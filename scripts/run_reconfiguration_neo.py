import tyro
import os
import shutil
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Union

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from neo.utils.commands import run_command
from rich.table import Table

from scripts.pipeline_progress_bar import PipelineProgress
from scripts.experiment_config import NeoDataConfig

from scripts.editing.edit import edit
from scripts.editing.loopback import loopback
from neo.utils.experiment_io import load_experiment_config, save_experiment_config


@dataclass
class AblationSetupElements():
    """Ablation setup elements to run"""

    _execute_all: bool = True
    """Prepare ablation outputs from a full run"""
    prepare_outputs: bool = True
    """Copy LERF and bbox artifacts from the full outputs folder"""

@dataclass
class DataPreperationElements():
    """Legacy reconfiguration config elements.

    Older experiment YAMLs reference this misspelled class. Keep the alias so
    those configs can be loaded and normalized into the current ablation setup.
    """

    _execute_all: bool = True

@dataclass
class TrainingElements():
    """Training elements to run"""

    _execute_all: bool = True
    """Execute all training elements"""
    train_neo: bool = True
    """Train Neo"""
    loopback_neo: bool = True
    """Loopback Neo"""

@dataclass
class EvaluationElements():
    """Evaluation elements to run"""

    _execute_all: bool = True
    """Execute all evaluation elements"""
    render_images: bool = True
    """Render evaluation images"""
    calculate_image_metrics: bool = True
    """Calculate PSNR, SSIM, and LPIPS"""
    calculate_depth_metrics: bool = True
    """Calculate depth RMSE and MAE"""
    export_pointcloud: bool = True
    """Export pointcloud for the trained model"""
    calculate_reprojection: bool = True
    """Calculate dense reprojection error"""

@dataclass
class PipelineElements():

    ablation_setup: AblationSetupElements = field(default_factory=AblationSetupElements)
    """Ablation setup elements to run"""
    training: TrainingElements = field(default_factory=TrainingElements)
    """Training elements to run"""
    evaluation: EvaluationElements = field(default_factory=EvaluationElements)
    """Evaluation elements to run"""

@dataclass
class ExperimentConfig(NeoDataConfig, InstantiateConfig):
    """Configuration for running experiments."""

    _target: Type = field(default_factory=lambda: Experiment)
    """target class to instantiate"""
    experiment_name: str = "default"
    """Name of the experiment"""
    load_config: Union[str, None] = None
    """Load configuration file from name"""
    generate_config: bool = False
    """Generate configuration file and exit program"""
    pipeline_elements: PipelineElements = field(default_factory=PipelineElements)
    """Pipeline elements to run"""
    lerf_prompt: str = "What is the object in the scene?"
    """Prompt for LERF-TOGO"""
    full_outputs_root: Path = Path("outputs_full")
    """Outputs root containing the full-system run"""
    ablation_outputs_root: Path = Path("outputs")
    """Outputs root used by ablation runs"""
    full_renders_root: Path = Path("renders_full")
    """Renders root containing the full-system rendered images"""
    renders_root: Path = Path("renders")
    """Root used for rendered evaluation images"""
    pointclouds_root: Path = Path("exports/pcd")
    """Root used for pointcloud exports"""
    eval_render_name: str = "full_edited_dist_appear_neo"
    """Render folder name for edited Neo evaluation images"""
    eval_gt_render_name: str = "full"
    """Render folder name for after/nerfacto GT depth"""
    reprojection_step: int = 20
    """Frame step used for reprojection evaluation"""
    eval_max_depth: float = 2.0
    """Maximum depth used by eval_depth_rmse.py and eval_reprojection_tree.py"""
    reprojection_pixel_stride: int = 4
    """Pixel stride used by eval_reprojection_tree.py"""
    pointcloud_num_points: int = 1_000_000
    """Number of points exported by ns-export"""
    replace_nerf_use_field_distillation: bool = True
    """Use per-sample RGB and density distillation in the Replace-NeRF stage"""
    replace_nerf_field_distillation_steps: int = 100
    """Number of Replace-NeRF steps that use per-sample RGB and density distillation"""
    replace_nerf_use_render_rgb_distillation: bool = True
    """Use rendered RGB distillation in the Replace-NeRF stage"""
    replace_nerf_use_render_depth_distillation: bool = True
    """Use rendered depth distillation in the Replace-NeRF stage"""

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir =Path("experiments")
        assert base_dir is not None, f"Data directory does not exist: {base_dir}"
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / f"{self.experiment_name}.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        save_experiment_config(self, config_yaml_path)

def _copy_existing_attrs(source: object, target: object) -> None:
    """Copy attributes that exist on target from a loaded YAML object."""
    if source is None:
        return

    for attr in vars(target):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))

def _normalize_loaded_config(config: ExperimentConfig) -> ExperimentConfig:
    """Normalize older experiment YAMLs to this runner's config schema."""
    pipeline_elements = getattr(config, "pipeline_elements", None)
    normalized_pipeline = PipelineElements()

    _copy_existing_attrs(getattr(pipeline_elements, "ablation_setup", None), normalized_pipeline.ablation_setup)
    legacy_data_preparation = getattr(pipeline_elements, "data_preparation", None)
    if legacy_data_preparation is not None and not hasattr(pipeline_elements, "ablation_setup"):
        normalized_pipeline.ablation_setup._execute_all = getattr(legacy_data_preparation, "_execute_all", True)
        normalized_pipeline.ablation_setup.prepare_outputs = True

    _copy_existing_attrs(getattr(pipeline_elements, "training", None), normalized_pipeline.training)
    _copy_existing_attrs(getattr(pipeline_elements, "evaluation", None), normalized_pipeline.evaluation)

    config.pipeline_elements = normalized_pipeline
    for attr, value in vars(ExperimentConfig()).items():
        if not hasattr(config, attr):
            setattr(config, attr, value)

    config._target = Experiment
    return config

def _replace_root(path: Path, old_root: Path, new_root: Path) -> Path:
    """Build an output path using the config's data path convention."""
    path = Path(path)
    try:
        return new_root / path.relative_to(old_root)
    except ValueError:
        return Path(str(path).replace(str(old_root), str(new_root), 1))

def _latest_timestamp_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Required ablation source does not exist: {base_dir}")

    timestamp_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp directories found in {base_dir}")

    return timestamp_dirs[0]

def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required ablation source does not exist: {src}")

    CONSOLE.log(f"Copying {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

def _round_bbox_dimensions(bbox_json: dict) -> dict:
    """Round bbox dimensions to avoid precision issues in bbox tests."""
    rounded = json.loads(json.dumps(bbox_json))
    for bbox in rounded.get("bboxes", []):
        data = bbox.get("data", {})
        dimensions = data.get("dimensions")
        if dimensions is None:
            continue
        data["dimensions"] = [round(float(value), 2) for value in dimensions]
    return rounded

def _write_bbox_json(path: Path, bbox_json: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as bbox_file:
        json.dump(_round_bbox_dimensions(bbox_json), bbox_file, indent=4)

def _copy_model_artifacts(src_base_dir: Path, dst_base_dir: Path, timestamp: str, model_type: str) -> None:
    source_candidates = [
        src_base_dir / "before" / model_type / timestamp / "bbox.json",
    ]

    src_bbox = next((path for path in source_candidates if path.exists()), None)
    if src_bbox is None:
        candidates = "\n".join(str(path) for path in source_candidates)
        raise FileNotFoundError(f"No bbox.json found for {model_type}. Tried:\n{candidates}")

    dst_model_dir = dst_base_dir / "before" / model_type / timestamp
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    with src_bbox.open("r", encoding="utf8") as bbox_file:
        _write_bbox_json(dst_model_dir / "bbox.json", json.load(bbox_file))
    CONSOLE.log(f"Copied bbox for {model_type}: {src_bbox} -> {dst_model_dir / 'bbox.json'}")

    for artifact_name in ("grasps.npy",):
        src_artifact = src_base_dir / "before" / model_type / timestamp / artifact_name
        if src_artifact.exists():
            shutil.copyfile(src_artifact, dst_model_dir / artifact_name)
            CONSOLE.log(f"Copied {artifact_name} for {model_type}: {src_artifact} -> {dst_model_dir / artifact_name}")

def _validate_ablation_outputs(dst_base_dir: Path, timestamp: str, model_type: str) -> None:
    required_paths = [
        dst_base_dir / "before" / "lerf-lite" / timestamp / "config.yml",
        dst_base_dir / "before" / model_type / timestamp / "bbox.json",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    lerf_checkpoint_dir = dst_base_dir / "before" / "lerf-lite" / timestamp / "nerfstudio_models"
    if not any(lerf_checkpoint_dir.glob("*.ckpt")):
        missing_paths.append(lerf_checkpoint_dir / "*.ckpt")

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Ablation setup is missing required artifacts:\n{missing}")

def _copy_gt_depth_renders(config: ExperimentConfig) -> None:
    """Copy after/nerfacto GT renders required by depth metrics."""
    data_base_dir = config.get_base_dir()
    src_render_base_dir = _replace_root(data_base_dir, config.root_input, config.full_renders_root)
    dst_render_base_dir = _replace_root(data_base_dir, config.root_input, config.renders_root)

    src_after_nerfacto_dir = src_render_base_dir / "after" / "nerfacto"
    if config.timestamp:
        src_gt_run_dir = src_after_nerfacto_dir / config.timestamp
    else:
        src_gt_run_dir = _latest_timestamp_dir(src_after_nerfacto_dir)
        config.timestamp = src_gt_run_dir.name

    src_depth_dir = src_gt_run_dir / f"{config.eval_gt_render_name}_edited" / "depth"
    if not src_depth_dir.exists():
        raise FileNotFoundError(f"Missing full-run GT depth renders: {src_depth_dir}")

    dst_gt_run_dir = dst_render_base_dir / "after" / "nerfacto" / config.timestamp
    _copy_tree(src_gt_run_dir, dst_gt_run_dir)

def prepare_ablation_outputs(config: ExperimentConfig) -> None:
    """Copy full-run artifacts required to train ablation Neo models."""
    data_base_dir = config.get_base_dir()
    src_base_dir = _replace_root(data_base_dir, config.root_input, config.full_outputs_root)
    dst_base_dir = _replace_root(data_base_dir, config.root_input, config.ablation_outputs_root)

    lerf_base_dir = src_base_dir / "before" / "lerf-lite"
    src_lerf_dir = lerf_base_dir / config.timestamp if config.timestamp else _latest_timestamp_dir(lerf_base_dir)
    config.timestamp = src_lerf_dir.name

    dst_lerf_dir = dst_base_dir / "before" / "lerf-lite" / config.timestamp
    _copy_tree(src_lerf_dir, dst_lerf_dir)

    _copy_model_artifacts(src_base_dir, dst_base_dir, config.timestamp, "neo")
    _validate_ablation_outputs(dst_base_dir, config.timestamp, "neo")
    _copy_gt_depth_renders(config)

    CONSOLE.log(f"Ablation outputs are ready in: {dst_base_dir}")

def _output_base_dir(config: ExperimentConfig) -> Path:
    return _replace_root(config.get_base_dir(), config.root_input, config.ablation_outputs_root)

def _render_base_dir(config: ExperimentConfig) -> Path:
    return _replace_root(config.get_base_dir(), config.root_input, config.renders_root) / "before" / "neo" / config.get_timestamp()

def _replace_nerf_config_path(config: ExperimentConfig) -> Path:
    return _output_base_dir(config) / "before" / "neo" / config.get_timestamp() / "replace-nerf" / "config.yml"

def _eval_output_dir(config: ExperimentConfig) -> Path:
    return Path("eval") / _replace_nerf_config_path(config).parent.relative_to(config.ablation_outputs_root)

def _depth_eval_output_dir(config: ExperimentConfig) -> Path:
    return Path("eval_depth_rmse") / _replace_nerf_config_path(config).parent.parent.relative_to(config.ablation_outputs_root)

def _reprojection_eval_output_dir(config: ExperimentConfig) -> Path:
    return Path("eval_reprojection_tree") / _replace_nerf_config_path(config).parent.parent.relative_to(config.ablation_outputs_root)

def _after_transforms_path(config: ExperimentConfig) -> Path:
    return config.get_base_dir() / "after" / "transforms.json"

def _load_bbox_json(config: ExperimentConfig) -> dict:
    bbox_path = _output_base_dir(config) / "before" / "neo" / config.get_timestamp() / "bbox.json"
    if not bbox_path.exists():
        raise FileNotFoundError(f"Missing bbox file for evaluation: {bbox_path}")

    with bbox_path.open("r", encoding="utf8") as bbox_file:
        return _round_bbox_dimensions(json.load(bbox_file))

def _reconfigured_bbox_json(bbox_json: dict) -> dict:
    bbox_after = json.loads(json.dumps(bbox_json))
    for bbox in bbox_after.get("bboxes", []):
        data = bbox.get("data", {})
        if not data.get("is_reconfigured", False):
            continue

        position = data.get("position", [0.0, 0.0, 0.0])
        offset_position = data.get("offset_position", [0.0, 0.0, 0.0])
        euler = data.get("euler", [0.0, 0.0, 0.0])
        offset_euler = data.get("offset_euler", [0.0, 0.0, 0.0])
        data["position"] = [float(value) + float(offset) for value, offset in zip(position, offset_position)]
        data["euler"] = [float(value) + float(offset) for value, offset in zip(euler, offset_euler)]

    return bbox_after

def prepare_evaluation_assets(config: ExperimentConfig) -> None:
    """Copy GT images and generate before/after bbox masks for evaluation metrics."""
    import numpy as np

    from neo.utils.mask_generator import generate_masks

    render_base_dir = _render_base_dir(config)
    gt_data_path = config.get_base_dir() / "after"
    transforms_path = gt_data_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing after transforms for evaluation: {transforms_path}")

    shutil.copytree(gt_data_path, render_base_dir / f"{config.eval_gt_render_name}_gt", dirs_exist_ok=True)

    with transforms_path.open("r", encoding="utf8") as transforms_file:
        camera_data = json.load(transforms_file)

    bbox_json = _load_bbox_json(config)
    generate_masks(
        camera_data=camera_data,
        bboxes=bbox_json["bboxes"],
        output_path=render_base_dir / f"{config.eval_gt_render_name}_masks",
        dataparser_transform=np.eye(4)[0:3, :],
        dataparser_scale=1.0,
    )
    generate_masks(
        camera_data=camera_data,
        bboxes=_reconfigured_bbox_json(bbox_json)["bboxes"],
        output_path=render_base_dir / f"{config.eval_gt_render_name}_masks_after",
        dataparser_transform=np.eye(4)[0:3, :],
        dataparser_scale=1.0,
    )
    if config.eval_render_name != config.eval_gt_render_name:
        shutil.copytree(
            render_base_dir / f"{config.eval_gt_render_name}_masks",
            render_base_dir / f"{config.eval_render_name}_masks",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            render_base_dir / f"{config.eval_gt_render_name}_masks_after",
            render_base_dir / f"{config.eval_render_name}_masks_after",
            dirs_exist_ok=True,
        )

def render_evaluation_images(config: ExperimentConfig) -> None:
    """Render edited Neo images from the after transforms."""
    config_path = _replace_nerf_config_path(config)
    transforms_path = _after_transforms_path(config)
    output_path = _render_base_dir(config) / config.eval_render_name

    if not config_path.exists():
        raise FileNotFoundError(f"Missing replace-nerf config for evaluation render: {config_path}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms for evaluation render: {transforms_path}")

    command = [
        "ns-render-neo", "render-from-transforms",
        "--load-config", config_path,
        "--transforms-filename", transforms_path,
        "--image-format", "png",
        "--output-format", "images",
        "--rendered-output-names", "rgb", "depth",
        "--output-path", output_path,
    ]
    run_command(command, verbose=True)
    prepare_evaluation_assets(config)

def _load_rgb_image(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.float32) / 255.0

def _load_mask(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return (np.array(image.convert("L"), dtype=np.uint8) > 0).astype(np.float32)

def _collect_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    pred_files = sorted(path for path in pred_dir.iterdir() if path.is_file())
    gt_by_name = {path.name: path for path in gt_dir.iterdir() if path.is_file()}
    pairs = [(pred_path, gt_by_name[pred_path.name]) for pred_path in pred_files if pred_path.name in gt_by_name]
    if not pairs:
        raise ValueError(f"No matching rendered/GT images found between {pred_dir} and {gt_dir}")

    return pairs

def calculate_image_metrics(config: ExperimentConfig) -> None:
    """Calculate PSNR, SSIM, and LPIPS for full/out/in image regions."""
    import numpy as np
    import torch

    from neo.eval.metrics import LPIPSModule, PSNRModule, SSIMModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = PSNRModule().to(device)
    ssim_metric = SSIMModule().to(device)
    lpips_metric = LPIPSModule().to(device)

    render_base_dir = _render_base_dir(config)
    pred_dir = render_base_dir / config.eval_render_name / "rgb"
    gt_dir = render_base_dir / f"{config.eval_gt_render_name}_gt" / "images"
    mask_dirs = {
        "all": None,
        "out": render_base_dir / f"{config.eval_gt_render_name}_masks" / "temp_masks",
        "in": render_base_dir / f"{config.eval_gt_render_name}_masks_after" / "temp_masks",
    }

    pairs = _collect_pairs(pred_dir, gt_dir)
    results = {}
    for region_name, mask_dir in mask_dirs.items():
        values = {"psnr": [], "ssim": [], "lpips": []}
        for pred_path, gt_path in pairs:
            pred_image = _load_rgb_image(pred_path)
            gt_image = _load_rgb_image(gt_path)
            pred_tensor = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

            mask_tensor = None
            if mask_dir is not None:
                mask_path = mask_dir / pred_path.name
                if not mask_path.exists():
                    continue
                mask = _load_mask(mask_path)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                psnr = psnr_metric(pred_tensor, gt_tensor, mask_tensor).item()
                ssim = ssim_metric(pred_tensor, gt_tensor, mask_tensor).item()
                lpips = lpips_metric(pred_tensor, gt_tensor, mask_tensor).item()

            if not (np.isnan(psnr) or np.isinf(psnr)):
                values["psnr"].append(psnr)
                values["ssim"].append(ssim)
                values["lpips"].append(lpips)

        if values["psnr"]:
            results[region_name] = {
                "frames": len(values["psnr"]),
                "psnr": float(np.mean(values["psnr"])),
                "ssim": float(np.mean(values["ssim"])),
                "lpips": float(np.mean(values["lpips"])),
            }

    eval_output_path = _eval_output_dir(config)
    eval_output_path.mkdir(parents=True, exist_ok=True)
    output_json = eval_output_path / f"image_metrics_{config.eval_render_name}.json"
    with output_json.open("w", encoding="utf8") as output_file:
        json.dump(results, output_file, indent=4)

    CONSOLE.log(f"Saved image metrics to {output_json}")

def calculate_depth_metrics(config: ExperimentConfig) -> None:
    """Calculate depth RMSE and MAE for the edited render."""
    from neo.eval.eval_depth_rmse import calculate_depth_rmse_for_render

    render_base_dir = _render_base_dir(config)
    info = {
        "task": config.action,
        "object_group": config.scene_name,
        "object_name": config.object_name,
        "method": "neo",
        "run": config.get_timestamp(),
        "render_name": config.eval_render_name,
        "render_run_dir": render_base_dir,
        "pred_depth_dir": render_base_dir / config.eval_render_name / "depth",
    }
    calculate_depth_rmse_for_render(
        info=info,
        renders_root=config.renders_root,
        render_name=config.eval_render_name,
        gt_render_name=config.eval_gt_render_name,
        max_depth=config.eval_max_depth,
    )

def export_pointcloud(config: ExperimentConfig) -> None:
    """Export a pointcloud for the edited Neo model."""
    config_path = _replace_nerf_config_path(config)
    output_dir = config.pointclouds_root / config_path.parent.relative_to(config.ablation_outputs_root)
    command = [
        "ns-export", "pointcloud",
        "--load-config", config_path,
        "--output-dir", output_dir,
        "--num-points", config.pointcloud_num_points,
        "--remove-outliers", "True",
        "--normal-method", "open3d",
        "--save-world-frame", "False",
    ]
    run_command(command, verbose=True)

def calculate_reprojection_error(config: ExperimentConfig) -> None:
    """Calculate reprojection error using eval_reprojection_tree.py."""
    from tqdm import tqdm

    from neo.eval.eval_reprojection_tree import pair_errors, stats

    render_run_dir = _render_base_dir(config)
    render_dir = render_run_dir / config.eval_render_name
    data_path = config.get_base_dir() / "after"
    transforms_path = data_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing {transforms_path}")

    with transforms_path.open("r", encoding="utf8") as transforms_file:
        meta = json.load(transforms_file)

    frames = meta["frames"]
    total_pairs = max(0, len(frames) - config.reprojection_step)
    perceptual = []
    perceptual_out = []
    perceptual_in = []
    perceptual_unmasked = []
    geometric = []
    geometric_out = []
    geometric_in = []
    geometric_unmasked = []
    pairs = []

    CONSOLE.print(
        f"Evaluating {config.action}/{config.object_name} neo "
        f"({total_pairs} pairs) with eval_reprojection_tree.py procedure..."
    )
    for i in tqdm(range(total_pairs)):
        try:
            result = pair_errors(
                render_dir=render_dir,
                data_path=data_path,
                out_mask_dir=render_run_dir / f"{config.eval_render_name.removesuffix('_edited')}_masks" / "temp_masks",
                in_mask_dir=render_run_dir / f"{config.eval_render_name.removesuffix('_edited')}_masks_after" / "temp_masks",
                use_in_mask=config.action == "translate_and_rotate",
                frames=frames,
                meta=meta,
                i=i,
                step=config.reprojection_step,
                pixel_stride=config.reprojection_pixel_stride,
                depth_scale=0.001,
                min_depth=1e-6,
                max_depth=config.eval_max_depth,
                depth_consistency_thresh=0.05,
                depth_is_distance=True,
            )
        except Exception:
            result = None
        if result is None:
            continue

        perceptual.extend(result["perceptual"].detach().cpu().tolist())
        perceptual_out.extend(result["perceptual_out"].detach().cpu().tolist())
        perceptual_in.extend(result["perceptual_in"].detach().cpu().tolist())
        perceptual_unmasked.extend(result["perceptual_unmasked"].detach().cpu().tolist())
        geometric.extend(result["geometric"].detach().cpu().tolist())
        geometric_out.extend(result["geometric_out"].detach().cpu().tolist())
        geometric_in.extend(result["geometric_in"].detach().cpu().tolist())
        geometric_unmasked.extend(result["geometric_unmasked"].detach().cpu().tolist())
        pairs.append(
            {
                "frame_idx_1": result["pair"][0],
                "frame_idx_2": result["pair"][1],
                "perceptual_mean": float(result["perceptual"].mean().item()),
                "perceptual_out_mean": (
                    float(result["perceptual_out"].mean().item()) if len(result["perceptual_out"]) else None
                ),
                "perceptual_in_mean": (
                    float(result["perceptual_in"].mean().item()) if len(result["perceptual_in"]) else None
                ),
                "perceptual_unmasked_mean": (
                    float(result["perceptual_unmasked"].mean().item()) if len(result["perceptual_unmasked"]) else None
                ),
                "geometric_mean": float(result["geometric"].mean().item()) if len(result["geometric"]) else None,
                "geometric_out_mean": (
                    float(result["geometric_out"].mean().item()) if len(result["geometric_out"]) else None
                ),
                "geometric_in_mean": (
                    float(result["geometric_in"].mean().item()) if len(result["geometric_in"]) else None
                ),
                "geometric_unmasked_mean": (
                    float(result["geometric_unmasked"].mean().item()) if len(result["geometric_unmasked"]) else None
                ),
                "perceptual_pixels": int(len(result["perceptual"])),
                "perceptual_out_pixels": int(len(result["perceptual_out"])),
                "perceptual_in_pixels": int(len(result["perceptual_in"])),
                "perceptual_unmasked_pixels": int(len(result["perceptual_unmasked"])),
                "geometric_pixels": int(len(result["geometric"])),
                "geometric_out_pixels": int(len(result["geometric_out"])),
                "geometric_in_pixels": int(len(result["geometric_in"])),
                "geometric_unmasked_pixels": int(len(result["geometric_unmasked"])),
            }
        )

    if not perceptual or not geometric:
        raise ValueError("No valid reprojection pairs found.")

    result_dict = {
        "task": config.action,
        "object_name": config.object_name,
        "method": "neo",
        "run": config.get_timestamp(),
        "step": config.reprojection_step,
        "pixel_stride": config.reprojection_pixel_stride,
        "depth_scale": 0.001,
        "depth_is_distance": True,
        "min_depth": 1e-6,
        "max_depth": config.eval_max_depth,
        "depth_consistency_thresh": 0.05,
        "debug_limit": 0,
        "debug_max_points": 3000,
        "render_dir": str(render_dir),
        "transforms_path": str(transforms_path),
        "total_valid_pairs": len(pairs),
        "all": {
            "perceptual": stats(perceptual),
            "geometric": stats(geometric),
        },
        "pairs": pairs,
    }
    if perceptual_out or geometric_out:
        result_dict["out"] = {
            "perceptual": stats(perceptual_out),
            "geometric": stats(geometric_out),
        }
    if perceptual_in or geometric_in:
        result_dict["in"] = {
            "perceptual": stats(perceptual_in),
            "geometric": stats(geometric_in),
        }
    if perceptual_unmasked or geometric_unmasked:
        result_dict["unmasked"] = {
            "perceptual": stats(perceptual_unmasked),
            "geometric": stats(geometric_unmasked),
        }

    output_dir = _reprojection_eval_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reprojection_step_{config.reprojection_step}_{config.eval_render_name.removesuffix('_edited')}.json"
    output_path.write_text(json.dumps(result_dict, indent=4), encoding="utf8")
    CONSOLE.log(f"Saved reprojection metrics to {output_path}")

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf8") as json_file:
        return json.load(json_file)

def _fmt_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)

def _region_metrics(metrics: dict, region: str) -> dict:
    return metrics.get(region, {}) if metrics else {}

def _reprojection_region_metrics(metrics: dict, region: str) -> dict:
    if not metrics:
        return {}
    region_metrics = metrics.get(region, {})
    if "perceptual" in region_metrics:
        perceptual = region_metrics.get("perceptual", {})
        return {
            "pixels": perceptual.get("count"),
            "mean_error": perceptual.get("mean"),
            "median_error": perceptual.get("median"),
        }
    if region == "out":
        return metrics.get("masked", {})
    return region_metrics

def _reprojection_metrics_path(config: ExperimentConfig) -> Path:
    candidates = [
        _reprojection_eval_output_dir(config) / f"reprojection_step_{config.reprojection_step}_{config.eval_render_name.removesuffix('_edited')}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _metrics_summary_markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"

def print_metrics_summary_table(config: ExperimentConfig) -> None:
    """Print a compact summary of all evaluation metrics written by this run."""
    image_metrics_path = _eval_output_dir(config) / f"image_metrics_{config.eval_render_name}.json"
    depth_metrics_path = _depth_eval_output_dir(config) / f"depth_rmse_results_{config.eval_render_name.removesuffix('_edited')}.json"
    reprojection_metrics_path = _reprojection_metrics_path(config)

    image_metrics = _read_json(image_metrics_path)
    depth_metrics = _read_json(depth_metrics_path)
    reprojection_metrics = _read_json(reprojection_metrics_path)

    if not (image_metrics or depth_metrics or reprojection_metrics):
        CONSOLE.log("No evaluation metric files found; skipping metrics summary table.")
        return

    table = Table(title="Evaluation Metrics")
    table.add_column("Region", style="bold")
    table.add_column("Frames/Pixels", justify="right")
    table.add_column("PSNR", justify="right")
    table.add_column("SSIM", justify="right")
    table.add_column("LPIPS", justify="right")
    table.add_column("Depth RMSE", justify="right")
    table.add_column("Depth MAE", justify="right")
    table.add_column("Reproj Mean", justify="right")
    table.add_column("Reproj Median", justify="right")

    headers = [
        "Region",
        "Frames/Pixels",
        "PSNR",
        "SSIM",
        "LPIPS",
        "Depth RMSE",
        "Depth MAE",
        "Reproj Mean",
        "Reproj Median",
    ]
    rows = []

    for region in ("all", "out", "in"):
        image_region = _region_metrics(image_metrics, region)
        depth_region = _region_metrics(depth_metrics, region)
        reprojection_region = _reprojection_region_metrics(reprojection_metrics, region)
        count = image_region.get("frames") or depth_region.get("pixels") or reprojection_region.get("pixels")

        row = [
            region,
            str(count) if count is not None else "N/A",
            _fmt_metric(image_region.get("psnr")),
            _fmt_metric(image_region.get("ssim")),
            _fmt_metric(image_region.get("lpips")),
            _fmt_metric(depth_region.get("rmse")),
            _fmt_metric(depth_region.get("mae")),
            _fmt_metric(reprojection_region.get("mean_error")),
            _fmt_metric(reprojection_region.get("median_error")),
        ]
        rows.append(row)
        table.add_row(*row)

    CONSOLE.print(table)
    summary_output_path = _eval_output_dir(config) / f"metrics_summary_{config.eval_render_name}.md"
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(_metrics_summary_markdown(headers, rows), encoding="utf8")
    CONSOLE.log(f"Saved metrics summary table to {summary_output_path}")

class Experiment():
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _execute_pipeline(self, pipeline_elements):
        """Execute a pipeline of elements."""
        with PipelineProgress(pipeline_elements, console=CONSOLE) as progress:
            for item in progress:
                progress.start_step()
                if item.get("kwargs"):
                    item["func"](self.config, **item["kwargs"])
                else:
                    item["func"](self.config)
                progress.end_step()

    def run(self):
        # Ablation setup
        if self.config.pipeline_elements.ablation_setup._execute_all:
            elements = self.config.pipeline_elements.ablation_setup
            preparing_outputs = {"name": "Prepare Outputs", "active": elements.prepare_outputs, "idx": 1, "func": prepare_ablation_outputs}

            pipeline_elements = [[preparing_outputs]]

            self._execute_pipeline(pipeline_elements)

        # Training editing networks
        if self.config.pipeline_elements.training._execute_all:
            elements = self.config.pipeline_elements.training

            train_neo = {"name": "Editing Neo", "active": elements.train_neo, "idx": 1, "func": edit, "kwargs": {"model_type": "neo"}}
            loopback_neo = {"name": "Looping Neo", "active": elements.loopback_neo, "idx": 2, "func": loopback, "kwargs": {"model_type": "neo"}}
            
            pipeline_elements = [[train_neo, "→", loopback_neo]]

            self._execute_pipeline(pipeline_elements)

        # Evaluation
        if self.config.pipeline_elements.evaluation._execute_all:
            elements = self.config.pipeline_elements.evaluation
            rendering_images = {"name": "Render Eval", "active": elements.render_images, "idx": 1, "func": render_evaluation_images}
            image_metrics = {"name": "Image Metrics", "active": elements.calculate_image_metrics, "idx": 2, "func": calculate_image_metrics}
            depth_metrics = {"name": "Depth Metrics", "active": elements.calculate_depth_metrics, "idx": 3, "func": calculate_depth_metrics}
            exporting_pointcloud = {"name": "Export Pointcloud", "active": elements.export_pointcloud, "idx": 4, "func": export_pointcloud}
            reprojection = {"name": "Reprojection", "active": elements.calculate_reprojection, "idx": 5, "func": calculate_reprojection_error}

            pipeline_elements = [[rendering_images, "→", image_metrics, "→", depth_metrics, "→", exporting_pointcloud, "→", reprojection]]

            self._execute_pipeline(pipeline_elements)


def main(config: ExperimentConfig) -> None:
    """Main function."""

    # Create experiments directory if it does not exist.
    os.makedirs("experiments", exist_ok=True)

    # Load configuration from file.
    if config.load_config is not None:
        config_path = Path(f"experiments/{config.load_config}")
        assert config_path.exists(), f"Config file does not exist: {config_path}"

        CONSOLE.log(f"Loading pre-set config from: experiments/{config.load_config}")
        config = load_experiment_config(config, config_path)
        config = _normalize_loaded_config(config)

    else:
        # Save configuration to file.
        config.save_config()
        
    config.print_to_terminal()
    if config.generate_config:
        config.generate_config = False
        config.save_config()
        CONSOLE.log("Configuration file generated. Exiting program.")
        return

    # Instantiate the target class.
    target = config.setup()

    # Run the experiment.
    target.run()
    print_metrics_summary_table(config)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")

    AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            ExperimentConfig
        ]
    ]

    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )

if __name__ == "__main__":
    entrypoint()
