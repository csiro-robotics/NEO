from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from neo.utils.commands import run_command
from neo.utils.experiment_io import load_experiment_config, save_experiment_config
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.ray_generators import RayGenerator
from scripts.experiment_config import NeoDataConfig
from scripts.run_assembly import ExperimentConfig as AssemblyConfig
from scripts.run_reconfiguration import ExperimentConfig as ReconfigurationConfig
from scripts.run_removal import ExperimentConfig as RemovalConfig


@dataclass
class NestedConfig:
    enabled: bool = True


@dataclass
class ExampleConfig:
    data_path: Path = Path("data")
    nested: NestedConfig = field(default_factory=NestedConfig)
    _private: str = "not serialized"


def test_experiment_config_round_trip_uses_plain_yaml(tmp_path: Path) -> None:
    source = ExampleConfig(data_path=Path("data with spaces"), nested=NestedConfig(enabled=False))
    config_path = tmp_path / "config.yaml"

    save_experiment_config(source, config_path)
    text = config_path.read_text(encoding="utf8")
    loaded = load_experiment_config(ExampleConfig(), config_path)

    assert "!!python" not in text
    assert "_private" not in text
    assert loaded.data_path == Path("data with spaces")
    assert loaded.nested.enabled is False


def test_unknown_config_keys_fail_with_context(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("nested:\n  unknown: true\n", encoding="utf8")

    with pytest.raises(ValueError, match="nested"):
        load_experiment_config(ExampleConfig(), config_path)


def test_scene_paths_are_mirrored_without_string_replacement() -> None:
    config = NeoDataConfig(
        root_input=Path("datasets/input"),
        outputs_root=Path("artifacts/models"),
        data_type="real",
        action="remove",
        scene_type="single",
        scene_name="data_in_scene_name",
        object_name="object",
    )

    assert config.get_relative_scene_dir() == Path("real/remove/single/data_in_scene_name/object")
    assert config.get_output_dir() == Path("artifacts/models/real/remove/single/data_in_scene_name/object")
    assert config.get_raw_capture_base() == Path(
        "datasets/input/raw_bagfiles/real/remove/single/data_in_scene_name/object"
    )


def test_timestamp_resolves_latest_lerf_output(tmp_path: Path) -> None:
    config = NeoDataConfig(
        root_input=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
        scene_name="scene",
        object_name="object",
    )
    lerf_root = config.get_output_dir() / "before" / "lerf-lite"
    (lerf_root / "2026-07-15_134640").mkdir(parents=True)
    (lerf_root / "2026-07-15_174051").mkdir()
    (lerf_root / "not-a-run").mkdir()

    assert config.get_timestamp() == "2026-07-15_174051"


def test_timestamp_is_generated_when_no_lerf_output_exists(tmp_path: Path) -> None:
    config = NeoDataConfig(
        root_input=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
        scene_name="scene",
        object_name="object",
    )

    timestamp = config.get_timestamp()

    assert timestamp is not None
    assert len(timestamp) == len("2026-07-15_193521")


def test_checked_command_preserves_argument_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(args, check):
        observed["args"] = args
        observed["check"] = check

    monkeypatch.setattr("neo.utils.commands.subprocess.run", fake_run)
    run_command(["tool", "--path", Path("directory with spaces")], verbose=False)

    assert observed == {"args": ["tool", "--path", "directory with spaces"], "check": True}


@pytest.mark.parametrize("channel_first", [False, True])
def test_mask_rejection_sampling_supports_both_channel_layouts(channel_first: bool) -> None:
    sampler = PixelSampler(PixelSamplerConfig())
    mask = torch.ones((2, 8, 10, 1), dtype=torch.bool)
    if channel_first:
        mask = mask.permute(0, 3, 1, 2)

    indices = sampler.rejection_sample_mask(
        mask=mask,
        num_samples=32,
        num_images=2,
        image_height=8,
        image_width=10,
        device="cpu",
    )

    assert indices.shape == (32, 3)
    assert torch.all(indices[:, 0] < 2)
    assert torch.all(indices[:, 1] < 8)
    assert torch.all(indices[:, 2] < 10)


def test_ray_generator_coordinates_follow_camera_device() -> None:
    cameras = Cameras(
        camera_to_worlds=torch.eye(4)[:3].unsqueeze(0),
        fx=100.0,
        fy=100.0,
        cx=5.0,
        cy=4.0,
        width=10,
        height=8,
    )

    ray_generator = RayGenerator(cameras)

    assert ray_generator.image_coords.device == cameras.device


@pytest.mark.parametrize(
    "relative_path, expected_action, config_type",
    [
        ("removal/tomato_soup_can.yaml", "remove", RemovalConfig),
        ("relocation/banana.yaml", "translate_and_rotate", ReconfigurationConfig),
    ],
)
def test_paper_scene_manifests_load(relative_path: str, expected_action: str, config_type) -> None:
    config_path = Path("experiments/paper") / relative_path
    config = load_experiment_config(config_type(), config_path)

    assert config.action == expected_action
    assert config.scene_name is not None
    assert config.object_name is not None


def test_assembly_manifest_runs_all_four_steps() -> None:
    config = load_experiment_config(AssemblyConfig(), Path("experiments/paper/assembly.yaml"))

    assert config.object_names == ["lego_bricks_1", "red_cup", "lego_bricks_2", "golf_ball"]
    assert config.steps_to_run == [0, 1, 2, 3]
    assert not hasattr(config, "neo_model_type")


def test_only_one_public_neo_method_is_registered() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf8")

    assert "neo = 'neo.neo_config:neo_method'" in pyproject
    assert "neo-big" not in pyproject


def test_basic_conda_environment_is_named_neo() -> None:
    environment = Path("environment.yml").read_text(encoding="utf8")

    assert environment.startswith("name: neo\n")
    assert "colmap=3.6.0" in environment
    assert "cuda-toolkit=12.4.0" in environment
    assert "-e ./submodules/tinycudann" not in environment
    assert ".[language]" in environment
    assert ".[capture]" not in environment
    assert ".[baselines]" not in environment


def test_required_dependencies_are_physically_vendored() -> None:
    required_files = [
        Path("submodules/tinycudann/LICENSE.txt"),
        Path("submodules/tinycudann/include/tiny-cuda-nn/common.h"),
        Path("submodules/tinycudann/src/network.cu"),
        Path("submodules/tinycudann/bindings/torch/setup.py"),
        Path("submodules/tinycudann/dependencies/cutlass/LICENSE.txt"),
        Path("submodules/tinycudann/dependencies/json/LICENSE.MIT"),
        Path("submodules/nerfstudio/nerfstudio/__init__.py"),
        Path("submodules/lerf_bbox/lerf/__init__.py"),
        Path("submodules/neo_lerftogo/robot_lerf/scripts/gen_bbox.py"),
        Path(
            "submodules/neo_lerftogo/robot_lerf/graspnet_baseline/"
            "pointnet2/pointnet2/_ext.cpython-310-x86_64-linux-gnu.so"
        ),
        Path(
            "submodules/neo_lerftogo/robot_lerf/graspnet_baseline/"
            "knn/knn_pytorch/knn_pytorch.cpython-310-x86_64-linux-gnu.so"
        ),
        Path("submodules/neo_nerfiller/pyproject.toml"),
    ]

    assert all(path.is_file() and not path.is_symlink() for path in required_files)


def test_robot_lerf_declares_grasp_nms_runtime_dependency() -> None:
    pyproject = Path("submodules/neo_lerftogo/pyproject.toml").read_text(
        encoding="utf8"
    )

    assert '"grasp-nms"' in pyproject


def test_tinycudann_does_not_vendor_compiled_extensions() -> None:
    tinycudann = Path("submodules/tinycudann")

    assert not list(tinycudann.rglob("*.so"))
