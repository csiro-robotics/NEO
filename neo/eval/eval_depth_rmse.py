import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TASKS = ("remove", "reconfigure")


def render_name_candidates(render_name: str) -> tuple[str, ...]:
    if render_name.endswith("_edited"):
        base_name = render_name.removesuffix("_edited")
        return (render_name, base_name)
    return (render_name, f"{render_name}_edited")


def load_depth(depth_path: Path, depth_scale: float) -> torch.Tensor:
    depth_np = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_np is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    return torch.tensor(np.asarray(depth_np, dtype=np.float32), device=device) * depth_scale


def load_depth_with_valid_mask(depth_path: Path, depth_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    depth_np = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_np is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")

    depth_raw = np.asarray(depth_np)
    invalid_raw = depth_raw <= 0
    if np.issubdtype(depth_raw.dtype, np.integer):
        invalid_raw |= depth_raw >= np.iinfo(depth_raw.dtype).max

    depth = torch.tensor(depth_raw.astype(np.float32), device=device) * depth_scale
    valid = torch.tensor(~invalid_raw, dtype=torch.bool, device=device)
    return depth, valid


def get_depth_path(data_path: Path, frame: dict) -> Path:
    if "depth_file_path" in frame:
        return data_path / frame["depth_file_path"]

    file_path = str(frame["file_path"])
    return data_path / file_path.replace("images", "depth").replace("rgb", "depth")


def get_rendered_depth_path(render_output_path: Path, eval_exp_name: str, frame: dict) -> Path:
    file_name = Path(frame["file_path"]).name
    return render_output_path / f"{eval_exp_name}_edited" / "depth" / file_name


def get_depth_render_path(render_depth_dir: Path, frame: dict) -> Path:
    file_name = Path(frame["file_path"]).name
    return render_depth_dir / file_name


def get_render_mask_path(render_run_dir: Path, render_name: str, frame: dict) -> Path:
    file_name = Path(frame["file_path"]).name
    eval_exp_name = render_name.removesuffix("_edited")
    return render_run_dir / f"{eval_exp_name}_masks" / "temp_masks" / file_name


def get_render_mask_after_path(render_run_dir: Path, render_name: str, frame: dict) -> Path:
    file_name = Path(frame["file_path"]).name
    eval_exp_name = render_name.removesuffix("_edited")
    return render_run_dir / f"{eval_exp_name}_masks_after" / "temp_masks" / file_name


def resolve_after_nerfacto_render_depth_dir(base_path: Path, gt_render_name: str) -> Path:
    """Resolve GT depth renders from the matching after/nerfacto scene."""
    parts = base_path.parts
    state_idx = None
    for state_name in ("before", "after"):
        if state_name in parts:
            state_idx = parts.index(state_name)
            break

    if state_idx is None:
        raise ValueError(f"Could not resolve scene root from config base path: {base_path}")

    scene_root = Path(*parts[:state_idx])
    run_name = parts[state_idx + 2] if state_idx + 2 < len(parts) else None
    after_nerfacto_root = scene_root / "after" / "nerfacto"

    candidate_runs = []
    if run_name is not None:
        candidate_runs.append(after_nerfacto_root / run_name)
    if after_nerfacto_root.exists():
        candidate_runs.extend(
            path for path in sorted(after_nerfacto_root.iterdir()) if path.is_dir() and path.name != run_name
        )

    for candidate_run in candidate_runs:
        depth_dir = Path(str(candidate_run).replace("outputs", "renders")) / gt_render_name / "depth"
        if depth_dir.exists():
            return depth_dir

    raise FileNotFoundError(
        f"Could not find after/nerfacto GT depth renders under {after_nerfacto_root} "
        f"for render '{gt_render_name}'."
    )


def parse_render_depth_dir(renders_root: Path, depth_dir: Path) -> Optional[dict]:
    parts = depth_dir.relative_to(renders_root).parts
    if len(parts) < 10:
        return None

    try:
        state_idx = parts.index("before")
    except ValueError:
        return None

    if depth_dir.name != "depth":
        return None
    render_name = depth_dir.parent.name
    if state_idx + 3 >= len(parts):
        return None

    return {
        "task": parts[1],
        "object_group": parts[3],
        "object_name": parts[4],
        "method": parts[state_idx + 1],
        "run": parts[state_idx + 2],
        "render_name": render_name,
        "render_run_dir": depth_dir.parents[1],
        "pred_depth_dir": depth_dir,
    }


def discover_render_depth_dirs(renders_root: Path, tasks: tuple[str, ...], render_name: str) -> list[dict]:
    discovered = []
    seen = set()
    for task in tasks:
        task_root = renders_root / "real" / task
        if not task_root.exists():
            print(f"Skipping missing task render directory: {task_root}")
            continue

        for candidate_render_name in render_name_candidates(render_name):
            for depth_dir in sorted(task_root.glob(f"**/before/*/*/{candidate_render_name}/depth")):
                if depth_dir in seen:
                    continue
                seen.add(depth_dir)
                parsed = parse_render_depth_dir(renders_root, depth_dir)
                if parsed is not None:
                    discovered.append(parsed)
    return discovered


def data_path_from_render_info(info: dict) -> Path:
    return (
        # TODO how to remove data as the hard code path
        Path("data")
        / "processed"
        / info["task"]
        / info["object_group"]
        / info["object_name"]
        / "after"
    )


def resolve_gt_render_depth_dir_from_info(renders_root: Path, info: dict, gt_render_name: str) -> Path:
    after_nerfacto_root = (
        renders_root
        / info["task"]
        / info["object_group"]
        / info["object_name"]
        / "after"
        / "nerfacto"
    )
    for candidate_gt_render_name in render_name_candidates(gt_render_name):
        preferred = after_nerfacto_root / info["run"] / candidate_gt_render_name / "depth"
        if preferred.exists():
            return preferred

    for candidate_gt_render_name in render_name_candidates(gt_render_name):
        candidates = sorted(after_nerfacto_root.glob(f"*/{candidate_gt_render_name}/depth"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Could not find GT depth renders under {after_nerfacto_root}")


def eval_depth_dir_from_render_info(info: dict) -> Path:
    return (
        Path("eval_depth_rmse")
        / info["task"]
        / info["object_group"]
        / info["object_name"]
        / "before"
        / info["method"]
        / info["run"]
    )


def get_mask_path(render_output_path: Path, eval_exp_name: str, frame: dict) -> Path:
    file_name = Path(frame["file_path"]).name
    return render_output_path / f"{eval_exp_name}_masks" / "temp_masks" / file_name


def get_frame_intrinsics(frame: dict, meta: dict) -> tuple[float, float, float, float]:
    fl_x = frame.get("fl_x", meta.get("fl_x"))
    fl_y = frame.get("fl_y", meta.get("fl_y"))
    cx = frame.get("cx", meta.get("cx"))
    cy = frame.get("cy", meta.get("cy"))
    if None in (fl_x, fl_y, cx, cy):
        raise ValueError("Missing camera intrinsics. Expected fl_x, fl_y, cx and cy in frame or transforms root.")
    return float(fl_x), float(fl_y), float(cx), float(cy)


def z_depth_to_distance(depth: torch.Tensor, fl_x: float, fl_y: float, cx: float, cy: float) -> torch.Tensor:
    height, width = depth.shape[-2], depth.shape[-1]
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x_norm = (grid_x - cx) / fl_x
    y_norm = (grid_y - cy) / fl_y
    ray_norm = torch.sqrt(x_norm**2 + y_norm**2 + 1.0)
    return depth * ray_norm


def distance_to_z_depth(distance: torch.Tensor, fl_x: float, fl_y: float, cx: float, cy: float) -> torch.Tensor:
    height, width = distance.shape[-2], distance.shape[-1]
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x_norm = (grid_x - cx) / fl_x
    y_norm = (grid_y - cy) / fl_y
    ray_norm = torch.sqrt(x_norm**2 + y_norm**2 + 1.0)
    return distance / ray_norm


def convert_depth_for_comparison(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    frame: dict,
    meta: dict,
    compare_space: str,
    pred_depth_is_distance: bool,
    gt_depth_is_distance: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if compare_space not in {"distance", "z_depth"}:
        raise ValueError("--compare-space must be either 'distance' or 'z_depth'.")

    fl_x, fl_y, cx, cy = get_frame_intrinsics(frame, meta)

    if compare_space == "distance":
        if not pred_depth_is_distance:
            pred_depth = z_depth_to_distance(pred_depth, fl_x, fl_y, cx, cy)
        if not gt_depth_is_distance:
            gt_depth = z_depth_to_distance(gt_depth, fl_x, fl_y, cx, cy)
    else:
        if pred_depth_is_distance:
            pred_depth = distance_to_z_depth(pred_depth, fl_x, fl_y, cx, cy)
        if gt_depth_is_distance:
            gt_depth = distance_to_z_depth(gt_depth, fl_x, fl_y, cx, cy)

    return pred_depth, gt_depth


def calculate_rmse(errors: torch.Tensor) -> float:
    if len(errors) == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(errors**2)).item())


def calculate_mae(errors: torch.Tensor) -> float:
    if len(errors) == 0:
        return float("nan")
    return float(torch.mean(torch.abs(errors)).item())


def empty_stats() -> dict:
    return {"sum_sq": 0.0, "sum_abs": 0.0, "count": 0, "pred_missing": 0}


def update_stats(stats: dict, errors: torch.Tensor, pred_values: Optional[torch.Tensor] = None, min_depth: float = 1e-6) -> None:
    if len(errors) == 0:
        return
    stats["sum_sq"] += float(torch.sum(errors**2).item())
    stats["sum_abs"] += float(torch.sum(torch.abs(errors)).item())
    stats["count"] += int(len(errors))
    if pred_values is not None:
        stats["pred_missing"] += int(torch.count_nonzero(pred_values <= min_depth).item())


def summarize_stats(stats: dict) -> dict:
    if stats["count"] == 0:
        return {}

    rmse = float(np.sqrt(stats["sum_sq"] / stats["count"]))
    mae = float(stats["sum_abs"] / stats["count"])
    return {
        "pixels": int(stats["count"]),
        "pred_missing_pixels": int(stats["pred_missing"]),
        "rmse": rmse,
        "mae": mae,
        "mean_error": rmse,
    }


def depth_errors(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_raw_valid: torch.Tensor,
    gt_raw_valid: torch.Tensor,
    min_depth: float,
    max_depth: Optional[float],
    require_pred_valid: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"Depth shape mismatch: predicted={tuple(pred_depth.shape)}, gt={tuple(gt_depth.shape)}")
    if pred_raw_valid.shape != gt_depth.shape:
        raise ValueError(f"Pred valid mask shape mismatch: mask={tuple(pred_raw_valid.shape)}, depth={tuple(gt_depth.shape)}")
    if gt_raw_valid.shape != gt_depth.shape:
        raise ValueError(f"GT valid mask shape mismatch: mask={tuple(gt_raw_valid.shape)}, depth={tuple(gt_depth.shape)}")

    valid = pred_raw_valid & gt_raw_valid & (gt_depth > min_depth)
    if require_pred_valid:
        valid &= pred_depth > min_depth

    errors = pred_depth - gt_depth
    all_errors = errors[valid]
    all_pred = pred_depth[valid]
    return all_errors, all_pred, valid


def region_depth_errors(
    pred_depth: torch.Tensor,
    errors: torch.Tensor,
    valid: torch.Tensor,
    mask: Optional[torch.Tensor],
    gt_depth_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        empty = torch.empty(0, device=device)
        return empty, empty
    if mask.shape != gt_depth_shape:
        raise ValueError(f"Mask shape mismatch: mask={tuple(mask.shape)}, depth={tuple(gt_depth_shape)}")

    region = mask > 0.5
    return errors[valid & region], pred_depth[valid & region]


def calculate_depth_rmse_for_render(
    info: dict,
    renders_root: Path,
    render_name: str,
    gt_render_name: str,
    depth_scale: float = 0.001,
    min_depth: float = 1e-6,
    max_depth: Optional[float] = 10.0,
    require_pred_valid: bool = False,
    compare_space: str = "z_depth",
    pred_depth_is_distance: bool = True,
    gt_depth_is_distance: bool = True,
    max_frames: Optional[int] = None,
    save_results: bool = True,
) -> Optional[dict]:
    """Compare one rendered depth folder against after/nerfacto rendered GT depth."""
    data_path = data_path_from_render_info(info)
    gt_render_depth_dir = resolve_gt_render_depth_dir_from_info(renders_root, info, gt_render_name)
    pred_depth_dir = info["pred_depth_dir"]
    render_run_dir = info["render_run_dir"]
    eval_depth_dir = eval_depth_dir_from_render_info(info)

    transforms_path = data_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing {transforms_path}")

    with open(transforms_path, "r") as f:
        meta = json.load(f)

    all_stats = empty_stats()
    out_stats = empty_stats()
    in_stats = empty_stats()
    unmasked_stats = empty_stats()
    frame_results = []
    skipped_missing = 0

    frames = meta["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]
    print(
        f"Evaluating {info['task']}/{info['object_name']} "
        f"{info['method']} {info['run']} ({len(frames)} frames)..."
    )
    for frame_idx, frame in enumerate(tqdm(frames)):
        pred_depth_path = get_depth_render_path(pred_depth_dir, frame)
        gt_depth_path = get_depth_render_path(gt_render_depth_dir, frame)
        out_mask_path = get_render_mask_path(render_run_dir, render_name, frame)
        in_mask_path = get_render_mask_after_path(render_run_dir, render_name, frame)

        if not pred_depth_path.exists() or not gt_depth_path.exists():
            skipped_missing += 1
            continue

        pred_depth, pred_raw_valid = load_depth_with_valid_mask(pred_depth_path, depth_scale)
        gt_depth, gt_raw_valid = load_depth_with_valid_mask(gt_depth_path, depth_scale)
        pred_depth, gt_depth = convert_depth_for_comparison(
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            frame=frame,
            meta=meta,
            compare_space=compare_space,
            pred_depth_is_distance=pred_depth_is_distance,
            gt_depth_is_distance=gt_depth_is_distance,
        )
        if max_depth is not None:
            pred_depth = torch.clamp(pred_depth, max=max_depth)
            gt_depth = torch.clamp(gt_depth, max=max_depth)

        out_mask = None
        if out_mask_path.exists():
            out_mask = torch.tensor(cv2.imread(str(out_mask_path), cv2.IMREAD_GRAYSCALE), device=device).float() / 255.0
        in_mask = None
        if info["task"] == "reconfigure" and in_mask_path.exists():
            in_mask = torch.tensor(cv2.imread(str(in_mask_path), cv2.IMREAD_GRAYSCALE), device=device).float() / 255.0

        frame_all, frame_all_pred, valid_mask = depth_errors(
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            pred_raw_valid=pred_raw_valid,
            gt_raw_valid=gt_raw_valid,
            min_depth=min_depth,
            max_depth=max_depth,
            require_pred_valid=require_pred_valid,
        )
        pixel_errors = pred_depth - gt_depth
        frame_out, frame_out_pred = region_depth_errors(
            pred_depth=pred_depth,
            errors=pixel_errors,
            valid=valid_mask,
            mask=out_mask,
            gt_depth_shape=gt_depth.shape,
        )
        frame_unmasked = pixel_errors[valid_mask & ~(out_mask > 0.5)] if out_mask is not None else torch.empty(0, device=device)
        frame_unmasked_pred = pred_depth[valid_mask & ~(out_mask > 0.5)] if out_mask is not None else torch.empty(0, device=device)
        frame_in, frame_in_pred = region_depth_errors(
            pred_depth=pred_depth,
            errors=pixel_errors,
            valid=valid_mask,
            mask=in_mask,
            gt_depth_shape=gt_depth.shape,
        )

        if len(frame_all) == 0:
            continue

        update_stats(all_stats, frame_all, frame_all_pred, min_depth)
        update_stats(out_stats, frame_out, frame_out_pred, min_depth)
        update_stats(in_stats, frame_in, frame_in_pred, min_depth)
        update_stats(unmasked_stats, frame_unmasked, frame_unmasked_pred, min_depth)
        frame_results.append(
            {
                "frame_idx": frame_idx,
                "rmse": calculate_rmse(frame_all),
                "mae": calculate_mae(frame_all),
                "valid_pixels": int(len(frame_all)),
            }
        )

    if all_stats["count"] == 0:
        print(f"No valid depth frames found or missing depth maps for {pred_depth_dir}")
        return None

    results_dict = {
        "task": info["task"],
        "object_name": info["object_name"],
        "method": info["method"],
        "run": info["run"],
        "total_valid_frames_evaluated": len(frame_results),
        "skipped_missing_frames": skipped_missing,
        "depth_scale": depth_scale,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "require_pred_valid": require_pred_valid,
        "compare_space": compare_space,
        "pred_depth_is_distance": pred_depth_is_distance,
        "gt_depth_is_distance": gt_depth_is_distance,
        "gt_render_name": gt_render_name,
        "gt_render_depth_dir": str(gt_render_depth_dir),
        "render_name": render_name,
        "pred_depth_dir": str(pred_depth_dir),
        "all": summarize_stats(all_stats),
        "frames": frame_results,
    }

    if out_stats["count"] > 0:
        results_dict["out"] = summarize_stats(out_stats)
    if in_stats["count"] > 0:
        results_dict["in"] = summarize_stats(in_stats)
    if unmasked_stats["count"] > 0:
        results_dict["unmasked"] = summarize_stats(unmasked_stats)

    if save_results:
        os.makedirs(eval_depth_dir, exist_ok=True)
        out_json_path = eval_depth_dir / f"depth_rmse_results_{render_name.removesuffix('_edited')}.json"
        with open(out_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Saved results to {out_json_path}")

    return results_dict


def print_summary_table(results: list[dict]) -> None:
    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"rmse": [], "mae": [], "objects": set(), "files": 0})
        )
    )

    for result in results:
        task = result["task"]
        method = result["method"]
        object_name = result["object_name"]
        for region in ("all", "out", "in", "unmasked"):
            if region not in result:
                continue
            stats = grouped[region][task][method]
            stats["rmse"].append(result[region]["rmse"])
            stats["mae"].append(result[region]["mae"])
            stats["objects"].add(object_name)
            stats["files"] += 1

    print(f"\n{'=' * 72}")
    print("Depth RMSE Summary")
    print(f"{'=' * 72}")

    for region in ("all", "out", "in", "unmasked"):
        if region not in grouped:
            continue

        print(f"\n\n{'#' * 72}")
        print(f"REGION: {region.upper()}")
        print(f"{'#' * 72}")

        for task, methods in grouped[region].items():
            print(f"\nTask: {task.upper()}")
            print(f"{'-' * 72}")
            print(
                f"{'Method':<25} | {'Avg RMSE (m)':<14} | {'Avg MAE (m)':<12} | "
                f"{'Objects':<7} | {'Files':<5}"
            )
            print(f"{'-' * 72}")

            rows = []
            for method, stats in methods.items():
                avg_rmse = sum(stats["rmse"]) / len(stats["rmse"])
                avg_mae = sum(stats["mae"]) / len(stats["mae"])
                rows.append((method, avg_rmse, avg_mae, len(stats["objects"]), stats["files"]))

            for method, avg_rmse, avg_mae, object_count, file_count in sorted(rows, key=lambda row: row[1]):
                print(
                    f"{method:<25} | {avg_rmse:<14.4f} | {avg_mae:<12.4f} | "
                    f"{object_count:<7} | {file_count:<5}"
                )


def calculate_depth_rmse(
    renders_root: Path = Path("renders"),
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    render_name: str = "full",
    gt_render_name: str = "full",
    depth_scale: float = 0.001,
    min_depth: float = 1e-6,
    max_depth: Optional[float] = 2.0,
    require_pred_valid: bool = False,
    compare_space: str = "z_depth",
    pred_depth_is_distance: bool = True,
    gt_depth_is_distance: bool = True,
    max_frames: Optional[int] = None,
    save_results: bool = True,
):
    """
    Discover rendered depth folders and compare them against after/nerfacto depth.

    The script searches renders/real/{remove,reconfigure}/.../before for
    render_name/depth folders, uses the matching after/nerfacto render as ground
    truth, writes per-render JSON results, and prints an aggregate table.
    Rendered depths are true z-depth PNGs in millimeters, so z-depth comparison
    is the default. Depths beyond max_depth are treated as invalid to exclude
    far-plane/saturated background pixels.
    """
    render_infos = discover_render_depth_dirs(renders_root, tasks, render_name)
    if not render_infos:
        print(f"No render depth folders found under {renders_root} for {tasks} and render '{render_name}'.")
        return

    print(f"Found {len(render_infos)} render depth folders.")
    results = []
    failures = []
    for info in render_infos:
        try:
            result = calculate_depth_rmse_for_render(
                info=info,
                renders_root=renders_root,
                render_name=render_name,
                gt_render_name=gt_render_name,
                depth_scale=depth_scale,
                min_depth=min_depth,
                max_depth=max_depth,
                require_pred_valid=require_pred_valid,
                compare_space=compare_space,
                pred_depth_is_distance=pred_depth_is_distance,
                gt_depth_is_distance=gt_depth_is_distance,
                max_frames=max_frames,
                save_results=save_results,
            )
        except Exception as exc:
            failures.append((info, exc))
            print(f"Failed {info['task']}/{info['object_name']} {info['method']} {info['run']}: {exc}")
            continue

        if result is not None:
            results.append(result)

    if results:
        print_summary_table(results)

    if failures:
        print(f"\nFailed evaluations: {len(failures)}")
        for info, exc in failures:
            print(f"- {info['task']}/{info['object_name']} {info['method']} {info['run']}: {exc}")


if __name__ == "__main__":
    tyro.cli(calculate_depth_rmse)
