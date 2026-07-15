import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TASKS = ("remove", "translate_and_rotate")


def load_depth(path: Path, depth_scale: float) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(path)
    return torch.tensor(depth.astype(np.float32), device=device) * depth_scale


def load_rgb(path: Path) -> torch.Tensor:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.tensor(rgb, device=device)


def get_rgb_path(data_path: Path, frame: dict) -> Path:
    return data_path / frame["file_path"]


def load_mask(path: Path, shape: torch.Size) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_t = torch.tensor(mask.astype(np.float32), device=device) / 255.0
    if mask_t.shape != shape:
        return None
    return mask_t > 0.5


def frame_name(frame: dict) -> str:
    return Path(frame["file_path"]).name


def get_intrinsics(frame: dict, meta: dict) -> tuple[float, float, float, float]:
    fl_x = frame.get("fl_x", meta.get("fl_x"))
    fl_y = frame.get("fl_y", meta.get("fl_y"))
    cx = frame.get("cx", meta.get("cx"))
    cy = frame.get("cy", meta.get("cy"))
    if None in (fl_x, fl_y, cx, cy):
        raise ValueError("Missing intrinsics in frame or transforms root.")
    return float(fl_x), float(fl_y), float(cx), float(cy)


def pixel_grid(height: int, width: int, stride: int) -> torch.Tensor:
    ys = torch.arange(0, height, stride, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, stride, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def sample_values(image: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[:2]
    u = points[:, 0]
    v = points[:, 1]
    in_bounds = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    if len(points) == 0:
        return torch.empty((0,) + image.shape[2:], device=device), in_bounds

    gx = (u / (width - 1)) * 2.0 - 1.0
    gy = (v / (height - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    if image.ndim == 2:
        image_chw = image.view(1, 1, height, width)
        sampled = torch.nn.functional.grid_sample(
            image_chw, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        ).view(-1)
    else:
        image_chw = image.permute(2, 0, 1).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            image_chw, grid, align_corners=True, mode="bilinear", padding_mode="zeros"
        ).squeeze(0).squeeze(-1).T
    return sampled, in_bounds


def sample_debug_points(num_points: int, max_points: int) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)
    return np.linspace(0, num_points - 1, max_points, dtype=np.int64)


def tensor_rgb_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    rgb = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def colorize_depth(depth: torch.Tensor, min_depth: float, max_depth: Optional[float]) -> np.ndarray:
    depth_np = depth.detach().cpu().numpy()
    valid = depth_np > min_depth
    if max_depth is not None:
        valid &= depth_np < max_depth
    if valid.any():
        near = min_depth
        far = max_depth if max_depth is not None else float(np.percentile(depth_np[valid], 95))
        far = max(far, near + 1e-6)
        normalized = np.clip((depth_np - near) / (far - near), 0.0, 1.0)
    else:
        normalized = np.zeros_like(depth_np, dtype=np.float32)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def save_tree_debug_visualizations(
    debug_out_dir: Path,
    pair_idx: int,
    rgb1: torch.Tensor,
    rgb2: torch.Tensor,
    depth1: torch.Tensor,
    depth2: torch.Tensor,
    pixels1: torch.Tensor,
    pixels2: torch.Tensor,
    pixels1_roundtrip: Optional[torch.Tensor],
    mask_values: Optional[torch.Tensor],
    min_depth: float,
    max_depth: Optional[float],
    max_debug_points: int,
) -> None:
    os.makedirs(debug_out_dir, exist_ok=True)
    if len(pixels1) == 0:
        return

    img1 = tensor_rgb_to_bgr_uint8(rgb1)
    img2 = tensor_rgb_to_bgr_uint8(rgb2)
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    sample_idx = sample_debug_points(len(pixels1), max_debug_points)
    np_pixels1 = pixels1.detach().cpu().numpy()[sample_idx]
    np_pixels2 = pixels2.detach().cpu().numpy()[sample_idx]
    np_pixels1_roundtrip = None
    if pixels1_roundtrip is not None:
        np_pixels1_roundtrip = pixels1_roundtrip.detach().cpu().numpy()[sample_idx]
    if mask_values is None:
        np_mask = np.zeros(len(np_pixels1), dtype=bool)
    else:
        np_mask = mask_values.detach().cpu().numpy()[sample_idx].astype(bool)

    def selectors() -> list[tuple[str, np.ndarray]]:
        all_selector = np.ones(len(np_pixels1), dtype=bool)
        if mask_values is None:
            return [("all", all_selector)]
        out = [("all", all_selector)]
        if np_mask.any():
            out.append(("out", np_mask))
        if (~np_mask).any():
            out.append(("unmasked", ~np_mask))
        return out

    def draw_points(base_img: np.ndarray, points: np.ndarray, suffix: str, kind: str, color: tuple[int, int, int]) -> None:
        canvas = base_img.copy()
        for pt in points:
            cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 2, color, -1)
        cv2.imwrite(str(debug_out_dir / f"tree_{kind}_{suffix}_{pair_idx:04d}.jpg"), canvas)

    def draw_correspondences(suffix: str, selector: np.ndarray) -> None:
        if np_pixels1_roundtrip is None:
            return
        canvas = img1.copy()
        for src_pt, dst_pt in zip(np_pixels1[selector], np_pixels1_roundtrip[selector]):
            src = (int(round(src_pt[0])), int(round(src_pt[1])))
            dst = (int(round(dst_pt[0])), int(round(dst_pt[1])))
            if not (0 <= src[0] < width1 and 0 <= src[1] < height1):
                continue
            if not (0 <= dst[0] < width1 and 0 <= dst[1] < height1):
                continue
            cv2.circle(canvas, src, 2, (255, 0, 0), -1)
            cv2.circle(canvas, dst, 2, (0, 0, 255), -1)
            cv2.line(canvas, src, dst, (0, 255, 255), 1)
        cv2.imwrite(str(debug_out_dir / f"tree_correspondence_{suffix}_{pair_idx:04d}.jpg"), canvas)

    def draw_roundtrip(suffix: str, selector: np.ndarray) -> None:
        if np_pixels1_roundtrip is None:
            return
        canvas = img1.copy()
        for src_pt, roundtrip_pt in zip(np_pixels1[selector], np_pixels1_roundtrip[selector]):
            src = (int(round(src_pt[0])), int(round(src_pt[1])))
            roundtrip = (int(round(roundtrip_pt[0])), int(round(roundtrip_pt[1])))
            if not (0 <= src[0] < width1 and 0 <= src[1] < height1):
                continue
            if not (0 <= roundtrip[0] < width1 and 0 <= roundtrip[1] < height1):
                continue
            cv2.circle(canvas, src, 2, (255, 0, 0), -1)
            cv2.circle(canvas, roundtrip, 2, (0, 0, 255), -1)
            cv2.line(canvas, src, roundtrip, (0, 255, 255), 1)
        cv2.imwrite(str(debug_out_dir / f"tree_roundtrip_{suffix}_{pair_idx:04d}.jpg"), canvas)

    def draw_pointcloud_cast(suffix: str, selector: np.ndarray) -> None:
        canvas = np.zeros_like(img2)
        for src_pt, dst_pt in zip(np_pixels1[selector], np_pixels2[selector]):
            src_x = int(round(src_pt[0]))
            src_y = int(round(src_pt[1]))
            dst_x = int(round(dst_pt[0]))
            dst_y = int(round(dst_pt[1]))
            if not (0 <= src_x < width1 and 0 <= src_y < height1):
                continue
            if not (0 <= dst_x < width2 and 0 <= dst_y < height2):
                continue
            cv2.circle(canvas, (dst_x, dst_y), 1, tuple(int(v) for v in img1[src_y, src_x]), -1)
        cv2.imwrite(str(debug_out_dir / f"tree_pointcloud_cast_{suffix}_{pair_idx:04d}.jpg"), canvas)

    cv2.imwrite(str(debug_out_dir / f"tree_depth_src_{pair_idx:04d}.jpg"), colorize_depth(depth1, min_depth, max_depth))
    cv2.imwrite(str(debug_out_dir / f"tree_depth_dst_{pair_idx:04d}.jpg"), colorize_depth(depth2, min_depth, max_depth))

    for suffix, selector in selectors():
        draw_points(img1, np_pixels1[selector], suffix, "source_points", (255, 0, 0))
        draw_points(img2, np_pixels2[selector], suffix, "projected_points", (0, 0, 255))
        draw_correspondences(suffix, selector)
        draw_roundtrip(suffix, selector)
        draw_pointcloud_cast(suffix, selector)


def unproject(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    depth_is_distance: bool,
) -> torch.Tensor:
    fl_x, fl_y, cx, cy = intrinsics
    x_norm = (pixels[:, 0] - cx) / fl_x
    y_norm = (pixels[:, 1] - cy) / fl_y
    if depth_is_distance:
        ray_norm = torch.sqrt(x_norm**2 + y_norm**2 + 1.0)
        z_depth = depth / ray_norm
    else:
        z_depth = depth
    x = x_norm * z_depth
    y = y_norm * z_depth
    return torch.stack([x, y, z_depth], dim=-1)


def opencv_to_opengl(points: torch.Tensor) -> torch.Tensor:
    out = points.clone()
    out[:, 1] *= -1
    out[:, 2] *= -1
    return out


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=-1)
    return (transform @ points_h.T).T[:, :3]


def project(points_cv: torch.Tensor, intrinsics: tuple[float, float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    fl_x, fl_y, cx, cy = intrinsics
    z = points_cv[:, 2]
    u = points_cv[:, 0] * fl_x / z + cx
    v = points_cv[:, 1] * fl_y / z + cy
    return torch.stack([u, v], dim=-1), z


def project_frame_to_frame(
    depth1: torch.Tensor,
    c2w1: torch.Tensor,
    c2w2: torch.Tensor,
    intrinsics1: tuple[float, float, float, float],
    intrinsics2: tuple[float, float, float, float],
    pixel_stride: int,
    min_depth: float,
    max_depth: Optional[float],
    depth_is_distance: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = depth1.shape
    pixels1 = pixel_grid(height, width, pixel_stride)
    z1 = depth1[pixels1[:, 1].long(), pixels1[:, 0].long()]
    valid = z1 > min_depth
    if max_depth is not None:
        valid &= z1 < max_depth

    pixels1 = pixels1[valid]
    z1 = z1[valid]
    if len(pixels1) == 0:
        empty = torch.empty((0, 2), device=device)
        return empty, empty, torch.empty(0, device=device)

    pts1_cv = unproject(pixels1, z1, intrinsics1, depth_is_distance)
    pts1_gl = opencv_to_opengl(pts1_cv)
    pts_world = transform_points(pts1_gl, c2w1)
    pts2_gl = transform_points(pts_world, torch.linalg.inv(c2w2))
    pts2_cv = opencv_to_opengl(pts2_gl)
    pixels2, z2 = project(pts2_cv, intrinsics2)
    return pixels1, pixels2, z2


def pair_errors(
    render_dir: Path,
    data_path: Path,
    out_mask_dir: Path,
    in_mask_dir: Optional[Path],
    use_in_mask: bool,
    frames: list[dict],
    meta: dict,
    i: int,
    step: int,
    pixel_stride: int,
    depth_scale: float,
    min_depth: float,
    max_depth: Optional[float],
    depth_consistency_thresh: Optional[float],
    depth_is_distance: bool,
    debug_out_dir: Optional[Path] = None,
    debug_max_points: int = 3000,
) -> Optional[dict]:
    f1 = frames[i]
    f2 = frames[i + step]
    name1 = frame_name(f1)
    name2 = frame_name(f2)

    depth1_path = render_dir / "depth" / name1
    depth2_path = render_dir / "depth" / name2
    rgb1_path = get_rgb_path(data_path, f1)
    rgb2_path = get_rgb_path(data_path, f2)
    if not all(path.exists() for path in (depth1_path, depth2_path, rgb1_path, rgb2_path)):
        return None

    depth1 = load_depth(depth1_path, depth_scale)
    depth2 = load_depth(depth2_path, depth_scale)
    rgb1 = load_rgb(rgb1_path)
    rgb2 = load_rgb(rgb2_path)
    out_mask1 = load_mask(out_mask_dir / name1, depth1.shape)
    out_mask2 = load_mask(out_mask_dir / name2, depth2.shape)
    in_mask1 = load_mask(in_mask_dir / name1, depth1.shape) if use_in_mask and in_mask_dir is not None else None
    in_mask2 = load_mask(in_mask_dir / name2, depth2.shape) if use_in_mask and in_mask_dir is not None else None

    c2w1 = torch.tensor(f1["transform_matrix"], device=device).float()
    c2w2 = torch.tensor(f2["transform_matrix"], device=device).float()
    intr1 = get_intrinsics(f1, meta)
    intr2 = get_intrinsics(f2, meta)

    pixels1, pixels2, z2_projected = project_frame_to_frame(
        depth1, c2w1, c2w2, intr1, intr2, pixel_stride, min_depth, max_depth, depth_is_distance
    )
    if len(pixels1) == 0:
        return None

    depth2_samples, in_bounds = sample_values(depth2, pixels2)
    valid = in_bounds & (z2_projected > min_depth) & (depth2_samples > min_depth)
    if max_depth is not None:
        valid &= (z2_projected < max_depth) & (depth2_samples < max_depth)
    depth2_points_cv = unproject(pixels2, depth2_samples, intr2, depth_is_distance)
    if depth_consistency_thresh is not None:
        valid &= torch.abs(depth2_points_cv[:, 2] - z2_projected) < depth_consistency_thresh

    pixels1 = pixels1[valid]
    pixels2 = pixels2[valid]
    depth2_points_cv = depth2_points_cv[valid]
    if len(pixels1) == 0:
        return None

    depth2_points_gl = opencv_to_opengl(depth2_points_cv)
    world_from_depth2 = transform_points(depth2_points_gl, c2w2)
    points1_roundtrip_gl = transform_points(world_from_depth2, torch.linalg.inv(c2w1))
    points1_roundtrip_cv = opencv_to_opengl(points1_roundtrip_gl)
    pixels1_roundtrip, _ = project(points1_roundtrip_cv, intr1)

    rgb2_samples, _ = sample_values(rgb2, pixels2)
    rgb1_values = rgb1[pixels1[:, 1].long(), pixels1[:, 0].long()]
    perceptual = torch.mean(torch.abs(rgb1_values - rgb2_samples), dim=-1)

    def geometric_one_way(src_depth, dst_depth, src_frame, dst_frame, src_intr, dst_intr):
        src_c2w = torch.tensor(src_frame["transform_matrix"], device=device).float()
        dst_c2w = torch.tensor(dst_frame["transform_matrix"], device=device).float()
        src_pixels, dst_pixels, dst_z_projected = project_frame_to_frame(
            src_depth, src_c2w, dst_c2w, src_intr, dst_intr, pixel_stride, min_depth, max_depth, depth_is_distance
        )
        dst_depth_samples, dst_in_bounds = sample_values(dst_depth, dst_pixels)
        good = dst_in_bounds & (dst_z_projected > min_depth) & (dst_depth_samples > min_depth)
        if max_depth is not None:
            good &= (dst_z_projected < max_depth) & (dst_depth_samples < max_depth)
        dst_pts_cv = unproject(dst_pixels, dst_depth_samples, dst_intr, depth_is_distance)
        if depth_consistency_thresh is not None:
            good &= torch.abs(dst_pts_cv[:, 2] - dst_z_projected) < depth_consistency_thresh
        src_pixels = src_pixels[good]
        dst_pixels = dst_pixels[good]
        dst_pts_cv = dst_pts_cv[good]
        if len(src_pixels) == 0:
            return torch.empty(0, device=device), torch.empty((0, 2), device=device)

        dst_pts_gl = opencv_to_opengl(dst_pts_cv)
        world_from_dst_depth = transform_points(dst_pts_gl, dst_c2w)
        src_pts_gl = transform_points(world_from_dst_depth, torch.linalg.inv(src_c2w))
        src_pts_cv = opencv_to_opengl(src_pts_gl)
        src_roundtrip_pixels, src_roundtrip_z = project(src_pts_cv, src_intr)
        valid_roundtrip = src_roundtrip_z > min_depth
        if max_depth is not None:
            valid_roundtrip &= src_roundtrip_z < max_depth

        errors = torch.linalg.norm(src_roundtrip_pixels[valid_roundtrip] - src_pixels[valid_roundtrip], dim=-1)
        src_pixels = src_pixels[valid_roundtrip]
        return errors, src_pixels

    def split_errors_by_mask(errors: torch.Tensor, pixels: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or len(errors) == 0:
            return torch.empty(0, device=device)
        mask_values = mask[pixels[:, 1].long(), pixels[:, 0].long()]
        return errors[mask_values]

    def split_errors_by_inverse_mask(errors: torch.Tensor, pixels: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None or len(errors) == 0:
            return torch.empty(0, device=device)
        mask_values = mask[pixels[:, 1].long(), pixels[:, 0].long()]
        return errors[~mask_values]

    geom_forward, geom_forward_pixels = geometric_one_way(
        depth1, depth2, f1, f2, intr1, intr2
    )
    geom_backward, geom_backward_pixels = geometric_one_way(
        depth2, depth1, f2, f1, intr2, intr1
    )
    geometric = torch.cat([geom_forward, geom_backward], dim=0)
    geometric_out = torch.cat(
        [
            split_errors_by_mask(geom_forward, geom_forward_pixels, out_mask1),
            split_errors_by_mask(geom_backward, geom_backward_pixels, out_mask2),
        ],
        dim=0,
    )
    geometric_in = torch.cat(
        [
            split_errors_by_mask(geom_forward, geom_forward_pixels, in_mask1),
            split_errors_by_mask(geom_backward, geom_backward_pixels, in_mask2),
        ],
        dim=0,
    )
    geometric_unmasked = torch.cat(
        [
            split_errors_by_inverse_mask(geom_forward, geom_forward_pixels, out_mask1),
            split_errors_by_inverse_mask(geom_backward, geom_backward_pixels, out_mask2),
        ],
        dim=0,
    )

    if out_mask1 is None:
        perceptual_out = torch.empty(0, device=device)
        perceptual_unmasked = torch.empty(0, device=device)
        perceptual_out_mask_values = None
    else:
        perceptual_out_mask_values = out_mask1[pixels1[:, 1].long(), pixels1[:, 0].long()]
        perceptual_out = perceptual[perceptual_out_mask_values]
        perceptual_unmasked = perceptual[~perceptual_out_mask_values]

    if in_mask1 is None:
        perceptual_in = torch.empty(0, device=device)
    else:
        perceptual_in_mask_values = in_mask1[pixels1[:, 1].long(), pixels1[:, 0].long()]
        perceptual_in = perceptual[perceptual_in_mask_values]

    if debug_out_dir is not None:
        save_tree_debug_visualizations(
            debug_out_dir=debug_out_dir,
            pair_idx=i,
            rgb1=rgb1,
            rgb2=rgb2,
            depth1=depth1,
            depth2=depth2,
            pixels1=pixels1,
            pixels2=pixels2,
            pixels1_roundtrip=pixels1_roundtrip,
            mask_values=perceptual_out_mask_values,
            min_depth=min_depth,
            max_depth=max_depth,
            max_debug_points=debug_max_points,
        )

    return {
        "pair": (i, i + step),
        "perceptual": perceptual,
        "perceptual_out": perceptual_out,
        "perceptual_in": perceptual_in,
        "perceptual_unmasked": perceptual_unmasked,
        "geometric": geometric,
        "geometric_out": geometric_out,
        "geometric_in": geometric_in,
        "geometric_unmasked": geometric_unmasked,
    }


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def parse_render_depth_dir(renders_root: Path, depth_dir: Path) -> Optional[dict]:
    try:
        parts = depth_dir.relative_to(renders_root).parts
        before_idx = parts.index("before")
    except ValueError:
        return None
    if len(parts) <= before_idx + 4:
        return None
    return {
        "task": parts[1],
        "object_group": parts[3],
        "object_name": parts[4],
        "method": parts[before_idx + 1],
        "run": parts[before_idx + 2],
        "render_dir": depth_dir.parents[0],
        "run_dir": depth_dir.parents[1],
    }


def discover_render_infos(renders_root: Path, tasks: tuple[str, ...], render_name: str) -> list[dict]:
    infos = []
    for task in tasks:
        for depth_dir in sorted((renders_root / "real" / task).glob(f"**/before/*/*/{render_name}/depth")):
            parsed = parse_render_depth_dir(renders_root, depth_dir)
            if parsed is not None:
                infos.append(parsed)
    return infos


def data_path_from_info(info: dict) -> Path:
    return (
        Path("data")
        / "real"
        / info["task"]
        / "single"
        / info["object_group"]
        / info["object_name"]
        / "after"
    )


def mask_dir_from_info(info: dict, render_name: str) -> Path:
    eval_exp_name = render_name.removesuffix("_edited")
    return info["run_dir"] / f"{eval_exp_name}_masks" / "temp_masks"


def mask_after_dir_from_info(info: dict, render_name: str) -> Path:
    eval_exp_name = render_name.removesuffix("_edited")
    return info["run_dir"] / f"{eval_exp_name}_masks_after" / "temp_masks"


def output_dir_from_info(info: dict) -> Path:
    return (
        Path("eval_reprojection_tree")
        / "real"
        / info["task"]
        / "single"
        / info["object_group"]
        / info["object_name"]
        / "before"
        / info["method"]
        / info["run"]
    )


def print_table(results: list[dict]) -> None:
    grouped = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"perceptual": [], "geometric": [], "objects": set(), "files": 0}))
    )
    for result in results:
        for region in ("all", "out", "in", "unmasked"):
            if region not in result:
                continue
            bucket = grouped[region][result["task"]][result["method"]]
            bucket["perceptual"].append(result[region]["perceptual"]["mean"])
            bucket["geometric"].append(result[region]["geometric"]["mean"])
            bucket["objects"].add(result["object_name"])
            bucket["files"] += 1

    print(f"\n{'=' * 86}")
    print("Reprojection Summary")
    print(f"{'=' * 86}")

    for region in ("all", "out", "in", "unmasked"):
        if region not in grouped:
            continue

        print(f"\n\n{'#' * 86}")
        print(f"REGION: {region.upper()}")
        print(f"{'#' * 86}")

        for task, methods in grouped[region].items():
            print(f"\nTask: {task.upper()}")
            print("-" * 86)
            print(f"{'Method':<25} | {'Perceptual L1':<14} | {'Geometric px':<14} | {'Objects':<7} | {'Files':<5}")
            print("-" * 86)
            rows = []
            for method, values in methods.items():
                rows.append(
                    (
                        method,
                        float(np.mean(values["perceptual"])),
                        float(np.mean(values["geometric"])),
                        len(values["objects"]),
                        values["files"],
                    )
                )
            for method, perceptual, geometric, objects, files in sorted(rows, key=lambda row: row[2]):
                print(f"{method:<25} | {perceptual:<14.5f} | {geometric:<14.4f} | {objects:<7} | {files:<5}")


def calculate_reprojection_tree(
    renders_root: Path = Path("renders"),
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    render_name: str = "full_edited",
    step: int = 20,
    pixel_stride: int = 4,
    depth_scale: float = 0.001,
    depth_is_distance: bool = True,
    min_depth: float = 1e-6,
    max_depth: Optional[float] = 2.0,
    depth_consistency_thresh: Optional[float] = 0.05,
    max_pairs: Optional[int] = None,
    debug_limit: int = 0,
    debug_out_dir: Optional[Path] = None,
    debug_max_points: int = 3000,
    save_results: bool = True,
) -> None:
    """
    Evaluate reprojection errors for every before render folder.

    Perceptual error is one-way RGB L1 after projecting frame i into i+step.
    Geometric error is two-way round-trip pixel reprojection error: i->i+step
    and i+step->i, concatenated before aggregation.

    NEO render folders store range/distance depth, so depth_is_distance=True is
    the default. Set it to False only when evaluating camera z-depth maps.

    Set debug_limit > 0 to save debug images for that many valid pairs per
    render folder. If debug_out_dir is omitted, images are written beside the
    JSON output for each render folder.
    """
    infos = discover_render_infos(renders_root, tasks, render_name)
    if not infos:
        print(f"No render folders found under {renders_root}.")
        return

    print(f"Found {len(infos)} render folders.")
    all_results = []
    failures = []
    for info in infos:
        data_path = data_path_from_info(info)
        transforms_path = data_path / "transforms.json"
        if not transforms_path.exists():
            failures.append((info, f"missing transforms: {transforms_path}"))
            continue
        meta = json.loads(transforms_path.read_text())
        frames = meta["frames"]
        total_pairs = max(0, len(frames) - step)
        if max_pairs is not None:
            total_pairs = min(total_pairs, max_pairs)

        perceptual = []
        perceptual_out = []
        perceptual_in = []
        perceptual_unmasked = []
        geometric = []
        geometric_out = []
        geometric_in = []
        geometric_unmasked = []
        pairs = []
        debug_saved = 0
        print(f"Evaluating {info['task']}/{info['object_name']} {info['method']} ({total_pairs} pairs)...")
        for i in tqdm(range(total_pairs)):
            pair_debug_out_dir = None
            if debug_limit > 0 and debug_saved < debug_limit:
                if debug_out_dir is None:
                    pair_debug_out_dir = (
                        output_dir_from_info(info) / f"debug_reprojection_step_{step}_{render_name.removesuffix('_edited')}"
                    )
                else:
                    pair_debug_out_dir = (
                        debug_out_dir
                        / info["task"]
                        / info["object_group"]
                        / info["object_name"]
                        / info["method"]
                        / info["run"]
                    )
            try:
                result = pair_errors(
                    render_dir=info["render_dir"],
                    data_path=data_path,
                    out_mask_dir=mask_dir_from_info(info, render_name),
                    in_mask_dir=mask_after_dir_from_info(info, render_name),
                    use_in_mask=info["task"] == "translate_and_rotate",
                    frames=frames,
                    meta=meta,
                    i=i,
                    step=step,
                    pixel_stride=pixel_stride,
                    depth_scale=depth_scale,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    depth_consistency_thresh=depth_consistency_thresh,
                    depth_is_distance=depth_is_distance,
                    debug_out_dir=pair_debug_out_dir,
                    debug_max_points=debug_max_points,
                )
            except Exception:
                result = None
            if result is None:
                continue
            if pair_debug_out_dir is not None:
                debug_saved += 1

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
                        float(result["perceptual_out"].mean().item())
                        if len(result["perceptual_out"])
                        else None
                    ),
                    "perceptual_in_mean": (
                        float(result["perceptual_in"].mean().item())
                        if len(result["perceptual_in"])
                        else None
                    ),
                    "perceptual_unmasked_mean": (
                        float(result["perceptual_unmasked"].mean().item())
                        if len(result["perceptual_unmasked"])
                        else None
                    ),
                    "geometric_mean": float(result["geometric"].mean().item()) if len(result["geometric"]) else None,
                    "geometric_out_mean": (
                        float(result["geometric_out"].mean().item()) if len(result["geometric_out"]) else None
                    ),
                    "geometric_in_mean": (
                        float(result["geometric_in"].mean().item()) if len(result["geometric_in"]) else None
                    ),
                    "geometric_unmasked_mean": (
                        float(result["geometric_unmasked"].mean().item())
                        if len(result["geometric_unmasked"])
                        else None
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
            failures.append((info, "no valid reprojection pairs"))
            continue

        result_dict = {
            "task": info["task"],
            "object_name": info["object_name"],
            "method": info["method"],
            "run": info["run"],
            "step": step,
            "pixel_stride": pixel_stride,
            "depth_scale": depth_scale,
            "depth_is_distance": depth_is_distance,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "depth_consistency_thresh": depth_consistency_thresh,
            "debug_limit": debug_limit,
            "debug_max_points": debug_max_points,
            "render_dir": str(info["render_dir"]),
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
        all_results.append(result_dict)
        if save_results:
            out_dir = output_dir_from_info(info)
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / f"reprojection_step_{step}_{render_name.removesuffix('_edited')}.json"
            out_path.write_text(json.dumps(result_dict, indent=4))

    if all_results:
        print_table(all_results)
    if failures:
        print(f"\nFailures: {len(failures)}")
        for info, reason in failures:
            print(f"- {info['task']}/{info['object_name']} {info['method']} {info['run']}: {reason}")


if __name__ == "__main__":
    tyro.cli(calculate_reprojection_tree)
