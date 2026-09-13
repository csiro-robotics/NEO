#!/usr/bin/env python3
"""Render RGB and true depth images for after/nerfacto scenes."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


RENDER_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_CONDA_ENV = "neo"
DEFAULT_TASKS = ("remove", "translate_and_rotate")
STATE_NAME = "after"
METHOD_NAME = "nerfacto"


def default_ns_render_command() -> str:
    conda_ns_render = Path.home() / ".conda" / "envs" / DEFAULT_CONDA_ENV / "bin" / "ns-render-neo"
    if conda_ns_render.exists():
        return str(conda_ns_render)

    ns_render = shutil.which("ns-render-neo")
    if ns_render:
        return ns_render

    sibling_ns_render = Path(sys.executable).with_name("ns-render-neo")
    if sibling_ns_render.exists():
        return str(sibling_ns_render)

    return "ns-render-neo"


def normalize_tasks(tasks: list[str]) -> list[str]:
    normalized = []
    for task in tasks:
        if task == "reomve":
            task = "remove"
        if task not in DEFAULT_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Choose from: {', '.join(DEFAULT_TASKS)}"
            )
        if task not in normalized:
            normalized.append(task)
    return normalized


def discover_configs(outputs_root: Path, tasks: list[str]) -> list[Path]:
    search_roots = [outputs_root / task for task in tasks]
    configs: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            print(f"Skipping missing directory: {search_root}")
            continue

        for config_path in sorted(search_root.rglob("config.yml")):
            parts = config_path.parts
            if STATE_NAME not in parts:
                continue

            state_index = parts.index(STATE_NAME)
            if state_index + 1 < len(parts) and parts[state_index + 1] == METHOD_NAME:
                configs.append(config_path)

    return configs


def build_scene_render_dir(config_path: Path, outputs_root: Path, output_root: Path) -> Path:
    return output_root / config_path.parent.relative_to(outputs_root)


def has_images(image_dir: Path) -> bool:
    return image_dir.exists() and any(
        path.is_file() and path.suffix.lower() in RENDER_SUFFIXES
        for path in image_dir.rglob("*")
    )


def has_true_depth_images(depth_dir: Path) -> bool:
    depth_paths = sorted(path for path in depth_dir.rglob("*.png") if path.is_file())
    if not depth_paths:
        return False

    import cv2

    depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_ANYDEPTH)
    return depth is not None and str(depth.dtype) == "uint16"


def has_existing_render(render_dir: Path, render_name: str) -> bool:
    return has_images(render_dir / render_name / "rgb") and has_true_depth_images(render_dir / render_name / "depth")


def read_config_data_path(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf8") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    data_path = getattr(config, "data", None)
    if data_path is None:
        datamanager = getattr(config.pipeline, "datamanager", None)
        data_path = getattr(datamanager, "data", None)
    if data_path is None:
        raise ValueError(f"Could not resolve data path from config: {config_path}")
    return Path(data_path)


def resolve_after_transforms_path(config_path: Path, outputs_root: Path) -> Path:
    config_data_path = read_config_data_path(config_path)
    transforms_path = config_data_path / "transforms.json"
    if transforms_path.exists():
        return transforms_path

    relative_parts = config_path.parent.relative_to(outputs_root).parts
    state_index = relative_parts.index(STATE_NAME)
    inferred_data_path = Path("data") / "real" / Path(*relative_parts[: state_index + 1])
    transforms_path = inferred_data_path / "transforms.json"
    if transforms_path.exists():
        return transforms_path

    raise FileNotFoundError(f"Could not find after transforms.json for {config_path}")


def build_command(
    ns_render_command: str,
    config_path: Path,
    output_path: Path,
    transforms_path: Path,
    eval_num_rays_per_chunk: int | None,
) -> list[str]:
    command = [
        ns_render_command,
        "render-from-transforms",
        "--load-config",
        str(config_path),
        "--output-path",
        str(output_path),
        "--output-format",
        "images",
        "--image-format",
        "png",
        "--rendered-output-names",
        "rgb",
        "depth",
        "--transforms-filename",
        str(transforms_path),
    ]

    if eval_num_rays_per_chunk is not None:
        command.extend(["--eval-num-rays-per-chunk", str(eval_num_rays_per_chunk)])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render rgb/depth image folders only for after/nerfacto configs under remove and "
            "translate_and_rotate using after transforms and the existing "
            "renders/.../full_edited/{rgb,depth} layout."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root searched for after/nerfacto/<run>/config.yml files. before configs are ignored.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("renders"),
        help="Root where rendered folders will be mirrored.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task directories below outputs-root to render. Supports: remove, translate_and_rotate.",
    )
    parser.add_argument(
        "--ns-render",
        default=default_ns_render_command(),
        help="Path to the ns-render-neo executable.",
    )
    parser.add_argument(
        "--render-name",
        default="full_edited",
        help="Leaf folder name containing rgb and depth render folders.",
    )
    parser.add_argument(
        "--eval-num-rays-per-chunk",
        type=int,
        default=None,
        help="Optional ray chunk size passed to ns-render-neo.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Render scenes even when rgb and true-depth images already exist.",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("renders/failed_after_nerfacto_renders.txt"),
        help="Path where failed render commands will be recorded.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed render.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running ns-render-neo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    configs = discover_configs(args.outputs_root, tasks)
    if not configs:
        print(f"No after/nerfacto configs found under {args.outputs_root} for tasks: {', '.join(tasks)}")
        return

    print(f"Found {len(configs)} after/nerfacto configs.")
    skipped = 0
    failures = []

    for index, config_path in enumerate(configs, start=1):
        render_dir = build_scene_render_dir(config_path, args.outputs_root, args.output_root)
        output_path = render_dir / args.render_name

        if has_existing_render(render_dir, args.render_name) and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(configs)}] Skipping existing true-depth render: {output_path}")
            continue

        try:
            transforms_path = resolve_after_transforms_path(config_path, args.outputs_root)
            command = build_command(
                ns_render_command=args.ns_render,
                config_path=config_path,
                output_path=output_path,
                transforms_path=transforms_path,
                eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
            )
        except Exception as exc:
            failures.append((config_path, render_dir, None, [], str(exc)))
            print(f"[{index}/{len(configs)}] Failed to prepare {config_path}: {exc}")
            if args.fail_fast:
                break
            continue

        print(f"[{index}/{len(configs)}] Rendering {config_path} -> {output_path}")
        print(f"[{index}/{len(configs)}] Using transforms: {transforms_path}")
        if args.dry_run:
            print(" ".join(command))
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed_process = subprocess.run(command, check=False)
        if completed_process.returncode != 0:
            failures.append((config_path, render_dir, completed_process.returncode, command, ""))
            print(f"[{index}/{len(configs)}] Failed with exit code {completed_process.returncode}: {config_path}")
            if args.fail_fast:
                break

    if skipped:
        print(f"Skipped {skipped} existing render directories. Use --overwrite to rerun them.")

    if failures:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failure_log.open("w", encoding="utf8") as log_file:
            for config_path, render_dir, returncode, command, error in failures:
                if returncode is not None:
                    log_file.write(f"returncode: {returncode}\n")
                if error:
                    log_file.write(f"error: {error}\n")
                log_file.write(f"config: {config_path}\n")
                log_file.write(f"render_dir: {render_dir}\n")
                if command:
                    log_file.write(f"command: {' '.join(command)}\n")
                log_file.write("\n")
        print(f"Failed renders: {len(failures)}. See {args.failure_log}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
