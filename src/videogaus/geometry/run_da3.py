"""Run Depth Anything 3 and normalize its geometry outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from videogaus.utils.commands import run_command
from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json


def run_da3(
    video_or_frames: str | Path,
    output_dir: str | Path,
    *,
    repo_dir: str | Path | None = None,
    model_dir: str | None = None,
    fps: float | None = None,
    export_format: str = "colmap-mini_npz-depth_vis-glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    ref_view_strategy: str = "middle",
    use_ray_pose: bool = True,
    conf_thresh_percentile: float = 95,
    num_max_points: int = 1000000,
    da3_bin: str = "da3",
    dry_run: bool = False,
) -> dict[str, Any]:
    source = Path(video_or_frames).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "da3_commands.jsonl"
    command = [
        da3_bin,
        "video" if source.is_file() else "auto",
        str(source),
        "--export-dir",
        str(out),
        "--export-format",
        export_format,
        "--process-res",
        str(process_res),
        "--process-res-method",
        process_res_method,
        "--ref-view-strategy",
        ref_view_strategy,
        "--conf-thresh-percentile",
        str(conf_thresh_percentile),
        "--num-max-points",
        str(num_max_points),
        "--auto-cleanup",
    ]
    if model_dir:
        command += ["--model-dir", model_dir]
    if fps is not None and source.is_file():
        command += ["--fps", str(fps)]
    if use_ray_pose:
        command.append("--use-ray-pose")
    run_command(command, cwd=repo_dir, dry_run=dry_run, log_path=log)

    manifest = {
        "source": str(source),
        "output_dir": str(out),
        "repo_dir": str(repo_dir) if repo_dir else None,
        "model_dir": model_dir,
        "depth_dir": _first_existing_dir(out, ["depth", "depths", "depth_npy"]),
        "confidence_dir": _first_existing_dir(out, ["confidence", "conf", "confidence_npy"]),
        "input_images_dir": _first_existing_dir(out, ["input_images", "images"]),
        "cameras_bin": _maybe_path(out / "cameras.bin"),
        "images_bin": _maybe_path(out / "images.bin"),
        "points3d_bin": _maybe_path(out / "points3D.bin"),
        "mini_npz": _maybe_path(out / "exports" / "mini_npz" / "results.npz")
        or _maybe_path(out / "mini_npz" / "results.npz")
        or _maybe_path(out / "results.npz"),
        "export_format": export_format,
    }
    write_json(out / "da3_manifest.json", manifest)
    return manifest


def _maybe_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _first_existing_dir(base: Path, names: list[str]) -> str | None:
    for name in names:
        path = base / name
        if path.is_dir():
            return str(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Video path or frame directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--export-format", default=None)
    parser.add_argument("--process-res", type=int, default=None)
    parser.add_argument("--process-res-method", default=None)
    parser.add_argument("--ref-view-strategy", default=None)
    parser.add_argument("--conf-thresh-percentile", type=float, default=None)
    parser.add_argument("--num-max-points", type=int, default=None)
    parser.add_argument("--no-ray-pose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.input or first_config_value(cfg, ["data.video", "paths.video", "data.frames_dir"])
    if source is None:
        raise SystemExit("--input or data.video is required.")
    manifest = run_da3(
        source,
        args.output_dir or first_config_value(cfg, ["da3.output_dir", "paths.da3_output_dir"], "outputs/da3"),
        repo_dir=args.repo_dir or first_config_value(cfg, ["third_party.depth_anything_3", "paths.depth_anything_repo"]),
        model_dir=args.model_dir or first_config_value(cfg, ["models.da3_checkpoint", "paths.model_dir", "da3.model_dir"]),
        fps=args.fps if args.fps is not None else first_config_value(cfg, ["data.fps", "da3.fps"]),
        export_format=args.export_format or first_config_value(cfg, ["da3.export_format"], "colmap-mini_npz-depth_vis-glb"),
        process_res=args.process_res or int(first_config_value(cfg, ["da3.process_res"], 504)),
        process_res_method=args.process_res_method or first_config_value(cfg, ["da3.process_res_method"], "upper_bound_resize"),
        ref_view_strategy=args.ref_view_strategy or first_config_value(cfg, ["da3.ref_view_strategy"], "middle"),
        use_ray_pose=not args.no_ray_pose and bool(first_config_value(cfg, ["da3.use_ray_pose"], True)),
        conf_thresh_percentile=args.conf_thresh_percentile
        if args.conf_thresh_percentile is not None
        else float(first_config_value(cfg, ["da3.conf_thresh_percentile"], 95)),
        num_max_points=args.num_max_points
        if args.num_max_points is not None
        else int(first_config_value(cfg, ["da3.num_max_points"], 1000000)),
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
