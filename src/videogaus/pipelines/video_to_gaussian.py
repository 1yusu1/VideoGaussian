"""Compatibility pipeline from one video to gsplat training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from videogaus.gaussian.train_gsplat import train_gsplat
from videogaus.geometry.prepare_gsplat_dataset import prepare_gsplat_dataset
from videogaus.geometry.run_da3 import run_da3
from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--depth-anything-repo", default=None)
    parser.add_argument("--gsplat-examples-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--export-format", default=None)
    parser.add_argument("--process-res", type=int, default=None)
    parser.add_argument("--process-res-method", default=None)
    parser.add_argument("--ref-view-strategy", default=None)
    parser.add_argument("--conf-thresh-percentile", type=float, default=None)
    parser.add_argument("--num-max-points", type=int, default=None)
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--no-ray-pose", action="store_true")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--data-factor", type=int, default=None)
    parser.add_argument("--enable-viewer", action="store_true")
    parser.add_argument("--save-ply-during-train", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--skip-da3", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--da3-output-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, _unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    video = args.video or first_config_value(cfg, ["data.video", "paths.video"])
    if video is None:
        raise SystemExit("--video or data.video is required.")
    scene = args.scene or first_config_value(cfg, ["scene", "run.scene"], Path(video).stem)
    if args.test_mode:
        scene = f"{scene}_test"
    work_dir = Path(args.work_dir or first_config_value(cfg, ["paths.output_root", "paths.runs_dir", "paths.work_dir"], "outputs")).expanduser()
    run_dir = work_dir / scene
    da3_dir = Path(args.da3_output_dir or first_config_value(cfg, ["da3.output_dir", "paths.da3_output_dir"], run_dir / "da3"))
    dataset_dir = Path(args.dataset_dir or first_config_value(cfg, ["gsplat.data_dir", "paths.dataset_dir"], run_dir / "gsplat_dataset"))
    result_dir = Path(args.result_dir or first_config_value(cfg, ["gsplat.result_dir", "paths.result_dir"], run_dir / "gsplat_result"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_da3:
        run_da3(
            video,
            da3_dir,
            repo_dir=args.depth_anything_repo or first_config_value(cfg, ["third_party.depth_anything_3", "paths.depth_anything_repo"]),
            model_dir=args.model_dir or first_config_value(cfg, ["models.da3_checkpoint", "paths.model_dir", "da3.model_dir"]),
            fps=args.fps if args.fps is not None else first_config_value(cfg, ["data.fps", "da3.fps"]),
            export_format=args.export_format or first_config_value(cfg, ["da3.export_format"], "colmap-mini_npz-depth_vis-glb"),
            process_res=args.process_res or int(first_config_value(cfg, ["da3.process_res"], 504)),
            process_res_method=args.process_res_method or first_config_value(cfg, ["da3.process_res_method"], "upper_bound_resize"),
            ref_view_strategy=args.ref_view_strategy or first_config_value(cfg, ["da3.ref_view_strategy"], "middle"),
            use_ray_pose=(args.use_ray_pose or (not args.no_ray_pose and bool(first_config_value(cfg, ["da3.use_ray_pose"], True)))),
            conf_thresh_percentile=args.conf_thresh_percentile if args.conf_thresh_percentile is not None else float(first_config_value(cfg, ["da3.conf_thresh_percentile"], 95)),
            num_max_points=args.num_max_points if args.num_max_points is not None else int(first_config_value(cfg, ["da3.num_max_points"], 1000000)),
            dry_run=args.dry_run,
        )
    prepare_gsplat_dataset(da3_dir, dataset_dir, overwrite=True)
    if not args.skip_train:
        cfg.setdefault("gsplat", {})
        cfg["gsplat"]["data_dir"] = str(dataset_dir)
        cfg["gsplat"]["result_dir"] = str(result_dir)
        if args.max_steps is not None:
            cfg["gsplat"]["iterations"] = 100 if args.test_mode else args.max_steps
        elif args.test_mode:
            cfg["gsplat"]["iterations"] = 100
        if args.eval_steps is not None:
            cfg["gsplat"]["eval_steps"] = args.eval_steps
        if args.data_factor is not None:
            cfg["gsplat"]["data_factor"] = args.data_factor
        if args.enable_viewer:
            cfg["gsplat"]["enable_viewer"] = True
        if args.save_ply_during_train:
            cfg["gsplat"]["save_ply_during_train"] = True
        if args.gpus is not None:
            cfg["gsplat"]["cuda_visible_devices"] = args.gpus
        if args.gsplat_examples_dir is not None:
            cfg.setdefault("paths", {})["gsplat_examples_dir"] = args.gsplat_examples_dir
        train_gsplat(cfg, data_dir=dataset_dir, result_dir=result_dir, dry_run=args.dry_run)
    manifest: dict[str, Any] = {
        "scene": scene,
        "run_dir": str(run_dir),
        "da3_output_dir": str(da3_dir),
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
    }
    write_json(run_dir / "pipeline_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
