#!/usr/bin/env python3
"""Convert a VideoGaussian YAML config to shell assignments."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - server environment check
    raise SystemExit("PyYAML is required for --config. Install with: pip install pyyaml") from exc


def get_nested(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_value(data: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = get_nested(data, path)
        if value is not None:
            return value
    return None


def emit(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    print(f"{name}={shlex.quote(text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to VideoGaussian YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {config_path}")

    mappings = {
        "VIDEO": ["paths.video", "inputs.video"],
        "DEPTH_ANYTHING_REPO": ["paths.depth_anything_repo", "repos.depth_anything_3"],
        "GSPLAT_EXAMPLES_DIR": ["paths.gsplat_examples_dir"],
        "WORK_DIR": ["paths.runs_dir", "paths.work_dir"],
        "MODEL_DIR": ["paths.model_dir", "da3.model_dir"],
        "SCENE_NAME": ["run.scene", "scene"],
        "DA3_OUTPUT_DIR": ["paths.da3_output_dir", "inputs.da3_output_dir"],
        "DATASET_DIR": ["paths.dataset_dir", "gsplat.data_dir"],
        "RESULT_DIR": ["paths.result_dir", "gsplat.result_dir", "gsplat.result_root"],
        "FINAL_PLY": ["paths.final_ply"],
        "FPS": ["da3.fps"],
        "EXPORT_FORMAT": ["da3.export_format"],
        "PROCESS_RES": ["da3.process_res"],
        "PROCESS_RES_METHOD": ["da3.process_res_method"],
        "REF_VIEW_STRATEGY": ["da3.ref_view_strategy"],
        "CONF_THRESH_PERCENTILE": ["da3.conf_thresh_percentile"],
        "NUM_MAX_POINTS": ["da3.num_max_points"],
        "USE_RAY_POSE": ["da3.use_ray_pose"],
        "CUDA_DEVICES": ["gsplat.cuda_visible_devices", "gsplat.gpus"],
        "MAX_STEPS": ["gsplat.max_steps"],
        "EVAL_STEPS": ["gsplat.eval_steps"],
        "DATA_FACTOR": ["gsplat.data_factor"],
        "SAVE_PLY_DURING_TRAIN": ["gsplat.save_ply_during_train"],
        "ENABLE_VIEWER": ["gsplat.enable_viewer"],
        "CONDA_ENV": ["run.conda_env"],
        "TEST_MODE": ["run.test_mode"],
        "SKIP_DA3": ["run.skip_da3"],
        "SKIP_TRAIN": ["run.skip_train"],
        "MERGE": ["run.merge"],
    }

    for env_name, paths in mappings.items():
        emit(env_name, first_value(data, paths))

    gsplat_repo = first_value(data, ["paths.gsplat_repo", "repos.gsplat"])
    gsplat_examples = first_value(data, ["paths.gsplat_examples_dir"])
    if gsplat_repo is not None and gsplat_examples is None:
        emit("GSPLAT_EXAMPLES_DIR", str(gsplat_repo).rstrip("/") + "/examples")


if __name__ == "__main__":
    main()
