"""Train gsplat from a VideoGaussian config."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from videogaus.utils.commands import run_command
from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json


def train_gsplat(
    config: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    result_dir: str | Path | None = None,
    gsplat_examples_dir: str | Path | None = None,
    iterations: int | None = None,
    eval_steps_override: int | None = None,
    save_steps_override: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    gsplat_repo = first_config_value(config, ["third_party.gsplat", "paths.gsplat_repo"])
    examples_dir = gsplat_examples_dir or first_config_value(config, ["paths.gsplat_examples_dir"], None)
    if examples_dir is None and gsplat_repo is not None:
        examples_dir = str(Path(gsplat_repo) / "examples")
    data_value = data_dir or first_config_value(config, ["gsplat.data_dir", "paths.dataset_dir", "data.dataset_dir"])
    if data_value is None:
        raise SystemExit("--data-dir or gsplat.data_dir is required.")
    data = Path(data_value).expanduser()
    result = Path(result_dir or first_config_value(config, ["gsplat.result_dir", "paths.result_dir"], "outputs/gsplat")).expanduser()
    result.mkdir(parents=True, exist_ok=True)

    max_steps = int(iterations or first_config_value(config, ["gsplat.iterations", "gsplat.max_steps"], 30000))
    eval_steps = int(eval_steps_override or first_config_value(config, ["gsplat.eval_steps"], max_steps))
    save_steps = int(save_steps_override or first_config_value(config, ["gsplat.save_steps"], eval_steps))
    data_factor = int(first_config_value(config, ["gsplat.data_factor"], 1))
    strategy = str(first_config_value(config, ["gsplat.strategy"], "default")).lower()
    if strategy not in {"default", "mcmc"}:
        raise ValueError("gsplat.strategy must be 'default' or 'mcmc'.")
    mcmc_cap_max = int(first_config_value(config, ["gsplat.mcmc.cap_max", "gsplat.strategy_cap_max"], 1800000))
    pose_opt_enabled = bool(first_config_value(config, ["gsplat.pose_optimization.enabled", "gsplat.pose_opt"], False))
    pose_opt_lr = float(first_config_value(config, ["gsplat.pose_optimization.lr", "gsplat.pose_opt_lr"], 1e-5))
    pose_opt_reg = float(first_config_value(config, ["gsplat.pose_optimization.reg", "gsplat.pose_opt_reg"], 1e-6))
    depth_reg_enabled = bool(first_config_value(config, ["gsplat.depth_regularization.enabled"], False))
    depth_reg_type = str(first_config_value(config, ["gsplat.depth_regularization.type"], "sparse")).lower()
    depth_weight = float(first_config_value(config, ["gsplat.depth_regularization.weight", "depth_regularization.weight"], 0.0))
    sparse_depth_enabled = depth_reg_enabled and depth_reg_type in {"sparse", "both", "sparse+dense"}
    dense_depth_enabled = bool(
        first_config_value(
            config,
            ["gsplat.depth_regularization.dense.enabled", "gsplat.dense_depth_loss.enabled"],
            depth_reg_enabled and depth_reg_type in {"dense", "both", "sparse+dense"},
        )
    )
    dense_depth_weight = float(
        first_config_value(
            config,
            [
                "gsplat.depth_regularization.dense.weight",
                "gsplat.dense_depth_loss.weight",
                "gsplat.depth_regularization.weight",
            ],
            depth_weight,
        )
    )
    dense_depth_conf_percentile = float(
        first_config_value(
            config,
            [
                "gsplat.depth_regularization.dense.conf_percentile",
                "gsplat.dense_depth_loss.conf_percentile",
            ],
            0.0,
        )
    )
    dense_depth_conf_weighted = bool(
        first_config_value(
            config,
            [
                "gsplat.depth_regularization.dense.conf_weighted",
                "gsplat.dense_depth_loss.conf_weighted",
            ],
            True,
        )
    )
    cuda_devices = first_config_value(config, ["gsplat.cuda_visible_devices", "gsplat.gpus"], None)
    command_template = first_config_value(config, ["gsplat.train_command"], None)

    if command_template:
        command = [
            str(part).format(
                data_dir=str(data),
                result_dir=str(result),
                max_steps=max_steps,
                eval_steps=eval_steps,
                depth_weight=depth_weight,
            )
            for part in command_template
        ]
        cwd = examples_dir
    else:
        command = [
            "python",
            "simple_trainer.py",
            strategy,
            "--data-dir",
            str(data),
            "--result-dir",
            str(result),
            "--max-steps",
            str(max_steps),
            "--eval-steps",
            str(eval_steps),
            "--save-steps",
            str(save_steps),
            "--data-factor",
            str(data_factor),
            "--disable-video",
        ]
        if strategy == "mcmc":
            command += ["--strategy.cap-max", str(mcmc_cap_max)]
        if pose_opt_enabled:
            command += ["--pose-opt", "--pose-opt-lr", str(pose_opt_lr), "--pose-opt-reg", str(pose_opt_reg)]
        if not bool(first_config_value(config, ["gsplat.enable_viewer"], False)):
            command.append("--disable-viewer")
        if bool(first_config_value(config, ["gsplat.save_ply_during_train"], False)):
            command.append("--save-ply")
        for extra in first_config_value(config, ["gsplat.extra_args"], []) or []:
            command.append(str(extra))
        if sparse_depth_enabled:
            command += ["--depth-loss", "--depth-lambda", str(depth_weight)]
        if dense_depth_enabled:
            command += [
                "--dense-depth-loss",
                "--dense-depth-lambda",
                str(dense_depth_weight),
                "--dense-depth-conf-percentile",
                str(dense_depth_conf_percentile),
            ]
            if dense_depth_conf_weighted:
                command.append("--dense-depth-conf-weighted")
        cwd = examples_dir

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    started = time.time()
    run_command(command, cwd=cwd, env=env, dry_run=dry_run, log_path=result / "gsplat_commands.jsonl")
    elapsed = 0.0 if dry_run else time.time() - started
    metrics = {
        "method": first_config_value(config, ["method"], "gsplat"),
        "data_dir": str(data),
        "result_dir": str(result),
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "strategy": strategy,
        "mcmc_cap_max": mcmc_cap_max if strategy == "mcmc" else None,
        "pose_optimization": {
            "enabled": pose_opt_enabled,
            "lr": pose_opt_lr,
            "reg": pose_opt_reg,
        },
        "depth_regularization": {
            "enabled": depth_reg_enabled,
            "type": depth_reg_type,
            "sparse_enabled": sparse_depth_enabled,
            "sparse_weight": depth_weight,
            "dense_enabled": dense_depth_enabled,
            "dense_weight": dense_depth_weight,
            "dense_conf_percentile": dense_depth_conf_percentile,
            "dense_conf_weighted": dense_depth_conf_weighted,
        },
        "training_time_sec": elapsed,
        "cuda_visible_devices": cuda_devices,
    }
    write_json(result / "runtime_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--gsplat-examples-dir", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(
        json.dumps(
            train_gsplat(
                cfg,
                data_dir=args.data_dir,
                result_dir=args.result_dir,
                gsplat_examples_dir=args.gsplat_examples_dir,
                iterations=args.iterations,
                eval_steps_override=args.eval_steps,
                save_steps_override=args.save_steps,
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
