"""Render train/test views with a trained gsplat model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from videogaus.utils.commands import run_command
from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json


def render_views(
    config: dict[str, Any],
    *,
    result_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    result = Path(result_dir or first_config_value(config, ["gsplat.result_dir", "paths.result_dir"], "outputs/gsplat")).expanduser()
    out = Path(output_dir or first_config_value(config, ["render.output_dir"], result / "renders")).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    command_template = first_config_value(config, ["render.command"], None)
    if not command_template:
        raise SystemExit(
            "No render.command configured. Set render.command to the gsplat render/eval command for this environment."
        )
    command = [
        str(part).format(result_dir=str(result), output_dir=str(out), split=split)
        for part in command_template
    ]
    run_command(command, cwd=first_config_value(config, ["paths.gsplat_examples_dir"], None), dry_run=dry_run, log_path=out / "render_commands.jsonl")
    manifest = {
        "result_dir": str(result),
        "output_dir": str(out),
        "split": split,
        "rendered_images_dir": str(out / "images"),
        "rendered_depth_dir": str(out / "depth"),
    }
    write_json(out / "render_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(render_views(cfg, result_dir=args.result_dir, output_dir=args.output_dir, split=args.split, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
