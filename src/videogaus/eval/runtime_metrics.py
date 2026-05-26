"""Collect runtime, render FPS, and memory metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from videogaus.utils.jsonio import read_json, write_json


def collect_runtime_metrics(
    output: str | Path,
    *,
    training_metrics: str | Path | None = None,
    render_dir: str | Path | None = None,
    gpu_log: str | Path | None = None,
    method: str | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"method": method, "scene": scene}
    if training_metrics and Path(training_metrics).exists():
        data = read_json(training_metrics)
        metrics.update({k: data.get(k) for k in ["training_time_sec", "max_steps", "cuda_visible_devices"] if k in data})
    if render_dir and Path(render_dir).exists():
        images = [p for p in Path(render_dir).iterdir() if p.is_file()]
        metrics["num_rendered_frames"] = len(images)
        time_file = Path(render_dir) / "render_time_sec.txt"
        if time_file.exists():
            render_time = float(time_file.read_text(encoding="utf-8").strip())
            metrics["render_time_sec"] = render_time
            metrics["render_fps"] = len(images) / render_time if render_time > 0 else None
    if gpu_log and Path(gpu_log).exists():
        metrics["peak_gpu_memory_mib"] = parse_peak_gpu_memory(gpu_log)
    write_json(output, metrics)
    return metrics


def parse_peak_gpu_memory(path: str | Path) -> int | None:
    peak = None
    pattern = re.compile(r"(\d+)")
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = [int(match.group(1)) for match in pattern.finditer(line)]
            for num in nums:
                peak = num if peak is None else max(peak, num)
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-metrics", default=None)
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--gpu-log", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()
    print(json.dumps(collect_runtime_metrics(args.output, training_metrics=args.training_metrics, render_dir=args.render_dir, gpu_log=args.gpu_log, method=args.method, scene=args.scene), indent=2))


if __name__ == "__main__":
    main()
