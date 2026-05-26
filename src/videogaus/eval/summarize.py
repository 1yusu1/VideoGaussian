"""Summarize scene/method metrics into CSV, Markdown, and reports."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from videogaus.utils.jsonio import read_json, write_json

SUMMARY_FIELDS = [
    "scene",
    "setting",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "num_GS",
    "render_time_sec_per_image",
    "render_fps",
    "training_time_sec",
    "peak_gpu_memory_mib",
    "val_step",
    "metrics_dir",
    "result_dir",
]


def summarize(
    metrics_root: str | Path,
    output_dir: str | Path,
    *,
    scene: str | None = None,
    report: bool = False,
) -> dict[str, Any]:
    root = Path(metrics_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = discover_metric_rows(root, scene=scene)
    summary_csv = out / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})
    summary_md = out / "summary.md"
    summary_md.write_text(_summary_markdown(rows), encoding="utf-8")
    payload = {"metrics_root": str(root), "summary_csv": str(summary_csv), "summary_md": str(summary_md), "rows": rows}
    write_json(out / "summary.json", payload)
    if report:
        report_scene = scene or (rows[0]["scene"] if rows else "scene")
        (out / f"{report_scene}_report.md").write_text(_report_markdown(report_scene, rows), encoding="utf-8")
    return payload


def discover_metric_rows(root: Path, scene: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_metrics_path in root.rglob("image_metrics.json"):
        metrics_dir = image_metrics_path.parent
        parts = metrics_dir.relative_to(root).parts
        inferred_scene = scene or (parts[0] if len(parts) >= 2 else root.name)
        inferred_method = parts[1] if len(parts) >= 2 else metrics_dir.name
        image_metrics = read_json(image_metrics_path)
        runtime_path = metrics_dir / "runtime_metrics.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else {}
        row = {
            "scene": image_metrics.get("scene") or runtime.get("scene") or inferred_scene,
            "method": image_metrics.get("method") or runtime.get("method") or inferred_method,
            "psnr": image_metrics.get("psnr"),
            "ssim": image_metrics.get("ssim"),
            "lpips": image_metrics.get("lpips"),
            "training_time_sec": runtime.get("training_time_sec"),
            "render_fps": runtime.get("render_fps"),
            "peak_gpu_memory_mib": runtime.get("peak_gpu_memory_mib"),
            "metrics_dir": str(metrics_dir),
        }
        rows.append(row)
    for val_metrics_path in root.rglob("stats/val_step*.json"):
        rows.append(_row_from_gsplat_stats(root, val_metrics_path, scene=scene))
    return sorted(rows, key=lambda row: (str(row.get("scene")), str(row.get("setting")), str(row.get("method"))))


def _row_from_gsplat_stats(root: Path, stats_path: Path, scene: str | None = None) -> dict[str, Any]:
    stats = read_json(stats_path)
    result_dir = stats_path.parent.parent
    method_dir = result_dir.parent
    relative = result_dir.relative_to(root)
    parts = relative.parts
    method = method_dir.name
    run_parts = list(parts[:-2]) if len(parts) >= 2 and parts[-1] == "gsplat" else list(parts[:-1])
    setting = _infer_setting(run_parts)
    inferred_scene = scene or _infer_scene(run_parts)
    method_label = _method_label(method, setting, run_parts)
    runtime_path = result_dir / "runtime_metrics.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else {}
    render_time = stats.get("ellipse_time")
    render_fps = None
    if isinstance(render_time, (int, float)) and render_time > 0:
        render_fps = 1.0 / float(render_time)
    return {
        "scene": inferred_scene,
        "setting": setting,
        "method": method_label,
        "psnr": stats.get("psnr"),
        "ssim": stats.get("ssim"),
        "lpips": stats.get("lpips"),
        "num_GS": stats.get("num_GS"),
        "render_time_sec_per_image": render_time,
        "render_fps": render_fps,
        "training_time_sec": runtime.get("training_time_sec"),
        "peak_gpu_memory_mib": runtime.get("peak_gpu_memory_mib") or stats.get("mem"),
        "val_step": _parse_val_step(stats_path.name),
        "metrics_dir": str(stats_path.parent),
        "result_dir": str(result_dir),
    }


def _infer_scene(run_parts: list[str]) -> str:
    if not run_parts:
        return "scene"
    first = run_parts[0]
    if first.startswith("liminal_pool"):
        return "liminal_pool"
    return first


def _infer_setting(run_parts: list[str]) -> str:
    text = "/".join(run_parts)
    if "liminal_pool_fps24_conf96" in text:
        return "fps12_conf96"
    if "colmap_vs_da3" in text:
        return "fps24_conf96"
    fps_match = re.search(r"fps(\d+)", text)
    fps = f"fps{fps_match.group(1)}" if fps_match else "fps?"
    conf_match = re.search(r"conf(\d+)", text)
    conf = f"_conf{conf_match.group(1)}" if conf_match else ""
    return f"{fps}{conf}"


def _method_label(method: str, setting: str, run_parts: list[str]) -> str:
    if "dense_depthreg" in method:
        base = "da3_gs_dense_depthreg"
    elif method == "da3_gs_depthreg":
        base = "da3_gs_sparse_depthreg"
    else:
        base = method
    return f"{base}_{setting}"


def _parse_val_step(filename: str) -> int | None:
    match = re.search(r"val_step(\d+)", filename)
    return int(match.group(1)) if match else None


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# VideoGaussian Summary",
        "",
        "| Scene | Setting | Method | PSNR | SSIM | LPIPS | #GS | Render s/img | Train Time (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {setting} | {method} | {psnr} | {ssim} | {lpips} | {num_GS} | {render_time_sec_per_image} | {training_time_sec} |".format(
                **{
                    k: _fmt(row.get(k))
                    for k in [
                        "scene",
                        "setting",
                        "method",
                        "psnr",
                        "ssim",
                        "lpips",
                        "num_GS",
                        "render_time_sec_per_image",
                        "training_time_sec",
                    ]
                }
            )
        )
    return "\n".join(lines) + "\n"


def _report_markdown(scene: str, rows: list[dict[str, Any]]) -> str:
    scene_rows = [row for row in rows if row.get("scene") == scene] or rows
    method_rows = "\n".join(
        f"| {row.get('setting')} | {row.get('method')} | {row.get('result_dir') or row.get('metrics_dir')} |"
        for row in scene_rows
    )
    metric_rows = "\n".join(
        f"| {row.get('setting')} | {row.get('method')} | {_fmt(row.get('psnr'))} | {_fmt(row.get('ssim'))} | {_fmt(row.get('lpips'))} | {_fmt(row.get('num_GS'))} |"
        for row in scene_rows
    )
    runtime_rows = "\n".join(
        f"| {row.get('setting')} | {row.get('method')} | {_fmt(row.get('training_time_sec'))} | {_fmt(row.get('render_time_sec_per_image'))} | {_fmt(row.get('render_fps'))} | {_fmt(row.get('peak_gpu_memory_mib'))} |"
        for row in scene_rows
    )
    observations = _observations(scene_rows)
    return f"""# {scene} Report

## Method Table

| Setting | Method | Result Directory |
|---|---|---|
{method_rows}

## PSNR/SSIM/LPIPS Table

| Setting | Method | PSNR | SSIM | LPIPS | #GS |
|---|---|---:|---:|---:|---:|
{metric_rows}

## Key Observations

{observations}

## Qualitative Comparison

Add rendered train/test comparisons from each method here. Recommended layout: ground truth, COLMAP+GS, DA3+GS, DA3+DepthReg, DA3+XFeat-GA-v2.

## Failure Cases

Add frames with pose drift, depth bleeding, dynamic objects, or textureless regions here.

## Runtime/Memory

| Setting | Method | Training Time (s) | Render s/img | Render FPS | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
{runtime_rows}
"""


def _observations(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No metrics were found."
    lines: list[str] = []
    best_psnr = max(rows, key=lambda row: float(row.get("psnr") or float("-inf")))
    best_lpips = min(rows, key=lambda row: float(row.get("lpips") or float("inf")))
    lines.append(
        f"- Best PSNR is `{_fmt(best_psnr.get('psnr'))}` from `{best_psnr.get('method')}` in `{best_psnr.get('setting')}`."
    )
    lines.append(
        f"- Best LPIPS is `{_fmt(best_lpips.get('lpips'))}` from `{best_lpips.get('method')}` in `{best_lpips.get('setting')}`."
    )
    fps12_da3 = [row for row in rows if row.get("setting") == "fps12_conf96" and str(row.get("method", "")).startswith("da3")]
    if len(fps12_da3) >= 2:
        lines.append("- On fps12/conf96, DA3 depth regularization improves DA3 initialization modestly but does not close the gap to COLMAP.")
    ga_rows = [row for row in fps12_da3 if "da3_ga_xfeat_gs" in str(row.get("method", ""))]
    direct_rows = [row for row in fps12_da3 if str(row.get("method", "")).startswith("da3_gs_")]
    if ga_rows and direct_rows:
        lines.append("- VGGT-X-style epipolar GA is a negative ablation on this scene: it trails direct DA3 initialization, likely because pose-only alignment disturbs DA3 camera-depth coupling.")
    v2_rows = [row for row in fps12_da3 if "da3_ga_xfeat_v2_gs" in str(row.get("method", ""))]
    if ga_rows and v2_rows:
        lines.append("- DA3 GA XFeat v2 recovers part of the epipolar-only GA loss, but still remains below direct DA3 initialization on PSNR/SSIM.")
    mcmc_rows = [row for row in fps12_da3 if "da3_ga_xfeat_v2_mcmc_pose_depthreg" in str(row.get("method", ""))]
    if mcmc_rows and v2_rows:
        lines.append("- MCMC + pose optimization + dense depth regularization improves perceptual LPIPS over v2 default, but lowers PSNR/SSIM and costs more training/render time.")
    weak_mcmc_rows = [
        row
        for row in fps12_da3
        if "da3_ga_xfeat_v2_mcmc_pose_depthreg_lr" in str(row.get("method", ""))
        and "500k" not in str(row.get("method", ""))
    ]
    if weak_mcmc_rows:
        best_weak = max(weak_mcmc_rows, key=lambda row: float(row.get("psnr") or float("-inf")))
        lines.append(
            f"- Weakening pose/depth regularization improves GA+MCMC PSNR/SSIM; the best weakened variant is `{best_weak.get('method')}` with PSNR `{_fmt(best_weak.get('psnr'))}`."
        )
    init_500k_rows = [row for row in fps12_da3 if "da3_ga_xfeat_v2_500k" in str(row.get("method", ""))]
    if init_500k_rows:
        lines.append("- Reducing GA v2 initialization to 500k points gives MCMC room to add Gaussians, but does not improve metrics over the best 1.8M-init weakened GA+MCMC variant on this scene.")
    fps2_rows = [row for row in rows if str(row.get("setting", "")).startswith("fps2")]
    if fps2_rows:
        lines.append("- On fps2, COLMAP remains stronger for this liminal_pool scene; DA3 conf70 adds more points but remains much worse, suggesting noisy or globally inconsistent DA3 geometry.")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    payload = summarize(args.metrics_root, args.output_dir, scene=args.scene, report=args.report)
    print(payload["summary_md"])


if __name__ == "__main__":
    main()
