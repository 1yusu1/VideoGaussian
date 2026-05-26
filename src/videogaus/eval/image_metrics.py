"""Compute PSNR, SSIM, and LPIPS for rendered images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from videogaus.utils.jsonio import write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def compute_image_metrics(
    pred_dir: str | Path,
    gt_dir: str | Path,
    output_dir: str | Path,
    *,
    require_lpips: bool = False,
) -> dict[str, Any]:
    pred = Path(pred_dir).expanduser().resolve()
    gt = Path(gt_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lpips_model = _build_lpips(require_lpips)
    rows = []
    for pred_path in sorted(p for p in pred.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        gt_path = gt / pred_path.name
        if not gt_path.exists():
            continue
        pred_img = _read_image(pred_path)
        gt_img = _read_image(gt_path)
        if pred_img.shape != gt_img.shape:
            gt_img = _resize_like(gt_img, pred_img)
        row = {
            "image": pred_path.name,
            "psnr": psnr(pred_img, gt_img),
            "ssim": ssim(pred_img, gt_img),
            "lpips": lpips_distance(lpips_model, pred_img, gt_img) if lpips_model is not None else None,
        }
        rows.append(row)
    summary = {
        "pred_dir": str(pred),
        "gt_dir": str(gt),
        "num_images": len(rows),
        "psnr": _mean(rows, "psnr"),
        "ssim": _mean(rows, "ssim"),
        "lpips": _mean(rows, "lpips"),
        "per_image": rows,
    }
    write_json(out / "image_metrics.json", summary)
    with (out / "image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    pil = pil.resize((ref.shape[1], ref.shape[0]), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))
    except ImportError:
        return _global_ssim(pred, gt)


def _global_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()
    return float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))


def _build_lpips(require: bool) -> Any | None:
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex")
        model.eval()
        return model
    except ImportError:
        if require:
            raise SystemExit("LPIPS requested but lpips/torch is not installed.")
        return None


def lpips_distance(model: Any, pred: np.ndarray, gt: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return float(model(p, g).item())


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None and not math.isinf(float(row[key]))]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-lpips", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compute_image_metrics(args.pred_dir, args.gt_dir, args.output_dir, require_lpips=args.require_lpips), indent=2))


if __name__ == "__main__":
    main()
