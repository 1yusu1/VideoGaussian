#!/usr/bin/env python3
"""Export DA3 mini_npz depths into a 3DGS dataset depths directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DA3 depth maps for 3DGS datasets.")
    parser.add_argument("--npz", required=True, help="Path to DA3 exports/mini_npz/results.npz.")
    parser.add_argument("--images-dir", required=True, help="3DGS dataset images directory.")
    parser.add_argument("--depths-dir", required=True, help="Output depths directory.")
    parser.add_argument(
        "--ext",
        default=".npy",
        choices=[".npy"],
        help="Depth file extension. .npy stores float32 metric depth.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    depths_dir = Path(args.depths_dir).expanduser().resolve()
    depths_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    depths = data["depth"].astype(np.float32)
    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    )

    if len(image_paths) != len(depths):
        raise SystemExit(
            f"Image/depth count mismatch: {len(image_paths)} images, {len(depths)} depths"
        )

    for image_path, depth in zip(image_paths, depths):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"Failed to read image: {image_path}")
        target_h, target_w = image.shape[:2]
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out_path = depths_dir / f"{image_path.stem}{args.ext}"
        np.save(out_path, depth.astype(np.float32))

    print(f"Exported {len(image_paths)} depth maps to {depths_dir}")
    print("Depth filenames match image stems and are stored as float32 .npy files.")


if __name__ == "__main__":
    main()
