#!/usr/bin/env python3
"""Prepare a gsplat-compatible COLMAP dataset from a DA3 export directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy DA3 input images and COLMAP binary files into gsplat dataset layout."
    )
    parser.add_argument(
        "--da3-output-dir",
        required=True,
        help="DA3 export directory containing input_images/ and cameras/images/points3D .bin files.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Output dataset directory with images/ and sparse/0/ subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output dataset directory.",
    )
    args = parser.parse_args()

    da3_output_dir = Path(args.da3_output_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    input_images_dir = da3_output_dir / "input_images"

    if not input_images_dir.is_dir():
        raise SystemExit(f"Missing DA3 input_images directory: {input_images_dir}")

    image_paths = sorted(
        path for path in input_images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No images found in: {input_images_dir}")

    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for src in image_paths:
        copy_file(src, images_dir / src.name, overwrite=args.overwrite)

    for name in COLMAP_FILES:
        src = da3_output_dir / name
        if not src.is_file():
            raise SystemExit(f"Missing DA3 COLMAP file: {src}")
        copy_file(src, sparse_dir / name, overwrite=args.overwrite)

    print(f"Copied {len(image_paths)} image(s) to {images_dir}")
    print(f"Copied COLMAP files to {sparse_dir}")
    print(f"gsplat dataset ready: {dataset_dir}")


if __name__ == "__main__":
    main()
