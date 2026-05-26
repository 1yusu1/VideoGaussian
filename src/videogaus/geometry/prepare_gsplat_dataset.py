"""Prepare a gsplat-compatible COLMAP dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from videogaus.utils.jsonio import write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def prepare_gsplat_dataset(
    source_dir: str | Path,
    dataset_dir: str | Path,
    *,
    images_dir: str | Path | None = None,
    sparse_dir: str | Path | None = None,
    dense_depth_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    dataset = Path(dataset_dir).expanduser().resolve()
    input_images = Path(images_dir).expanduser().resolve() if images_dir else _find_images_dir(source)
    input_sparse = Path(sparse_dir).expanduser().resolve() if sparse_dir else _find_sparse_dir(source, input_images)
    out_images = dataset / "images"
    out_sparse = dataset / "sparse" / "0"
    out_images.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in input_images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for image in image_paths:
        _copy(image, out_images / image.name, overwrite)
    for name in COLMAP_FILES:
        _copy(input_sparse / name, out_sparse / name, overwrite)
    dense_depth = Path(dense_depth_path).expanduser().resolve() if dense_depth_path else _find_dense_depth_npz(source)
    dense_depth_out = None
    if dense_depth is not None:
        dense_depth_out = dataset / "dense_depth.npz"
        _write_dense_depth_npz(dense_depth, dense_depth_out, image_paths, overwrite)
    manifest = {
        "source_dir": str(source),
        "dataset_dir": str(dataset),
        "images_dir": str(out_images),
        "sparse_dir": str(out_sparse),
        "dense_depth": str(dense_depth_out) if dense_depth_out else None,
        "num_images": len(image_paths),
    }
    write_json(dataset / "dataset_manifest.json", manifest)
    return manifest


def _find_images_dir(source: Path) -> Path:
    for candidate in [source / "input_images", source / "images", source / "undistorted" / "images"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find image directory under {source}")


def _find_sparse_dir(source: Path, images_dir: Path | None = None) -> Path:
    candidates = [source]
    if images_dir == (source / "undistorted" / "images").resolve():
        candidates.append(source / "undistorted" / "sparse")
    candidates.extend([source / "sparse" / "0", source / "sparse", source / "undistorted" / "sparse"])
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / name).is_file() for name in COLMAP_FILES):
            return candidate
    raise FileNotFoundError(f"Could not find COLMAP cameras/images/points3D .bin files under {source}")


def _find_dense_depth_npz(source: Path) -> Path | None:
    for candidate in [
        source / "dense_depth.npz",
        source / "exports" / "mini_npz" / "results.npz",
        source / "mini_npz" / "results.npz",
        source / "results.npz",
    ]:
        if candidate.is_file():
            with np.load(candidate) as data:
                if "depth" in data:
                    return candidate
    return None


def _write_dense_depth_npz(src: Path, dst: Path, image_paths: list[Path], overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst}")
    with np.load(src) as data:
        if "depth" not in data:
            raise KeyError(f"{src} does not contain a 'depth' array")
        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data else None
        if depth.shape[0] != len(image_paths):
            raise ValueError(
                f"Dense depth frame count ({depth.shape[0]}) does not match image count ({len(image_paths)})"
            )
        payload: dict[str, Any] = {
            "depth": depth,
            "image_names": np.asarray([p.name for p in image_paths]),
        }
        if conf is not None:
            if conf.shape != depth.shape:
                raise ValueError(f"Confidence shape {conf.shape} does not match depth shape {depth.shape}")
            payload["conf"] = conf
        for key in ("intrinsics", "extrinsics"):
            if key in data:
                payload[key] = data[key]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)


def _copy(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", "--da3-output-dir", dest="source_dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--sparse-dir", default=None)
    parser.add_argument("--dense-depth-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        prepare_gsplat_dataset(
            args.source_dir,
            args.dataset_dir,
            images_dir=args.images_dir,
            sparse_dir=args.sparse_dir,
            dense_depth_path=args.dense_depth_path,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
