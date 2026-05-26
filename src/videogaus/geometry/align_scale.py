"""Align predicted geometry scale to a COLMAP reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from videogaus.utils.jsonio import write_json
from videogaus.utils.pointcloud import load_points, save_points_npz, write_ascii_ply


def align_scale(
    source_points: str | Path,
    reference_points: str | Path,
    output_npz: str | Path,
    *,
    method: str = "median_radius",
    output_ply: str | Path | None = None,
) -> dict[str, Any]:
    source = load_points(source_points)
    reference = load_points(reference_points)
    src = source["points"].astype(np.float64)
    ref = reference["points"].astype(np.float64)
    if method == "umeyama" and len(src) == len(ref):
        scale, rotation, translation = _umeyama(src, ref)
        aligned = scale * (rotation @ src.T).T + translation
    else:
        src_center = np.median(src, axis=0)
        ref_center = np.median(ref, axis=0)
        src_radius = np.median(np.linalg.norm(src - src_center, axis=1))
        ref_radius = np.median(np.linalg.norm(ref - ref_center, axis=1))
        scale = float(ref_radius / max(src_radius, 1e-8))
        rotation = np.eye(3)
        translation = ref_center - scale * src_center
        aligned = scale * src + translation

    save_points_npz(output_npz, aligned.astype(np.float32), source.get("colors"), confidence=source.get("confidence"))
    if output_ply:
        write_ascii_ply(output_ply, aligned.astype(np.float32), source.get("colors"))
    transform = {
        "method": method,
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": np.asarray(translation).reshape(3).tolist(),
        "source_points": str(source_points),
        "reference_points": str(reference_points),
        "output_npz": str(output_npz),
        "output_ply": str(output_ply) if output_ply else None,
        "num_source_points": int(len(src)),
        "num_reference_points": int(len(ref)),
    }
    write_json(Path(output_npz).with_suffix(".transform.json"), transform)
    return transform


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    cov = dst0.T @ src0 / len(src)
    u, singular_values, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src0 * src0, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / max(variance, 1e-8))
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["median_radius", "umeyama"], default="median_radius")
    parser.add_argument("--output-ply", default=None)
    args = parser.parse_args()
    print(align_scale(args.source, args.reference, args.output, method=args.method, output_ply=args.output_ply))


if __name__ == "__main__":
    main()
