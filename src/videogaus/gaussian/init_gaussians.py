"""Initialize Gaussian Splatting inputs from geometry priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from videogaus.utils.jsonio import write_json
from videogaus.utils.pointcloud import load_points, save_points_npz, write_ascii_ply


def init_gaussians(
    source: str | Path,
    output_dir: str | Path,
    *,
    source_type: str = "auto",
    max_points: int | None = None,
    min_confidence: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cloud = load_points(source)
    points = cloud["points"]
    colors = cloud["colors"]
    confidence = cloud.get("confidence")
    mask = np.ones(len(points), dtype=bool)
    if min_confidence is not None and confidence is not None:
        mask &= confidence >= min_confidence
    points = points[mask]
    colors = colors[mask]
    if confidence is not None:
        confidence = confidence[mask]
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
        if confidence is not None:
            confidence = confidence[keep]

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "gaussian_init.npz"
    ply_path = out / "gaussian_init.ply"
    radii = _estimate_initial_radii(points)
    save_points_npz(npz_path, points, colors, confidence=confidence, scales=radii)
    write_ascii_ply(ply_path, points, colors)
    manifest = {
        "source": str(source),
        "source_type": source_type,
        "output_dir": str(out),
        "init_npz": str(npz_path),
        "init_ply": str(ply_path),
        "num_gaussians": int(len(points)),
        "max_points": max_points,
        "min_confidence": min_confidence,
    }
    write_json(out / "gaussian_init_manifest.json", manifest)
    return manifest


def _estimate_initial_radii(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.full((len(points), 3), 0.01, dtype=np.float32)
    center = np.median(points, axis=0)
    radius = np.median(np.linalg.norm(points - center, axis=1))
    scale = max(float(radius) / max(len(points) ** (1.0 / 3.0), 1.0), 1e-4)
    return np.full((len(points), 3), scale, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="points3D.txt/bin, .npz, .npy, or ASCII .ply.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-type", choices=["auto", "colmap", "da3"], default="auto")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(init_gaussians(args.source, args.output_dir, source_type=args.source_type, max_points=args.max_points, min_confidence=args.min_confidence, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
