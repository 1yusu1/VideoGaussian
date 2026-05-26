"""Point cloud loading and writing utilities."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np


def load_points(path: str | Path) -> dict[str, np.ndarray]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        data = np.load(input_path)
        points = _first_existing(data, ["points", "world_points", "xyz", "means"])
        colors = _first_existing(data, ["colors", "rgb", "rgbs"], required=False)
        confidence = _first_existing(data, ["confidence", "conf", "opacity"], required=False)
        return _normalize_cloud(points, colors, confidence)
    if suffix == ".npy":
        return _normalize_cloud(np.load(input_path), None, None)
    if suffix == ".ply":
        return read_ascii_ply(input_path)
    if input_path.name == "points3D.txt":
        return read_colmap_points3d_txt(input_path)
    if input_path.name == "points3D.bin":
        return read_colmap_points3d_bin(input_path)
    raise ValueError(f"Unsupported point cloud format: {input_path}")


def _first_existing(data: Any, keys: list[str], required: bool = True) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    if required:
        raise KeyError(f"Missing point array. Tried keys: {keys}")
    return None


def _normalize_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    confidence: np.ndarray | None,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.max(initial=0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors.reshape(-1, 3), 0, 255).astype(np.uint8)
    out = {"points": points, "colors": colors}
    if confidence is not None:
        out["confidence"] = np.asarray(confidence).reshape(-1).astype(np.float32)
    return out


def save_points_npz(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None, **extra: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"points": np.asarray(points, dtype=np.float32)}
    if colors is not None:
        payload["colors"] = np.asarray(colors)
    payload.update({key: value for key, value in extra.items() if value is not None})
    np.savez_compressed(output, **payload)


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cloud = _normalize_cloud(points, colors, None)
    pts = cloud["points"]
    rgb = cloud["colors"]
    with output.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def read_ascii_ply(path: str | Path) -> dict[str, np.ndarray]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        rows = [line.split() for line in f if line.strip()]
    if not rows:
        return _normalize_cloud(np.empty((0, 3), dtype=np.float32), None, None)
    arr = np.asarray(rows, dtype=np.float32)
    points = arr[:, :3]
    colors = arr[:, 3:6] if arr.shape[1] >= 6 else None
    return _normalize_cloud(points, colors, None)


def read_colmap_points3d_txt(path: str | Path) -> dict[str, np.ndarray]:
    points: list[list[float]] = []
    colors: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return _normalize_cloud(np.asarray(points), np.asarray(colors), None)


def read_colmap_points3d_bin(path: str | Path) -> dict[str, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    with Path(path).open("rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            _point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, 1)
            points.append(xyz)
            colors.append(rgb)
    return _normalize_cloud(np.asarray(points), np.asarray(colors), None)
