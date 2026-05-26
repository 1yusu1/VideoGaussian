"""Minimal COLMAP model helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def qvec_to_rotmat(qvec: list[float] | np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
        ],
        dtype=np.float64,
    )


def camera_center_from_colmap(qvec: list[float], tvec: list[float]) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec)
    translation = np.asarray(tvec, dtype=np.float64)
    return -rotation.T @ translation


def read_cameras_txt(path: str | Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(v) for v in parts[4:]],
            }
    return cameras


def read_images_txt(path: str | Path) -> dict[int, dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = [float(v) for v in parts[1:5]]
        tvec = [float(v) for v in parts[5:8]]
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images[image_id] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "camera_center": camera_center_from_colmap(qvec, tvec).tolist(),
        }
        idx += 1  # skip 2D points line
    return images


def camera_params_to_intrinsics(camera: dict[str, Any]) -> dict[str, float]:
    model = camera["model"].upper()
    params = camera["params"]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    else:
        fx = fy = max(camera["width"], camera["height"]) / (2 * math.tan(math.radians(60) / 2))
        cx = camera["width"] / 2
        cy = camera["height"] / 2
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def build_cameras_json(model_dir: str | Path) -> dict[str, Any]:
    model_path = Path(model_dir)
    cameras = read_cameras_txt(model_path / "cameras.txt")
    images = read_images_txt(model_path / "images.txt")
    frames = []
    for image in sorted(images.values(), key=lambda item: item["name"]):
        camera = cameras[image["camera_id"]]
        rotation_wc = qvec_to_rotmat(image["qvec"]).T
        frames.append(
            {
                "file_path": image["name"],
                "image_id": image["image_id"],
                "camera_id": image["camera_id"],
                "width": camera["width"],
                "height": camera["height"],
                "intrinsics": camera_params_to_intrinsics(camera),
                "qvec_world_to_camera": image["qvec"],
                "tvec_world_to_camera": image["tvec"],
                "camera_to_world": {
                    "rotation": rotation_wc.tolist(),
                    "translation": image["camera_center"],
                },
            }
        )
    return {"camera_models": list(cameras.values()), "frames": frames}
