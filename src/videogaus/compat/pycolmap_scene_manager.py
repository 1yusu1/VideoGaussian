"""Compatibility SceneManager for newer pycolmap builds.

gsplat 1.5.x imports ``pycolmap.SceneManager``, which existed in old pycolmap
packages but is absent from recent official pycolmap wheels. This adapter
implements the subset of that API used by ``examples/datasets/colmap.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _Camera:
    camera_id: int
    camera_type: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0


@dataclass
class _Image:
    image_id: int
    camera_id: int
    name: str
    tvec: np.ndarray
    _rotation: np.ndarray

    def R(self) -> np.ndarray:
        return self._rotation


class SceneManager:
    """Small wrapper around ``pycolmap.Reconstruction``.

    It lazily reads a COLMAP binary/text model and exposes old SceneManager-like
    attributes expected by gsplat's dataset parser.
    """

    def __init__(self, colmap_dir: str | Path):
        import pycolmap

        self.colmap_dir = str(colmap_dir)
        self._reconstruction = pycolmap.Reconstruction(self.colmap_dir)
        self.cameras: dict[int, _Camera] = {}
        self.images: dict[int, _Image] = {}
        self.name_to_image_id: dict[str, int] = {}
        self.points3D = np.zeros((0, 3), dtype=np.float64)
        self.point3D_errors = np.zeros((0,), dtype=np.float64)
        self.point3D_colors = np.zeros((0, 3), dtype=np.uint8)
        self.point3D_id_to_images: dict[int, list[tuple[int, int]]] = {}
        self.point3D_id_to_point3D_idx: dict[int, int] = {}

    def load_cameras(self) -> None:
        self.cameras = {
            int(camera_id): _camera_from_pycolmap(camera)
            for camera_id, camera in self._reconstruction.cameras.items()
        }

    def load_images(self) -> None:
        self.images = {
            int(image_id): _image_from_pycolmap(image)
            for image_id, image in self._reconstruction.images.items()
            if bool(getattr(image, "registered", True))
        }
        self.name_to_image_id = {image.name: image_id for image_id, image in self.images.items()}

    def load_points3D(self) -> None:
        point_items = sorted(self._reconstruction.points3D.items(), key=lambda item: int(item[0]))
        points = []
        errors = []
        colors = []
        self.point3D_id_to_images = {}
        self.point3D_id_to_point3D_idx = {}
        for idx, (point_id, point) in enumerate(point_items):
            point_id = int(point_id)
            points.append(np.asarray(point.xyz, dtype=np.float64))
            errors.append(float(point.error))
            colors.append(np.asarray(point.color, dtype=np.uint8))
            self.point3D_id_to_point3D_idx[point_id] = idx
            self.point3D_id_to_images[point_id] = [
                (int(track_element.image_id), int(track_element.point2D_idx))
                for track_element in point.track.elements
            ]
        self.points3D = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.point3D_errors = np.asarray(errors, dtype=np.float64)
        self.point3D_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)


def _camera_from_pycolmap(camera: Any) -> _Camera:
    params = np.asarray(camera.params, dtype=np.float64)
    model = _camera_model_name(camera)
    fx, fy, cx, cy = _intrinsics_from_params(model, params)
    distortion = _distortion_from_params(model, params)
    return _Camera(
        camera_id=int(camera.camera_id),
        camera_type=model,
        width=int(camera.width),
        height=int(camera.height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        **distortion,
    )


def _image_from_pycolmap(image: Any) -> _Image:
    transform = image.cam_from_world
    return _Image(
        image_id=int(image.image_id),
        camera_id=int(image.camera_id),
        name=str(image.name),
        tvec=np.asarray(transform.translation, dtype=np.float64),
        _rotation=np.asarray(transform.rotation.matrix(), dtype=np.float64),
    )


def _camera_model_name(camera: Any) -> str:
    model = camera.model
    name = getattr(model, "name", None)
    if name:
        return str(name)
    text = str(model)
    return text.split(".")[-1]


def _intrinsics_from_params(model: str, params: np.ndarray) -> tuple[float, float, float, float]:
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(f"Unsupported COLMAP camera model for gsplat compatibility: {model}")


def _distortion_from_params(model: str, params: np.ndarray) -> dict[str, float]:
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return {}
    if model == "SIMPLE_RADIAL":
        return {"k1": float(params[3])}
    if model == "RADIAL":
        return {"k1": float(params[3]), "k2": float(params[4])}
    if model == "OPENCV":
        return {"k1": float(params[4]), "k2": float(params[5]), "p1": float(params[6]), "p2": float(params[7])}
    if model == "OPENCV_FISHEYE":
        return {"k1": float(params[4]), "k2": float(params[5]), "k3": float(params[6]), "k4": float(params[7])}
    if model == "FULL_OPENCV":
        return {"k1": float(params[4]), "k2": float(params[5]), "p1": float(params[6]), "p2": float(params[7])}
    return {}
