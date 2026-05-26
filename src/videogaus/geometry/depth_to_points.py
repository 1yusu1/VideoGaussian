"""Convert depth maps and camera metadata to world-space points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from videogaus.utils.jsonio import read_json, write_json
from videogaus.utils.pointcloud import save_points_npz, write_ascii_ply


def depth_to_world_points(
    depth_path: str | Path,
    cameras_json: str | Path,
    output_npz: str | Path,
    *,
    confidence_path: str | Path | None = None,
    frame_name: str | None = None,
    stride: int = 1,
    max_depth: float | None = None,
    min_confidence: float | None = None,
    output_ply: str | Path | None = None,
) -> dict[str, Any]:
    depth = _load_array(depth_path)
    confidence = _load_array(confidence_path) if confidence_path else None
    cameras = read_json(cameras_json)
    frame = _select_frame(cameras["frames"], frame_name)
    intr = frame["intrinsics"]
    rotation = np.asarray(frame["camera_to_world"]["rotation"], dtype=np.float64)
    translation = np.asarray(frame["camera_to_world"]["translation"], dtype=np.float64)

    yy, xx = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled_depth = depth[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if max_depth is not None:
        valid &= sampled_depth <= max_depth
    sampled_conf = None
    if confidence is not None:
        sampled_conf = confidence[::stride, ::stride]
        if min_confidence is not None:
            valid &= sampled_conf >= min_confidence

    z = sampled_depth[valid]
    x = (xx[valid] - intr["cx"]) / intr["fx"] * z
    y = (yy[valid] - intr["cy"]) / intr["fy"] * z
    cam_points = np.stack([x, y, z], axis=1)
    world_points = (rotation @ cam_points.T).T + translation
    extra: dict[str, Any] = {"frame_name": frame["file_path"]}
    if sampled_conf is not None:
        extra["confidence"] = sampled_conf[valid].astype(np.float32)
    save_points_npz(output_npz, world_points.astype(np.float32), **extra)
    if output_ply:
        write_ascii_ply(output_ply, world_points.astype(np.float32))
    manifest = {
        "depth_path": str(depth_path),
        "cameras_json": str(cameras_json),
        "frame_name": frame["file_path"],
        "output_npz": str(output_npz),
        "output_ply": str(output_ply) if output_ply else None,
        "num_points": int(world_points.shape[0]),
        "stride": stride,
    }
    write_json(Path(output_npz).with_suffix(".json"), manifest)
    return manifest


def _load_array(path: str | Path | None) -> np.ndarray:
    if path is None:
        raise ValueError("Array path is required.")
    input_path = Path(path)
    if input_path.suffix.lower() == ".npz":
        data = np.load(input_path)
        key = next((k for k in ["depth", "depths", "arr_0", "confidence", "conf"] if k in data), data.files[0])
        return np.asarray(data[key])
    return np.asarray(np.load(input_path))


def _select_frame(frames: list[dict[str, Any]], frame_name: str | None) -> dict[str, Any]:
    if frame_name is None:
        return frames[0]
    for frame in frames:
        if Path(frame["file_path"]).name == Path(frame_name).name or frame["file_path"] == frame_name:
            return frame
    raise KeyError(f"Frame not found in cameras.json: {frame_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", default=None)
    parser.add_argument("--frame-name", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--output-ply", default=None)
    args = parser.parse_args()
    manifest = depth_to_world_points(
        args.depth,
        args.cameras_json,
        args.output,
        confidence_path=args.confidence,
        frame_name=args.frame_name,
        stride=args.stride,
        max_depth=args.max_depth,
        min_confidence=args.min_confidence,
        output_ply=args.output_ply,
    )
    print(manifest)


if __name__ == "__main__":
    main()
