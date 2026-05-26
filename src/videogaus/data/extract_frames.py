"""Extract frames from a casual video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from videogaus.utils.jsonio import write_json

IMAGE_EXT = ".png"


def _parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    text = value.lower().replace(",", "x")
    if "x" not in text:
        side = int(text)
        return side, side
    width, height = text.split("x", 1)
    return int(width), int(height)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    image_ext: str = IMAGE_EXT,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("OpenCV is required for frame extraction. Install with: pip install opencv-python") from exc

    input_path = Path(video_path).expanduser().resolve()
    frames_dir = Path(output_dir).expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / source_fps if source_fps > 0 else None
    frame_interval = 1
    if fps and fps > 0 and source_fps > 0:
        frame_interval = max(1, round(source_fps / fps))

    start_frame = int(round((start_time or 0.0) * source_fps)) if source_fps > 0 else 0
    end_frame = int(round(end_time * source_fps)) if end_time and source_fps > 0 else total_frames
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved: list[dict[str, Any]] = []
    frame_idx = start_frame
    output_idx = 0
    while True:
        if end_frame and frame_idx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            name = f"frame_{output_idx:06d}{image_ext}"
            out_path = frames_dir / name
            cv2.imwrite(str(out_path), frame)
            saved.append(
                {
                    "index": output_idx,
                    "source_frame": frame_idx,
                    "time_sec": frame_idx / source_fps if source_fps > 0 else None,
                    "file_name": name,
                }
            )
            output_idx += 1
            if max_frames is not None and output_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()

    manifest = {
        "video": str(input_path),
        "frames_dir": str(frames_dir),
        "source_fps": source_fps,
        "requested_fps": fps,
        "frame_interval": frame_interval,
        "source_frame_count": total_frames,
        "duration_sec": duration,
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "resize": list(resize) if resize else None,
        "num_frames": len(saved),
        "frames": saved,
    }
    write_json(frames_dir / "frames.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted frames.")
    parser.add_argument("--fps", type=float, default=None, help="Target sampling FPS.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to save.")
    parser.add_argument("--resize", default=None, help="Resize as WIDTHxHEIGHT, or one integer for square.")
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds.")
    args = parser.parse_args()

    manifest = extract_frames(
        args.video,
        args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        resize=_parse_resize(args.resize),
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(json.dumps({"frames_dir": manifest["frames_dir"], "num_frames": manifest["num_frames"]}, indent=2))


if __name__ == "__main__":
    main()
