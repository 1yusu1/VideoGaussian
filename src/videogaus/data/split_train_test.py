"""Create reproducible train/test frame splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from videogaus.utils.jsonio import read_json, write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_frames(frames_dir: str | Path, manifest: str | Path | None = None) -> list[str]:
    if manifest is not None and Path(manifest).is_file():
        data = read_json(manifest)
        return [frame["file_name"] for frame in data.get("frames", [])]
    path = Path(frames_dir)
    return sorted(p.name for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def split_frames(
    frames: list[str],
    *,
    test_every: int | None = None,
    test_ratio: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("No frames were found.")
    indices = list(range(len(frames)))
    if test_every is not None and test_every > 0:
        test_idx = [idx for idx in indices if idx % test_every == 0]
    elif test_ratio is not None and test_ratio > 0:
        rng = random.Random(seed)
        shuffled = indices[:]
        rng.shuffle(shuffled)
        count = max(1, int(round(len(indices) * test_ratio)))
        test_idx = sorted(shuffled[:count])
    else:
        test_idx = [idx for idx in indices if idx % 8 == 0]
    test_set = set(test_idx)
    train_idx = [idx for idx in indices if idx not in test_set]
    if not train_idx:
        train_idx = [idx for idx in indices if idx != test_idx[0]]
        test_idx = [test_idx[0]]
    return {
        "train": [frames[idx] for idx in train_idx],
        "test": [frames[idx] for idx in test_idx],
        "train_indices": train_idx,
        "test_indices": test_idx,
        "num_frames": len(frames),
        "num_train": len(train_idx),
        "num_test": len(test_idx),
        "strategy": {
            "test_every": test_every,
            "test_ratio": test_ratio,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, help="Directory containing extracted frames.")
    parser.add_argument("--manifest", default=None, help="Optional frames.json from extract_frames.")
    parser.add_argument("--output", default=None, help="Output splits.json path. Defaults to frames-dir/splits.json.")
    parser.add_argument("--test-every", type=int, default=None, help="Use every Nth frame as test.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Random test ratio, e.g. 0.1.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frames = list_frames(args.frames_dir, args.manifest)
    splits = split_frames(frames, test_every=args.test_every, test_ratio=args.test_ratio, seed=args.seed)
    output = Path(args.output) if args.output else Path(args.frames_dir) / "splits.json"
    write_json(output, splits)
    print(f"Wrote split with {splits['num_train']} train and {splits['num_test']} test frames: {output}")


if __name__ == "__main__":
    main()
