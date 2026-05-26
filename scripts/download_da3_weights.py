#!/usr/bin/env python3
"""Download DA3 weights to a fixed local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_LOCAL_DIR = (
    Path.home()
    / "DepthAnything3"
    / "src"
    / "depth_anything_3"
    / "models"
    / "DA3NESTED-GIANT-LARGE-1.1"
)


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Depth Anything 3 weights.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help="Directory to store the downloaded model files.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Keep partial downloads and resume when supported by huggingface_hub.",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo_id}")
    print(f"Local dir: {local_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=args.resume_download,
    )

    required_files = ["config.json", "model.safetensors"]
    missing = [name for name in required_files if not (local_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required file(s): {', '.join(missing)}")

    model_file = local_dir / "model.safetensors"
    print("Download complete.")
    print(f"config.json: {(local_dir / 'config.json').stat().st_size} bytes")
    print(f"model.safetensors: {format_size(model_file.stat().st_size)}")
    print()
    print("Use this model path with DA3:")
    print(f"  --model-dir {local_dir}")


if __name__ == "__main__":
    main()
