#!/usr/bin/env python3
"""Merge multi-rank gsplat checkpoints and export one PLY file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gsplat import export_splats


SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if "splats" not in checkpoint:
        raise KeyError(f"{path} does not contain a 'splats' entry")
    return checkpoint["splats"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge rank-local gsplat checkpoints into a single PLY."
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        required=True,
        help="Checkpoint paths, e.g. ckpt_29999_rank0.pt ckpt_29999_rank1.pt.",
    )
    parser.add_argument("--output", required=True, help="Output .ply path.")
    parser.add_argument(
        "--format",
        default="ply",
        choices=["ply", "splat"],
        help="Export format supported by gsplat.export_splats.",
    )
    args = parser.parse_args()

    ckpt_paths = [Path(p).expanduser().resolve() for p in args.ckpt]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in ckpt_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing checkpoint(s):\n" + "\n".join(missing))

    print(f"Loading {len(ckpt_paths)} checkpoint(s)...")
    splat_parts = [load_checkpoint(path) for path in ckpt_paths]

    merged = {}
    for key in SPLAT_KEYS:
        if any(key not in splats for splats in splat_parts):
            raise KeyError(f"Missing splat key: {key}")
        merged[key] = torch.cat([splats[key] for splats in splat_parts], dim=0)

    print(f"Merged Gaussians: {merged['means'].shape[0]}")
    export_splats(
        means=merged["means"],
        scales=merged["scales"],
        quats=merged["quats"],
        opacities=merged["opacities"],
        sh0=merged["sh0"],
        shN=merged["shN"],
        format=args.format,
        save_to=str(output_path),
    )
    print(f"Saved merged splats to: {output_path}")


if __name__ == "__main__":
    main()
