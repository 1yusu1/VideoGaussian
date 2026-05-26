#!/usr/bin/env python3
"""Compatibility wrapper for videogaus.geometry.prepare_gsplat_dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from videogaus.geometry.prepare_gsplat_dataset import main


if __name__ == "__main__":
    main()
