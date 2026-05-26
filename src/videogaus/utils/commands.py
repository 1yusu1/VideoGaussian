"""Subprocess helpers with reproducible logging."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable


def format_command(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run_command(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    started_at = time.time()
    command_text = format_command(args)
    record: dict[str, object] = {
        "command": args,
        "command_text": command_text,
        "cwd": str(cwd) if cwd is not None else None,
        "dry_run": dry_run,
        "started_at": started_at,
    }
    if dry_run:
        record["returncode"] = 0
        record["elapsed_sec"] = 0.0
        _append_log(log_path, record)
        print(f"[dry-run] {command_text}")
        return record

    print(f"[run] {command_text}")
    completed = subprocess.run(args, cwd=cwd, env=env, check=False)
    record["returncode"] = completed.returncode
    record["elapsed_sec"] = time.time() - started_at
    _append_log(log_path, record)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, args)
    return record


def _append_log(log_path: str | Path | None, record: dict[str, object]) -> None:
    if log_path is None:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=None, sort_keys=True) + "\n")
