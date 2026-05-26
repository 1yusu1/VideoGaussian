"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    data.setdefault("_config_path", str(config_path.resolve()))
    return data


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def first_config_value(data: dict[str, Any], paths: list[str], default: Any = None) -> Any:
    for path in paths:
        value = get_nested(data, path, None)
        if value is not None:
            return value
    return default


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path | None, base: str | Path | None = None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base is not None:
        return Path(base).expanduser().resolve() / path
    return (project_root() / path).resolve()
