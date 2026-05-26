"""Run a COLMAP baseline and export normalized metadata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from videogaus.utils.colmap import build_cameras_json
from videogaus.utils.commands import run_command
from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json


def run_colmap(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "cli",
    colmap_bin: str = "colmap",
    camera_model: str = "OPENCV",
    matcher: str = "sequential",
    single_camera: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    images = Path(image_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    database = out / "database.db"
    sparse = out / "sparse"
    sparse0 = sparse / "0"
    undistorted = out / "undistorted"
    txt_model = out / "sparse_txt"
    log = out / "colmap_commands.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    if backend == "pycolmap":
        return _run_pycolmap(
            images,
            out,
            database=database,
            sparse=sparse,
            sparse0=sparse0,
            undistorted=undistorted,
            txt_model=txt_model,
            log=log,
            camera_model=camera_model,
            matcher=matcher,
            single_camera=single_camera,
            dry_run=dry_run,
        )
    if backend != "cli":
        raise ValueError(f"Unsupported COLMAP backend: {backend}")

    feature_args = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.camera_model",
        camera_model,
    ]
    if single_camera:
        feature_args += ["--ImageReader.single_camera", "1"]
    run_command(feature_args, dry_run=dry_run, log_path=log)

    if matcher == "exhaustive":
        match_args = [colmap_bin, "exhaustive_matcher", "--database_path", str(database)]
    elif matcher == "vocab_tree":
        match_args = [colmap_bin, "vocab_tree_matcher", "--database_path", str(database)]
    else:
        match_args = [colmap_bin, "sequential_matcher", "--database_path", str(database)]
    run_command(match_args, dry_run=dry_run, log_path=log)

    run_command(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ],
        dry_run=dry_run,
        log_path=log,
    )

    if not dry_run:
        if not sparse0.exists():
            model_dirs = sorted(p for p in sparse.iterdir() if p.is_dir())
            if not model_dirs:
                raise RuntimeError(f"COLMAP mapper did not create a sparse model under {sparse}")
            sparse0 = model_dirs[0]
        txt_model.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(sparse0),
            "--output_path",
            str(txt_model),
            "--output_type",
            "TXT",
        ],
        dry_run=dry_run,
        log_path=log,
    )

    run_command(
        [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(images),
            "--input_path",
            str(sparse0),
            "--output_path",
            str(undistorted),
            "--output_type",
            "COLMAP",
        ],
        dry_run=dry_run,
        log_path=log,
    )

    cameras_json: dict[str, Any] = {"frames": [], "camera_models": []}
    if not dry_run:
        cameras_json = build_cameras_json(txt_model)
        write_json(out / "cameras.json", cameras_json)

    manifest = {
        "backend": backend,
        "image_dir": str(images),
        "output_dir": str(out),
        "database_path": str(database),
        "sparse_model_dir": str(sparse0),
        "sparse_txt_dir": str(txt_model),
        "undistorted_dir": str(undistorted),
        "undistorted_sparse_dir": str(undistorted / "sparse"),
        "cameras_json": str(out / "cameras.json"),
        "images_bin": str(sparse0 / "images.bin"),
        "images_txt": str(txt_model / "images.txt"),
        "points3d_bin": str(sparse0 / "points3D.bin"),
        "points3d_txt": str(txt_model / "points3D.txt"),
        "num_registered_images": len(cameras_json.get("frames", [])),
    }
    write_json(out / "colmap_manifest.json", manifest)
    return manifest


def _run_pycolmap(
    images: Path,
    out: Path,
    *,
    database: Path,
    sparse: Path,
    sparse0: Path,
    undistorted: Path,
    txt_model: Path,
    log: Path,
    camera_model: str,
    matcher: str,
    single_camera: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if matcher == "vocab_tree":
        raise ValueError("pycolmap backend supports sequential or exhaustive matching, not vocab_tree.")

    if dry_run:
        for step in ["extract_features", f"match_{matcher}", "incremental_mapping", "write_text", "undistort_images"]:
            _log_pycolmap_step(log, step, dry_run=True)
        cameras_json: dict[str, Any] = {"frames": [], "camera_models": []}
    else:
        try:
            import pycolmap
        except ImportError as exc:
            raise RuntimeError("Install a recent pycolmap build, or use --backend cli with a COLMAP binary.") from exc

        _require_pycolmap_api(pycolmap, ["extract_features", "incremental_mapping", "undistort_images"])
        camera_mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
        _run_pycolmap_step(
            log,
            "extract_features",
            pycolmap.extract_features,
            str(database),
            str(images),
            camera_mode=camera_mode,
            camera_model=camera_model,
        )

        if matcher == "exhaustive":
            _require_pycolmap_api(pycolmap, ["match_exhaustive"])
            _run_pycolmap_step(log, "match_exhaustive", pycolmap.match_exhaustive, str(database))
        else:
            _require_pycolmap_api(pycolmap, ["match_sequential"])
            _run_pycolmap_step(log, "match_sequential", pycolmap.match_sequential, str(database))

        reconstructions = _run_pycolmap_step(
            log,
            "incremental_mapping",
            pycolmap.incremental_mapping,
            str(database),
            str(images),
            str(sparse),
        )
        if not reconstructions:
            raise RuntimeError(f"pycolmap did not create a sparse model under {sparse}")
        best_key, reconstruction = _select_best_reconstruction(reconstructions)
        sparse0 = sparse / str(best_key)
        sparse0.mkdir(parents=True, exist_ok=True)
        reconstruction.write(str(sparse0))

        txt_model.mkdir(parents=True, exist_ok=True)
        reconstruction.write_text(str(txt_model))

        _run_pycolmap_step(
            log,
            "undistort_images",
            pycolmap.undistort_images,
            str(undistorted),
            str(sparse0),
            str(images),
            output_type="COLMAP",
        )
        cameras_json = build_cameras_json(txt_model)
        write_json(out / "cameras.json", cameras_json)

    manifest = {
        "backend": "pycolmap",
        "image_dir": str(images),
        "output_dir": str(out),
        "database_path": str(database),
        "sparse_model_dir": str(sparse0),
        "sparse_txt_dir": str(txt_model),
        "undistorted_dir": str(undistorted),
        "undistorted_sparse_dir": str(undistorted / "sparse"),
        "cameras_json": str(out / "cameras.json"),
        "images_bin": str(sparse0 / "images.bin"),
        "images_txt": str(txt_model / "images.txt"),
        "points3d_bin": str(sparse0 / "points3D.bin"),
        "points3d_txt": str(txt_model / "points3D.txt"),
        "num_registered_images": len(cameras_json.get("frames", [])),
    }
    write_json(out / "colmap_manifest.json", manifest)
    return manifest


def _require_pycolmap_api(pycolmap: Any, names: list[str]) -> None:
    missing = [name for name in names if not hasattr(pycolmap, name)]
    if missing:
        raise RuntimeError(f"Installed pycolmap is missing required APIs: {', '.join(missing)}")


def _run_pycolmap_step(log: Path, name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    started = time.time()
    print(f"[pycolmap] {name}")
    try:
        result = func(*args, **kwargs)
    except Exception:
        _append_pycolmap_log(log, {"step": name, "returncode": 1, "elapsed_sec": time.time() - started})
        raise
    _append_pycolmap_log(log, {"step": name, "returncode": 0, "elapsed_sec": time.time() - started})
    return result


def _log_pycolmap_step(log: Path, name: str, *, dry_run: bool) -> None:
    record = {"backend": "pycolmap", "step": name, "dry_run": dry_run, "returncode": 0, "elapsed_sec": 0.0}
    _append_pycolmap_log(log, record)
    print(f"[dry-run pycolmap] {name}")


def _append_pycolmap_log(log: Path, record: dict[str, Any]) -> None:
    record = {"backend": "pycolmap", "started_at": time.time(), **record}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=None, sort_keys=True) + "\n")


def _select_best_reconstruction(reconstructions: dict[int, Any]) -> tuple[int, Any]:
    def score(item: tuple[int, Any]) -> tuple[int, int]:
        reconstruction = item[1]
        return (_colmap_count(reconstruction, "num_reg_images"), _colmap_count(reconstruction, "num_points3D"))

    return max(reconstructions.items(), key=score)


def _colmap_count(obj: Any, name: str) -> int:
    value = getattr(obj, name, 0)
    if callable(value):
        value = value()
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backend", choices=["cli", "pycolmap"], default=None)
    parser.add_argument("--colmap-bin", default=None)
    parser.add_argument("--camera-model", default=None)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive", "vocab_tree"], default=None)
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = args.image_dir or first_config_value(cfg, ["data.frames_dir", "paths.frames_dir", "paths.images_dir"])
    output_dir = args.output_dir or first_config_value(cfg, ["colmap.output_dir", "paths.colmap_dir"], "outputs/colmap")
    if image_dir is None:
        raise SystemExit("--image-dir or data.frames_dir is required.")
    manifest = run_colmap(
        image_dir,
        output_dir,
        backend=args.backend or first_config_value(cfg, ["colmap.backend"], "cli"),
        colmap_bin=args.colmap_bin or first_config_value(cfg, ["colmap.binary"], "colmap"),
        camera_model=args.camera_model or first_config_value(cfg, ["colmap.camera_model"], "OPENCV"),
        matcher=args.matcher or first_config_value(cfg, ["colmap.matcher"], "sequential"),
        single_camera=args.single_camera or bool(first_config_value(cfg, ["colmap.single_camera"], False)),
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
