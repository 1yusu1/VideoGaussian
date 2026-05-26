"""Refine DA3 cameras with epipolar matches and rebuild COLMAP init geometry."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from videogaus.utils.config import first_config_value, load_config
from videogaus.utils.jsonio import write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class MatchSet:
    indexes_i: np.ndarray
    indexes_j: np.ndarray
    points_i: np.ndarray
    points_j: np.ndarray
    epipolar_errors: np.ndarray
    weights: np.ndarray
    num_pairs: int
    num_matches: int
    median_epipolar_error: float


def align_da3_cameras(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    matcher: str = "opencv",
    xfeat_repo_dir: str | Path | None = None,
    match_batch_size: int = 16,
    angle_threshold: float = 30.0,
    temporal_window: int | None = None,
    max_keypoints: int = 4096,
    max_matches_per_pair: int = 2048,
    niter: int = 300,
    lr_base: float | None = None,
    lr_end: float | None = None,
    loss_mode: str = "epi",
    lambda_epipolar: float = 1.0,
    lambda_3d: float = 0.1,
    lambda_pose_reg: float = 1e-3,
    pose_rot_clamp: float = 0.0,
    pose_trans_clamp: float = 0.0,
    conf_percentile: float = 70.0,
    conf_percentile_mode: str = "per-frame",
    min_points_per_frame: int = 32,
    use_match_mask: bool = False,
    match_mask_weight_threshold: float = 0.1,
    match_mask_dilation: int = 3,
    max_points: int = 1000000,
    seed: int = 0,
    device: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    results_npz = _find_results_npz(source)
    image_dir = _find_images_dir(source)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(results_npz) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data else np.ones_like(depth, dtype=np.float32)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)

    if len(image_paths) != depth.shape[0]:
        raise ValueError(f"Image count {len(image_paths)} does not match DA3 depth count {depth.shape[0]}")
    if extrinsics.shape[:2] != (depth.shape[0], 3):
        raise ValueError(f"Expected extrinsics shape [N,3,4], got {extrinsics.shape}")
    if intrinsics.shape[:2] != (depth.shape[0], 3):
        raise ValueError(f"Expected intrinsics shape [N,3,3], got {intrinsics.shape}")

    height, width = depth.shape[1:3]
    if dry_run:
        manifest = {
            "source_dir": str(source),
            "output_dir": str(out),
            "results_npz": str(results_npz),
            "image_dir": str(image_dir),
            "num_images": len(image_paths),
            "depth_shape": list(depth.shape),
            "matcher": matcher,
            "xfeat_repo_dir": str(xfeat_repo_dir) if xfeat_repo_dir is not None else None,
            "loss_mode": loss_mode,
            "dry_run": True,
        }
        write_json(out / "da3_global_alignment_manifest.json", manifest)
        return manifest

    started = time.time()
    match_images = _load_resized_images(image_paths, width=width, height=height)
    pairs = _select_pairs(extrinsics, angle_threshold=angle_threshold, temporal_window=temporal_window)
    raw_matches, matcher_used = _extract_feature_matches(
        match_images,
        pairs,
        matcher=matcher,
        xfeat_repo_dir=xfeat_repo_dir,
        batch_size=match_batch_size,
        device=device,
        max_keypoints=max_keypoints,
        max_matches_per_pair=max_matches_per_pair,
    )
    match_set = _weight_matches(raw_matches, extrinsics, intrinsics, image_shape=(height, width))
    if lr_base is None or lr_end is None:
        lr_base, lr_end = _default_lr(match_set.median_epipolar_error)
    match_mask = _build_match_conf_mask(
        match_set,
        image_shape=(height, width),
        num_frames=len(image_paths),
        weight_threshold=match_mask_weight_threshold,
        dilation=match_mask_dilation,
    )
    refined_extrinsics, loss_history = _optimize_extrinsics(
        match_set,
        extrinsics,
        intrinsics,
        depth=depth,
        conf=conf,
        niter=niter,
        lr_base=lr_base,
        lr_end=lr_end,
        loss_mode=loss_mode,
        lambda_epipolar=lambda_epipolar,
        lambda_3d=lambda_3d,
        lambda_pose_reg=lambda_pose_reg,
        pose_rot_clamp=pose_rot_clamp,
        pose_trans_clamp=pose_trans_clamp,
        device=device,
    )

    points3d, points_xyf, colors = _points_from_depth(
        depth,
        conf,
        refined_extrinsics,
        intrinsics,
        match_images,
        conf_percentile=conf_percentile,
        conf_percentile_mode=conf_percentile_mode,
        min_points_per_frame=min_points_per_frame,
        match_mask=match_mask if use_match_mask else None,
        max_points=max_points,
        seed=seed,
    )
    _write_colmap_reconstruction(
        out,
        image_paths=image_paths,
        image_size=(width, height),
        points3d=points3d,
        points_xyf=points_xyf,
        colors=colors,
        extrinsics=refined_extrinsics,
        intrinsics=intrinsics,
    )
    _copy_images(image_paths, out / "input_images")
    np.savez_compressed(
        out / "alignment_outputs.npz",
        extrinsics=refined_extrinsics,
        intrinsics=intrinsics,
        loss_history=np.asarray(loss_history, dtype=np.float32),
        match_indexes_i=match_set.indexes_i,
        match_indexes_j=match_set.indexes_j,
        match_points_i=match_set.points_i,
        match_points_j=match_set.points_j,
        match_epipolar_errors=match_set.epipolar_errors,
        match_weights=match_set.weights,
    )
    match_filename = "matches_xfeat.npz" if matcher_used == "xfeat" else f"matches_{matcher_used}.npz"
    np.savez_compressed(
        out / match_filename,
        indexes_i=match_set.indexes_i,
        indexes_j=match_set.indexes_j,
        points_i=match_set.points_i,
        points_j=match_set.points_j,
        epipolar_errors=match_set.epipolar_errors,
        weights=match_set.weights,
    )
    np.savez_compressed(
        out / "match_conf_mask.npz",
        mask=match_mask,
        weight_threshold=np.asarray([match_mask_weight_threshold], dtype=np.float32),
        dilation=np.asarray([match_mask_dilation], dtype=np.int32),
    )
    manifest = {
        "source_dir": str(source),
        "output_dir": str(out),
        "results_npz": str(results_npz),
        "image_dir": str(image_dir),
        "num_images": len(image_paths),
        "num_pairs": match_set.num_pairs,
        "num_matches": match_set.num_matches,
        "matcher": matcher_used,
        "xfeat_repo_dir": str(xfeat_repo_dir) if xfeat_repo_dir is not None else None,
        "match_batch_size": match_batch_size,
        "initial_median_epipolar_error": match_set.median_epipolar_error,
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "loss_mode": loss_mode,
        "lr_base": lr_base,
        "lr_end": lr_end,
        "lambda_epipolar": lambda_epipolar,
        "lambda_3d": lambda_3d,
        "lambda_pose_reg": lambda_pose_reg,
        "pose_rot_clamp": pose_rot_clamp,
        "pose_trans_clamp": pose_trans_clamp,
        "angle_threshold": angle_threshold,
        "temporal_window": temporal_window,
        "conf_percentile": conf_percentile,
        "conf_percentile_mode": conf_percentile_mode,
        "min_points_per_frame": min_points_per_frame,
        "use_match_mask": use_match_mask,
        "match_mask_weight_threshold": match_mask_weight_threshold,
        "match_mask_dilation": match_mask_dilation,
        "match_mask_pixels": int(match_mask.sum()),
        "matches_file": str(out / match_filename),
        "match_conf_mask_file": str(out / "match_conf_mask.npz"),
        "num_points": int(points3d.shape[0]),
        "sparse_model_dir": str(out),
        "input_images_dir": str(out / "input_images"),
        "elapsed_sec": time.time() - started,
    }
    write_json(out / "da3_global_alignment_manifest.json", manifest)
    return manifest


def _find_results_npz(source: Path) -> Path:
    for candidate in [
        source / "exports" / "mini_npz" / "results.npz",
        source / "mini_npz" / "results.npz",
        source / "results.npz",
    ]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find DA3 mini_npz results under {source}")


def _find_images_dir(source: Path) -> Path:
    for candidate in [source / "input_images", source / "images"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find DA3 input images under {source}")


def _load_resized_images(image_paths: list[Path], *, width: int, height: int) -> np.ndarray:
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images, axis=0)


def _select_pairs(
    extrinsics: np.ndarray,
    *,
    angle_threshold: float,
    temporal_window: int | None,
) -> list[tuple[int, int]]:
    rotations = extrinsics[:, :3, :3]
    pairs: list[tuple[int, int]] = []
    for i in range(len(rotations) - 1):
        max_j = len(rotations) if temporal_window is None else min(len(rotations), i + temporal_window + 1)
        for j in range(i + 1, max_j):
            relative = rotations[i] @ rotations[j].T
            trace = float(np.trace(relative))
            angle = math.degrees(math.acos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
            if angle <= angle_threshold:
                pairs.append((i, j))
    if not pairs:
        raise RuntimeError("No candidate image pairs found for epipolar alignment.")
    return pairs


def _extract_feature_matches(
    images: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    matcher: str,
    xfeat_repo_dir: str | Path | None,
    batch_size: int,
    device: str,
    max_keypoints: int,
    max_matches_per_pair: int,
) -> tuple[dict[str, np.ndarray], str]:
    matcher = matcher.lower()
    if matcher not in {"opencv", "xfeat", "auto"}:
        raise ValueError(f"Unsupported matcher {matcher!r}; expected opencv, xfeat, or auto.")

    if matcher in {"xfeat", "auto"}:
        try:
            return (
                _extract_xfeat_matches(
                    images,
                    pairs,
                    max_keypoints=max_keypoints,
                    max_matches_per_pair=max_matches_per_pair,
                    batch_size=batch_size,
                    xfeat_repo_dir=xfeat_repo_dir,
                    device=device,
                ),
                "xfeat",
            )
        except Exception as exc:
            if matcher == "xfeat":
                raise RuntimeError(
                    "XFeat matching failed. If the server is offline, clone "
                    "verlab/accelerated_features and pass --xfeat-repo-dir."
                ) from exc
            print(f"XFeat matching failed ({exc}); falling back to OpenCV SIFT/ORB.")

    return (
        _extract_opencv_matches(
            images,
            pairs,
            max_keypoints=max_keypoints,
            max_matches_per_pair=max_matches_per_pair,
        ),
        "opencv",
    )


def _extract_xfeat_matches(
    images: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    max_keypoints: int,
    max_matches_per_pair: int,
    batch_size: int,
    xfeat_repo_dir: str | Path | None,
    device: str,
) -> dict[str, np.ndarray]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0).to(torch_device)
    xfeat = _load_xfeat(max_keypoints=max_keypoints, repo_dir=xfeat_repo_dir, device=torch_device)

    all_i = []
    all_j = []
    all_pi = []
    all_pj = []
    used_pairs = 0
    batch_size = max(int(batch_size), 1)
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            pair_batch = pairs[start : start + batch_size]
            indexes_i = [i for i, _ in pair_batch]
            indexes_j = [j for _, j in pair_batch]
            outputs = xfeat.match_xfeat_star(image_tensor[indexes_i], image_tensor[indexes_j])
            matches_batch = _normalize_xfeat_outputs(outputs, len(pair_batch), torch_device)
            for (i, j), matches in zip(pair_batch, matches_batch, strict=True):
                if matches.numel() == 0:
                    continue
                matches = matches[:max_matches_per_pair, :4].detach().cpu().numpy().astype(np.float32)
                all_i.append(np.full(len(matches), i, dtype=np.int64))
                all_j.append(np.full(len(matches), j, dtype=np.int64))
                all_pi.append(matches[:, :2])
                all_pj.append(matches[:, 2:])
                used_pairs += 1

    if not all_pi:
        raise RuntimeError("XFeat produced no correspondences.")
    return {
        "indexes_i": np.concatenate(all_i),
        "indexes_j": np.concatenate(all_j),
        "points_i": np.concatenate(all_pi, axis=0),
        "points_j": np.concatenate(all_pj, axis=0),
        "num_pairs": np.asarray([used_pairs], dtype=np.int64),
    }


def _load_xfeat(
    *,
    max_keypoints: int,
    repo_dir: str | Path | None,
    device: torch.device,
) -> Any:
    if repo_dir is not None:
        repo = Path(repo_dir).expanduser().resolve()
        if not (repo / "modules" / "xfeat.py").is_file():
            raise FileNotFoundError(f"Could not find XFeat modules/xfeat.py under {repo}")
        sys.path.insert(0, str(repo))
        try:
            from modules.xfeat import XFeat as LocalXFeat

            weights_path = repo / "weights" / "xfeat.pt"
            if weights_path.is_file():
                model = LocalXFeat(weights=str(weights_path), top_k=max_keypoints)
            else:
                model = LocalXFeat(top_k=max_keypoints)
        finally:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass
    else:
        kwargs = {"pretrained": True, "top_k": max_keypoints}
        try:
            model = torch.hub.load("verlab/accelerated_features", "XFeat", trust_repo=True, **kwargs)
        except TypeError:
            model = torch.hub.load("verlab/accelerated_features", "XFeat", **kwargs)
    if hasattr(model, "eval"):
        model = model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def _normalize_xfeat_outputs(outputs: Any, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    if batch_size == 1 and isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return [_concat_xfeat_pair(outputs[0], outputs[1], device)]
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim == 2:
            return [outputs.to(device)]
        if outputs.ndim == 3:
            return [outputs[idx].to(device) for idx in range(outputs.shape[0])]
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == batch_size and all(not isinstance(item, (tuple, list)) for item in outputs):
            return [torch.as_tensor(item, device=device) for item in outputs]
        if len(outputs) == batch_size and all(isinstance(item, (tuple, list)) and len(item) == 2 for item in outputs):
            return [_concat_xfeat_pair(item[0], item[1], device) for item in outputs]
    raise RuntimeError(f"Unexpected XFeat match output type/shape: {type(outputs)!r}")


def _concat_xfeat_pair(points_i: Any, points_j: Any, device: torch.device) -> torch.Tensor:
    pts_i = torch.as_tensor(points_i, dtype=torch.float32, device=device)
    pts_j = torch.as_tensor(points_j, dtype=torch.float32, device=device)
    if pts_i.ndim != 2 or pts_j.ndim != 2 or pts_i.shape[-1] < 2 or pts_j.shape[-1] < 2:
        raise RuntimeError("Unexpected single-pair XFeat output shape.")
    return torch.cat([pts_i[:, :2], pts_j[:, :2]], dim=-1)


def _extract_opencv_matches(
    images: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    max_keypoints: int,
    max_matches_per_pair: int,
) -> dict[str, np.ndarray]:
    import cv2

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_keypoints)
        norm = cv2.NORM_L2
        ratio = 0.75
    else:
        detector = cv2.ORB_create(nfeatures=max_keypoints)
        norm = cv2.NORM_HAMMING
        ratio = 0.85

    keypoints: list[Any] = []
    descriptors: list[np.ndarray | None] = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(desc)

    matcher = cv2.BFMatcher(norm)
    all_i = []
    all_j = []
    all_pi = []
    all_pj = []
    used_pairs = 0
    for i, j in pairs:
        if descriptors[i] is None or descriptors[j] is None:
            continue
        knn = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
        good = []
        for item in knn:
            if len(item) < 2:
                continue
            m, n = item
            if m.distance < ratio * n.distance:
                good.append(m)
        good = sorted(good, key=lambda match: match.distance)[:max_matches_per_pair]
        if not good:
            continue
        pts_i = np.asarray([keypoints[i][m.queryIdx].pt for m in good], dtype=np.float32)
        pts_j = np.asarray([keypoints[j][m.trainIdx].pt for m in good], dtype=np.float32)
        all_i.append(np.full(len(good), i, dtype=np.int64))
        all_j.append(np.full(len(good), j, dtype=np.int64))
        all_pi.append(pts_i)
        all_pj.append(pts_j)
        used_pairs += 1
    if not all_pi:
        raise RuntimeError("Feature matching produced no correspondences.")
    return {
        "indexes_i": np.concatenate(all_i),
        "indexes_j": np.concatenate(all_j),
        "points_i": np.concatenate(all_pi, axis=0),
        "points_j": np.concatenate(all_pj, axis=0),
        "num_pairs": np.asarray([used_pairs], dtype=np.int64),
    }


def _weight_matches(
    raw_matches: dict[str, np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_shape: tuple[int, int],
    err_range: float = 20.0,
    alpha: float = 0.5,
) -> MatchSet:
    indexes_i = raw_matches["indexes_i"]
    indexes_j = raw_matches["indexes_j"]
    points_i = raw_matches["points_i"]
    points_j = raw_matches["points_j"]
    errors = _epipolar_errors_numpy(points_i, points_j, indexes_i, indexes_j, extrinsics, intrinsics)
    height, width = image_shape
    valid = (
        np.isfinite(errors)
        & (points_i[:, 0] >= 0)
        & (points_i[:, 0] < width)
        & (points_i[:, 1] >= 0)
        & (points_i[:, 1] < height)
        & (points_j[:, 0] >= 0)
        & (points_j[:, 0] < width)
        & (points_j[:, 1] >= 0)
        & (points_j[:, 1] < height)
    )
    indexes_i = indexes_i[valid]
    indexes_j = indexes_j[valid]
    points_i = points_i[valid]
    points_j = points_j[valid]
    errors = errors[valid]
    if len(errors) == 0:
        raise RuntimeError("All matches were filtered out before weighting.")
    clipped = np.clip(errors, 0.0, err_range)
    hist, bin_edges = np.histogram(clipped, bins=100, range=(0.0, err_range), density=True)
    bin_ids = np.clip(np.searchsorted(bin_edges, clipped, side="right") - 1, 0, len(hist) - 1)
    weights = hist[bin_ids].astype(np.float32)
    if not np.isfinite(weights).all() or float(weights.mean()) <= 0:
        weights = np.ones_like(errors, dtype=np.float32)
    weights = weights / max(float(weights.mean()), 1e-8)
    weights = np.power(weights, alpha).astype(np.float32)
    weights = weights / max(float(weights.mean()), 1e-8)
    return MatchSet(
        indexes_i=indexes_i,
        indexes_j=indexes_j,
        points_i=points_i.astype(np.float32),
        points_j=points_j.astype(np.float32),
        epipolar_errors=errors.astype(np.float32),
        weights=weights,
        num_pairs=int(raw_matches["num_pairs"][0]),
        num_matches=int(len(points_i)),
        median_epipolar_error=float(np.median(errors)),
    )


def _default_lr(epipolar_error: float, bound1: float = 2.5, bound2: float = 7.5) -> tuple[float, float]:
    if epipolar_error > bound2:
        lr_base = 1e-2
    elif epipolar_error < bound1:
        lr_base = 5e-4
    else:
        lr_base = 1e-3
    return lr_base, lr_base / 10.0


def _optimize_extrinsics(
    matches: MatchSet,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    niter: int,
    lr_base: float,
    lr_end: float,
    loss_mode: str,
    lambda_epipolar: float,
    lambda_3d: float,
    lambda_pose_reg: float,
    pose_rot_clamp: float,
    pose_trans_clamp: float,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    loss_mode = loss_mode.lower()
    if loss_mode not in {"epi", "epi3d"}:
        raise ValueError("loss_mode must be 'epi' or 'epi3d'.")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float32
    rotations0 = torch.as_tensor(extrinsics[:, :3, :3], dtype=dtype, device=device)
    translations0 = torch.as_tensor(extrinsics[:, :3, 3], dtype=dtype, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=dtype, device=device)
    idx_i = torch.as_tensor(matches.indexes_i, dtype=torch.long, device=device)
    idx_j = torch.as_tensor(matches.indexes_j, dtype=torch.long, device=device)
    pts_i = torch.as_tensor(matches.points_i, dtype=dtype, device=device)
    pts_j = torch.as_tensor(matches.points_j, dtype=dtype, device=device)
    weights = torch.as_tensor(matches.weights, dtype=dtype, device=device)
    weights = weights / weights.mean().clamp_min(1e-8)

    match_depth_i: torch.Tensor | None = None
    match_depth_j: torch.Tensor | None = None
    match_conf_weights: torch.Tensor | None = None
    valid_3d: torch.Tensor | None = None
    if loss_mode == "epi3d":
        depth_t = torch.as_tensor(depth, dtype=dtype, device=device)
        conf_t = torch.as_tensor(conf, dtype=dtype, device=device)
        match_depth_i = _bilinear_sample_frames(depth_t, idx_i, pts_i)
        match_depth_j = _bilinear_sample_frames(depth_t, idx_j, pts_j)
        conf_i = _bilinear_sample_frames(conf_t, idx_i, pts_i).clamp_min(0.0)
        conf_j = _bilinear_sample_frames(conf_t, idx_j, pts_j).clamp_min(0.0)
        valid_3d = torch.isfinite(match_depth_i) & torch.isfinite(match_depth_j) & (match_depth_i > 0.0) & (match_depth_j > 0.0)
        if bool(valid_3d.any()):
            match_conf_weights = torch.sqrt(conf_i[valid_3d] * conf_j[valid_3d]).clamp_min(1e-8)
            match_conf_weights = match_conf_weights / match_conf_weights.mean().clamp_min(1e-8)
        else:
            loss_mode = "epi"

    rot_delta = torch.zeros((len(extrinsics), 3), dtype=dtype, device=device, requires_grad=True)
    trans_delta = torch.zeros((len(extrinsics), 3), dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rot_delta, trans_delta], lr=lr_base, betas=(0.9, 0.9))
    scene_scale = _scene_scale_torch(rotations0, translations0).clamp_min(1e-3)
    history: list[float] = []
    for step in range(max(niter, 1)):
        alpha = step / max(niter - 1, 1)
        lr = lr_end + (lr_base - lr_end) * (1.0 + math.cos(alpha * math.pi)) / 2.0
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        rot_used = rot_delta.clone()
        trans_used = trans_delta.clone()
        rot_used[0] = 0.0
        trans_used[0] = 0.0
        rotations = _so3_exp(rot_used) @ rotations0
        translations = translations0 + trans_used
        errors = _epipolar_errors_torch(pts_i, pts_j, idx_i, idx_j, rotations, translations, intrinsics_t)
        epi_loss = (errors * weights).mean() * lambda_epipolar
        consistency_loss = torch.zeros((), dtype=dtype, device=device)
        if loss_mode == "epi3d" and valid_3d is not None and match_depth_i is not None and match_depth_j is not None:
            world_i = _unproject_match_points_torch(
                pts_i[valid_3d],
                match_depth_i[valid_3d],
                idx_i[valid_3d],
                rotations,
                translations,
                intrinsics_t,
            )
            world_j = _unproject_match_points_torch(
                pts_j[valid_3d],
                match_depth_j[valid_3d],
                idx_j[valid_3d],
                rotations,
                translations,
                intrinsics_t,
            )
            distances = torch.linalg.norm(world_i - world_j, dim=-1) / scene_scale
            robust = torch.sqrt(distances.square() + 1e-6)
            consistency_weights = weights[valid_3d]
            if match_conf_weights is not None:
                consistency_weights = consistency_weights * match_conf_weights
            consistency_weights = consistency_weights / consistency_weights.mean().clamp_min(1e-8)
            consistency_loss = (robust * consistency_weights).mean() * lambda_3d
        reg_loss = lambda_pose_reg * (rot_used.square().mean() + (trans_used / scene_scale).square().mean())
        loss = epi_loss + consistency_loss + reg_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if pose_rot_clamp > 0:
                rot_delta[1:].clamp_(min=-pose_rot_clamp, max=pose_rot_clamp)
            if pose_trans_clamp > 0:
                trans_limit = pose_trans_clamp * scene_scale
                trans_delta[1:].clamp_(min=-trans_limit, max=trans_limit)
        history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        rot_used = rot_delta.clone()
        trans_used = trans_delta.clone()
        rot_used[0] = 0.0
        trans_used[0] = 0.0
        rotations = (_so3_exp(rot_used) @ rotations0).detach().cpu().numpy()
        translations = (translations0 + trans_used).detach().cpu().numpy()
    refined = extrinsics.copy()
    refined[:, :3, :3] = rotations
    refined[:, :3, 3] = translations
    return refined, history


def _epipolar_errors_numpy(
    points_i: np.ndarray,
    points_j: np.ndarray,
    indexes_i: np.ndarray,
    indexes_j: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        errors = _epipolar_errors_torch(
            torch.as_tensor(points_i, dtype=torch.float32),
            torch.as_tensor(points_j, dtype=torch.float32),
            torch.as_tensor(indexes_i, dtype=torch.long),
            torch.as_tensor(indexes_j, dtype=torch.long),
            torch.as_tensor(extrinsics[:, :3, :3], dtype=torch.float32),
            torch.as_tensor(extrinsics[:, :3, 3], dtype=torch.float32),
            torch.as_tensor(intrinsics, dtype=torch.float32),
        )
    return errors.cpu().numpy()


def _bilinear_sample_frames(values: torch.Tensor, indexes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[1:3]
    x = points[:, 0].clamp(0, width - 1)
    y = points[:, 1].clamp(0, height - 1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0.to(points.dtype)
    wy = y - y0.to(points.dtype)
    v00 = values[indexes, y0, x0]
    v01 = values[indexes, y0, x1]
    v10 = values[indexes, y1, x0]
    v11 = values[indexes, y1, x1]
    return (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v01 * wx * (1.0 - wy)
        + v10 * (1.0 - wx) * wy
        + v11 * wx * wy
    )


def _unproject_match_points_torch(
    points: torch.Tensor,
    depths: torch.Tensor,
    indexes: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    k = intrinsics[indexes]
    z = depths
    x = (points[:, 0] - k[:, 0, 2]) / k[:, 0, 0].clamp_min(1e-8) * z
    y = (points[:, 1] - k[:, 1, 2]) / k[:, 1, 1].clamp_min(1e-8) * z
    cam_points = torch.stack([x, y, z], dim=-1)
    r = rotations[indexes]
    t = translations[indexes]
    return (r.transpose(-1, -2) @ (cam_points - t)[..., None]).squeeze(-1)


def _epipolar_errors_torch(
    points_i: torch.Tensor,
    points_j: torch.Tensor,
    indexes_i: torch.Tensor,
    indexes_j: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    r_i = rotations[indexes_i]
    r_j = rotations[indexes_j]
    t_i = translations[indexes_i]
    t_j = translations[indexes_j]
    k_i = intrinsics[indexes_i]
    k_j = intrinsics[indexes_j]
    r_ji = r_j @ r_i.transpose(-1, -2)
    t_ji = t_j - (r_ji @ t_i[..., None]).squeeze(-1)
    essential = _skew(t_ji) @ r_ji
    fundamental = torch.linalg.inv(k_j).transpose(-1, -2) @ essential @ torch.linalg.inv(k_i)
    ones = torch.ones((points_i.shape[0], 1), dtype=points_i.dtype, device=points_i.device)
    x_i = torch.cat([points_i, ones], dim=-1)
    x_j = torch.cat([points_j, ones], dim=-1)
    fx_i = (fundamental @ x_i[..., None]).squeeze(-1)
    ftx_j = (fundamental.transpose(-1, -2) @ x_j[..., None]).squeeze(-1)
    xfx = (x_j * fx_i).sum(dim=-1).abs()
    denom_i = torch.linalg.norm(fx_i[:, :2], dim=-1).clamp_min(1e-8)
    denom_j = torch.linalg.norm(ftx_j[:, :2], dim=-1).clamp_min(1e-8)
    return 0.5 * xfx * (1.0 / denom_i + 1.0 / denom_j)


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[:, 0])
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    omega = _skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[0], 3, 3)
    a = torch.sin(theta)[..., None] / theta[..., None]
    b = (1.0 - torch.cos(theta))[..., None] / (theta[..., None] ** 2)
    return eye + a * omega + b * (omega @ omega)


def _scene_scale_torch(rotations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    centers = -(rotations.transpose(-1, -2) @ translations[..., None]).squeeze(-1)
    center = centers.mean(dim=0, keepdim=True)
    return torch.linalg.norm(centers - center, dim=-1).median()


def _build_match_conf_mask(
    matches: MatchSet,
    *,
    image_shape: tuple[int, int],
    num_frames: int,
    weight_threshold: float,
    dilation: int,
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((num_frames, height, width), dtype=bool)
    selected = matches.weights > weight_threshold
    for indexes, points in [
        (matches.indexes_i[selected], matches.points_i[selected]),
        (matches.indexes_j[selected], matches.points_j[selected]),
    ]:
        xs = np.rint(points[:, 0]).astype(np.int64)
        ys = np.rint(points[:, 1]).astype(np.int64)
        valid = (
            (indexes >= 0)
            & (indexes < num_frames)
            & (xs >= 0)
            & (xs < width)
            & (ys >= 0)
            & (ys < height)
        )
        mask[indexes[valid], ys[valid], xs[valid]] = True
    dilation = int(dilation)
    if dilation <= 0 or not mask.any():
        return mask
    try:
        import cv2

        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=np.uint8)
        for frame_idx in range(num_frames):
            mask[frame_idx] = cv2.dilate(mask[frame_idx].astype(np.uint8), kernel, iterations=1).astype(bool)
        return mask
    except Exception:
        padded = np.pad(mask, ((0, 0), (dilation, dilation), (dilation, dilation)), mode="constant")
        dilated = np.zeros_like(mask)
        for dy in range(2 * dilation + 1):
            for dx in range(2 * dilation + 1):
                dilated |= padded[:, dy : dy + height, dx : dx + width]
        return dilated


def _points_from_depth(
    depth: np.ndarray,
    conf: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    images: np.ndarray,
    *,
    conf_percentile: float,
    conf_percentile_mode: str,
    min_points_per_frame: int,
    match_mask: np.ndarray | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf_percentile_mode = conf_percentile_mode.lower()
    if conf_percentile_mode not in {"global", "per-frame"}:
        raise ValueError("conf_percentile_mode must be 'global' or 'per-frame'.")
    if match_mask is not None and match_mask.shape != depth.shape:
        raise ValueError(f"match_mask shape {match_mask.shape} must match depth shape {depth.shape}.")

    base_valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf)
    global_threshold = float(np.percentile(conf[base_valid], conf_percentile)) if np.any(base_valid) else math.inf
    frame_candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    empty_frames: list[int] = []
    for frame_idx in range(depth.shape[0]):
        frame_base = base_valid[frame_idx]
        if not np.any(frame_base):
            empty_frames.append(frame_idx)
            continue
        if conf_percentile_mode == "per-frame":
            threshold = float(np.percentile(conf[frame_idx][frame_base], conf_percentile))
        else:
            threshold = global_threshold
        frame_valid = frame_base & (conf[frame_idx] >= threshold)
        if match_mask is not None:
            frame_valid |= frame_base & match_mask[frame_idx]
        ys_i, xs_i = np.nonzero(frame_valid)
        if len(ys_i) == 0:
            empty_frames.append(frame_idx)
            continue
        fs_i = np.full(len(ys_i), frame_idx, dtype=np.int64)
        frame_candidates.append((fs_i, ys_i.astype(np.int64), xs_i.astype(np.int64)))

    if not frame_candidates:
        raise RuntimeError("No DA3 depth pixels survived confidence filtering.")
    if empty_frames:
        raise RuntimeError(
            "Some frames have no DA3 depth pixels after confidence filtering: "
            f"{empty_frames[:10]}{'...' if len(empty_frames) > 10 else ''}. "
            "Try --conf-percentile-mode per-frame or a lower --conf-percentile."
        )

    rng = np.random.default_rng(seed)
    counts = np.asarray([len(frame[0]) for frame in frame_candidates], dtype=np.int64)
    if int(counts.sum()) > max_points:
        quotas = _allocate_frame_point_quotas(counts, max_points=max_points, min_points_per_frame=min_points_per_frame)
        sampled_frames = []
        for (fs_i, ys_i, xs_i), quota in zip(frame_candidates, quotas, strict=True):
            if quota < len(fs_i):
                keep = rng.choice(len(fs_i), size=int(quota), replace=False)
                sampled_frames.append((fs_i[keep], ys_i[keep], xs_i[keep]))
            else:
                sampled_frames.append((fs_i, ys_i, xs_i))
        frame_candidates = sampled_frames

    fs = np.concatenate([frame[0] for frame in frame_candidates])
    ys = np.concatenate([frame[1] for frame in frame_candidates])
    xs = np.concatenate([frame[2] for frame in frame_candidates])

    points = np.empty((len(fs), 3), dtype=np.float32)
    for frame_idx in np.unique(fs):
        mask = fs == frame_idx
        z = depth[frame_idx, ys[mask], xs[mask]].astype(np.float64)
        k = intrinsics[frame_idx]
        x = (xs[mask] - k[0, 2]) / k[0, 0] * z
        y = (ys[mask] - k[1, 2]) / k[1, 1] * z
        cam_points = np.stack([x, y, z], axis=-1)
        r = extrinsics[frame_idx, :3, :3].astype(np.float64)
        t = extrinsics[frame_idx, :3, 3].astype(np.float64)
        points[mask] = ((r.T @ (cam_points - t).T).T).astype(np.float32)

    colors = images[fs, ys, xs].astype(np.uint8)
    points_xyf = np.stack([xs, ys, fs], axis=-1).astype(np.float32)
    return points, points_xyf, colors


def _allocate_frame_point_quotas(
    counts: np.ndarray,
    *,
    max_points: int,
    min_points_per_frame: int,
) -> np.ndarray:
    num_frames = int(len(counts))
    if max_points < num_frames:
        raise ValueError(f"max_points={max_points} is smaller than the number of frames with valid points ({num_frames}).")
    min_points_per_frame = max(int(min_points_per_frame), 1)
    guaranteed = np.minimum(counts, min_points_per_frame).astype(np.int64)
    if int(guaranteed.sum()) > max_points:
        guaranteed = np.ones_like(counts, dtype=np.int64)
    quotas = guaranteed.copy()
    remaining = int(max_points - quotas.sum())
    capacity = counts - quotas
    if remaining > 0 and int(capacity.sum()) > 0:
        raw = capacity.astype(np.float64) / float(capacity.sum()) * remaining
        extra = np.floor(raw).astype(np.int64)
        extra = np.minimum(extra, capacity)
        quotas += extra
        remaining = int(max_points - quotas.sum())
        if remaining > 0:
            order = np.argsort(-(raw - extra))
            for idx in order:
                if remaining <= 0:
                    break
                if quotas[idx] < counts[idx]:
                    quotas[idx] += 1
                    remaining -= 1
    return quotas


def _write_colmap_reconstruction(
    output_dir: Path,
    *,
    image_paths: list[Path],
    image_size: tuple[int, int],
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    point_ids = []
    for xyz, rgb in zip(points3d, colors, strict=True):
        point_ids.append(reconstruction.add_point3D(xyz, pycolmap.Track(), rgb))

    width, height = image_size
    for frame_idx, image_path in enumerate(image_paths):
        params = np.asarray(
            [
                intrinsics[frame_idx, 0, 0],
                intrinsics[frame_idx, 1, 1],
                intrinsics[frame_idx, 0, 2],
                intrinsics[frame_idx, 1, 2],
            ],
            dtype=np.float64,
        )
        camera = pycolmap.Camera(model="PINHOLE", width=width, height=height, params=params, camera_id=frame_idx + 1)
        reconstruction.add_camera(camera)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[frame_idx, :3, :3]),
            extrinsics[frame_idx, :3, 3],
        )
        image = pycolmap.Image(
            id=frame_idx + 1,
            name=image_path.name,
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )
        point_rows = np.nonzero(points_xyf[:, 2].astype(np.int32) == frame_idx)[0]
        points2d = []
        for point2d_idx, point_row in enumerate(point_rows):
            point_id = int(point_ids[point_row])
            xy = points_xyf[point_row, :2].astype(np.float64)
            points2d.append(pycolmap.Point2D(xy, point_id))
            reconstruction.points3D[point_id].track.add_element(frame_idx + 1, point2d_idx)
        image.points2D = pycolmap.ListPoint2D(points2d)
        image.registered = True
        reconstruction.add_image(image)

    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_dir))


def _copy_images(image_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        target = output_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional VideoGaussian YAML config.")
    parser.add_argument("--source-dir", default=None, help="DA3 output directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory for refined DA3 COLMAP model.")
    parser.add_argument("--matcher", choices=["opencv", "xfeat", "auto"], default=None)
    parser.add_argument("--xfeat-repo-dir", default=None, help="Optional local verlab/accelerated_features checkout.")
    parser.add_argument("--match-batch-size", type=int, default=None)
    parser.add_argument("--angle-threshold", type=float, default=None)
    parser.add_argument("--temporal-window", type=int, default=None)
    parser.add_argument("--max-keypoints", type=int, default=None)
    parser.add_argument("--max-matches-per-pair", type=int, default=None)
    parser.add_argument("--niter", type=int, default=None)
    parser.add_argument("--lr-base", type=float, default=None)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument("--loss-mode", choices=["epi", "epi3d"], default=None)
    parser.add_argument("--lambda-epipolar", type=float, default=None)
    parser.add_argument("--lambda-3d", type=float, default=None)
    parser.add_argument("--lambda-pose-reg", type=float, default=None)
    parser.add_argument("--pose-rot-clamp", type=float, default=None)
    parser.add_argument("--pose-trans-clamp", type=float, default=None)
    parser.add_argument("--conf-percentile", type=float, default=None)
    parser.add_argument("--conf-percentile-mode", choices=["global", "per-frame"], default=None)
    parser.add_argument("--min-points-per-frame", type=int, default=None)
    parser.add_argument("--use-match-mask", action="store_true", default=None)
    parser.add_argument("--match-mask-weight-threshold", type=float, default=None)
    parser.add_argument("--match-mask-dilation", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    def value(cli_value: Any, paths: list[str], default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        return first_config_value(cfg, paths, default)

    source_dir = value(args.source_dir, ["paths.da3_output_dir", "global_alignment.source_dir"], None)
    output_dir = value(args.output_dir, ["paths.ga_output_dir", "global_alignment.output_dir"], None)
    if source_dir is None:
        raise SystemExit("--source-dir or paths.da3_output_dir is required.")
    if output_dir is None:
        raise SystemExit("--output-dir or paths.ga_output_dir is required.")

    manifest = align_da3_cameras(
        source_dir,
        output_dir,
        matcher=value(args.matcher, ["global_alignment.matcher"], "opencv"),
        xfeat_repo_dir=value(args.xfeat_repo_dir, ["paths.xfeat_repo_dir", "global_alignment.xfeat_repo_dir"], None),
        match_batch_size=int(value(args.match_batch_size, ["global_alignment.match_batch_size"], 16)),
        angle_threshold=float(value(args.angle_threshold, ["global_alignment.angle_threshold"], 30.0)),
        temporal_window=value(args.temporal_window, ["global_alignment.temporal_window"], None),
        max_keypoints=int(value(args.max_keypoints, ["global_alignment.max_keypoints"], 4096)),
        max_matches_per_pair=int(value(args.max_matches_per_pair, ["global_alignment.max_matches_per_pair"], 2048)),
        niter=int(value(args.niter, ["global_alignment.niter"], 300)),
        lr_base=args.lr_base,
        lr_end=args.lr_end,
        loss_mode=value(args.loss_mode, ["global_alignment.loss_mode"], "epi"),
        lambda_epipolar=float(value(args.lambda_epipolar, ["global_alignment.lambda_epipolar"], 1.0)),
        lambda_3d=float(value(args.lambda_3d, ["global_alignment.lambda_3d"], 0.1)),
        lambda_pose_reg=float(value(args.lambda_pose_reg, ["global_alignment.lambda_pose_reg"], 1e-3)),
        pose_rot_clamp=float(value(args.pose_rot_clamp, ["global_alignment.pose_rot_clamp"], 0.0)),
        pose_trans_clamp=float(value(args.pose_trans_clamp, ["global_alignment.pose_trans_clamp"], 0.0)),
        conf_percentile=float(value(args.conf_percentile, ["global_alignment.conf_percentile"], 70.0)),
        conf_percentile_mode=value(args.conf_percentile_mode, ["global_alignment.conf_percentile_mode"], "per-frame"),
        min_points_per_frame=int(value(args.min_points_per_frame, ["global_alignment.min_points_per_frame"], 32)),
        use_match_mask=bool(value(args.use_match_mask, ["global_alignment.use_match_mask"], False)),
        match_mask_weight_threshold=float(
            value(args.match_mask_weight_threshold, ["global_alignment.match_mask_weight_threshold"], 0.1)
        ),
        match_mask_dilation=int(value(args.match_mask_dilation, ["global_alignment.match_mask_dilation"], 3)),
        max_points=int(value(args.max_points, ["global_alignment.max_points"], 1000000)),
        seed=int(value(args.seed, ["global_alignment.seed"], 0)),
        device=value(args.device, ["global_alignment.device"], "cuda"),
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
