# VideoGaussian

VideoGaussian is a reproducible experiment framework for **Geometry-prior video-to-3D Gaussian Splatting reconstruction from casual videos**.

The repo keeps orchestration, configs, metrics, and reports in the main project, while external methods stay behind `third_party/` path or submodule boundaries:

- COLMAP baseline for SfM geometry.
- Depth Anything 3 for monocular video geometry priors.
- XFeat-mask DA3 target inspired by VGGT-X.
- gsplat for Gaussian Splatting training and rendering.

## Repository Layout

```text
configs/              Experiment YAML files.
scripts/              Thin shell entry points.
src/videogaus/        Python package for data, geometry, gsplat, and eval.
third_party/          Submodules or path notes only.
outputs/              Generated runs.
reports/              Markdown summaries.
assets/               Local videos and qualitative assets.
```

## Setup

Install the lightweight VideoGaussian helpers:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Install external systems separately following their upstream instructions:

- COLMAP CLI available as `colmap`.
- Depth Anything 3 CLI available as `da3`, or configured through `third_party.depth_anything_3`.
- XFeat available through a local `verlab/accelerated_features` checkout for DA3 support selection.
- gsplat examples available at `third_party/gsplat/examples` or `paths.gsplat_examples_dir`.

Public external repositories can be added as Git submodules; you usually do not need to contact maintainers, but you must follow each project license and model weight terms.

## One-Command Stages

Prepare frames and train/test split:

```bash
bash scripts/prepare_video.sh \
  --video assets/scene.mp4 \
  --frames-dir outputs/scene/frames \
  --fps 6 \
  --test-every 8
```

Run COLMAP baseline:

```bash
bash scripts/run_colmap_baseline.sh \
  --config configs/colmap_gs.yaml \
  --image-dir outputs/scene/frames \
  --output-dir outputs/scene/colmap
```

Run Depth Anything 3 pipeline:

```bash
bash scripts/run_da3_pipeline.sh \
  --config configs/da3_gs.yaml \
  --input assets/scene.mp4 \
  --output-dir outputs/scene/da3
```

Build the fixed-camera DA3 XFeat-mask target:

```bash
bash scripts/run_da3_global_align.sh \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml \
  --source-dir outputs/scene/da3 \
  --output-dir outputs/scene/da3_xfeat_mask \
  --xfeat-repo-dir /path/to/accelerated_features
```

Train gsplat from a config:

```bash
python -m videogaus.gaussian.train_gsplat --config configs/da3_gs.yaml
```

Render with a configured gsplat render command:

```bash
python -m videogaus.gaussian.render --config configs/da3_gs.yaml --split test
```

Evaluate novel-view synthesis:

```bash
bash scripts/eval_nvs.sh \
  --pred-dir outputs/scene/da3_gs/renders/test \
  --gt-dir outputs/scene/frames \
  --output-dir outputs/scene/metrics/da3_gs \
  --scene scene \
  --method da3_gs
```

Generate a report:

```bash
bash scripts/make_report.sh \
  --scene scene \
  --metrics-root outputs/scene/metrics \
  --output-dir reports
```

The script writes `reports/summary.json` plus a scene Markdown report.

## Configs

Tracked stable configs:

- `configs/colmap_gs.yaml`
- `configs/da3_gs.yaml`
- `configs/da3_gs_depthreg.yaml`
- `configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml`
- `configs/pipeline.example.yaml`

Each config includes data paths, split settings, model paths, gsplat iterations, learning rates, depth regularization weight, and output paths. Checkpoint paths and external repository paths are config fields or CLI arguments and are not hard-coded in the source.

The targets branch keeps only the balanced positive VGGTX/DA3 target in `configs/`; older ablation YAML files and non-selected result rows were removed from the tracked docs.

Current `liminal_pool fps12_conf96` DA3/VGGTX target result:

| Method | PSNR | SSIM | LPIPS | Note |
|---|---:|---:|---:|---|
| `da3_gs` | 26.4729 | 0.8723 | 0.2516 | Naive DA3 initialization baseline |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` | 27.4575 | 0.8868 | 0.1586 | Retained balanced positive target |
| `colmap_gs` | 34.5433 | 0.9600 | 0.0813 | COLMAP-friendly reference |

The retained target improves over naive DA3 by `+0.9846` PSNR, `+0.0145` SSIM, and `-0.0931` LPIPS. The compact evidence and run paths live in `reports/liminal_pool_report.md`.

## Useful Modules

```bash
python -m videogaus.data.extract_frames --video assets/scene.mp4 --output-dir outputs/scene/frames --fps 6
python -m videogaus.data.split_train_test --frames-dir outputs/scene/frames --test-every 8
python -m videogaus.geometry.depth_to_points --depth depth.npy --cameras-json cameras.json --output points.npz
python -m videogaus.geometry.align_scale --source da3_points.npz --reference outputs/scene/colmap/sparse_txt/points3D.txt --output da3_aligned.npz
python -m videogaus.geometry.align_cameras_epipolar --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml
python -m videogaus.gaussian.init_gaussians --source da3_aligned.npz --output-dir outputs/scene/init/da3
python -m videogaus.eval.summarize --metrics-root outputs/scene/metrics --output-dir reports --scene scene --report
```

## Legacy Compatibility

`scripts/run_video_to_gaussian.sh` is retained as a compatibility wrapper and now calls `src/videogaus/pipelines/video_to_gaussian.py`. New experiments should prefer the explicit stage scripts above so every geometry prior and metric artifact is easy to reproduce.
