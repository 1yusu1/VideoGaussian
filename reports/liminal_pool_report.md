# liminal_pool Report

## Scope

This report records the retained positive VGGTX/DA3 target for `liminal_pool` and keeps the tracked evidence compact. The goal is no longer to preserve the full ablation matrix in git; non-selected target YAML files and non-selected result rows were removed from tracked reports so the branch stays focused on the best balanced result.

Protocol: `liminal_pool fps12_conf96`, using the same DA3 output, same train/test split, and same 30k gsplat training budget.

Remote run root:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Retained artifacts:

```text
da3_xfeat_mask/
da3_xfeat_mask_dense_depthreg/dataset/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_train_20260601.log
```

## Retained Methods

| Setting | Method | Result Directory |
|---|---|---|
| fps12_conf96 | da3_gs_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_gs/gsplat |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat |
| fps24_conf96 | colmap_gs_fps24_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_colmap_vs_da3/colmap_gs/gsplat |

## PSNR/SSIM/LPIPS

| Setting | Method | PSNR | SSIM | LPIPS | #GS |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_gs_fps12_conf96 | 26.4729 | 0.8723 | 0.2516 | 1880719 |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | 27.4575 | 0.8868 | 0.1586 | 2600000 |
| fps24_conf96 | colmap_gs_fps24_conf96 | 34.5433 | 0.9600 | 0.0813 | 1745683 |

Delta over naive DA3:

| Metric | Delta |
|---|---:|
| PSNR | +0.9846 |
| SSIM | +0.0145 |
| LPIPS | -0.0931 |

Gap to COLMAP reference:

| Metric | Gap |
|---|---:|
| PSNR | -7.0859 |
| SSIM | -0.0732 |
| LPIPS | +0.0773 |

## Retained Components

The retained method combines these components:

- Fixed DA3 cameras; no pose refinement.
- VGGTX-style XFeat matches used only as a match-mask support signal.
- DA3 dense-depth point selection with per-frame confidence percentile `96`.
- XFeat match-mask dilation `3`, match-mask weight threshold `0.1`, max exported points `1,800,000`.
- gsplat MCMC with cap `2,600,000`.
- Dense DA3 depth regularization with weight `0.005`, confidence percentile `85`, confidence weighting enabled.
- SH degree `4`.

Promoted config:

```text
configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml
```

## Key Observations

- The retained VGGTX/DA3 target beats naive DA3 initialization by +0.9846 PSNR, +0.0145 SSIM, and -0.0931 LPIPS.
- The retained target improves DA3 while preserving the framing that DA3 is a geometry prior, not a guaranteed COLMAP replacement.
- COLMAP remains the reference upper bound on this COLMAP-friendly scene, with a gap of -7.0859 PSNR, -0.0732 SSIM, and +0.0773 LPIPS from the retained target.
- Non-selected target rows were removed from this tracked report; remote training outputs were not deleted.

## Runtime

| Setting | Method | Training Time (s) | Render s/img | Render FPS | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_gs_fps12_conf96 | 1396.4150 | 0.0094 | 106.7670 |  |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | 3408.8328 | 0.0175 | 57.1561 |  |
| fps24_conf96 | colmap_gs_fps24_conf96 | 1634.4473 | 0.0095 | 105.4122 |  |

## Reproduction Commands

Build the XFeat-mask DA3 model:

```bash
python -m videogaus.geometry.align_cameras_epipolar \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml \
  --source-dir "$RUN/da3" \
  --output-dir "$RUN/da3_xfeat_mask" \
  --xfeat-repo-dir "$XFEAT_REPO"
```

Prepare the gsplat dataset:

```bash
python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3_xfeat_mask" \
  --dense-depth-path "$RUN/da3/exports/mini_npz/results.npz" \
  --dataset-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --overwrite
```

Train the retained target:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

## Cleanup Record

Removed tracked target YAML files that were not the retained balanced positive method:

- `configs/da3_ga_xfeat_v2_gs.yaml`
- `configs/da3_ga_xfeat_v2_mcmc_pose_depthreg.yaml`
- `configs/da3_xfeat_mask_dense_depthreg.yaml`
- `configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01.yaml`
- `configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01.yaml`
- `configs/da3_xfeat_mask_mcmc_pose_dense_depthreg.yaml`

Tracked reports were filtered to the baseline, retained target, and COLMAP reference. Remote training outputs were not deleted.
