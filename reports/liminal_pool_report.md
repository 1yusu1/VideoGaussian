# liminal_pool Report

## Retained Positive

This report keeps the `liminal_pool` evidence focused on the single retained positive VGGTX/DA3 target:

```text
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4
```

Protocol: `fps12_conf96`, same DA3 output, same train/test split, and same 30k gsplat training budget.

Remote run root:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

## Metrics

| Role | Method | PSNR | SSIM | LPIPS | #GS |
|---|---|---:|---:|---:|---:|
| Naive DA3 baseline | `da3_gs_fps12_conf96` | 26.4729 | 0.8723 | 0.2516 | 1,880,719 |
| Retained positive | `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96` | 27.4575 | 0.8868 | 0.1586 | 2,600,000 |
| COLMAP reference | `colmap_gs_fps24_conf96` | 34.5433 | 0.9600 | 0.0813 | 1,745,683 |

Delta over naive DA3:

| Metric | Delta |
|---|---:|
| PSNR | +0.9846 |
| SSIM | +0.0145 |
| LPIPS | -0.0931 |

The target passes the original improvement gate on all three metrics. COLMAP remains stronger on this COLMAP-friendly scene, so the result should be framed as an effective DA3/VGGTX improvement, not as a COLMAP replacement.

## Components

The retained method keeps DA3 cameras fixed and uses VGGTX-style XFeat matches only as a support mask. The final gsplat training uses MCMC cap `2,600,000`, dense DA3 depth regularization weight `0.005`, dense confidence percentile `85`, and SH degree `4`.

Promoted config:

```text
configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml
```

## Runtime

| Role | Method | Training Time (s) | Render s/img | Render FPS |
|---|---|---:|---:|---:|
| Naive DA3 baseline | `da3_gs_fps12_conf96` | 1396.4150 | 0.0094 | 106.7670 |
| Retained positive | `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96` | 3408.8328 | 0.0175 | 57.1561 |
| COLMAP reference | `colmap_gs_fps24_conf96` | 1634.4473 | 0.0095 | 105.4122 |

## Retained Remote Artifacts

Remote results were pruned to the positive-only retained set. The run root changed from `41G` to `4.6G`.

Remaining paths:

```text
da3/
da3_gs/
da3_xfeat_mask/
da3_xfeat_mask_dense_depthreg/dataset/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_train_20260601.log
logs/remote_positive_cleanup_20260601.log
```

The cleanup record also remains on the remote server at `logs/remote_positive_cleanup_20260601.log`.

## Reproduce

```bash
python -m videogaus.geometry.align_cameras_epipolar \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml \
  --source-dir "$RUN/da3" \
  --output-dir "$RUN/da3_xfeat_mask" \
  --xfeat-repo-dir "$XFEAT_REPO"

python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3_xfeat_mask" \
  --dense-depth-path "$RUN/da3/exports/mini_npz/results.npz" \
  --dataset-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --overwrite

python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```
