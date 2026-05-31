# VGGTX DA3 Retained Target

## Scope

This targets branch now keeps one positive VGGTX/DA3 method: `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4`.

The goal is no longer to preserve the full ablation matrix in git. Non-selected target YAML files and non-selected result rows were removed from the tracked reports so the branch stays focused on the best balanced result.

Hard constraints kept during this cleanup:

- Do not kill processes that do not belong to this user or this experiment.
- Do not delete other users' files or existing shared experiment outputs.
- Only clean smoke-test outputs that were explicitly created for this branch.
- Leave a tracked record of what changed and which result is retained.

## Retained Result

Protocol: `liminal_pool fps12_conf96`, same DA3 output, same train/test split, same 30k gsplat budget.

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

Metrics:

| Method | PSNR | SSIM | LPIPS | #GS | Render s/img | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| `da3_gs` baseline | 26.4729 | 0.8723 | 0.2516 | 1,880,719 | 0.0094 | 1396.4150 |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` | 27.4575 | 0.8868 | 0.1586 | 2,600,000 | 0.0175 | 3408.8328 |
| `colmap_gs` reference | 34.5433 | 0.9600 | 0.0813 | 1,745,683 | 0.0095 | 1634.4473 |

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

The promoted config is:

```text
configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml
```

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
