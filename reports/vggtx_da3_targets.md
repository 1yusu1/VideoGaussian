# VGGTX DA3 Target Plan

## Hard Constraints

- Do not kill processes that do not belong to this user or this experiment.
- Do not delete other users' files or existing shared experiment outputs.
- Only clean smoke-test outputs that are clearly created for this branch's own smoke tests.
- Record code changes, experiment commands, run directories, and metric outcomes.

## Baseline And Gates

Protocol: `liminal_pool fps12_conf96`, same DA3 output, same train/test split, same 30k gsplat training budget.

Baseline naive DA3 initialization:

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `da3_gs` | 26.4729 | 0.8723 | 0.2516 |

Minimum success gate:

| Metric | Gate |
|---|---:|
| PSNR | `> 26.4729` |
| SSIM | `> 0.8723` |
| LPIPS | `< 0.2516` |

Engineering target:

| Metric | Gate |
|---|---:|
| PSNR | `>= 26.8` |
| SSIM | `>= 0.875` |
| LPIPS | `<= 0.240` |

## Next Candidate

`da3_xfeat_mask_dense_depthreg` keeps DA3 cameras unchanged and introduces VGGTX-style XFeat matches only as a confidence mask for DA3 depth point selection. This directly tests whether XFeat-guided spatial support helps without disturbing DA3's camera-depth coupling.

Expected stages:

```bash
python -m videogaus.geometry.align_cameras_epipolar \
  --config configs/da3_xfeat_mask_dense_depthreg.yaml \
  --source-dir "$RUN/da3" \
  --output-dir "$RUN/da3_xfeat_mask" \
  --xfeat-repo-dir "$XFEAT_REPO"

python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3_xfeat_mask" \
  --dense-depth-path "$RUN/da3/exports/mini_npz/results.npz" \
  --dataset-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --overwrite

python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_dense_depthreg.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_dense_depthreg/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

## Result

Remote run root:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Artifacts:

```text
da3_xfeat_mask/
da3_xfeat_mask_dense_depthreg/dataset/
da3_xfeat_mask_dense_depthreg/gsplat/
logs/da3_xfeat_mask_dense_depthreg_train_20260531.log
```

XFeat mask reconstruction:

| Item | Value |
|---|---:|
| Images | 181 |
| Candidate pairs | 1755 |
| XFeat matches | 2655834 |
| Initial median epipolar error | 1.1693 px |
| Skip pose refinement | true |
| Match-mask pixels | 20634192 |
| Exported points | 1800000 |
| Runtime | 54.0 s |

30k gsplat result:

| Method | PSNR | SSIM | LPIPS | #GS | Render s/img | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| `da3_gs` baseline | 26.4729 | 0.8723 | 0.2516 | 1880719 | 0.0094 | 1396.4150 |
| `da3_xfeat_mask_dense_depthreg` | 27.1033 | 0.8797 | 0.1845 | 2000636 | 0.0085 | 1748.4650 |

Gate status:

| Gate | Status |
|---|---|
| Minimum PSNR/SSIM/LPIPS gate | passed |
| Engineering PSNR/SSIM/LPIPS target | passed |

This result supports keeping DA3 cameras fixed while introducing VGGTX/XFeat as a support-selection signal. It also confirms that the previous negative GA results were not caused by XFeat matching itself; the damaging part was the pose-refinement path disturbing DA3 camera-depth coupling.
