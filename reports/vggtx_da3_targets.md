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

## Additional Component Results

After the fixed-camera XFeat-mask result passed the gate, two additional gsplat-side components were tested on the same dataset:

Note: the first historical MCMC config in this section was retired from tracked `configs/` after stronger targets superseded it. The metric evidence and remote artifacts are retained here; recreate it by copying the closest retained fixed-camera XFeat-mask config and setting MCMC cap `2.2M` with dense depth weight `0.02`/confidence percentile `50`.

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_dense_depthreg.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_dense_depthreg/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_pose_dense_depthreg.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_pose_dense_depthreg/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

Additional artifacts:

```text
da3_xfeat_mask_mcmc_dense_depthreg/gsplat/
da3_xfeat_mask_mcmc_pose_dense_depthreg/gsplat/
logs/da3_xfeat_mask_mcmc_dense_depthreg_train_20260531.log
logs/da3_xfeat_mask_mcmc_pose_dense_depthreg_train_20260531.log
```

Both smoke tests were run for 1000 steps and their `smoke_gsplat` directories were deleted only after `realpath` confirmed that they were inside this experiment's own output directory.

30k additional results:

| Method | PSNR | SSIM | LPIPS | #GS | Render s/img | Train Time (s) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `da3_xfeat_mask_dense_depthreg` | 27.1033 | 0.8797 | 0.1845 | 2000636 | 0.0085 | 1748.4650 | Best fixed-camera LPIPS without MCMC |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` | 27.4575 | 0.8868 | 0.1586 | 2600000 | 0.0175 | 3408.8328 | Best no-pose PSNR/SSIM/LPIPS target so far; promoted config |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01` | 27.4601 | 0.8854 | 0.1723 | 2600000 | 0.0179 | 3644.2475 | Best PSNR-only target; worse SSIM/LPIPS than the balanced SH4 target |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01` | 27.3809 | 0.8835 | 0.1549 | 2600000 | 0.0185 | 5836.3322 | Best no-pose LPIPS/perceptual target; promoted config |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2` | 27.4336 | 0.8850 | 0.1555 | 2600000 | 0.0185 | 5990.8590 | Batch-size 2 improves LPIPS but costs PSNR/SSIM and training time |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85` | 27.4335 | 0.8864 | 0.1616 | 2600000 | 0.0165 | 3049.0354 | Previous weak-depth target |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_shint500` | 27.4306 | 0.8861 | 0.1621 | 2600000 | 0.0176 | 3453.3246 | Earlier SH activation does not beat the balanced SH4 target |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_visible_adam` | 27.4007 | 0.8852 | 0.1639 | 2600000 | 0.0145 | 2305.2455 | Visible Adam is a speed component, not a quality improvement |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scaleslr01` | 27.3842 | 0.8850 | 0.1629 | 2600000 | 0.0174 | 3398.3758 | Higher scale LR alone does not improve final metrics |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_sh4` | 27.4235 | 0.8854 | 0.1654 | 2600000 | 0.0174 | 3465.6859 | SH degree 4 is close to best but slower |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf90` | 27.3766 | 0.8856 | 0.1645 | 2600000 | 0.0167 | 3069.3637 | Raising confidence to 90 hurts PSNR versus conf85 |
| `da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf90` | 27.3725 | 0.8859 | 0.1633 | 2600000 | 0.0168 | 3027.4567 | Weaker depth at conf90 improves LPIPS slightly but not PSNR |
| `da3_xfeat_mask_mcmc_cap2600_pose_lr1e7_dense_w0005_conf85` | 27.3694 | 0.8859 | 0.1623 | 2600000 | 0.0166 | 3126.4952 | Very-low-LR pose does not beat the fixed-camera weak-depth target |
| `da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf85` | 27.3245 | 0.8851 | 0.1664 | 2600000 | 0.0171 | 3076.6383 | Weight 0.0025 under-regularizes relative to w0005 |
| `da3_xfeat_mask_mcmc_cap3000_dense_w0005_conf85` | 27.3103 | 0.8845 | 0.1676 | 3000000 | 0.0181 | 3344.0500 | More capacity hurts with the weak-depth setting too |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70` | 27.3767 | 0.8852 | 0.1686 | 2600000 | 0.0163 | 3110.0417 | Previous no-pose target |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf85` | 27.3767 | 0.8856 | 0.1678 | 2600000 | 0.0169 | 3197.1380 | Raising confidence to 85 helps LPIPS but not PSNR unless depth weight is also reduced |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf70` | 27.3500 | 0.8852 | 0.1677 | 2600000 | 0.0170 | 3131.9398 | Weaker depth weight helps LPIPS, but conf85 is better |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_opacity001` | 27.1043 | 0.8808 | 0.1686 | 2600000 | 0.0123 | 2204.3540 | Opacity regularization speeds/render-cost profile but hurts PSNR/SSIM |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_bilateral` | 26.6432 | 0.8686 | 0.1698 | 2600000 | 0.0151 | 4094.9764 | Main metrics drop; color-corrected eval was 27.4238/0.8818/0.1580 |
| `da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_app` | 24.8177 | 0.8329 | 0.1834 | 2600000 | 0.0315 | 4583.1598 | Appearance optimization overfits train embeddings and hurts held-out views |
| `da3_xfeat_mask_2400k_mcmc_cap3000_dense_depthreg` | 27.2551 | 0.8827 | 0.1838 | 3000000 | 0.0182 | 3621.8573 | Larger 2.4M initialization plus 3.0M MCMC cap; slight gain over prior cap2600, but below weak-depth targets |
| `da3_xfeat_mask_mcmc_cap2600_dense_depthreg` | 27.2528 | 0.8825 | 0.1854 | 2600000 | 0.0168 | 3354.4538 | Previous DA3/VGGTX PSNR and SSIM target |
| `da3_xfeat_mask_mcmc_cap3000_dense_depthreg` | 27.1833 | 0.8812 | 0.1884 | 3000000 | 0.0181 | 3662.0717 | More MCMC capacity alone did not help |
| `da3_xfeat_mask_mcmc_cap2600_refine30k_dense_depthreg` | 27.1361 | 0.8804 | 0.1941 | 2600000 | 0.0207 | 3395.3490 | Extending MCMC refinement to 30k hurt all metrics |
| `da3_xfeat_mask_mcmc_dense_depthreg` | 27.2221 | 0.8821 | 0.1855 | 2200000 | 0.0160 | 3034.9239 | Previous 2.2M MCMC target |
| `da3_xfeat_mask_mcmc_pose_dense_depthreg` | 26.7115 | 0.8772 | 0.1536 | 2200000 | 0.0146 | 2865.3260 | Best DA3/VGGTX LPIPS so far |
| `da3_xfeat_mask_mcmc_pose_lr3e7_dense_depthreg` | 27.0781 | 0.8801 | 0.1863 | 2200000 | 0.0166 | 3179.5074 | Lower pose LR recovers PSNR/SSIM versus default pose, but not LPIPS |
| `da3_xfeat_mask_mcmc_pose_lr1e7_dense_depthreg` | 27.1980 | 0.8819 | 0.1817 | 2200000 | 0.0167 | 3070.6198 | Best balanced low-LR pose sweep |

Interpretation:

- MCMC is useful when DA3 cameras stay fixed: it improves PSNR from `27.1033` to `27.2221` and SSIM from `0.8797` to `0.8821`, while LPIPS is essentially flat/slightly worse.
- Increasing the no-pose MCMC cap from `2.2M` to `2.6M` gives a small PSNR/SSIM gain, reaching `27.2528`/`0.8825`.
- Weakening dense DA3 depth supervision from weight `0.02`/confidence percentile `50` to weight `0.01`/confidence percentile `70` improved the no-pose target to `27.3767`/`0.8852`/`0.1686`.
- Weakening it further to weight `0.005` and confidence percentile `85` improved the no-pose target to `27.4335`/`0.8864`/`0.1616`.
- Combining that weak-depth setting with SH degree 4 is the strongest no-pose component combination so far, reaching `27.4575`/`0.8868`/`0.1586`; it is now promoted to `configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4.yaml`.
- Reducing `ssim-lambda` to `0.1` nudges PSNR to `27.4601`, but lowers SSIM to `0.8854` and LPIPS to `0.1723`; promote it only as a PSNR-only target, not as the balanced target.
- Increasing batch size to `2` is a perceptual trade-off: `batch2` improves no-pose LPIPS to `0.1555`, but lowers PSNR/SSIM to `27.4336`/`0.8850` and nearly doubles training time.
- Combining `batch-size=2` with `scales-lr=0.01` gives the best no-pose LPIPS so far at `0.1549`, close to the pose-optimized LPIPS `0.1536`, but lowers PSNR/SSIM to `27.3809`/`0.8835`; promote it only as a no-pose perceptual target.
- Raising `scales-lr` to `0.01` without batch size 2 does not improve the final metrics (`27.3842`/`0.8850`/`0.1629`).
- Increasing the cap to `3.0M`, extending MCMC refinement through 30k, or rebuilding the initialization with `2.4M` XFeat-mask points did not beat the `2.6M + w0005/conf85` target. The 2.4M initialization is close, but costs more Gaussians and training time.
- Increasing the cap to `3.0M` still hurts under weak depth (`27.3103`/`0.8845`/`0.1676`), so the current target should stay at `2.6M`.
- Raising dense confidence to `90`, weakening depth further to `0.0025`, or adding very-low-LR pose optimization did not beat the fixed-camera `w0005/conf85_sh4` target.
- Visible Adam improves speed substantially (`2305.2455` s training, `0.0145` s/image render) but lowers quality to `27.4007`/`0.8852`/`0.1639`.
- Turning on SH degree 4 earlier with `--sh-degree-interval 500` does not help the final metrics (`27.4306`/`0.8861`/`0.1621`).
- Appearance optimization is a negative component here: it falls to `24.8177`/`0.8329`/`0.1834`, consistent with train-image appearance embeddings not transferring cleanly to held-out views.
- Bilateral grid also hurts the main validation metrics, though its color-corrected side metrics are strong (`27.4238`/`0.8818`/`0.1580`); keep it as a diagnostic, not the default target metric.
- Opacity regularization improves speed but gives back too much PSNR/SSIM.
- Low-LR gsplat pose optimization is not the best default if the target is PSNR/SSIM, but it is a strong perceptual component: LPIPS improves from `0.1855` to `0.1536`.
- Very low pose learning rates (`3e-7` and `1e-7`) recover most PSNR/SSIM lost by the default pose-optimized run. The `1e-7` run is the best balanced pose sweep at `27.1980`/`0.8819`/`0.1817`, but it does not beat the default pose run's LPIPS `0.1536`.
- The current recommended metric target is therefore split: use `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01` only for PSNR-only reporting, `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` for balanced no-pose PSNR/SSIM/LPIPS reporting, `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01` for no-pose LPIPS/perceptual reporting, and `da3_xfeat_mask_mcmc_pose_dense_depthreg` when prioritizing the absolute best perceptual LPIPS.
- The targets branch keeps only stable reproduction configs in `configs/`. Older intermediate configs such as the original 2.2M MCMC target, cap-2.6M dense-depth target, `w001/conf70`, and pre-SH4 `w0005/conf85` target were retired because stronger configs superseded them; their positive and negative evidence remains in this report and `reports/summary.*`.
- These methods still remain below COLMAP on this COLMAP-friendly scene. The best PSNR-only no-pose gap is PSNR `7.0832` relative to `colmap_gs_fps24_conf96`; the balanced target gap remains PSNR `7.0858`, SSIM `0.0732`, LPIPS `0.0773`; the best no-pose LPIPS gap is now `0.0736`.

## 2026-05-31 MCMC Cap And Pose LR Sweep

The sweep reused the existing fixed-camera XFeat-mask dataset:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_xfeat_mask_dense_depthreg/dataset
```

Smoke outputs were written under each method's own `smoke_gsplat` directory and were deleted only after explicit `realpath` checks confirmed they were inside:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Commands used the same 30k training budget:

The cap-2.6M dense-depth config in this historical command was retired from tracked `configs/` after weak-depth targets superseded it; the result remains part of the ablation evidence.

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_depthreg.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_depthreg/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

The two pose-LR sweeps used one-off configs outside git under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/sweep_configs/
```

Artifacts:

```text
da3_xfeat_mask_mcmc_cap2600_dense_depthreg/gsplat/
da3_xfeat_mask_mcmc_pose_lr3e7_dense_depthreg/gsplat/
da3_xfeat_mask_mcmc_pose_lr1e7_dense_depthreg/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_depthreg_train_20260531.log
logs/da3_xfeat_mask_mcmc_pose_lr3e7_dense_depthreg_train_20260531.log
logs/da3_xfeat_mask_mcmc_pose_lr1e7_dense_depthreg_train_20260531.log
```

Rejected smoke:

| Method | Outcome | Reason |
|---|---|---|
| `da3_xfeat_mask_mcmc_both_depthreg` | failed at step 0 | gsplat sparse+dense depth mode hit a tensor-rank mismatch in the sparse depth path: `input.dim() = 2` versus `len(dims) = 4` |

## 2026-05-31 Depth Weight, Capacity, And Initialization Sweep

The sweep reused the same fixed-camera XFeat-mask dataset for three training-only variants and exported one larger initialization for the `2400k` variant:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_xfeat_mask_dense_depthreg/dataset
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_xfeat_mask_2400k_dense_depthreg/dataset
```

The then-promoted command was retired from tracked `configs/` after stronger weak-depth and SH4 targets superseded it:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

The one-off sweep configs were kept outside git under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/sweep_configs/
```

Additional artifacts:

```text
da3_xfeat_mask_mcmc_cap3000_dense_depthreg/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70/gsplat/
da3_xfeat_mask_mcmc_cap2600_refine30k_dense_depthreg/gsplat/
da3_xfeat_mask_2400k_mcmc_cap3000_dense_depthreg/gsplat/
logs/da3_xfeat_mask_mcmc_cap3000_dense_depthreg_train_20260531.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_train_20260531.log
logs/da3_xfeat_mask_mcmc_cap2600_refine30k_dense_depthreg_train_20260531.log
logs/da3_xfeat_mask_2400k_mcmc_cap3000_dense_depthreg_train_20260531.log
```

All four variants passed 1000-step smoke tests before full training. Their `smoke_gsplat` outputs were deleted only after explicit path checks confirmed they were inside this experiment root.

## 2026-05-31 Weak Depth And gsplat Component Sweep

This sweep kept the same fixed DA3 cameras, XFeat mask initialization, MCMC cap `2.6M`, and 30k training budget. It tested whether the best `w001/conf70` setting still over-constrained training, and whether additional gsplat components should become part of the default target.

This once-promoted command was retired from tracked `configs/` after the SH4 target superseded it:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

The one-off configs remained outside git under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/sweep_configs/
```

Additional artifacts:

```text
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf70/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf85/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_app/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_bilateral/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_sh4/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_opacity001/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf70_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf85_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_app_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_bilateral_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_sh4_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_opacity001_train_20260601.log
```

Smoke tests also covered `nodepth` and `antialias`; both were rejected before full training. `nodepth` lagged the best smoke setting, and `antialias` dropped to `23.4689`/`0.8288`/`0.3700` at 1000 steps.

All smoke-test directories created in this sweep were deleted only after explicit path checks confirmed they were inside:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

## 2026-05-31 SH4, Capacity, Confidence, And Pose Sweep

This sweep reused the fixed-camera XFeat-mask dataset and started from the best weak-depth setting, `weight=0.005` and `conf_percentile=85`. It tested whether SH degree 4 should be combined with that setting, whether more MCMC capacity still helps, whether the dense confidence mask should be tightened, and whether a very-low-LR pose variant can beat the fixed-camera result.

The promoted command is:

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

The one-off sweep configs remained outside git under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/sweep_configs/
```

Additional artifacts:

```text
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat/
da3_xfeat_mask_mcmc_cap3000_dense_w0005_conf85/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf85/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf90/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf90/gsplat/
da3_xfeat_mask_mcmc_cap2600_pose_lr1e7_dense_w0005_conf85/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap3000_dense_w0005_conf85_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf85_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf90_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w00025_conf90_train_20260601.log
logs/da3_xfeat_mask_mcmc_cap2600_pose_lr1e7_dense_w0005_conf85_train_20260601.log
```

All six variants passed 1000-step smoke tests before full training. Their `smoke_gsplat` outputs were deleted only after explicit `realpath` checks confirmed they were inside:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

The SH4 combination is the new best fixed-camera target at PSNR `27.4575`, SSIM `0.8868`, and LPIPS `0.1586`. Increasing the cap to `3.0M`, reducing dense depth weight to `0.0025`, raising dense confidence to `90`, and adding very-low-LR pose optimization did not beat it.

## 2026-05-31 SH4 Loss And Optimizer Component Sweep

This sweep kept the same fixed-camera XFeat-mask dataset, weak dense depth setting, MCMC `2.6M` cap, and SH degree 4. It tested whether loss weighting, MCMC noise, optimizer variants, random background, and earlier SH activation should be added on top of the balanced SH4 target.

Full 30k commands used one-off configs from:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/sweep_configs/
```

The PSNR-only promoted command is:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

Smoke results at 1000 steps:

| Method | PSNR | SSIM | LPIPS | Outcome |
|---|---:|---:|---:|---|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_visible_adam` | 24.0013 | 0.8397 | 0.3451 | Promising smoke; full run is faster but lower quality |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01` | 23.8661 | 0.8284 | 0.3615 | Promoted to full for PSNR trade-off |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_shint500` | 23.6075 | 0.8330 | 0.3497 | Promoted to full; no final gain |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_noise250k` | 23.5423 | 0.8316 | 0.3523 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_noise1m` | 23.5149 | 0.8316 | 0.3530 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim03` | 23.1846 | 0.8309 | 0.3570 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_random_bkgd` | 23.3393 | 0.8279 | 0.3803 | Rejected after smoke |

Rejected smoke tests:

| Method | Outcome | Reason |
|---|---|---|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_sparse_visible_adam` | failed before eval | `sparse_grad` requires `packed=True` |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_packed_sparse_visible_adam` | failed before eval | after adding `--packed`, `SparseAdam.step(visibility_mask)` hit `TypeError: 'Tensor' object is not callable` |

Full-run artifacts:

```text
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_visible_adam/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_shint500/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_visible_adam_train_20260531_090003_sh4_component_full.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_shint500_train_20260531_090003_sh4_component_full.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim01_train_20260531_090003_sh4_component_full.log
```

All smoke outputs created for this sweep were deleted only after `realpath` confirmed that each path ended in this experiment's own `smoke_gsplat` directory under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Conclusion: `--ssim-lambda 0.1` gives the current best no-pose PSNR at `27.4601`, but it hurts SSIM/LPIPS enough that the balanced target remains `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4`. Visible Adam is useful only when speed matters, and sparse-gradient visible Adam is not compatible with this patched trainer stack.

## 2026-05-31 SH4 Batch Size And LR Component Sweep

This sweep kept the same fixed-camera XFeat-mask dataset, weak dense depth setting, MCMC `2.6M` cap, and SH degree 4. It tested training batch size, means/scales learning rates, scale regularization, and the 3DGUT `with_ut` component.

The no-pose LPIPS/perceptual promoted command is:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01.yaml \
  --data-dir "$RUN/da3_xfeat_mask_dense_depthreg/dataset" \
  --result-dir "$RUN/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

Smoke results at 1000 steps:

| Method | PSNR | SSIM | LPIPS | Outcome |
|---|---:|---:|---:|---|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01` | 25.0411 | 0.8528 | 0.3019 | Strongest smoke; promoted to full |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2` | 24.5978 | 0.8470 | 0.3164 | Strong smoke; promoted to full |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scaleslr01` | 24.2174 | 0.8422 | 0.3338 | Promoted to full; no final gain |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_ssim015` | 23.7196 | 0.8308 | 0.3545 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_meanslr24e5` | 23.5479 | 0.8319 | 0.3540 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_meanslr8e5` | 23.5326 | 0.8311 | 0.3538 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scalereg005` | 23.5226 | 0.8316 | 0.3534 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_with_ut_eval3d` | 23.4944 | 0.8301 | 0.3798 | Rejected after smoke |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scaleslr0025` | 22.2821 | 0.7925 | 0.4279 | Rejected after smoke |

Rejected setup smoke:

| Method | Outcome | Reason |
|---|---|---|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_with_ut` | failed before eval | trainer asserts that `--with-ut` requires `--with-eval3d` |

Full-run artifacts:

```text
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scaleslr01/gsplat/
da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01/gsplat/
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_train_20260531_101956_sh4_batch_scale_full.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_scaleslr01_train_20260531_101956_sh4_batch_scale_full.log
logs/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_batch2_scaleslr01_train_20260531_102357_sh4_batch_scale_combo_full.log
```

All smoke outputs created for this sweep were deleted only after `realpath` confirmed that each path ended in this experiment's own `smoke_gsplat` directory under:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Conclusion: batch size 2 is not a new PSNR/SSIM target, but it is a useful no-pose perceptual component. The `batch2 + scales-lr=0.01` combination reaches LPIPS `0.1549`, the best fixed-camera no-pose LPIPS so far, while keeping DA3 cameras unchanged.
