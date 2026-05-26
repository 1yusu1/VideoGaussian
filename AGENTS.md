# VideoGaussian Agent Notes

## Project Goal

VideoGaussian is an experiment framework for geometry-prior video-to-3D Gaussian Splatting reconstruction from casual videos.

Primary comparison methods:

- `colmap_gs`: COLMAP sparse SfM + gsplat.
- `da3_gs`: Depth Anything 3 COLMAP-style geometry + gsplat.
- `da3_gs_depthreg`: DA3 geometry + gsplat depth regularization.
- `da3_ga_xfeat_v2_gs`: VGGT-X-inspired XFeat global alignment + gsplat.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg`: XFeat GA + gsplat MCMC, pose optimization, and dense DA3 depth regularization.

## Remote Server Facts

- SSH: `panshihan@10.184.17.164 -p 45079`
- Node: `node02`
- GPUs: 8 x NVIDIA RTX 3090, 24GB each.
- Conda env for experiments: `project_310`
- Large outputs should go under `/data1`, not the home directory.
- Suggested run root: `/data1/panshihan/videogaussian_runs`
- Video for the current experiment:
  `/home/panshihan/DepthAnything3/videos/liminal_pool.mp4`
- gsplat repo/examples:
  `/home/panshihan/3DGS/gsplat-1.5.3/examples`
- Depth Anything 3 repo:
  `/home/panshihan/DepthAnything3`
- COLMAP is not currently available on PATH.
- XFeat repo:
  `/home/panshihan/accelerated_features`

## Current Liminal Pool Experiment

Scene name:

```text
liminal_pool_fps24_conf96
```

Recommended output root:

```text
/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
```

Note: this directory name is historical. Even though it contains `fps24`, the intended current protocol is fps12. Reuse this directory for the fps12 comparison instead of creating a separate `liminal_pool_fps12` directory.
A 2026-05-26 dry-run found 181 DA3 images under this directory, which matches the fps12 protocol for the 15-second source video.

DA3 settings:

- `fps=12`
- `conf_thresh_percentile=96`
- Use a local `--model-dir` path. Do not rely on the default Hugging Face model id.
- `--num-max-points` is a DA3 GLB/export visualization point-cloud cap. It should not be assumed to limit the COLMAP `points3D.bin` used by gsplat initialization.

The source video is about 15 seconds, 2560x1440, average ~24 fps, 361 frames. `fps=12` samples roughly half of the frames and is the current comparison protocol.

## Standard Remote Setup

```bash
cd ~/VideoGaussian
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project_310
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export VIDEO=/home/panshihan/DepthAnything3/videos/liminal_pool.mp4
export RUN=/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
export GSPLAT_EXAMPLES=/home/panshihan/3DGS/gsplat-1.5.3/examples
export DA3_MODEL_DIR=/path/to/local/DA3NESTED-GIANT-LARGE-1.1
```

## DA3 Command

```bash
bash scripts/run_da3_pipeline.sh \
  --config configs/da3_gs.yaml \
  --input "$VIDEO" \
  --output-dir "$RUN/da3" \
  --repo-dir /home/panshihan/DepthAnything3 \
  --model-dir "$DA3_MODEL_DIR" \
  --fps 12 \
  --conf-thresh-percentile 96
```

If DA3 runs out of memory on a 3090, first reduce `--process-res` to `384` or `336`.

## gsplat Dataset Preparation

```bash
python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3" \
  --dataset-dir "$RUN/da3_gs/dataset" \
  --overwrite
```

Before training, inspect initialization size:

```bash
ls -lh "$RUN/da3/points3D.bin" "$RUN/da3_gs/dataset/sparse/0/points3D.bin"
find "$RUN/da3_gs/dataset/images" -type f | wc -l
```

## gsplat Training

When using `python -m videogaus.gaussian.train_gsplat`, GPU selection is controlled by the config field:

```yaml
gsplat:
  cuda_visible_devices: "0,1,2,3,4,5,6,7"
```

The wrapper intentionally reads this config and sets `CUDA_VISIBLE_DEVICES` for the child `simple_trainer.py` process. Therefore, if the config says `"0"`, a shell prefix such as `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m videogaus.gaussian.train_gsplat ...` can still end up running single-GPU. Change the YAML config, or use a dedicated config copy, when selecting GPUs through the wrapper.

Use all 8 GPUs when a single 3090 OOMs:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_gs.yaml \
  --data-dir "$RUN/da3_gs/dataset" \
  --result-dir "$RUN/da3_gs/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

The wrapper passes `--disable-video`, so training should not render videos.

For smoke tests, use `--iterations 1000 --eval-steps 1000 --save-steps 1000`, then delete the smoke result directory after success.

## DA3 Depth Regularization Notes

DA3's advantage should be framed as providing fast neural geometry priors, not as guaranteeing better optimized 3DGS metrics than COLMAP on easy high-overlap videos. COLMAP+3DGS relies on optimized multi-view consistency; DA3+3DGS depends on the global consistency of predicted geometry. It is expected that COLMAP can outperform DA3 on COLMAP-friendly scenes.

To use DA3 geometry during training, run `configs/da3_gs_depthreg.yaml`, which now enables dense DA3 depth supervision by default:

```yaml
gsplat:
  depth_regularization:
    enabled: true
    type: dense
    dense:
      enabled: true
      weight: 0.02
      conf_percentile: 50
      conf_weighted: true
```

`prepare_gsplat_dataset.py` automatically converts DA3 `exports/mini_npz/results.npz` into `dataset/dense_depth.npz` with `depth`, optional `conf`, and `image_names`. The server gsplat 1.5.3 example has been patched with:

- `--dense-depth-loss`
- `--dense-depth-lambda`
- `--dense-depth-conf-percentile`
- `--dense-depth-conf-weighted`

The dense loss compares rendered expected depth against DA3 dense depth in disparity space, after downsampling the rendered depth to the DA3 depth resolution, and can confidence-mask/weight pixels.
On 2026-05-26, the server gsplat `simple_trainer.py` dense loss was further patched to avoid NaNs with MCMC: it now applies the valid depth mask before reciprocal disparity conversion and clamps valid depths to `1e-6`, instead of using `torch.where(pred > 0, 1 / pred, 0)`.

The older sparse projected-point depth loss still exists for `type: sparse` or `type: both`. Some DA3/COLMAP-style models have images with no visible sparse point tracks; the server gsplat copy has been patched so `datasets/colmap.py` returns empty sparse depth samples for those images and `simple_trainer.py` skips sparse depth loss for empty samples instead of raising `KeyError` or producing NaNs.

Before dense depth training, ensure the dataset has `dense_depth.npz`:

```bash
python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3" \
  --dataset-dir "$RUN/da3_gs/dataset" \
  --overwrite
```

Recommended smoke test:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_gs_depthreg.yaml \
  --data-dir "$RUN/da3_gs/dataset" \
  --result-dir "$RUN/da3_gs_dense_depthreg/smoke_gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 1000 \
  --eval-steps 1000 \
  --save-steps 1000
```

If the smoke test passes, run the full dense-depth-regularized DA3 experiment into `$RUN/da3_gs_dense_depthreg/gsplat`.

## DA3 Global Alignment Plan

Local VGGT-X repo inspected at:

```text
D:\VGGT-X
```

VGGT-X implements global alignment mainly in `utils/opt.py` and calls it from `demo_colmap.py --use_ga`.

Important details:

- Candidate image pairs are selected by relative viewing angle, default threshold 30 degrees.
- Matches are extracted with XFeat, with `max_query_pts=4096` for fewer than 500 images and 2048 otherwise.
- Initial epipolar errors are computed from predicted cameras and XFeat correspondences.
- Correspondence weights are derived from the epipolar-error histogram/PDF, then square-rooted/normalized during optimization.
- Learning rate is adaptive from median epipolar error:
  - `<2.5`: `5e-4`
  - `2.5-7.5`: `1e-3`
  - `>7.5`: `1e-2`
  - end LR is base / 10.
- It optimizes camera parameters first, then re-unprojects predicted depth with refined cameras and exports a new COLMAP sparse model. Do not only edit cameras while keeping old predicted points.

VideoGaussian DA3 adaptation supports two matcher backends:

- `--matcher opencv`: local SIFT/ORB fallback, no network dependency.
- `--matcher xfeat`: VGGT-X-style XFeat matching. Prefer passing a local `verlab/accelerated_features` checkout through `--xfeat-repo-dir`; otherwise PyTorch Hub may try to download it.
- `--matcher auto`: try XFeat first, then fall back to OpenCV.

For the next comparison, do not continue the fps2 run. Reuse the historical fps12 run root:

```bash
export RUN=/data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96
export XFEAT_REPO=/path/to/accelerated_features
```

The directory name says `fps24`, but the intended protocol is fps12. Treat it as the existing fps12 workspace.

Install XFeat through a GitHub mirror, not direct GitHub, then always pass the local checkout path:

```bash
cd /data1/panshihan

# Preferred clone mirror.
git clone https://gitclone.com/github.com/verlab/accelerated_features.git accelerated_features

# Fallback proxy form if gitclone.com is unavailable.
# git clone https://gh-proxy.com/https://github.com/verlab/accelerated_features.git accelerated_features

export XFEAT_REPO=/data1/panshihan/accelerated_features
```

If the mirror rewrites GitHub paths instead of proxying the full URL, clone with that mirror's native form, but keep the final checkout path as `$XFEAT_REPO`. The important part is that `run_da3_global_align.sh` receives `--xfeat-repo-dir "$XFEAT_REPO"` so the local XFeat code and `weights/xfeat.pt` are used without PyTorch Hub network access.

Run DA3 at fps12:

```bash
bash scripts/run_da3_pipeline.sh \
  --config configs/da3_gs.yaml \
  --input "$VIDEO" \
  --output-dir "$RUN/da3" \
  --repo-dir /home/panshihan/DepthAnything3 \
  --model-dir "$DA3_MODEL_DIR" \
  --fps 12 \
  --conf-thresh-percentile 96
```

Then run DA3 global alignment with XFeat:

```bash
bash scripts/run_da3_global_align.sh \
  --source-dir "$RUN/da3" \
  --output-dir "$RUN/da3_ga_xfeat" \
  --matcher xfeat \
  --xfeat-repo-dir "$XFEAT_REPO" \
  --angle-threshold 30 \
  --temporal-window 10 \
  --max-keypoints 4096 \
  --max-matches-per-pair 2048 \
  --match-batch-size 16 \
  --niter 300 \
  --conf-percentile 96 \
  --conf-percentile-mode per-frame \
  --max-points 1000000
```

The current implementation is `src/videogaus/geometry/align_cameras_epipolar.py`. It optimizes DA3 extrinsics under weighted epipolar loss with pose regularization, rebuilds points from DA3 dense depth/confidence, and writes a refined COLMAP-style model under the requested output directory.
For high confidence thresholds such as 96, use `--conf-percentile-mode per-frame`. A global percentile can leave some frames with zero retained points, and pycolmap refuses to write registered images with empty `points2D`.

Then prepare/train as usual:

```bash
python -m videogaus.geometry.prepare_gsplat_dataset \
  --source-dir "$RUN/da3_ga_xfeat" \
  --dense-depth-path "$RUN/da3/exports/mini_npz/results.npz" \
  --dataset-dir "$RUN/da3_ga_xfeat_gs/dataset" \
  --overwrite

python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_gs.yaml \
  --data-dir "$RUN/da3_ga_xfeat_gs/dataset" \
  --result-dir "$RUN/da3_ga_xfeat_gs/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

For the depth-regularized variant, reuse the same dataset and switch only config/result directory:

```bash
python -m videogaus.gaussian.train_gsplat \
  --config configs/da3_gs_depthreg.yaml \
  --data-dir "$RUN/da3_ga_xfeat_gs/dataset" \
  --result-dir "$RUN/da3_ga_xfeat_gs_dense_depthreg/gsplat" \
  --gsplat-examples-dir "$GSPLAT_EXAMPLES" \
  --iterations 30000 \
  --eval-steps 30000 \
  --save-steps 30000
```

## Current Liminal Pool Findings

Current on-disk metrics show that `liminal_pool` is COLMAP-friendly. COLMAP+GS remains the strongest baseline even at fps=2, while direct DA3 geometry initialization produces substantially worse NVS metrics.

fps12/conf96, stored in historical `liminal_pool_fps24_conf96` directory:

| Method | PSNR | SSIM | LPIPS | #GS |
|---|---:|---:|---:|---:|
| `colmap_gs` | 34.543 | 0.9600 | 0.081 | 1,745,683 |
| `da3_gs` | 26.473 | 0.8723 | 0.252 | 1,880,719 |
| `da3_gs_sparse_depthreg` | 26.637 | 0.8739 | 0.235 | 1,815,594 |
| `da3_gs_dense_depthreg` | 26.719 | 0.8746 | 0.232 | 1,497,448 |
| `da3_ga_xfeat_gs` | 21.064 | 0.8044 | 0.457 | 1,006,884 |
| `da3_ga_xfeat_v2_gs` | 22.756 | 0.8246 | 0.354 | 1,707,957 |
| `da3_ga_xfeat_v2_mcmc_pose_depthreg` | 21.747 | 0.7928 | 0.287 | 1,800,000 |
| `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001` | 22.732 | 0.8179 | 0.319 | 1,800,000 |
| `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001` | 22.955 | 0.8260 | 0.345 | 1,800,000 |
| `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005` | 22.668 | 0.8183 | 0.316 | 1,800,000 |
| `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85` | 22.775 | 0.8188 | 0.318 | 1,800,000 |
| `da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001` | 22.779 | 0.8234 | 0.355 | 1,800,000 |

DA3+XFeat GA manifest for this run:

- Images: 181.
- Candidate pairs: 1,755.
- XFeat matches: 2,655,834.
- Initial median epipolar error: 1.169 px.
- Final GA loss: 0.799.
- GA runtime: 186.8 s.

fps2:

| Method | PSNR | SSIM | LPIPS | #GS |
|---|---:|---:|---:|---:|
| `colmap_gs` | 26.018 | 0.8665 | 0.273 | 1,570,118 |
| `da3_gs_conf96` | 19.278 | 0.7265 | 0.406 | 888,395 |
| `da3_gs_conf70` | 17.981 | 0.7060 | 0.461 | 1,713,092 |
| `da3_gs_dense_depthreg_conf70` | 18.831 | 0.7162 | 0.474 | 1,443,703 |

Interpretation:

- DA3 dense/sparse depth regularization improves DA3 at fps12/conf96, with dense depthreg being the best DA3 variant.
- VGGT-X-style epipolar-only GA is a negative ablation on this scene. It is substantially worse than direct DA3 initialization, likely because it changes pose while keeping DA3 depth/intrinsics fixed and therefore disturbs DA3 camera-depth coupling.
- DA3 GA XFeat v2 recovers part of the epipolar-only GA loss, improving PSNR from 21.064 to 22.756 and LPIPS from 0.457 to 0.354, but it remains below direct DA3 and DA3 dense depth regularization.
- DA3 GA XFeat v2 with MCMC + pose optimization + dense depth regularization improves perceptual LPIPS further to 0.287, but PSNR/SSIM drop to 21.747/0.7928 and runtime increases, so the current settings over-regularize or move poses in a way that hurts pixel fidelity.
- Weakening pose/depth regularization recovers PSNR/SSIM: `lr1e6_w001` is the best GA+MCMC setting so far on PSNR/SSIM at 22.955/0.8260, while `lr3e6_w0005` keeps the best LPIPS among the weakened variants at 0.316.
- Raising dense-depth confidence to 85 (`lr3e6_w001_conf85`) only slightly changes the `lr3e6_w001` result and does not beat `lr1e6_w001`.
- Reducing GA v2 initialization to 500k points while keeping MCMC cap at 1.8M does not improve metrics; it lands near ordinary v2 and below `lr1e6_w001`, suggesting that simply leaving more add-room for MCMC is not enough.
- Lowering the DA3 confidence threshold at fps2 increases initialization point count but worsens NVS quality, indicating that many added points are noisy or globally inconsistent.
- Dense DA3 depth supervision improves fps2/conf70 PSNR and SSIM over DA3 initialization, but LPIPS remains worse and the gap to COLMAP is large.
- For this scene, DA3 geometry is useful as a regularizer but is not a drop-in replacement for COLMAP SfM when COLMAP works well.
- To demonstrate DA3's advantage, use COLMAP-degraded scenes or protocols: textureless/reflective/blurred videos, very short clips, low parallax, dynamic content, or train-only geometry with held-out evaluation.

## DA3 GA XFeat V2

The first XFeat GA result shows that pose-only epipolar alignment is not enough. V2 is implemented in `src/videogaus/geometry/align_cameras_epipolar.py` and should use `configs/da3_ga_xfeat_v2_gs.yaml` and `configs/da3_ga_xfeat_v2_mcmc_pose_depthreg.yaml`:

- Run GA with `--loss-mode epi3d`, `--lambda-3d 0.1`, `--niter 100`, `--lambda-pose-reg 0.01`, `--pose-rot-clamp 0.05`, `--pose-trans-clamp 0.05`.
- Keep `--conf-percentile 96 --conf-percentile-mode per-frame`.
- Enable `--use-match-mask` so DA3 conf96 points are unioned with a 3px-dilated high-weight XFeat match mask.
- Train one default gsplat run and one MCMC + pose optimization + dense depth regularization run.

Current v2 default gsplat result:

- `da3_ga_xfeat_v2_gs`: PSNR 22.756, SSIM 0.8246, LPIPS 0.354, render 0.009 s/image, #GS 1,707,957.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg`: PSNR 21.747, SSIM 0.7928, LPIPS 0.2869, render 0.016 s/image, training time 3000.9 s, #GS 1,800,000.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001`: PSNR 22.732, SSIM 0.8179, LPIPS 0.3186, render 0.0145 s/image, training time 2869.5 s, #GS 1,800,000.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001`: PSNR 22.955, SSIM 0.8260, LPIPS 0.3447, render 0.0152 s/image, training time 2941.6 s, #GS 1,800,000.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005`: PSNR 22.668, SSIM 0.8183, LPIPS 0.3159, render 0.0151 s/image, training time 2829.4 s, #GS 1,800,000.
- `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85`: PSNR 22.775, SSIM 0.8188, LPIPS 0.3182, render 0.0149 s/image, training time 2868.9 s, #GS 1,800,000.
- `da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001`: PSNR 22.779, SSIM 0.8234, LPIPS 0.3551, render 0.0154 s/image, training time 3088.3 s, #GS 1,800,000. GA exported 500,000 initial points, then MCMC grew to the 1.8M cap.

The historical MCMC + pose + dense sweep showed that weakening pose/depth regularization recovers PSNR/SSIM. The repository now keeps only one generic MCMC config, `configs/da3_ga_xfeat_v2_mcmc_pose_depthreg.yaml`, using the best PSNR/SSIM setting from that sweep: pose LR `1e-6`, pose reg `1e-5`, dense weight `0.01`, dense conf percentile `70`, and MCMC cap `1,800,000`.

Do not add one YAML file per ablation unless the ablation becomes a stable comparison method. For one-off sweeps, copy the generic config outside git or override fields from CLI/job scripts, then record the result in `reports/` and `AGENTS.md`.

GA parameters can now be read from config:

```bash
python -m videogaus.geometry.align_cameras_epipolar \
  --config configs/da3_ga_xfeat_v2_gs.yaml
```

CLI flags still override config fields for one-off parameter sweeps.

## Export gsplat Checkpoints to PLY

Use `scripts/merge_gsplat_checkpoints.py` to export trained gsplat checkpoints to a single PLY. This works for both single-GPU and multi-rank runs.

Single-GPU example:

```bash
cd ~/VideoGaussian
source ~/miniconda3/etc/profile.d/conda.sh
conda activate project_310
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/merge_gsplat_checkpoints.py \
  --ckpt "$RUN/colmap_gs/gsplat/ckpts/ckpt_29999_rank0.pt" \
  --output "$RUN/colmap_gs/gsplat/ply/final_rank0.ply"
```

Multi-GPU example:

```bash
python scripts/merge_gsplat_checkpoints.py \
  --ckpt "$RUN/colmap_gs/gsplat"/ckpts/ckpt_29999_rank*.pt \
  --output "$RUN/colmap_gs/gsplat/ply/final_merged.ply"
```

For the current COLMAP-vs-DA3 comparison run, use:

```bash
export RUN=/data1/panshihan/videogaussian_runs/liminal_pool_colmap_vs_da3

python scripts/merge_gsplat_checkpoints.py \
  --ckpt "$RUN/colmap_gs/gsplat/ckpts/ckpt_29999_rank0.pt" \
  --output "$RUN/colmap_gs/gsplat/ply/final_rank0.ply"
```

## OOM Response Order

1. Check for stale jobs with `nvidia-smi` and kill only processes that belong to this experiment.
2. Use all 8 GPUs via `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
3. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. Increase DA3 `--conf-thresh-percentile` to `97` or `98` and regenerate DA3 output if the COLMAP `points3D.bin` is too large.
5. Lower DA3 `--process-res` to `384` or `336` if DA3 inference OOMs.
6. As a last resort, sample fewer frames, e.g. `fps=12`.

Do not assume `--num-max-points` fixes gsplat initialization OOM; DA3 marks it as a GLB point-cloud export option.

## Deferred VGGT-Omega Work

VGGT-Omega was inspected but removed from the reproducible experiment path because the required checkpoint was unavailable. Keep VGGT-X/XFeat global-alignment notes above; those are part of the implemented DA3 GA v2 ablations. Do not advertise or run a VGGT-Omega pipeline unless the checkpoint and a validated adapter are restored.
