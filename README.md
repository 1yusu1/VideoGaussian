# VideoGaussian

VideoGaussian is a thin pipeline repo for training Gaussian Splats from a video.

It orchestrates three external projects:

- Depth Anything 3: video geometry and COLMAP-style export.
- gsplat 1.5.3: Gaussian Splatting training.
- Spark: browser preview for the final `.ply` or `.spz`.

The upstream source trees are dependencies, not vendored code. Keep them cloned and installed on the server, but do not copy them into this repo.

## Required Setup

Prepare a config from the template:

```bash
cp configs/pipeline.example.yaml configs/my_scene.yaml
```

Edit these fields in `configs/my_scene.yaml`:

```yaml
paths:
  video: <input_video>
  depth_anything_repo: <Depth-Anything-3_repo>
  gsplat_repo: <gsplat-1.5.3_repo>
  runs_dir: <output_root>
  model_dir: <local_DA3_model_dir>

gsplat:
  cuda_visible_devices: "0,1"
  max_steps: 30000
```

Required environment assumptions:

- `da3 video` works inside the current environment.
- `$GSPLAT_REPO/examples/simple_trainer.py` exists.
- `python -c "from gsplat import export_splats"` works.
- Node.js is available if you want to use the local Spark viewer.

## Test The Pipeline

Run a short smoke test first. This uses 100 gsplat steps and should finish with a merged PLY.

Single GPU:

```bash
cd <VideoGaussian_repo>

bash scripts/run_video_to_gaussian.sh \
  --config configs/my_scene.yaml \
  --gpus 0 \
  --test-mode
```

Two GPUs:

```bash
bash scripts/run_video_to_gaussian.sh \
  --config configs/my_scene.yaml \
  --test-mode
```

Command-line options override the YAML config, so the single-GPU test above can reuse the same config and temporarily override `gsplat.cuda_visible_devices`.

Expected output:

```text
<RUNS_DIR>/<scene>_test_<timestamp>/
  da3_output/
  gsplat_dataset/
  gsplat_result/
  <scene>_test_final.ply
  pipeline.log
```

The script disables gsplat's live viewer by default so it can finish and merge checkpoints. Add `--enable-viewer` only when you want to inspect training live; that mode keeps the gsplat process open after training.

## Full Training

After the smoke test succeeds, run full training:

```bash
bash scripts/run_video_to_gaussian.sh \
  --config configs/my_scene.yaml
```

Important parameters:

- `da3.fps`: frame sampling rate for DA3. Lower values reduce frames and training cost.
- `da3.conf_thresh_percentile`: confidence filter for DA3 COLMAP export. Higher values reduce `points3D.bin` and gsplat initialization size.
- `gsplat.cuda_visible_devices`: visible GPU ids for gsplat training.
- `gsplat.max_steps`: gsplat training iterations.
- `--save-ply-during-train`: disabled by default. In multi-GPU training, gsplat keeps a rank-local shard of the Gaussians on each GPU, so the built-in training-time PLY export writes only that rank's subset. We still need a post-processing merge step, so this pipeline uses `scripts/merge_gsplat_checkpoints.py` after training.

## View Results

Use the Spark viewer:

```bash
cd <VideoGaussian_repo>
node scripts/serve_viewer.js
```

Open:

```text
http://localhost:8090/
```

Load the final output:

```text
<RUNS_DIR>/<scene>_<timestamp>/<scene>_final.ply
```

If you are viewing from your local machine through SSH, forward the viewer port:

```bash
ssh -L 8090:localhost:8090 <user>@<server>
```

## Repository Contents

```text
scripts/run_video_to_gaussian.sh       # video -> DA3 -> gsplat -> merged PLY
scripts/prepare_gsplat_dataset.py      # DA3 output -> gsplat COLMAP dataset layout
scripts/merge_gsplat_checkpoints.py    # multi-rank ckpts -> final PLY
scripts/config_to_env.py               # YAML config -> shell variables for the main script
scripts/serve_viewer.js                # local static server for the Spark viewer
viewer/report-viewer.html              # copied viewer page
docs/                                 # dependency and workflow notes
configs/pipeline.example.yaml          # editable config template loaded by --config
```

## Notes

- Generated outputs, videos, checkpoints, model weights, and splats are ignored by git.
- The current pipeline uses DA3 COLMAP-style export, not dense DA3 depth supervision.
- See `docs/dependencies.md` for the external repo boundary.
