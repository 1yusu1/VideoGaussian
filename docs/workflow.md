# Workflow

VideoGaussian's current main path is:

```text
video
  -> Depth Anything 3 `da3 video`
  -> DA3 output directory
  -> gsplat-compatible COLMAP dataset
  -> gsplat `examples/simple_trainer.py`
  -> rank-local checkpoints
  -> merged PLY
  -> Spark viewer
```

DA3 should export at least:

```text
<RUN_DIR>/da3_output/
  input_images/
  cameras.bin
  images.bin
  points3D.bin
```

The bridge script creates the dataset layout expected by gsplat:

```text
<RUN_DIR>/gsplat_dataset/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

The training output is:

```text
<RUN_DIR>/gsplat_result/
  ckpts/
  renders/
  stats/
  vram_peak.log
```

The final merged result is:

```text
<RUN_DIR>/<scene>_final.ply
```

Dense DA3 depth supervision is not used in the current main path.
