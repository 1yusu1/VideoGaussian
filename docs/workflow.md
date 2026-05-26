# Workflow

VideoGaussian's current reproducible experiment paths are:

```text
video -> frames -> COLMAP/pycolmap -> gsplat dataset -> gsplat -> metrics/report
video -> Depth Anything 3 -> DA3 COLMAP-style export -> gsplat dataset -> gsplat -> metrics/report
video -> Depth Anything 3 -> XFeat DA3 global alignment -> gsplat dataset -> gsplat/MCMC -> metrics/report
```

DA3 should export at least:

```text
<RUN_DIR>/da3_output/
  input_images/
  cameras.bin
  images.bin
  points3D.bin
```

The bridge script creates the dataset layout expected by gsplat and, when available, copies dense DA3 depth supervision:

```text
<RUN_DIR>/gsplat_dataset/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  dense_depth.npz
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

The final report is generated with:

```bash
bash scripts/make_report.sh \
  --scene <scene> \
  --metrics-root <runs_dir> \
  --output-dir reports
```

VGGT-Omega is not part of the active workflow because its checkpoint was unavailable for the completed study.
