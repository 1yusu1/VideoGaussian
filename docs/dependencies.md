# Dependencies

VideoGaussian depends on three external repositories. They are intentionally not vendored into this repository.

## External Repositories

Recommended server layout:

```text
<WORKSPACE>/Depth-Anything-3
<WORKSPACE>/gsplat-1.5.3
<WORKSPACE>/spark
<WORKSPACE>/VideoGaussian
```

### Depth Anything 3

Used for video/image geometry prediction and DA3 exports.

Expected capabilities:

- `da3 video`
- `da3 auto`
- `exports/mini_npz/results.npz`
- COLMAP-style export files such as `cameras.bin`, `images.bin`, and `points3D.bin`

Install this environment following the upstream Depth Anything 3 instructions.

### gsplat 1.5.3

Used for Gaussian Splatting training and checkpoint export.

VideoGaussian currently assumes access to:

- `examples/simple_trainer.py`
- `examples/datasets/colmap.py`
- `gsplat.export_splats`

Install the gsplat package and the dependencies from its `examples/requirements.txt` in the remote training environment.

### Spark

Used only for browser-based preview of exported splat files.

The local viewer uses:

- Spark CDN/import map from `viewer/report-viewer.html`
- Node.js static server from `scripts/serve_viewer.js`

If working offline or modifying Spark itself, build the upstream Spark repo separately.

## VideoGaussian Python Requirements

`requirements.txt` intentionally does not install Depth Anything 3 or gsplat. Those projects should be installed following their upstream instructions in the training environment.

The scripts in this repo rely on:

- Python standard library for dataset preparation.
- `pyyaml` for reading `--config` YAML files.
- `torch` and `gsplat` from the gsplat environment for checkpoint merging.
- `da3` from the Depth Anything 3 environment for video inference.

Node.js is only needed for `scripts/serve_viewer.js`.
