# Third-Party Dependencies

`third_party/` is the boundary between VideoGaussian code and external research projects. The main repository should contain orchestration, adapters, configs, reports, and lightweight assets only. Do not vendor external source code by copying files into `src/videogaus`.

## Expected Dependencies

| Dependency | Purpose | Expected Path |
|---|---|---|
| Depth Anything 3 | Video depth, confidence, cameras, and COLMAP-style geometry export | `third_party/depth-anything-3` |
| gsplat | Gaussian Splatting training, rendering, dense depth supervision, and checkpoint export | `third_party/gsplat` |
| XFeat / accelerated_features | Feature matching for DA3 global alignment ablations | `third_party/accelerated_features` |

## Recommended Setup

The reproducible setup uses Git submodules:

```bash
git submodule update --init --recursive
```

Current submodule sources:

```text
third_party/depth-anything-3      https://github.com/ByteDance-Seed/depth-anything-3.git
third_party/gsplat                https://github.com/1yusu1/gsplat-videogaussian.git
third_party/accelerated_features  https://github.com/verlab/accelerated_features.git
```

`third_party/gsplat` intentionally points to the `videogaussian-depthreg` branch of the project fork. That branch records the experiment-specific hooks used by VideoGaussian:

- dense DA3 depth supervision through `dense_depth.npz`
- confidence thresholding/weighting for dense depth loss
- safe handling of images with no sparse projected point depths
- MCMC/pose optimization flags consumed by `src/videogaus/gaussian/train_gsplat.py`

Using public repositories as submodules normally does not require contacting maintainers, but every dependency still carries its own license, citation requirements, and model checkpoint terms. Model weights should stay outside git unless their license and size make tracking appropriate.

If a dependency is installed elsewhere on a server, you can still pass explicit paths in commands or configs:

```bash
bash scripts/run_da3_pipeline.sh --repo-dir /path/to/DepthAnything3
python -m videogaus.gaussian.train_gsplat --gsplat-examples-dir /path/to/gsplat/examples
bash scripts/run_da3_global_align.sh --xfeat-repo-dir /path/to/accelerated_features
```

## Project Policy

- External repositories live as submodules or documented paths, not copied code in `src/videogaus`.
- VideoGaussian adapters must call public CLI/API surfaces and save normalized outputs under `outputs/`.
- Generated outputs, checkpoints, splats, dense arrays, and rendered images do not belong in `third_party/`.
- If an upstream project is patched for an experiment, record the patch location and reason in `AGENTS.md` or `docs/`, and keep the patched code in the external checkout.

## Deferred Dependencies

VGGT-Omega was inspected during this project but is not part of the reproducible experiment path because the required checkpoint was unavailable. Do not add a runnable VGGT-Omega stage unless a checkpoint, validated adapter, and reproducible config are restored together.
