# Third-Party Dependencies

`third_party/` is the boundary between VideoGaussian code and external research projects. The main repository should contain orchestration, adapters, configs, reports, and lightweight assets only. Do not vendor external source code by copying files into `src/videogaus`.

## Expected Dependencies

| Dependency | Purpose | Expected Path |
|---|---|---|
| Depth Anything 3 | Video depth, confidence, cameras, and COLMAP-style geometry export | `third_party/depth-anything-3` or an external checkout passed with `--repo-dir` |
| gsplat | Gaussian Splatting training, rendering, and checkpoint export | `third_party/gsplat` or `--gsplat-examples-dir` |
| XFeat / accelerated_features | Feature matching for DA3 global alignment ablations | `third_party/accelerated_features` or `--xfeat-repo-dir` |

## Recommended Setup

Prefer Git submodules when the project should be portable:

```bash
git submodule add <Depth-Anything-3-url> third_party/depth-anything-3
git submodule add <gsplat-url> third_party/gsplat
git submodule add <accelerated-features-url> third_party/accelerated_features
```

Using public repositories as submodules normally does not require contacting maintainers, but every dependency still carries its own license, citation requirements, and model checkpoint terms. Model weights should stay outside git unless their license and size make tracking appropriate.

If a dependency is installed elsewhere on a server, keep this directory source-free and pass explicit paths in commands or configs:

```bash
bash scripts/run_da3_pipeline.sh --repo-dir /path/to/DepthAnything3
python -m videogaus.gaussian.train_gsplat --gsplat-examples-dir /path/to/gsplat/examples
bash scripts/run_da3_global_align.sh --xfeat-repo-dir /path/to/accelerated_features
```

## Project Policy

- External repositories live as submodules or documented paths, not copied code.
- VideoGaussian adapters must call public CLI/API surfaces and save normalized outputs under `outputs/`.
- Generated outputs, checkpoints, splats, dense arrays, and rendered images do not belong in `third_party/`.
- If an upstream project is patched for an experiment, record the patch location and reason in `AGENTS.md` or `docs/`, and keep the patched code in the external checkout.

## Deferred Dependencies

VGGT-Omega was inspected during this project but is not part of the reproducible experiment path because the required checkpoint was unavailable. Do not add a runnable VGGT-Omega stage unless a checkpoint, validated adapter, and reproducible config are restored together.
