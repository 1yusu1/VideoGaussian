# Remote Server Notes

The intended server setup is:

```text
<WORKSPACE>/Depth-Anything-3
<WORKSPACE>/gsplat-1.5.3
<WORKSPACE>/VideoGaussian
<RUNS_DIR>
```

Create a scene config:

```bash
cp configs/pipeline.example.yaml configs/my_scene.yaml
```

Edit `configs/my_scene.yaml` with the video path, upstream repo paths, model path, output root, and GPU ids.

Smoke test:

```bash
python -m compileall src scripts
```

For the liminal_pool study, follow `AGENTS.md`: use `project_310`, keep large outputs under `/data1/panshihan/videogaussian_runs`, and run the explicit stage scripts for DA3, global alignment, gsplat training, evaluation, and report generation. The legacy `scripts/run_video_to_gaussian.sh` wrapper is retained, but explicit stages are preferred for reproducible comparisons.

For live gsplat inspection, add `--enable-viewer` and forward port 8080:

```bash
ssh -L 8080:localhost:8080 <user>@<server>
```

For viewing the final PLY through the Spark viewer, forward port 8090:

```bash
ssh -L 8090:localhost:8090 <user>@<server>
```
