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
bash scripts/run_video_to_gaussian.sh \
  --config configs/my_scene.yaml \
  --test-mode
```

For live gsplat inspection, add `--enable-viewer` and forward port 8080:

```bash
ssh -L 8080:localhost:8080 <user>@<server>
```

For viewing the final PLY through the Spark viewer, forward port 8090:

```bash
ssh -L 8090:localhost:8090 <user>@<server>
```
