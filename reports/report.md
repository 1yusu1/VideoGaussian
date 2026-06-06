# VideoGaussian Results

## Liminal Pool

Protocol: real casual video, `fps12_conf96`, `181` frames, `30k` gsplat training. COLMAP registered `181/181` frames; all methods rendered `23` validation views.

| Method | PSNR | SSIM | LPIPS | #GS | Coverage |
|---|---:|---:|---:|---:|---|
| `colmap_gs` | 34.5433 | 0.9600 | 0.0813 | 1,745,683 | 181/181 |
| `da3_gs` | 26.4729 | 0.8723 | 0.2516 | 1,880,719 | 181/181 |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` | 27.4575 | 0.8868 | 0.1586 | 2,600,000 | 181/181 |

Delta over native DA3:

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4 - da3_gs` | +0.9846 | +0.0145 | -0.0931 |

## Generated Open-Plan Living Room

Protocol: LTX-generated indoor video, `1024x576`, `257` frames, `24 fps`, `10.71s`; extracted at `fps12` into `129` frames with `112` train and `17` held-out frames.

Source video:

```text
/data1/panshihan/videogaussian_generated/ltx_open_plan_living_1024x576/000_open_plan_living_walkthrough_seed20260606.mp4
```

| Geometry Source | Frames | Validation Views | Notes |
|---|---:|---:|---|
| `colmap` | 61 | 8 | Registered `61/129` extracted frames. |
| `da3` | 129 | 17 | Native DA3, `conf96`, `728,181` points. |
| `da3_xfeat_mask` | 129 | 17 | XFeat mask, `1,235` pairs, `896,385` matches, `1,800,000` points. |

| Method | PSNR | SSIM | LPIPS | #GS | Train Time (s) | Render s/img | Coverage |
|---|---:|---:|---:|---:|---:|---:|---|
| `colmap_gs` | 37.2170 | 0.9761 | 0.0636 | 177,930 | 422.8 | 0.0026 | 61/129 registered |
| `da3_gs` | 32.1479 | 0.9515 | 0.1256 | 890,495 | 600.2 | 0.0039 | 129/129 |
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4` | 35.6503 | 0.9714 | 0.0695 | 2,600,000 | 2311.9 | 0.0099 | 129/129 |

Delta over native DA3:

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4 - da3_gs` | +3.5023 | +0.0199 | -0.0562 |

COLMAP metrics on the generated video are for the registered subset only. DA3 and DA3/XFeat metrics cover all extracted frames.
