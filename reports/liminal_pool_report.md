# liminal_pool Report

## Retained Methods

| Setting | Method | Result Directory |
|---|---|---|
| fps12_conf96 | da3_gs_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_gs/gsplat |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4/gsplat |
| fps24_conf96 | colmap_gs_fps24_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_colmap_vs_da3/colmap_gs/gsplat |

## PSNR/SSIM/LPIPS

| Setting | Method | PSNR | SSIM | LPIPS | #GS |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_gs_fps12_conf96 | 26.4729 | 0.8723 | 0.2516 | 1880719 |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | 27.4575 | 0.8868 | 0.1586 | 2600000 |
| fps24_conf96 | colmap_gs_fps24_conf96 | 34.5433 | 0.9600 | 0.0813 | 1745683 |

## Key Observations

- The retained VGGTX/DA3 target beats naive DA3 initialization by +0.9846 PSNR, +0.0145 SSIM, and -0.0931 LPIPS.
- The retained target keeps DA3 cameras fixed, uses XFeat only as a support mask, trains with MCMC cap 2.6M, weak dense depth regularization, and SH degree 4.
- COLMAP remains the reference upper bound on this COLMAP-friendly scene, with a gap of -7.0859 PSNR, -0.0732 SSIM, and +0.0773 LPIPS from the retained target.
- Non-selected target rows were removed from this tracked report; remote training outputs were not deleted.

## Runtime

| Setting | Method | Training Time (s) | Render s/img | Render FPS | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_gs_fps12_conf96 | 1396.4150 | 0.0094 | 106.7670 |  |
| fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_sh4_fps12_conf96 | 3408.8328 | 0.0175 | 57.1561 |  |
| fps24_conf96 | colmap_gs_fps24_conf96 | 1634.4473 | 0.0095 | 105.4122 |  |
