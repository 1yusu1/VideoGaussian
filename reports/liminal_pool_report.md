# liminal_pool Report

## Method Table

| Setting | Method | Result Directory |
|---|---|---|
| fps12_conf96 | da3_ga_xfeat_gs_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_gs/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_gs_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_gs/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_mcmc_pose_depthreg/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85/gsplat |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001/gsplat |
| fps12_conf96 | da3_gs_dense_depthreg_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_gs_dense_depthreg/gsplat |
| fps12_conf96 | da3_gs_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_gs/gsplat |
| fps12_conf96 | da3_gs_sparse_depthreg_fps12_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_fps24_conf96/da3_gs_depthreg/gsplat |
| fps2 | colmap_gs_fps2 | /data1/panshihan/videogaussian_runs/liminal_pool_fps2_ablation/colmap_gs/gsplat |
| fps24_conf96 | colmap_gs_fps24_conf96 | /data1/panshihan/videogaussian_runs/liminal_pool_colmap_vs_da3/colmap_gs/gsplat |
| fps2_conf70 | da3_gs_dense_depthreg_fps2_conf70 | /data1/panshihan/videogaussian_runs/liminal_pool_fps2_conf_sweep/conf70/da3_gs_dense_depthreg/gsplat |
| fps2_conf70 | da3_gs_fps2_conf70 | /data1/panshihan/videogaussian_runs/liminal_pool_fps2_conf_sweep/conf70/da3_gs/gsplat |

## PSNR/SSIM/LPIPS Table

| Setting | Method | PSNR | SSIM | LPIPS | #GS |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_ga_xfeat_gs_fps12_conf96 | 21.0636 | 0.8044 | 0.4566 | 1006884 |
| fps12_conf96 | da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 22.7793 | 0.8234 | 0.3551 | 1800000 |
| fps12_conf96 | da3_ga_xfeat_v2_gs_fps12_conf96 | 22.7555 | 0.8246 | 0.3535 | 1707957 |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_fps12_conf96 | 21.7470 | 0.7928 | 0.2869 | 1800000 |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 22.9551 | 0.8260 | 0.3447 | 1800000 |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005_fps12_conf96 | 22.6683 | 0.8183 | 0.3159 | 1800000 |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85_fps12_conf96 | 22.7753 | 0.8188 | 0.3182 | 1800000 |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_fps12_conf96 | 22.7323 | 0.8179 | 0.3186 | 1800000 |
| fps12_conf96 | da3_gs_dense_depthreg_fps12_conf96 | 26.7192 | 0.8746 | 0.2316 | 1497448 |
| fps12_conf96 | da3_gs_fps12_conf96 | 26.4729 | 0.8723 | 0.2516 | 1880719 |
| fps12_conf96 | da3_gs_sparse_depthreg_fps12_conf96 | 26.6368 | 0.8739 | 0.2353 | 1815594 |
| fps2 | colmap_gs_fps2 | 26.0179 | 0.8665 | 0.2731 | 1570118 |
| fps24_conf96 | colmap_gs_fps24_conf96 | 34.5433 | 0.9600 | 0.0813 | 1745683 |
| fps2_conf70 | da3_gs_dense_depthreg_fps2_conf70 | 18.8310 | 0.7162 | 0.4738 | 1443703 |
| fps2_conf70 | da3_gs_fps2_conf70 | 17.9810 | 0.7060 | 0.4612 | 1713092 |

## Key Observations

- Best PSNR is `34.5433` from `colmap_gs_fps24_conf96` in `fps24_conf96`.
- Best LPIPS is `0.0813` from `colmap_gs_fps24_conf96` in `fps24_conf96`.
- On fps12/conf96, DA3 depth regularization improves DA3 initialization modestly but does not close the gap to COLMAP.
- VGGT-X-style epipolar GA is a negative ablation on this scene: it trails direct DA3 initialization, likely because pose-only alignment disturbs DA3 camera-depth coupling.
- DA3 GA XFeat v2 recovers part of the epipolar-only GA loss, but still remains below direct DA3 initialization on PSNR/SSIM.
- MCMC + pose optimization + dense depth regularization improves perceptual LPIPS over v2 default, but lowers PSNR/SSIM and costs more training/render time.
- Weakening pose/depth regularization improves GA+MCMC PSNR/SSIM; the best weakened variant is `da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96` with PSNR `22.9551`.
- Reducing GA v2 initialization to 500k points gives MCMC room to add Gaussians, but does not improve metrics over the best 1.8M-init weakened GA+MCMC variant on this scene.
- On fps2, COLMAP remains stronger for this liminal_pool scene; DA3 conf70 adds more points but remains much worse, suggesting noisy or globally inconsistent DA3 geometry.

## Qualitative Comparison

Add rendered train/test comparisons from each method here. Recommended layout: ground truth, COLMAP+GS, DA3+GS, DA3+DepthReg, DA3+XFeat-GA-v2.

## Failure Cases

Add frames with pose drift, depth bleeding, dynamic objects, or textureless regions here.

## Runtime/Memory

| Setting | Method | Training Time (s) | Render s/img | Render FPS | Peak GPU MiB |
|---|---|---:|---:|---:|---:|
| fps12_conf96 | da3_ga_xfeat_gs_fps12_conf96 | 1158.3905 | 0.0071 | 141.6705 |  |
| fps12_conf96 | da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 3088.2677 | 0.0154 | 65.0595 |  |
| fps12_conf96 | da3_ga_xfeat_v2_gs_fps12_conf96 | 1702.5219 | 0.0094 | 106.3239 |  |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_fps12_conf96 | 3000.8666 | 0.0160 | 62.5640 |  |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 2941.5947 | 0.0152 | 65.9958 |  |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005_fps12_conf96 | 2829.4486 | 0.0151 | 66.2042 |  |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85_fps12_conf96 | 2868.9370 | 0.0149 | 67.2954 |  |
| fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_fps12_conf96 | 2869.4625 | 0.0145 | 68.9060 |  |
| fps12_conf96 | da3_gs_dense_depthreg_fps12_conf96 | 1466.0963 | 0.0094 | 106.1708 |  |
| fps12_conf96 | da3_gs_fps12_conf96 | 1396.4150 | 0.0094 | 106.7670 |  |
| fps12_conf96 | da3_gs_sparse_depthreg_fps12_conf96 | 1610.1480 | 0.0100 | 100.1461 |  |
| fps2 | colmap_gs_fps2 | 1593.8011 | 0.0092 | 108.8031 |  |
| fps24_conf96 | colmap_gs_fps24_conf96 | 1634.4473 | 0.0095 | 105.4122 |  |
| fps2_conf70 | da3_gs_dense_depthreg_fps2_conf70 | 1840.1487 | 0.0199 | 50.1578 |  |
| fps2_conf70 | da3_gs_fps2_conf70 | 1602.2191 | 0.0102 | 98.0138 |  |
