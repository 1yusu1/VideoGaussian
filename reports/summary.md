# VideoGaussian Summary

| Scene | Setting | Method | PSNR | SSIM | LPIPS | #GS | Render s/img | Train Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| liminal_pool | fps12_conf96 | da3_ga_xfeat_gs_fps12_conf96 | 21.0636 | 0.8044 | 0.4566 | 1006884 | 0.0071 | 1158.3905 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_500k_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 22.7793 | 0.8234 | 0.3551 | 1800000 | 0.0154 | 3088.2677 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_gs_fps12_conf96 | 22.7555 | 0.8246 | 0.3535 | 1707957 | 0.0094 | 1702.5219 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_fps12_conf96 | 21.7470 | 0.7928 | 0.2869 | 1800000 | 0.0160 | 3000.8666 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr1e6_w001_fps12_conf96 | 22.9551 | 0.8260 | 0.3447 | 1800000 | 0.0152 | 2941.5947 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w0005_fps12_conf96 | 22.6683 | 0.8183 | 0.3159 | 1800000 | 0.0151 | 2829.4486 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_conf85_fps12_conf96 | 22.7753 | 0.8188 | 0.3182 | 1800000 | 0.0149 | 2868.9370 |
| liminal_pool | fps12_conf96 | da3_ga_xfeat_v2_mcmc_pose_depthreg_lr3e6_w001_fps12_conf96 | 22.7323 | 0.8179 | 0.3186 | 1800000 | 0.0145 | 2869.4625 |
| liminal_pool | fps12_conf96 | da3_gs_dense_depthreg_fps12_conf96 | 26.7192 | 0.8746 | 0.2316 | 1497448 | 0.0094 | 1466.0963 |
| liminal_pool | fps12_conf96 | da3_gs_fps12_conf96 | 26.4729 | 0.8723 | 0.2516 | 1880719 | 0.0094 | 1396.4150 |
| liminal_pool | fps12_conf96 | da3_gs_sparse_depthreg_fps12_conf96 | 26.6368 | 0.8739 | 0.2353 | 1815594 | 0.0100 | 1610.1480 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_2400k_mcmc_cap3000_dense_depthreg_fps12_conf96 | 27.2551 | 0.8827 | 0.1838 | 3000000 | 0.0182 | 3621.8573 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_dense_depthreg_fps12_conf96 | 27.1033 | 0.8797 | 0.1845 | 2000636 | 0.0085 | 1748.4650 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_depthreg_fps12_conf96 | 27.2528 | 0.8825 | 0.1854 | 2600000 | 0.0168 | 3354.4538 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf70_fps12_conf96 | 27.3500 | 0.8852 | 0.1677 | 2600000 | 0.0170 | 3131.9398 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w0005_conf85_fps12_conf96 | 27.4335 | 0.8864 | 0.1616 | 2600000 | 0.0165 | 3049.0354 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_app_fps12_conf96 | 24.8177 | 0.8329 | 0.1834 | 2600000 | 0.0315 | 4583.1598 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_bilateral_fps12_conf96 | 26.6432 | 0.8686 | 0.1698 | 2600000 | 0.0151 | 4094.9764 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_fps12_conf96 | 27.3767 | 0.8852 | 0.1686 | 2600000 | 0.0163 | 3110.0417 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_opacity001_fps12_conf96 | 27.1043 | 0.8808 | 0.1686 | 2600000 | 0.0123 | 2204.3540 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf70_sh4_fps12_conf96 | 27.4235 | 0.8854 | 0.1654 | 2600000 | 0.0174 | 3465.6859 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_dense_w001_conf85_fps12_conf96 | 27.3767 | 0.8856 | 0.1678 | 2600000 | 0.0169 | 3197.1380 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap2600_refine30k_dense_depthreg_fps12_conf96 | 27.1361 | 0.8804 | 0.1941 | 2600000 | 0.0207 | 3395.3490 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_cap3000_dense_depthreg_fps12_conf96 | 27.1833 | 0.8812 | 0.1884 | 3000000 | 0.0181 | 3662.0717 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_dense_depthreg_fps12_conf96 | 27.2221 | 0.8821 | 0.1855 | 2200000 | 0.0160 | 3034.9239 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_pose_dense_depthreg_fps12_conf96 | 26.7115 | 0.8772 | 0.1536 | 2200000 | 0.0146 | 2865.3260 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_pose_lr1e7_dense_depthreg_fps12_conf96 | 27.1980 | 0.8819 | 0.1817 | 2200000 | 0.0167 | 3070.6198 |
| liminal_pool | fps12_conf96 | da3_xfeat_mask_mcmc_pose_lr3e7_dense_depthreg_fps12_conf96 | 27.0781 | 0.8801 | 0.1863 | 2200000 | 0.0166 | 3179.5074 |
| liminal_pool | fps2 | colmap_gs_fps2 | 26.0179 | 0.8665 | 0.2731 | 1570118 | 0.0092 | 1593.8011 |
| liminal_pool | fps24_conf96 | colmap_gs_fps24_conf96 | 34.5433 | 0.9600 | 0.0813 | 1745683 | 0.0095 | 1634.4473 |
| liminal_pool | fps2_conf70 | da3_gs_dense_depthreg_fps2_conf70 | 18.8310 | 0.7162 | 0.4738 | 1443703 | 0.0199 | 1840.1487 |
| liminal_pool | fps2_conf70 | da3_gs_fps2_conf70 | 17.9810 | 0.7060 | 0.4612 | 1713092 | 0.0102 | 1602.2191 |
