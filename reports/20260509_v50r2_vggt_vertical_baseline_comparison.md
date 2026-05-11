# V50R2 与 VGGT baseline 的纵向比较

本轮比较只使用同一 case、同一 `frame0000`、同一 V42/V50R2 六视角协议：`00, 01, 06, 11, 16, 21`。这样可以避免再次出现 view / mask / camera id 混用导致的假对比。

## 比较对象

- `V25 base VGGT`：base VGGT research prediction，作为本轮 VGGT baseline。
- `V42 prior-enabled VGGT`：接入 SMPL-X prior / HumanPriorAdapter 后的 prior-enabled prediction。
- `V50R2 candidate package`：当前 strict candidate package。注意它的主 point map 与 V42 frame0000 前六视角完全一致，额外价值在 head/face patch、hand patch、normal evidence、visual gate 和 package/registry 交付。
- `V16 SMPL-X prior only`：SMPL-X native prior-only reference，不是 VGGT 输出，不参与 baseline 胜负结论。

## 关键数值结论

- V42 prior-enabled 与 V25 baseline 的全像素 mean L2 差异：`0.00053544`。
- V50R2 candidate 主点图与 V42 prior-enabled 前六视角 max abs 差异：`0.00000000`。
- 因此，如果只看主 point map 坐标，V50R2 不应被写成相对 V42 又发生了一次大幅几何提升；V50R2 的提升是 candidate package、normal/region evidence、visual gate 和正式解锁层面的闭环。

## 区域均值表

| method | region | valid_points_mean | coverage_mean | z_relief | neighbor_delta_p95 | L2 vs V25 | normal |
|---|---:|---:|---:|---:|---:|---:|---|
| V25 base VGGT | full | 11575.5 | 1 | 0.198347 | 0.00423296 | NA | False |
| V25 base VGGT | head_face | 4019.17 | 1 | 0.0829389 | 0.00404264 | NA | False |
| V25 base VGGT | left_hand | 3276 | 1 | 0.111296 | 0.00528568 | NA | False |
| V25 base VGGT | right_hand | 4905.67 | 1 | 0.129735 | 0.00378373 | NA | False |
| V42 prior-enabled VGGT | full | 11575.5 | 1 | 0.19824 | 0.00423233 | 0.000244902 | True |
| V42 prior-enabled VGGT | head_face | 4019.17 | 1 | 0.0828137 | 0.00404026 | 0.000296476 | True |
| V42 prior-enabled VGGT | left_hand | 3276 | 1 | 0.111313 | 0.005275 | 0.000296937 | True |
| V42 prior-enabled VGGT | right_hand | 4905.67 | 1 | 0.129656 | 0.00377863 | 0.000200385 | True |
| V50R2 candidate package | full | 11575.5 | 1 | 0.19824 | 0.00423233 | 0.000244902 | True |
| V50R2 candidate package | head_face | 2450.83 | 1 | 0.0659072 | 0.00484538 | 0.00048253 | True |
| V50R2 candidate package | left_hand | 1248.83 | 1 | 0.0808403 | 0.00525253 | 0.00029834 | True |
| V50R2 candidate package | right_hand | 842.833 | 1 | 0.0535645 | 0.00458839 | 0.000221999 | True |
| V16 SMPL-X prior only | full | 11575.5 | 1 | 0.632774 | 0.00378068 | NA | True |
| V16 SMPL-X prior only | head_face | 4019.17 | 1 | 0.551812 | 0.00328922 | NA | True |
| V16 SMPL-X prior only | left_hand | 3276 | 1 | 0.490339 | 0.00473891 | NA | True |
| V16 SMPL-X prior only | right_hand | 4905.67 | 1 | 0.496608 | 0.00309694 | NA | True |

## 点云图合集

### Full body RGB-colored point cloud
![full rgb](D:\vggt\vggt-main\output\mentor_report_v50r2\images\09_vertical_full_body_rgb.png)

### Head / face RGB-colored point cloud
![head face rgb](D:\vggt\vggt-main\output\mentor_report_v50r2\images\10_vertical_head_face_rgb.png)

### Hands RGB-colored point cloud
![hands rgb](D:\vggt\vggt-main\output\mentor_report_v50r2\images\11_vertical_hands_rgb.png)

### Geometry/depth-colored checks
![full depth](D:\vggt\vggt-main\output\mentor_report_v50r2\images\12_vertical_full_body_depth.png)

![head face depth](D:\vggt\vggt-main\output\mentor_report_v50r2\images\13_vertical_head_face_depth.png)

![hands depth](D:\vggt\vggt-main\output\mentor_report_v50r2\images\14_vertical_hands_depth.png)

## 给导师的表述建议

这一组结果说明：相比 base VGGT，prior-enabled 路线确实让输出进入了可审查、可打包、带 normal/region evidence 的 candidate 闭环；但从主 point map 的几何坐标看，V42 相对 V25 的变化幅度并不大，V50R2 主点图又与 V42 相同。因此当前版本不能夸大成“细节几何已经明显大幅超过 baseline”。更稳的结论是：baseline 的全身轮廓已经能出来，SMPL-X prior-enabled 路线主要补齐了人体区域、normal/region consistency、head/hand candidate evidence 和 D-line 交付闭环；下一步真正的改进空间仍在 head/face/hairline 局部表面、右手，以及能直接改变 target-view point map 的局部几何优化。

## 输出文件

- JSON: `D:\vggt\vggt-main\reports\20260509_v50r2_vggt_vertical_baseline_comparison.json`
- CSV: `D:\vggt\vggt-main\reports\20260509_v50r2_vggt_vertical_baseline_metrics.csv`
- Image dir: `D:\vggt\vggt-main\output\mentor_report_v50r2\vertical_vggt_baseline_comparison\images`
- PLY dir: `D:\vggt\vggt-main\output\mentor_report_v50r2\vertical_vggt_baseline_comparison\ply`
