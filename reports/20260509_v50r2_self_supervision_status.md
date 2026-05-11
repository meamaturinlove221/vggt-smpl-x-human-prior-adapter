# V50R2 自监督 / 几何自一致性状态

结论：目前版本已经实现导师录音里提到的 depth / point / normal 几何自一致约束，但它是辅助约束和审查证据，不是独立 teacher，也不是 V50R2 通过 candidate gate 的唯一原因。

## 如何实现

- `training/loss.py` 的 `compute_normal_loss` 从 dense point map 在线计算 target normal，并把 normal 监督限制在有效人体区域。
- 同一函数支持把预测 depth 反投影成 camera points，再计算 depth-derived normal，和网络输出 normal 做 cosine consistency。
- `compute_unproject_geometry_loss` 用预测 depth + 预测 camera pose 可微反投影到 world points，再和 world point / human prior pseudo point 对齐。
- `training/config` 下的 `depthnormal_coupled`、`xview_selfgeom`、`teacherless selfgeom`、`depth_point_normal_direct` 配置记录了 depth-normal、depth-point、point-normal 和 cross-view self-geometry 的实验路线。

## 效果

历史 r2 / r3 / r16 实验显示，normal-depth-point consistency 指标有改善；但同协议 6-view Open3D face/head 点云没有稳定超过 signfix/reference visual gate。因此汇报时应写成：自监督几何一致性已经实现并产生可测量改善，但它本身没有单独解决头脸连续表面问题。

## V50R2 中的定位

V50R2 把这条线作为 candidate consistency / normal-depth audit 使用。它支持 candidate 几何自洽审查，但不能包装成 strict teacher pass。
