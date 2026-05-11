# 人体高质量几何恢复的 SMPL-X Native 候选闭环

## 摘要

本阶段工作从“把 SMPL-X 接入 VGGT”继续推进到“把 SMPL-X native prior-enabled VGGT candidate 做成可复查、可归档、可向导师汇报的工程闭环”。这版报告不再沿用前面几轮错误的可视化结果：旧图把 V50R2 的 candidate point map 和 V15/V16 的 6-view 图像、mask、camera id 混在一起，导致点云在 MeshLab / Open3D 里出现撕裂、残影、多人体和薄片感。现在已经统一改回 V42/V50R2 对齐协议，正确 view 顺序是 `00, 01, 06, 11, 16, 21`。

当前结果达到的是 candidate 路线的可交付状态：`strict_candidate_passes = 1`，`formal_cloud_unblocked = true`。同时需要明确，`strict_teacher_passes = 0`，也就是现在不是 dense teacher 已完成，而是 SMPL-X native prior-enabled VGGT candidate route 已经形成。右手、头脸和发际线仍然属于需要向导师说明的视觉风险区，不能在汇报里写成完全解决。

导师前面问的几个问题，这版也都做了直接回应：SMPL-X 不能只当模板强压图像特征；depth、point、normal 必须有几何一致性，而不是三个分支各学各的；不能只展示上半身，full body、head / face / hairline、left / right hand 都要单独审；如果是真实拍摄、没有 SMPL-X 和相机参数，当前流程还需要前置姿态 / 相机优化，不能直接声称无标注实拍已经解决。

![full body](D:/vggt/vggt-main/output/mentor_report_v50r2/images/01_full_body_v42_consistent_pointcloud.png)

## 架构图

下面这张图是在原版 VGGT 的结构上加出的当前路线。主干仍然保留原 VGGT 的 image tokens、camera token、global / frame attention、camera head 和 DPT dense heads；新增部分是 SMPL-X native prior branch、HumanPriorAdapter、input-side / layer-wise prior fusion、self-geometry consistency，以及 D-line candidate gate。

![architecture](D:/vggt/vggt-main/output/mentor_report_v50r2/images/05_vggt_smplx_native_architecture.png)

## 一、当前路线的定位

当前主线不是继续找新的外部 hand / hair / FLAME / MANO 项目，而是把 SMPL-X native 当作人体结构先验，与 VGGT 的 depth、point map、normal 证据结合起来。和最早“只是把 SMPL-X 接进 VGGT”的阶段相比，现在的区别主要有两点。

第一，SMPL-X 不再被当作最终真值。它只提供 pose-aligned 的人体拓扑、手部大体结构、头脸区域、可见性和粗深度 / 法向提示。真正的输出仍然来自 VGGT 的 point / depth / normal 路线，以及后续的 candidate package。

第二，当前工作不把某个局部增强结果直接包装成 teacher。Teacher route 仍然冻结，因为还没有独立 dense sensor / MVS / 2DGS surface 能够通过 full / head / face / hair / hands 的严格 ownership 审查。现在能成立的是 candidate pass，而不是 teacher pass。

当前 active candidate 路径是：

`D:\vggt\vggt-main\output\frozen_candidates\V50R2_rebuilt_from_sessions_gdrive_modal`

对应 strict registry 路径是：

`D:\vggt\vggt-main\output\frozen_candidates\V50R2_rebuilt_from_sessions_gdrive_modal\strict_registry_entry_v50r2.json`

## 二、对导师问题的直接回应

导师问得比较关键的一点是：如果只有六张真人照片，没有 SMPL-X 参数，也没有相机参数，这个流程是否自洽？

当前这个 case 的答案是：本实验仍然建立在 4K4D / DNA 这类数据已经有 SMPL-X 和相机参数的前提下。也就是说，当前路线证明的是“当姿态、相机、SMPL-X 参数可用时，如何把这些人体先验送进 VGGT，并形成 candidate 级别的几何恢复”。它还不能直接声称已经解决了无标注真实拍摄场景。

如果转到普通真人实拍，下一步必须补一个前置模块：先从图像轮廓、关键点和多视角一致性中优化 SMPL-X / 相机参数，使 SMPL-X 投影和人像轮廓对齐，然后再进入现在的 prior-enabled VGGT 路线。这一步不能跳过，也不能在汇报里省略。

导师还提到 normal 是否是最有用的提升点。当前实验结论更接近：normal 是必要条件，但不是单独充分条件。只让网络输出 normal 图，或者只加一个 normal loss，并不会自动带来清晰连续的人脸点云。更有效的表述应当是：depth / point / normal 三者需要耦合，尤其是 depth 反投影得到的 point map，以及由点图或深度差分得到的 geometric normal，要和网络输出 normal 保持一致。

这也是为什么 V50R2 的 normal evidence 写成 candidate normal，而不是 teacher normal。它可以用来审查 candidate 几何是否自洽，但它不是独立 teacher。

## 三、SMPL-X Native Prior 的作用

SMPL-X 在当前路线里的作用可以概括为四个方面。

第一，它提供 pose-aligned 的人体大体结构。对 full body 来说，这解决的是人体在哪里、各身体部件大致在哪里、手和头脸的大致区域在哪里的问题。

第二，它提供 view-aligned prior maps。也就是把 SMPL-X mesh 投影到真实相机下，形成 prior depth、prior points、prior normals、visibility、part map 和 ROI mask。这些信息不是孤立 mesh，而是和输入 view 对齐。

第三，它帮助做 region ownership。当前 body、head、face、left hand、right hand 都有对应 region evidence，避免只看 full-body 总点数。

第四，它约束手部和头脸区域，但不能硬锁成模板。导师指出“SMPL-X 模板感太强”这个问题是准确的，所以当前报告不把 SMPL-X 当 dense teacher，只把它作为 weak structural prior 和 candidate consistency 的一部分。

![head face](D:/vggt/vggt-main/output/mentor_report_v50r2/images/03_head_face_v42_consistent_pointcloud.png)

## 四、Depth / Point / Normal 的几何耦合

当前路线保留了 depth unprojection，也就是从深度反投影得到 3D points。这个设置和导师提到的“深度图每一点更准，直接反投影，点云应该能出来”是一致的。

但现在没有把 depth 和 normal 当成两个完全独立的输出看待。更合理的理解是：

- depth 负责目标视角下的距离和可见表面位置；
- point map 是 depth / camera / world coordinate 下的三维表达；
- normal 用来约束局部表面方向；
- geometric normal 可以由 point map 或 depth 差分得到；
- 输出 normal 应该和 geometric normal 保持一致。

这也解释了为什么 V50R2 的 normal evidence 可以支持 candidate 审查，却不能单独支持 teacher pass。它证明的是候选几何自洽，不是独立几何来源。

## 五、自监督 / 几何自一致性目前做到哪里

导师录音里提到的“depth 可以转 normal，normal 又要反过来帮助几何”这一点，目前不是停留在想法上，代码里已经实现成了几何自一致约束。更准确地说，它不是一个单独的无监督 teacher，而是训练和审查阶段的一组 self-geometry consistency loss / audit。

实现上分三层。

第一层是 normal 分支不再只学一张 2D normal 图。`training/loss.py` 里 normal loss 的目标 normal 会从同一批 dense point map 在线计算出来，同时还支持把预测 depth 反投影成 camera points，再由这些点计算 depth-derived normal，和网络输出 normal 做 cosine consistency。这样 normal、depth、point 不再是三个互相独立的输出。

第二层是 depth-to-point 的几何链路。`compute_unproject_geometry_loss` 会用预测 depth 和预测 camera pose 可微地反投影成 world points，再和目标 world points / human prior pseudo points 对齐。这对应导师说的“深度图每一点深度准了，直接反投影，点云应该能出来”。

第三层是自监督实验配置。4 月底到 5 月初跑过 `depthnormal_coupled`、`xview_selfgeom`、`teacherless selfgeom`、`depth_point_normal_direct` 等配置，里面包含 depth-normal、depth-point、point-normal、cross-view geometry consistency、ROI 加权和 softmatte / crop 组合。这些不是 V50R2 最终打包时才临时加的，而是前面 normal/crop/SMPL 三条线反复验证后保留下来的辅助约束。

效果需要如实说：有改善，但不是单独决定性突破。历史 r2 / r3 / r16 结果显示，normal-depth-point consistency 指标确实变好，例如 pred-normal vs depth-normal、pred-normal vs point-normal、depth-normal vs point-normal 在多行指标上下降；但是同协议 6-view 的 Open3D face/head 点云没有稳定超过 signfix baseline，也没有直接得到导师要求的清晰连续人脸表面。也就是说，自监督几何一致性解决的是“输出之间互相矛盾”的问题，不能单独解决“缺少高质量连续头脸几何 teacher / 局部可见面不充分”的问题。

因此 V50R2 文档里应该把它写成：已实现并作为 candidate consistency / normal-depth audit 使用；它是必要的几何约束，不是当前 strict pass 的唯一来源，也不能被包装成 teacher pass。当前 V50R2 的正结果来自 SMPL-X native prior、prior-enabled candidate、V42/V50R2 对齐点云、formal replay、visual gate 和 D-line candidate transaction 的组合。

## 六、VGGT 系列纵向比较

按导师要求，这里先做同一条 VGGT 纵向路线的 baseline 对比，而不是直接和外部人体方法混在一起比较。比较对象限定在同一个 `frame0000`、同一个 V42/V50R2 六视角协议、同一套图像和 mask 下：V25 base VGGT、V42 prior-enabled VGGT、V50R2 candidate package，以及一个只作参考的 SMPL-X prior-only。

数值上看，V42 prior-enabled 相比 V25 base VGGT 的全像素 point map mean L2 差异是 `0.0005354409804567695`；V50R2 candidate 主点图和 V42 prior-enabled 前六视角的 max abs 差异是 `0.0`。这说明，如果只看主 point map 坐标，当前路线相对 baseline 的几何位移并不大，V50R2 也不是在 V42 之后又生成了一套全新的 full-body point map。

因此这一版不能夸大成“细节几何已经明显大幅超过 baseline”。更准确的说法是：baseline 已经能给出基本人体点云；prior-enabled / V50R2 路线主要补齐的是人体 prior 输入、normal availability、region evidence、head/hand candidate patch、formal replay 和 D-line candidate 交付闭环。导师说“看起来还是不太够”这一点是成立的，尤其在 head / face / hairline 的真实局部起伏、右手细节、衣物和模板差异方面，还有明显改进空间。

![vertical full body rgb](D:/vggt/vggt-main/output/mentor_report_v50r2/images/09_vertical_full_body_rgb.png)

![vertical head face rgb](D:/vggt/vggt-main/output/mentor_report_v50r2/images/10_vertical_head_face_rgb.png)

![vertical hands rgb](D:/vggt/vggt-main/output/mentor_report_v50r2/images/11_vertical_hands_rgb.png)

为了避免 RGB 上色点云看起来像照片裁剪，本轮同时生成了 depth / geometry 上色检查图。它们更适合看 point map 的深度起伏和局部几何厚度。

![vertical full body depth](D:/vggt/vggt-main/output/mentor_report_v50r2/images/12_vertical_full_body_depth.png)

![vertical head face depth](D:/vggt/vggt-main/output/mentor_report_v50r2/images/13_vertical_head_face_depth.png)

![vertical hands depth](D:/vggt/vggt-main/output/mentor_report_v50r2/images/14_vertical_hands_depth.png)

这组对比的结论是：V50R2 可以作为 candidate pass 汇报，但如果导师要求的是“整体细节相比 VGGT baseline 明显改善”，当前证据还不够强。下一步需要直接作用在 target-view point map 上的局部几何优化，重点是 head / face / hairline 和 right hand，而不是只继续增加 package 证据。

## 七、Full Body 审查结果

导师强调不能只看上半身，这一版把 full body 放进了最终 gate。当前 full body 的判断是 `PASS_WITH_RISK`，不是无风险通过。

全身结构整体可读，候选包可以加载，hash manifest 能够复查，formal replay 没有写入 forbidden output。但 full body 仍然有稀疏和局部模板感的问题，尤其是衣物与人体模板之间的差异没有完全恢复，所以不能说已经达到高质量 mesh / dense teacher 的水平。

![upper body](D:/vggt/vggt-main/output/mentor_report_v50r2/images/02_upper_body_v42_consistent_pointcloud.png)

## 八、Head / Face / Hairline 状态

头脸是导师最关注的区域。当前 V50R2 相比最早的工程状态，已经不是完全空洞或只有模板壳的状态；head / face evidence 是非空的，也进入了 final visual board。

但这里仍然要谨慎表述。当前 head / face / hairline 的结论是 `PASS_WITH_RISK`。也就是说，它可以作为候选结果给导师看，但不能说面部已经达到“清晰、连续、稳定的人脸几何”的最终标准。

发际线也不是 HairGS / GaussianHaircut 那种真实发丝拓扑，而是 SMPL-X native / candidate route 下的 hairline-level evidence。因此它能支持 candidate 审查，但不能支持 independent hair teacher。

## 九、手部状态

手部这次没有继续引入 MANO、HaMeR、WiLoR 等外部路线，而是按导师要求先使用 SMPL-X native 的手部结构处理。

当前 left hand 和 right hand 都有 evidence。右手不是完全没有，但它仍然是最弱的区域。V253 的结论是 `MERGE_FAIL_SOFT_REVIEW_ONLY`：局部 right-hand patch 可以改善一部分可见性，但没有足够证据证明 hard merge 后 full body、head、temporal 和 60-view 都不退化。所以当前没有强行把右手 patch 合进新的 candidate，而是保留 V50R2 作为 active candidate，并把右手风险写入导师包。

这个处理是保守的，但比把右手风险藏起来更稳。

![hands](D:/vggt/vggt-main/output/mentor_report_v50r2/images/04_hands_v42_consistent_pointcloud.png)

## 十、Formal Cloud / 60-view / Temporal 审查

当前 formal cloud 的状态是 unblocked。最终检查没有残留 Modal app / container，也没有后台 Python / Modal 进程继续占资源。

60-view 方面，当前有 raw 4K4D SMC triplet 和 V35 support evidence。需要强调的是，这里不是重新生成了一个完美 60-view dense teacher，而是用当前可用数据完成了 V50R2 candidate 的 support / robustness 审查。因此结论是 candidate robustness evidence，而不是 teacher evidence。

Temporal 方面，当前完成的是 frame0000 / frame0001 / frame0002 的三帧 stress / support。它可以说明当前候选不是只依赖单一截图，但不能扩展成多主体泛化结论。

## 十一、没有写 Teacher Pass 的原因

当前 `strict_teacher_passes = 0`，这是有意保留的，不是漏写。

teacher route 的要求比 candidate route 高。它需要独立 dense geometry source，例如 Kinect TSDF、MVS、2DGS / Gaussian surface，并且必须通过 known-camera alignment、depth / normal、full / head / face / hair / hands region ownership 和 Open3D visual gate。

现在 V50R2 的 evidence 是 candidate-derived。它可以证明 candidate route 成立，但不能反过来当成 independent teacher。也就是说，不能用自己的候选结果再包装成 teacher，这会破坏 D-line 的判定逻辑。

因此当前合理表述是：candidate pass 已完成，teacher route 冻结；如果后面有新的独立 dense sensor / MVS evidence，再重新打开 teacher resurrection。当前 teacher route 状态记录为 `FAIL_FROZEN`。

## 十二、和 4 月阶段相比的变化

4 月阶段的主要结论是：SMPL-X prior、dense prior maps、layer-wise fusion、output-side depth / point / normal supervision、detail normal refiner、external SMPL-X bridge 等工程链路都已经存在，但 6-view head / face 点云没有达到足够清晰稳定。

现在的变化是：项目已经从“工程链路是否能跑通”推进到“候选包能否被严格归档并通过 D-line candidate gate”。V50R2 已经给出了一个可加载、可 hash、可归档、可 formal replay 的 candidate package。

但最终科学问题还没有完全结束。当前仍然不能说 dense teacher 完成，也不能说右手和发际线完全解决。更准确的说法是：SMPL-X native prior-enabled VGGT candidate route 已经形成一个可交付版本；导师如果接受 candidate pass 和显式风险，这一阶段可以提交；如果导师要求 teacher pass 或更强右手 / 头发质量，还需要新的独立几何来源或更强的局部优化。

## 十三、当前成果

当前阶段完成了以下几件事：

1. 恢复并冻结 V50R2 candidate package。
2. 重新跑通 strict candidate gate，`strict_candidate_passes = 1`。
3. formal cloud gate 已解锁，且最终无 Modal app / container 残留。
4. full body、upper body、head / face、left / right hand 都有 V42/V50R2 一致协议下的点云图。
5. 修复了旧图把 V50R2 point map 和 V15/V16 view 混用的问题。
6. 右手风险已显式记录，没有强行 hard merge。
7. teacher route 没有伪写 pass，保持 `strict_teacher_passes = 0`。

## 十四、总结

这一版的汇报重点不应写成“已经得到完美人体重建”，而应写成“SMPL-X native prior-enabled VGGT candidate route 已经完成可交付闭环”。

它回应了导师提出的几个核心问题：SMPL-X 作为 pose-aligned prior 是有用的，但不能过强；depth / point / normal 要耦合，不能各自独立；全身和手部不能忽略；真实拍摄无 SMPL-X 参数时还需要前置姿态 / 相机优化；normal 是有效约束之一，但不是唯一答案。

当前最稳妥的结论是：V50R2 已达到 candidate pass 和 formal cloud unblock 的工程交付要求；头脸、发际线、右手仍是带风险的视觉区域；teacher pass 未完成且不应伪造。下一阶段如果继续推进，优先级应是右手局部非退化增强、头脸 / 发际线更强局部几何，以及独立 dense teacher source。
