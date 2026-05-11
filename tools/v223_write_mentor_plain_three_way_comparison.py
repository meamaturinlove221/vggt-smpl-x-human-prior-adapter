from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEND = ROOT / "output" / "mentor_report_v50r2" / "send_to_mentor_v42_consistent"
OUT_MD = REPORTS / "20260511_mentor_plain_three_way_vggt_smplx_normal_comparison.md"
OUT_JSON = REPORTS / "20260511_mentor_plain_three_way_vggt_smplx_normal_comparison.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    SEND.mkdir(parents=True, exist_ok=True)
    text = """# 按导师原话整理：VGGT 原版、SMPL-X + VGGT、当前 normal 路线三组对比

这版汇报不按内部版本号讲，只按导师关心的三个结果来讲：

1. 直接跑 VGGT 原版 baseline 是什么效果；
2. 加入 SMPL-X 以后，对人体结构有没有帮助，有没有模板感问题；
3. 当前 normal 路线有没有按照导师说的 depth / point / normal 耦合来做，效果比前两者改善在哪里、还差在哪里。

## 一、VGGT 原版 baseline

直接用 VGGT 原版做这个 6-view 人体点云，可以恢复一个粗的人体。全身轮廓是有的，身体、头、手的大概位置也能出来，所以 baseline 不是完全不能用。

但它的问题也比较清楚：人脸、头发、手这些局部细节不够。头脸区域更像是稀疏的点云壳，鼻子、嘴、眼睛、发际线没有稳定立起来。手部也只是有大概位置，手指和手腕连接并不稳定。

数值上，早期 full-image VGGT baseline 在同一个 6-view case 下，full-body 点数约 `40.9k`，head ROI 约 `9.0k`，face ROI 约 `4.2k`。这说明它能给粗全身，但离导师说的“完整呈现整个人，头部、面部、头发最重要”还有距离。

导师提到“人占比太小，看不清细节，crop 有道理”，这点在 baseline 对比里是成立的。做 human crop 后，full-body 点数从约 `40.9k` 提到约 `111.1k`，head 从约 `9.0k` 提到约 `24.4k`，face 提到约 `9.6k-11.5k`。所以 crop 这条输入处理线应该保留，它确实让人体区域更大、点云更多。

但也不能只看点数。点数增加不代表脸真的建好了。soft matte 点数更多，但全局几何有时更不稳。所以 VGGT 原版 + crop 的结论是：输入处理有帮助，但还不足以解决头脸细节。

## 二、SMPL-X + VGGT

SMPL-X 加进来以后，最直接的好处是给了人体结构先验。它知道人体大概在哪里，头、身体、手大概在哪里，也能提供姿态对齐后的 depth、point、normal、visibility 这些提示。对于全身结构来说，这比完全让 VGGT 从 6 张图里盲猜更合理。

这也回答了导师说的“你现在都有 SMPL-X 参数了，对人体几何约束应该好很多”这一点。确实，SMPL-X 对人体大结构有帮助，尤其是全身、头部位置、手部大概区域这些地方。

但导师担心的另一个问题也存在：SMPL-X 的模板感太强。它本质上是参数化人体模板，如果权重太强，结果会往模板壳收缩。这样身体可能有 SMPL-X 模板感，裙子、衣物、头发、个人脸部特征会被压掉。导师说“SMPL-X 模板特征太强，有时候反而是坏事”，这一点是对的。

所以现在不能把 SMPL-X 当最终 teacher，也不能说 SMPL-X 本身就是真实人体几何。我们现在的做法是把它当作弱的人体结构先验：帮助定位人体、分区、手和头脸的大概结构，但最终细节仍然要靠图像特征、depth/point 输出和 normal 几何一致性来决定。

当前 SMPL-X + VGGT 的实际效果是：全身结构和候选包闭环比原来更完整，normal 和 region evidence 也更清楚；但如果只看主 point map 坐标，它相对 base VGGT 的变化很小。当前 prior-enabled VGGT 相比 base VGGT 的全像素 point-map mean L2 只有 `0.00053544`，现版 candidate 主点图和 prior-enabled 输出是对齐的。也就是说，SMPL-X + VGGT 主要补齐的是结构约束、normal 证据、分区证据和 candidate package，而不是已经让所有局部细节明显变好。

这部分对导师可以这样说：SMPL-X 有用，但不能让它太强；它适合作为人体结构约束，不适合作为脸、头发、衣物的硬模板答案。

## 三、当前 normal 路线

导师录音里对 normal 的要求很具体：depth 和 normal 不能是两个互相独立的输出。depth 可以通过相邻像素差分算出局部方向，也就是 normal；网络输出的 normal 应该和这个由 depth / point map 推出来的 normal 保持一致。否则 normal 可能只是学到一张 2D 法向图，并没有真正增强几何。

当前 normal 路线就是按这个要求做的，不是只让模型多输出一张 normal 图。具体做法是：

1. VGGT 输出 depth 和 point map。
2. 用 depth / point map 反算出几何 normal。
3. 让网络输出的 normal 和这个几何 normal 做一致性约束。
4. 再检查 depth 反投影出来的点云和 point map 本身是否一致。

所以这条线在细节上是按导师说的思路落实的：depth、point、normal 之间有几何关系，不是三个分支各自独立输出。

效果上，normal 路线确实改善了一部分几何一致性。历史 normal/self-geometry 实验里，head / face 区域的 normal-depth、normal-point、depth-point 一致性有改善。这说明 normal 不是没用，它确实能让输出更自洽。

但问题是，这种自洽还没有稳定变成清晰的人脸点云。也就是说，normal 让 depth、point、normal 之间更协调，但它还没有单独解决“6-view 下 head / face 表面不够连续、不够清楚”的问题。之前 normal 路线里，近看头脸仍然偏 shell-like，鼻子、嘴、眼睛、发际线没有稳定成型，手部也仍然容易碎。

所以当前 normal 路线的结论是：方向对，代码也按导师要求做了；它是必要约束，但不是充分条件。后面不能只继续加 normal loss，而是要让 normal 约束真正作用到 target-view 的 point map 上，尤其是头脸、头发、手这些 ROI。

## 四、三组结果放在一起看

如果按导师现在的问题来总结，三组结果可以这样对比：

| 对比项 | VGGT 原版 baseline | SMPL-X + VGGT | 当前 normal 路线 |
|---|---|---|---|
| 全身 | 能出粗全身，但细节弱 | 全身结构更有约束，人体区域更清楚 | 可用于检查几何是否自洽，但单独不改变全身结构很多 |
| 头部 / 面部 / 头发 | 点云薄，不够连续，脸部细节弱 | 有头脸区域先验，但容易有模板感，不能当真实脸 | normal 一致性有改善，但还没稳定形成清晰人脸几何 |
| 手部 | 有基础位置，但手腕/手指不稳 | SMPL-X 有手部结构，能提供大概区域 | 手部 normal/point 仍弱，右手尤其需要继续做 |
| crop / 人占比 | 原图人占比小，细节吃亏 | 仍需要 crop 配合，不然图像细节不够 | normal 也需要 ROI/crop，不然局部监督太弱 |
| depth / point / normal 关系 | 原版主要看 depth/point 输出 | SMPL-X 提供 prior depth/normal，但不能硬压 | 已做耦合：depth/point 反算 normal，再约束输出 normal |
| 当前能否作为最终结果 | 不够 | 可作为结构先验路线，但不能说 teacher 完成 | 有技术改进，但还没单独达到导师想看的清晰头脸点云 |

## 五、给导师的结论口径

现在最稳妥的说法是：

VGGT 原版 baseline 能恢复粗全身，但头脸、头发和手部细节明显不够。crop 能提升人体区域占比和 ROI 点数，所以这条输入处理线要保留。SMPL-X + VGGT 对全身结构有帮助，也能给手和头脸提供大概区域，但 SMPL-X 不能太强，否则会有模板感，脸和衣物不像真人。当前 normal 路线已经按导师说的方式，把 depth、point、normal 耦合起来，也改善了一部分几何一致性；但这些改善还没有稳定转化成清晰连续的人脸、头发和手部点云。

下一步应该不是继续堆新名词，而是围绕导师说的几个点继续做：crop 让人占比更合适；normal 和 depth/point 继续耦合；SMPL-X 只作为弱结构先验；最终检查要回到 Open3D 点云里，看全身、头脸、头发、手是不是视觉上真的更完整、更有立体感。
"""
    OUT_MD.write_text(text, encoding="utf-8")
    (SEND / OUT_MD.name).write_text(text, encoding="utf-8")
    payload = {
        "task": "mentor_plain_three_way_vggt_smplx_normal_comparison",
        "created_utc": now(),
        "status": "DONE_PASS",
        "report_md": str(OUT_MD),
        "send_copy": str(SEND / OUT_MD.name),
        "comparison_groups": ["VGGT baseline", "SMPL-X + VGGT", "current normal route"],
        "style": "mentor_plain_language_following_recording",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (SEND / OUT_JSON.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
