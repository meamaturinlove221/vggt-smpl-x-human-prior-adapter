from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEND = ROOT / "output" / "mentor_report_v50r2" / "send_to_mentor_v42_consistent"
OUT_MD = REPORTS / "20260511_mentor_plain_vggt_baseline_vs_normal_route.md"
OUT_JSON = REPORTS / "20260511_mentor_plain_vggt_baseline_vs_normal_route.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    SEND.mkdir(parents=True, exist_ok=True)
    text = """# 按导师原话整理：VGGT baseline 和当前 normal 路线结果

这部分不再按内部版本号讲，只分成两件事：第一，直接跑 VGGT baseline 的点云是什么效果；第二，我们现在加 normal 之后，点云有没有真正改善。

## 一、VGGT baseline 的效果

直接用 VGGT 做这个 6-view 人体点云，能出来一个基本的人体形状。全身轮廓是能看的，身体、头部、手的大概位置也能恢复出来，所以 baseline 不是完全失败。

但问题也比较明显：它更像一个粗的 point map / depth 反投影结果，细节不够。导师关心的头部、面部、头发这些区域，baseline 里没有稳定清楚的人脸几何；脸部点云比较薄，局部连续性不够，鼻子、嘴、眼睛和发际线这些结构没有真正立起来。手部也是类似，有基础位置，但细节和连接关系不稳定。

从数值上看，早期 full-image VGGT baseline 在同一个 6-view case 下，full-body 点数大约是 `40.9k`，head ROI 大约是 `9.0k`，face ROI 大约是 `4.2k`。这说明 baseline 可以给一个粗人体，但如果按导师说的“完整呈现整个人，头部、面部、头发更重要”这个标准，它还不够。

后面做 crop 以后，点数确实明显上来了。human crop 后 full-body 点数到了约 `111.1k`，head 到约 `24.4k`，face 到约 `9.6k-11.5k`。这和导师说的“人占比太小，看不清细节，所以 crop 有道理”是对上的。也就是说，crop 这条线是有效的输入处理，它让人体区域占比更大，点云数量和可见区域变多。

但这里也要实话说：点数变多不等于脸真的建好了。soft matte 这一类结果点数更多，但有时全局几何会更不稳。所以 crop 可以保留，它是有用的 baseline 改进，但不能单独当作最终答案。

## 二、当前 normal 路线做到什么

导师录音里提到的核心点是：depth 和 normal 不能是两个互相独立的输出。depth 可以通过相邻像素差分算出局部方向，也就是 normal；网络输出的 normal 应该和这个由 depth / point map 推出来的 normal 保持一致。这样 normal 才不是只学一张看起来像法向图的 2D 图，而是要真的帮助几何。

我们现在 normal 这条线就是按这个思路做的。不是只让网络多输出一张 normal 图，而是把 depth、point map、normal 放到同一个几何关系里看：

1. VGGT 输出 depth 和 point map。
2. 用 depth / point map 反算出几何 normal。
3. 让网络输出的 normal 和这个几何 normal 做一致性约束。
4. 再检查 depth 反投影出来的点云和 point map 本身是否一致。

所以这条线是落实了导师说的“depth 可以转 normal，normal 和 depth / point 要耦合”这个要求的。

实验结果上，normal 路线确实让一部分几何一致性指标变好了。比如 head / face 区域里，预测 normal 和由 depth / point 推出来的 normal 更接近，depth 和 point 之间也更自洽。这说明 normal 不是完全没用，它对几何约束是有帮助的。

但目前最大的问题是：这些一致性改善还没有稳定转化成导师想看的清晰头脸点云。也就是说，normal 让输出之间更协调，但它没有单独解决“6-view 下人脸表面不够连续、不够清楚”的问题。之前 normal/self-geometry 的结果里，face/head 近看还是偏 shell-like，脸部没有稳定的鼻子、眼睛、嘴和发际线几何。

所以现在对 normal 线的判断应该是：方向是对的，代码上也按导师说的细节做了；它是必要约束，但还不是充分条件。它不能单独把 baseline 变成高质量人体点云。

## 三、现在能给导师看的结论

当前最稳妥的结论是：

VGGT baseline 能恢复粗全身，但头脸和手部细节不够。crop 能显著增加人体区域点数，说明输入预处理是有效的。normal 路线已经按导师说的方式，把 depth、point 和 normal 耦合起来，也改善了一部分几何一致性；但是它还没有让 6-view 的 head / face 点云达到清晰、连续、稳定的人脸几何。

所以后面应该继续保留两条线：

第一条是 crop 这条输入处理线，因为它确实让人体占比更合适。

第二条是 normal 这条几何约束线，但不能只做 normal 图本身，而要继续让 normal 和 depth 反投影、point map、局部 ROI 几何一起优化。

下一步真正需要突破的是：让这些 normal / depth / point 的一致性，直接反映到 target-view 的头脸点云上。也就是不仅指标变好，还要 Open3D 里能看到面部、头发、手部这些区域的几何真的更连续、更有立体感。
"""
    OUT_MD.write_text(text, encoding="utf-8")
    (SEND / OUT_MD.name).write_text(text, encoding="utf-8")
    payload = {
        "task": "mentor_plain_vggt_baseline_vs_normal_route",
        "created_utc": now(),
        "status": "DONE_PASS",
        "report_md": str(OUT_MD),
        "send_copy": str(SEND / OUT_MD.name),
        "style": "mentor_plain_language_no_internal_version_jargon",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (SEND / OUT_JSON.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
