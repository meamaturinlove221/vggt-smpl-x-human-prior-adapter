from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOC = ROOT / "output" / "mentor_report_v50r2"
IMG = DOC / "images"
RGB = DOC / "open3d_rgb_camera_view_pointcloud" / "images"
REPORT_JSON = REPORTS / "20260509_v50r2_rgb_pointcloud_final_image_list.json"
REPORT_MD = REPORTS / "20260509_v50r2_rgb_pointcloud_final_image_list.md"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    items = [
        ("01_full_body.png", IMG / "01_full_body.png", "main report full-body RGB point-cloud contact sheet"),
        ("02_head_face.png", IMG / "02_head_face.png", "main report head/face RGB point-cloud contact sheet"),
        ("03_hairline.png", IMG / "03_hairline.png", "main report head/hairline close RGB point-cloud view"),
        ("04_left_hand.png", IMG / "04_left_hand.png", "main report left-hand RGB point-cloud view"),
        ("05_right_hand.png", IMG / "05_right_hand.png", "main report right-hand RGB point-cloud view"),
        ("06_60view_support.png", IMG / "06_60view_support.png", "main report 60-view/support RGB point-cloud view"),
        ("07_temporal.png", IMG / "07_temporal.png", "main report temporal/support RGB point-cloud view"),
        ("08_vggt_smplx_native_architecture.png", IMG / "08_vggt_smplx_native_architecture.png", "architecture diagram based on original VGGT"),
        ("rgb_camera_view_full_body_contact_sheet.png", RGB / "rgb_camera_view_full_body_contact_sheet.png", "source Open3D RGB full-body point-cloud sheet"),
        ("rgb_camera_view_head_face_contact_sheet.png", RGB / "rgb_camera_view_head_face_contact_sheet.png", "source Open3D RGB head/face point-cloud sheet"),
        ("rgb_camera_view_hand_contact_sheet.png", RGB / "rgb_camera_view_hand_contact_sheet.png", "source Open3D RGB hand point-cloud sheet"),
        ("rgb_camera_view_human_pointcloud_report_sheet.png", RGB / "rgb_camera_view_human_pointcloud_report_sheet.png", "combined Open3D RGB point-cloud report sheet"),
    ]
    records = []
    for name, path, note in items:
        records.append({
            "name": name,
            "path": str(path.resolve()),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "note": note,
        })
    report = {
        "task": "v223_rgb_pointcloud_final_image_list",
        "created_utc": now(),
        "policy": "Main mentor report images now use RGB human point-cloud screenshots. Monochrome SMPL-X prior silhouette images are not used as main result figures.",
        "records": records,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# V50R2 Final RGB Point-Cloud Image List",
        "",
        "主汇报图已切换为 RGB 人体点云截图；单色 SMPL-X 先验轮廓不再作为主结果图。",
        "",
    ]
    for r in records:
        status = "exists" if r["exists"] else "missing"
        lines.append(f"- `{r['name']}` ({status})")
        lines.append(f"  - path: `{r['path']}`")
        lines.append(f"  - note: {r['note']}")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
