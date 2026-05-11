from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "output" / "mentor_report_v50r2" / "images"
REPORTS = ROOT / "reports"
PNG_MAIN = IMG_DIR / "05_vggt_smplx_native_architecture.png"
PNG_ALIAS = IMG_DIR / "08_vggt_smplx_native_architecture.png"
SVG_ALIAS = IMG_DIR / "08_vggt_smplx_native_architecture.svg"
JSON_REPORT = REPORTS / "20260509_v50r2_architecture_diagram.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 3) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=width)


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    *,
    title_size: int = 27,
    sub_size: int = 18,
    fill: str = "#111827",
) -> None:
    x0, y0, x1, y1 = box
    f1 = font(title_size, bold=True)
    f2 = font(sub_size)
    lines = [title]
    lines += textwrap.wrap(subtitle, width=24)
    heights = []
    widths = []
    for i, line in enumerate(lines):
        f = f1 if i == 0 else f2
        b = draw.textbbox((0, 0), line, font=f)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total = sum(heights) + 10 * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for i, line in enumerate(lines):
        f = f1 if i == 0 else f2
        x = x0 + (x1 - x0 - widths[i]) / 2
        draw.text((x, y), line, font=f, fill=fill)
        y += heights[i] + 10


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#4B5563") -> None:
    draw.line((start, end), fill=color, width=4)
    ex, ey = end
    sx, sy = start
    if ex >= sx:
        pts = [(ex, ey), (ex - 14, ey - 9), (ex - 14, ey + 9)]
    else:
        pts = [(ex, ey), (ex + 14, ey - 9), (ex + 14, ey + 9)]
    draw.polygon(pts, fill=color)


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    w, h = 2400, 760
    img = Image.new("RGB", (w, h), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    title = "VGGT + SMPL-X Native Prior Candidate Route"
    subtitle = "single-row view, based on original VGGT pipeline; no stacked layers"
    draw.text((70, 34), title, font=font(36, bold=True), fill="#111827")
    draw.text((70, 84), subtitle, font=font(22), fill="#4B5563")

    y = 190
    box_h = 210
    boxes = [
        ((70, y, 300, y + box_h), "#F3F4F6", "#6B7280", "Input", "6-view RGB + mask"),
        ((340, y, 570, y + box_h), "#E0F2FE", "#0284C7", "DINO Tokens", "image patch tokens"),
        ((610, y, 840, y + box_h), "#FFE4E6", "#E11D48", "SMPL-X Prior", "prior maps + tokens"),
        ((880, y, 1110, y + box_h), "#FDE68A", "#D97706", "Camera Token", "original VGGT path"),
        ((1150, y, 1430, y + box_h), "#EDE9FE", "#7C3AED", "VGGT Backbone", "global + frame attention"),
        ((1470, y, 1700, y + box_h), "#DCFCE7", "#16A34A", "Dense Heads", "camera / depth / point / normal"),
        ((1740, y, 1970, y + box_h), "#ECFCCB", "#65A30D", "Self-Geom", "depth -> point -> normal consistency"),
        ((2010, y, 2240, y + box_h), "#DBEAFE", "#2563EB", "D-line Gate", "candidate / teacher split"),
    ]

    for box, fill, outline, label, text in boxes:
        rounded(draw, box, fill, outline)
        center_text(draw, box, label, text)

    for i in range(len(boxes) - 1):
        b0 = boxes[i][0]
        b1 = boxes[i + 1][0]
        arrow(draw, (b0[2] + 16, y + box_h // 2), (b1[0] - 16, y + box_h // 2))

    # SMPL-X branch cue is inline, not stacked: a thin annotation band below the single row.
    band = (70, 470, 2240, 610)
    draw.rounded_rectangle(band, radius=18, fill="#FAFAFA", outline="#D1D5DB", width=2)
    notes = [
        ("SMPL-X native evidence", "posed mesh, part map, visibility, weak depth / normal prior"),
        ("HumanPriorAdapter", "injects prior maps into the VGGT token path without replacing RGB evidence"),
        ("Self-supervised geometry", "depth unprojection, point-map normal, depth-normal and point-normal checks"),
        ("Output status", "strict candidate pass = 1; strict teacher pass = 0"),
    ]
    x = 105
    for header, body in notes:
        draw.text((x, 500), header, font=font(21, bold=True), fill="#111827")
        wrapped = textwrap.wrap(body, width=34)
        yy = 530
        for line in wrapped:
            draw.text((x, yy), line, font=font(17), fill="#374151")
            yy += 24
        x += 510

    footer = (
        "Key point: self-geometry improves consistency checks, but V50R2 remains a candidate pass, not an independent teacher pass."
    )
    draw.text((70, 675), footer, font=font(22), fill="#111827")

    img.save(PNG_MAIN)
    img.save(PNG_ALIAS)
    SVG_ALIAS.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"2400\" height=\"760\">"
        "<text x=\"70\" y=\"80\" font-size=\"36\">VGGT + SMPL-X Native Prior Candidate Route</text>"
        "<text x=\"70\" y=\"130\" font-size=\"22\">See PNG for the rendered single-row architecture diagram.</text>"
        "</svg>\n",
        encoding="utf-8",
    )
    report = {
        "task": "v223_make_vggt_smplx_architecture_diagram",
        "created_utc": now(),
        "status": "DONE_PASS",
        "png": PNG_MAIN.resolve().as_posix(),
        "png_alias": PNG_ALIAS.resolve().as_posix(),
        "svg_alias": SVG_ALIAS.resolve().as_posix(),
        "design": "single-row architecture, no stacked layer layout, bottom margin kept clear; includes explicit self-geometry consistency module",
    }
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
