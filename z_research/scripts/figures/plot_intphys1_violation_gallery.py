#!/usr/bin/env python3
"""IntPhys 1 dev 위반 시점 갤러리 (2026-10-17) — 행 = 쌍 (불가능 영상), 열 = 위반 시점 d 주변 프레임.

d = 가능 · 불가능 영상이 처음 픽셀로 달라지는 raw 프레임 (0-based, 15 fps). 출처 z_research/previous/OcclusionStats/intphys1/pairs.csv
(정본 Archive/INTPHYS1_OCCLUSION_2026-09-26.md — 180 쌍 모두 first_div_sensitive − 1 과 같다).
열 = d + {−10, −8, …, +10} (빨간 테두리 = d). 영상 범위 밖이면 빈 칸. 그룹 (moving/occluded · static/occluded · moving/visible · static/visible) 마다 PNG 하나.
행 라벨: 불가능 영상 id · d · event_kind (gap_contains_d = d 가 가림 구간 안 / not_seen_before_d = d 전에 물체가 안 보임 …).

  python z_research/scripts/figures/plot_intphys1_violation_gallery.py
  → z_research/previous/OcclusionStats/intphys1/figures/gallery_violation_<group>.png
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
PAIRS = REPO / "z_research/previous/OcclusionStats/intphys1/pairs.csv"
OUT = REPO / "z_research/previous/OcclusionStats/intphys1/figures"
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
OFFS = list(range(-10, 11, 2))
TH, LEFT, TOP, GAP = 96, 230, 44, 4
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 13); FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def frame(vid, t):
    p, s, r = vid.split("_")
    f = FR / p / s / r / "scene" / f"scene_{t + 1:03d}.png"
    return Image.open(f).convert("RGB").resize((TH, TH)) if f.exists() else None


def main():
    rows = list(csv.DictReader(open(PAIRS)))
    groups = sorted({r["group"] for r in rows})
    OUT.mkdir(parents=True, exist_ok=True)
    for g in groups:
        rs = sorted([r for r in rows if r["group"] == g], key=lambda r: (r["event_kind"], int(r["d"]), r["imp"]))
        W = LEFT + len(OFFS) * (TH + GAP); H = TOP + len(rs) * (TH + GAP) + 10
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"IntPhys 1 dev · {g} · impossible clip around violation frame d (red) · n={len(rs)} pairs", fill=(0, 0, 0), font=FONT_B)
        for j, o in enumerate(OFFS):
            dr.text((LEFT + j * (TH + GAP) + TH // 2 - 14, TOP - 18), "d" if o == 0 else f"d{o:+d}", fill=(200, 0, 0) if o == 0 else (0, 0, 0), font=FONT)
        for i, r in enumerate(rs):
            y = TOP + i * (TH + GAP); d = int(r["d"])
            dr.multiline_text((8, y + 12), f"{r['imp']}  d={d} ({d / 15:.1f}s)\n{r['event_kind']}\nskip2_w32 correct={r['correct_skip2_w32']}", fill=(0, 0, 0), font=FONT, spacing=3)
            for j, o in enumerate(OFFS):
                x = LEFT + j * (TH + GAP); fr = frame(r["imp"], d + o) if 0 <= d + o < 100 else None
                if fr is not None: im.paste(fr, (x, y))
                else: dr.rectangle([x, y, x + TH - 1, y + TH - 1], fill=(235, 235, 235))
                if o == 0: dr.rectangle([x - 2, y - 2, x + TH + 1, y + TH + 1], outline=(220, 0, 0), width=3)
        f = OUT / f"gallery_violation_{g.replace('/', '_')}.png"; im.save(f); print("saved", f, im.size)


if __name__ == "__main__":
    main()
