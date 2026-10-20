#!/usr/bin/env python3
"""render_predictor.py 가 만든 GIF 들을 표 한 장으로 (2026-10-20, 사용자 요청: 열 = 프레임, 행 = 샘플).
  /venv/vjepa/bin/python z_pixel_decoder/grid_from_gifs.py z_pixel_decoder/runs/vobench_vits/renders/visible [--cols 15,16,18,...] [--size 128]
샘플마다 두 줄: 원본 · 릴리즈 predictor (GIF 의 왼쪽 · 가운데 칸). 열은 32 장 중 고른 장 (기본: 문맥 마지막 15 + 미래 16..31 짝수)."""
import argparse, glob, os
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--cols", default="13,15,16,18,20,22,24,26,28,30,31")
ap.add_argument("--size", type=int, default=128); a = ap.parse_args()
cols = [int(c) for c in a.cols.split(",")]; s = a.size; L = 230; top = 18
gifs = sorted(glob.glob(os.path.join(a.dir, "*.gif")))
W, H = L + s * len(cols), top + 2 * s * len(gifs)
sheet = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
for j, t in enumerate(cols):
    dr.text((L + j * s + 3, 3), f"t{t} (f{3 * t})" + (" ctx" if t < 16 else ""), fill=(0, 0, 0) if t >= 16 else (110, 110, 110))
for i, g in enumerate(gifs):
    im = Image.open(g); name = os.path.basename(g)[:-4].split("__")
    for r, (panel, lab) in enumerate(((0, "original"), (1, "release pred"))):
        y = top + (2 * i + r) * s
        dr.text((3, y + 3), "/".join(name[:2]) if r == 0 else "", fill=(0, 0, 0)); dr.text((3, y + 17), name[3][:30] if r == 0 else "", fill=(60, 60, 60))
        dr.text((3, y + s - 16), lab, fill=(0, 0, 160) if r else (0, 0, 0))
        for j, t in enumerate(cols):
            im.seek(t); fr = im.convert("RGB").crop((panel * 256, 0, panel * 256 + 256, 256)).resize((s, s))
            sheet.paste(fr, (L + j * s, y))
        if r == 1:
            dr.line([(0, y + s - 1), (W, y + s - 1)], fill=(200, 200, 200))
out = os.path.join(a.dir, "grid.png"); sheet.save(out); print(out, sheet.size)
