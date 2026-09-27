#!/usr/bin/env python3
"""RollOut v3 — 16 개 (context, prediction) 창을 **4x4 격자 영상**으로.

열 = 예측 길이 P ∈ {4,8,16,32}, 행 = 문맥 길이 C ∈ {4,8,16,32}.
사건은 f32 (`semantic_event_frame`), 문맥 `[32-C, 32)`, 예측 `[32, 32+P)`.
각 칸은 **자기 창을 잘라서** 튼다 — 창이 시작하기 전에는 첫 프레임, 끝난 뒤에는 **마지막 예측 프레임에 멈춘다**.
그래서 P=4 칸이 f35 에서 얼어붙어 있는 동안 P=32 칸만 계속 가는 게 그대로 보인다.
테두리: 문맥 파랑 · 예측 주황 · 멈춤 회색.

`wall`·`ledge` 는 가능·불가능이 문맥을 공유하므로 (f32 까지 픽셀 동일, f33 부터 갈라짐)
가능 프레임 위에 불가능 위치를 빨간 테두리로 얹고, 칸마다 그 창의 최대 분리량을 px 로 적는다.
**자 한 칸 = 18 px** 이라 그보다 작은 창은 라벨이 빨갛다 (= 자로는 판정 불가).

출력: z_research/RollOutV3/figures/windows_gif/<scenario>.gif

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_rollout3_windows_gif.py
  $P z_research/scripts/figures/plot_rollout3_windows_gif.py --scenarios wall ledge --speed 10.5 --panel 200
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import matplotlib.font_manager as fm

DATA = Path("/data2/local_datasets/world/world_analysis/RollOut_v3")
import sys; sys.path.insert(0, "/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/analysis")   # noqa: E702
from rollout3_paths import fig   # noqa: E402  (R3_OUT 로 폴더를 고른다. 내용은 자와 무관하다)
OUT = fig("windows_gif")
NF, SPLIT, CELL, SRC = 64, 32, 18.0, 288
CTX, PRD = (4, 8, 16, 32), (4, 8, 16, 32)
BG, INK, MUTED, DIM = (18, 18, 22), (240, 240, 240), (150, 150, 160), (70, 70, 78)
BLUE, ORANGE, GREEN, RED = (42, 120, 214), (235, 104, 52), (27, 175, 122), (220, 70, 70)
F = lambda s, b=False: ImageFont.truetype(
    fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if b else "normal")), s)


def pick(scenario, speed):
    rs = [x for x in csv.DictReader((DATA / "metadata.csv").open()) if x["scenario"] == scenario]
    e = min((x for x in rs if x["role"] == "roll"), key=lambda x: (abs(float(x["primary"]) - speed), x["name"]))
    imp = next((x for x in rs if x["role"] == "impossible_roll" and x["block"] == e["block"]), None)
    return e, imp


def track(e):
    a = lambda s: np.array([float(v) for v in s.split()], np.float32)
    return a(e["object_px_x_by_sample"]), a(e["object_px_y_by_sample"]), a(e["in_frame_by_sample"]) > 0


def fpath(e, i):
    return DATA / "Images" / e["scenario"] / "roll" / e["name"] / f"{i:06d}.png"


def build(scenario, speed, ms, PS):
    e, imp = pick(scenario, speed)
    px, py, inf = track(e)
    sep = None
    if imp is not None:
        qx, qy, _ = track(imp)
        sep = np.hypot(px - qx, py - qy)
    s = PS / SRC                                             # 원본 288 → 패널
    GX, GY, HEAD, LAB = 8, 26, 78, 40                        # 칸 간격 · 헤더 · 행 라벨 폭
    W = LAB + 4 * PS + 3 * GX + 24
    H = HEAD + 4 * PS + 3 * GY + 54   # 마지막 줄 라벨 + 범례 자리
    f9, f10, f11 = F(9), F(10), F(11)

    cache = {}
    def panel(j):
        if j not in cache:
            cache[j] = Image.open(fpath(e, j)).convert("RGB").resize((PS, PS), Image.BILINEAR)
        return cache[j]

    out = []
    for i in range(NF):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((14, 10), f"RollOut v3 · {scenario}", font=F(17, True), fill=INK)
        d.text((14, 33), f"{e['name']}   speed {e['primary']} px/2f   env {e['env']}   "
                         f"event @ f{SPLIT}   1 cell = 18 px", font=f10, fill=MUTED)
        d.text((W - 210, 12), f"frame {i:02d} / 63", font=F(15, True), fill=INK)
        d.text((W - 210, 34), "CONTEXT" if i < SPLIT else "PREDICTION", font=F(12, True),
               fill=BLUE if i < SPLIT else ORANGE)
        for c, p in enumerate(PRD):
            d.text((LAB + c * (PS + GX) + PS // 2 - 16, HEAD - 17), f"P = {p}", font=F(12, True), fill=INK)

        for r, C in enumerate(CTX):
            y = HEAD + r * (PS + GY)
            d.text((6, y + PS // 2 - 6), f"C={C}", font=F(12, True), fill=INK)
            for c, P in enumerate(PRD):
                x = LAB + c * (PS + GX)
                lo, hi = SPLIT - C, SPLIT + P - 1
                j = min(max(i, lo), hi)                  # 창 밖이면 양 끝 프레임에서 멈춘다 (자르기)
                lit = lo <= i <= hi
                im.paste(panel(j), (x, y))
                dd = ImageDraw.Draw(im)
                if inf[j]:
                    rr = max(4, int(PS * 0.030))
                    dd.ellipse([px[j]*s+x-rr, py[j]*s+y-rr, px[j]*s+x+rr, py[j]*s+y+rr],
                               fill=GREEN if imp is not None else (255, 255, 255), outline=(255, 255, 255))
                    if imp is not None and j > SPLIT:
                        r2 = rr + 4
                        dd.ellipse([qx[j]*s+x-r2, qy[j]*s+y-r2, qx[j]*s+x+r2, qy[j]*s+y+r2],
                                   outline=RED, width=2)
                col = (BLUE if j < SPLIT else ORANGE) if lit else DIM
                dd.rectangle([x, y, x + PS - 1, y + PS - 1], outline=col, width=3)
                dd.text((x + PS - 34, y + 4), f"f{j:02d}", font=F(11, True), fill=col if lit else MUTED)
                if not lit:
                    dd.text((x + 5, y + 4), "HOLD", font=F(11, True), fill=MUTED)
                tag = f"f{SPLIT-C}..{SPLIT-1} | f{SPLIT}..{SPLIT+P-1}"
                if sep is not None:
                    m = sep[SPLIT:SPLIT + P].max()
                    dd.text((x + 1, y + PS + 3), tag, font=f9, fill=MUTED if lit else DIM)
                    dd.text((x + PS - 52, y + PS + 3), f"{m:5.1f}px", font=F(9, True),
                            fill=(GREEN if lit else (30, 90, 64)) if m > CELL else (RED if lit else (90, 38, 38)))
                else:
                    dd.text((x + 1, y + PS + 3), tag, font=f9, fill=MUTED if lit else DIM)
        if sep is not None:
            d.text((14, H - 26), f"●  possible (rendered)      ○  impossible (shares the context)      "
                                 f"|pos - imp| at this frame = {sep[i]:5.1f} px", font=f11,
                   fill=INK if sep[i] > CELL else MUTED)
        else:
            d.text((14, H - 26), "possible only (this law has no impossible partner)", font=f11, fill=MUTED)
        out.append(im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{scenario}.gif"
    out[0].save(f, save_all=True, append_images=out[1:], duration=ms, loop=0, optimize=True)
    tail = "  sep " + " ".join(f"P{p}:{sep[SPLIT:SPLIT+p].max():.1f}" for p in PRD) if sep is not None else ""
    print(f"→ {f.name}  {W}x{H}  {f.stat().st_size/2**20:.1f} MB  ({e['name']}){tail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*",
                    default=["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "wall", "ledge"])
    ap.add_argument("--speed", type=float, default=8.25)
    ap.add_argument("--ms", type=int, default=110)
    ap.add_argument("--panel", type=int, default=176, help="칸 한 변 px (원본 288)")
    a = ap.parse_args()
    for s in a.scenarios:
        build(s, a.speed, a.ms, a.panel)
