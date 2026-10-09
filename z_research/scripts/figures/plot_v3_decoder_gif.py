#!/usr/bin/env python3
"""RollOut v3 — **h 로 학습한 자**로 읽은 p 와 h 의 위치를 영상 위에 겹친다 (시나리오 8 개를 한 GIF 에, 2 × 4).

  흰 고리 = 진실 (튜블릿 단위)        × = 마지막으로 본 자리 (문맥 끝)
  주황 점 = predictor p 를 h 로 학습한 자로 읽은 위치 (`p@h`)
  파랑 점 = h (target encoder, 창 전체를 본다) 를 같은 자로 읽은 위치 — 자가 실제 프레임에서 읽는 값
  '있다' 일 때만 점을 찍고, '없다' 면 글자로 적는다. 문턱 · 좌표 치우침은 h 의 것 (p 기준 재보정은 안 했다).
  아래 막대 = 진회색 문맥 · 밝은 회색 미래 · ▲ 지금. 창 전체 프레임을 **같은 속도로** 보여 준다 (멈춤 없음).
  clip 은 seed 로 한 번 고르고 모든 창에 같은 것을 쓴다 — 창끼리 비교할 수 있게.

⚠️ 점 위치는 자의 읽기다 — attention 이 퍼지면 기본값 쪽으로 끌린다 (figures/README.md §2, CLAUDE.md §7-3).
⚠️ p 를 h 자로 읽으면 y 가 고정으로 어긋난다 (flat·wall t0 부터 약 20 px) — h 로 잰 보정이 p 에 안 맞는다.
읽은 값은 `exp_results/windows/readings.npz` (자 지문 확인). GPU 불필요.

  $P z_research/scripts/figures/plot_v3_decoder_gif.py                          # P ≤ 16 인 창 12 개
  $P z_research/scripts/figures/plot_v3_decoder_gif.py --windows 16x32 32x32
  → z_research/RollOutV3/figures/v3_gif/gif_C{c}_P{p}.gif
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, check      # noqa: E402

SRC = Path("/data2/local_datasets/world/world_analysis/RollOut_v3/Images")
INDEX = ROOT / "data_csv/rollout_v3/index.csv"
OUT = ROOT / "z_research/RollOutV3/figures/v3_gif"
RES, R, Z, SPLIT, MS = 144.0, 288, 1.0, 32, 150          # MS = 프레임당 ms (모두 같다)
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
WHITE, ORANGE, BLUE, FUT, YEL, GRAY, CTX = (255, 255, 255), (235, 104, 52), (42, 120, 214), (215, 215, 215), (255, 230, 90), (170, 170, 170), (95, 95, 95)


def tile(fname, f, c, p, truth, reads, law):
    """프레임 f 의 한 칸. truth (64, 2) px, reads = [(색, 이름, xy (tp,2) px, say (tp,))]."""
    W = int(R * Z); HEAD, BAR = 34, 26
    im = Image.open(SRC / fname / f"{f:06d}.png").convert("RGB").resize((W, W), Image.NEAREST)
    cv = Image.new("RGB", (W, W + HEAD + BAR), (20, 20, 20)); cv.paste(im, (0, HEAD)); d = ImageDraw.Draw(cv)
    P = lambda xy: (float(xy[0]) * Z, float(xy[1]) * Z + HEAD)
    s0 = SPLIT - c
    tt = truth.reshape(32, 2, 2).mean(1)                                     # 튜블릿 단위 진실
    pts = [P(tt[q // 2]) for q in range(s0, f + 1)]
    if len(pts) > 1: d.line(pts, fill=WHITE, width=1)
    x, y = pts[-1]; d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=WHITE, width=2)
    status = []
    if f >= SPLIT:
        lx, ly = P(tt[(SPLIT - 1) // 2]); d.line([lx - 5, ly - 5, lx + 5, ly + 5], fill=WHITE, width=2); d.line([lx - 5, ly + 5, lx + 5, ly - 5], fill=WHITE, width=2)
        j = (f - SPLIT) // 2
        for col, nm, xy, say in reads:
            tr = [P(xy[q]) for q in range(j + 1) if say[q]]
            if len(tr) > 1: d.line(tr, fill=col, width=2)
            if say[j]:
                x, y = P(xy[j]); d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=col, outline=WHITE, width=2)
            status.append((col, f"{nm}: {'present' if say[j] else 'ABSENT'}"))
    d.text((5, 3), f"{law}   frame {f}  ({'context' if f < SPLIT else 'future t' + str((f - SPLIT) // 2)})", fill=WHITE)
    xo = 5
    for col, txt in status:
        d.text((xo, 18), txt, fill=col if "present" in txt else YEL); xo += 8 * len(txt) + 10
    y0 = HEAD + W + 6; x0, x1 = 6, W - 6; n = c + p; seg = (x1 - x0) / n
    for q in range(n):
        d.rectangle([x0 + q * seg, y0, x0 + (q + 1) * seg - 1, y0 + 7], fill=CTX if q < c else FUT)
    cx = x0 + (f - s0 + 0.5) * seg; d.polygon([(cx, y0 + 8), (cx - 5, y0 + 15), (cx + 5, y0 + 15)], fill=WHITE)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="*", default=None, metavar="CxP", help="기본: P ≤ 16 인 창 12 개")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    wins = ([tuple(int(v) for v in w.lower().split("x")) for w in a.windows] if a.windows
            else [(c, p) for c in (4, 8, 16, 32) for p in (4, 8, 16)])
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text()); thr = S["reps"]["h"]["attn"]["thr_val_fpr5"]
    bias = np.array(json.loads((PRES / "attn_bias_px.json").read_text())["h"])
    fname = {r["video_id"]: r["file_name"] for r in csv.DictReader(INDEX.open())}
    POS = np.where(z["role"] == "roll")[0]; rng = np.random.default_rng(a.seed)
    picks = [int(rng.choice(POS[z["scenario"][POS] == law])) for law in LAWS]          # 모든 창에 같은 clip
    OUT.mkdir(parents=True, exist_ok=True)
    marks = [ORANGE, BLUE, WHITE, YEL, FUT, GRAY, CTX, (20, 20, 20), (0, 0, 0)]

    def quant(fr):
        # **프레임마다** 팔레트를 따로 (영상 색 + 표시색을 정확히). 2026-09-24:
        #   전체 공통 MEDIANCUT → 주황 점이 배경 황갈색과 섞여 갈색이 됐다
        #   공통 240 색 + 표시색 고정 → 작은 물체 색이 밀려났다 (분홍 물체가 흰색·주황으로)
        nb = 256 - len(marks)
        base = fr.quantize(colors=nb, method=Image.Quantize.FASTOCTREE).getpalette()[:nb * 3]
        pl = base + [v for cl in marks for v in cl]; pl += [0] * (768 - len(pl))
        pal = Image.new("P", (1, 1)); pal.putpalette(pl)
        return fr.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)

    for c, p in wins:
        def reads(i):
            out = []
            for key, col, nm in (("p@h", ORANGE, "p"), ("h", BLUE, "h")):
                v = z[f"C{c}_P{p}_{key}"][i]
                out.append((col, nm, (v[:, 0:2] + 1) * RES - bias, v[:, 2] > thr))
            return out
        frames = []
        for f in range(SPLIT - c, SPLIT + p):
            tiles = [tile(fname[str(z["video_id"][i])], f, c, p, (z["truth"][i] + 1) * RES, reads(i), law) for i, law in zip(picks, LAWS)]
            w, h = tiles[0].size
            cv = Image.new("RGB", (w * 4 + 6 * 3, h * 2 + 6), (0, 0, 0))
            for q, t_ in enumerate(tiles):
                cv.paste(t_, ((q % 4) * (w + 6), (q // 4) * (h + 6)))
            frames.append(cv)
        q_ = [quant(fr) for fr in frames]
        path = OUT / f"gif_C{c}_P{p}.gif"
        q_[0].save(path, save_all=True, append_images=q_[1:], duration=MS, loop=0)
        print(path)
    print("clips:", [str(z["video_id"][i]) for i in picks])


if __name__ == "__main__":
    main()
