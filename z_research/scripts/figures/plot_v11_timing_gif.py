#!/usr/bin/env python3
"""IntPhysGen v11 — 가림 타이밍별로 p 의 미래를 나란히 본다: 열 = visible | 문맥 중간 가림 (mid) | 문맥 끝 가림 (late).

한 GIF = 한 운동 (static · flat · ramp) × 같은 k. 행마다 진행 방향을 맞춘 clip 을 한 줄씩 (vanish 의 pos_a, 물체가 내내 있다).
  흰 고리   = 진실 위치 (튜블릿 단위). 가려진 샘플에서는 회색 고리 (물체가 가림막 뒤 어디 있는지)
  ×        = 마지막으로 본 자리
  주황 점   = p 를 자로 읽은 위치 (미래만, p 가 '있다' 고 할 때만). '없다' 면 점을 안 찍고 글자로 적는다
  아래 막대 = 파랑 문맥 · 밝은 회색 미래 · 진회색 가려진 샘플 · ▲ 지금

⚠️ p 는 문맥 16 샘플만 본다. 읽은 값은 `v11_readout_online.py` 의 readings (자 지문 확인).
⚠️ 점 위치는 자의 읽기다 — 물체가 흐려지면 자가 기본값 쪽으로 끌린다 (CLAUDE.md §7-3). '있다/없다' 를 먼저 본다.

  $P z_research/scripts/figures/plot_v11_timing_gif.py                # static · flat · ramp, k=3, 2 줄
  $P z_research/scripts/figures/plot_v11_timing_gif.py --k 2 --rows 3
  $P z_research/scripts/figures/plot_v11_timing_gif.py --motions moving     # flat · ramp 를 한 파일에 (행 = flat 좌→우 · flat 우→좌 · ramp 좌→우 · ramp 우→좌)
  → figures/v11/gif_timing_<motion>_k<k>.gif
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, EXP, fig, check      # noqa: E402

FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
RES, R, Z = 144.0, 288, 1.25
SAMPLES = list(range(0, 94, 3))
WHITE, GRAYR, ORANGE, BLUE, FUT, HID, YEL = (255, 255, 255), (150, 150, 150), (235, 104, 52), (42, 120, 214), (215, 215, 215), (80, 80, 80), (255, 230, 90)
COND = {"static": ("static_visible", "static_occlusion_mid", "static_occlusion"),
        "flat": ("moving_visible_flat", "moving_occlusion_flat_mid", "moving_occlusion_flat"),
        "ramp": ("moving_visible", "moving_occlusion_mid", "moving_occlusion")}
COL_NAME = ("visible", "occluded mid-context (k={k})", "occluded at context end (k={k})")


def load(name):
    z = np.load(EXP / name / "readings.npz", allow_pickle=True); check(z, PRES); return z


def pick(z, cond, k, tdir, rng):
    m = ((z["condition"] == cond) & (z["sym_k"].astype(int) == k) & (z["violation_type"] == "vanish")
         & (z["variant"] == "pos_a") & z["obj"])
    if tdir != 0:
        m &= z["travel_dir"] == tdir
    idx = np.where(m)[0]
    return int(rng.choice(idx)) if len(idx) else None


def panel(z, i, fname, t, title, thr, bias):
    """샘플 t 의 한 칸 (PIL 이미지)."""
    W = int(R * Z); HEAD, BAR = 34, 30
    im = Image.open(FR / fname / f"{SAMPLES[t]:06d}.png").convert("RGB").resize((W, W), Image.NEAREST)
    c = Image.new("RGB", (W, W + HEAD + BAR), (20, 20, 20)); c.paste(im, (0, HEAD)); d = ImageDraw.Draw(c)
    P = lambda xy: (float(xy[0]) * Z, float(xy[1]) * Z + HEAD)
    truth = (z["truth"][i] + 1) * RES                                  # (32, 2) px
    tt = truth.reshape(16, 2, 2).mean(1); hid = z["hidden"][i]
    seen = [s for s in range(16) if not hid[s]]; ls = truth[seen[-1]] if seen else None
    # 진실 궤적 (튜블릿 단위)
    pts = [P(tt[s // 2]) for s in range(t + 1)]
    if len(pts) > 1: d.line(pts, fill=WHITE, width=1)
    x, y = pts[-1]; col = GRAYR if hid[t] else WHITE
    d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=col, width=2)
    if ls is not None and t >= 16:
        lx, ly = P(ls); d.line([lx - 5, ly - 5, lx + 5, ly + 5], fill=WHITE, width=2); d.line([lx - 5, ly + 5, lx + 5, ly - 5], fill=WHITE, width=2)
    # p (미래만)
    msg = "p: (reads the future only)"
    if t >= 16:
        v = z["p"][i]; say = v[:, 2] > thr; xy = (v[:, 0:2] + 1) * RES - np.array(bias)
        j = (t - 16) // 2
        tr = [P(xy[q]) for q in range(j + 1) if say[q]]
        if len(tr) > 1: d.line(tr, fill=ORANGE, width=3)
        if say[j]:
            x, y = P(xy[j]); d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=ORANGE, outline=WHITE, width=2)
            msg = "p: object PRESENT"
        else:
            msg = "p: object ABSENT"
    d.text((5, 3), title, fill=WHITE)
    phase = "context" if t < 16 else "future"
    d.text((5, 18), f"sample {t:2d} ({phase}){'  hidden' if hid[t] else ''}   {msg}",
           fill=ORANGE if "PRESENT" in msg else (YEL if "ABSENT" in msg else (190, 190, 190)))
    y0 = HEAD + W + 6; x0, x1 = 6, W - 6; seg = (x1 - x0) / 32
    for s in range(32):
        d.rectangle([x0 + s * seg, y0, x0 + (s + 1) * seg - 1, y0 + 8], fill=HID if hid[s] else (BLUE if s < 16 else FUT))
    cx = x0 + (t + 0.5) * seg; d.polygon([(cx, y0 + 9), (cx - 5, y0 + 17), (cx + 5, y0 + 17)], fill=WHITE)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--motions", nargs="+", default=["static", "flat", "ramp"], choices=["static", "flat", "ramp", "moving"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    zl, zm = load("v11"), load("v11_mid")
    S = json.loads((PRES / "summary.json").read_text()); thr = S["reps"]["p"]["attn"]["thr_val_fpr5"]
    bias = json.loads((PRES / "attn_bias_px.json").read_text())["p"]
    csv.field_size_limit(10 ** 9)
    fname = {r["video_id"]: r["file_name"] for r in csv.DictReader(INDEX.open())}
    rng = np.random.default_rng(a.seed); out = fig("v11"); out.mkdir(parents=True, exist_ok=True)
    for mo in a.motions:
        grid = []                                                        # 행마다 [(z, i, title)] x 3 열
        spec = ([("flat", 1), ("flat", -1), ("ramp", 1), ("ramp", -1)] if mo == "moving" else
                [(mo, 0 if mo == "static" else (1 if r % 2 == 0 else -1)) for r in range(a.rows)])
        for m_, tdir in spec:
            row = []
            for col, (cond, z, kk) in enumerate(zip(COND[m_], (zl, zm, zl), (0, a.k, a.k))):
                i = pick(z, cond, kk, tdir, rng)
                if i is None: sys.exit(f"clip 없음: {cond} k={kk} dir={tdir}")
                arrow = {1: "  (left to right)", -1: "  (right to left)", 0: ""}[tdir]
                row.append((z, i, f"{m_} · " + COL_NAME[col].format(k=a.k) + arrow))
            grid.append(row)
        frames = []
        for t in range(32):
            tiles = [[panel(z, i, fname[str(z["video_id"][i])], t, tl, thr, bias) for z, i, tl in row] for row in grid]
            w, h = tiles[0][0].size
            canvas = Image.new("RGB", (w * 3 + 8 * 2, h * len(grid) + 8 * (len(grid) - 1)), (0, 0, 0))
            for rr, row in enumerate(tiles):
                for cc, tile in enumerate(row):
                    canvas.paste(tile, (cc * (w + 8), rr * (h + 8)))
            frames.append(canvas)
        ref = Image.new("RGB", (frames[0].width, frames[0].height * 3))
        for j, fi in enumerate((5, 20, 30)): ref.paste(frames[fi], (0, frames[0].height * j))
        dd = ImageDraw.Draw(ref)
        for q, cl in enumerate((ORANGE, WHITE, BLUE, HID, FUT, YEL, GRAYR)):          # 마커 색을 팔레트에 강제로 넣는다
            dd.rectangle([10 + 40 * q, 10, 40 + 40 * q, 40], fill=cl)
        pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        p = out / f"gif_timing_{mo}_k{a.k}.gif"
        q[0].save(p, save_all=True, append_images=q[1:], duration=[400] * 15 + [1500] + [450] * 15 + [2000], loop=0)
        print(p, [[str(z["video_id"][i]) for z, i, _ in row] for row in grid])


if __name__ == "__main__":
    main()
