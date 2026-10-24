#!/usr/bin/env python3
"""intphys1_late 샘플 그림 (2026-10-17) — 모델이 맞힌 쌍 / 틀린 쌍을 실제로 채점에 들어간 32 장 (문맥 16 | 미래 16) 으로 그린다.

쌍마다 두 줄: 위 = 가능 (pos), 아래 = 불가능 (imp). 왼쪽 16 장 = 문맥 (두 줄이 픽셀 단위로 같다), 굵은 세로선 뒤 16 장 = 미래 (채점 대상).
라벨: 영상 id · 원리 · 움직임 · d · 모델 ΔS = S(imp) − S(pos) (양수 = 정답) · 복사 기준선 ΔS (copy_z · copy_h).

  python cvpr/eval/intphys1_late/plot_samples.py [--run cvpr/results/eval/intphys1_late/intphys1_late_vith] [--n 10]
  → <run>/figures/samples_{correct,wrong}.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
FR = Path("/local_datasets/world/world_analysis/IntPhys1_late")
TH, LEFT, GAPC, ROWG, PAIRG, TOP = 64, 300, 2, 2, 14, 46
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 12); FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=str(REPO / "cvpr/results/eval/intphys1_late/intphys1_late_vith")); ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(); run = Path(a.run)
    m = json.load(open(run / "per_block.json"))["per_video_surprise"]; cb = json.load(open(run / "copy_baseline.json"))["per_video"]
    P = [p for p in csv.DictReader(open(REPO / "z_research/previous/OcclusionStats/intphys1/pairs.csv")) if "occluded" in p["group"]]
    for p in P:
        p["dm"] = m[p["imp"]] - m[p["pos"]]; p["dz"] = cb["copy_z"][p["imp"]] - cb["copy_z"][p["pos"]]; p["dh"] = cb["copy_h"][p["imp"]] - cb["copy_h"][p["pos"]]
    rng = np.random.default_rng(0)
    right = [p for p in P if p["dm"] > 0]; wrong = [p for p in P if p["dm"] <= 0]
    sel = {"correct": [right[i] for i in sorted(rng.choice(len(right), min(a.n, len(right)), replace=False))],
           "wrong": sorted(wrong, key=lambda p: p["dm"])[:a.n]}
    (run / "figures").mkdir(exist_ok=True)
    for name, rs in sel.items():
        W = LEFT + 32 * (TH + GAPC) + 12; H = TOP + len(rs) * (2 * TH + ROWG + PAIRG) + 10
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"intphys1_late · release · {name} pairs ({len(rs)} of {len(right) if name == 'correct' else len(wrong)}) · top row possible / bottom row impossible · "
                        f"context 16 | future 16 (skip 2, context end = d-1)", fill=(0, 0, 0), font=FONT_B)
        xs = [LEFT + k * (TH + GAPC) + (12 if k >= 16 else 0) for k in range(32)]
        for k in (0, 15, 16, 31): dr.text((xs[k] + 4, TOP - 16), f"{'c' if k < 16 else 'f'}{k + 1 if k < 16 else k - 15}", fill=(0, 0, 0), font=FONT)
        for i, p in enumerate(rs):
            y = TOP + i * (2 * TH + ROWG + PAIRG)
            ok = lambda v: "O" if v > 0 else "X"
            dr.multiline_text((8, y + 6), f"{p['imp']} vs {p['pos']}\n{p['principle']} · {p['group'].split('/')[0]} · d={p['d']}\n{p['event_kind']}\n"
                                          f"model dS {p['dm']:+.4f} {ok(p['dm'])}\ncopy_z dS {p['dz']:+.4f} {ok(p['dz'])} · copy_h {ok(p['dh'])}",
                              fill=(0, 0, 0), font=FONT, spacing=2)
            for r, vid in enumerate((p["pos"], p["imp"])):
                for k in range(32):
                    f = Image.open(FR / vid / f"{k:06d}.png").convert("RGB").resize((TH, TH)); im.paste(f, (xs[k], y + r * (TH + ROWG)))
            dr.line([(xs[16] - 7, y - 2), (xs[16] - 7, y + 2 * TH + ROWG + 2)], fill=(220, 0, 0), width=3)
        out = run / "figures" / f"samples_{name}.png"; im.save(out); print("saved", out, im.size)


if __name__ == "__main__":
    main()
