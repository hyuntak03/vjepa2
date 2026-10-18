#!/usr/bin/env python3
"""intphys1_late_hid 전체 쌍 그림 (2026-10-18) — 채점에 실제로 들어간 32 장 (문맥 16 | 미래 16, skip 2) 을 쌍마다 그린다.

쌍마다 두 줄: 위 = 가능 (pos), 아래 = 불가능 (imp). 문맥 16 장은 두 줄이 픽셀 단위로 같다 (build.py --hidden-end).
문맥 마지막 튜블릿 (c15 · c16) 은 초록 테두리 — 이 두 장에 물체가 안 보이도록 고른 창이다. 빨간 세로선 = 문맥 | 미래.
라벨: 쌍 · 원리 · d · 문맥 끝 e (e − d) · c16 / f1 의 원 프레임 번호 · 모델별 정답 (O/X): release · release+PFT (VOBench+gravity) · vast AR · 복사 copy_h.
  python cvpr/eval/intphys1_late/plot_hid_gallery.py [--ds intphys1_late_hid | intphys1_late_rev | intphys1_late_rev_dn1]
  → cvpr/results/eval/intphys1_late/<ds>_vith/figures/gallery_{moving,static}.png
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "cvpr/results/eval/intphys1_late"
MODELS = [("release", "vith"), ("PFT", "vith_release_vobftg"), ("vastAR", "vith_vast_ar_state_ep70"), ("vastAR+PFT", "vith_vast_ar_state_ep70_vobftg")]
TH, LEFT, GAPC, ROWG, PAIRG, TOP = 72, 330, 2, 2, 16, 52
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 12); FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="intphys1_late_hid"); a = ap.parse_args(); DS = a.ds
    FR = Path("/local_datasets/world/world_analysis/IntPhys1_late" + DS[len("intphys1_late"):])
    idx = {r["video_id"]: r for r in csv.DictReader(open(REPO / f"data_csv/{DS}/index.csv"))}
    P = [p for p in csv.DictReader(open(REPO / "z_research/previous/OcclusionStats/intphys1/pairs.csv")) if p["pos"] in idx]
    S = {k: json.load(open(RES / f"{DS}_{m}/per_block.json"))["per_video_surprise"] for k, m in MODELS}
    S["copy_h"] = json.load(open(RES / f"{DS}_vith/copy_baseline.json"))["per_video"]["copy_h"]
    out = RES / f"{DS}_vith/figures"; out.mkdir(exist_ok=True)
    for grp in ("moving", "static"):
        rs = sorted([p for p in P if p["group"].startswith(grp)], key=lambda p: (p["principle"], p["scene"]))
        W = LEFT + 32 * (TH + GAPC) + 14; H = TOP + len(rs) * (2 * TH + ROWG + PAIRG) + 10
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"{DS} · {grp} | occlusion · {len(rs)} pairs · top possible / bottom impossible · context 16 | future 16 (stride {sorted({int(idx[q['pos']]['skip']) for q in rs})}) · "
                        f"green = last context tubelet (object hidden), red line = context | future", fill=(0, 0, 0), font=FONT_B)
        xs = [LEFT + k * (TH + GAPC) + (14 if k >= 16 else 0) for k in range(32)]
        for k in range(32):
            if k in (0, 14, 15, 16, 31) or k % 4 == 0: dr.text((xs[k] + 4, TOP - 18), f"c{k + 1}" if k < 16 else f"f{k - 15}", fill=(0, 0, 0), font=FONT)
        for i, p in enumerate(rs):
            y = TOP + i * (2 * TH + ROWG + PAIRG); r0 = idx[p["pos"]]; fr = [int(x) for x in r0["frames"].split()]
            marks = " ".join(f"{k} {'O' if S[k][p['imp']] > S[k][p['pos']] else 'X'}" for k in ("release", "PFT", "vastAR", "vastAR+PFT", "copy_h"))
            dr.multiline_text((8, y + 4), f"{p['imp']} vs {p['pos']}\n{p['principle']} · d={p['d']} · e={r0['ctx_end']} (e-d {int(r0['ctx_end']) - int(p['d']):+d})\n"
                                          f"stride {r0['skip']} · raw: c16={fr[15]} · f1={fr[16]} · f16={fr[31]}\n{p['event_kind']}\n{marks}", fill=(0, 0, 0), font=FONT, spacing=2)
            for r, vid in enumerate((p["pos"], p["imp"])):
                for k in range(32):
                    f = Image.open(FR / vid / f"{k:06d}.png").convert("RGB").resize((TH, TH)); im.paste(f, (xs[k], y + r * (TH + ROWG)))
            for k in (14, 15):
                dr.rectangle([xs[k] - 1, y - 1, xs[k] + TH, y + 2 * TH + ROWG], outline=(0, 170, 0), width=2)
            dr.line([(xs[16] - 8, y - 3), (xs[16] - 8, y + 2 * TH + ROWG + 3)], fill=(220, 0, 0), width=3)
        f = out / f"gallery_{grp}.png"; im.save(f); print("saved", f, im.size)


if __name__ == "__main__":
    main()
