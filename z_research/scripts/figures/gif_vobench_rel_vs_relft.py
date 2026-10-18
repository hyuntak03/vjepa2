"""late (문맥 끝 가림) 예시 GIF — 왼쪽 release, 오른쪽 release + VOBench post-FT (held-out), 2026-10-14.
행 = 클립 (split v2 test block · primitive · 가능 영상 · 운동별 하나), 열 = 모델. 원본 프레임 (stride 3, 32 장) 위에
  흰 원 = 실제 물체 중심 (label.json, 가려져 있으면 회색) · 주황 십자 (검정 테두리) = 그 모델의 p 를 그 모델의 p 자로 읽은 위치 (t8–t15).
  정체는 argmax (fig_timeline_rel_vs_relft 와 같은 규칙): '없음' 이면 십자를 안 그리고 'none' 으로 적는다. 글자색 초록 = 정체 맞음, 빨강 = 틀림.

  python z_research/scripts/figures/gif_vobench_rel_vs_relft.py [--occ late] [--n-per-motion 1]
  → z_research/VOBench_v1/vjepa2_pft/readout_figures/examples_<occ>_rel_vs_relft.gif (+ _t12.png 정지 화면)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import vobench_ar_readout as V  # noqa: E402

OUT = REPO / "z_research/VOBench_v1/vjepa2_pft/readout_figures"
MODELS = (("rel", "release V-JEPA 2"), ("relft", "+ VOBench post-FT"))
MOTIONS = ("flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up")
PX, BAR = 256, 30


FIXED = [(255, 120, 20), (0, 0, 0), (255, 255, 255), (150, 150, 150), (0, 150, 70), (200, 30, 30)]   # 주황 십자 · 테두리 · 원 · 글자


def shared_palette(frames):
    """모든 프레임 공통 팔레트: 프레임 몇 장으로 250 색을 뽑고 표식 색 (FIXED) 을 정확히 끼운다."""
    pick = frames[::4]; w, h = pick[0].size
    comp = Image.new("RGB", (w, h * len(pick)))
    for i, f in enumerate(pick): comp.paste(f, (0, i * h))
    base = comp.quantize(256 - len(FIXED), dither=Image.Dither.NONE).getpalette()[: 3 * (256 - len(FIXED))]
    pal = Image.new("P", (1, 1)); pal.putpalette(base + [c for rgb in FIXED for c in rgb]); return pal


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late"); ap.add_argument("--n-per-motion", type=int, default=1)
    a = ap.parse_args()
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    rows = [r for r in V.rows_primitive() if r["block_id"] in test and r["occlusion"] == a.occ and r["variant"] == "pos_a" and r["obj_a"] != "empty"]
    picks = []
    for mo in MOTIONS:
        c = sorted((r for r in rows if r["motion"] == mo), key=lambda r: (r["violation"] != "shape", r["k"] != "2", r["video_id"]))
        picks += c[:a.n_per_motion]
    Z = {n: V.load(n) for n, _ in MODELS}; vid = {n: {v: i for i, v in enumerate(z["video_id"])} for n, z in Z.items()}
    lab = {r["video_id"]: json.load(open(V.FR / Path(r["file_name"]).parent / "label.json"))["labels"]["A"] for r in picks}
    W, H = PX * len(MODELS), (PX + BAR) * len(picks) + 22
    frames = []
    for s in range(32):
        cv = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(cv)
        for c, (_, title) in enumerate(MODELS):
            d.text((c * PX + 6, 5), f"{title}   t{s // 2} ({'context' if s < 16 else 'predicted'})", fill=(0, 0, 0))
        for ri, r in enumerate(picks):
            y0 = 22 + ri * (PX + BAR)
            im = Image.open(V.FR / r["file_name"] / f"{3 * s:06d}.png").convert("RGB").resize((PX, PX))
            L = lab[r["video_id"]][3 * s]; tu, tv = L["u"] / 2, L["v"] / 2
            sh, co = r["obj_a"].split("/"); cls = V.SHAPES.index(sh) * len(V.COLORS) + V.COLORS.index(co)
            for c, (n, _) in enumerate(MODELS):
                x0 = c * PX; cv.paste(im, (x0, y0))
                d.ellipse([x0 + tu - 10, y0 + tv - 10, x0 + tu + 10, y0 + tv + 10], outline=(150, 150, 150) if L["hidden"] else (255, 255, 255), width=2)
                txt, col = "context", (0, 0, 0)
                if s >= 16:
                    j = (s - 16) // 2; i = vid[n][r["video_id"]]
                    pb = Z[n]["pp_prob"][i, j]; k = int(pb.argmax())
                    if k == V.NONE:
                        txt, col = f"p: none  (P none {pb[V.NONE]:.2f})", (200, 30, 30)
                    else:
                        au, av = V.to512(Z[n]["pp_xy"][i, j]) / 2; ok = k == cls
                        for cc, w, r_ in (((0, 0, 0), 7, 12), ((255, 120, 20), 4, 11)):        # 검정 테두리 위에 주황
                            d.line([x0 + au - r_, y0 + av, x0 + au + r_, y0 + av], fill=cc, width=w); d.line([x0 + au, y0 + av - r_, x0 + au, y0 + av + r_], fill=cc, width=w)
                        txt, col = f"p: {V.SHAPES[k // 8]}/{V.COLORS[k % 8]}", (0, 150, 70) if ok else (200, 30, 30)
                d.text((x0 + 4, y0 + PX + 2), f"{r['motion']} · k{r['k']} · true {r['obj_a']}" + ("  [hidden]" if L["hidden"] else ""), fill=(0, 0, 0))
                d.text((x0 + 4, y0 + PX + 15), txt, fill=col)
            d.line([PX, 0, PX, H], fill=(0, 0, 0), width=1)
        frames.append(cv)
    OUT.mkdir(parents=True, exist_ok=True); g = OUT / f"examples_{a.occ}_rel_vs_relft.gif"
    pal = shared_palette(frames)                                    # 프레임마다 따로 줄이면 주황이 프레임마다 다른 색으로 바뀐다 (2026-10-14)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(g, save_all=True, append_images=q[1:], duration=[200] * 16 + [600] * 16, loop=0)
    frames[24].save(g.with_name(g.stem + "_t12.png"))
    print("saved →", g, "|", [r["video_id"] for r in picks])


if __name__ == "__main__":
    main()
