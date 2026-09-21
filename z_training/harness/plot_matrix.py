#!/usr/bin/env python3
"""전이 행렬 그림 — z_training/runs/_matrix/MATRIX.json (harness/matrix.py 산출) 을 그린다. 2026-09-19.

(a) 짝 정확도 3 열 (v11 test / IntPhys1 dev / IntPhys2 test): 색 = 릴리즈 대비 Δ (pt), 글자 = 값 + Δ. 대각 (학습한 도메인) 은 굵은 테두리.
(b) Predictor_v1 holdout L1 (가능 영상뿐 — 단위가 다른 열): 색 = −ΔL1 (낮아지면 좋음 쪽 색), 글자 = L1 + Δ.
빈 칸 = 아직 안 돈 칸 (pending, 빗금). 출력: z_training/figures/fig_transfer_matrix.{png,pdf}
  python z_training/harness/matrix.py && python z_training/harness/plot_matrix.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
M = json.load(open(ROOT / "z_training/runs/_matrix/MATRIX.json"))
FIG = ROOT / "z_training/figures"; FIG.mkdir(parents=True, exist_ok=True)
# 2026-09-21 (사용자): Predictor_v1 행을 뺐다 -> 학습 3 x 시험 3 정방 행렬 + 기준선 한 줄.
#   그 행이 빠지면서 (b) Predictor_v1 holdout L1 열도 짝을 잃어 같이 뺐다.
#   되살리려면 ROWS 에 ("predictor_v1_postft", "Predictor_v1") 를 넣고 PANEL_B = True 로.
ROWS = [("release", "Release (no FT)"), ("v11_postft", "IntPhysGen v11"),
        ("intphys1_postft", "IntPhys1"), ("intphys2_postft", "IntPhys2")]
PANEL_B = False
ACC = [("v11", "v11 test\n(C16/T16, 10,752 pairs)", "v11_postft"), ("ip1", "IntPhys1 dev\n(sliding s2_w32/avg, 180)", "intphys1_postft"),
       ("ip2", "IntPhys2 test\n(growing ctx, 252 pairs)", "intphys2_postft")]
CMAP = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f7f7f7", "#2a78d6"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})


def v(run, col):
    c = M[run].get(col); return None if c is None else c["value"]


fig = plt.figure(figsize=(8.6, 4.2) if PANEL_B else (6.6, 3.8))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.15], wspace=0.08) if PANEL_B else fig.add_gridspec(1, 1)
# ---- (a) accuracy
ax = fig.add_subplot(gs[0]); nr, nc = len(ROWS), len(ACC)
D = np.full((nr, nc), np.nan)
for i, (r, _) in enumerate(ROWS):
    for j, (c, _, _) in enumerate(ACC):
        x, b = v(r, c), v("release", c)
        if x is not None and b is not None:
            D[i, j] = 0.0 if r == "release" else x - b
lim = max(5.0, np.nanmax(np.abs(D)) if np.isfinite(D).any() else 5.0)
ax.imshow(np.ma.masked_invalid(D), cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i, (r, _) in enumerate(ROWS):
    for j, (c, _, dom) in enumerate(ACC):
        x = v(r, c)
        if x is None:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#eeeeee", hatch="///", edgecolor="#bbbbbb", lw=0))
            ax.text(j, i, "pending", ha="center", va="center", color="#888", fontsize=8); continue
        s = f"{x:.2f}" + ("" if r == "release" else f"\n({x - v('release', c):+.2f})")
        ax.text(j, i, s, ha="center", va="center", fontsize=9, color="#111", fontweight="bold" if r == dom else "normal")
        if r == dom:
            ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, edgecolor="#222", lw=1.8))
ax.set_xticks(range(nc)); ax.set_xticklabels([n for _, n, _ in ACC], fontsize=8)
ax.set_yticks(range(nr)); ax.set_yticklabels([n for _, n in ROWS]); ax.set_ylabel("predictor trained on")
ax.xaxis.tick_top(); ax.tick_params(length=0); ax.axhline(0.5, color="#222", lw=0.8)
for s in ax.spines.values(): s.set_visible(False)
ax.annotate("(a) pairwise accuracy (%), color = Δ vs release", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -14),
            textcoords="offset points", ha="center", va="top", fontsize=9)
# ---- (b) L1  (PANEL_B 일 때만)
if PANEL_B:
  bx = fig.add_subplot(gs[1]); L = np.full((nr, 1), np.nan); b0 = v("release", "pv1")
  for i, (r, _) in enumerate(ROWS):
      x = v(r, "pv1")
      if x is not None and b0 is not None: L[i, 0] = 0.0 if r == "release" else -(x - b0)
  ll = max(0.01, np.nanmax(np.abs(L)) if np.isfinite(L).any() else 0.01)
  bx.imshow(np.ma.masked_invalid(L), cmap=CMAP, norm=TwoSlopeNorm(0, -ll, ll), aspect="auto")
  for i, (r, _) in enumerate(ROWS):
      x = v(r, "pv1")
      if x is None:
          bx.add_patch(Rectangle((-.5, i - .5), 1, 1, facecolor="#eeeeee", hatch="///", edgecolor="#bbbbbb", lw=0))
          bx.text(0, i, "pending", ha="center", va="center", color="#888", fontsize=8); continue
      s = f"{x:.4f}" + ("" if r == "release" else f"\n({x - b0:+.4f})")
      bx.text(0, i, s, ha="center", va="center", fontsize=9, fontweight="bold" if r == "predictor_v1_postft" else "normal")
      if r == "predictor_v1_postft":
          bx.add_patch(Rectangle((-.47, i - .47), .94, .94, fill=False, edgecolor="#222", lw=1.8))
  bx.set_xticks([0]); bx.set_xticklabels(["Predictor_v1 holdout\nL1 ↓ (possible only)"], fontsize=8); bx.xaxis.tick_top()
  bx.set_yticks([]); bx.tick_params(length=0); bx.axhline(0.5, color="#222", lw=0.8)
  for s in bx.spines.values(): s.set_visible(False)
  bx.annotate("(b) prediction L1, blue = lower", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -14),
              textcoords="offset points", ha="center", va="top", fontsize=9)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_transfer_matrix.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08)
print(f"-> {FIG / 'fig_transfer_matrix.png'}  (pending 칸 {int(np.isnan(D).sum())} 개)")