#!/usr/bin/env python3
"""[X1] R3 그림 — (a) E34 국소 판독 검사 Δ (참 자리 − 대조 3 곳 평균) · (b) E33/E33b 같은 장면 두 판 속도 차 기울기 (2026-10-30).

입력  z_analysis/transfer_gap/results/q1q2/{E34_arlocal.json, E33_velsep.json, E33b_transfer.json}
출력  z_analysis/transfer_gap/results/q1q2/fig/q1q2_r3_summary.{png,pdf}
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/figures/plot_transfer_gap_q1q2_r3.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "z_analysis/transfer_gap/results/q1q2"
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

e34 = json.load(open(RES / "E34_arlocal.json"))
e33 = json.load(open(RES / "E33_velsep.json")); e33b = json.load(open(RES / "E33b_transfer.json"))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))

srcs = [("rel_p", "release p"), ("rel_pft_p", "release+PFT p"), ("h", "h (32-frame enc.)"), (None, ""),
        ("ar_h_blk", "h_blk (per-block enc.)"), ("vast70_p", "vast AR p"), ("vast70_pft_p", "vast AR+PFT p"), ("ariel_ar38_p", "Ariel AR p")]
tasks = [("primitive|shape", "prim. shape"), ("realistic|shape", "real. shape"), ("primitive|color", "prim. color")]
ax.axhspan(-5, 5, color="#999999", alpha=0.15, lw=0)
ax.axhline(0, color="#666666", lw=0.8)
for k, (tk, tl) in enumerate(tasks):
    xs, ys, lo, hi = [], [], [], []
    for x, (s, _) in enumerate(srcs):
        if s is None:
            continue
        e = e34["tasks"][tk][s]; d = e["delta_true_minus_ctrl"]; c = e["delta_CI"]
        xs.append(x + (k - 1) * 0.22); ys.append(d); lo.append(d - c[0]); hi.append(c[1] - d)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.5, color=COL[k], ecolor=COL[k], elinewidth=1.2, capsize=0, label=tl,
                markeredgecolor="white", markeredgewidth=0.8)
ax.axvline(3, color="#bbbbbb", lw=0.8, ls=":")
ax.set_xticks([x for x, (s, _) in enumerate(srcs) if s]); ax.set_xticklabels([n for s, n in srcs if s], rotation=30, ha="right")
ax.set_ylabel("true-location minus control (pt)")
ax.text(1, 70, "release encoder space", ha="center", fontsize=7, color="#555555")
ax.text(5.5, 70, "AR encoder space (per-block)", ha="center", fontsize=7, color="#555555")
ax.set_ylim(-12, 76)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
ax.set_title("(a)", y=-0.52)

late = e33["sets"]["late23"]["u|s0|F7"]
bars = [("object\narms", e33b["F7"]["s0"]["slope_obj_all"]), ("empty arms\n(same readout)", e33b["F7"]["s0"]["slope_empty"]),
        ("presence\ncontrast", e33b["deltaz|F7"]["s0"]["within_scene_slope"])]
bx.axhline(1.0, color="#666666", lw=0.8, ls="--"); bx.axhline(0, color="#666666", lw=0.8)
for j, (n, v) in enumerate(bars):
    bx.bar(j, v, width=0.6, color=COL[0] if j != 1 else COL[1], edgecolor="white", linewidth=2)
    bx.text(j, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
bx.set_xticks(range(len(bars))); bx.set_xticklabels([b[0] for b in bars], fontsize=7)
bx.set_ylabel("within-scene slope"); bx.set_ylim(0, 1.15)
bx.set_title("(b)", y=-0.52)
fig.tight_layout()
(RES / "fig").mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(RES / "fig" / f"q1q2_r3_summary.{ext}", dpi=200, bbox_inches="tight")
print("saved", RES / "fig" / "q1q2_r3_summary.png", "late23 F7 within-scene slope", late["within_scene_slope"])
