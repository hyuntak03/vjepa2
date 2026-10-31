#!/usr/bin/env python3
"""[X1] Q1 · Q2 요약 그림 (2026-10-30) — (a) S2 정렬 사다리 h 판독 · (b) 물체 영역 정체 대비 크기 ρ_O 걸음별 · (c) late 사라짐 A 주입.
입력: z_analysis/transfer_gap/results/q1q2/{E4_ladder,E2_pairdiff,E7_inject,E6_rescore}.json → 출력 results/q1q2/fig/q1q2_summary.{png,pdf}
  $P z_research/scripts/figures/plot_transfer_gap_q1q2.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[3] / "z_analysis/transfer_gap/results/q1q2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#222222", "#777777"
lad = json.load(open(R / "E4_ladder.json")); e2 = json.load(open(R / "E2_pairdiff.json"))["summary"]
e7 = json.load(open(R / "E7_inject.json")); e6 = json.load(open(R / "E6_rescore.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1.15]))
# (a)
rungs = [("L0", "none"), ("L1", "per-ch"), ("L2", "scalar"), ("L2b", "per-PC"), ("L3", "rotate")]
w = 0.38
for q, (o, col) in enumerate((("primitive", C1), ("realistic", C2))):
    cell = lad[o]["S2_vis_move_t11-15"]
    ys = [cell["maps"][r]["acc"]["mean"] for r, _ in rungs]
    xs = [i + (q - 0.5) * w for i in range(len(rungs))]
    ax[0].bar(xs, ys, w * 0.92, color=col, label=o)
    ax[0].axhline(cell["ceiling_p_reader_L0"], color=col, lw=1, ls="--")
ax[0].set_xticks(range(len(rungs))); ax[0].set_xticklabels([n for _, n in rungs], rotation=35, ha="right"); ax[0].set_xlabel("map rung\n(a)")
ax[0].set_ylabel("h-reader accuracy (%)"); ax[0].set_ylim(0, 100)
ax[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
# (b)
ts = list(range(8, 16))
for lab, key, col in (("release, visible", "id|primitive|rel|bystep|vis_move|t{}", C1), ("release, late", "id|primitive|rel|bystep|late_move|t{}", C2),
                      ("PFT, late", "id|primitive|pft|bystep|late_move|t{}", C3)):
    ys = [e2[key.format(t)]["rho_O"]["mean"] for t in ts]
    ax[1].plot(ts, ys, color=col, lw=2, marker="o", ms=4, label=lab)
ax[1].set_xlabel("future tubelet\n(b)"); ax[1].set_ylabel(r"identity contrast size $\rho_O$")
ax[1].set_ylim(0, 0.6); ax[1].legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
# (c)
k = "both|all|late_move|permanence|A"
vals = [("release", e7[k]["L0raw"]["acc"] if isinstance(e7[k]["L0raw"], dict) else e7[k]["L0raw"]), ("copy", e7[k]["copy"]),
        ("id @ last seen", e7[k]["d2"]["acc"]), ("id @ true place", e7[k]["d3"]["acc"]), ("true future", e7[k]["d1"]["acc"]),
        ("PFT (test)", e6["both|test|late_move|permanence|A"]["PFT_L0raw"]["acc"])]
ax[2].bar(range(len(vals)), [v for _, v in vals], 0.7, color=C1)
for i, (_, v) in enumerate(vals):
    ax[2].text(i, v + 2, f"{v:.0f}", ha="center", color=INK, fontsize=7)
ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels([n for n, _ in vals], rotation=35, ha="right")
ax[2].set_ylabel("late vanish (A) correct (%)"); ax[2].set_xlabel("(c)"); ax[2].set_ylim(0, 110)
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
fig.tight_layout()
(R / "fig").mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(R / "fig" / f"q1q2_summary.{ext}", dpi=200, bbox_inches="tight")
print("→", R / "fig/q1q2_summary.png")
