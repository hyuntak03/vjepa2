#!/usr/bin/env python3
"""VOBench k 2·3 — 물체가 보이는 블록 / 가린 블록에서 obj (있음 vs 빈 장면) · shape · color 가 읽히나: (a) 문맥 encoder (b) predictor 출력 1280
(c) 문맥 encoder 로 학습한 판독기를 predictor 출력의 같은 범주 블록에 그대로 건다 (2026-10-29, 사용자 요청).

판독기는 보임 · 가림 따로 학습, 각자 자기 블록에서 시험 (vobench_attnprobe_release.py probe_fig). 입력 = 블록 하나 (256, 1280).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_fig_vis.py
입력: z_analysis/results/vobench_attnprobe_release/fig_vis/{z,p}_{obj,shape,color}.json · 출력: z_analysis/figures/vobench_fig_vis.{png,pdf}
색 (2026-10-29 사용자 "색깔 다시"): obj 는 있음/없음 2 지선다라 속성 (shape · color) 과 구분해 무채색 회색, shape 파랑 · color 주황.
  검사 (dataviz validate_palette, all-pairs): 분리 · 대비 통과, 회색은 채도 하한만 의도적으로 미달. 모든 막대에 값을 직접 단다.
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# 2026-10-29 정정: fig_vis (학습 표본 수만 맞춤) → fig_vis_pm (가린 블록과 같은 위치의 보이는 블록으로 짝, 시험도 그 위치). 앞 판은 p 의 보이는 블록이 미래 1–8 이 섞여 낮게 나왔다
RES = os.path.join(REPO, "z_analysis/results/vobench_attnprobe_release/fig_vis_pm")
TG = ("obj", "shape", "color")
COL = {"obj": "#6b6a66", "shape": "#2a78d6", "color": "#eb6834"}
CH = {"obj": 50.0, "shape": 100 / 6, "color": 100 / 8}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
w = 0.27
for ax, (rep, lab) in zip(axes, (("z", "(a) context encoder"), ("p", "(b) predictor output"), ("zp", "(c) encoder → predictor"))):
    for j, t in enumerate(TG):
        R = json.load(open(os.path.join(RES, f"{rep}_{t}.json")))
        for xi, c in enumerate(("visible", "occluded")):
            v = 100 * float(np.mean([r["pred"] == r["y"] for r in R if r["cat"] == c]))
            x = xi + (j - 1) * w
            ax.bar(x, v, w - 0.03, color=COL[t], label=t if xi == 0 else None, zorder=3)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#333")
            ax.plot([x - (w - 0.03) / 2, x + (w - 0.03) / 2], [CH[t]] * 2, color="#000", lw=1.4, ls=(0, (2, 1.5)), zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["visible", "occluded"])
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 108); ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_xlabel(lab, labelpad=6)
axes[0].set_ylabel("probe accuracy (%)")
h_, l_ = axes[0].get_legend_handles_labels()
h_.append(matplotlib.lines.Line2D([], [], color="#000", lw=1.4, ls=(0, (2, 1.5)))); l_.append("chance")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(h_, l_, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8)
OD = os.path.join(REPO, "z_analysis/figures"); os.makedirs(OD, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OD, f"vobench_fig_vis.{ext}"), dpi=200, bbox_inches="tight")
print(os.path.join(OD, "vobench_fig_vis.png"))
