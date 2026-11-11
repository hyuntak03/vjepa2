#!/usr/bin/env python3
"""정답 표현으로 학습한 물체 존재 head 를 예측에 붙인 결과 (2026-11-10) — z_analysis/results/presence_probe_20261110/transfer_probe.json 에서 다시 그린다.

(a) 물체가 가려졌다 다시 나오는 장면: 예측 미래를 head 가 '물체 있음' 으로 본 비율 (미래 블록 3–8 평균)
(b) 같은 장면의 빈 버전: 예측 미래를 '물체 없음' 으로 본 비율
막대 = 출발 모델 (Release / AR) × 학습 (그대로 · v4bc4 L1 이어 학습 · v4bc4 음의 손실 p 0.2). 색은 검증된 팔레트 1–3 (초록은 대비 경고 → 값 직접 라벨).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_presence_transfer.py
"""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = f"{REPO}/z_analysis/results/presence_probe_20261110/transfer_probe.json"
OUT = f"{REPO}/z_analysis/figures/presence_transfer_20261110"
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42,
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 7.5, "axes.linewidth": 0.6})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
GROUPS = [("Release", ["vith", "vith_release_v4bc4", "vith_release_v4bc4_negimp_vanish_app02"]),
          ("AR", ["vast_ar_vw_state", "vast_ar_vw_state_v4bc4", "vast_ar_vw_state_v4bc4_negimp_vanish_app02"])]
COND = ["Base", "+ L1 post-training", "+ negative loss"]


def main():
    R = json.load(open(SRC))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    w = 0.26
    for ax, key, lab in zip(axes, ("ball_as_ball", "empty_as_empty"),
                            ("(a) Object scenes judged ‘object present’", "(b) Empty scenes judged ‘empty’")):
        for gi, (gname, ms) in enumerate(GROUPS):
            for ci, m in enumerate(ms):
                v = float(np.mean(R[m][key][2:]))
                x = gi + (ci - 1) * (w + 0.03)
                ax.bar([x], [v], width=w, color=COL[ci], edgecolor="none", zorder=2, label=COND[ci] if gi == 0 else None)
                ax.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="#222222")
        ax.axhline(50, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks([0, 1], [g for g, _ in GROUPS])
        ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.text(0.5, -0.22, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    axes[0].set_ylabel("Predicted futures (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 0.9), w_pad=1.5)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    print("저장", OUT + ".pdf · .png")


if __name__ == "__main__":
    main()
