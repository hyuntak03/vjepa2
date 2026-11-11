#!/usr/bin/env python3
"""진단 → 처방 그림 (2026-11-10): 예측 속 물체 신호의 세기와 방향 — z_analysis/results/presence_probe_20261110/vector_signal.json 에서 다시 그린다.

물체 벡터 = 같은 장면의 (물체 장면 − 빈 장면) 차이, 실제 표현 · 예측 표현 각각 (블록 평균 풀링, 미래 블록 3–8 평균).
(a) 진단 2 — 분리도 d′ = 실제 물체 방향 위에서 물체 장면과 빈 장면의 평균 차 ÷ 장면 간 흩어짐 (점선 = 같은 계열의 실제 표현 값)
(b) 검증 — 실제 영상 영속성 (real_world_permanence_uniform) 짝 A 사라짐 정답률, 오차 막대 Wilson 95 % (짝 95 개)
값은 presence_vector_signal.py 가 쓴 vector_signal.json 에서 읽는다 (방향 cos 은 표로만, 2026-11-10 진단 → 처방).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_presence_vector_signal.py
"""
import json, math, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = f"{REPO}/z_analysis/results/presence_probe_20261110/vector_signal.json"
OUT = f"{REPO}/z_analysis/figures/presence_vector_signal_20261110"
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42,
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 7.5, "axes.linewidth": 0.6})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
GROUPS = [("Release", ["vith", "vith_release_v4bc4", "vith_release_v4bc4_negimp_vanish_app02"]),
          ("AR", ["vast_ar_vw_state", "vast_ar_vw_state_v4bc4", "vast_ar_vw_state_v4bc4_negimp_vanish_app02"])]
COND = ["Base", "+ L1 post-training", "+ negative loss"]


def wilson(p, n, z=1.96):
    p = p / 100; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    R = json.load(open(SRC))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    w = 0.26
    for ax, key, lab, ylab in zip(axes, ("dp", "rwp_A"), ("(a) Object-signal separation in predicted futures", "(b) Real-video vanish accuracy"),
                                  ("$d'$ along real object direction", "Pairwise accuracy (%)")):
        for gi, (gname, ms) in enumerate(GROUPS):
            for ci, m in enumerate(ms):
                x = gi + (ci - 1) * (w + 0.03)
                if key == "dp":
                    v = R[m]["late"]["dp"]; txt = f"{v:.2f}"; top = v
                else:
                    v = R[m]["rwp"]["rwp_A"]; lo, hi = wilson(v, R[m]["rwp"]["n_A"]); txt = f"{v:.1f}"; top = hi
                    ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#5c5c5c", elinewidth=0.55, capsize=1.3, capthick=0.55, zorder=3)
                ax.bar([x], [v], width=w, color=COL[ci], edgecolor="none", zorder=2, label=COND[ci] if gi == 0 else None)
                ax.annotate(txt, xy=(x, top), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="#222222")
            if key == "dp":                                    # 같은 계열의 실제 표현 값 (기준)
                ref = R[ms[0]]["late"]["dh"]
                ax.plot([gi - 1.6 * w, gi + 1.6 * w], [ref, ref], color="#222222", lw=0.9, ls=(0, (3, 2)), zorder=3)
                ax.annotate("real", xy=(gi + 1.6 * w, ref), xytext=(2, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color="#222222")
            else:
                ax.axhline(50, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks([0, 1], [g for g, _ in GROUPS])
        ax.set_ylim(0, 2.0) if key == "dp" else (ax.set_ylim(0, 100), ax.set_yticks([0, 25, 50, 75, 100]))
        ax.set_ylabel(ylab)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.text(0.5, -0.22, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 0.9), w_pad=2.0)
    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=300)
    print("저장", OUT + ".pdf · .png")


if __name__ == "__main__":
    main()
