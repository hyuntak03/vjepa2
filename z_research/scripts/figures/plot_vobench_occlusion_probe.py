#!/usr/bin/env python3
"""VOBench 가림 판독 그림 (2026-10-29) — 문맥 경계 블록에서 물체 있음 · 모양 · 색: 보이는 블록 vs 완전히 가린 블록.

(a) 문맥 encoder (문맥 마지막 블록) (b) predictor 출력 (미래 첫 블록) (c) (a) 의 판독기를 (b) 에 그대로 (d) target encoder (실제 미래 첫 블록, 대조)
네 패널 모두 같은 영상 (보임 = early · mid k2·3, 가림 = late k2·3) · 같은 학습 표본 수 (obj 36 · shape 90 · color 90) · 블록 위치 하나.
막대 = 씨앗 3 × 5-fold 시험 정확도, 오차 막대 = block 단위 부트스트랩 95 % CI, 점선 = 우연 수준 (1 / 클래스 수).
색 (사용자 "파스텔 톤"): 민트 · 연파랑 · 복숭아. dataviz 검사 (all-pairs) 에서 색끼리 분리는 통과 (CVD ΔE 9.5 · 정상 15.4),
  밝기 · 대비는 파스텔이라 미달 → 같은 계열의 진한 테두리 + 모든 막대에 값 라벨.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_occlusion_probe.py
입력: z_analysis/vobench_occlusion_probe/results/summary.json · 출력: z_analysis/vobench_occlusion_probe/figures/occlusion_probe{,_motion}.{png,pdf}
운동 그림 (2026-10-29 추가): 속도 R² (연보라) · 가속도 R² (버터) · 가속 3 지선다 정확도 (버터 + 빗금). 두 색은 검사 통과 (all-pairs).
  × = 같은 장면 · 같은 가림판에서 물체만 없는 영상에 같은 판독기를 건 값 (장면 단서 대조). 0 아래는 −15 에 ▼ 로 표시
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASE = os.path.join(REPO, "z_analysis/vobench_occlusion_probe")
R = json.load(open(os.path.join(BASE, "results/summary.json")))
TG = ("obj", "shape", "color")
FILL = {"obj": "#b9e3cf", "shape": "#9db4ee", "color": "#f6a98a"}
EDGE = {"obj": "#4f9c7a", "shape": "#4a68b8", "color": "#c8653f"}
PANELS = (("z_boundary", "(a) context encoder\nlast context block"), ("p", "(b) predictor output\nfirst future block"),
          ("z→p(경계)", "(c) context encoder probe\napplied to predictor"), ("h", "(d) target encoder\nactual first future block"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OD = os.path.join(BASE, "figures"); os.makedirs(OD, exist_ok=True)


def draw(TG, FILL, EDGE, HATCH, LAB, fname, empty=False, ylo=0):
    l_lab = "x" * (4 if empty else 3)              # 운동 그림은 값 라벨이 붙으므로 글자를 줄인다
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.85), sharey=True)
    w = 0.27
    for ax, (key, lab) in zip(axes, PANELS):
        for j, t in enumerate(TG):
            for xi, c in enumerate(("visible", "occluded")):
                r = R[f"{key}|{t}|{c}"]; x = xi + (j - 1) * w; v = r["acc"]; lo, hi = r["ci"]
                ax.bar(x, max(v, ylo), w - 0.03, bottom=0, color=FILL[t], edgecolor=EDGE[t], linewidth=0.9, hatch=HATCH[t],
                       label=LAB[t] if xi == 0 else None, zorder=3)
                ax.errorbar(x, v, yerr=[[v - max(lo, ylo)], [hi - v]], fmt="none", ecolor="#555", elinewidth=0.8, capsize=1.6, zorder=5)
                ax.text(x, hi + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8 if len(l_lab) < 4 else 5.6, color="#333")
                if r["chance"] > 0:
                    ax.plot([x - (w - 0.03) / 2, x + (w - 0.03) / 2], [r["chance"]] * 2, color="#333", lw=1.1, ls=(0, (2, 1.5)), zorder=4)
                ek = f"{key}·빈장면|{t}|{c}"
                if empty and ek in R:
                    e = R[ek]["acc"]
                    ax.scatter([x], [max(e, ylo + 3)], marker="x" if e >= ylo + 3 else "v", s=22, color="#222", linewidths=1.2, zorder=6)
        ax.axhline(0, color="#999", lw=0.7, zorder=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["visible", "occluded"])
        ax.set_xlim(-0.55, 1.55); ax.set_ylim(ylo, 112); ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
        ax.set_xlabel(lab, labelpad=5, fontsize=8)
    axes[0].set_ylabel("probe accuracy / R² (%)" if empty else "probe accuracy (%)")
    h_, l_ = axes[0].get_legend_handles_labels()
    if any(R[f"{PANELS[0][0]}|{t}|visible"]["chance"] > 0 for t in TG):
        h_.append(matplotlib.lines.Line2D([], [], color="#333", lw=1.1, ls=(0, (2, 1.5)))); l_.append("chance")
    if empty:
        h_.append(matplotlib.lines.Line2D([], [], color="#222", marker="x", lw=0, markersize=5)); l_.append("same scene, no object")
    nc = len(l_) if len(l_) <= 4 else 3
    fig.tight_layout(rect=(0, 0, 1, 0.92 if nc == len(l_) else 0.87))
    fig.legend(h_, l_, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=nc, fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OD, f"{fname}.{ext}"), dpi=220, bbox_inches="tight")
    print(os.path.join(OD, f"{fname}.png"))


draw(TG, FILL, EDGE, {t: None for t in TG}, {t: t for t in TG}, "occlusion_probe")
MT = ("vel", "acc", "acc3")
draw(MT, {"vel": "#bba7ec", "acc": "#f4d06f", "acc3": "#f4d06f"}, {"vel": "#6a55b8", "acc": "#b8901d", "acc3": "#b8901d"},
     {"vel": None, "acc": None, "acc3": "////"}, {"vel": "velocity (R²)", "acc": "acceleration (R²)", "acc3": "speeding up / const / slowing (acc.)"},
     "occlusion_probe_motion", empty=True, ylo=-20)
