#!/usr/bin/env python3
"""VOBench late k1 — 미래 블록 t 별 모양 · 색 판독 정확도, 선 그래프 (2026-10-29, 사용자: "x 축 시간 · shape acc, color acc ·
(a) encoder (b) predictor (c) encoder → predictor · 나온 결과로").

결과: vobench_attnprobe_release.py probe_t (판독기 = 미래 블록 t 하나의 토큰 (256, 1280) → attention pooling → 분류,
  전 조건 600 영상 block 단위 5-fold 로 학습, late k1 60 영상에서 시험). 출처 z_analysis/results/vobench_attnprobe_release/per_t/summary.json
  (a) encoder = target encoder 의 실제 미래 (h→h) · (b) predictor = 릴리즈 predictor 예측 (p→p) · (c) h 로 학습한 판독기를 p 에 (h→p)
띠 = Wilson 95 % 구간 (n = 60), 점선 = 우연 수준. 블록 1 은 첫 장이 가림판 뒤 (late k1 = 문맥 끝 1 장 + 미래 첫 1 장 가림).
색: 파스텔 채움 + 같은 계열의 진한 선 (사용자 "파스텔 톤", 선은 대비를 위해 진하게).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_late_k1_per_t.py
출력: z_analysis/vobench_occlusion_probe/figures/late_k1_per_block.{png,pdf}
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines, matplotlib.patches
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
R = json.load(open(os.path.join(REPO, "z_analysis/results/vobench_attnprobe_release/per_t/summary.json")))
N = 60
T = np.arange(1, 9)
COL = {"shape": ("#4a68b8", "#9db4ee"), "color": ("#c8653f", "#f6a98a")}     # (선, 띠)
CH = {"shape": 100 / 6, "color": 100 / 8}
IDX = {"h->h": (4, 5), "p->p": (0, 1), "h->p": (2, 3)}                        # summary 행 = [p→p 모양, 색, h→p 모양, 색, h→h 모양, 색]
PANELS = (("h->h", "(a) encoder (actual future)"), ("p->p", "(b) predictor"), ("h->p", "(c) encoder → predictor"))


def wilson(p, n, z=1.96):
    p = np.asarray(p) / 100; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for ax, (key, lab) in zip(axes, PANELS):
    ax.axvspan(0.5, 1.5, color="#e9e9e9", zorder=0, lw=0)
    for j, t in enumerate(("shape", "color")):
        v = np.array([R[f"all|late k1|t{k}"][IDX[key][j]] for k in T]); lo, hi = wilson(v, N)
        ax.fill_between(T, lo, hi, color=COL[t][1], alpha=0.45, lw=0, zorder=1)
        ax.plot(T, v, color=COL[t][0], lw=1.8, marker="o", ms=4, mfc=COL[t][1], mec=COL[t][0], mew=1.1, label=f"{t} acc", zorder=3)
        ax.axhline(CH[t], color=COL[t][0], lw=0.9, ls=(0, (3, 2)), alpha=0.7, zorder=2)
    ax.set_xticks(T); ax.set_xlim(0.5, 8.5); ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    ax.set_xlabel(f"future block (0.2 s each)\n{lab}", labelpad=4)
axes[0].set_ylabel("probe accuracy (%)")
h_, l_ = axes[0].get_legend_handles_labels()
h_ += [matplotlib.lines.Line2D([], [], color="#666", lw=0.9, ls=(0, (3, 2))), matplotlib.patches.Patch(color="#e9e9e9")]
l_ += ["chance", "first frame occluded"]
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.legend(h_, l_, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8)
OD = os.path.join(REPO, "z_analysis/vobench_occlusion_probe/figures"); os.makedirs(OD, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OD, f"late_k1_per_block.{ext}"), dpi=220, bbox_inches="tight")
print(os.path.join(OD, "late_k1_per_block.png"))
