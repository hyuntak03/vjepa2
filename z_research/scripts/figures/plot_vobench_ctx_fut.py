#!/usr/bin/env python3
"""VOBench late k2 — 블록별 판독, 선 그래프 (2026-10-29, 사용자: "x 축 시간 · shape acc, color acc · 정상 색 ·
(a) encoder 는 context 1–16 장 (15·16 가려짐) k=2 · (b) predictor 17–32 장 (17·18 가려짐) · (c) transfer · 속도 그래프도").

결과: vobench_occlusion_probe.py probe_cf → z_analysis/vobench_occlusion_probe/results/ctx_fut/summary.json
  (a) 문맥 encoder 의 문맥 블록 b 로 판독기 학습 · 시험 · (b) predictor 미래 블록 b 로 학습 · 시험
  (c) 문맥 encoder 블록 전부로 학습한 판독기 하나를 predictor 미래 블록 b 에 그대로
  판독기 입력 = 블록 하나의 토큰 (256, 1280) → attention pooling → (1, 1280) → 선형. 학습 = 전 조건 영상, 시험 = late k2.
윗줄 = 모양 · 색 정확도, 아랫줄 = 속도 R². 띠 = block 단위 부트스트랩 95 % CI, 점선 = 우연 수준, 회색 칸 = 두 장 모두 가린 블록.
색 = 검증 팔레트 slot 1–3 (#2a78d6 · #eb6834 · #1baf7a). aqua 는 밝은 바탕 대비 3:1 미만이라 점마다 값을 단다 (CLAUDE.md §8-4).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_ctx_fut.py
출력: z_analysis/vobench_occlusion_probe/figures/late_k2_ctx_fut.{png,pdf}
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines, matplotlib.patches
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASE = os.path.join(REPO, "z_analysis/vobench_occlusion_probe")
R = json.load(open(os.path.join(BASE, "results/ctx_fut/summary.json")))
B = np.arange(1, 9)
COL = {"shape": "#2a78d6", "color": "#eb6834", "vel": "#1baf7a"}
CH = {"shape": 100 / 6, "color": 100 / 8}
PANELS = (("z", "(a) context encoder", 1, 8), ("p", "(b) predictor", 17, 1), ("z->p", "(c) context encoder → predictor", 17, 1))  # (키, 이름, 첫 장, 가린 블록)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8), sharey="row")
for col, (key, lab, f0, occ) in enumerate(PANELS):
    for row, tgts in enumerate((("shape", "color"), ("vel",))):
        ax = axes[row, col]
        ax.axvspan(occ - 0.5, occ + 0.5, color="#e6e6e6", lw=0, zorder=0)
        for t in tgts:
            v = np.array([R[f"{t}|late k2|{key}|b{b}"]["score"] for b in B])
            ci = np.array([R[f"{t}|late k2|{key}|b{b}"]["ci"] for b in B])
            ax.fill_between(B, ci[:, 0], ci[:, 1], color=COL[t], alpha=0.16, lw=0, zorder=1)
            ax.plot(B, v, color=COL[t], lw=1.8, marker="o", ms=3.8, label={"shape": "object shape", "color": "object color", "vel": "object velocity"}[t], zorder=3)
            if t in CH:
                ax.axhline(CH[t], color=COL[t], lw=0.9, ls=(0, (3, 2)), alpha=0.8, zorder=2)
            if t == "vel":
                for b, vv in zip(B, v):
                    ax.text(b, vv + 6, f"{vv:.0f}", ha="center", va="bottom", fontsize=6.3, color="#333")
            elif key == "p":                                       # 2026-10-29 사용자: "(b) 는 수치가 보이면" — 모양은 점 위, 색은 점 아래
                for b, vv in zip(B, v):
                    ax.text(b, vv + (3 if t == "shape" else -3), f"{vv:.0f}", ha="center", va="bottom" if t == "shape" else "top",
                            fontsize=6.3, color=COL[t])
        if row == 1:
            ax.axhline(0, color="#888", lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.set_ylim(-130, 118)            # transfer 의 속도 R² 가 −113 까지 내려간다 (자르지 않는다)
        else:
            ax.set_ylim(0, 105)
        ax.set_xticks(B); ax.set_xticklabels([f"{f0 + 2 * (b - 1)}–{f0 + 2 * b - 1}" for b in B], fontsize=6.5, rotation=40)
        ax.set_xlim(0.5, 8.5); ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
        if row == 1:
            ax.set_xlabel(f"frames\n{lab}", labelpad=3)
axes[0, 0].set_ylabel("probe accuracy (%)"); axes[1, 0].set_ylabel("velocity R² (%)")
h_, l_ = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
h_ += h2 + [matplotlib.lines.Line2D([], [], color="#666", lw=0.9, ls=(0, (3, 2))), matplotlib.patches.Patch(color="#e6e6e6")]
l_ += l2 + ["chance", "both frames occluded"]
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(h_, l_, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=8)
OD = os.path.join(BASE, "figures"); os.makedirs(OD, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OD, f"late_k2_ctx_fut.{ext}"), dpi=220, bbox_inches="tight")
print(os.path.join(OD, "late_k2_ctx_fut.png"))
