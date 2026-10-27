"""고정 선형 probe 정확도 — 원본 vs 정답 개념 벡터 빼기 (z − c_정답), 막대 그림 (2026-10-26).
[2026-10-26 저녁 갱신: 물체 칸 기준 (probe_subtract_objcell.json), 점선 = 최빈 클래스. ⚠️ 같은 공간에서 빼면 선형 판독기는 수학적으로 우연 쪽으로 간다 — 계산 확인용]
사용자: "linear probe steering 결과를 figure 로. 막대 2 개, x 축 개념, y 축 acc. subtraction 하면 성능 이만큼 감소한다".

입력: z_analysis/results/concept_steering.json (concept_steering.py: 블록 전체 평균 (1, D), 표현마다 그 공간에서 개념 벡터 · ridge probe, val 350 편, α = 1).
패널: encoder (문맥 블록 0–7) · AR 굴린 예측 출력 걸음 1–2 · 걸음 6–8. 점선 = 우연 (모양 1/6, 색 1/8).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_probe_subtract_bars.py
출력: z_analysis/figures/probe_subtract_bars.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
S = json.load(open(f"{R}/results/probe_subtract_objcell.json"))
PANELS = [("encoder", "encoder (object cells)"), ("ar_rollout_s12", "AR rollout output, steps 1–2 (object cells)"), ("ar_rollout_s68", "AR rollout output, steps 6–8 (object cells)")]
FAMS = ["shape", "color"]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
w = 0.36
for ax, (key, title) in zip(axes, PANELS):
    for i, f in enumerate(FAMS):
        ch = S[key][f]["val_majority"]
        o, s = S[key][f]["original"], S[key][f]["subtract_a1"]
        for dx, v, col, lab in ((-w / 2, o, "#2a78d6", "original"), (w / 2, s, "#eb6834", "minus own concept vector")):
            ax.bar(i + dx, v, w, color=col, label=lab if i == 0 else None)
            ax.text(i + dx, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
        ax.hlines(ch, i - 0.45, i + 0.45, colors="gray", linestyles=":", lw=1.2, label="majority class" if i == 0 else None)
    ax.set_xticks(range(len(FAMS))); ax.set_xticklabels(FAMS, fontsize=10)
    ax.set_ylim(0, 100); ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("frozen linear probe accuracy (%)")
for ax, lab_ in zip(axes, ("(a)", "(b)", "(c)")):
    ax.text(0.5, -0.17, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/probe_subtract_bars.png", dpi=170, bbox_inches="tight")
print({k: {f: (S[k][f]["original"], S[k][f]["subtract_a1"]) for f in FAMS} for k, _ in PANELS})
