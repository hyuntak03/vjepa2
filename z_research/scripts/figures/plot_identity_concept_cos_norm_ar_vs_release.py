"""정체 (모양 · 색) 개념 벡터 — encoder vs 예측, 물체 칸 기준, AR 굴린 예측과 릴리즈 predictor (2026-10-26 저녁 갱신).
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘" · "figure 전부 갱신". 앞판 (화면 평균 개념) 은 화면 전체에 퍼진 정체 신호였다 (concept_vector_regions.json)
  → z_analysis/figures/_superseded/identity_concept_cos_norm_ar_vs_release_wholeframe.png
입력 z_analysis/results/objcell_concept_compare.json (objcell_concept_compare.py). 개념 벡터 = 클래스 평균 − 전체 평균, train 8,000 편, 직접 잰 cos · 크기 비 (클래스마다 → 평균, 띠 ±1 표준편차).
  위 줄 = 예측을 블록 b 의 진짜 물체 칸에서 평균 · 아래 줄 = 예측을 블록 7 (마지막에 본) 물체 칸에서 평균. 기준은 둘 다 encoder 의 블록 b 진짜 물체 칸.
  색 = 모델 (AR 파랑 · 릴리즈 주황), 선 모양 = 개념 (모양 실선 ● · 색 점선 ■).
  /venv/vjepa/bin/python z_research/scripts/figures/plot_identity_concept_cos_norm_ar_vs_release.py
출력: z_analysis/figures/identity_concept_cos_norm_ar_vs_release.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/objcell_concept_compare.json"))
MODELS = {"ar": ("AR (block rollout)", "#2a78d6"), "release": ("release (one-shot)", "#eb6834")}
FSTY = {"shape": ("-", "o"), "color": ("--", "s")}
ROWS = [("pred_true", "prediction at true object cells"), ("pred_last", "prediction at last-seen object cells")]
steps = list(range(1, 9))
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
for r, (where, rname) in enumerate(ROWS):
    for c, (key, ylab, ylim) in enumerate((("cos", "cosine, predictor vs encoder concept", (0, 1.05)), ("norm_ratio", "norm ratio, predictor / encoder concept", (0, 1.2)))):
        ax = axes[r, c]
        for model, (mname, col) in MODELS.items():
            if model not in J:
                continue
            for fam, (ls, mk) in FSTY.items():
                dx = {("ar", "shape"): -0.06, ("ar", "color"): -0.02, ("release", "shape"): 0.02, ("release", "color"): 0.06}[(model, fam)]; xs = np.array(steps) + dx   # 겹침 방지
                m = np.array([J[model][fam]["concept"][str(k)][where][key] for k in steps]); s = np.array([J[model][fam]["concept"][str(k)][where][key + "_sd"] for k in steps])
                ax.plot(xs, m, ls=ls, marker=mk, color=col, lw=2, ms=5, label=f"{mname}, {fam}"); ax.fill_between(xs, m - s, m + s, color=col, alpha=0.08, lw=0)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":"); ax.set_xticks(steps); ax.set_xlim(0.8, 8.2); ax.set_ylim(*ylim)
        ax.set_xlabel("prediction step (block = 2 frames)"); ax.set_ylabel(ylab); ax.set_title(rname, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
for n, ax in enumerate(axes.flat):
    ax.set_xlabel(ax.get_xlabel() + f"\n({'abcd'[n]})")   # 2026-10-26 사용자: "겹치는 것들 모두 해결" — 패널 표시는 x 축 이름 아래 줄로 (자동 배치)
fig.suptitle("Identity concept vectors in the 1280-d output space (encoder vs predictor output), object-cell mean", fontsize=12, y=0.995)   # 2026-10-26 사용자: 상단 가운데 제목 (어느 공간에서 했나)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9, frameon=False, handlelength=3)
fig.savefig(f"{R}/figures/identity_concept_cos_norm_ar_vs_release.png", dpi=170, bbox_inches="tight")
print("saved")
