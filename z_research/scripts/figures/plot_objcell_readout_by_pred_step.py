"""물체 칸 판독기 이식 — encoder 물체 칸으로 학습한 선형 판독기 (모양 · 색 따로) 를 예측의 물체 칸에, 예측 걸음별 (2026-10-26 저녁).
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘" · "figure 전부 갱신".
입력 z_analysis/results/objcell_concept_compare.json["<모델>"][<개념>]["readout"] (objcell_concept_compare.py; 블록 8–15 합쳐 학습, val 걸음별).
  위 줄 = AR · 아래 줄 = 릴리즈, 열 = 모양 · 색. 파랑 점선 = 상한 (encoder → encoder) · 주황 = encoder 판독기 → 예측 (진짜 물체 칸) ·
  초록 = encoder 판독기 → 예측 (마지막에 본 물체 칸) · 회색 = 예측 (진짜 물체 칸) 으로 학습한 판독기 → 같은 것 (정보가 있나) · 검정 점선 = 최빈 클래스.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_objcell_readout_by_pred_step.py
출력: z_analysis/figures/objcell_readout_by_pred_step.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/objcell_concept_compare.json"))
x = np.arange(1, 9)
LINES = [("enc_head_on_target", "upper bound: encoder head → encoder", "#2a78d6", "--", "o"), ("enc_head_on_pred_true", "encoder head → prediction, true object cells", "#eb6834", "-", "s"),
         ("enc_head_on_pred_last", "encoder head → prediction, last-seen object cells", "#1baf7a", "-", "D"), ("pred_head_on_pred_true", "prediction head → prediction, true object cells", "#777777", "-", "x")]
MODELS = [m for m in ("ar", "release") if m in J]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(len(MODELS), 2, figsize=(11, 4.2 * len(MODELS)), squeeze=False)
for r, model in enumerate(MODELS):
    for c, fam in enumerate(("shape", "color")):
        ax = axes[r, c]; d = J[model][fam]["readout"]
        for key, name, col, ls, mk in LINES:
            ax.plot(x, d[key], ls=ls, marker=mk, color=col, lw=2, ms=5, label=name)
        ax.axhline(d["val_majority"], color="black", ls=":", lw=1.0, label="majority class")
        ax.set_xticks(x); ax.set_ylim(0, 100); ax.set_xlabel("prediction step (block = 2 frames)"); ax.set_ylabel(f"{fam} accuracy (%)")
        ax.set_title(f"{'AR (block rollout)' if model == 'ar' else 'release (one-shot)'}, {fam}", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
for n, ax in enumerate(axes.flat):
    ax.text(0.5, -0.24, f"({'abcd'[n]})", transform=ax.transAxes, ha="center", fontsize=11)
fig.suptitle("Linear readout in the 1280-d output space, object-cell mean: encoder-trained head applied to predictor output", fontsize=12, y=0.995)   # 2026-10-26 사용자: 상단 가운데 제목 (어느 공간에서 했나)
fig.tight_layout(rect=(0, 0.08, 1, 0.96))
h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/objcell_readout_by_pred_step.png", dpi=170, bbox_inches="tight")
print("saved")
