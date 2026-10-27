"""예측 걸음별 판독기 이식 그림 (2026-10-26). 판독기 = encoder 임베딩 → 입력 투영 (N, 384) 으로 학습 (블록마다 하나).
사용자: "y 축 점선 하나 (encoder embedding 상한), 나머지는 prediction block test".
입력 z_analysis/results/readout_by_pred_step.json (readout_by_pred_step.py). 가로 = 예측 걸음 1–8 (블록 8–15).
  파랑 점선 = 상한 (encoder 임베딩 → 입력 투영, val) · 주황 = predictor 출력 (384, 마지막 정규화 상태) ·
  초록 = 예측 출력 (1280) → 같은 입력 투영 (판독기와 같은 좌표계) · 회색 가는 선 = 실제 encoder 를 입력으로 넣은 경로의 같은 384 출력 (좌표계 대조).
  R² 는 −1 에서 자르고 그 아래는 속 빈 아래 삼각형.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_readout_by_pred_step.py
출력: z_analysis/figures/readout_by_pred_step.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/readout_by_pred_step.json"))
x = np.arange(1, 9)
PANELS = [("identity", "shape × color accuracy (%)", (0, 45), None), ("velocity", "velocity R²", (-1.1, 1.0), -1.0), ("position", "position R²", (-1.1, 1.0), -1.0)]
LINES = [("upper_bound_encoder", "upper bound: encoder embedding", "#2a78d6", "--", "o", 2.0),
         ("pred_output_via_input_proj", "prediction: output (1280) → input projection", "#1baf7a", "-", "D", 2.0),
         ("predictor_output_state", "prediction: predictor output (384 state)", "#eb6834", "-", "s", 2.0),
         ("encoder_path_state", "reference: predictor output (384) with real encoder input", "#888888", "-", "x", 1.0)]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (task, ylab, ylim, floor) in zip(axes, PANELS):
    for key, name, col, ls, mk, lw in LINES:
        v = np.array(J[task][key], dtype=float)
        if floor is None:
            ax.plot(x, v, ls=ls, marker=mk, color=col, lw=lw, ms=5, label=name)
        else:
            c = v < floor; vv = np.maximum(v, floor)
            ax.plot(x, vv, ls=ls, color=col, lw=lw, label=name)
            ax.plot(x[~c], vv[~c], mk, color=col, ms=5, ls="none")
            ax.plot(x[c], vv[c], "v", mfc="white", mec=col, ms=7, mew=1.5, ls="none")
    if task == "identity":
        ax.plot(x, J["identity_val_majority"], ":", color="black", lw=1.0, label="majority class")
    else:
        ax.axhline(0, color="gray", ls=":", lw=1.0)
        ax.plot([], [], "v", mfc="white", mec="gray", ms=7, mew=1.5, ls="none", label="below −1 (clipped)")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(blk {7 + k})" for k in x], fontsize=8); ax.set_ylim(*ylim)
    ax.set_xlabel("prediction step"); ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
    ax.text(0.5, -0.27, lab, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.12, 1, 1))
hs, ls_ = [], []
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
fig.legend(hs, ls_, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/readout_by_pred_step.png", dpi=170, bbox_inches="tight")
print("saved")
