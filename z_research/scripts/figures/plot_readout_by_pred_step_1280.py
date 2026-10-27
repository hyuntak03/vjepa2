"""예측 걸음별 판독기 이식 그림, 1280 공간 (2026-10-26). 판독기 = encoder 임베딩 (1280) 으로 학습 (블록마다 하나).
사용자: "1280 에서 비교. encoder embedding (1280), predictor output (1280), prediction step 을 x 축으로".
입력 z_analysis/results/readout_by_pred_step_1280.json (readout_by_pred_step_1280.py).
  파랑 점선 = 상한 (encoder 임베딩, val) · 주황 = AR predictor 출력 (1280) · 검정 점선 = 최빈 클래스. R² 는 −1 에서 자르고 그 아래는 속 빈 아래 삼각형.
대체: 384 판 (readout_by_pred_step.png) → z_analysis/figures/_superseded/
  /venv/vjepa/bin/python z_research/scripts/figures/plot_readout_by_pred_step_1280.py
출력: z_analysis/figures/readout_by_pred_step_1280.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/readout_by_pred_step_1280.json"))
x = np.arange(1, 9)
PANELS = [("identity", "shape × color accuracy (%)", (0, 100), None),   # 2026-10-26 사용자: 정확도는 100 까지
          ("velocity", "velocity R²", (-1.1, 1.0), -1.0), ("position", "position R²", (-1.1, 1.0), -1.0)]
LINES = [("upper_bound_encoder", "upper bound: encoder embedding (1280)", "#2a78d6", "--", "o"),
         ("predictor_output", "test: AR predictor output (1280)", "#eb6834", "-", "s")]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (task, ylab, ylim, floor) in zip(axes, PANELS):
    for key, name, col, ls, mk in LINES:
        v = np.array(J[task][key], dtype=float)
        c = v < floor if floor is not None else np.zeros(8, bool); vv = np.maximum(v, floor) if floor is not None else v
        ax.plot(x, vv, ls=ls, color=col, lw=2, label=name)
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
fig.tight_layout(rect=(0, 0.08, 1, 1))
hs, ls_ = [], []
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
fig.legend(hs, ls_, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/readout_by_pred_step_1280.png", dpi=170, bbox_inches="tight")
print("saved")
