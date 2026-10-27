"""층별 판독기 이식 — 선 그래프 (2026-10-26). 판독기 = encoder 임베딩의 predictor 입력 투영 (N, 384) 으로만 학습 (예측 블록 8–14 마다 하나).
사용자: "그림이 너무 보기 어려운데 그래프로 그려주던가". 앞판 (층 × 블록 열지도) → z_analysis/figures/_superseded/layer_step_readout_transfer_heatmap.png
입력 z_analysis/results/layer_step_readout_transfer.json (layer_step_readout_transfer.py).
가로 = predictor 층 (input = 첫 층에 들어가기 직전, 표시 type 뺌 · 1–12 층 · norm = 마지막 정규화 = 상태). 세로 = 점수.
  파랑 = 실제 encoder 입력 경로 (teacher forcing) · 주황 = 굴린 예측 (입력 = 앞 자리 상태). 선 = 블록 8–14 평균, 띠 = 블록 사이 최소–최대.
  초록 점 = 굴린 예측 출력 (1280) 을 같은 입력 투영에 통과시킨 것 (판독기와 같은 공간), 블록 평균 ± 최소–최대.
  R² 는 [−1, 1] 로 자른다. −1 아래 값은 −1 에 속 빈 아래 삼각형으로.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_layer_step_readout_transfer.py
출력: z_analysis/figures/layer_step_readout_transfer.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/layer_step_readout_transfer.json"))
layers = ["input"] + [str(i) for i in range(1, 13)] + ["norm"]; x = np.arange(len(layers))
maj = np.mean([J["meta"]["blocks"][str(k)]["identity_val_majority"] for k in J["steps"]])
PANELS = [("identity", "shape × color accuracy (%)", (0, 45), None), ("velocity", "velocity R²", (-1.1, 1.0), -1.0), ("position", "position R²", (-1.1, 1.0), -1.0)]
LINES = [("encoder_input_path", "encoder input (teacher forcing)", "#2a78d6", -0.1), ("rollout", "rollout (state fed back)", "#eb6834", 0.1)]   # 가로 어긋남 (−1 에 겹치지 않게)
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (task, ylab, ylim, floor) in zip(axes, PANELS):
    for key, name, col, dx in LINES:
        xx = x + dx
        A = np.array(J[task][key], dtype=float)                       # (14 층, 7 블록)
        m, lo, hi = A.mean(1), A.min(1), A.max(1)
        if floor is not None:
            clipped = m < floor; m, lo, hi = np.maximum(m, floor), np.maximum(lo, floor), np.maximum(hi, floor)
        else:
            clipped = np.zeros(len(m), bool)
        ax.fill_between(xx, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(xx, m, color=col, lw=2, label=name)
        ax.plot(xx[~clipped], m[~clipped], "o", color=col, ms=5)
        ax.plot(xx[clipped], m[clipped], "v", mfc="white", mec=col, ms=7, mew=1.5)
    v = np.array(J[task]["rollout_output_projected"], dtype=float); vm = v.mean()
    vplot = max(vm, floor) if floor is not None else vm
    ax.errorbar([0.35], [vplot], yerr=[[vplot - max(v.min(), ylim[0])], [max(v.max(), ylim[0]) - vplot]], fmt="D" if (floor is None or vm >= floor) else "v",
                color="#1baf7a", mfc="#1baf7a" if (floor is None or vm >= floor) else "white", ms=7, capsize=3, lw=1.5, label="rollout output → input projection")
    if task == "identity":
        ax.axhline(maj, color="gray", ls=":", lw=1.2, label=f"majority class ({maj:.0f}%)")
    else:
        ax.axhline(0, color="gray", ls=":", lw=1.0)
        ax.plot([], [], "v", mfc="white", mec="gray", ms=7, mew=1.5, ls="none", label="below −1 (clipped)")
    ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=9); ax.set_xlim(-0.5, len(layers) - 0.5); ax.set_ylim(*ylim)
    ax.set_xlabel("predictor layer"); ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
    ax.text(0.5, -0.22, lab, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 1))
hs, ls = [], []
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls:
            hs.append(h_); ls.append(l_)
fig.legend(hs, ls, loc="lower center", ncol=5, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/layer_step_readout_transfer.png", dpi=170, bbox_inches="tight")
print("saved")
