"""attentive probe (모양×색 · 속도 · 위치 한 번에) — 예측 걸음별, 블록 AR (2026-10-26).
사용자: "probe 를 attentive probe 로. color × shape 한 번에, velocity, position 까지" · "figure 전부 갱신".
입력: z_analysis/results/attnprobe_enc.json · attnprobe_ro.json (attnprobe_train.py). 판독기 = 블록 토큰 (256, 1280) 위 attentive probe (질의 3 개, 16 head, 7.0 M).
  파랑 점선 = 상한 (encoder 토큰으로 학습 → encoder val, 블록 8–15) · 주황 = encoder 로 학습한 판독기 → AR 예측 출력 토큰 ·
  초록 = AR 예측 출력 토큰으로 학습한 판독기 → AR 예측 출력 (예측 안에 정보가 있나) · 검정 점선 = 모양×색 최빈 클래스.
대체: readout_by_pred_step_1280.png (화면 평균 선형) · rollout_readout_pos_vel.png (화면 평균 선형 위치 · 속도) → z_analysis/figures/_superseded/
  /venv/vjepa/bin/python z_research/scripts/figures/plot_attnprobe_by_pred_step.py
출력: z_analysis/figures/attnprobe_by_pred_step.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
E, P = json.load(open(f"{R}/results/attnprobe_enc.json")), json.load(open(f"{R}/results/attnprobe_ro.json"))
x = np.arange(1, 9); g = lambda d, key: np.array([d[str(k)][key] for k in x], dtype=float)
DODGE = (-0.07, 0.0, 0.07)   # 선이 같은 값에서 포개지지 않게 가로로 어긋냄
LINES = [(E["test_encoder_val_blocks8to15_by_step"], "upper bound: encoder probe → encoder", "#2a78d6", "--", "o"),
         (E["test_rollout_val_by_step"], "encoder probe → AR prediction", "#eb6834", "-", "s"),
         (P["test_rollout_val_by_step"], "AR-prediction probe → AR prediction", "#1baf7a", "-", "D")]
PANELS = [("shape_x_color", "shape × color accuracy (%)", (0, 100), None), ("velocity_r2", "velocity R²", (-1.1, 1.0), -1.0)]   # 2026-10-26 사용자: 위치 패널 삭제 (마지막 본 위치가 거의 다 설명 — position_readout_meaningless.png)
yid = np.load("/workspace/auto_research_cache/attnprobe/val_ro_lab.npy"); maj = np.mean([np.bincount((yid[yid[:, 7] == b, 0] * 8 + yid[yid[:, 7] == b, 1]).astype(int)).max() / (yid[:, 7] == b).sum() * 100 for b in range(8, 16)])
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, (key, ylab, ylim, floor) in zip(axes, PANELS):
    for (d, name, col, ls, mk), dx in zip(LINES, DODGE):
        xx = x + dx; v = g(d, key); c = v < floor if floor is not None else np.zeros(8, bool); vv = np.maximum(v, floor) if floor is not None else v
        ax.plot(xx, vv, ls=ls, color=col, lw=2, label=name); ax.plot(xx[~c], vv[~c], mk, color=col, ms=5, ls="none"); ax.plot(xx[c], vv[c], "v", mfc="white", mec=col, ms=7, mew=1.5, ls="none")
    if key == "shape_x_color":
        ax.axhline(maj, color="black", ls=":", lw=1.0, label="majority class")
    else:
        ax.axhline(0, color="gray", ls=":", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(blk {7 + k})" for k in x], fontsize=8); ax.set_ylim(*ylim)
    ax.set_xlabel("prediction step"); ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
for ax, lab in zip(axes, ("(a)", "(b)")):
    ax.set_xlabel(ax.get_xlabel() + "\n" + lab)   # 2026-10-26 사용자: "겹치는 것들 모두 해결" — 패널 표시는 x 축 이름 아래 줄로 (자동 배치)
fig.suptitle("Attentive probe on 1280-d block tokens (256 × 1280): encoder vs AR predictor output", fontsize=12, y=0.995)   # 2026-10-26 사용자: 상단 가운데 제목 (어느 공간에서 했나)
fig.tight_layout(rect=(0, 0.08, 1, 0.96))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/attnprobe_by_pred_step.png", dpi=170, bbox_inches="tight")
print("saved")
