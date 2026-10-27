"""predictor 은닉 (384) 안의 정보 — 층별 · 예측 걸음별 attentive probe (모양×색 · 속도 · 위치 한 번에, 은닉 자체로 학습) (2026-10-26 저녁 갱신).
사용자: "(층별 그림) attentive probing 으로 진행해줘. shape × color 로 할 수 있잖아". 앞판 (선형, 화면 평균 / 물체 칸 평균) 의 수치는 predictor_info_by_layer.json 에 남는다.
입력: z_analysis/results/attnprobe384/{ro,tf}_L{0..13}.json (attnprobe384_train.py). 판독기 = 은닉 토큰 (256, 384) 위 attentive probe (질의 3 개, 16 head), 층마다 따로, 걸음 1–7 합쳐 학습.
가로 = predictor 층 (input · 1–12 · norm). 실선 = 굴린 예측, 걸음 1 (밝음) → 7 (어두움).
  회색 점선 = 상한: encoder 임베딩 → predictor 입력 투영 (384) 토큰으로 학습 · 시험한 probe (enc_L0.json, 걸음 평균; 2026-10-26 사용자: teacher forcing 대신). 검정 점선 = 모양×색 최빈 클래스.
  (a) 모양 × 색 정확도 · (b) 속도 R² · (c) 위치 R². R² 는 −1 에서 자른다.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_predictor_info_by_layer.py
출력: z_analysis/figures/predictor_info_by_layer.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
LAYERS = ["input"] + [str(i) for i in range(1, 13)] + ["norm"]; steps = list(range(1, 8)); x = np.arange(14)
J = {"ro": [json.load(open(f"{R}/results/attnprobe384/ro_L{l}.json")) for l in range(14)]}
UB = json.load(open(f"{R}/results/attnprobe384/enc_L0.json"))
A = lambda p, key: np.array([[J[p][l]["val_by_step"][str(k)][key] for k in steps] for l in range(14)], dtype=float)   # (14 층, 7 걸음)
lab = np.load("/workspace/auto_research_cache/attnprobe384/val_lab.npy")
maj = np.mean([np.bincount((lab[lab[:, 7] == k, 0] * 8 + lab[lab[:, 7] == k, 1]).astype(int)).max() / (lab[:, 7] == k).sum() * 100 for k in steps])
cols = plt.cm.viridis(np.linspace(0.9, 0.05, len(steps)))
PANELS = [("shape_x_color", "shape × color accuracy (%)", (0, 100), None), ("velocity_r2", "velocity R²", (-1.1, 1.0), -1.0)]   # 2026-10-26 사용자: 위치 패널 삭제 (position_readout_meaningless.png)
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, (key, ylab, ylim, floor) in zip(axes, PANELS):
    ro = A("ro", key); ub = np.mean([UB["val_by_step"][str(k)][key] for k in steps])
    if floor is not None:
        ro, ub = np.maximum(ro, floor), max(ub, floor)
    for j, k in enumerate(steps):
        ax.plot(x + (j - 3) * 0.05, ro[:, j], "-o", color=cols[j], lw=1.8, ms=3.5, label=f"rollout, step {k} (blk {7 + k})")   # 걸음마다 가로로 조금 어긋냄 (겹침 방지)
    ax.axhline(ub, color="#666666", ls="--", lw=2, label="upper bound: encoder embedding → predictor input projection (384), mean over steps")
    if key == "shape_x_color":
        ax.axhline(maj, color="black", ls=":", lw=1.0, label="majority class")
    else:
        ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS, fontsize=9); ax.set_ylim(*ylim)
    ax.set_xlabel("predictor layer"); ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
for ax, lab_ in zip(axes, ("(a)", "(b)")):
    ax.set_xlabel(ax.get_xlabel() + "\n" + lab_)   # 2026-10-26 사용자: "겹치는 것들 모두 해결" — 패널 표시는 x 축 이름 아래 줄로 (자동 배치)
fig.suptitle("Attentive probe on AR predictor hidden tokens, 384-d per layer (256 × 384); probes trained on the hidden states", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0.2, 1, 0.95))
hs, ls_ = [], []
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
fig.legend(hs, ls_, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/predictor_info_by_layer.png", dpi=170, bbox_inches="tight")
print("saved")
