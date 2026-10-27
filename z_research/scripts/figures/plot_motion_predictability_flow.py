"""학습 영상에서 지금 블록의 움직임이 다음 블록에 얼마나 이어지나 (2026-10-26).
입력: z_analysis/results/motion_predictability_flow.json (motion_predictability_flow.py). 값 = 따라간 칸 기준 r (median ρ), 움직이는 칸 |w1| ≥ 2 px.
  (a) 학습 영상 3 종: 0.25 s 뒤 (블록 사이 움직임) · 0.5 s 뒤 = 다음 블록 안 움직임 (예측 대상) · 같은 것 카메라 보정
  (b) 답을 아는 경우 (같은 RAFT · 같은 칸 평균, 0.5 s 뒤): 등속 패치 (3 속도 평균) · 멈춘 패치 · VOBench 보임 5 운동 (점 = 정답 라벨) · phys3d (라벨)
  /venv/vjepa/bin/python z_research/scripts/figures/plot_motion_predictability_flow.py
출력: z_analysis/figures/motion_predictability_flow.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/motion_predictability_flow.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

ax = axes[0]; srcs = [("k710", "Kinetics-710"), ("ssv2", "SSv2"), ("howto", "HowTo100M 1%")]
bars = [("raw_T2", "between_lag1", "0.25 s later (raw)"), ("raw_T2", "adv", "0.5 s later = next block (raw)"), ("cam_T2", "adv", "0.5 s later, camera motion removed")]
w = 0.26
for i, (mode, key, name) in enumerate(bars):
    v = [J[s][mode][key]["r"] for s, _ in srcs]
    b_ = ax.bar(np.arange(3) + (i - 1) * w, v, w, color=C[i], label=name)
    for bb, vv in zip(b_, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in srcs]); ax.set_xlabel("training videos\n(a)")
ax.legend(loc="center right", bbox_to_anchor=(1, 0.6), fontsize=8.5, frameon=False)   # 1.0 점선 아래 · 막대 위

ax = axes[1]
P = J["check_synthetic_patch"]; V = J["check_vobench_visible"]; mos = ["flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
names = ["constant\npatch", "stopping\npatch"] + [m.replace("_", " ") for m in mos] + ["phys3d\n(labels)"]
vals = [np.mean([P[f"constant_{s}px"]["adv"]["r"] for s in (4, 8, 16)]), np.mean([P[f"stop_{s}px"]["adv"]["r"] for s in (4, 8, 16)])] + [V[m]["raft"]["adv"]["r"] for m in mos] + [J["phys3d_gt"]["T2"]["r"]]
cols = ["#888888", "#888888"] + [C[0]] * len(mos) + ["#555555"]
b_ = ax.bar(range(len(vals)), vals, 0.6, color=cols)
for bb, vv in zip(b_, vals):
    inside = vv > 0.2                                                    # 1.0 점선과 겹치지 않게 값은 막대 안쪽 위에
    ax.text(bb.get_x() + bb.get_width() / 2, vv - 0.07 if inside else max(vv, 0) + 0.02, f"{abs(vv):.2f}" if abs(vv) < 0.005 else f"{vv:.2f}",
            ha="center", fontsize=8, color="white" if inside else "black")
ax.plot(np.arange(2, 2 + len(mos)), [V[m]["gt_r"] for m in mos], "k_", ms=18, mew=2, ls="none", label="VOBench ground truth (labels)")
ax.text(4, -0.2, "VOBench visible (RAFT)", ha="center", fontsize=9, color=C[0])
ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8); ax.set_xlabel("\n\nknown-answer checks, 0.5 s later\n(b)")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
for ax in axes:
    ax.set_ylim(-0.05, 1.38); ax.axhline(1, color="black", ls=":", lw=0.8); ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("share of current motion that continues\n(median, moving cells)"); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("How much of the current block's motion continues into the future (RAFT flow, 16×16 cells, 4 fps)", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{R}/figures/motion_predictability_flow.png", dpi=170, bbox_inches="tight")
print("saved")
