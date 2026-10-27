"""굴린 예측 걸음별 개념 벡터 — (a) predictor vs encoder cos, (b) L2 크기 비 (predictor / encoder) (2026-10-26).
사용자: "x 축을 roll out step, y 축을 cos sim. figure (b) 는 y 축을 L2 norm (predictor / encoder) ratio 로 해서 파일 하나".
입력 z_analysis/results/output_concept_compare.json (output_concept_compare.py): 블록 전체 평균 (1, 1280), 같은 블록 · 같은 표본,
개념 = shape 6 · color 8 (클래스 평균 − 클래스 평균들의 평균) · velocity (vx, vy) · position (x, y) (라벨 1 px 당 평균 이동). 선 = 클래스 (축) 평균, 띠 = ±1 표준편차.

  /venv/vjepa/bin/python z_research/scripts/figures/plot_rollout_concept_cos_norm.py
출력: z_analysis/figures/rollout_concept_cos_norm.png
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--fams", default="shape,color,velocity,position"); ap.add_argument("--out", default="rollout_concept_cos_norm.png")
a = ap.parse_args()   # 2026-10-26 사용자: "position, velocity 는 제외하고" → --fams shape,color --out rollout_concept_cos_norm_identity.png
R = "/workspace/world_model/vjepa2/z_analysis"
D = json.load(open(f"{R}/results/output_concept_compare.json"))["ro"]
steps = list(range(1, 9))
FAM = [f for f in [("shape", "#2a78d6", "o"), ("color", "#eb6834", "s"), ("velocity", "#1baf7a", "^"), ("position", "#444444", "D")] if f[0] in a.fams.split(",")]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (key, ylab, ylim) in zip(axes, (("per_item_cos", "cosine (predictor vs encoder)", (0, 1.0)), ("per_item_ratio", "L2 norm ratio (predictor / encoder)", (0, 1.2)))):
    for fam, col, mk in FAM:
        A = np.array([D[str(k)][fam][key] for k in steps])               # (8, 클래스 수)
        m, s = A.mean(1), A.std(1)
        ax.plot(steps, m, marker=mk, color=col, lw=2, ms=5)
        ax.fill_between(steps, m - s, m + s, color=col, alpha=0.12, lw=0)
        ax.text(8.15, m[-1], fam, color=col, fontsize=9, va="center")
    if key == "per_item_ratio":
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(steps); ax.set_xlim(0.8, 9.0); ax.set_ylim(*ylim)
    ax.set_xlabel("rollout step"); ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
for ax, lab in zip(axes, ("(a)", "(b)")):
    ax.text(0.5, -0.27, lab, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout()
fig.savefig(f"{R}/figures/{a.out}", dpi=170, bbox_inches="tight")
print("saved", f"{R}/figures/{a.out}")
