"""릴리즈 predictor 판 — 예측 블록별 정체 (모양 · 색) 개념 벡터의 (a) cos (predictor vs 정답), (b) L2 크기 비 (predictor / 정답) (2026-10-26).
사용자: "(AR 그림) release 버전도 하나". AR 판 = plot_rollout_concept_cos_norm.py --fams shape,color (rollout_concept_cos_norm_identity.png).

입력: release_pooled_cache.py 출력 (블록마다 전체 평균 (1, 1280)) + ar_layers_pooled 라벨 (같은 클립 · 같은 순서는 i 로 맞춘다).
  정답 = 32 장 한 번에 LN(target_encoder) (릴리즈의 타깃 공간), 예측 = 문맥 16 장 → 미래 8 블록 한 번에 (걸음 k = 블록 7+k).
개념 벡터 (같은 블록 · 같은 표본으로 정답 · 예측 각각): c_k = 클래스 평균 − 클래스 평균들의 평균 (모양 6 · 색 8). 선 = 클래스 평균, 띠 = ±1 표준편차.

  /venv/vjepa/bin/python z_research/scripts/figures/plot_release_identity_cos_norm.py
출력: z_analysis/figures/rollout_concept_cos_norm_identity_release.png · z_analysis/results/release_identity_concept_compare.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
RP = np.load("/workspace/auto_research_cache/release_pooled/train.npz"); LP = np.load("/workspace/auto_research_cache/ar_layers_pooled/train.npz")
order = {int(v): n for n, v in enumerate(LP["i"])}
lab = LP["lab"][[order[int(v)] for v in RP["i"]]]
pred, gt = RP["pred"], RP["gt"]
FAM = {"shape": (1, 6, "#2a78d6", "o"), "color": (2, 8, "#eb6834", "s")}
res = {}
for k in range(1, 9):
    b = 7 + k; m = lab[:, b, 0] > 0; res[k] = {}
    for f, (col, K, _, _) in FAM.items():
        y = lab[m, b, col].astype(int)
        cp = np.stack([pred[m, k - 1][y == c].astype(np.float64).mean(0) for c in range(K)]); cp -= cp.mean(0)
        cg = np.stack([gt[m, b][y == c].astype(np.float64).mean(0) for c in range(K)]); cg -= cg.mean(0)
        cos = (cp * cg).sum(1) / (np.linalg.norm(cp, axis=1) * np.linalg.norm(cg, axis=1)); ratio = np.linalg.norm(cp, axis=1) / np.linalg.norm(cg, axis=1)
        res[k][f] = {"n": int(m.sum()), "per_item_cos": cos.round(3).tolist(), "per_item_ratio": ratio.round(3).tolist(), "cos": round(float(cos.mean()), 3), "ratio": round(float(ratio.mean()), 3)}
json.dump(res, open(f"{R}/results/release_identity_concept_compare.json", "w"), indent=1)
plt.rcParams["font.family"] = "DejaVu Sans"
steps = list(range(1, 9))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (key, ylab, ylim) in zip(axes, (("per_item_cos", "cosine (predictor vs target encoder)", (0, 1.0)), ("per_item_ratio", "L2 norm ratio (predictor / target encoder)", (0, 1.2)))):
    for f, (_, _, col, mk) in FAM.items():
        A = np.array([res[k][f][key] for k in steps]); mu, sd = A.mean(1), A.std(1)
        ax.plot(steps, mu, marker=mk, color=col, lw=2, ms=5); ax.fill_between(steps, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
        ax.text(8.15, mu[-1], f, color=col, fontsize=9, va="center")
    if key == "per_item_ratio":
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(steps); ax.set_xlim(0.8, 9.0); ax.set_ylim(*ylim); ax.set_xlabel("prediction step (block)"); ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)
for ax, lab_ in zip(axes, ("(a)", "(b)")):
    ax.text(0.5, -0.27, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(); fig.savefig(f"{R}/figures/rollout_concept_cos_norm_identity_release.png", dpi=170, bbox_inches="tight")
for f in FAM:
    print(f, "cos", [res[k][f]["cos"] for k in steps], "ratio", [res[k][f]["ratio"] for k in steps])
