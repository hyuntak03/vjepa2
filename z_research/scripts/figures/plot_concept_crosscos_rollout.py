"""굴린 예측 출력의 개념 벡터가 encoder 의 어느 개념 벡터와 같은 방향인가 — 18 × 18 cos (2026-10-26, 학습 없음, CPU).
사용자: "속도랑 위치는 다른 concept vector 랑 cos sim 비슷한지 봐바. 다르게 표현된 걸 수도 있으니까. figure 로 명확하게 (roll out 버전만)".

개념 벡터 (output_concept_compare.py 와 같은 정의, 블록 전체 평균 (1, 1280), 같은 블록 · 같은 표본): 모양 6 · 색 8 (클래스 평균 − 클래스 평균들의 평균) ·
vx · vy (블록 안 변위) · x · y (물체 중심) (라벨 1 px 당 특징 평균 이동). 굴린 예측 걸음 k → 블록 7+k, 문맥 16 장 · 예측 16 장.
그림 1: 걸음 1 · 2 · 4 · 8 의 cos(predictor 개념 i, encoder 개념 j) + 기준 encoder–encoder (블록 8).
그림 2: vx · vy · x · y 마다 걸음별 — 같은 개념 encoder 와의 cos vs 다른 encoder 개념 중 |cos| 최대 (그 개념 이름).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_concept_crosscos_rollout.py
출력: z_analysis/figures/concept_crosscos_rollout.png · concept_crosscos_rollout_motion.png · z_analysis/results/concept_crosscos_rollout.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/workspace/auto_research_cache/ar_layers_pooled"; R = "/workspace/world_model/vjepa2/z_analysis"
TR = dict(np.load(f"{P}/train.npz")); lab = TR["lab"]; ok = lab[..., 0] > 0
SHAPES = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]
COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
NAMES = SHAPES + COLORS + ["vx", "vy", "x", "y"]


def concepts(X, L):
    rows = []
    for col, K in ((1, 6), (2, 8)):
        y = L[:, col].astype(int); M = np.stack([X[y == k].mean(0) for k in range(K)]); rows.append(M - M.mean(0))
    for c0 in (4, 6):
        Y = L[:, c0:c0 + 2].astype(np.float64); Yc = Y - Y.mean(0); Xc = X - X.mean(0)
        rows.append(((Xc.T @ Yc / len(Y)) @ np.linalg.inv(Yc.T @ Yc / len(Y))).T)
    C = np.concatenate(rows); return C / np.linalg.norm(C, axis=1, keepdims=True)


S, Eref = {}, None
for k in range(1, 9):
    b = 7 + k; m = ok[:, b]
    Ce = concepts(TR["enc"][m, b].astype(np.float32), lab[m, b]); Cp = concepts(TR["roout"][m, k - 1].astype(np.float32), lab[m, b])
    S[k] = Cp @ Ce.T
    if k == 1:
        Eref = Ce @ Ce.T
json.dump({"names": NAMES, "pred_x_enc": {k: np.round(v, 3).tolist() for k, v in S.items()}, "enc_x_enc_block8": np.round(Eref, 3).tolist()},
          open(f"{R}/results/concept_crosscos_rollout.json", "w"))

plt.rcParams["font.family"] = "DejaVu Sans"
# ---- 그림 1: 히트맵
steps = (1, 2, 4, 8)
fig, axes = plt.subplots(1, 5, figsize=(26, 5.8))
mats = [("encoder vs encoder (block 8)", Eref)] + [(f"rollout step {k}: predictor vs encoder", S[k]) for k in steps]
for ax, (title, A) in zip(axes, mats):
    im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(18)); ax.set_xticklabels(NAMES, rotation=90, fontsize=7); ax.set_yticks(range(18)); ax.set_yticklabels(NAMES, fontsize=7)
    for g in (5.5, 13.5):
        ax.axhline(g, color="k", lw=0.8); ax.axvline(g, color="k", lw=0.8)
    ax.set_xlabel("encoder concept", fontsize=8); ax.set_ylabel("encoder concept" if "vs encoder (" in title else "predictor concept", fontsize=8)
    ax.set_title(title, fontsize=9)
fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="cosine")
fig.savefig(f"{R}/figures/concept_crosscos_rollout.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 그림 2: 속도 · 위치 — 같은 개념 vs 가장 닮은 다른 개념
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
xs = list(range(1, 9))
for ax, name in zip(axes, ["vx", "vy", "x", "y"]):
    i = NAMES.index(name)
    same = [S[k][i, i] for k in xs]
    other_idx = [j for j in range(18) if j != i]
    best = [max(other_idx, key=lambda j: abs(S[k][i, j])) for k in xs]
    bestv = [S[k][i, j] for k, j in zip(xs, best)]
    ax.plot(xs, same, "o-", color="#2a78d6", lw=2, label=f"same concept (enc {name})")
    ax.plot(xs, bestv, "s--", color="#eb6834", lw=2, label="most similar other enc concept")
    for k, j, v in zip(xs, best, bestv):
        ax.annotate(NAMES[j], (k, v), textcoords="offset points", xytext=(0, -12 if v < same[k - 1] else 6), ha="center", fontsize=7, color="#eb6834")
    ref = sorted(((abs(Eref[i, j]), NAMES[j]) for j in other_idx), reverse=True)[0]
    ax.axhline(ref[0], color="gray", lw=1, ls=":")
    ax.text(8.1, ref[0], f"enc-enc max |cos|\n({ref[1]})", fontsize=7, color="gray", va="center")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"predictor {name}", fontsize=10); ax.set_xlabel("rollout step", fontsize=9); ax.set_xticks(xs); ax.set_ylim(-1, 1)
    if name == "vx":
        ax.set_ylabel("cosine with encoder concept", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(f"{R}/figures/concept_crosscos_rollout_motion.png", dpi=150); plt.close(fig)

for name in ("vx", "vy", "x", "y"):
    i = NAMES.index(name)
    print(name, "same:", [round(float(S[k][i, i]), 2) for k in range(1, 9)])
    for k in (1, 4, 8):
        top = sorted(((float(S[k][i, j]), NAMES[j]) for j in range(18) if j != i), key=lambda t: -abs(t[0]))[:3]
        print("   step", k, "top other:", [(n, round(v, 2)) for v, n in top])
    print("   enc-enc (block 8) top other:", sorted(((round(float(Eref[i, j]), 2), NAMES[j]) for j in range(18) if j != i), key=lambda t: -abs(t[0]))[:3])
print("saved")
