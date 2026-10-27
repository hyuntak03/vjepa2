"""encoder 정답으로 학습한 위치 · 속도 판독기를 굴린 예측에 적용 — R² (2026-10-26, 학습 없음, CPU).
사용자: "metric 을 R² 로" (앞판 rollout_readout_pos_vel.py 의 '블록 7 에서의 직선거리 + 16 px 이상만' 지표는 진자만 남겨 버렸다 → 버림).

특징: 블록마다 칸 256 개 평균 (1, 1280) — encoder 정답 · 굴린 예측 출력 (문맥 블록 0–7 = 16 장 → 걸음 k = 블록 7+k).
판독기: 블록 b (8–15) 마다, train 8,000 편의 encoder 정답만으로 선형 ridge. 위치 판독기 · 속도 판독기를 따로, λ 도 각각 (train 클립 80/20).
  위치 = 물체 중심 (x, y) px (블록 두 장 평균) · 속도 = 블록 안 이동 (dx, dy) px (블록 두 장 사이, 0.25 초)
표본: 물체가 보이는 블록 전부 (시나리오 · 움직임으로 고르지 않음).
지표: val 350 편, 걸음마다 R² = 1 − Σ|읽은 − 정답|² / Σ|정답 − 정답 평균|² (x · y 또는 dx · dy 를 묶어서).

  /venv/vjepa/bin/python z_research/scripts/analysis/rollout_readout_r2.py
출력: z_analysis/results/rollout_readout_r2.json · z_analysis/figures/rollout_readout_pos_vel.png (앞판 그림을 이 판으로 바꿈)
"""
import json
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"; R = "/workspace/world_model/vjepa2/z_analysis"
TR, VA = dict(np.load(f"{P}/train.npz")), dict(np.load(f"{P}/val.npz"))
LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
TGT = {"position": slice(6, 8), "velocity": slice(4, 6)}


def fit(X, Y, clip):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; tr = (clip % 5) != 0
    ev, V = np.linalg.eigh(Z[tr].T @ Z[tr]); ym = Y[tr].mean(0); ZtY = Z[tr].T @ (Y[tr] - ym)
    errs = [(((Z[~tr] @ (V @ ((V.T @ ZtY) / (ev + lam)[:, None])) + ym) - Y[~tr]) ** 2).sum() for lam in LAMS]
    lam = LAMS[int(np.argmin(errs))]
    ev, V = np.linalg.eigh(Z.T @ Z); ym = Y.mean(0); W = V @ ((V.T @ (Z.T @ (Y - ym))) / (ev + lam)[:, None])
    return (lambda Xn: ((Xn - mu) / sd) @ W + ym), lam


r2 = lambda P_, Y: float(1 - ((P_ - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
lab = VA["lab"]; out = {}
for t, sl in TGT.items():
    out[t] = {}
    for k in range(1, 9):
        b = 7 + k
        m_tr = TR["lab"][:, b, 0] > 0
        h, lam = fit(TR["enc"][m_tr, b].astype(np.float32), TR["lab"][m_tr, b, sl].astype(np.float64), np.nonzero(m_tr)[0])
        m = lab[:, b, 0] > 0; Y = lab[m, b, sl]
        out[t][k] = {"n_val": int(m.sum()), "lambda": lam, "encoder": round(r2(h(VA["enc"][m, b].astype(np.float32)), Y), 3),
                     "rollout": round(r2(h(VA["roout"][m, k - 1].astype(np.float32)), Y), 3)}
json.dump(out, open(f"{R}/results/rollout_readout_r2.json", "w"), indent=1)
for t in TGT:
    print(t, {k: (v["encoder"], v["rollout"], v["lambda"]) for k, v in out[t].items()})

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
ks = list(range(1, 9))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, t, title in zip(axes, TGT, ("position (x, y)", "velocity (dx, dy)")):
    for s_, col, mk, nm in (("encoder", "#1baf7a", "D", "encoder embedding -> head"), ("rollout", "#eb6834", "s", "rollout output -> head")):
        ax.plot(ks, [out[t][k][s_] for k in ks], marker=mk, color=col, lw=2, ms=5, label=nm)
    ax.axhline(0, color="gray", lw=0.8); ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xticks(ks); ax.set_xlabel("rollout step"); ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("R² (read vs ground truth)")
lo = min(min(out[t][k][s_] for k in ks for s_ in ("encoder", "rollout")) for t in TGT)
axes[0].set_ylim(min(-0.2, lo - 0.1), 1.05)
for ax, lab_ in zip(axes, ("(a)", "(b)")):
    ax.text(0.5, -0.24, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 1))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/rollout_readout_pos_vel.png", dpi=170, bbox_inches="tight")
print("saved")
