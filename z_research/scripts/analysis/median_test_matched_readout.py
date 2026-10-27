"""중앙값 시험 다시 — 속도 · 색에 같은 꼴의 판독기 (2026-10-27, CPU).
왜: velocity_lowrank_median.py 의 중앙값 시험은 속도 판독기 (전체 표본 → 연속 vx 회귀) 와 색 판독기 (빨강 · 파랑 두 무리 → ±1 회귀) 의 꼴이 달랐다.
    velocity_dim_snr.json 에서 차원별 분리도는 속도가 색보다 크게 나와 (중앙값 0.29 vs 0.23, 1 넘는 차원 43 vs 9) "속도는 차원마다 신호가 작아 중앙값 ≈ 평균" 설명이 맞지 않았다.
    → 판독기 꼴 차이가 결과를 만든 것인지 가른다.
판독기 셋을 두 속성에 똑같이: (A) 두 무리 ±1 최소제곱 (색에 쓰던 것) · (B) 두 무리 평균 차 방향으로 사영 · (C) 원래 판독기 (속도 = 전체 vx 회귀, 색 = (A)).
나머지 (묶음 31 개, p = 0 … 1, 반복 200, 순수 −1 / +1 눈금) 는 같다. 벡터 캐시 velocity_vectors (velocity_dim_snr.py).
  /venv/vjepa/bin/python z_research/scripts/analysis/median_test_matched_readout.py → z_analysis/results/median_test_matched_readout.json
"""
import json
import numpy as np
CA = "/workspace/auto_research_cache/velocity_vectors"; rng = np.random.default_rng(0)
tr, va = np.load(f"{CA}/train.npz"), np.load(f"{CA}/val.npz"); Xtr, Ltr = tr["X"], tr["L"]
X = np.concatenate([Xtr, va["X"]]); L = np.concatenate([Ltr, va["L"]])
grp = {"velocity": ((L[:, 0] >= 6) & (np.abs(L[:, 1]) < 4), (L[:, 0] <= -6) & (np.abs(L[:, 1]) < 4)), "color": (L[:, 5] == 0, L[:, 5] == 5)}
grp_tr = {"velocity": ((Ltr[:, 0] >= 6) & (np.abs(Ltr[:, 1]) < 4), (Ltr[:, 0] <= -6) & (np.abs(Ltr[:, 1]) < 4)), "color": (Ltr[:, 5] == 0, Ltr[:, 5] == 5)}
def ls_pm1(ma, mb):
    A = np.concatenate([Xtr[ma], Xtr[mb]]); y = np.r_[np.ones(ma.sum()), -np.ones(mb.sum())]
    w = np.linalg.lstsq(np.c_[A, np.ones(len(A))], y, rcond=None)[0]; return lambda v: v @ w[:-1] + w[-1]
def meandiff(ma, mb):
    u = Xtr[ma].mean(0) - Xtr[mb].mean(0); u /= np.linalg.norm(u); return lambda v: v @ u
wv = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], Ltr[:, 0], rcond=None)[0]; vx_reg = lambda v: v @ wv[:-1] + wv[-1]
R = {}
for nm in ("velocity", "color"):
    ma, mb = grp[nm]; Xp, Xn = X[ma], X[mb]
    reads = {"A_two_group_pm1": ls_pm1(*grp_tr[nm]), "B_mean_difference_direction": meandiff(*grp_tr[nm])}
    reads["C_original"] = vx_reg if nm == "velocity" else reads["A_two_group_pm1"]
    R[nm] = {}
    for rn, rd in reads.items():
        lo, hi = float(np.mean(rd(Xn))), float(np.mean(rd(Xp))); norm = lambda v: 2 * (v - lo) / (hi - lo) - 1
        res = {}
        for p in np.round(np.linspace(0, 1, 11), 1):
            med, mean = [], []
            for _ in range(200):
                k = int(round(p * 31)); G = np.concatenate([Xp[rng.choice(len(Xp), k, replace=False)], Xn[rng.choice(len(Xn), 31 - k, replace=False)]])
                med.append(norm(rd(np.median(G, 0)))); mean.append(norm(rd(G.mean(0))))
            res[str(p)] = {"median_vector": round(float(np.mean(med)), 3), "mean_vector": round(float(np.mean(mean)), 3)}
        R[nm][rn] = res
        print(f"{nm:8s} {rn:30s} p0.6/0.7/0.8 median " + " ".join(f"{res[q]['median_vector']:+.2f}" for q in ("0.6", "0.7", "0.8")) +
              " | mean " + " ".join(f"{res[q]['mean_vector']:+.2f}" for q in ("0.6", "0.7", "0.8")), flush=True)
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/median_test_matched_readout.json", "w"), indent=1)
