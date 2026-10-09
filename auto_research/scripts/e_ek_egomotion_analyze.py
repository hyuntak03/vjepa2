#!/usr/bin/env python3
"""e_ek_egomotion.py 산출물 → frozen encoder 토큰이 운동 (ego-motion · 칸별 flow · 미래 변위) 을 담는가. python e_ek_egomotion_analyze.py <dir>  (OUT_TAG)

ridge (닫힌 해; Gram 을 fold 마다 한 번만 만들고 모든 표적 · α 를 그 위에서 푼다), α 는 안쪽 3-fold 격자 {1e-2 … 1e3}·n,
**clip 5-fold** 교차검증 R² (성분별). 무작위 대조 = 표적 clip 셔플 (같은 Gram).
  (a) 전역: 토큰 평균 z̄ (1280) → gflow (2)               ego-motion 을 encoder 가 읽는가
  (b) 국소: 토큰 z[s] (1280) → flow[s] (2)                 칸별 문맥 끝 속도 (RollOutV2 09-19 의 z→v 실영상 판)
  (b') 국소 잔차: flow[s] − gflow                          물체 · 시차 운동 (전역 카메라 운동을 뺀 것)
  (c) 미래: 토큰 z[s] (1280) → fut[i][s] (2), i = 0..7     "칸 s 의 미래 내용이 어디서 오나" 를 문맥 끝 토큰 하나가 담는가
z (online, predictor 입력) 와 h (target, LN) 둘 다. (b)(c) 는 clip 당 64 칸 부표본.
2026-09-29 재작성: 이전 판은 표적 · α 마다 X 를 다시 표준화 · Gram 을 다시 만들어 (1,200 회 × 84 GFLOP) 로그인 노드에서 수 시간 걸렸다.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); n = meta["n"]; rng = np.random.default_rng(0)
Z = np.load(I / "z_last.npy", mmap_mode="r"); Hh = np.load(I / "h_last.npy", mmap_mode="r")
FL = np.load(I / "flow.npy"); GF = np.load(I / "gflow.npy"); FU = np.load(I / "fut.npy")
done = np.isfinite(GF[:, 0]) & np.isfinite(np.asarray(Z[:, 0, 0], dtype=np.float32)); idx = np.where(done)[0]; print("완료", len(idx), "/", n, flush=True)
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1000]


def r2(Y, P):
    ss = ((Y - Y.mean(0)) ** 2).sum(0); return 1 - ((Y - P) ** 2).sum(0) / np.maximum(ss, 1e-9)


def _gram(Xs, Y):
    return Xs.T @ Xs, Xs.T @ Y


def _solve(G, XY, a, n_tr, d):
    return np.linalg.solve(G + a * n_tr * np.eye(d), XY)


def cv_r2_multi(X, Y, clip, k=5):
    """X (N, d), Y (N, m): 모든 표적 열을 한 번에. α 는 열마다 안쪽 3-fold 로 고른다. return R² (m,), 예측 P (N, m)."""
    N, d = X.shape; m = Y.shape[1]
    uc = np.unique(clip); rng.shuffle(uc); folds = np.array_split(uc, k); P = np.zeros_like(Y, dtype=np.float64)
    for f in folds:
        te = np.isin(clip, f); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = ((X - mu) / sd).astype(np.float64); ym = Y[tr].mean(0); Yc = (Y - ym).astype(np.float64)
        # 안쪽 3-fold: 각 inner fold 의 Gram 을 한 번씩 → 모든 α · 열
        utr = np.unique(clip[tr]); inner = np.array_split(rng.permutation(utr), 3); sc = np.zeros((len(ALPHAS), m))
        for g in inner:
            ite = tr & np.isin(clip, g); itr = tr & ~ite
            G, XY = _gram(Xs[itr], Yc[itr])
            for ai, a in enumerate(ALPHAS):
                W = _solve(G, XY, a, itr.sum(), d); sc[ai] += r2(Yc[ite], Xs[ite] @ W)
        best = sc.argmax(0)                                                     # (m,)
        G, XY = _gram(Xs[tr], Yc[tr])
        for ai in np.unique(best):
            cols = np.where(best == ai)[0]; W = _solve(G, XY[:, cols], ALPHAS[ai], tr.sum(), d)
            P[np.ix_(te, cols)] = Xs[te] @ W + ym[cols]
    return r2(Y, P), P


res = dict(n=int(len(idx)), source=meta["source"])
for nm, T in (("z", Z), ("h", Hh)):
    # (a) 전역
    Xg = np.stack([np.asarray(T[c], dtype=np.float32).mean(0) for c in idx]); Yg = GF[idx]
    Ysh = Yg[rng.permutation(len(idx))]
    rg, _ = cv_r2_multi(Xg, np.concatenate([Yg, Ysh], 1), idx)
    res[f"{nm}_global_R2"] = dict(x=round(float(rg[0]), 3), y=round(float(rg[1]), 3), shuffled=round(float(rg[2:].mean()), 3), target_sd=[round(float(v), 2) for v in Yg.std(0)])
    # (b)(b')(c) 국소 — 한 X, 여러 표적
    Xl, Yl, Cl, Yf = [], [], [], []
    for c in idx:
        s = rng.choice(256, 64, replace=False); Xl.append(np.asarray(T[c][s], dtype=np.float32)); Yl.append(FL[c][s]); Cl.append(np.full(64, c)); Yf.append(FU[c][:, s])
    Xl, Yl, Cl = np.concatenate(Xl), np.concatenate(Yl), np.concatenate(Cl); Yf = np.concatenate(Yf, 1)                  # Yf (8, N, 2)
    Yres = Yl - GF[Cl]; Ysh = Yl[rng.permutation(len(Yl))]
    Yfut = np.nan_to_num(np.transpose(Yf, (1, 0, 2)).reshape(len(Yl), 16), nan=0.0)
    Yall = np.concatenate([Yl, Yres, Ysh, Yfut], 1)                                                                        # (N, 2+2+2+16)
    rl, _ = cv_r2_multi(Xl, Yall, Cl)
    res[f"{nm}_local_R2"] = dict(x=round(float(rl[0]), 3), y=round(float(rl[1]), 3), shuffled=round(float(rl[4:6].mean()), 3), target_sd=[round(float(v), 2) for v in Yl.std(0)])
    res[f"{nm}_local_residual_R2"] = dict(x=round(float(rl[2]), 3), y=round(float(rl[3]), 3), target_sd=[round(float(v), 2) for v in Yres.std(0)])
    res[f"{nm}_future_disp_R2_by_slot"] = [[round(float(rl[6 + 2 * i]), 3), round(float(rl[7 + 2 * i]), 3)] for i in range(8)]
    print(nm, "done", flush=True)
json.dump(res, open(OUT / f"ego_motion{TAG}.json", "w"), indent=1)
for k, v in res.items(): print(k, v)
