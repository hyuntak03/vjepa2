#!/usr/bin/env python3
"""[L] 운반 신뢰 b 의 L1 최적 판 (auto_research, 2026-10-22 KST 16 시).

물음: "블록 AR 은 데이터가 보상하는 운반의 6–26 % 만 옮긴다" 의 데이터 쪽 기준 b 는 최소제곱 회귀 (= 선형 꼴 안의 L2 최적) 다.
블록 AR 은 L1 (칸 · 차원별 중앙값) 로 학습한다. 같은 선형 꼴 안의 **L1 최적** b (최소절대편차 회귀) 가 모델 b 근처라면,
모델은 L1 최적에 있고 간격은 학습 실패가 아니라 L1 과 L2 최적의 차이다. L2 최적 근처라면 학습 실패가 선다.

  OMP_NUM_THREADS=32 /venv/vjepa/bin/python -W ignore auto_research/scripts/lad_transport.py

정의 (transport_placebo.py 와 같다): 블록 8 개 창, t = 4 · 5 · 6, 움직이는 칸 |U[t+1]| ≥ 0.5, m = 같은 칸 블록 t−4..t 평균,
W_true = h_t 를 U[t+1] 로 옮김, W_rot = 90° 돌린 흐름으로 옮김 (위약). y = h_{t+1} (데이터) · h_t (복사 정답, 0 이어야).
L2 = 최소제곱, L1 = 최소절대편차 (IRLS). 위약 뺀 b = b(W_true) − b(W_rot). L1 운반 보상 = (L1(m 만) − L1(m, W)) / L1(m 만)."""
import json, os
import numpy as np, torch

torch.set_num_threads(32)
CACHE = "/workspace/auto_research_cache/ssv2_classes"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
rng = np.random.RandomState(0)


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def fit(y, cols, l1, iters=40):
    X = torch.stack(cols, 1)                                     # (N, k)
    w = torch.ones_like(y)
    for _ in range(iters if l1 else 1):
        Xw = X * w[:, None]
        beta = torch.linalg.solve(Xw.T @ X, Xw.T @ y)
        if l1:
            w = 1.0 / (y - X @ beta).abs().clamp_min(1e-3)
    r = y - X @ beta
    return beta.tolist(), float(r.abs().mean()), float((r ** 2).mean())


res = {}
for grp in ("passive", "actor"):
    Hs = np.load(os.path.join(CACHE, f"{grp}_h.npy"), mmap_mode="r"); Us = np.load(os.path.join(CACHE, f"{grp}_U.npy")).astype(np.float32)
    Y, Cp, M, Wt, Wr, WIN = [], [], [], [], [], []
    for w in range(Hs.shape[0]):
        h = np.asarray(Hs[w]).astype(np.float32); U = Us[w].reshape(-1, S, 2)
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            Y.append(h[t + 1][sel]); Cp.append(h[t][sel]); M.append(h[t - 4:t + 1].mean(0)[sel])
            Wt.append(warp(h[t], U[t + 1])[sel]); Wr.append(warp(h[t], U[t + 1] @ R90.T)[sel]); WIN.append(np.full(sel.sum(), w))
    T_ = lambda L: torch.from_numpy(np.concatenate(L).ravel())
    y, cp, m, wt, wr = T_(Y), T_(Cp), T_(M), T_(Wt), T_(Wr)
    print(f"{grp}: 점 {y.numel():,}", flush=True)
    e = {}
    for nm, yy in (("data", y), ("copy", cp)):
        for loss in ("L2", "L1"):
            l1 = loss == "L1"
            bt, L1t, L2t = fit(yy, [m, wt], l1)
            br, _, _ = fit(yy, [m, wr], l1)
            b0, L10, L20 = fit(yy, [m], l1)
            e[f"{nm}_{loss}"] = dict(b_true=bt[1], b_rot=br[1], net=bt[1] - br[1], a_true=bt[0],
                                     reward_L1=(L10 - L1t) / L10, reward_L2=(L20 - L2t) / L20)
            print(f"  {nm:4s} {loss}: b_true {bt[1]:.4f} · b_rot {br[1]:.4f} · 위약 뺀 {bt[1] - br[1]:.4f} · "
                  f"운반 보상 L1 {100 * (L10 - L1t) / L10:.2f} % · L2 {100 * (L20 - L2t) / L20:.2f} %", flush=True)
    # 데이터 L1 판의 창 반반 흔들림 (bootstrap 대신 반쪽 둘)
    win = torch.from_numpy(np.concatenate(WIN).repeat(1280))
    uw = np.unique(np.concatenate(WIN)); rng.shuffle(uw); half = torch.from_numpy(np.isin(np.concatenate(WIN), uw[: len(uw) // 2]).repeat(1280))
    for k, msk in (("half_A", half), ("half_B", ~half)):
        bt, _, _ = fit(y[msk], [m[msk], wt[msk]], True); br, _, _ = fit(y[msk], [m[msk], wr[msk]], True)
        e[f"data_L1_{k}_net"] = bt[1] - br[1]
        print(f"  data L1 {k}: 위약 뺀 {bt[1] - br[1]:.4f}", flush=True)
    res[grp] = e
out = os.path.join(os.path.dirname(__file__), "../results/lad_transport.json")
json.dump(res, open(out, "w"), indent=1)
print("저장:", os.path.abspath(out))
