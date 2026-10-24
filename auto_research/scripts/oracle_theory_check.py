#!/usr/bin/env python3
"""[T] 정리 3′ 이 [P] 의 "완벽한 흐름으로 옮기면 자연 영상 토큰 L1 이 나빠진다" 를 정량으로 예측하나 (학습 없음, CPU) (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/oracle_theory_check.py

[E] 캐시 (K710 · HT100M · 3D 합성, 블록 8 개 창) + 칸 흐름. 움직이는 칸 (|U[t+1]| ≥ 0.5 칸), t = 4, 5, 6.
예측기: 복사 h_t(c) · 흐름으로 옮긴 마지막 블록 h_t(c + U) (= 오라클 운반) · 같은 칸 5 블록 평균 m(c) ·
        회귀 최적 a*·m + κ·h_t(c+U) (맞춤 반 / 평가 반) · "옮긴 블록을 κ 만큼만 쓰고 나머지는 같은 칸 평균" 꼴 (회귀 최적과 같음).
제곱 오차 분해 (이론): 오라클 (a, b) = (0, 1) 의 초과 오차 = (β_o − β*)ᵀ A (β_o − β*) / n 를 L1 측정과 함께 낸다."""
import json, os
import numpy as np

EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


res = {}
for src in ("k710", "ht100m", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    A = np.zeros((2, 2)); bv = np.zeros(2); yy = 0.0; n = 0
    rows = []
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = np.asarray(T_[w, 0], np.float32)
            for t in (4, 5, 6):
                u1 = U[w0 + w, t + 1].reshape(S, 2)
                mov = np.linalg.norm(u1, axis=1) >= 0.5
                js, ok = cell_of(P0 + u1); sel = np.where(mov & ok)[0]
                if sel.size == 0:
                    continue
                m5 = h[t - 4:t + 1].mean(0)[sel]; mv = h[t][js[sel]]; cp = h[t][sel]; y = h[t + 1][sel]
                rows.append((w % 2, m5, mv, cp, y))
                if w % 2 == 0:
                    X = np.stack([m5.ravel(), mv.ravel()], 1).astype(np.float64)
                    A += X.T @ X; bv += X.T @ y.ravel(); yy += float((y.astype(np.float64) ** 2).sum()); n += y.size
        w0 += T_.shape[0]
    beta = np.linalg.solve(A, bv)
    sse_opt = yy - 2 * beta @ bv + beta @ A @ beta
    bo = np.array([0.0, 1.0]); sse_or = yy - 2 * bo @ bv + bo @ A @ bo
    l1 = {k: [0.0, 0] for k in ("copy", "oracle", "mean5", "opt")}
    for half, m5, mv, cp, y in rows:
        if half != 1:
            continue
        for k, p in (("copy", cp), ("oracle", mv), ("mean5", m5), ("opt", beta[0] * m5 + beta[1] * mv)):
            l1[k][0] += float(np.abs(p - y).sum()); l1[k][1] += y.size
    l1 = {k: v[0] / v[1] for k, v in l1.items()}
    r = {"a_opt": float(beta[0]), "kappa": float(beta[1]), "mse_opt": sse_opt / n, "mse_oracle": sse_or / n,
         "mse_excess_oracle_pct": 100 * (sse_or - sse_opt) / sse_opt, "l1": l1,
         "l1_oracle_vs_opt_pct": 100 * (l1["oracle"] - l1["opt"]) / l1["opt"], "l1_copy_vs_opt_pct": 100 * (l1["copy"] - l1["opt"]) / l1["opt"]}
    res[src] = r
    print(f"{src:7s}: 최적 (a, κ) = ({beta[0]:.3f}, {beta[1]:.3f}) · 움직이는 칸 제곱 오차 초과 (오라클 vs 최적) {r['mse_excess_oracle_pct']:+.1f} % · "
          f"L1: 최적 {l1['opt']:.4f} · 오라클 {l1['oracle']:.4f} ({r['l1_oracle_vs_opt_pct']:+.1f} %) · 복사 {l1['copy']:.4f} ({r['l1_copy_vs_opt_pct']:+.1f} %) · 같은 칸 평균 {l1['mean5']:.4f}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/oracle_theory_check.json"), "w"), indent=1)
