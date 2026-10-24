#!/usr/bin/env python3
"""[T] (다) 확인 (학습 없음, CPU): 옮겨 붙인 토큰을 토큰 L1 로 채점하면, 토큰 L1 은 변위 u 를 어느 쪽으로 당기나 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/u_pull_check.py

[E] 캐시 (K710 · HT100M · 3D 합성, 블록 8 개 창) + RAFT 칸 흐름 (역방향 U[t+1]: 블록 t+1 칸 c 의 내용이 블록 t 의 c + U 에서 옴).
움직이는 칸 (|U| ≥ 0.5 칸), t = 4, 5, 6. 변위를 λ 배로 줄인 쌍선형 옮기기 warp(h_t, c + λ·U) 의 토큰 L1 (정답 h_{t+1}) 을 λ = 0 … 1.25 로.
  λ = 1 (정답 흐름) 이 최소가 아니면 → 옮겨 붙이기를 거친 토큰 L1 기울기가 u 를 정답 흐름에서 떼어 낸다 (방법 B 에서 u 도 끊어야 하는 근거).
같은 꼴로 '옮긴 토큰 κ 배 + 같은 칸 평균' (정리 3′ 최적 꼴) 에서도 λ 를 훑는다 (κ · a 는 λ 마다 맞춤 반에서 다시 맞춤)."""
import json, os
import numpy as np

EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
LAM = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
yy, xx = np.divmod(np.arange(S), G)


def bilin(h, px, py):                                                     # h (S, D), 칸 좌표 (가장자리 고정)
    px = np.clip(px, 0, G - 1); py = np.clip(py, 0, G - 1)
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int); x1 = np.minimum(x0 + 1, G - 1); y1 = np.minimum(y0 + 1, G - 1)
    wx = (px - x0)[:, None]; wy = (py - y0)[:, None]
    g = lambda yv, xv: h[yv * G + xv]
    return (1 - wy) * ((1 - wx) * g(y0, x0) + wx * g(y0, x1)) + wy * ((1 - wx) * g(y1, x0) + wx * g(y1, x1))


res = {}
for src in ("k710", "ht100m", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    l1 = {l: [0.0, 0] for l in LAM}
    fit = {l: [np.zeros((2, 2)), np.zeros(2)] for l in LAM}; rows = []
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = np.asarray(T_[w, 0], np.float32)
            for t in (4, 5, 6):
                u1 = U[w0 + w, t + 1].reshape(S, 2)
                sel = np.where(np.linalg.norm(u1, axis=1) >= 0.5)[0]
                if sel.size == 0:
                    continue
                y = h[t + 1][sel]; m5 = h[t - 4:t + 1].mean(0)[sel]
                wl = {l: bilin(h[t], xx[sel] + l * u1[sel, 0], yy[sel] + l * u1[sel, 1]) for l in LAM}
                for l in LAM:
                    l1[l][0] += float(np.abs(wl[l] - y).sum()); l1[l][1] += y.size
                    if w % 2 == 0:
                        X = np.stack([m5.ravel(), wl[l].ravel()], 1).astype(np.float64)
                        fit[l][0] += X.T @ X; fit[l][1] += X.T @ y.ravel()
                if w % 2 == 1:
                    rows.append((y, m5, wl))
        w0 += T_.shape[0]
    beta = {l: np.linalg.solve(*fit[l]) for l in LAM}
    l1s = {l: [0.0, 0] for l in LAM}
    for y, m5, wl in rows:
        for l in LAM:
            l1s[l][0] += float(np.abs(beta[l][0] * m5 + beta[l][1] * wl[l] - y).sum()); l1s[l][1] += y.size
    r = {"warp_l1": {str(l): v[0] / v[1] for l, v in l1.items()}, "shrunk_l1": {str(l): v[0] / v[1] for l, v in l1s.items()},
         "kappa": {str(l): float(beta[l][1]) for l in LAM}}
    r["argmin_warp"] = min(LAM, key=lambda l: r["warp_l1"][str(l)]); r["argmin_shrunk"] = min(LAM, key=lambda l: r["shrunk_l1"][str(l)])
    res[src] = r
    print(f"{src:7s} 옮기기만 L1: " + " · ".join(f"λ {l:.2f} {r['warp_l1'][str(l)]:.4f}" for l in LAM) + f"  → 최소 λ {r['argmin_warp']}", flush=True)
    print(f"{'':7s} 수축 꼴 L1: " + " · ".join(f"λ {l:.2f} {r['shrunk_l1'][str(l)]:.4f} (κ {r['kappa'][str(l)]:.2f})" for l in LAM) + f"  → 최소 λ {r['argmin_shrunk']}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/u_pull_check.json"), "w"), indent=1)
