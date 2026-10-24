#!/usr/bin/env python3
"""[T] 가닥 2 정량 예측: "절반 운반" (기준 AR 의 옮긴 복사 계수 0.18 = 정답 최적 0.35 의 절반) 을 자연 영상의 데이터 최적 계수로 설명하나 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/transfer_beta.py

자연 영상 ([E] 수집 캐시, 블록 8 개 창, K710 · SSv2 · HT100M) 에서, 움직이는 칸 (실제 흐름 |U[t+1](c)| ≥ 0.5 칸) 의 정답 h_{t+1}(c) 를
  h_{t+1}(c) ≈ a · (같은 칸 5 블록 평균) + b · (h_t 를 어떤 변위로 옮긴 토큰)
으로 최소제곱 (모든 칸 · 차원 풀링, 맞춤 반 / 평가 반). 옮기는 변위 둘:
  참   = 실제 흐름 U[t+1](c) (정답 블록을 보고 아는 변위 — 3D 합성에서 [P] 가 쓴 "정답 변위" 와 같은 역할)
  지속 = 문맥 흐름 U[t](c) (문맥만으로 알 수 있는 등속 변위 — 자연 영상에서 predictor 가 쓸 수 있는 것)
예측: 자연 영상으로 학습한 predictor 는 자기 변위 추정에 b_지속 만큼의 비중을 배운다. 결정적 세계에 가져가면 옮긴 복사 계수는
  β_모델 ≈ β_정답(3D 합성) × b_지속 / b_참 (자연 영상).  [P] 측정: β_모델 0.18, β_정답 0.35 → 비 0.51."""
import json, os
import numpy as np

EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
BINS = [(0.5, 1), (1, 2), (2, 4), (0.5, 99)]


def warp(ht, u):                     # ht (256,D), u (256,2) 칸 단위: 정답 칸 c 의 내용 = ht[c + u] (가까운 칸, 화면 밖은 제자리)
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


import sys
SRCS = sys.argv[1:] or ["k710", "ssv2", "ht100m"]
OUT = os.path.join(os.path.dirname(__file__), "../results/transfer_beta" + ("" if not sys.argv[1:] else "_" + "_".join(SRCS)) + ".json")
def splat(ht, u0):                  # 라그랑주 등속 외삽: 블록 t 의 칸 s 내용을 s − U[t](s) (앞으로 같은 변위) 로 보낸다. 겹치면 평균, 빈 칸은 제자리
    d = np.floor(P0 - u0).astype(int)
    ok = (d >= 0).all(1) & (d < G).all(1)
    j = d[:, 1] * G + d[:, 0]
    acc = np.zeros_like(ht); n = np.zeros(S)
    np.add.at(acc, j[ok], ht[ok]); np.add.at(n, j[ok], 1)
    out = ht.copy(); m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


res = {}
for src in SRCS:
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)        # (W, 8, 16, 16, 2)
    acc = {}                                                                             # (bin, kind, half) → [XtX(2x2), Xty(2), yy, n]
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = np.asarray(T_[w, 0]).astype(np.float32)                                  # (8,256,D)
            Uw = U[w0 + w].reshape(8, S, 2)
            half = w % 2
            for t in (4, 5, 6):
                y = h[t + 1]; m = h[t - 4:t + 1].mean(0)
                mag = np.linalg.norm(Uw[t + 1], axis=1)
                for kind, u in (("true", Uw[t + 1]), ("pers", Uw[t]), ("lag", None)):
                    W_ = warp(h[t], u) if kind != "lag" else splat(h[t], Uw[t])
                    for lo, hi in BINS:
                        sel = (mag >= lo) & (mag < hi)
                        if not sel.any():
                            continue
                        X = np.stack([m[sel].ravel(), W_[sel].ravel()], 1).astype(np.float64); yy = y[sel].ravel().astype(np.float64)
                        k = (lo, hi, kind, half)
                        if k not in acc:
                            acc[k] = [np.zeros((2, 2)), np.zeros(2), 0.0, 0]
                        acc[k][0] += X.T @ X; acc[k][1] += X.T @ yy; acc[k][2] += yy @ yy; acc[k][3] += sel.sum()
        w0 += T_.shape[0]
    r = {}
    print(f"\n== {src}: 움직이는 칸 회귀 h_(t+1) ≈ a·(같은 칸 평균) + b·(옮긴 h_t)  [맞춤 = 짝수 창, 홀수 창은 확인용]")
    for lo, hi in BINS:
        row = {}
        for kind in ("true", "pers", "lag"):
            for half in (0, 1):
                k = (lo, hi, kind, half)
                if k not in acc:
                    continue
                A, bv, yy, n = acc[k]
                coef = np.linalg.solve(A, bv)
                row[f"{kind}_h{half}"] = dict(a=float(coef[0]), b=float(coef[1]), n_cells=int(n))
        if "true_h0" in row and "pers_h0" in row:
            ratio = [row[f"pers_h{hh}"]["b"] / row[f"true_h{hh}"]["b"] for hh in (0, 1) if f"pers_h{hh}" in row and row[f"true_h{hh}"]["b"] > 0]
            row["ratio_pers_over_true"] = ratio
            rl = [row[f"lag_h{hh}"]["b"] / row[f"true_h{hh}"]["b"] for hh in (0, 1) if f"lag_h{hh}" in row and row[f"true_h{hh}"]["b"] > 0]
            row["ratio_lag_over_true"] = rl
            print(f"  실제 속도 [{lo}, {hi}) 칸 {row['true_h0']['n_cells']}: b_참 {row['true_h0']['b']:.3f} / {row['true_h1']['b']:.3f} · "
                  f"b_지속(오일러) {row['pers_h0']['b']:.3f} / {row['pers_h1']['b']:.3f} · b_지속(라그랑주) {row['lag_h0']['b']:.3f} / {row['lag_h1']['b']:.3f} · "
                  f"비 (라그랑주/참) {' / '.join(f'{x:.2f}' for x in rl)}")
        r[f"{lo}-{hi}"] = row
    res[src] = r
json.dump(res, open(OUT, "w"), indent=1)
