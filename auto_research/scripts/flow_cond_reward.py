"""[물리] 흐름 공간의 조건화 보상 — 토큰 공간 0.13 % (고르기 최대 0.38 %, 장난감 문턱 8–13 %) 와 같은 정의로 (학습 없음, CPU).

y = U[t+1](c) (움직이는 칸, |y| >= 0.5 칸), x = 같은 내용의 앞 걸음 변위 U[t](s), s = c + y 의 칸 (κ 와 같은 오라클 고르기).
문맥만으로 정하는 신뢰 특징 (미래 안 봄):
  r = |x − x′| / (|x| + |x′|), x′ = U[t−1](s′), s′ = s + x 의 칸  (문맥 안 등속 정도: 0 = 등속)
  cam = 블록 t 의 칸 흐름 크기 중앙값 > 0.5 칸 (문맥 카메라 움직임)
구간: r 5 분위 (자연 영상 통합 경계) × cam 2 = 10 구간.
보상 (모두 움직이는 칸의 전역 회귀 잔차 제곱합 대비 %): 운반 보상 = SSE(k=0) − SSE(전역 k); 조건화 보상 = SSE(전역 k) − SSE(구간별 k).
"""
import json, os, numpy as np
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G = 16
cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)

def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok

def collect(src):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    N = U.shape[0]; U = U.reshape(N, 8, G * G, 2)
    Y, X, R, C = [], [], [], []
    for w in range(N):
        for t in (4, 5, 6):
            u1, u0, um = U[w, t + 1], U[w, t], U[w, t - 1]
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            cam = np.median(np.linalg.norm(u0, axis=1)) > 0.5
            s, ok = cell_of(P0 + u1)
            s2, ok2 = cell_of(P0[s] + u0[s])
            sel = np.where(mov & ok & ok2)[0]
            if sel.size == 0:
                continue
            x = u0[s[sel]]; xp = um[s2[sel]]
            r = np.linalg.norm(x - xp, axis=1) / (np.linalg.norm(x, axis=1) + np.linalg.norm(xp, axis=1) + 1e-6)
            Y.append(u1[sel]); X.append(x); R.append(r); C.append(np.full(sel.size, cam))
    return np.concatenate(Y), np.concatenate(X), np.concatenate(R), np.concatenate(C)

def fitk(Y, X):
    return (Y * X).sum() / max((X * X).sum(), 1e-9)

def sse(Y, X, k):
    return ((Y - k * X) ** 2).sum()

D = {s: collect(s) for s in ("k710", "ht100m", "ssv2", "phys3d")}
edges = np.quantile(np.concatenate([D[s][2] for s in ("k710", "ht100m", "ssv2")]), [0.2, 0.4, 0.6, 0.8])
out = {"r_edges": edges.tolist()}
for name, srcs in (("natural", ("k710", "ht100m", "ssv2")), ("k710", ("k710",)), ("ht100m", ("ht100m",)), ("ssv2", ("ssv2",)), ("phys3d", ("phys3d",)),
                   ("natural+phys3d_25pct", None)):
    if srcs is None:  # 자연 영상에 3D 합성 칸을 25 % 비율로 섞은 경우 (학습 섞기 비율)
        Yn, Xn, Rn, Cn = [np.concatenate([D[s][i] for s in ("k710", "ht100m", "ssv2")]) for i in range(4)]
        Yp, Xp, Rp, Cp = D["phys3d"]
        rep = max(1, int(round(0.25 * len(Yn) / len(Yp))))
        Y, X, R, C = (np.concatenate([Yn] + [Yp] * rep), np.concatenate([Xn] + [Xp] * rep), np.concatenate([Rn] + [Rp] * rep), np.concatenate([Cn] + [Cp] * rep))
    else:
        Y, X, R, C = [np.concatenate([D[s][i] for s in srcs]) for i in range(4)]
    kg = fitk(Y, X); Sg = sse(Y, X, kg); S0 = sse(Y, X, 0.0)
    b = np.digitize(R, edges)
    Sb = 0.0; bins = []
    for cam in (False, True):
        for q in range(5):
            m = (b == q) & (C == cam)
            if m.sum() < 30:
                continue
            kb = fitk(Y[m], X[m]); Sb += sse(Y[m], X[m], kb)
            bins.append(dict(cam=bool(cam), r_bin=q, n=int(m.sum()), share=round(float(m.mean()), 3), k=round(float(kb), 3)))
    # 구간에 안 든 칸은 전역 k 로
    used = np.zeros(len(Y), bool)
    for bb in bins:
        used |= (b == bb["r_bin"]) & (C == bb["cam"])
    Sb += sse(Y[~used], X[~used], kg)
    out[name] = dict(n=int(len(Y)), k_global=round(float(kg), 3), transport_pct=round(float(100 * (S0 - Sg) / Sg), 2),
                     cond_pct=round(float(100 * (Sg - Sb) / Sg), 2), bins=bins)
    print(name, {k: v for k, v in out[name].items() if k != "bins"})
    for bb in bins:
        print("   ", bb)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/flow_cond_reward.json"), "w"), indent=1)
