"""[물리] 지름길 점검 (학습 없음 · CPU): 문맥 흐름만 보고 "자연 영상 vs 3D 합성" 을 가를 수 있나.
가를 수 있으면 물리층 입력을 흐름으로 제한해도 "합성처럼 보이면 관성" 지름길이 구조로 막히지 않는다.
특징 (움직이는 칸마다, 문맥만): 앞 변위 크기 |x|, 그 앞 |x′|, 등속 정도 r, 방향 변화 cos(x, x′), 카메라 움직임 (칸 흐름 중앙값 크기), 화면 흐름 크기 분산.
작은 로지스틱 회귀 (반 맞춤 · 반 평가) AUROC. 칸 단위 + 창 단위 (칸 특징 평균)."""
import os, json, numpy as np
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G = 16
cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok
def feats(src):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32); N = U.shape[0]; U = U.reshape(N, 8, G * G, 2)
    Fc, Fw = [], []
    for w in range(N):
        for t in (4, 5, 6):
            u0, um = U[w, t], U[w, t - 1]
            m0 = np.linalg.norm(u0, axis=1)
            mov = m0 >= 0.5                                  # 문맥에서 움직이는 칸 (미래 안 봄)
            s2, ok = cell_of(P0 + u0)
            sel = np.where(mov & ok)[0]
            if sel.size == 0: continue
            x, xp = u0[sel], um[s2[sel]]
            nx, nxp = np.linalg.norm(x, axis=1), np.linalg.norm(xp, axis=1)
            r = np.linalg.norm(x - xp, axis=1) / (nx + nxp + 1e-6)
            cs = (x * xp).sum(1) / (nx * nxp + 1e-6)
            cam = np.full(sel.size, np.median(m0)); var = np.full(sel.size, m0.std())
            f = np.stack([nx, nxp, r, cs, cam, var], 1)
            Fc.append(f); Fw.append(f.mean(0))
    return np.concatenate(Fc), np.stack(Fw)
def auroc(s, y):
    rk = s.argsort().argsort() + 1; npos = y.sum(); nneg = len(y) - npos
    return float((rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def logit_fit(X, y, it=300, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; Z = np.concatenate([Z, Z ** 2, np.ones((len(Z), 1))], 1)
    w = np.zeros(Z.shape[1]); wpos = 0.5 / y.mean(); wneg = 0.5 / (1 - y.mean())
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ ((p - y) * np.where(y == 1, wpos, wneg)) / len(y); w -= lr * g
    return lambda X2: np.concatenate([(X2 - mu) / sd, ((X2 - mu) / sd) ** 2, np.ones((len(X2), 1))], 1) @ w
out = {}
nat = [feats(s) for s in ("k710", "ht100m", "ssv2")]; syn = feats("phys3d")
for lvl, i in (("cell", 0), ("window", 1)):
    Xn = np.concatenate([n[i] for n in nat]); Xs = syn[i]
    X = np.concatenate([Xn, Xs]); y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xs))])
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    f = logit_fit(X[tr], y[tr]); out[lvl] = dict(n_nat=int(len(Xn)), n_syn=int(len(Xs)), auroc=round(auroc(f(X[te]), y[te]), 3))
    # 특징 하나씩
    out[lvl]["single"] = {nm: round(max(auroc(X[te][:, j], y[te]), 1 - auroc(X[te][:, j], y[te])), 3) for j, nm in enumerate(["|x|", "|x'|", "r", "cos", "cam", "var"])}
    print(lvl, out[lvl])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/flow_domain_leak.json"), "w"), indent=1)
