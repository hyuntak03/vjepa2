"""[물리] 흐름 지속성 보충: (1) 문맥 두 걸음 선형 예측 (x_t, x_{t−1} 내용 따라가기) 의 R², (2) 카메라 성분 (창의 칸 흐름 중앙값) 의 지속성, (3) 큰 움직임 (≥ 1 칸).
학습 없음, CPU. 정의는 flow_persistence.py 와 같다."""
import json, os, numpy as np
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G = 16
cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok
def r2_ls(Y, F):  # Y (n,2), F (n,2,p): 성분 공통 계수 최소제곱
    A = F.transpose(0, 2, 1).reshape(-1, F.shape[2]) if False else np.concatenate([F[:, 0, :], F[:, 1, :]], 0)
    y = np.concatenate([Y[:, 0], Y[:, 1]])
    half = len(y) // 2
    idx = np.random.default_rng(0).permutation(len(y)); tr, te = idx[:half], idx[half:]
    b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    return float(1 - ((y[te] - A[te] @ b) ** 2).sum() / (y[te] ** 2).sum()), b.round(3).tolist()
out = {}
for src in ("k710", "ht100m", "ssv2", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32); N = U.shape[0]; U = U.reshape(N, 8, G * G, 2)
    Y, X1, X2, big, camY, camX = [], [], [], [], [], []
    for w in range(N):
        for t in (4, 5, 6):
            u1, u0, um = U[w, t + 1], U[w, t], U[w, t - 1]
            camY.append(np.median(u1, 0)); camX.append(np.median(u0, 0))
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            s, ok = cell_of(P0 + u1); s2, ok2 = cell_of(P0[s] + u0[s])
            sel = np.where(mov & ok & ok2)[0]
            if sel.size == 0: continue
            Y.append(u1[sel]); X1.append(u0[s[sel]]); X2.append(um[s2[sel]]); big.append(np.linalg.norm(u1[sel], axis=1) >= 1.0)
    Y, X1, X2, big = map(np.concatenate, (Y, X1, X2, big))
    F1 = X1[:, :, None]; F2 = np.stack([X1, X2], 2)
    camY, camX = np.array(camY), np.array(camX)
    r1, b1 = r2_ls(Y, F1); r2, b2 = r2_ls(Y, F2); rb, bb = r2_ls(Y[big], F1[big])
    rc, bc = r2_ls(camY, camX[:, :, None])
    out[src] = dict(R2_one_step=round(r1, 3), k_one=b1, R2_two_step=round(r2, 3), k_two=b2, R2_big=round(rb, 3), k_big=bb,
                    cam_R2=round(rc, 3), cam_k=bc, n=int(len(Y)), n_big=int(big.sum()))
    print(src, out[src])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/flow_persistence_ctx.json"), "w"), indent=1)
