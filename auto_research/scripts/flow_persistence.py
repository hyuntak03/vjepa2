"""[물리] 흐름 공간의 "운반 신뢰" — 다음 블록 흐름을 문맥 흐름이 얼마나 맞히나 (학습 없음, CPU).

토큰 공간 κ (정답 h_{t+1}(c) 를 옮긴 h_t(s) 에 회귀한 계수, 자연 0.21 · 3D 합성 0.32) 와 같은 꼴을 흐름 공간에서 잰다.
U[b](c) = 블록 b 의 칸 c 내용이 블록 b-1 에서 있던 자리까지의 칸 변위 (RAFT, 칸 평균; natural_transport_ctx5 캐시).
움직이는 칸 c: |U[t+1](c)| >= 0.5 칸, t = 4,5,6 (token κ 와 같은 고르기).
 (a) 내용 따라가기: x = U[t](s), s = c + U[t+1](c) 의 칸 (같은 내용의 앞 걸음 변위; 고르기에 정답 흐름 = κ 와 같은 오라클)
 (b) 같은 칸 (문맥만): x = U[t](c)
y = U[t+1](c). 계수 k_L2 = Σ y·x / Σ x·x, k_L1 = argmin_k Σ|y − k x|_1 (점 손실 최적 계수), 상대 L1 = Σ|y − x|_1 / Σ|y|_1 (등속 외삽 vs "멈춤" 예측).
창 나누기: 카메라 움직임 = 블록 t+1 의 칸 흐름 크기 중앙값 > 0.5 칸.
"""
import json, os, numpy as np
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G = 16
cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)

def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok

def coef(Y, X):
    k2 = (Y * X).sum() / max((X * X).sum(), 1e-9)
    ks = np.linspace(0, 1.5, 151)
    l1 = [np.abs(Y - k * X).sum() for k in ks]
    k1 = ks[int(np.argmin(l1))]
    rel = np.abs(Y - X).sum() / max(np.abs(Y).sum(), 1e-9)
    relk = min(l1) / max(np.abs(Y).sum(), 1e-9)
    r2 = 1 - ((Y - k2 * X) ** 2).sum() / max((Y ** 2).sum(), 1e-9)
    cosd = np.mean((Y * X).sum(1) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(X, axis=1) + 1e-9) > 0.7071)
    return dict(n=int(len(Y)), k_L2=round(float(k2), 3), k_L1=round(float(k1), 3), R2=round(float(r2), 3),
                relL1_constvel=round(float(rel), 3), relL1_bestk=round(float(relk), 3), dir_within45=round(float(cosd), 3))

out = {}
for src in ("k710", "ht100m", "ssv2", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)  # (N,8,16,16,2)
    N = U.shape[0]; U = U.reshape(N, 8, G * G, 2)
    acc = {k: {"Y": [], "Xa": [], "Xb": []} for k in ("all", "cam_static", "cam_moving")}
    for w in range(N):
        for t in (4, 5, 6):
            u1, u0 = U[w, t + 1], U[w, t]
            m1 = np.linalg.norm(u1, axis=1)
            cam = np.median(m1) > 0.5
            mov = m1 >= 0.5
            s, ok = cell_of(P0 + u1)
            sel = np.where(mov & ok)[0]
            if sel.size == 0:
                continue
            for k in ("all", "cam_moving" if cam else "cam_static"):
                acc[k]["Y"].append(u1[sel]); acc[k]["Xa"].append(u0[s[sel]]); acc[k]["Xb"].append(u0[sel])
    out[src] = {}
    for k, d in acc.items():
        if not d["Y"]:
            continue
        Y = np.concatenate(d["Y"]); Xa = np.concatenate(d["Xa"]); Xb = np.concatenate(d["Xb"])
        out[src][k] = {"follow_content": coef(Y, Xa), "same_cell": coef(Y, Xb)}
    print(src, json.dumps(out[src]))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/flow_persistence.json"), "w"), indent=1)
