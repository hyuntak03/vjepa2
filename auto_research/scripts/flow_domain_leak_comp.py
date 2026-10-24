"""[물리] 지름길 점검 2 (학습 없음 · CPU): 카메라 보정 흐름 (블록마다 화면 칸 흐름 중앙값을 뺌) 으로 바꾸면 "자연 vs 3D 합성" 이 얼마나 가려지나.
flow_domain_leak.py 와 같은 특징 · 같은 로지스틱 회귀. 보정 뒤 '움직이는 칸' 은 보정 흐름 크기 ≥ 0.5 칸 (= 화면에 대해 따로 움직이는 내용)."""
import os, json, numpy as np
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
src0 = open(os.path.join(os.path.dirname(__file__), "flow_domain_leak.py")).read().split("def feats")[0]
ns = {"__file__": os.path.join(os.path.dirname(__file__), "flow_domain_leak.py")}; exec(src0, ns)
G = 16; P0 = ns["P0"]; cell_of = ns["cell_of"]
full = open(os.path.join(os.path.dirname(__file__), "flow_domain_leak.py")).read()
exec(full[full.index("def auroc"):full.index("out = {}")], ns)
def feats(src, comp):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32); N = U.shape[0]; U = U.reshape(N, 8, G * G, 2)
    if comp:
        U = U - np.median(U, axis=2, keepdims=True)
    Fc = []
    for w in range(N):
        for t in (4, 5, 6):
            u0, um = U[w, t], U[w, t - 1]
            m0 = np.linalg.norm(u0, axis=1); sel0 = m0 >= 0.5
            s2, ok = cell_of(P0 + u0); sel = np.where(sel0 & ok)[0]
            if sel.size == 0: continue
            x, xp = u0[sel], um[s2[sel]]
            nx, nxp = np.linalg.norm(x, axis=1), np.linalg.norm(xp, axis=1)
            r = np.linalg.norm(x - xp, axis=1) / (nx + nxp + 1e-6); cs = (x * xp).sum(1) / (nx * nxp + 1e-6)
            frac = np.full(sel.size, sel0.mean())               # 따로 움직이는 칸 몫
            Fc.append(np.stack([nx, nxp, r, cs, frac], 1))
    return np.concatenate(Fc)
out = {}
for comp in (0, 1):
    Xn = np.concatenate([feats(s, comp) for s in ("k710", "ht100m", "ssv2")]); Xs = feats("phys3d", comp)
    for nm, cols in (("motion_shape(|x|,|x'|,r,cos)", [0, 1, 2, 3]), ("+moving_fraction", [0, 1, 2, 3, 4])):
        X = np.concatenate([Xn[:, cols], Xs[:, cols]]); y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xs))])
        idx = np.random.default_rng(0).permutation(len(y)); tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
        f = ns["logit_fit"](X[tr], y[tr]); a = ns["auroc"](f(X[te]), y[te])
        out[f"comp{comp}|{nm}"] = dict(auroc=round(a, 3), n_nat=int(len(Xn)), n_syn=int(len(Xs)))
        print(comp, nm, out[f"comp{comp}|{nm}"])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/flow_domain_leak_comp.json"), "w"), indent=1)
