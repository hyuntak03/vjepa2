#!/usr/bin/env python3
"""SSV2T part 3: rho split into 'arrive' (<dp_i, h~_{9+i}>) vs 'depart' (-<dp_i, h~_{8+i}>), and
clip-centred correlation matrix C[i,j] = corr(p~_i, h~_j) over all 16 h slots (index signature removed by centring)."""
import json
import numpy as np
I = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e3"
n = json.load(open(f"{I}/meta.json"))["n"]; PR = np.load(f"{I}/proj.npy", mmap_mode="r")
R = {}; r3 = lambda x: np.round(np.asarray(x, float), 3).tolist()
rng = np.random.default_rng(3)
for di, dn in enumerate(("forward", "reversed")):
    A = np.asarray(PR[:, di], np.float32); A = A - A.mean(0, keepdims=True)          # centre every (slot, token, dim) across clips
    H, P, HC = A[:, 0:16], A[:, 16:24], A[:, 24]
    dP = np.diff(P, axis=1); dH = np.diff(H[:, 8:16], axis=1)
    f = lambda a, b: float((a.astype(np.float64) * b).sum())
    arr, dep, tot = [], [], []
    for i in range(7):
        den = np.sqrt(f(dP[:, i], dP[:, i]) * f(dH[:, i], dH[:, i]))
        arr.append(f(dP[:, i], H[:, 9 + i]) / den); dep.append(-f(dP[:, i], H[:, 8 + i]) / den); tot.append(f(dP[:, i], dH[:, i]) / den)
    # clip bootstrap for arrive at each transition
    na = np.stack([(dP[:, i].astype(np.float64) * H[:, 9 + i]).sum((1, 2)) for i in range(7)], 1)
    nd = np.stack([-(dP[:, i].astype(np.float64) * H[:, 8 + i]).sum((1, 2)) for i in range(7)], 1)
    pp = np.stack([(dP[:, i].astype(np.float64) ** 2).sum((1, 2)) for i in range(7)], 1)
    hh = np.stack([(dH[:, i].astype(np.float64) ** 2).sum((1, 2)) for i in range(7)], 1)
    bsA, bsD = [], []
    for _ in range(1000):
        s = rng.integers(0, n, n); d = np.sqrt(pp[s].sum(0) * hh[s].sum(0)); bsA.append(na[s].sum(0) / d); bsD.append(nd[s].sum(0) / d)
    rec = dict(rho_total=r3(tot), rho_arrive=r3(arr), rho_depart=r3(dep),
               arrive_ci=[r3(np.percentile(bsA, 2.5, 0)), r3(np.percentile(bsA, 97.5, 0))], depart_ci=[r3(np.percentile(bsD, 2.5, 0)), r3(np.percentile(bsD, 97.5, 0))])
    # correlation matrix p~_i vs h~_j (levels), and vs hc7
    nrm = lambda X: np.sqrt((X.astype(np.float64) ** 2).sum())
    C = np.array([[f(P[:, i], H[:, j]) / (nrm(P[:, i]) * nrm(H[:, j])) for j in range(16)] for i in range(8)])
    rec["corr_p_vs_h_levels"] = r3(C); rec["argmax_j"] = [int(C[i].argmax()) for i in range(8)]
    rec["corr_p_vs_hc7"] = r3([f(P[:, i], HC) / (nrm(P[:, i]) * nrm(HC)) for i in range(8)])
    Ch = np.array([[f(H[:, 8 + i], H[:, j]) / (nrm(H[:, 8 + i]) * nrm(H[:, j])) for j in range(16)] for i in range(8)])
    rec["corr_hfuture_vs_h_levels"] = r3(Ch)
    R[dn] = rec
    print(dn, json.dumps({k: rec[k] for k in ("rho_total", "rho_arrive", "rho_depart", "arrive_ci", "argmax_j", "corr_p_vs_hc7")}), flush=True)
    print(" C diag (p_i vs h_8+i):", r3([C[i, 8 + i] for i in range(8)]), " C p_i vs h_7:", r3(C[:, 7]), " C p_i vs h_8:", r3(C[:, 8]), flush=True)
    del A, H, P, HC, dP, dH
json.dump(R, open("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/SSV2T/s3_arrive.json", "w"), indent=1)
print("DONE")
