"""(IP2) per_video.csv 에서 다시: 쌍 정확도 (avg · max), 조건×카메라 scene bootstrap, release−copy 짝 차이 (각자 best C · 같은 C · 전 C 평균)."""
import json, numpy as np, pandas as pd
from pathlib import Path
from vcommon import dump
R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/intphys2")
TAGS = {"release_w48": "bench_intphys2_main_vith", "release_w32": "bench_intphys2_main_vith_w32", "release_w16": "bench_intphys2_main_vith_w16", "copy_w48": "bench_intphys2_main_vith_copy_w48", "copy_w32": "bench_intphys2_main_vith_copy_w32",
        "ariel43_w32": "bench_intphys2_main_ariel_bc_ep43_w32", "ariel43_w16": "bench_intphys2_main_ariel_bc_ep43_w16", "ariel30_w48": "bench_intphys2_main_ariel_bc_ep30_w48"}
rng = np.random.default_rng(7)
def pair_table(df, col="surprise_avg"):
    p = df.pivot_table(index=["scene_index", "pair_id", "condition", "camera", "context_length"], columns="is_impossible", values=col, aggfunc="first").reset_index()
    p = p.dropna(subset=[True, False]); p["acc"] = np.where(p[True] > p[False], 1.0, np.where(p[True] == p[False], 0.5, 0.0)); return p
def boot(a, sc, B=2000):
    uc = np.unique(sc); ix = {c: np.where(sc == c)[0] for c in uc}
    bs = [a[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(B)]
    return [int(len(a)), round(float(a.mean() * 100), 1), round(float(np.percentile(bs, 2.5) * 100), 1), round(float(np.percentile(bs, 97.5) * 100), 1)]
D = {}; res = {}
for k, t in TAGS.items():
    f = R / t / "per_video.csv"
    if not f.exists(): res[k] = "없음"; continue
    df = pd.read_csv(f); s = json.load(open(R / t / "summary.json")); C = int(s["best_context_length"]); D[k] = (df, C)
    P = pair_table(df); Pm = pair_table(df, "surprise_max"); Pc = P[P.context_length == C]; Pmc = Pm[Pm.context_length == C]
    rec = dict(best_C=C, Cs=sorted(df.context_length.unique().tolist()), n_pairs=int(len(Pc)), overall=boot(Pc.acc.values, Pc.scene_index.values), overall_allC_mean=round(float(P.acc.mean() * 100), 1),
               summary_json_keys=list(s.keys())[:12], summary_acc=s.get("accuracy", s.get("pairwise_accuracy", None)))
    for cond in ("permanence", "continuity", "immutability", "solidity"):
        for cam in ("Fixed", "Moving", "all"):
            m = (Pc.condition == cond) & ((Pc.camera == cam) if cam != "all" else True); mm = (Pmc.condition == cond) & ((Pmc.camera == cam) if cam != "all" else True)
            rec[f"{cond}|{cam}"] = dict(avg=boot(Pc.acc.values[m], Pc.scene_index.values[m]), max=boot(Pmc.acc.values[mm], Pmc.scene_index.values[mm], B=300),
                                        allC_avg=round(float(P.acc.values[(P.condition == cond) & ((P.camera == cam) if cam != "all" else True)].mean() * 100), 1))
    res[k] = rec
# 짝 차이
def diff(ka, kb, Ca=None, Cb=None):
    (da, ca), (db, cb) = D[ka], D[kb]; Ca = Ca or ca; Cb = Cb or cb
    A = pair_table(da); A = A[A.context_length == Ca]; B = pair_table(db); B = B[B.context_length == Cb]
    M = A.merge(B, on=["scene_index", "pair_id", "condition", "camera"], suffixes=("_a", "_b")); M["d"] = M.acc_a - M.acc_b; out = {}
    for cond in ("permanence", "continuity", "immutability", "solidity", "all"):
        for cam in ("Fixed", "Moving", "all"):
            m = np.ones(len(M), bool) & ((M.condition == cond).values if cond != "all" else True) & ((M.camera == cam).values if cam != "all" else True)
            if m.sum() == 0: continue
            out[f"{cond}|{cam}"] = boot(M.d.values[m], M.scene_index.values[m], B=1000)
    return dict(Ca=Ca, Cb=Cb, n=int(len(M)), diff=out)
res["diff"] = {}
for ka, kb in (("release_w48", "copy_w48"), ("release_w32", "copy_w32"), ("ariel30_w48", "copy_w48"), ("ariel43_w32", "copy_w32")):
    if ka in D and kb in D:
        res["diff"][f"{ka}-{kb}_bestC"] = diff(ka, kb)
        res["diff"][f"{ka}-{kb}_sameC{D[ka][1]}"] = diff(ka, kb, D[ka][1], D[ka][1]) if D[ka][1] in D[kb][0].context_length.unique() else "copy 에 그 C 없음"
dump("ip2", res)
for k, r in res.items():
    if k == "diff":
        for kk, v in r.items(): print(kk, v)
    elif isinstance(r, dict): print(k, "C", r["best_C"], "Cs", r["Cs"], "n", r["n_pairs"], "overall", r["overall"], "allC", r["overall_allC_mean"], "summary_acc", r["summary_acc"]); [print("   ", c, r[c]) for c in r if "|" in c]
    else: print(k, r)
