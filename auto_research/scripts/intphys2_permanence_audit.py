#!/usr/bin/env python3
"""IntPhys 2 Main permanence 감사 (2026-09-26): predictor (release · Ariel) vs 복사 기준선 (IP2_COPY=1) 을 조건 × 카메라로, scene bootstrap CI.
쌍 = 같은 scene_index · pair_id 의 (Possible, Impossible). 정답 = surprise(imp) > surprise(pos), tie = 0.5. 창당 C 는 run 의 best_context_length (summary.json) 와
"열별 최고 C" (논문 Table 2 방식) 둘 다 보고. 짝 차이 = 같은 쌍의 (pred − copy) 정답 지표.
  python intphys2_permanence_audit.py   → auto_research/exp_results/verify/intphys2_permanence_audit.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/intphys2")
OUT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/exp_results/verify"); OUT.mkdir(parents=True, exist_ok=True)
RUNS = {"release": {48: "bench_intphys2_main_vith", 32: "bench_intphys2_main_vith_w32", 16: "bench_intphys2_main_vith_w16"},
        "copy": {48: "bench_intphys2_main_vith_copy_w48", 32: "bench_intphys2_main_vith_copy_w32"},
        "ariel43": {32: "bench_intphys2_main_ariel_bc_ep43_w32", 16: "bench_intphys2_main_ariel_bc_ep43_w16"},
        "ariel30": {48: "bench_intphys2_main_ariel_bc_ep30_w48"}}
rng = np.random.default_rng(0)
CONDS = ["permanence", "continuity", "immutability", "solidity"]; CAMS = ["Fixed", "Moving", "all"]


def pairs(df):
    rows = []
    for (sc, pid), d in df.groupby(["scene_index", "pair_id"]):
        if len(d) != 2 or d.is_impossible.sum() != 1: continue
        si = d[d.is_impossible].surprise_avg.values[0]; sp = d[~d.is_impossible].surprise_avg.values[0]
        rows.append((int(sc), int(pid), 1.0 if si > sp else (0.5 if si == sp else 0.0), d.condition.iloc[0], d.camera.iloc[0], d.difficulty.iloc[0]))
    return pd.DataFrame(rows, columns=["scene", "pair", "acc", "condition", "camera", "difficulty"])


def boot(a, sc, B=2000):
    if len(a) == 0: return [np.nan] * 3
    uc = np.unique(sc); ix = {c: np.where(sc == c)[0] for c in uc}
    bs = [a[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(B)]
    return [round(float(a.mean() * 100), 1), round(float(np.percentile(bs, 2.5) * 100), 1), round(float(np.percentile(bs, 97.5) * 100), 1)]


def load(tag):
    d = R / tag
    if not (d / "per_video.csv").exists(): return None, None
    return pd.read_csv(d / "per_video.csv"), int(json.load(open(d / "summary.json"))["best_context_length"])


res = {}; P = {}
for model, ws in RUNS.items():
    for w, tag in ws.items():
        df, C = load(tag)
        if df is None: res[f"{model}_w{w}"] = "없음"; continue
        rec = dict(tag=tag, best_C=C, n_pairs=None, at_best_C={}, column_best={})
        Pc = pairs(df[df.context_length == C]); P[(model, w)] = {C: Pc}; rec["n_pairs"] = int(len(Pc))
        allC = {int(c): pairs(df[df.context_length == c]) for c in sorted(df.context_length.unique())}; P[(model, w)] = allC
        for cond in CONDS:
            for cam in CAMS:
                m = (Pc.condition == cond) & ((Pc.camera == cam) if cam != "all" else True)
                rec["at_best_C"][f"{cond}|{cam}"] = [int(m.sum())] + boot(Pc.acc.values[m], Pc.scene.values[m])
                best = max(((c, float(p.acc.values[(p.condition == cond) & ((p.camera == cam) if cam != "all" else True)].mean())) for c, p in allC.items()), key=lambda t: t[1])
                rec["column_best"][f"{cond}|{cam}"] = [best[0], round(best[1] * 100, 1)]
        rec["at_best_C"]["overall"] = [int(len(Pc))] + boot(Pc.acc.values, Pc.scene.values)
        res[f"{model}_w{w}"] = rec
# 짝 차이 pred − copy (같은 창 · 각자 best C)
for w in (48, 32):
    if ("copy", w) not in P: continue
    Cc = res[f"copy_w{w}"]["best_C"]; pc = P[("copy", w)][Cc]
    for model in ("release", "ariel43", "ariel30"):
        if (model, w) not in P: continue
        Cm = res[f"{model}_w{w}"]["best_C"]; pm = P[(model, w)][Cm]
        M = pm.merge(pc, on=["scene", "pair", "condition", "camera"], suffixes=("_m", "_c"))
        d = {}
        for cond in CONDS + ["all"]:
            for cam in CAMS:
                mm = ((M.condition == cond) if cond != "all" else pd.Series(True, index=M.index)) & ((M.camera == cam) if cam != "all" else pd.Series(True, index=M.index))
                d[f"{cond}|{cam}"] = [int(mm.sum())] + boot((M.acc_m - M.acc_c).values[mm], M.scene.values[mm])
        res[f"diff_{model}_minus_copy_w{w}"] = d
json.dump(res, open(OUT / "intphys2_permanence_audit.json", "w"), indent=1, default=float)
for k, v in res.items():
    if isinstance(v, str): print(k, v); continue
    if k.startswith("diff"):
        print(f"== {k}: " + "  ".join(f"{kk} n{vv[0]} {vv[1]:+.1f}[{vv[2]:+.1f},{vv[3]:+.1f}]" for kk, vv in v.items() if kk.endswith("|Fixed") or kk.endswith("|all")))
    else:
        print(f"== {k} (best C {v['best_C']}, n {v['n_pairs']}) overall {v['at_best_C']['overall'][1:]}")
        for cond in CONDS: print(f"   {cond:12s} Fixed {v['at_best_C'][cond+'|Fixed']}  Moving {v['at_best_C'][cond+'|Moving']}  all {v['at_best_C'][cond+'|all']}  | 열별 최고 Fixed {v['column_best'][cond+'|Fixed']} all {v['column_best'][cond+'|all']}")
