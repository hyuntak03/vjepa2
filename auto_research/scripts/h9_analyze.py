#!/usr/bin/env python3
"""H9 분석 — 개입 변형별 v11 matched pair 정확도 (timing × motion × 위반 × 방향, 슬롯 구간). clean 은 표준과 대조 (검증)."""
import collections, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h9")
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
OUT = ROOT / "auto_research/exp_results/h9"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); n = meta["n"]; L = np.load(I / "l1.npy"); V = meta["vars"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
ref = json.load(open(REF))["per_video_surprise"]; s = L[:, 0].mean(1); r = np.array([ref[v] for v in M["video_id"]])
print(f"검증 clean vs 표준: max {np.abs(s-r).max():.2e} median {np.median(np.abs(s-r)):.2e}")
pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
P = np.array([(d["pos"], d["imp"]) for d in pairs.values() if len(d) == 2]); ps, im = P[:, 0], P[:, 1]
out = {}
for sl, nm in ((slice(0, 8), "all"), (slice(0, 4), "t0-3"), (slice(4, 8), "t4-7")):
    S = L[:, :, sl].mean(2)
    ok = (S[im] > S[ps]).astype(float) + 0.5 * (S[im] == S[ps])
    groups = {"ALL": np.ones(len(P), bool)}
    for t in ("visible", "early", "mid", "late"):
        groups[t] = tim[ps] == t
        for mo in ("static", "moving_flat", "moving"):
            groups[f"{t}|{mo}"] = (tim[ps] == t) & (M["motion"][ps] == mo)
        for vt in ("vanish", "shape", "color"):
            groups[f"{t}|{vt}"] = (tim[ps] == t) & (M["violation_type"][ps] == vt)
        for d in ("A", "B"):
            groups[f"{t}|vanish|{d}"] = (tim[ps] == t) & (M["violation_type"][ps] == "vanish") & (M["pair_id"][ps] == d)
    for g, m in groups.items():
        out[f"{nm}|{g}"] = {v: round(100 * float(ok[m, vi].mean()), 1) for vi, v in enumerate(V)} | {"n": int(m.sum())}
json.dump(out, open(OUT / "h9_v11.json", "w"), indent=1)
show = ["all|ALL", "all|visible", "all|early", "all|mid", "all|late", "all|visible|moving", "all|visible|moving_flat", "all|late|static",
        "all|late|moving_flat", "all|late|moving", "all|late|vanish|A", "all|late|vanish|B", "t4-7|visible", "t4-7|late", "t4-7|visible|moving"]
for k in show:
    v = out[k]; print(f"{k:26s} " + "  ".join(f"{x} {v[x]:5.1f}" for x in V) + f"  n {v['n']}")
