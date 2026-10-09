#!/usr/bin/env python3
"""H6f 분석 — v11 matched pair 정확도: p vs 복사 × 표준 표적 vs 인과 표적, timing × motion × 위반 (+ 방향 A/B).
검증: p_full 의 슬롯 평균 = 표준 per_video surprise (v11_full per_block.json)."""
import collections, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f")
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
OUT = ROOT / "auto_research/exp_results/h6"
meta = json.load(open(I / "meta.json")); n = meta["n"]; L = np.load(I / "l1.npy"); cols = meta["cols"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
ref = json.load(open(REF))["per_video_surprise"]; s = L[:, :, 0].mean(1); r = np.array([ref[v] for v in M["video_id"]])
print(f"검증 p_full vs 표준: max {np.abs(s-r).max():.2e} median {np.median(np.abs(s-r)):.2e}")
pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
P = np.array([(d["pos"], d["imp"]) for d in pairs.values() if len(d) == 2]); ps, im = P[:, 0], P[:, 1]
out = {}
for sl, nm in ((slice(0, 8), "all"), (slice(0, 4), "t0-3"), (slice(4, 8), "t4-7")):
    S = L[:, sl].mean(1)                                                    # (n, 4)
    ok = (S[im] > S[ps]).astype(float) + 0.5 * (S[im] == S[ps])
    for g in ["ALL"] + [f"{t}" for t in ("visible", "early", "mid", "late")] + \
             [f"{t}|{m}" for t in ("visible", "early", "mid", "late") for m in ("static", "moving_flat", "moving")] + \
             [f"{t}|{v}" for t in ("visible", "late") for v in ("vanish", "shape", "color")] + \
             [f"{t}|{v}|{d}" for t in ("visible", "late") for v in ("vanish",) for d in ("A", "B")]:
        parts = g.split("|"); m = np.ones(len(P), bool)
        if parts[0] != "ALL": m &= tim[ps] == parts[0]
        for q in parts[1:]:
            if q in ("static", "moving_flat", "moving"): m &= M["motion"][ps] == q
            elif q in ("vanish", "shape", "color"): m &= M["violation_type"][ps] == q
            elif q in ("A", "B"): m &= M["pair_id"][ps] == q
        out[f"{nm}|{g}"] = {c: round(100 * float(ok[m, ci].mean()), 1) for ci, c in enumerate(cols)} | {"n": int(m.sum())}
json.dump(out, open(OUT / "h6f_v11.json", "w"), indent=1)
for k, v in out.items():
    if k.startswith("all|") or k in ("t0-3|late", "t4-7|late", "t0-3|visible", "t4-7|visible"):
        print(f"{k:28s} " + "  ".join(f"{c} {v[c]:5.1f}" for c in cols) + f"  n {v['n']}")
