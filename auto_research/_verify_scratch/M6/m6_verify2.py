#!/usr/bin/env python3
"""M6 verify part 2: paired hit-null / objectness diffs, splice-cost contrasts, all-slot base-hist_none."""
import csv, json
from pathlib import Path
import numpy as np
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m6")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CELL, K = 18.0, 16
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
M = np.load(I / "m6.npy").astype(np.float64); A = {a: i for i, a in enumerate(arms)}
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(rows[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(rows[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(rows[v]["in_frame_by_sample"]) for v in ids]) > 0.5
scen = np.array([rows[v]["scenario"] for v in ids]); traj = np.array([f"{rows[v]['scenario']}|{rows[v]['primary']}|{rows[v]['secondary']}" for v in ids])
ut = np.unique(traj); tix = [np.where(traj == t)[0] for t in ut]
rng = np.random.default_rng(7); sel = [np.concatenate([tix[j] for j in rng.integers(0, len(ut), len(ut))]) for _ in range(2000)]
def boot(v):
    v = np.asarray(v, float); bs = np.array([np.nanmean(v[s]) for s in sel])
    return [round(float(np.nanmean(v)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
pos = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
L = pos(30); Tk = np.stack([pos(32 + 2 * t) for t in range(K)], 1)
u = Tk[:, -1] - L; u = u / np.linalg.norm(u, axis=-1)[:, None]
a = np.stack([M[..., 1], M[..., 0]], -1); al = ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1)
con = M[..., 2] - M[..., 3]; ct = np.floor(Tk)
same = [np.where((scen == scen[i]) & (traj != traj[i]))[0] for i in range(n)]
hit = {k: (np.abs(a[:, A[k]] - ct).max(-1) <= 1).astype(float) for k in arms}
nul = {k: np.stack([np.array([(np.abs(a[i, A[k], t] - ct[same[i], t]).max(-1) <= 1).mean() for i in range(n)]) for t in range(K)], 1) for k in arms}
hn = {k: hit[k] - nul[k] for k in arms}
out = {}
pairs = [("base", "hist_none"), ("base_iso", "hist_none"), ("hist+4", "hist_none"), ("hist-8", "base_iso"), ("hist+4", "base_iso"),
         ("histZ+bndM", "bndM_only"), ("bndM_only", "base"), ("histZ+bndM", "base"), ("base", "base_iso")]
for p, q in pairs:
    out[f"{p}-{q}"] = {"along": [boot(al[:, A[p], t] - al[:, A[q], t]) for t in range(K)],
                       "hitnull": [boot(hn[p][:, t] - hn[q][:, t]) for t in range(K)],
                       "obj357": [boot((con[:, A[p], t] >= .357).astype(float) - (con[:, A[q], t] >= .357).astype(float)) for t in range(K)],
                       "contrast": [boot(con[:, A[p], t] - con[:, A[q], t]) for t in range(K)]}
# lag = true along - arm along; natural lag vs no-history lag
tru = ((np.floor(Tk) + 0.5 - L[:, None]) * u[:, None]).sum(-1)
out["lag_ratio_mean_t4_15"] = {k: round(float(np.mean(al[:, A[k], 4:]) / np.mean(tru[:, 4:])), 3) for k in arms}
# window content: object pixel displacement between last hist tubelet and first boundary tubelet (seam jump, cells, along u)
for k, s in (("base_iso", 16), ("hist-8", 8), ("hist-4", 12), ("hist+4", 20)):
    out[f"seam_jump_{k}"] = boot(((pos(28) - pos(s + 10)) * u).sum(-1))
json.dump(out, open(Path(__file__).resolve().parent / "m6_verify2_out.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "along" in v:
        print(f"\n{k}")
        for m in ("along", "hitnull", "obj357", "contrast"):
            print(f"  {m:9s}", " ".join(f"t{t}:{v[m][t][0]:+.3f}[{v[m][t][1]:+.3f},{v[m][t][2]:+.3f}]" for t in (2, 4, 6, 8, 10, 12)))
    else:
        print(k, v)
