#!/usr/bin/env python3
"""M6 verify part 3: does the hist-8 / hist-4 pull depend on whether the object is actually present in the shifted history window?"""
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
ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
rng = np.random.default_rng(3)
def boot(v, idx):
    v = np.asarray(v, float); tt = np.unique(traj[idx]); ti = [idx[traj[idx] == t] for t in tt]
    bs = [np.nanmean(v[np.concatenate([ti[j] for j in rng.integers(0, len(tt), len(tt))])]) for _ in range(1000)]
    return [round(float(np.nanmean(v[idx])), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3), len(idx), len(tt)]
pos = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
L = pos(30); Tk = np.stack([pos(32 + 2 * t) for t in range(K)], 1)
u = Tk[:, -1] - L; u = u / np.linalg.norm(u, axis=-1)[:, None]
a = np.stack([M[..., 1], M[..., 0]], -1); al = ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1)
aa = np.stack([M[..., 5], M[..., 4]], -1); ala = ((aa + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1)
out = {}
for arm, s in (("hist-8", 8), ("hist-4", 12)):
    fr = INF[:, s:s + 12].mean(1)
    groups = {"all_in": np.where(fr == 1)[0], "partial(0.5-1)": np.where((fr >= 0.5) & (fr < 1))[0], "mostly_out(<0.5)": np.where(fr < 0.5)[0]}
    out[arm] = {}
    for g, idx in groups.items():
        if len(idx) == 0: continue
        out[arm][g] = {tpl: {f"t{t}": boot(A_[:, A[arm], t] - A_[:, A["base_iso"], t], idx) for t in (4, 6, 8, 10)} for tpl, A_ in (("truth", al), ("app", ala))}
        out[arm][g]["scen"] = dict(zip(*[x.tolist() for x in np.unique(scen[idx], return_counts=True)]))
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent / "m6_verify3_out.json", "w"), indent=1)
