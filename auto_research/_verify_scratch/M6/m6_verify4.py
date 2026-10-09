#!/usr/bin/env python3
"""M6 verify part 4: d(hist-8 - base_iso) ~ displacement(cells) + out-of-frame fraction + scenario FE, trajectory bootstrap."""
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
rng = np.random.default_rng(5); sel = [np.concatenate([tix[j] for j in rng.integers(0, len(ut), len(ut))]) for _ in range(1000)]
pos = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
L = pos(30); Tk = np.stack([pos(32 + 2 * t) for t in range(K)], 1)
u = Tk[:, -1] - L; u = u / np.linalg.norm(u, axis=-1)[:, None]
a = np.stack([M[..., 1], M[..., 0]], -1); al = ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1)
S = sorted(set(scen)); FE = np.stack([(scen == s).astype(float) for s in S], 1)
out = {}
for arm, s in (("hist-8", 8), ("hist-4", 12)):
    # mean along-u displacement of history object over the window, only in-frame samples, relative to base_iso window (raw 16-27)
    xx = np.stack([pos(s + 2 * j) for j in range(6)], 1); x0 = np.stack([pos(16 + 2 * j) for j in range(6)], 1)
    disp = ((xx - x0) * u[:, None]).sum(-1).mean(1)                         # geometric shift (cells), uses sim positions even if off-frame
    outf = 1 - INF[:, s:s + 12].mean(1)
    Xd = np.column_stack([disp, outf, FE])
    out[arm] = {"corr_disp_outf": round(float(np.corrcoef(disp, outf)[0, 1]), 3)}
    for t in (4, 6, 8, 10):
        y = al[:, A[arm], t] - al[:, A["base_iso"], t]
        fit = lambda idx: np.linalg.lstsq(Xd[idx], y[idx], rcond=None)[0][:2]
        b = fit(np.arange(n)); bs = np.array([fit(ss) for ss in sel])
        out[arm][f"t{t}"] = {"b_disp_per_cell": [round(float(b[0]), 3)] + np.percentile(bs[:, 0], [2.5, 97.5]).round(3).tolist(),
                             "b_outframe_frac": [round(float(b[1]), 3)] + np.percentile(bs[:, 1], [2.5, 97.5]).round(3).tolist()}
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent / "m6_verify4_out.json", "w"), indent=1)
