#!/usr/bin/env python3
"""Independent P3 re-analysis across predictors: j3-7 along-slope (truth template and appearance template),
new-cell landing j4-7, hit-null (exhaustive null) at j7, paired vs release; trajectory-cluster bootstrap."""
import csv, json
from pathlib import Path
import numpy as np

C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
TAGS = ["", "_pv1", "_ariel", "_v11ft"]; J = 8; CELL = 18.0; B = 2000
metas = {t: json.load(open(C / f"v3_p3{t}" / "meta.json")) for t in TAGS}
ids = metas[""]["video_ids"]
for t in TAGS: assert metas[t]["video_ids"] == ids and list(metas[t]["arms"]) == list(metas[""]["arms"])
arms = list(metas[""]["arms"]); n = len(ids)
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fl = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([fl(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
traj = np.array(["|".join(map(str, t)) for t in metas[""]["traj"]]); scen = np.array(metas[""]["scenario"])
ut = np.unique(traj); tix = {u: np.where(traj == u)[0] for u in ut}
rng = np.random.default_rng(99); BI = [np.concatenate([tix[u] for u in rng.choice(ut, len(ut))]) for _ in range(B)]
xy = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
cl = lambda v: np.clip(np.floor(v), 0, 15)
L = xy(30); T = np.stack([cl(xy(32 + 2 * j)) for j in range(J)], 1)
OK = np.stack([INF[:, 30] & INF[:, 31] & INF[:, 32 + 2 * j] & INF[:, 33 + 2 * j] for j in range(J)], 1)
u = xy(32 + 2 * (J - 1)) - L; un = np.linalg.norm(u, axis=-1); u = u / np.maximum(un, 1e-6)[:, None]
okc = OK.all(1) & (un >= 1)
js = np.arange(3, J) - np.arange(3, J).mean()


def ci(v):
    v = np.asarray(v, float); bs = np.array([np.nanmean(v[b]) for b in BI])
    return [round(float(np.nanmean(v)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


def feats(a):
    al = ((a + 0.5 - L[:, None]) * u[:, None]).sum(-1)
    sl = np.where(okc, (np.nan_to_num(al)[:, 3:] * js).sum(1) / (js ** 2).sum(), np.nan)
    ln = []
    for j in range(4, J):
        mv = OK[:, j] & (np.abs(T[:, j] - T[:, j - 1]).max(-1) > 0)
        ln.append(np.where(mv, (a[:, j] == T[:, j]).all(-1), np.nan))
    ln = np.nanmean(np.stack(ln, 1).astype(float), 1)
    j = 7; h = np.where(OK[:, j], np.abs(a[:, j] - T[:, j]).max(-1) <= 1, np.nan).astype(float)
    nx = np.full(n, np.nan)
    for i in range(n):
        if not OK[i, j]: continue
        p = np.where((scen == scen[i]) & (traj != traj[i]) & OK[:, j])[0]
        nx[i] = (np.abs(a[i, j][None] - T[p, j]).max(-1) <= 1).mean()
    return dict(sl=sl, ln=ln, hn7=h - nx, along=al)


true_sl = np.where(okc, (((T + 0.5 - L[:, None]) * u[:, None]).sum(-1)[:, 3:] * js).sum(1) / (js ** 2).sum(), np.nan)
out = {"n": n, "n_traj": len(ut), "true_slope": ci(true_sl)}
F = {}
for t in TAGS:
    M = np.load(C / f"v3_p3{t}" / "p3.npy")
    out.setdefault("nan", {})[t or "release"] = int(np.isnan(M[..., 0]).sum())
    for ai, arm in enumerate(arms):
        if arm not in ("ar_z", "ar_pA", "ar_hcA"): continue
        for tpl, cols in (("truth", [1, 0]), ("app", [5, 4])):
            F[(t, arm, tpl)] = feats(M[:, :, ai, cols])
    # j0 identity check: before any feedback all arms must be identical within a predictor
    out.setdefault("j0_arms_identical", {})[t or "release"] = bool(np.allclose(M[:, 0, :, 8], M[:, 0, :1, 8], atol=1e-4))
res = {}
for (t, arm, tpl), f in F.items():
    r = dict(slope=ci(f["sl"]), land_new_j4_7=ci(f["ln"]), hit_null_j7=ci(f["hn7"]),
             along_mean=[round(float(np.nanmean(f["along"][:, j])), 2) for j in range(J)])
    if t:
        g = F[("", arm, tpl)]
        r["paired_slope_minus_release"] = ci(f["sl"] - g["sl"])
    res[f"{t or 'release'}|{arm}|{tpl}"] = r
out["res"] = res
out["true_along_mean"] = [round(float(np.nanmean(((T[:, j] + 0.5 - L) * u).sum(-1))), 2) for j in range(J)]
json.dump(out, open(Path(__file__).resolve().parent / "v_p3_out.json", "w"), indent=1)
print(json.dumps(out, indent=0)[:6000])
