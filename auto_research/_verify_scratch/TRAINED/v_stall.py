#!/usr/bin/env python3
"""Do trained predictors also stall? Along-path displacement slope of the argmax (appearance and truth templates)
in slot bands, s1_C16 (16 future tubelets; trained horizon of pv1/v11ft/Ariel(C=8) = 8), trajectory-cluster CI."""
import csv, json
from pathlib import Path
import numpy as np

C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
TAGS = ["", "_pv1", "_ariel", "_v11ft"]; CELL = 18.0; B = 1000
meta = json.load(open(C / "v3_p2" / "meta.json")); ids = meta["video_ids"]; n = len(ids); arms = meta["arms"]
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fl = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids])
scen = np.array(meta["scenario"]); traj = np.array(["|".join(map(str, t)) for t in meta["traj"]])
ut = np.unique(traj); tix = {u: np.where(traj == u)[0] for u in ut}
rng = np.random.default_rng(3); BI = [np.concatenate([tix[u] for u in rng.choice(ut, len(ut))]) for _ in range(B)]
out = {}
for arm_name, bands in (("s1_C16", [(0, 3), (3, 8), (8, 12), (12, 16)]), ("s2_C16", [(0, 3), (3, 8)])):
    ai = [k for k, A in enumerate(arms) if A["name"] == arm_name][0]; A = arms[ai]
    fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
    L = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
    CT = np.stack([np.stack([(X[:, fr[2 * (nc + t)]] + X[:, fr[2 * (nc + t) + 1]]) / 2, (Y[:, fr[2 * (nc + t)]] + Y[:, fr[2 * (nc + t) + 1]]) / 2], -1) / CELL for t in range(nf)], 1)
    u = CT[:, -1] - L; u = u / np.maximum(np.linalg.norm(u, axis=-1), 1e-6)[:, None]
    true_al = ((CT - L[:, None]) * u[:, None]).sum(-1)
    keep = ~(scen == "arc")                                    # arc curves; use straight-ish laws for along-slope
    res = {}
    def bslope(al, lo, hi):
        tt = np.arange(lo, hi) - np.arange(lo, hi).mean()
        s = (al[:, lo:hi] * tt).sum(1) / (tt ** 2).sum()
        s = np.where(keep, s, np.nan)
        bs = [np.nanmean(s[b]) for b in BI]
        return [round(float(np.nanmean(s)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    res["truth"] = {f"t{lo}-{hi-1}": bslope(true_al, lo, hi) for lo, hi in bands}
    for t in TAGS:
        for tpl, nm in (("app", "loc_app"), ("tru", "loc_tru")):
            Lc = np.load(C / f"v3_p2{t}" / f"{nm}.npy")[:, ai, :nf][..., [1, 0]]
            al = ((Lc + 0.5 - L[:, None]) * u[:, None]).sum(-1)
            res[f"{t or 'release'}|{tpl}"] = {f"t{lo}-{hi-1}": bslope(al, lo, hi) for lo, hi in bands}
            res[f"{t or 'release'}|{tpl}|along_mean_noarc"] = [round(float(np.nanmean(al[keep, s])), 2) for s in range(nf)]
    res["truth_along_mean_noarc"] = [round(float(np.nanmean(true_al[keep, s])), 2) for s in range(nf)]
    out[arm_name] = res
json.dump(out, open(Path(__file__).resolve().parent / "v_stall_out.json", "w"), indent=1)
for a, r in out.items():
    print("==", a)
    for k, v in r.items(): print(k, v)
