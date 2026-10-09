#!/usr/bin/env python3
"""M6 adversarial verification (independent re-implementation, CPU only). Reads v3_m6 raw arrays."""
import csv, json, sys
from pathlib import Path
import numpy as np

I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m6")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
OUT = Path(__file__).resolve().parent / "m6_verify_out.json"
CELL, K, G = 18.0, 16, 16
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
M = np.load(I / "m6.npy").astype(np.float64)                                  # (n, arms, K, 8)
print("shape", M.shape, "nan", int(np.isnan(M).sum()), "arms", arms)
A = {a: i for i, a in enumerate(arms)}
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(rows[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(rows[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(rows[v]["in_frame_by_sample"]) for v in ids]) > 0.5
VIS = np.stack([f(rows[v]["visible_by_sample"]) for v in ids]) > 0.5
scen = np.array([rows[v]["scenario"] for v in ids])
traj = np.array([f"{rows[v]['scenario']}|{rows[v]['primary']}|{rows[v]['secondary']}" for v in ids])
# meta traj must agree with index
mt = np.array(["|".join(map(str, t)) for t in meta["traj"]])
print("traj agree", np.mean([a.split('|')[0] == b.split('|')[0] for a, b in zip(mt, traj)]))
ut = np.unique(traj); tix = [np.where(traj == t)[0] for t in ut]
print("n clips", n, "n traj", len(ut), "scen", dict(zip(*np.unique(scen, return_counts=True))))
rng = np.random.default_rng(12345)
B = 2000
boot_sel = [np.concatenate([tix[j] for j in rng.integers(0, len(ut), len(ut))]) for _ in range(B)]


def boot(v):
    v = np.asarray(v, float)
    m = np.nanmean(v)
    bs = np.array([np.nanmean(v[s]) for s in boot_sel])
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


def pos(f0):  # continuous cell position of tubelet starting at sample f0
    return np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL


L = pos(30); Tk = np.stack([pos(32 + 2 * t) for t in range(K)], 1)
u = Tk[:, -1] - L; un = np.linalg.norm(u, axis=-1); u = u / np.maximum(un, 1e-6)[:, None]
OK = np.stack([INF[:, 32 + 2 * t] & INF[:, 33 + 2 * t] for t in range(K)], 1) & (un >= 1)[:, None]
print("OK frac per slot", np.round(OK.mean(0), 3).tolist(), "un>=1", (un >= 1).mean())
tru_al = np.where(OK, ((np.floor(Tk) + 0.5 - L[:, None]) * u[:, None]).sum(-1), np.nan)

res = {}
for tpl, cx, cy, cmax, cmed in (("truth", 1, 0, 2, 3), ("appearance", 5, 4, 6, 7)):
    a = np.stack([M[..., cx], M[..., cy]], -1)                                  # (n, arms, K, 2) x,y
    al = np.where(OK[:, None], ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1), np.nan)
    con = M[..., cmax] - M[..., cmed]                                           # contrast (objectness)
    r = {}
    r["true_along"] = [boot(tru_al[:, t]) for t in range(K)]
    for arm in arms:
        r[arm] = {"along": [boot(al[:, A[arm], t]) for t in range(K)],
                  "d_base_iso": [boot(al[:, A[arm], t] - al[:, A["base_iso"], t]) for t in range(K)],
                  "d_base": [boot(al[:, A[arm], t] - al[:, A["base"], t]) for t in range(K)],
                  "d_hist_none": [boot(al[:, A[arm], t] - al[:, A["hist_none"], t]) for t in range(K)],
                  "contrast": [boot(np.where(OK[:, t], con[:, A[arm], t], np.nan)) for t in range(K)],
                  "obj_frac357": [boot(np.where(OK[:, t], (con[:, A[arm], t] >= 0.357).astype(float), np.nan)) for t in range(K)]}
    # monotone check hist-8 < hist-4 < base_iso < hist+4 per slot (means)
    mono = []
    for t in range(K):
        m = [np.nanmean(al[:, A[k], t]) for k in ("hist-8", "hist-4", "base_iso", "hist+4")]
        mono.append(bool(m[0] < m[1] < m[2] < m[3]))
    r["monotone_per_slot"] = mono
    # objectness-conditioned along diff: only slots where both arms pass common threshold 0.357
    cond = {}
    for arm in ("hist-8", "hist-4", "hist+4", "base", "hist_none", "bndM_only", "histZ+bndM"):
        both = (con[:, A[arm]] >= 0.357) & (con[:, A["base_iso"]] >= 0.357) & OK
        cond[arm] = [boot(np.where(both[:, t], al[:, A[arm], t] - al[:, A["base_iso"], t], np.nan)) for t in range(K)]
        cond[arm + "_n"] = [int(both[:, t].sum()) for t in range(K)]
    r["d_base_iso_both_obj357"] = cond
    res[tpl] = r
    res.setdefault("_al", {})[tpl] = al

# hit - null (truth template), independent null: all other trajectories of same scenario (mean), not one random draw
a = np.stack([M[..., 1], M[..., 0]], -1)
ct = np.floor(Tk)
hn = {}
same = [np.where((scen == scen[i]) & (traj != traj[i]))[0] for i in range(n)]
for arm in arms:
    hit = (np.abs(a[:, A[arm]] - ct).max(-1) <= 1).astype(float)              # (n, K)
    nul = np.stack([np.array([(np.abs(a[i, A[arm], t] - ct[same[i], t]).max(-1) <= 1).mean() for i in range(n)]) for t in range(K)], 1)
    hn[arm] = [boot(np.where(OK[:, t], hit[:, t] - nul[:, t], np.nan)) for t in range(K)]
res["hit_minus_fullnull"] = hn

# ---- off-screen audit of history windows
win = {"hist-8": (8, 20), "hist-4": (12, 24), "base_iso": (16, 28), "hist+4": (20, 32)}
res["inframe_hist_windows"] = {k: {"all_inframe_frac": float(INF[:, s:e].all(1).mean()),
                                   "mean_inframe_frac": float(INF[:, s:e].mean()),
                                   "all_visible_frac": float(VIS[:, s:e].all(1).mean())} for k, (s, e) in win.items()}
full_in = INF[:, 8:32].all(1)
res["n_all_inframe_8_32"] = int(full_in.sum())
al_t = res["_al"]["truth"]
res["d_base_iso_inframe_only"] = {arm: [boot(np.where(full_in, al_t[:, A[arm], t] - al_t[:, A["base_iso"], t], np.nan)) for t in (4, 6, 8, 10)]
                                  for arm in ("hist-8", "hist-4", "hist+4")}

# ---- dose: per-clip displacement along u caused by the shift (cells), history last tubelet position
def last_hist_pos(s):  # object position at last tubelet of history window starting at raw s (samples s+10, s+11)
    return pos(s + 10)
disp = {k: ((last_hist_pos(s) - last_hist_pos(16)) * u).sum(-1) for k, (s, e) in win.items()}  # along-u shift of history object
speed = np.linalg.norm(pos(30) - pos(16), axis=-1) / 14.0                    # cells / frame over context
res["speed_quantiles"] = np.percentile(speed, [10, 25, 50, 75, 90]).round(3).tolist()
terc = np.quantile(speed, [1 / 3, 2 / 3]); tb = np.digitize(speed, terc)
dose = {}
for arm in ("hist-8", "hist-4", "hist+4"):
    dd = {}
    for t in (4, 6, 8, 10):
        d = al_t[:, A[arm], t] - al_t[:, A["base_iso"], t]
        dd[f"t{t}_by_speed_tercile"] = [boot(np.where(tb == q, d, np.nan)) for q in range(3)]
        # slope of d on history displacement (cells), trajectory bootstrap
        x = disp[arm]; ok = ~np.isnan(d)
        def slope(sel):
            s = sel[ok[sel]]
            xx, yy = x[s], d[s]
            return np.polyfit(xx, yy, 1)[0] if len(s) > 10 and np.std(xx) > 0 else np.nan
        allsel = np.arange(n)
        bs = [slope(s) for s in boot_sel[:500]]
        dd[f"t{t}_slope_per_cell_disp"] = [round(float(slope(allsel)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
    dd["mean_hist_disp_cells"] = boot(disp[arm])
    dose[arm] = dd
res["dose"] = dose

# ---- direct retrieval test: does the arm's argmax land near the history object positions of THAT arm's window?
def near_hist(arm_i, s, t, tol=1):
    # object cells at each history tubelet of window [s, s+12)
    hc = np.stack([np.floor(pos(s + 2 * j)) for j in range(6)], 1)            # (n, 6, 2)
    aa = a[:, arm_i, t][:, None]                                              # (n, 1, 2)
    return (np.abs(aa - hc).max(-1) <= tol).any(1).astype(float)
ret = {}
for arm, (s, e) in win.items():
    ret[arm] = {}
    for t in (4, 6, 8, 10):
        own = near_hist(A[arm], s, t)
        base_on_own = near_hist(A["base_iso"], s, t)                           # does base_iso land in the same (shifted) cells?
        ret[arm][f"t{t}"] = {"arm_near_own_hist": boot(np.where(OK[:, t], own, np.nan)),
                             "base_iso_near_that_hist": boot(np.where(OK[:, t], base_on_own, np.nan)),
                             "diff": boot(np.where(OK[:, t], own - base_on_own, np.nan))}
res["near_history_cells"] = ret

# ---- overlap (hist+4 window shares raw 28-31 with boundary) and gap sizes
res["gap_frames_hist_to_boundary"] = {k: 28 - e for k, (s, e) in win.items()}

del res["_al"]
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)

# ---- compact print
for tpl in ("truth", "appearance"):
    r = res[tpl]
    print(f"\n=== {tpl}  true_along t4/6/8/10:", [r['true_along'][t][0] for t in (4, 6, 8, 10)])
    print(" monotone per slot:", "".join("M" if m else "." for m in r["monotone_per_slot"]))
    for arm in arms:
        print(f"  {arm:11s} along", [r[arm]['along'][t][0] for t in (0, 2, 4, 6, 8, 10, 12, 14)])
        print(f"  {'':11s} d_iso", [r[arm]['d_base_iso'][t] for t in (4, 6, 8, 10)])
        print(f"  {'':11s} d_base", [r[arm]['d_base'][t] for t in (4, 6, 8, 10)])
        print(f"  {'':11s} d_none", [r[arm]['d_hist_none'][t] for t in (4, 6, 8, 10)])
        print(f"  {'':11s} contrast", [r[arm]['contrast'][t][0] for t in (0, 2, 4, 6, 8, 10)], "obj357", [r[arm]['obj_frac357'][t][0] for t in (0, 2, 4, 6, 8, 10)])
    for arm, v in r["d_base_iso_both_obj357"].items():
        if arm.endswith("_n"):
            print(f"  cond n {arm}", [v[t] for t in (4, 6, 8, 10)])
        else:
            print(f"  cond(obj357 both) {arm}", [v[t] for t in (4, 6, 8, 10)])
print("\nhit-fullnull t4/6/8/10")
for arm in arms:
    print(f"  {arm:11s}", [res['hit_minus_fullnull'][arm][t] for t in (4, 6, 8, 10)])
print("\ninframe windows", json.dumps(res["inframe_hist_windows"]), "n all in-frame 8-32", res["n_all_inframe_8_32"])
print("d_base_iso in-frame-only", json.dumps(res["d_base_iso_inframe_only"]))
print("speed quantiles", res["speed_quantiles"])
print("dose", json.dumps(res["dose"], indent=0))
print("near history", json.dumps(res["near_history_cells"], indent=0))
