#!/usr/bin/env python3
"""Independent re-analysis of v3_p3b (does not import p3_analyze). CPU only."""
import csv, json, sys
from pathlib import Path
import numpy as np

I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3b")
H5 = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h5")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
OUTJ = Path(__file__).resolve().parent / "verify_p3b_out.json"
CELL, J, THR = 18.0, 8, 0.357
meta = json.load(open(I / "meta.json")); arms = list(meta["arms"]); ids = meta["video_ids"]; n = len(ids)
M = np.load(I / "p3b.npy").astype(np.float64)
print("shape", M.shape, "arms", arms, "NaN count", int(np.isnan(M).sum()))
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fa = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fa(rows[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fa(rows[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([fa(rows[v]["in_frame_by_sample"]) for v in ids]) > 0.5
SF = np.stack([fa(rows[v]["sample_frames"]) for v in ids])
print("sample_frames stride-1 0..63 for all:", bool((SF[:, :64] == np.arange(64)).all()), "n samples", SF.shape[1])
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); scen = np.array(meta["scenario"])
ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
print("n", n, "n_traj", len(ut), "scen counts", {s: int((scen == s).sum()) for s in np.unique(scen)})

def cxy(f0):
    return np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
cl = lambda v: np.clip(np.floor(v), 0, 15)
L = cxy(30); Lc = cl(L)
T = np.stack([cl(cxy(32 + 2 * j)) for j in range(J)], 1)                  # (n,J,2) x,y
OK = np.stack([INF[:, 30] & INF[:, 31] & INF[:, 32 + 2 * j] & INF[:, 33 + 2 * j] for j in range(J)], 1)
u = cxy(46) - L; un = np.linalg.norm(u, axis=-1); u = u / np.maximum(un, 1e-6)[:, None]
okc = OK.all(1) & (un >= 1)
print("okc clips", int(okc.sum()), "of", n)

rng = np.random.default_rng(777); B = 2000
BI = [np.concatenate([tix[t] for t in rng.choice(ut, len(ut))]) for _ in range(B)]
def ci(v):
    v = np.asarray(v, float); bs = np.array([np.nanmean(v[b]) for b in BI])
    return [round(float(np.nanmean(v)), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]

def along(a):  # a (n,J,2) xy cells
    return ((a + 0.5 - L[:, None]) * u[:, None]).sum(-1)
def slope(al, js=range(3, J)):
    js = np.array(list(js)); w = js - js.mean()
    s = (al[:, js] * w).sum(1) / (w ** 2).sum()
    return np.where(okc, s, np.nan)

A = {t: {arm: M[:, :, k, [1, 0]] if t == "tru" else M[:, :, k, [5, 4]] for k, arm in enumerate(arms)} for t in ("tru", "app")}
out = {"nan": int(np.isnan(M).sum()), "n": n, "n_traj": int(len(ut)), "okc": int(okc.sum())}

# 0) degeneracy checks at j=1 (tf_first == ar_z; tf_last == ar_pA); j=0 all equal
k = {a: i for i, a in enumerate(arms)}
out["j1_tf_first_eq_ar_z_maxabs"] = float(np.abs(M[:, 1, k["tf_first"]] - M[:, 1, k["ar_z"]]).max())
out["j1_tf_last_eq_ar_pA_maxabs"] = float(np.abs(M[:, 1, k["tf_last"]] - M[:, 1, k["ar_pA"]]).max())
out["j0_all_arms_eq_maxabs"] = float(max(np.abs(M[:, 0, i] - M[:, 0, 0]).max() for i in range(len(arms))))
out["j2_tf_first_eq_ar_z_argmax_agree"] = float((M[:, 2, k["tf_first"], :2] == M[:, 2, k["ar_z"], :2]).all(-1).mean())

# 1) recompute slopes and paired diffs, both templates
true_sl = slope(along(T))
out["true_slope"] = ci(true_sl)
SL = {t: {arm: slope(along(A[t][arm])) for arm in arms} for t in A}
for t in A:
    out[f"slope_{t}"] = {arm: ci(SL[t][arm]) for arm in arms}
    out[f"paired_vs_ar_z_{t}"] = {arm: ci(SL[t][arm] - SL[t]["ar_z"]) for arm in arms if arm != "ar_z"}
    out[f"tf_first_minus_tf_last_{t}"] = ci(SL[t]["tf_first"] - SL[t]["tf_last"])
    out[f"tf_first_minus_ar_pA_{t}"] = ci(SL[t]["tf_first"] - SL[t]["ar_pA"])
    out[f"tf_last_minus_ar_pA_{t}"] = ci(SL[t]["tf_last"] - SL[t]["ar_pA"])
    # recovery fraction of the ar_z - ar_pA gap, with bootstrap on ratio of means
    def rec(arm):
        num = SL[t][arm] - SL[t]["ar_pA"]; den = SL[t]["ar_z"] - SL[t]["ar_pA"]
        m = np.nanmean(num) / np.nanmean(den)
        bs = np.array([np.nanmean(num[b]) / np.nanmean(den[b]) for b in BI])
        return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
    out[f"recovery_frac_{t}"] = {arm: rec(arm) for arm in ("tf_first", "tf_last", "ar_zfull", "ar_hA", "ar_hcA")}
    # interaction: (ar_z - tf_first) - (tf_last - ar_pA): history effect given true last vs given self last
    out[f"interaction_{t}"] = ci((SL[t]["ar_z"] - SL[t]["tf_first"]) - (SL[t]["tf_last"] - SL[t]["ar_pA"]))
    # sensitivity: j3-6 and j2-7 slopes
    for nm, js in (("j3_6", range(3, 7)), ("j2_7", range(2, 8)), ("j4_7", range(4, 8))):
        out[f"tf_first_minus_ar_z_{t}_{nm}"] = ci(slope(along(A[t]["tf_first"]), js) - slope(along(A[t]["ar_z"]), js))
        out[f"tf_last_minus_ar_z_{t}_{nm}"] = ci(slope(along(A[t]["tf_last"]), js) - slope(along(A[t]["ar_z"]), js))
    # per-j along difference
    alz = along(A[t]["ar_z"]); alf = along(A[t]["tf_first"]); all_ = along(A[t]["tf_last"])
    out[f"along_diff_tf_first_minus_ar_z_{t}"] = [ci(np.where(okc, alf[:, j] - alz[:, j], np.nan)) for j in range(J)]
    out[f"along_diff_tf_last_minus_ar_z_{t}"] = [ci(np.where(okc, all_[:, j] - alz[:, j], np.nan)) for j in range(J)]
    out[f"along_mean_{t}"] = {arm: [round(float(np.nanmean(np.where(okc, along(A[t][arm])[:, j], np.nan))), 3) for j in range(J)] for arm in arms}
    # per-scenario tf_first - ar_z
    out[f"tf_first_minus_ar_z_by_scen_{t}"] = {s: round(float(np.nanmean((SL[t]["tf_first"] - SL[t]["ar_z"])[scen == s])), 3) for s in np.unique(scen)}
    out[f"tf_last_minus_ar_z_by_scen_{t}"] = {s: round(float(np.nanmean((SL[t]["tf_last"] - SL[t]["ar_z"])[scen == s])), 3) for s in np.unique(scen)}
out["true_along_mean"] = [round(float(np.nanmean(np.where(okc, along(T)[:, j], np.nan))), 3) for j in range(J)]

# 2) copy baselines: slope of "copy the position of the last fed token"
Tprev = np.concatenate([Lc[:, None], T[:, :-1]], 1)                     # T_{j-1} (T_{-1}=L)
out["copy_last_true_slope"] = ci(slope(along(Tprev)))                   # tf_first / ar_z copy baseline
out["copy_T_jminus2_slope"] = ci(slope(along(np.concatenate([Lc[:, None], Lc[:, None], T[:, :-2]], 1))))
for t in A:
    az = A[t]["ar_z"]; azprev = np.concatenate([az[:, :1], az[:, :-1]], 1)   # position of p^z_{j-1} (fed in tf_last)
    out[f"copy_fed_pz_slope_{t}"] = ci(slope(along(azprev)))
    # fraction of arm argmax equal to T_j / T_{j-1} / T_{j-2} among clips where these three are distinct (moving), j3..7
    res = {}
    for arm in arms:
        a = A[t][arm]; r = {"T_j": [], "T_j-1": [], "T_j-2": [], "n": []}
        for j in range(3, J):
            mv = OK[:, j] & (np.abs(T[:, j] - T[:, j - 1]).max(-1) > 0) & (np.abs(T[:, j - 1] - T[:, j - 2]).max(-1) > 0) & (np.abs(T[:, j] - T[:, j - 2]).max(-1) > 0)
            r["n"].append(int(mv.sum()))
            for key, ref in (("T_j", T[:, j]), ("T_j-1", T[:, j - 1]), ("T_j-2", T[:, j - 2])):
                r[key].append(round(float((a[mv, j] == ref[mv]).all(-1).mean()), 3))
        res[arm] = r
    out[f"landing_distinct3_{t}"] = res
    # tf_last: does it land on the fed predicted token's own cell (copy) vs one step past it?
    tl = A[t]["tf_last"]; r = {"eq_fed_pz_cell": [], "eq_T_j": [], "n": []}
    for j in range(3, J):
        mv = OK[:, j] & (np.abs(T[:, j] - azprev[:, j]).max(-1) > 0)
        r["n"].append(int(mv.sum()))
        r["eq_fed_pz_cell"].append(round(float((tl[mv, j] == azprev[mv, j]).all(-1).mean()), 3))
        r["eq_T_j"].append(round(float((tl[mv, j] == T[mv, j]).all(-1).mean()), 3))
    out[f"tf_last_copy_vs_step_{t}"] = r

# 3) template agreement: fraction of (clip, j) where tru and app argmax coincide per arm
out["template_argmax_agree"] = {arm: [round(float((A["tru"][arm][:, j] == A["app"][arm][:, j]).all(-1).mean()), 3) for j in range(J)] for arm in arms}
# argmax at truth cell under each template for ar_z at j0 (template leak check)
out["j0_hit_exact_T0"] = {t: round(float(((A[t]["ar_z"][:, 0] == T[:, 0]).all(-1) & OK[:, 0]).sum() / OK[:, 0].sum()), 3) for t in A}

# 4) template-free: L1 to h_j (col 8), paired vs ar_z
out["l1_paired_vs_ar_z"] = {arm: [ci(M[:, j, k[arm], 8] - M[:, j, k["ar_z"], 8]) for j in range(J)] for arm in arms if arm != "ar_z"}
# objectness contrast (truth template) and app contrast
out["contrast_tru_mean"] = {arm: [round(float(np.nanmean(np.where(OK[:, j], M[:, j, k[arm], 2] - M[:, j, k[arm], 3], np.nan))), 3) for j in range(J)] for arm in arms}
out["contrast_app_mean"] = {arm: [round(float(np.nanmean(np.where(OK[:, j], M[:, j, k[arm], 6] - M[:, j, k[arm], 7], np.nan))), 3) for j in range(J)] for arm in arms}

# 5) ar_z backward pull: at j6/j7, P(a_j == T_{j-3}) etc., conditioned on T_j != T_{j-3}
for t in A:
    res = {}
    for arm in ("ar_z", "tf_first", "tf_last", "ar_hcA"):
        a = A[t][arm]; d = {}
        for j in (6, 7):
            for back in (2, 3, 4):
                mv = OK[:, j] & (np.abs(T[:, j] - T[:, j - back]).max(-1) > 1)
                d[f"j{j}_eq_T_j-{back}"] = round(float((np.abs(a[mv, j] - T[mv, j - back]).max(-1) <= 0).mean()), 3)
            mv = OK[:, j]
            d[f"j{j}_behind_by_ge2cells"] = round(float(((along(a)[:, j] < along(T)[:, j] - 2) & mv & okc).sum() / (mv & okc).sum()), 3)
        res[arm] = d
    out[f"backward_pull_{t}"] = res

# 6) H5 matched arms (os, ar_p, ar_h) slopes on the same clips
if (H5 / "h5.npy").exists():
    m5 = json.load(open(H5 / "meta.json")); pos5 = {v: i for i, v in enumerate(m5["video_ids"])}; a5 = list(m5["arms"])
    Hh = np.asarray(np.load(H5 / "h5.npy", mmap_mode="r")[[pos5[v] for v in ids], m5["Cs"].index(16)], dtype=np.float64)
    out["H5_arms"] = a5
    out["H5_j0_l1_absdiff_vs_ar_pA"] = float(np.nanmean(np.abs(Hh[:, 0, a5.index("ar_p"), 8] - M[:, 0, k["ar_pA"], 8])))
    for t, cols in (("tru", [1, 0]), ("app", [5, 4])):
        out[f"H5_slope_{t}"] = {a: ci(slope(along(Hh[:, :, i, cols]))) for i, a in enumerate(a5)}
        if "os" in a5:
            os_sl = slope(along(Hh[:, :, a5.index("os"), cols]))
            out[f"tf_last_minus_os_{t}"] = ci(SL[t]["tf_last"] - os_sl)
            out[f"tf_first_minus_os_{t}"] = ci(SL[t]["tf_first"] - os_sl)

json.dump(out, open(OUTJ, "w"), indent=1, default=float)
print(json.dumps(out, default=float)[:200000])
