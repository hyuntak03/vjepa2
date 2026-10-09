#!/usr/bin/env python3
"""Independent re-analysis (verifier, round 2) of TRAINED_PREDICTORS_SIGNATURES — P2 raw arrays.
Does NOT import repo analysis scripts. Reads vll6 cache v3_p2{,_pv1,_ariel,_v11ft}.
Readouts: truth-cell template argmax (loc_tru), appearance template argmax (loc_app), apph (h ceiling).
Null = same law, different trajectory, all valid clips (exhaustive mean). CI = trajectory-cluster bootstrap.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
OUTF = Path(__file__).resolve().parent / "v_p2_out.json"
TAGS = ["", "_pv1", "_ariel", "_v11ft"]
CELL, THR, B = 18.0, 0.357, 1000

metas = {t: json.load(open(C / f"v3_p2{t}" / "meta.json")) for t in TAGS}
ids = metas[""]["video_ids"]
for t in TAGS:
    assert metas[t]["video_ids"] == ids, f"video_ids differ for {t}"
    assert metas[t]["arms"] == metas[""]["arms"]
arms = metas[""]["arms"]; n = len(ids)
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fl = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([fl(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
scen = np.array(metas[""]["scenario"]); traj = np.array(["|".join(map(str, t)) for t in metas[""]["traj"]])
ut = np.unique(traj); tix = {u: np.where(traj == u)[0] for u in ut}
rng = np.random.default_rng(7)
BI = [np.concatenate([tix[u] for u in rng.choice(ut, len(ut))]) for _ in range(B)]
LAWS = np.unique(scen); SAME = {L: np.where(scen == L)[0] for L in LAWS}
D = {t: {k: np.load(C / f"v3_p2{t}" / f"{k}.npy") for k in ("loc_tru", "loc_app", "loc_apph", "l1")} for t in TAGS}
out = {"n": n, "n_traj": len(ut), "laws": {L: int(len(v)) for L, v in SAME.items()}}


def ci(v):
    v = np.asarray(v, float); bs = np.array([np.nanmean(v[b]) for b in BI])
    return [round(float(np.nanmean(v)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


def null_ex(am, cti, okt):
    o = np.full(n, np.nan)
    for L, ix in SAME.items():
        M = np.abs(am[ix][:, None] - cti[ix][None]).max(-1) <= 1
        V = (traj[ix][:, None] != traj[ix][None]) & okt[ix][None]
        o[ix] = (M & V).sum(1) / np.maximum(V.sum(1), 1)
    return o


def geom(A):
    fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
    cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
    fa2, fb2 = fr[2 * nc - 4], fr[2 * nc - 3]
    cl2 = np.stack([(X[:, fa2] + X[:, fb2]) / 2, (Y[:, fa2] + Y[:, fb2]) / 2], -1) / CELL
    okl = INF[:, fa] & INF[:, fb]
    slots = []
    for t in range(nf):
        g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
        ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
        okt = INF[:, g0] & INF[:, g1]
        slots.append(dict(ct=ct, cti=np.clip(np.floor(ct), 0, 15), okt=okt, ts=(g0 + g1) / 2 - (fa + fb) / 2))
    return cl, cl2, okl, slots, (fb + fa) / 2 - (fb2 + fa2) / 2


# ---- 0. sanity: NaN counts, identical h (apph must be identical across predictors: same encoder) ----
san = {}
for t in TAGS:
    san[t or "release"] = {k: int(np.isnan(D[t][k][..., 0]).sum()) for k in ("loc_tru", "loc_app", "loc_apph")}
san["apph_identical_across_tags"] = bool(all(np.array_equal(np.nan_to_num(D[t]["loc_apph"][..., :2]), np.nan_to_num(D[""]["loc_apph"][..., :2])) for t in TAGS))
san["l1c_copy_identical_across_tags"] = bool(all(np.allclose(np.nan_to_num(D[t]["l1"][..., 1]), np.nan_to_num(D[""]["l1"][..., 1]), atol=1e-3) for t in TAGS))
out["sanity"] = san

res = {}
KEYS = {"s1_C16": [3, 4, 6, 8, 10, 12, 14, 15], "s2_C16": [2, 3, 4, 5, 6, 7], "s4_C8": [0, 1, 2, 3], "s2_C8": [2, 3, 4, 5, 6, 7], "s1_C32": [3, 8, 10, 14]}
for ai, A in enumerate(arms):
    if A["name"] not in KEYS: continue
    cl, cl2, okl, slots, dstep = geom(A)
    vel = (cl - cl2) / dstep                       # cells per raw frame (context end)
    ra = {}
    per = {t: {} for t in TAGS}
    oracle = {}
    for s, S in enumerate(slots):
        ok = okl & S["okt"] & ~np.isnan(D[""]["loc_tru"][:, ai, s, 0])
        # oracles (position only): copy last cell, CV extrapolation (true context-end velocity), stop-after-3 (truth at slot min(s,2))
        orc = {"copy": np.clip(np.floor(cl), 0, 15),
               "cv": np.clip(np.floor(cl + vel * S["ts"]), 0, 15),
               "stop3": slots[min(s, 2)]["cti"]}
        for nm, am in orc.items():
            h = (np.abs(am - S["cti"]).max(-1) <= 1).astype(float); nu = null_ex(am, S["cti"], S["okt"])
            h[~ok] = np.nan; nu[~ok] = np.nan
            oracle.setdefault(nm, {})[s] = (h, nu)
        for t in TAGS:
            L = D[t]
            am = L["loc_tru"][:, ai, s][:, [1, 0]]; ap = L["loc_app"][:, ai, s][:, [1, 0]]
            hit = (np.abs(am - S["cti"]).max(-1) <= 1).astype(float); nu = null_ex(am, S["cti"], S["okt"])
            app = (np.abs(ap - S["cti"]).max(-1) <= 1).astype(float); nua = null_ex(ap, S["cti"], S["okt"])
            obj = ((L["loc_tru"][:, ai, s, 2] - L["loc_tru"][:, ai, s, 3]) >= THR).astype(float)
            dtr = np.abs(am - S["cti"]).max(-1)
            for v in (hit, nu, app, nua, obj): v[~ok] = np.nan
            per[t][s] = dict(hn=hit - nu, an=app - nua, hit=hit, nu=nu, app=app, nua=nua, obj=obj, dtr=np.where(ok, dtr, np.nan),
                             mx=np.where(ok, L["loc_tru"][:, ai, s, 2], np.nan), md=np.where(ok, L["loc_tru"][:, ai, s, 3], np.nan),
                             apx=ap, ok=ok)
    for t in TAGS:
        rt = {}
        for s in KEYS[A["name"]]:
            if s >= len(slots): continue
            P = per[t][s]
            r = dict(hit_null=ci(P["hn"]), app_null=ci(P["an"]), app=ci(P["app"]), app_nullrate=ci(P["nua"]), obj=ci(P["obj"]),
                     maxcos=round(float(np.nanmean(P["mx"])), 3), medcos=round(float(np.nanmean(P["md"])), 3))
            # objectness decoupling: objectness among far-miss clips (Chebyshev >= 3 from truth) vs hits
            fm = P["dtr"] >= 3; hh = P["dtr"] <= 1
            r["obj_given_farmiss"] = [round(float(np.nanmean(np.where(fm, P["obj"], np.nan))), 3), int(np.nansum(fm))]
            r["obj_given_hit"] = [round(float(np.nanmean(np.where(hh, P["obj"], np.nan))), 3), int(np.nansum(hh))]
            if t:
                r["paired_hn_minus_release"] = ci(P["hn"] - per[""][s]["hn"])
                r["paired_an_minus_release"] = ci(P["an"] - per[""][s]["an"])
            rt[s] = r
        ra[t or "release"] = rt
    ra["oracles"] = {nm: {s: dict(hit_null=ci(v[s][0] - v[s][1]), hit=round(float(np.nanmean(v[s][0])), 3), null=round(float(np.nanmean(v[s][1])), 3))
                          for s in KEYS[A["name"]] if s < len(slots)} for nm, v in oracle.items()}
    # censored cliff by speed tercile (integer rule: first slot where hit-null < half of slot-0 value), per tag
    spd = np.linalg.norm(vel, axis=-1); q = np.nanpercentile(spd[okl], [33.33, 66.67]); terc = np.digitize(spd, q)
    cl_out = {}
    for t in TAGS:
        cc = []
        for k in range(3):
            m = terc == k
            curve = [float(np.nanmean(np.where(m, per[t][s]["hn"], np.nan))) for s in range(len(slots))]
            c0 = curve[0]; cs = next((s for s, v in enumerate(curve) if v < 0.5 * c0), None)
            cc.append(dict(cliff=cs, censored=cs is None, curve=[round(v, 3) for v in curve]))
        cl_out[t or "release"] = cc
    ra["cliff_terciles"] = cl_out
    # displacement model on appearance argmax (s1_C16 only): along = alpha + beta * true_along
    if A["name"] in ("s1_C16", "s2_C16", "s4_C8"):
        dm = {}
        for t in TAGS:
            rows = []
            for s in range(len(slots)):
                P = per[t][s]; ct = slots[s]["ct"]; d = ct - cl; dist = np.linalg.norm(d, axis=-1)
                u = d / np.maximum(dist, 1e-6)[:, None]
                al = ((P["apx"] + 0.5 - cl) * u).sum(-1)
                m = P["ok"] & (dist >= 0.5) & np.isfinite(al)
                rows.append((s, dist[m], al[m], traj[m]))
            fits = {}
            for lo, hi in ((0, 3), (3, 7), (7, 11), (11, 16)):
                xs = np.concatenate([r[1] for r in rows if lo <= r[0] < hi] or [np.zeros(0)]); ys = np.concatenate([r[2] for r in rows if lo <= r[0] < hi] or [np.zeros(0)])
                if len(xs) < 20: continue
                bta = np.polyfit(xs, ys, 1)
                fits[f"slots{lo}-{hi-1}"] = dict(beta=round(float(bta[0]), 3), alpha_cells=round(float(bta[1]), 3), n=int(len(xs)),
                                                 mean_true=round(float(xs.mean()), 2), mean_pred=round(float(ys.mean()), 2),
                                                 median_ratio=round(float(np.median(ys / xs)), 3))
            # per-slot mean along and true along
            fits["per_slot_pred_true"] = [[s, round(float(np.mean(r[2])), 2) if len(r[2]) else None, round(float(np.mean(r[1])), 2) if len(r[1]) else None] for s, r in [(r[0], r) for r in rows]]
            dm[t or "release"] = fits
        ra["displacement_app"] = dm
        # location-prior concentration of app argmax: share of clips (per law) at the law's modal cell, averaged over laws
        conc = {}
        for t in TAGS:
            cs = []
            for s in range(len(slots)):
                P = per[t][s]; sh = []
                for L, ix in SAME.items():
                    ii = ix[P["ok"][ix]]
                    if len(ii) < 10: continue
                    cells = (P["apx"][ii, 1] * 16 + P["apx"][ii, 0]).astype(int)
                    sh.append(np.bincount(cells, minlength=256).max() / len(ii))
                    # truth concentration for reference
                cs.append(round(float(np.mean(sh)), 3) if sh else None)
            conc[t or "release"] = cs
        tc = []
        for s in range(len(slots)):
            sh = []
            for L, ix in SAME.items():
                ii = ix[(okl & slots[s]["okt"])[ix]]
                if len(ii) < 10: continue
                cells = (slots[s]["cti"][ii, 1] * 16 + slots[s]["cti"][ii, 0]).astype(int)
                sh.append(np.bincount(cells, minlength=256).max() / len(ii))
            tc.append(round(float(np.mean(sh)), 3) if sh else None)
        conc["truth"] = tc
        ra["modal_cell_share"] = conc
    # per-law app-null at a far slot
    far = {"s1_C16": 10, "s2_C16": 6, "s4_C8": 3, "s2_C8": 6, "s1_C32": 10}[A["name"]]
    if far < len(slots):
        ra["per_law_far"] = {t or "release": {L: [round(float(np.nanmean(per[t][far]["an"][ix])), 3), round(float(np.nanmean(per[t][far]["hn"][ix])), 3)] for L, ix in SAME.items()} for t in TAGS}
        ra["per_law_far_slot"] = far
    res[A["name"]] = ra
out["arms"] = res
json.dump(out, open(OUTF, "w"), indent=1, default=float)
print("wrote", OUTF)
