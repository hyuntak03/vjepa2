#!/usr/bin/env python3
"""VERIFY AR40_READOUT_2026-09-30 — independent recomputation (α variants, readout confound, presence rule, v11, empty FA)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); E = ROOT / "z_research/RollOutV3/exp_results"
OUT = ROOT / "auto_research/exp_results/verify"; OUT.mkdir(parents=True, exist_ok=True)
R, SPLIT, NONE = 144.0, 32, 56
rng = np.random.default_rng(1)
DEC = {"release": "identity_r8", "AR40": "identity_ar40"}
res = {}


def boot(v, blk, B=2000):
    ok = np.isfinite(v); v, blk = v[ok], blk[ok]
    if len(v) == 0: return [np.nan] * 3 + [0]
    ub = np.unique(blk); ix = {u: np.where(blk == u)[0] for u in ub}
    bs = np.array([v[np.concatenate([ix[u] for u in rng.choice(ub, len(ub))])].mean() for _ in range(B)])
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), int(len(v))]


W = {}
for tag, dec in DEC.items():
    z = np.load(E / f"windows_{dec}" / "readings.npz", allow_pickle=True); b = json.load(open(E / dec / "attn_bias_px.json"))
    W[tag] = (z, b)

# ---------------------------------------------------------------- (1) alpha variants
SC_ACC = ["arc", "ledge", "ramp_a", "flat_a"]; SC_DEC = ["flat_d", "ramp_d"]
def alpha_table(tag, rep, win="C16_P32", vsrc=(28, 31), off=1.5, thr=9.0, trange=None):
    z, b = W[tag]
    if f"{win}_{rep}" not in z.files: return None
    P = int(win.split("_P")[1]); tp = P // 2
    pos_ok = z["role"] == "roll"; L = z["truth"] * R; blk = z["block_id"]
    gt = L[:, SPLIT:SPLIT + P].reshape(len(L), tp, 2, 2).mean(2)
    last = L[:, 30:32].mean(1); vctx = (L[:, vsrc[1]] - L[:, vsrc[0]]) / float(vsrc[1] - vsrc[0])
    dt = (2 * np.arange(tp) + off)[None, :, None]; cv = last[:, None] + vctx[:, None] * dt
    v = z[f"{win}_{rep}"]; pr = z[f"{win}_{rep}_prob"].astype(np.float32); present = pr.argmax(-1) != NONE
    bias = np.array(b[rep.split("@")[-1]] if "@" in rep else b[rep], np.float32)   # cross head p@h uses h bias
    pos = v[..., :2] * R - bias
    d = gt - cv; dn = (d ** 2).sum(-1); alpha = ((pos - cv) * d).sum(-1) / np.maximum(dn, 1e-6); sep = np.sqrt(dn) >= thr
    tmask = np.zeros(tp, bool); tmask[trange[0]:trange[1]] = True
    out = {}
    for sc in SC_ACC + SC_DEC:
        m = (pos_ok & (z["scenario"] == sc))[:, None] & tmask[None] & present & sep
        B_ = np.broadcast_to(blk[:, None], m.shape)
        out[sc] = boot(np.clip(alpha[m], -1, 2), B_[m], B=2000 if trange == (0, 8) else 300)
    return out

res["alpha"] = {}
variants = {"doc(28-31,off1.5,thr9)": dict(vsrc=(28, 31), off=1.5, thr=9.0), "v24-31": dict(vsrc=(24, 31), off=1.5, thr=9.0),
            "v30-31": dict(vsrc=(30, 31), off=1.5, thr=9.0), "off0.5": dict(vsrc=(28, 31), off=0.5, thr=9.0), "thr18": dict(vsrc=(28, 31), off=1.5, thr=18.0)}
for name, kw in variants.items():
    for tr in ((4, 8), (8, 16), (0, 8)):
        for tag, rep in (("release", "p"), ("AR40", "p"), ("AR40", "h"), ("release", "z"), ("AR40", "p@h"), ("release", "p@h")):
            t = alpha_table(tag, rep, trange=tr, **kw)
            if t: res["alpha"][f"{name}|t{tr[0]}-{tr[1]-1}|{tag}:{rep}"] = t
# C16/P16 (rollout 8 blocks = in-distribution window) alpha
for tag, rep in (("release", "p"), ("AR40", "p"), ("AR40", "h")):
    t = alpha_table(tag, rep, win="C16_P16", trange=(4, 8))
    if t: res["alpha"][f"C16P16|t4-7|{tag}:{rep}"] = t

# ---------------------------------------------------------------- (2) readout confound: errors, ceilings, cross head
def err_table(tag, rep, win="C16_P32"):
    z, b = W[tag]
    if f"{win}_{rep}" not in z.files: return None
    P = int(win.split("_P")[1]); tp = P // 2; pos_ok = z["role"] == "roll"; L = z["truth"] * R; blk = z["block_id"]
    gt = L[:, SPLIT:SPLIT + P].reshape(len(L), tp, 2, 2).mean(2); last = L[:, 30:32].mean(1)
    v = z[f"{win}_{rep}"]; pr = z[f"{win}_{rep}_prob"].astype(np.float32); present = pr.argmax(-1) != NONE
    bias = np.array(b[rep.split("@")[-1]] if "@" in rep else b[rep], np.float32); pos = v[..., :2] * R - bias
    e_t = np.linalg.norm(pos - gt, axis=-1); e_copy = np.linalg.norm(pos - last[:, None], axis=-1)
    out = {}
    for sc in sorted(set(z["scenario"])):
        row = {}
        for nm, sl in (("t0-3", (0, 4)), ("t4-7", (4, 8)), ("t8-15", (8, 16))):
            tm = np.zeros(tp, bool); tm[sl[0]:sl[1]] = True
            m = (pos_ok & (z["scenario"] == sc))[:, None] & tm[None]; pm = m & present
            B_ = np.broadcast_to(blk[:, None], m.shape)
            row[nm] = dict(present=boot(present[m].astype(float), B_[m], 300), err_truth=boot(e_t[pm], B_[pm], 300), err_copy=boot(e_copy[pm], B_[pm], 300),
                           present_score=boot((v[..., 2] > 0)[m].astype(float), B_[m], 300))
        out[sc] = row
    return out
res["err"] = {f"{tag}:{rep}": err_table(tag, rep) for tag, rep in (("release", "p"), ("AR40", "p"), ("AR40", "h"), ("release", "z"), ("AR40", "p@h"), ("release", "p@h"), ("release", "z@h"))}
res["err"] = {k: v for k, v in res["err"].items() if v}
res["train_center_cells"] = {tag: {r: json.load(open(E / dec / "summary.json"))["reps"][r]["center_mean"] for r in json.load(open(E / dec / "summary.json"))["reps"]} for tag, dec in DEC.items()}

# ---------------------------------------------------------------- (4)(5) v11
res["v11"] = {}
for tag, dec in DEC.items():
    z = np.load(E / f"v11_{dec}" / "readings.npz", allow_pickle=True); b = json.load(open(E / dec / "attn_bias_px.json"))
    L = z["truth"] * R; k = z["sym_k"].astype(int); obj = z["obj"]; cond = z["condition"]; blk = z["block_id"]; hid = z["hidden"]
    gt = L[:, 16:32].reshape(len(L), 8, 2, 2).mean(2)
    last_seen = np.array([L[i, max(0, 16 - kk - 1)] for i, kk in enumerate(k)])     # 마지막 본 자리 (샘플 16−k−1)
    for rep in ("p", "h"):
        v = z[rep][:, -8:] if rep == "h" else z[rep]; pr = (z[f"{rep}_prob"][:, -8:] if rep == "h" else z[f"{rep}_prob"]).astype(np.float32)
        present = pr.argmax(-1) != NONE; pos = v[..., :2] * R - np.array(b[rep], np.float32)
        e_t = np.linalg.norm(pos - gt, axis=-1); e_last = np.linalg.norm(pos - last_seen[:, None], axis=-1)
        nanflag = int(np.isnan(v).sum() + np.isnan(pr).sum() + (v == -1).all(-1).sum())
        for motion, cm in (("static", ["static_visible", "static_occlusion"]), ("moving_flat", ["moving_visible_flat", "moving_occlusion_flat"]), ("moving_ramp", ["moving_visible", "moving_occlusion"])):
            for kk in range(5):
                m = obj & np.isin(cond, cm) & (k == kk)
                if not m.any(): continue
                B_ = np.broadcast_to(blk[:, None], (len(L), 8)); mm = np.zeros((len(L), 8), bool); mm[m] = True; pm = mm & present; t37 = np.arange(8)[None] >= 3
                res["v11"].setdefault(motion, {}).setdefault(f"k{kk}", {})[f"{tag}:{rep}"] = dict(
                    n=int(m.sum()), nan=nanflag, present_by_t=[round(float(present[m, t].mean()), 3) for t in range(8)],
                    present_all=boot(present[mm].astype(float), B_[mm], 500), present_t3_7=boot(present[mm & t37].astype(float), B_[mm & t37], 500),
                    err_truth_t3_7=boot(e_t[pm & t37], B_[pm & t37], 500), err_lastseen_t3_7=boot(e_last[pm & t37], B_[pm & t37], 500))
        me = (~obj) & np.isin(cond, ["static_occlusion", "moving_occlusion_flat", "moving_occlusion"])
        res["v11"].setdefault("empty_fa", {})[f"{tag}:{rep}"] = dict(all=round(float(present[me].mean()), 4), by_t=[round(float(present[me, t].mean()), 3) for t in range(8)], n=int(me.sum()))
json.dump(res, open(OUT / "verify_ar40rd.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- print
print("### α variants (mean [CI] n)")
for key in sorted(res["alpha"]):
    row = res["alpha"][key]; print(f"{key:46s} " + " ".join(f"{sc}:{row[sc][0]:+.2f}[{row[sc][1]:+.2f},{row[sc][2]:+.2f}]n{row[sc][3]}" for sc in SC_ACC + SC_DEC))
print("\n### errors px (present cells) — err_truth / err_copy / present% / present(score>0)%")
for key, tab in res["err"].items():
    for sc in ("arc", "ledge", "ramp_a", "flat_a", "flat_d", "ramp_d", "flat_v", "wall"):
        r = tab[sc]; print(f"{key:12s} {sc:7s} " + " | ".join(f"{nm}: {r[nm]['err_truth'][0]:.0f}[{r[nm]['err_truth'][1]:.0f},{r[nm]['err_truth'][2]:.0f}] / {r[nm]['err_copy'][0]:.0f} / {100*r[nm]['present'][0]:.0f}% / {100*r[nm]['present_score'][0]:.0f}%" for nm in ("t0-3", "t4-7", "t8-15")))
print("\n### train center (cells)", res["train_center_cells"])
print("\n### v11")
for motion, ks in res["v11"].items():
    if motion == "empty_fa": print("empty FA", ks); continue
    for kk, rows in ks.items():
        for key, r in rows.items():
            print(f"{motion:12s} {kk} {key:10s} nan{r['nan']} present_by_t {r['present_by_t']} all {r['present_all'][:3]} t3-7 {r['present_t3_7'][:3]} err_truth {r['err_truth_t3_7'][:3]} err_last {r['err_lastseen_t3_7'][:3]}")
