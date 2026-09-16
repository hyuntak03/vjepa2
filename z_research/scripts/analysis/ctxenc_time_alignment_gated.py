#!/usr/bin/env python3
"""Time-alignment matrix with an OBJECT-SPECIFIC gate — does p's slot i align with true time 8+i because of the object, or only
because every token carries a per-tubelet time signature (background tokens were also diagonal in the v1 pilot)?  Cache-only, no GPU, 0 parameters.

Same clips, same null partners and same z–h control as z_research/scripts/analysis/ctxenc_time_alignment.py (identical rng sequence:
RollOut_v2 plausible, total future displacement >= 2 cells, <= 60/scenario, seed 0; v11 flat/ramp k=0 vanish pos_a, first 224 blocks).

Per clip we first compute token-level distances d[i][j][s] = mean_d |p_i,s - LN(h)_j,s| (8 x 16 x 256; h rows are already LN'd in the
cache), then every token subset is a mean over s.  Three additions to v1:
 (1) union gate:   c_S(i) = (median_j - min_j)/median_j of D_S[i] with S = 3x3 trajectory UNION (v1 set), c_BG(i) the same on the complement;
                   dc(i) = c_S(i) - c_BG(i).  Null A: same S / BG split but h from ANOTHER clip (v1's partner: same scenario for v2, opposite
                   travel direction for v11).  Null B (extra, within clip): S' = the trajectory windows translated by a random in-frame shift
                   disjoint from S, BG' = complement of S u S'.  Gate per slot = 95th percentile of the null dc over clips; matched if dc > gate.
 (2) per-tubelet:  S_j = only the 3x3 at the true position at time j (in-frame tubelets only), so D_pt[i][j] = mean_{s in S_j} d[i][j][s] compares
                   p_i with h_j at the place the object IS at time j; BG_j = complement of S_j.  Same c / dc / nulls / gate.  j with the object
                   out of frame is NaN; a clip enters a slot only if >= MIN_VALID_J of the 16 columns are valid.
 (3) two curves per slot: argmin alignment (fraction of clips with argmin_j within +-1 of 8+i, S / BG / per-tubelet / z-h) and
                   contrast relative to the per-slot null (mean c, mean null c, fraction above the per-slot null 95th pct).
                   'break slot' of a curve = first slot i where it drops below 0.5 (8 = never).
Control: LN(z)[8+i] (isolated_ctx over all 32 frames) in place of p_i, through exactly the same sets, nulls and gates.
CIs: block bootstrap over clips (v2: 1 plausible clip per block, so clip = block; v11: block = pos_a clip), B resamples, the gate is
recomputed inside every resample from the resampled null.  No fitted readout anywhere, so no split / ridge is involved.
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated/{results.json, RESULTS.md, per_clip.npz}
Run:
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_time_alignment_gated.py [--per-scen 60] [--v11-n 224] [--boot 500]
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxenc_time_alignment as v1                      # data selection, ln, n3, token_set, refine, contrast — reused verbatim

ROOT = v1.ROOT
OUT_DEFAULT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated"
S_TOK, T_FUT, T_ALL, D, CELL_PX = v1.S_TOK, v1.T_FUT, v1.T_ALL, v1.D, v1.CELL_PX
TARGET = np.arange(T_FUT) + T_FUT
MIN_VALID_J = 6
GATE_PCT = 95.0
SHIFT_MIN_CELLS = 4
ALL_TOK = np.arange(S_TOK)


def token_dist(a, b):
    """a (8,256,D) fp32, b (16,256,D) fp32 -> d (8,16,256) = mean over D of |a_i,s - b_j,s|."""
    out = np.empty((a.shape[0], b.shape[0], S_TOK), np.float32)
    for i in range(a.shape[0]):
        out[i] = np.abs(a[i][None] - b).mean(-1)
    return out


def contrast_nan(row):
    v = row[np.isfinite(row)]
    if v.size < MIN_VALID_J:
        return np.nan
    m = float(np.median(v)); return (m - float(v.min())) / m if m > 0 else 0.0


def refine_nan(row):
    ok = np.isfinite(row)
    if ok.sum() < MIN_VALID_J:
        return np.nan
    r = np.where(ok, row, np.inf); j = int(r.argmin())
    if 0 < j < len(r) - 1 and ok[j - 1] and ok[j + 1]:
        a, b, c = r[j - 1], r[j], r[j + 1]; den = a - 2 * b + c
        if den > 0:
            j = j + 0.5 * (a - c) / den
    return float(j)


def argmin_nan(row):
    ok = np.isfinite(row)
    if ok.sum() < MIN_VALID_J:
        return -99
    return int(np.where(ok, row, np.inf).argmin())


def per_tubelet_windows(tub, inf):
    """list over j of the 3x3 token window at the true position at time j (None if out of frame / empty)."""
    W = []
    for t in range(T_ALL):
        w = np.array(sorted(set(v1.n3(*tub[t]))), int) if inf[t] else np.zeros(0, int)
        W.append(w if w.size else None)
    return W


def shifted_windows(tub, inf, S_union, rng, tries=200):
    """translate the whole trajectory by an integer cell shift (|shift| >= SHIFT_MIN_CELLS) so that every in-frame window stays in frame and
    the shifted union is disjoint from S_union.  Returns (windows list, union) or (None, None)."""
    for _ in range(tries):
        dx, dy = rng.integers(-12, 13, size=2)
        if max(abs(dx), abs(dy)) < SHIFT_MIN_CELLS:
            continue
        shift = np.array([dx, dy], float) * CELL_PX
        W, U = [], set()
        ok = True
        for t in range(T_ALL):
            if not inf[t]:
                W.append(None); continue
            x, y = tub[t] + shift
            if not (0 <= x < 288 and 0 <= y < 288):
                ok = False; break
            w = np.array(sorted(set(v1.n3(x, y))), int); W.append(w); U.update(w.tolist())
        if not ok or not U:
            continue
        U = np.array(sorted(U), int)
        if np.intersect1d(U, S_union).size:
            continue
        return W, U
    return None, None


def row_stats(Dm):
    """Dm (8,16) possibly NaN -> dict of per-slot arrays: contrast, jstar, argmin."""
    return (np.array([contrast_nan(Dm[i]) for i in range(T_FUT)]), np.array([refine_nan(Dm[i]) for i in range(T_FUT)]),
            np.array([argmin_nan(Dm[i]) for i in range(T_FUT)]))


def subset_matrix(d, S):
    return d[:, :, S].mean(-1) if S is not None and len(S) else np.full((T_FUT, T_ALL), np.nan, np.float32)


def per_tubelet_matrix(d, W, complement=False, exclude=None):
    M = np.full((T_FUT, T_ALL), np.nan, np.float32)
    for j in range(T_ALL):
        w = W[j]
        if w is None:
            continue
        if complement:
            drop = w if exclude is None or exclude[j] is None else np.union1d(w, exclude[j])
            w = np.setdiff1d(ALL_TOK, drop)
        M[:, j] = d[:, j, w].mean(-1)
    return M


def analyse_clip(d_ph, d_pu, d_zh, d_zu, S, W, Wsh, Ush):
    """d_* : (8,16,256) token distances for p-h(own), p-h(other clip), z-h(own), z-h(other).  Returns dict of per-slot arrays."""
    BG = np.setdiff1d(ALL_TOK, S)
    out = {}
    for tag, d in (("p", d_ph), ("pnull", d_pu), ("z", d_zh), ("znull", d_zu)):
        if d is None:
            continue
        cS, jS, aS = row_stats(subset_matrix(d, S)); cB, jB, aB = row_stats(subset_matrix(d, BG))
        cT, jT, aT = row_stats(per_tubelet_matrix(d, W)); cTB, jTB, aTB = row_stats(per_tubelet_matrix(d, W, complement=True))
        out[tag] = dict(cS=cS, jS=jS, aS=aS, cB=cB, jB=jB, aB=aB, dc=cS - cB, cT=cT, jT=jT, aT=aT, cTB=cTB, jTB=jTB, aTB=aTB, dcT=cT - cTB)
        if tag in ("p", "z") and Ush is not None:                                   # null B: shifted trajectory, same clip's h
            BGx = np.setdiff1d(BG, Ush)
            cS2, _, _ = row_stats(subset_matrix(d, Ush)); cB2, _, _ = row_stats(subset_matrix(d, BGx))
            cT2, _, _ = row_stats(per_tubelet_matrix(d, Wsh)); cTB2, _, _ = row_stats(per_tubelet_matrix(d, Wsh, complement=True, exclude=W))
            out[tag]["dc_shift"] = cS2 - cB2; out[tag]["dcT_shift"] = cT2 - cTB2
    out["D_pS"] = subset_matrix(d_ph, S); out["D_pT"] = per_tubelet_matrix(d_ph, W); out["D_pB"] = subset_matrix(d_ph, BG)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- summaries
def nanpct(x, q):
    x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if x.size else np.nan


def gated_metrics(dc, dc_null, jstar, arg, idx=None):
    """dc, dc_null, jstar (n,8); returns per-slot gate, match frac, mean j* matched, within±1 frac, n valid."""
    if idx is not None:
        dc, dc_null, jstar, arg = dc[idx], dc_null[idx], jstar[idx], arg[idx]
    gate = np.array([nanpct(dc_null[:, i], GATE_PCT) for i in range(T_FUT)])
    valid = np.isfinite(dc)
    match = valid & (dc > gate[None])
    nval = valid.sum(0)
    frac = np.where(nval > 0, match.sum(0) / np.maximum(nval, 1), np.nan)
    mj = np.array([float(np.nanmean(jstar[match[:, i], i])) if match[:, i].any() else np.nan for i in range(T_FUT)])
    aligned = (np.abs(arg - TARGET[None]) <= 1) & valid
    w1 = np.where(nval > 0, aligned.sum(0) / np.maximum(nval, 1), np.nan)
    both = np.where(nval > 0, (match & aligned).sum(0) / np.maximum(nval, 1), np.nan)      # object-specific AND at the right time
    return gate, frac, mj, w1, nval, both


def break_slot(curve, thr=0.5):
    for i, v in enumerate(curve):
        if not np.isfinite(v) or v < thr:
            return i
    return T_FUT


def fl(a):
    return [None if (v is None or not np.isfinite(v)) else float(v) for v in np.asarray(a, float)]


def summarise(clips, B, rng):
    n = len(clips)
    if n == 0:
        return {"n": 0}
    st = lambda tag, key: np.array([c[tag][key] for c in clips], float)                                        # (n,8)
    res = {"n": n, "n_S_mean": float(np.mean([c["n_S"] for c in clips])), "n_shift_ok": int(sum(1 for c in clips if "dc_shift" in c["p"]))}
    have_z = all("z" in c for c in clips)
    variants = {"union": ("dc", "jS", "aS", "dc_shift", "cS", "cB", "jB", "aB"), "per_tubelet": ("dcT", "jT", "aT", "dcT_shift", "cT", "cTB", "jTB", "aTB")}
    for vname, (kdc, kj, ka, kdc2, kcS, kcB, kjB, kaB) in variants.items():
        for who in (("p",) + (("z",) if have_z else ())):
            dc, dcn, js, ar = st(who, kdc), st(who + "null", kdc), st(who, kj), st(who, ka)
            cS, cSn, cB, cBn, jB, aB = st(who, kcS), st(who + "null", kcS), st(who, kcB), st(who + "null", kcB), st(who, kjB), st(who, kaB)
            has2 = [i for i, c in enumerate(clips) if kdc2 in c[who]]
            dc2 = np.array([clips[i][who][kdc2] for i in has2], float) if has2 else None
            gate, frac, mj, w1, nval, both = gated_metrics(dc, dcn, js, ar)
            # bootstrap over clips (== blocks)
            bf, bj, bw, bg, bbrk, bbrk_w1, bfB, bboth, bbrk_both = [], [], [], [], [], [], [], [], []
            for _ in range(B):
                idx = rng.integers(n, size=n)
                g, f, m, w, _, bo = gated_metrics(dc, dcn, js, ar, idx)
                bf.append(f); bj.append(m); bw.append(w); bg.append(g); bbrk.append(break_slot(f)); bbrk_w1.append(break_slot(w)); bboth.append(bo); bbrk_both.append(break_slot(bo))
            bf, bj, bw, bg, bboth = np.array(bf), np.array(bj), np.array(bw), np.array(bg), np.array(bboth)
            ci = lambda a: (fl(np.nanpercentile(a, 2.5, axis=0)), fl(np.nanpercentile(a, 97.5, axis=0)))
            # BG-token argmin curve (time signature) + CI
            validB = np.isfinite(cB); w1B = ((np.abs(aB - TARGET[None]) <= 1) & validB).sum(0) / np.maximum(validB.sum(0), 1)
            for _ in range(B):
                idx = rng.integers(n, size=n); vb = validB[idx]; bfB.append(((np.abs(aB[idx] - TARGET[None]) <= 1) & vb).sum(0) / np.maximum(vb.sum(0), 1))
            bfB = np.array(bfB)
            # per-slot null-95 of the S contrast alone (curve (3): contrast relative to null)
            cS_gate = np.array([nanpct(cSn[:, i], GATE_PCT) for i in range(T_FUT)])
            cS_above = np.where(np.isfinite(cS).sum(0) > 0, ((cS > cS_gate[None]) & np.isfinite(cS)).sum(0) / np.maximum(np.isfinite(cS).sum(0), 1), np.nan)
            # null-B (shifted trajectory) gate and match fraction
            nb = {}
            if dc2 is not None and len(has2) >= 10:
                g2 = np.array([nanpct(dc2[:, i], GATE_PCT) for i in range(T_FUT)]); dcs = dc[has2]; al2 = (np.abs(ar[has2] - TARGET[None]) <= 1)
                v2_ = np.isfinite(dcs); f2 = np.where(v2_.sum(0) > 0, ((dcs > g2[None]) & v2_).sum(0) / np.maximum(v2_.sum(0), 1), np.nan)
                b2 = np.where(v2_.sum(0) > 0, ((dcs > g2[None]) & v2_ & al2).sum(0) / np.maximum(v2_.sum(0), 1), np.nan)
                bf2, bb2, bbrk2 = [], [], []
                for _ in range(B):
                    idx = rng.integers(len(has2), size=len(has2)); gg = np.array([nanpct(dc2[idx][:, i], GATE_PCT) for i in range(T_FUT)])
                    vv = np.isfinite(dcs[idx]); mm = (dcs[idx] > gg[None]) & vv
                    bf2.append(mm.sum(0) / np.maximum(vv.sum(0), 1)); bo = (mm & al2[idx]).sum(0) / np.maximum(vv.sum(0), 1); bb2.append(bo); bbrk2.append(break_slot(bo))
                bf2, bb2 = np.array(bf2), np.array(bb2)
                nb = {"n": len(has2), "gate_shift": fl(g2), "match_frac_shift_gate": fl(f2), "match_frac_shift_gate_ci": ci(bf2),
                      "dc_shift_mean": fl(np.nanmean(dc2, 0)), "break_slot_shift_gate": break_slot(f2),
                      "frac_shift_gated_and_aligned": fl(b2), "frac_shift_gated_and_aligned_ci": ci(bb2),
                      "break_slot_shift_gated_and_aligned": break_slot(b2), "break_slot_shift_gated_and_aligned_ci": [int(np.percentile(bbrk2, 2.5)), int(np.percentile(bbrk2, 97.5))],
                      "slots_passing_shift_gate_and_aligned": [i for i in range(T_FUT) if np.isfinite(b2[i]) and b2[i] > 0.5 and np.nanpercentile(bb2[:, i], 2.5) > 0.05]}
            res.setdefault(vname, {})[who] = {
                "n_valid_per_slot": [int(v) for v in nval],
                "gate_per_slot": fl(gate), "gate_ci": ci(bg),
                "dc_mean": fl(np.nanmean(dc, 0)), "dc_null_mean": fl(np.nanmean(dcn, 0)), "dc_null_p95": fl(gate),
                "match_frac_gated": fl(frac), "match_frac_gated_ci": ci(bf),
                "frac_gated_and_aligned": fl(both), "frac_gated_and_aligned_ci": ci(bboth),
                "break_slot_gated_and_aligned": break_slot(both), "break_slot_gated_and_aligned_ci": [int(np.percentile(bbrk_both, 2.5)), int(np.percentile(bbrk_both, 97.5))],
                "slots_passing_gate_and_aligned": [i for i in range(T_FUT) if np.isfinite(both[i]) and both[i] > 0.5 and np.nanpercentile(bboth[:, i], 2.5) > 0.05],
                "mean_jstar_matched": fl(mj), "mean_jstar_matched_ci": ci(bj),
                "mean_jstar_all": fl(np.nanmean(js, 0)), "mean_offset_all": fl(np.nanmean(js - TARGET[None], 0)),
                "frac_argmin_within1": fl(w1), "frac_argmin_within1_ci": ci(bw),
                "frac_argmin_within1_bg": fl(w1B), "frac_argmin_within1_bg_ci": ci(bfB), "mean_jstar_bg": fl(np.nanmean(jB, 0)),
                "contrast_S_mean": fl(np.nanmean(cS, 0)), "contrast_S_null_mean": fl(np.nanmean(cSn, 0)), "contrast_S_null_p95": fl(cS_gate),
                "frac_contrast_S_above_null95": fl(cS_above),
                "contrast_BG_mean": fl(np.nanmean(cB, 0)), "contrast_BG_null_mean": fl(np.nanmean(cBn, 0)),
                "break_slot_gated": break_slot(frac), "break_slot_gated_ci": [int(np.percentile(bbrk, 2.5)), int(np.percentile(bbrk, 97.5))],
                "break_slot_argmin_within1": break_slot(w1), "break_slot_argmin_within1_ci": [int(np.percentile(bbrk_w1, 2.5)), int(np.percentile(bbrk_w1, 97.5))],
                "break_slot_argmin_within1_bg": break_slot(w1B),
                "slots_passing_gate": [i for i in range(T_FUT) if np.isfinite(frac[i]) and frac[i] > 0.5 and np.nanpercentile(bf[:, i], 2.5) > 0.05],
                "null_B_shifted": nb,
            }
    res["mean_D_pS"] = np.nanmean([c["D_pS"] for c in clips], 0).round(4).tolist()
    res["mean_D_pT"] = np.nanmean([c["D_pT"] for c in clips], 0).round(4).tolist()
    res["mean_D_pB"] = np.nanmean([c["D_pB"] for c in clips], 0).round(4).tolist()
    return res


# ----------------------------------------------------------------------------------------------------------------------------- per-clip driver
def process(p, hl, zl, hl_u, tub, inf, rng_shift):
    S = v1.token_set(tub, inf); W = per_tubelet_windows(tub, inf); Wsh, Ush = shifted_windows(tub, inf, S, rng_shift)
    d_ph = token_dist(p, hl); d_pu = token_dist(p, hl_u)
    d_zh = token_dist(zl[T_FUT:], hl) if zl is not None else None; d_zu = token_dist(zl[T_FUT:], hl_u) if zl is not None else None
    c = analyse_clip(d_ph, d_pu, d_zh, d_zu, S, W, Wsh, Ush); c["n_S"] = len(S); c["n_inframe_future"] = int(inf[T_FUT:].sum())
    return c


def run_v2(per_scen, rng, rng_shift, B, rng_boot):
    idx = [r for r in csv.DictReader(v1.V2_INDEX.open()) if r["plausible"] == "1"]
    vids = json.loads((v1.V2_CACHE / "meta.json").read_text())["video_ids"]; vids2 = json.loads((v1.V2_CTX32 / "meta.json").read_text())["video_ids"]
    assert vids == vids2
    row = {v: i for i, v in enumerate(vids)}
    P = np.load(v1.V2_CACHE / "predictor.npy", mmap_mode="r"); H = np.load(v1.V2_CACHE / "target.npy", mmap_mode="r"); Z = np.load(v1.V2_CTX32 / "isolated_ctx_0_32.npy", mmap_mode="r")
    geo = {}
    for r in idx:
        assert r["resolution"] == "288"
        t32 = np.stack([v1.arr(r["px_x_by_sample"]), v1.arr(r["px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1); inf = (v1.arr(r["in_frame_by_sample"]) > 0).reshape(16, 2).all(1)
        geo[r["video_id"]] = (tub, inf)
    res, per_clip = {}, {}
    for sc in v1.SCEN:
        pool = [r["video_id"] for r in idx if r["scenario"] == sc]
        qual = [v for v in pool if np.linalg.norm(geo[v][0][15] - geo[v][0][8]) / CELL_PX >= v1.MIN_DISP_CELLS and geo[v][1][T_FUT:].any()]
        sel = list(rng.choice(qual, min(per_scen, len(qual)), replace=False)) if qual else []            # identical draw to v1
        clips = []; t0 = time.time()
        for v in sel:
            tub, inf = geo[v]; others = [u for u in pool if u != v]; u = others[rng.integers(len(others))]                 # identical partner to v1
            p = np.asarray(P[row[v]], np.float32).reshape(T_FUT, S_TOK, D)
            hl = v1.ln(np.asarray(H[row[v]], np.float32)).reshape(T_ALL, S_TOK, D); zl = v1.ln(np.asarray(Z[row[v]], np.float32)).reshape(T_ALL, S_TOK, D)
            hl_u = v1.ln(np.asarray(H[row[u]], np.float32)).reshape(T_ALL, S_TOK, D)
            c = process(p, hl, zl, hl_u, tub, inf, rng_shift); c["video_id"] = v; c["null_partner"] = u; clips.append(c)
        res[sc] = summarise(clips, B, rng_boot); res[sc]["counts"] = {"plausible": len(pool), "qualified_disp_ge_2cells": len(qual), "used": len(sel)}
        per_clip[sc] = clips
        print(f"[v2] {sc:<7} used {len(sel):3d}/{len(qual):3d} of {len(pool)}  ({time.time()-t0:.0f}s)", flush=True); _print(sc, res[sc])
    return res, per_clip


def run_v11(n_per_cond, rng, rng_shift, B, rng_boot):
    idx = list(csv.DictReader(v1.V11_INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(v1.V11_META.open())}
    row = {v: i for i, v in enumerate(json.loads((v1.V11_CACHE / "meta.json").read_text())["video_ids"])}
    zrow = {v: i for i, v in enumerate(json.loads((v1.V11_CTX32 / "meta.json").read_text())["video_ids"])}
    P = np.load(v1.V11_CACHE / "predictor.npy", mmap_mode="r"); H = np.load(v1.V11_CACHE / "target.npy", mmap_mode="r"); Z = np.load(v1.V11_CTX32 / "isolated_ctx_0_32.npy", mmap_mode="r")
    res, per_clip = {}, {}
    for name, cond, K in v1.V11_CONDS:
        pos = {r["block_id"]: r for r in idx if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == "pos_a"}
        blocks = sorted(pos)[:n_per_cond]; geo = {}
        for b in blocks:
            m = meta[pos[b]["video_id"]]; assert m["resolution"] == "288"
            t32 = np.stack([v1.arr(m["object_px_x_by_sample"]), v1.arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1)
            inf = np.array([(0 <= tub[t, 0] < 288) and (0 <= tub[t, 1] < 288) for t in range(T_ALL)]); geo[b] = (tub, inf, m["travel_direction"])
        qual = [b for b in blocks if np.linalg.norm(geo[b][0][15] - geo[b][0][8]) / CELL_PX >= v1.MIN_DISP_CELLS and geo[b][1][T_FUT:].any()]
        have_z = all(pos[b]["video_id"] in zrow for b in qual); clips = []; t0 = time.time(); nfb = 0
        for b in qual:
            v = pos[b]["video_id"]; tub, inf, d = geo[b]
            opp = [c for c in qual if c != b and geo[c][2] != d]
            if not opp:
                opp = [c for c in qual if c != b]; nfb += 1
            u = opp[rng.integers(len(opp))]                                                                                     # identical partner to v1
            p = np.asarray(P[row[v]], np.float32).reshape(T_FUT, S_TOK, D); hl = v1.ln(np.asarray(H[row[v]], np.float32)).reshape(T_ALL, S_TOK, D)
            zl = v1.ln(np.asarray(Z[zrow[v]], np.float32)).reshape(T_ALL, S_TOK, D) if have_z else None
            hl_u = v1.ln(np.asarray(H[row[pos[u]["video_id"]]], np.float32)).reshape(T_ALL, S_TOK, D)
            c = process(p, hl, zl, hl_u, tub, inf, rng_shift); c["video_id"] = v; c["null_partner"] = pos[u]["video_id"]; clips.append(c)
        res[name] = summarise(clips, B, rng_boot); res[name]["counts"] = {"blocks_selected": len(blocks), "qualified_disp_ge_2cells": len(qual), "used": len(clips), "control_z_available": have_z, "null_no_opposite_direction_fallback": nfb}
        per_clip[name] = clips
        print(f"[v11] {name:<9} used {len(clips)}/{len(blocks)}  z {'yes' if have_z else 'NO'} ({time.time()-t0:.0f}s)", flush=True); _print(name, res[name])
    return res, per_clip


def _f(vs, f="{:5.2f}"):
    return " ".join("  -  " if v is None else f.format(v) for v in vs)


def _print(name, r):
    if r.get("n", 0) == 0:
        print("   (no clips)"); return
    for vn in ("union", "per_tubelet"):
        for who in ("p", "z"):
            if who not in r[vn]:
                continue
            q = r[vn][who]
            print(f"   {vn:<11} {who}  gated match {_f(q['match_frac_gated'])}  | argmin±1 {_f(q['frac_argmin_within1'])}  | bg argmin±1 {_f(q['frac_argmin_within1_bg'])}")
            print(f"   {'':<11} {who}  dc {_f(q['dc_mean'], '{:5.3f}')}  gate {_f(q['gate_per_slot'], '{:5.3f}')}  j*(m) {_f(q['mean_jstar_matched'])}  break gated {q['break_slot_gated']} argmin {q['break_slot_argmin_within1']} bg {q['break_slot_argmin_within1_bg']}")


# ----------------------------------------------------------------------------------------------------------------------------- report
def ci_s(v, ci, i, f="{:.2f}"):
    if v[i] is None:
        return "–"
    lo, hi = ci[0][i], ci[1][i]
    return f.format(v[i]) + (f" [{f.format(lo)}, {f.format(hi)}]" if lo is not None and hi is not None else "")


def write_md(out, path):
    L = ["# Time alignment with an object-specific gate (p slot i vs true tubelet j) — cache-only, 0 parameters", "",
         f"Generated from `results.json` in this folder ({out['generated']}). Same clips / null partners / z–h control as `time_alignment/` (v1).", "",
         "Definitions — `c(i)` = (median_j − min_j)/median_j of the row D[i][·]; **Δc(i) = c_S(i) − c_BG(i)**; gate = per-slot 95th percentile of Δc on the null "
         "(null A: same S/BG split, h from another clip — v2 same scenario, v11 opposite travel direction; null B: same clip's h, trajectory windows translated ≥ 4 cells, disjoint from S). "
         "*union* = S is the 3×3 union over all in-frame tubelets (v1's set); *per-tubelet* = S_j is only the 3×3 at the true position at time j (D[i][j] compares p_i with h_j where the object IS at time j). "
         "`argmin±1` = fraction of clips whose argmin_j is within ±1 of 8+i (S tokens); `bg argmin±1` = the same on the complement (time signature). "
         "**break slot** = first slot where a curve drops below 0.5 (8 = never). `passing` = slots with gated match > 0.5 and bootstrap CI lower bound > 0.05 (= the gate's own false-positive rate). "
         f"CIs: block bootstrap over clips, B = {out['boot']} resamples, gate recomputed inside each resample. Slots i = 0..7 ↔ true tubelets 8..15.", ""]
    for arm, res in (("RollOut_v2 (plausible, future displacement ≥ 2 cells, ≤ 60/scenario, seed 0)", out["v2"]), ("IntPhysGen v11 k=0 vanish pos_a (first 224 blocks per condition)", out.get("v11", {}))):
        if not res or "skipped" in res:
            L += [f"## {arm}", "", f"skipped: {res.get('skipped') if res else '-'}", ""]; continue
        L += [f"## {arm}", ""]
        for vn, title in (("union", "(1) trajectory-union S with object-specific gate Δc = c_S − c_BG"), ("per_tubelet", "(2) per-tubelet S_j (3×3 where the object is at time j)")):
            L += [f"### {title}", "", "| scenario | n | who | metric | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | break |", "|---|---:|---|---|---|---|---|---|---|---|---|---|---|"]
            for sc, r in res.items():
                if r.get("n", 0) == 0:
                    L.append(f"| {sc} | 0 | | (no clip passes the ≥ 2-cell filter) | | | | | | | | | |"); continue
                for who in ("p", "z"):
                    if who not in r[vn]:
                        continue
                    q = r[vn][who]; nn = r["n"]
                    L.append(f"| {sc} | {nn} | {who} | **gated match (null A)** | " + " | ".join(ci_s(q["match_frac_gated"], q["match_frac_gated_ci"], i) for i in range(T_FUT)) + f" | {q['break_slot_gated']} [{q['break_slot_gated_ci'][0]}, {q['break_slot_gated_ci'][1]}] |")
                    L.append(f"| | | {who} | Δc mean / gate | " + " | ".join(("–" if q["dc_mean"][i] is None else f"{q['dc_mean'][i]:.3f} / {q['gate_per_slot'][i]:.3f}") for i in range(T_FUT)) + " | |")
                    L.append(f"| | | {who} | mean j* (matched) | " + " | ".join(ci_s(q["mean_jstar_matched"], q["mean_jstar_matched_ci"], i, "{:.1f}") for i in range(T_FUT)) + " | |")
                    L.append(f"| | | {who} | **gated AND argmin±1** | " + " | ".join(ci_s(q["frac_gated_and_aligned"], q["frac_gated_and_aligned_ci"], i) for i in range(T_FUT)) + f" | {q['break_slot_gated_and_aligned']} [{q['break_slot_gated_and_aligned_ci'][0]}, {q['break_slot_gated_and_aligned_ci'][1]}] |")
                    L.append(f"| | | {who} | argmin±1 (S) | " + " | ".join(ci_s(q["frac_argmin_within1"], q["frac_argmin_within1_ci"], i) for i in range(T_FUT)) + f" | {q['break_slot_argmin_within1']} [{q['break_slot_argmin_within1_ci'][0]}, {q['break_slot_argmin_within1_ci'][1]}] |")
                    L.append(f"| | | {who} | argmin±1 (BG, time signature) | " + " | ".join(ci_s(q["frac_argmin_within1_bg"], q["frac_argmin_within1_bg_ci"], i) for i in range(T_FUT)) + f" | {q['break_slot_argmin_within1_bg']} |")
                    L.append(f"| | | {who} | c_S mean / null-95 | " + " | ".join(("–" if q["contrast_S_mean"][i] is None else f"{q['contrast_S_mean'][i]:.3f} / {q['contrast_S_null_p95'][i]:.3f}") for i in range(T_FUT)) + " | |")
                    L.append(f"| | | {who} | frac c_S > null-95 | " + " | ".join(("–" if q["frac_contrast_S_above_null95"][i] is None else f"{q['frac_contrast_S_above_null95'][i]:.2f}") for i in range(T_FUT)) + " | |")
                    nb = q["null_B_shifted"]
                    if nb:
                        L.append(f"| | | {who} | gated match (null B, n={nb['n']}) | " + " | ".join(ci_s(nb["match_frac_shift_gate"], nb["match_frac_shift_gate_ci"], i) for i in range(T_FUT)) + f" | {nb['break_slot_shift_gate']} |")
                        L.append(f"| | | {who} | **null-B gated AND argmin±1** | " + " | ".join(ci_s(nb["frac_shift_gated_and_aligned"], nb["frac_shift_gated_and_aligned_ci"], i) for i in range(T_FUT)) + f" | {nb['break_slot_shift_gated_and_aligned']} [{nb['break_slot_shift_gated_and_aligned_ci'][0]}, {nb['break_slot_shift_gated_and_aligned_ci'][1]}] |")
                    L.append(f"| | | {who} | n valid | " + " | ".join(str(v) for v in q["n_valid_per_slot"]) + " | |")
                    L.append(f"| | | {who} | passing slots (A gate / A gate AND aligned / B gate AND aligned) | {q['slots_passing_gate']} / {q['slots_passing_gate_and_aligned']} / {nb.get('slots_passing_shift_gate_and_aligned', 'n/a')} | | | | | | | | |")
            L.append("")
        L += ["### Mean D rows (p, per-tubelet S_j; NaN = object out of frame at j)", ""]
        for sc, r in res.items():
            if r.get("n", 0) == 0:
                continue
            L += [f"**{sc}** (n={r['n']})", "", "| i \\ j | " + " | ".join(str(j) for j in range(16)) + " |", "|---|" + "---|" * 16]
            for i, rowv in enumerate(r["mean_D_pT"]):
                vals = [v for v in rowv if v is not None]; mn = min(vals) if vals else None
                L.append(f"| p{i} (true {8+i}) | " + " | ".join(("–" if v is None else (f"**{v:.3f}**" if v == mn else f"{v:.3f}")) for v in rowv) + " |")
            L.append("")
    L += ["## Verdict per scenario (from the numbers above)", ""] + out["verdict_lines"] + ["", "## Caveats", ""] + [f"- {c}" for c in out["caveats"]]
    path.write_text("\n".join(L) + "\n")


def verdicts(out):
    L = []
    for arm in ("v2", "v11"):
        res = out.get(arm, {})
        if not res or "skipped" in res:
            continue
        for sc, r in res.items():
            if r.get("n", 0) == 0:
                L.append(f"- {arm} {sc}: no clips."); continue
            pu, pt = r["union"]["p"], r["per_tubelet"]["p"]
            mj = lambda q, s: "–" if not s else ", ".join(f"{q['mean_jstar_matched'][i]:.1f}" for i in s)
            nbu, nbt = pu["null_B_shifted"], pt["null_B_shifted"]
            L.append(f"- **{arm} {sc}** (n={r['n']}): union gate (null A) passes at slots {pu['slots_passing_gate']} (mean j* there: {mj(pu, pu['slots_passing_gate'])}; expected {[8+i for i in pu['slots_passing_gate']]}), "
                     f"A-gate AND argmin±1 at {pu['slots_passing_gate_and_aligned']}, B-gate AND argmin±1 at {nbu.get('slots_passing_shift_gate_and_aligned', 'n/a')}; "
                     f"per-tubelet A-gate passes at {pt['slots_passing_gate']} (mean j*: {mj(pt, pt['slots_passing_gate'])}), A-gate AND argmin±1 at {pt['slots_passing_gate_and_aligned']}, B-gate AND argmin±1 at {nbt.get('slots_passing_shift_gate_and_aligned', 'n/a')}. "
                     f"Break slots — A-gated Δc: union {pu['break_slot_gated']}, per-tubelet {pt['break_slot_gated']}; A-gated AND aligned: union {pu['break_slot_gated_and_aligned']}, per-tubelet {pt['break_slot_gated_and_aligned']}; "
                     f"B-gated AND aligned: union {nbu.get('break_slot_shift_gated_and_aligned', 'n/a')}, per-tubelet {nbt.get('break_slot_shift_gated_and_aligned', 'n/a')}; "
                     f"argmin±1: S {pu['break_slot_argmin_within1']}, BG {pu['break_slot_argmin_within1_bg']}, per-tubelet S_j {pt['break_slot_argmin_within1']}."
                     + (f" z control (per-tubelet): A-gated break {r['per_tubelet']['z']['break_slot_gated']}, B-gated AND aligned break {r['per_tubelet']['z']['null_B_shifted'].get('break_slot_shift_gated_and_aligned', 'n/a')}." if "z" in r["union"] else " z control: n/a."))
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-scen", type=int, default=60); ap.add_argument("--v11-n", type=int, default=224); ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--skip-v11", action="store_true"); ap.add_argument("--out", default=str(OUT_DEFAULT)); a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "boot": a.boot, "gate_pct": GATE_PCT, "min_valid_j": MIN_VALID_J, "shift_min_cells": SHIFT_MIN_CELLS,
           "args": vars(a), "token_layout": "idx = tubelet*256 + row*16 + col; cell = px // 18 on the 288 render; tubelet pos = mean of samples 2t, 2t+1 (v11_token_object_test.py n3)"}
    rng = np.random.default_rng(0); rng_shift = np.random.default_rng(1); rng_boot = np.random.default_rng(2)
    t0 = time.time(); out["v2"], pc2 = run_v2(a.per_scen, rng, rng_shift, a.boot, rng_boot); out["v2_seconds"] = time.time() - t0
    pc = dict(pc2)
    if a.skip_v11:
        out["v11"] = {"skipped": "--skip-v11"}
    else:
        try:
            t0 = time.time(); out["v11"], pc11 = run_v11(a.v11_n, np.random.default_rng(0), rng_shift, a.boot, rng_boot); out["v11_seconds"] = time.time() - t0; pc.update(pc11)
        except Exception as e:                                             # noqa: BLE001
            out["v11"] = {"skipped": f"{type(e).__name__}: {e}"}; print("v11 arm failed:", e)
    out["verdict_lines"] = verdicts(out)
    out["caveats"] = [
        "Null A keeps the clip's own S/BG split and swaps in another clip's h (v1 used the partner's S_B); so the v1 'gate' numbers are not directly comparable.",
        "Δc on the union S is a difference of two contrasts on token sets of different size (|S| ≈ 30–55 vs ≈ 200–225); the null has exactly the same split so the size asymmetry is in the gate.",
        "The z–h control is an encoder of the SAME frames, so on union S both S and BG match h at j = 8+i and Δc(z) can be ≈ 0 while the argmin is perfectly diagonal; "
        "the per-tubelet S_j version is the meaningful control for z (S_j moves with the object). Read the z rows with that in mind.",
        "Per-tubelet D[i][j] mixes token sets across j; the contrast then also reflects per-token baseline differences. Null A (same S_j, other h) carries the same mixing.",
        "Per-tubelet rows need ≥ 6 in-frame columns; clips with the object out of frame for most tubelets drop out of that slot (n valid per slot is in the tables).",
        "Null B (shifted trajectory) needs a ≥ 4-cell in-frame shift disjoint from S; clips where no such shift exists are excluded from null B (n reported).",
        "Null A is degenerate wherever the partner clip's object sits at (almost) the same place at the same time: v2 ramp_a / arc / ledge use one trajectory geometry per scenario, "
        "and v11 opposite-direction partners cross the same central cells near the boundary. There the A-gate is high and the z control itself fails it; read null B (within-clip shift) in those cells.",
        "Bootstrap is over clips; in RollOut_v2 every plausible clip is its own block (verified: 4704 clips / 4704 blocks), in v11 the unit is the pos_a block.",
        "The 95th-percentile gate has a 5 % false-positive rate by construction; 'passing' requires the CI lower bound of the match fraction to exceed 0.05.",
        "wall (plausible) is stationary in the future → no clip passes the ≥ 2-cell displacement filter.",
    ]
    (OUT / "results.json").write_text(json.dumps(out, indent=1)); write_md(out, OUT / "RESULTS.md")
    keys = ["cS", "cB", "dc", "jS", "aS", "cT", "cTB", "dcT", "jT", "aT", "jB", "aB", "jTB", "aTB"]
    npz = {}
    for sc, clips in pc.items():
        for who in ("p", "pnull", "z", "znull"):
            if all(who in c for c in clips) and clips:
                for k in keys:
                    npz[f"{sc}/{who}/{k}"] = np.array([c[who][k] for c in clips], np.float32)
                for k in ("dc_shift", "dcT_shift"):                                          # NaN row where no valid shift existed
                    npz[f"{sc}/{who}/{k}"] = np.array([c[who].get(k, np.full(T_FUT, np.nan)) for c in clips], np.float32)
        npz[f"{sc}/video_id"] = np.array([c["video_id"] for c in clips]); npz[f"{sc}/null_partner"] = np.array([c["null_partner"] for c in clips])
    np.savez_compressed(OUT / "per_clip.npz", **npz)
    print(f"→ {OUT}/results.json  RESULTS.md  per_clip.npz")


if __name__ == "__main__":
    main()
