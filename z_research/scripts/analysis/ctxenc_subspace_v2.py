#!/usr/bin/env python3
"""E3-v2 — does a global linear p→h map (fit on visible POSSIBLE clips, held-out blocks) change matched-pair surprise scoring (accuracy / margin) in visible, late, early × static / flat / ramp? Plus the subspace-overlap / CKA / Procrustes tables of ctxenc_subspace_overlap.py extended with the accelerating (ramp) family, per-k late rows with block-bootstrap CIs, the z→h control beside every p→h row, absolute L1 after each map and a norm-matched Procrustes variant.

Frozen test of CLAUDE.md §10 '정렬 사상 개입' (item 4/5). Cache only (no model, no GPU), deterministic (seed 0, closed-form ridge / Procrustes).

Data  : data_csv/intphysgen_v11_full/index_probe.csv (= index.csv + sym_k / probe_type), cache /local_datasets/world/world_analysis/cache/v11_full_vith (float16, LN'd z/h, raw p).
        9 conditions: {static, moving_flat, moving(ramp)} × {visible k=0, occlusion late k=1..4, occlusion_early}; *_mid excluded (budget).
Part A (subspace, as v1 + ramp): 9 sets × N_CLIPS obj clips (one per block, stratified violation × k; early sets k=4 as in v1) × N_TOK random tokens.
        per-k rows: late sets, N_PERK blocks per k (all 256 tokens/clip subsampled to N_TOK), overlaps with block-bootstrap 95 % CI (N_BOOT_SUB resamples, block-level sufficient statistics).
        alignment maps p→h fit on visible train blocks (60/20/20 block split of the set) — Procrustes, Procrustes+scale, norm-matched Procrustes (p rescaled to h's per-token std before fitting/applying), ridge (λ by validation fold), shift-only; z→h control on the same rows; absolute L1 after each map.
Part B (matched-pair RE-SCORING under the map):
        visible blocks split 50/50 by block_id (stratified violation_type, seed 0): fit half → N_FIT train + N_VAL val blocks per visible condition (possible clips pos_a/pos_b only, all 2048 future tokens, streaming sufficient statistics); test half → N_TEST blocks per visible condition.
        occlusion conditions: N_TEST blocks per condition stratified by violation_type × sym_k (all their blocks are disjoint from the fit blocks by construction).
        surprise(v) = mean over 2048 future tokens × 1280 dims of |f(p) − h_v| with p shared inside a matched pair (pos_a, imp_ab) / (pos_b, imp_ba) [pair_id A / B];
        accuracy = fraction of pairs with surprise(imp) > surprise(pos) (tie = 0.5), margin = surprise(imp) − surprise(pos).
        f ∈ {raw, shift-only, per-token std rescale (= affine-free LN of p), global scalar rescale, Procrustes, Procrustes+scale, norm-matched Procrustes, ridge, norm-matched ridge,
             per-family Procrustes / ridge, CONTROL ridge fit p→h of a different clip, CONTROL z-scoring |z − h| and |ridge_z→h(z) − h|}.
        Verification: raw per-video surprise vs per_video_surprise of the established run (max |Δ|), and raw accuracy on the SAME blocks vs that run's per_block.json.
        Block-bootstrap 95 % CI (N_BOOT resamples over blocks) for accuracy, margin and their paired deltas vs raw.
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/{results.json, RESULTS.md}
Run (CPU only, ~20–30 min; env overrides N_CLIPS N_TOK N_PERK N_FIT N_VAL N_TEST N_BOOT N_BOOT_SUB WORKERS):
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2 && /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_subspace_v2.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith"
OUT = Path(os.environ.get("OUT_DIR", ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2"))
SEED = 0
N_CLIPS = int(os.environ.get("N_CLIPS", 120))
N_TOK = int(os.environ.get("N_TOK", 256))
N_PERK = int(os.environ.get("N_PERK", 60))
N_FIT = int(os.environ.get("N_FIT", 160))
N_VAL = int(os.environ.get("N_VAL", 40))
N_TEST = int(os.environ.get("N_TEST", 288))
N_BOOT = int(os.environ.get("N_BOOT", 1000))
N_BOOT_SUB = int(os.environ.get("N_BOOT_SUB", 200))
WORKERS = int(os.environ.get("WORKERS", 10))
KS = (16, 64)
D = 1280
HALF = 2048
REPS = ("z", "p", "h")
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

# subspace sets (v1 six + ramp family); name -> (index condition, allowed sym_k)
SETS = {
    "static_visible": ("static_visible", [0]), "static_late": ("static_occlusion", [1, 2, 3, 4]), "static_early": ("static_occlusion_early", [4]),
    "movflat_visible": ("moving_visible_flat", [0]), "movflat_late": ("moving_occlusion_flat", [1, 2, 3, 4]), "movflat_early": ("moving_occlusion_flat_early", [4]),
    "ramp_visible": ("moving_visible", [0]), "ramp_late": ("moving_occlusion", [1, 2, 3, 4]), "ramp_early": ("moving_occlusion_early", [4]),
}
FAMILIES = {"static": ("static_visible", "static_late", "static_early"), "movflat": ("movflat_visible", "movflat_late", "movflat_early"),
            "ramp": ("ramp_visible", "ramp_late", "ramp_early")}
# rescoring conditions: family -> (visible, late, early) index conditions
RS_FAM = {"static": ("static_visible", "static_occlusion", "static_occlusion_early"),
          "movflat": ("moving_visible_flat", "moving_occlusion_flat", "moving_occlusion_flat_early"),
          "ramp": ("moving_visible", "moving_occlusion", "moving_occlusion_early")}
RS_CONDS = [c for fam in RS_FAM.values() for c in fam]
FAM_OF = {c: f for f, cs in RS_FAM.items() for c in cs}

_G = {}  # globals for forked workers


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def open_cache():
    row_of = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r")
    assert Z.shape[1] == HALF and P.shape[1] == HALF and H.shape[1] == 2 * HALF and Z.shape[2] == D
    return row_of, Z, P, H


def rd(A, r, sl=None):
    return np.asarray(A[r] if sl is None else A[r][sl], dtype=np.float32)


# ----------------------------------------------------------------------------- sampling (v1)
def stratified(df, n, strata, rng):
    """n rows stratified equally over `strata` columns, top-up from the rest; deterministic given rng."""
    groups = df.groupby(strata)
    per = n // groups.ngroups
    picks = [g.sample(n=min(per, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in groups]
    out = pd.concat(picks)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)), random_state=int(rng.integers(1 << 31)))])
    return out.sort_values("block_id").reset_index(drop=True)


def sample_blocks(df, cond, ks, n, rng, strata=("violation_type", "sym_k")):
    sub = df[(df.probe_type == "obj") & (df.condition == cond) & (df.sym_k.isin(ks))]
    sub = sub.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).drop_duplicates("block_id")
    return stratified(sub, n, list(strata), rng)


def load_set(clips, row_of, Z, P, H, rng, with_other=True):
    n = len(clips)
    tok = np.stack([np.sort(rng.choice(HALF, N_TOK, replace=False)) for _ in range(n)])
    perm = rng.permutation(n)
    while n > 1 and np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    z = np.empty((n, N_TOK, D), np.float32); p = np.empty_like(z); h = np.empty_like(z); h_other = np.empty_like(z) if with_other else None
    for i, vid in enumerate(clips.video_id):
        r = row_of[vid]; idx = tok[i]
        z[i] = Z[r][idx]; p[i] = P[r][idx]; h[i] = H[r][HALF + idx]
        if with_other:
            h_other[i] = H[row_of[clips.video_id.iloc[perm[i]]]][HALF + idx]
    meta = dict(n_clips=n, n_tok_per_clip=N_TOK, n_tokens=n * N_TOK, n_blocks=int(clips.block_id.nunique()),
                k_counts={int(k): int(v) for k, v in clips.sym_k.value_counts().sort_index().items()},
                violation_counts={k: int(v) for k, v in clips.violation_type.value_counts().sort_index().items()})
    return dict(z=z, p=p, h=h, h_other=h_other, clips=clips, tok=tok, meta=meta)


# ----------------------------------------------------------------------------- (a) subspaces (v1)
def pca_basis(X, kmax=max(KS)):
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    w = w[::-1]; V = V[:, ::-1]
    ev = {k: float(w[:k].sum() / w.sum()) for k in KS}
    pr = float(w.sum() ** 2 / (w ** 2).sum())
    return V[:, :kmax], ev, pr


def overlap(U1, U2, k):
    s = np.linalg.svd(U1[:, :k].T @ U2[:, :k], compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def random_overlap(k, rng, n_draw=10):
    vals = []
    for _ in range(n_draw):
        A = np.linalg.qr(rng.standard_normal((D, k)))[0]; B = np.linalg.qr(rng.standard_normal((D, k)))[0]
        vals.append(overlap(A, B, k))
    return float(np.mean(vals)), float(np.std(vals))


def linear_cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    return float(np.linalg.norm(Xc.T @ Yc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)))


# ----------------------------------------------------------------------------- maps (closed form; from sufficient statistics)
def tokstd(X, s_h=1.0):
    """norm-matched p: rescale every token to h's per-token std (h is affine-free LN'd → std = s_h ≈ 1)."""
    return X * (s_h / (X.std(axis=-1, keepdims=True) + 1e-6))


class Stats:
    """streaming sufficient statistics for a linear map X -> Y: n, Σx, Σy, XᵀX, XᵀY (float64)."""

    def __init__(self):
        self.n = 0; self.sx = np.zeros(D); self.sy = np.zeros(D); self.xx = np.zeros((D, D)); self.xy = np.zeros((D, D)); self.sy2 = 0.0

    def add(self, X, Y):
        self.n += len(X); self.sx += X.sum(0, dtype=np.float64); self.sy += Y.sum(0, dtype=np.float64)
        self.xx += (X.T @ X).astype(np.float64); self.xy += (X.T @ Y).astype(np.float64)

    def __iadd__(self, o):
        self.n += o.n; self.sx += o.sx; self.sy += o.sy; self.xx += o.xx; self.xy += o.xy; return self

    def centered(self):
        mx, my = self.sx / self.n, self.sy / self.n
        G = self.xx - self.n * np.outer(mx, mx); B = self.xy - self.n * np.outer(mx, my)
        return mx, my, G, B


def stats_of(X, Y):
    s = Stats(); s.add(X, Y); return s


def fit_procrustes_stats(st):
    mx, my, G, B = st.centered()
    U, S, Vt = np.linalg.svd(B)
    W = (U @ Vt).astype(np.float32)
    scale = float(S.sum() / np.trace(G))
    return dict(kind="procrustes", W=W, mx=mx.astype(np.float32), my=my.astype(np.float32), scale=scale)


def fit_ridge_stats(st, Xv, Yv):
    mx, my, G, B = st.centered()
    tr = np.trace(G) / D
    best = None
    for lam in LAMS:
        W = np.linalg.solve(G + lam * tr * np.eye(D), B).astype(np.float32)
        l1 = float(np.abs((Xv - mx.astype(np.float32)) @ W + my.astype(np.float32) - Yv).mean())
        if best is None or l1 < best[0]:
            best = (l1, lam, W)
    return dict(kind="ridge", W=best[2], mx=mx.astype(np.float32), my=my.astype(np.float32), lam=best[1], val_l1=best[0])


def apply_map(m, X, variant="plain"):
    if variant == "shift":
        return X - m["mx"] + m["my"]
    if variant == "scaled":
        return m["scale"] * (X - m["mx"]) @ m["W"] + m["my"]
    return (X - m["mx"]) @ m["W"] + m["my"]


def l1(A, B):
    return float(np.abs(A - B).mean())


def fit_family_maps(Xtr, Ytr, Xva, Yva, s_h):
    """the map set used in part A(c): proc, proc+scale, ridge on p; norm-matched proc/ridge on tokstd(p)."""
    st = stats_of(Xtr, Ytr); stn = stats_of(tokstd(Xtr, s_h), Ytr)
    return dict(proc=fit_procrustes_stats(st), ridge=fit_ridge_stats(st, Xva, Yva),
                nm_proc=fit_procrustes_stats(stn), nm_ridge=fit_ridge_stats(stn, tokstd(Xva, s_h), Yva), s_h=s_h)


def eval_maps(maps, X, Y):
    base = l1(X, Y); ynorm = float(np.abs(Y).mean())
    Xn = tokstd(X, maps["s_h"])
    after = {"shift_only": l1(apply_map(maps["proc"], X, "shift"), Y), "tokstd_only": l1(Xn, Y),
             "procrustes": l1(apply_map(maps["proc"], X), Y), "procrustes_scaled": l1(apply_map(maps["proc"], X, "scaled"), Y),
             "nm_procrustes": l1(apply_map(maps["nm_proc"], Xn), Y), "ridge": l1(apply_map(maps["ridge"], X), Y), "nm_ridge": l1(apply_map(maps["nm_ridge"], Xn), Y)}
    out = dict(l1_before=base, n_tokens=int(len(X)), mean_abs_target=ynorm)
    for k, v in after.items():
        out[k] = dict(l1=v, frac_removed=float(1 - v / base) if base > 0 else float(1 - v / ynorm), rel_residual=float(v / ynorm))
    return out


def split_blocks(clips, rng, fr=(0.6, 0.2, 0.2)):
    tr, va, te = [], [], []
    for _, g in clips.groupby(["violation_type", "sym_k"]):
        ids = rng.permutation(g.index.values)
        n = len(ids); a = int(round(fr[0] * n)); b = int(round((fr[0] + fr[1]) * n))
        tr += list(ids[:a]); va += list(ids[a:b]); te += list(ids[b:])
    return np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int), np.array(sorted(te), dtype=int)


def flat(a, sel=None):
    a = a if sel is None else a[sel]
    return a.reshape(-1, D)


# ----------------------------------------------------------------------------- per-k bootstrap worker
def _perk_worker(args):
    name, k, clips_json, seed = args
    row_of, Z, P, H = _G["row_of"], _G["Z"], _G["P"], _G["H"]
    clips = pd.read_json(io.StringIO(clips_json))
    rng = np.random.default_rng(seed)
    n = len(clips)
    tok = np.stack([np.sort(rng.choice(HALF, N_TOK, replace=False)) for _ in range(n)])
    # block-level sufficient statistics (sum, XᵀX) per rep
    S = {r: np.zeros((n, D)) for r in REPS}; M = {r: np.zeros((n, D, D)) for r in REPS}
    pf, hf, zf = [], [], []
    for i, vid in enumerate(clips.video_id):
        r = row_of[vid]; idx = tok[i]
        x = {"z": rd(Z, r, idx), "p": rd(P, r, idx), "h": rd(H, r, HALF + idx)}
        pf.append(x["p"]); hf.append(x["h"]); zf.append(x["z"])
        for rep in REPS:
            S[rep][i] = x[rep].sum(0, dtype=np.float64); M[rep][i] = (x[rep].T @ x[rep]).astype(np.float64)
    pf, hf, zf = np.concatenate(pf), np.concatenate(hf), np.concatenate(zf)

    def basis(sel, rep):
        s = S[rep][sel].sum(0); m = M[rep][sel].sum(0); N = len(sel) * N_TOK
        C = (m - np.outer(s, s) / N) / (N - 1)
        w, V = sla.eigh(C, subset_by_index=[D - max(KS), D - 1])
        return V[:, ::-1]

    def all_ov(sel):
        Up, Uh, Uz = basis(sel, "p"), basis(sel, "h"), basis(sel, "z")
        return {f"p-vs-h|{kk}": overlap(Up, Uh, kk) for kk in KS} | {f"z-vs-h|{kk}": overlap(Uz, Uh, kk) for kk in KS} | {f"z-vs-p|{kk}": overlap(Uz, Up, kk) for kk in KS}

    point = all_ov(np.arange(n))
    boots = {key: [] for key in point}
    for _ in range(N_BOOT_SUB):
        sel = rng.integers(0, n, n)
        for key, v in all_ov(sel).items():
            boots[key].append(v)
    ci = {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in boots.items()}
    # transfer of the family's visible-fit maps (passed in _G["maps"]) — L1 before/after on these tokens
    fam = name.split("_")[0]; maps = _G["maps"][fam]
    ev = eval_maps(maps, pf, hf)
    out = dict(set=name, k=int(k), n_blocks=n, n_tokens=int(n * N_TOK), n_boot=N_BOOT_SUB, overlap=point, overlap_ci95=ci,
               cka_p_h=linear_cka(pf, hf), cka_z_h=linear_cka(zf, hf),
               l1_before=ev["l1_before"], l1_after={m: ev[m]["l1"] for m in ("shift_only", "tokstd_only", "procrustes", "procrustes_scaled", "nm_procrustes", "ridge", "nm_ridge")},
               frac_removed={m: ev[m]["frac_removed"] for m in ("shift_only", "tokstd_only", "procrustes", "procrustes_scaled", "nm_procrustes", "ridge", "nm_ridge")},
               k_counts={int(kk): int(v) for kk, v in clips.sym_k.value_counts().sort_index().items()},
               violation_counts={kk: int(v) for kk, v in clips.violation_type.value_counts().sort_index().items()})
    return out


# ----------------------------------------------------------------------------- Part B: rescoring
def _fit_worker(args):
    """streaming sufficient statistics over the possible clips of a list of visible blocks (one visible condition)."""
    cond, vids_json, seed = args
    row_of, Z, P, H = _G["row_of"], _G["Z"], _G["P"], _G["H"]
    vids = json.loads(vids_json)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vids))
    st = {k: Stats() for k in ("p", "pn", "z", "p_other")}
    h_prev = rd(H, row_of[vids[order[-1]]], slice(HALF, None))  # h of a different clip (wrap-around of a random order)
    sum_pstd = 0.0; sum_hstd = 0.0; n_tok = 0
    for j in order:
        r = row_of[vids[j]]
        p = rd(P, r); z = rd(Z, r); h = rd(H, r, slice(HALF, None))
        sum_pstd += float(p.std(-1).sum()); sum_hstd += float(h.std(-1).sum()); n_tok += len(p)
        st["p"].add(p, h); st["z"].add(z, h); st["p_other"].add(p, h_prev); st["pn"].add(tokstd(p), h)
        h_prev = h
    return cond, st, sum_pstd, sum_hstd, n_tok, len(vids)


def _val_worker(args):
    cond, vids_json, seed = args
    row_of, Z, P, H = _G["row_of"], _G["Z"], _G["P"], _G["H"]
    vids = json.loads(vids_json); rng = np.random.default_rng(seed)
    out = {k: [] for k in ("p", "z", "h")}
    for vid in vids:
        r = row_of[vid]; idx = np.sort(rng.choice(HALF, N_TOK, replace=False))
        out["p"].append(rd(P, r, idx)); out["z"].append(rd(Z, r, idx)); out["h"].append(rd(H, r, HALF + idx))
    return cond, {k: np.concatenate(v) for k, v in out.items()}


def _score_worker(args):
    """one block: read p_a, p_b, z_a, z_b, h_a, h_b, h_ab, h_ba; surprise under every variant for pair A (pos_a, imp_ab) and pair B (pos_b, imp_ba)."""
    block = args  # dict(block_id, condition, violation_type, sym_k, vids{variant: video_id})
    row_of, Z, P, H = _G["row_of"], _G["Z"], _G["P"], _G["H"]
    M = _G["rs_maps"]; fam = FAM_OF[block["condition"]]; s_h = M["s_h"]
    out = dict(block_id=block["block_id"], condition=block["condition"], violation_type=block["violation_type"], sym_k=block["sym_k"], pairs={})
    for pair, (pos, imp) in (("A", ("pos_a", "imp_ab")), ("B", ("pos_b", "imp_ba"))):
        rp, ri = row_of[block["vids"][pos]], row_of[block["vids"][imp]]
        p = rd(P, rp); z = rd(Z, rp); hp = rd(H, rp, slice(HALF, None)); hi = rd(H, ri, slice(HALF, None))
        pn = tokstd(p, s_h)
        f = {"raw": p, "shift": apply_map(M["proc"], p, "shift"), "tokstd": pn, "gscale": p * M["gscale"],
             "proc": apply_map(M["proc"], p), "proc_scaled": apply_map(M["proc"], p, "scaled"), "nm_proc": apply_map(M["nm_proc"], pn),
             "ridge": apply_map(M["ridge"], p), "nm_ridge": apply_map(M["nm_ridge"], pn),
             "fam_proc": apply_map(M["fam"][fam]["proc"], p), "fam_ridge": apply_map(M["fam"][fam]["ridge"], p),
             "ctrl_ridge_other": apply_map(M["ridge_other"], p), "z_raw": z, "z_ridge": apply_map(M["ridge_z"], z)}
        out["pairs"][pair] = {k: (l1(v, hp), l1(v, hi)) for k, v in f.items()}
        out["pairs"][pair]["_vids"] = (block["vids"][pos], block["vids"][imp])
        if block["block_id"] % 25 == 0:  # p must be bit-identical across the matched pair (pixel-identical context); checked on every 25th block
            out["pairs"][pair]["_p_max_abs_diff_pos_imp"] = float(np.abs(p - rd(P, ri)).max())
    return out


VARIANTS = ["raw", "shift", "tokstd", "gscale", "proc", "proc_scaled", "nm_proc", "ridge", "nm_ridge", "fam_proc", "fam_ridge", "ctrl_ridge_other", "z_raw", "z_ridge"]


def boot_ci(vals, rng, n_boot, idx=None):
    vals = np.asarray(vals, float); n = len(vals)
    if idx is None:
        idx = rng.integers(0, n, (n_boot, n))
    bs = vals[idx].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], idx


def summarize_condition(rows, rng):
    """rows: list of per-block dicts (same condition). Returns per-variant acc / margin with block-bootstrap CIs and paired deltas vs raw."""
    n = len(rows)
    idx = rng.integers(0, n, (N_BOOT, n))
    res = dict(n_blocks=n, n_pairs=2 * n, n_boot=N_BOOT, variants={}, by_violation={}, by_k={}, by_direction={})
    per = {}
    for v in VARIANTS:
        hit = np.array([[(1.0 if r["pairs"][pr][v][1] > r["pairs"][pr][v][0] else (0.5 if r["pairs"][pr][v][1] == r["pairs"][pr][v][0] else 0.0)) for pr in "AB"] for r in rows])
        mar = np.array([[r["pairs"][pr][v][1] - r["pairs"][pr][v][0] for pr in "AB"] for r in rows])
        spos = np.array([[r["pairs"][pr][v][0] for pr in "AB"] for r in rows])
        per[v] = dict(hit=hit, mar=mar, spos=spos)
    for v in VARIANTS:
        hit, mar, spos = per[v]["hit"].mean(1), per[v]["mar"].mean(1), per[v]["spos"].mean(1)
        d = dict(acc=float(hit.mean()), acc_ci95=boot_ci(hit, rng, N_BOOT, idx)[0], margin=float(mar.mean()), margin_ci95=boot_ci(mar, rng, N_BOOT, idx)[0],
                 surprise_pos=float(spos.mean()), rel_margin=float(mar.mean() / spos.mean()), n_ties=int((per[v]["hit"] == 0.5).sum()),
                 acc_A=float(per[v]["hit"][:, 0].mean()), acc_B=float(per[v]["hit"][:, 1].mean()))
        if v != "raw":
            dh = hit - per["raw"]["hit"].mean(1); dm = mar - per["raw"]["mar"].mean(1)
            d["d_acc"] = float(dh.mean()); d["d_acc_ci95"] = boot_ci(dh, rng, N_BOOT, idx)[0]
            d["d_margin"] = float(dm.mean()); d["d_margin_ci95"] = boot_ci(dm, rng, N_BOOT, idx)[0]
        res["variants"][v] = d
    viol = np.array([r["violation_type"] for r in rows]); kk = np.array([r["sym_k"] for r in rows])
    for vt in sorted(set(viol)):
        m = viol == vt
        res["by_violation"][vt] = dict(n_blocks=int(m.sum()), acc={v: float(per[v]["hit"][m].mean()) for v in VARIANTS}, acc_A={v: float(per[v]["hit"][m, 0].mean()) for v in VARIANTS},
                                       acc_B={v: float(per[v]["hit"][m, 1].mean()) for v in VARIANTS}, margin={v: float(per[v]["mar"][m].mean()) for v in VARIANTS})
    for k in sorted(set(kk.tolist())):
        m = kk == k
        res["by_k"][str(k)] = dict(n_blocks=int(m.sum()), acc={v: float(per[v]["hit"][m].mean()) for v in VARIANTS}, margin={v: float(per[v]["mar"][m].mean()) for v in VARIANTS})
    return res


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INDEX)
    df["sym_k"] = df.sym_k.astype(int)
    row_of, Z, P, H = open_cache()
    _G.update(row_of=row_of, Z=Z, P=P, H=H)
    res = dict(config=dict(seed=SEED, n_clips=N_CLIPS, n_tok=N_TOK, n_perk=N_PERK, n_fit=N_FIT, n_val=N_VAL, n_test=N_TEST, n_boot=N_BOOT, n_boot_sub=N_BOOT_SUB, ks=list(KS),
                           lambda_grid=LAMS, sets={k: dict(condition=v[0], ks=v[1]) for k, v in SETS.items()}, rescoring_conditions=RS_CONDS, cache=str(CACHE), index=str(INDEX), reference_run=str(REF)))

    # ======================================================================= Part A
    rng = np.random.default_rng(SEED)
    data = {}
    for name, (cond, ks) in SETS.items():
        clips = sample_blocks(df, cond, ks, N_CLIPS, rng)
        data[name] = load_set(clips, row_of, Z, P, H, rng)
        log(f"A: loaded {name}: {data[name]['meta']}")
    res["sets"] = {n: d["meta"] | dict(condition=SETS[n][0], ks=SETS[n][1]) for n, d in data.items()}
    res["ln_check"] = {name: {r: dict(token_mean_abs=float(np.abs(flat(d[r]).mean(1)).mean()), token_std=float(flat(d[r]).std(1).mean()), token_l2=float(np.linalg.norm(flat(d[r]), axis=1).mean())) for r in REPS}
                       for name, d in data.items()}

    log("A(a) PCA subspaces")
    bases, spectra = {}, {}
    for name, d in data.items():
        for r in REPS:
            U, ev, pr = pca_basis(flat(d[r])); bases[(name, r)] = U
            spectra[f"{r}|{name}"] = dict(explained_var={str(k): v for k, v in ev.items()}, participation_ratio=pr, n_tokens=int(flat(d[r]).shape[0]))
    a = dict(spectra=spectra, random_baseline={}, split_half={}, within_set={}, across_sets={}, containment={}, mean_vectors={})
    rr = np.random.default_rng(SEED + 1)
    for k in KS:
        m, s = random_overlap(k, rr); a["random_baseline"][str(k)] = dict(mean=m, std=s, expectation=k / D)
    log("A(a) bases done")
    for name, d in data.items():
        blocks = rng.permutation(len(d["clips"])); h1, h2 = np.sort(blocks[: len(blocks) // 2]), np.sort(blocks[len(blocks) // 2:])
        for r in REPS:
            U1, _, _ = pca_basis(flat(d[r], h1)); U2, _, _ = pca_basis(flat(d[r], h2))
            a["split_half"][f"{r}|{name}"] = {str(k): overlap(U1, U2, k) for k in KS} | dict(n_tokens_half=int(len(h1) * N_TOK))
    log("A(a) split-half done")
    for name in data:
        for r1, r2 in (("p", "h"), ("z", "h"), ("z", "p")):
            a["within_set"][f"{r1}-vs-{r2}|{name}"] = {str(k): overlap(bases[(name, r1)], bases[(name, r2)], k) for k in KS}
    pairs = []
    for fam, (vis, late, early) in FAMILIES.items():
        pairs += [(vis, late), (vis, early), (late, early)]
    pairs += [("static_visible", "movflat_visible"), ("static_late", "movflat_late"), ("static_early", "movflat_early"),
              ("movflat_visible", "ramp_visible"), ("movflat_late", "ramp_late"), ("movflat_early", "ramp_early"), ("static_visible", "ramp_visible"), ("static_late", "ramp_late")]
    for s1, s2 in pairs:
        for r in REPS:
            a["across_sets"][f"{r}|{s1}-vs-{s2}"] = {str(k): overlap(bases[(s1, r)], bases[(s2, r)], k) for k in KS}
    for fam, (vis, late, early) in FAMILIES.items():
        for s2 in (late, early):
            a["across_sets"][f"p({vis})-vs-h({s2})"] = {str(k): overlap(bases[(vis, "p")], bases[(s2, "h")], k) for k in KS}
            a["across_sets"][f"p({s2})-vs-h({vis})"] = {str(k): overlap(bases[(s2, "p")], bases[(vis, "h")], k) for k in KS}
    log("A(a) within/across done")
    for name, d in data.items():
        big = {r: pca_basis(flat(d[r]), kmax=256)[0] for r in REPS}
        for ra, rb in (("p", "h"), ("h", "p"), ("z", "h"), ("p", "z")):
            for kb in (64, 256):
                a["containment"][f"{ra}top16-in-{rb}top{kb}|{name}"] = float(np.mean(np.linalg.norm(big[rb][:, :kb].T @ big[ra][:, :16], axis=0) ** 2))
        blocks = np.random.default_rng(SEED + 3).permutation(len(d["clips"])); h1, h2 = np.sort(blocks[: len(blocks) // 2]), np.sort(blocks[len(blocks) // 2:])
        for r in ("p", "h"):
            U1 = pca_basis(flat(d[r], h1), kmax=16)[0]; U2 = pca_basis(flat(d[r], h2), kmax=256)[0]
            for kb in (64, 256):
                a["containment"][f"SPLIT {r}top16-in-{r}top{kb}|{name}"] = float(np.mean(np.linalg.norm(U2[:, :kb].T @ U1, axis=0) ** 2))
        mp_, mh, mz = flat(d["p"]).mean(0), flat(d["h"]).mean(0), flat(d["z"]).mean(0)
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        a["mean_vectors"][name] = dict(cos_p_h=cos(mp_, mh), cos_z_h=cos(mz, mh), norm_p=float(np.linalg.norm(mp_)), norm_h=float(np.linalg.norm(mh)), l1_mean_gap=float(np.abs(mp_ - mh).mean()))
    res["a_subspace"] = a

    log("A(b) CKA")
    b = {}
    for name, d in data.items():
        p, h, ho = flat(d["p"]), flat(d["h"]), flat(d["h_other"])
        perm_tok = np.stack([rng.permutation(N_TOK) for _ in range(len(d["clips"]))])
        h_perm = np.take_along_axis(d["h"], perm_tok[:, :, None], axis=1).reshape(-1, D)
        b[name] = dict(n_tokens=int(len(p)), cka_matched=linear_cka(p, h), cka_other_clip_same_token=linear_cka(p, ho), cka_same_clip_perm_token=linear_cka(p, h_perm),
                       cka_h_vs_h_other=linear_cka(h, ho), cka_z_vs_h_sametokidx=linear_cka(flat(d["z"]), h))
    res["b_cka"] = b

    log("A(c) alignment maps (+ z->h control, norm-matched, absolute L1)")
    c = dict(fits={})
    splits = {name: split_blocks(d["clips"], np.random.default_rng(SEED + 2)) for name, d in data.items()}
    fam_maps = {}

    def fit_on(src_sets, src="p", dst="h"):
        Xtr = np.concatenate([flat(data[s][src], splits[s][0]) for s in src_sets]); Ytr = np.concatenate([flat(data[s][dst], splits[s][0]) for s in src_sets])
        Xva = np.concatenate([flat(data[s][src], splits[s][1]) for s in src_sets]); Yva = np.concatenate([flat(data[s][dst], splits[s][1]) for s in src_sets])
        s_h = float(Ytr.std(-1).mean())
        maps = fit_family_maps(Xtr, Ytr, Xva, Yva, s_h)
        info = dict(n_train_tokens=int(len(Xtr)), n_train_blocks=int(sum(len(splits[s][0]) for s in src_sets)), n_val_tokens=int(len(Xva)), ridge_lambda_rel=maps["ridge"]["lam"],
                    nm_ridge_lambda_rel=maps["nm_ridge"]["lam"], procrustes_scale=maps["proc"]["scale"], nm_procrustes_scale=maps["nm_proc"]["scale"], s_h=s_h)
        return maps, info

    def evals_for(maps, zmaps, targets):
        out = {}
        for label, sel in targets:
            s = label.split(" ")[0]
            e = eval_maps(maps, flat(data[s]["p"], sel), flat(data[s]["h"], sel))
            ez = eval_maps(zmaps, flat(data[s]["z"], sel), flat(data[s]["h"], sel))
            e["z_control"] = dict(l1_before=ez["l1_before"], ridge=ez["ridge"], procrustes=ez["procrustes"], procrustes_scaled=ez["procrustes_scaled"])
            e["n_blocks"] = int(len(sel) if sel is not None else len(data[s]["clips"]))
            out[label] = e
        return out

    for fam, (vis, late, early) in FAMILIES.items():
        maps, info = fit_on([vis]); zmaps, zinfo = fit_on([vis], src="z"); fam_maps[fam] = maps
        info["z_control"] = zinfo
        c["fits"][f"{fam}: p->h fit on {vis}"] = dict(info=info, evals=evals_for(maps, zmaps, [(f"{vis} (train)", splits[vis][0]), (f"{vis} (held-out test)", splits[vis][2]), (late, None), (early, None)]))
        maps_l, info_l = fit_on([late]); zmaps_l, zinfo_l = fit_on([late], src="z"); info_l["z_control"] = zinfo_l
        c["fits"][f"{fam}: p->h fit on {late}"] = dict(info=info_l, evals=evals_for(maps_l, zmaps_l, [(f"{late} (train)", splits[late][0]), (f"{late} (held-out test)", splits[late][2]), (f"{vis} (all)", None), (early, None)]))
        maps_e, info_e = fit_on([early]); zmaps_e, zinfo_e = fit_on([early], src="z"); info_e["z_control"] = zinfo_e
        c["fits"][f"{fam}: p->h fit on {early}"] = dict(info=info_e, evals=evals_for(maps_e, zmaps_e, [(f"{early} (held-out test)", splits[early][2]), (f"{vis} (all)", None), (late, None)]))
        maps_hh, info_hh = fit_on([vis], src="h")
        e_hh = {lab: eval_maps(maps_hh, flat(data[lab.split(' ')[0]]["h"], sel), flat(data[lab.split(' ')[0]]["h"], sel)) for lab, sel in [(f"{vis} (held-out test)", splits[vis][2]), (late, None)]}
        c["fits"][f"{fam}: SANITY h->h fit on {vis}"] = dict(info=info_hh, evals=e_hh)
    vis3 = [FAMILIES[f][0] for f in FAMILIES]
    maps, info = fit_on(vis3); zmaps, zinfo = fit_on(vis3, src="z"); info["z_control"] = zinfo
    c["fits"]["pooled: p->h fit on " + "+".join(vis3)] = dict(info=info, evals=evals_for(maps, zmaps, [(f"{v} (held-out test)", splits[v][2]) for v in vis3] + [(s, None) for f in FAMILIES for s in FAMILIES[f][1:]]))
    res["c_alignment"] = c
    del data

    # per-k late rows with bootstrap CIs (parallel over (set, k))
    log("A(per-k) late sets per k with block-bootstrap")
    _G["maps"] = fam_maps
    jobs = []
    for fam, (vis, late, early) in FAMILIES.items():
        for k in (1, 2, 3, 4):
            clips = sample_blocks(df, SETS[late][0], [k], N_PERK, rng, strata=("violation_type",))
            jobs.append((late, k, clips.to_json(), int(rng.integers(1 << 31))))
    with mp.get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
        perk = pool.map(_perk_worker, jobs)
    res["a_perk_late"] = perk
    log(f"A done ({time.time() - t0:.0f}s)")

    # ======================================================================= Part B: rescoring
    log("B: block split + sampling")
    rs = dict(fit_blocks={}, val_blocks={}, test_blocks={})
    blocks = df.drop_duplicates("block_id")[["block_id", "condition", "violation_type", "sym_k"]].reset_index(drop=True)
    rng_b = np.random.default_rng(SEED + 10)
    fit_vids, val_vids, test_blocks = {}, {}, []
    vids_of = {b: dict(zip(g.variant, g.video_id)) for b, g in df.groupby("block_id")}
    for cond in RS_CONDS:
        cb = blocks[blocks.condition == cond]
        if cond in [f[0] for f in RS_FAM.values()]:  # visible: 50/50 split by block, stratified by violation_type
            fit_half, test_half = [], []
            for _, g in cb.groupby("violation_type"):
                ids = rng_b.permutation(g.block_id.values); fit_half += list(ids[: len(ids) // 2]); test_half += list(ids[len(ids) // 2:])
            fit_df = cb[cb.block_id.isin(fit_half)]; test_df = cb[cb.block_id.isin(test_half)]
            fit_pick = stratified(fit_df, N_FIT + N_VAL, ["violation_type"], rng_b)
            tr = stratified(fit_pick, N_FIT, ["violation_type"], rng_b); va = fit_pick[~fit_pick.block_id.isin(tr.block_id)]
            fit_vids[cond] = [vids_of[b][v] for b in tr.block_id for v in ("pos_a", "pos_b")]
            val_vids[cond] = [vids_of[b][v] for b in va.block_id for v in ("pos_a", "pos_b")]
            te = stratified(test_df, N_TEST, ["violation_type"], rng_b)
            rs["fit_blocks"][cond] = [int(x) for x in tr.block_id]; rs["val_blocks"][cond] = [int(x) for x in va.block_id]
        else:
            te = stratified(cb, N_TEST, ["violation_type", "sym_k"], rng_b)
        rs["test_blocks"][cond] = [int(x) for x in te.block_id]
        for _, r in te.iterrows():
            test_blocks.append(dict(block_id=int(r.block_id), condition=cond, violation_type=r.violation_type, sym_k=int(r.sym_k), vids=vids_of[r.block_id]))
    assert not (set(sum(rs["fit_blocks"].values(), [])) | set(sum(rs["val_blocks"].values(), []))) & set(sum(rs["test_blocks"].values(), []))
    rs["n_test_blocks_total"] = len(test_blocks)

    log(f"B: fitting maps (streaming over {sum(len(v) for v in fit_vids.values())} possible clips × 2048 tokens)")
    with mp.get_context("fork").Pool(min(WORKERS, 3 * 4)) as pool:
        # split each visible condition's clip list into 4 chunks for parallel streaming
        jobs = []
        for cond, vids in fit_vids.items():
            for ch in np.array_split(np.arange(len(vids)), 4):
                jobs.append((cond, json.dumps([vids[i] for i in ch]), int(rng_b.integers(1 << 31))))
        parts = pool.map(_fit_worker, jobs)
        vals = dict(pool.map(_val_worker, [(cond, json.dumps(vids), int(rng_b.integers(1 << 31))) for cond, vids in val_vids.items()]))
    fam_st, tot = {}, {}
    fit_info = {}
    for cond, st, sp, sh, nt, nv in parts:
        f = FAM_OF[cond]
        if f not in fam_st:
            fam_st[f] = dict(st={k: Stats() for k in st}, sp=0.0, sh=0.0, nt=0, nv=0)
        for k in st:
            fam_st[f]["st"][k] += st[k]
        fam_st[f]["sp"] += sp; fam_st[f]["sh"] += sh; fam_st[f]["nt"] += nt; fam_st[f]["nv"] += nv
    pooled = dict(st={k: Stats() for k in ("p", "pn", "z", "p_other")}, sp=0.0, sh=0.0, nt=0, nv=0)
    for f, d in fam_st.items():
        for k in d["st"]:
            pooled["st"][k] += d["st"][k]
        pooled["sp"] += d["sp"]; pooled["sh"] += d["sh"]; pooled["nt"] += d["nt"]; pooled["nv"] += d["nv"]
    Xv = np.concatenate([vals[c]["p"] for c in vals]); Yv = np.concatenate([vals[c]["h"] for c in vals]); Zv = np.concatenate([vals[c]["z"] for c in vals])
    s_h = pooled["sh"] / pooled["nt"]; s_p = pooled["sp"] / pooled["nt"]
    M = dict(s_h=s_h, gscale=float(s_h / s_p), proc=fit_procrustes_stats(pooled["st"]["p"]), ridge=fit_ridge_stats(pooled["st"]["p"], Xv, Yv),
             nm_proc=fit_procrustes_stats(pooled["st"]["pn"]), nm_ridge=fit_ridge_stats(pooled["st"]["pn"], tokstd(Xv, s_h), Yv),
             ridge_other=fit_ridge_stats(pooled["st"]["p_other"], Xv, Yv), ridge_z=fit_ridge_stats(pooled["st"]["z"], Zv, Yv), fam={})
    fit_info["pooled"] = dict(n_train_clips=pooled["nv"], n_train_tokens=pooled["nt"], n_val_tokens=int(len(Xv)), s_h=s_h, s_p=s_p, gscale=M["gscale"],
                              procrustes_scale=M["proc"]["scale"], nm_procrustes_scale=M["nm_proc"]["scale"],
                              ridge_lambda_rel=M["ridge"]["lam"], ridge_val_l1=M["ridge"]["val_l1"], nm_ridge_lambda_rel=M["nm_ridge"]["lam"], nm_ridge_val_l1=M["nm_ridge"]["val_l1"],
                              ridge_other_lambda_rel=M["ridge_other"]["lam"], ridge_other_val_l1=M["ridge_other"]["val_l1"], ridge_z_lambda_rel=M["ridge_z"]["lam"], ridge_z_val_l1=M["ridge_z"]["val_l1"],
                              val_l1_raw=l1(Xv, Yv), val_l1_z_raw=l1(Zv, Yv))
    for f, d in fam_st.items():
        cv = RS_FAM[f][0]
        M["fam"][f] = dict(proc=fit_procrustes_stats(d["st"]["p"]), ridge=fit_ridge_stats(d["st"]["p"], vals[cv]["p"], vals[cv]["h"]))
        fit_info[f] = dict(n_train_clips=d["nv"], n_train_tokens=d["nt"], n_val_tokens=int(len(vals[cv]["p"])), procrustes_scale=M["fam"][f]["proc"]["scale"],
                           ridge_lambda_rel=M["fam"][f]["ridge"]["lam"], ridge_val_l1=M["fam"][f]["ridge"]["val_l1"], val_l1_raw=l1(vals[cv]["p"], vals[cv]["h"]))
    rs["fit_info"] = fit_info
    del vals, Xv, Yv, Zv
    _G["rs_maps"] = M
    log(f"B: maps fit ({time.time() - t0:.0f}s); scoring {len(test_blocks)} blocks")

    with mp.get_context("fork").Pool(WORKERS) as pool:
        scored = pool.map(_score_worker, test_blocks, chunksize=4)
    log(f"B: scored ({time.time() - t0:.0f}s)")

    # verification against the established run
    ref_pb = json.loads((REF / "per_block.json").read_text())
    ref_pv = ref_pb["per_video_surprise"]; ref_blk = {r["block_id"]: r for r in ref_pb["per_block"]}
    ref_sum = json.loads((REF / "summary.json").read_text())["surprise"]["by_block_type"]
    ver = {}
    for cond in RS_CONDS:
        rows = [r for r in scored if r["condition"] == cond]
        dpos, dimp, my_acc, ref_acc, ref_pair_acc, flips, ref_margins, mine, refv, pdiff = [], [], [], [], [], [], [], [], [], []
        for r in rows:
            for pr in "AB":
                vp, vi = r["pairs"][pr]["_vids"]; sp, si = r["pairs"][pr]["raw"]; rp, ri = ref_pv[vp], ref_pv[vi]
                dpos.append(abs(sp - rp)); dimp.append(abs(si - ri)); mine += [sp, si]; refv += [rp, ri]
                my_acc.append(1.0 if si > sp else (0.5 if si == sp else 0.0)); ref_pair_acc.append(1.0 if ri > rp else (0.5 if ri == rp else 0.0))
                ref_margins.append(ri - rp)
                if (si > sp) != (ri > rp):
                    flips.append(dict(block_id=r["block_id"], pair=pr, ref_margin=ri - rp, my_margin=si - sp))
                if "_p_max_abs_diff_pos_imp" in r["pairs"][pr]:
                    pdiff.append(r["pairs"][pr]["_p_max_abs_diff_pos_imp"])
            ref_acc.append(ref_blk[str(r["block_id"])]["acc"])
        ref_margins = np.array(ref_margins)
        ver[cond] = dict(n_blocks=len(rows), n_pairs=2 * len(rows), max_abs_diff_surprise=float(max(dpos + dimp)), mean_abs_diff_surprise=float(np.mean(dpos + dimp)),
                         corr_per_video_surprise=float(np.corrcoef(mine, refv)[0, 1]),
                         acc_recomputed_same_blocks=float(np.mean(my_acc)), acc_reference_same_blocks=float(np.mean(ref_acc)),
                         acc_reference_per_video_recomputed_with_my_pairing=float(np.mean(ref_pair_acc)),
                         n_flipped_pairs=len(flips), max_abs_ref_margin_among_flips=float(max([abs(f["ref_margin"]) for f in flips], default=0.0)),
                         frac_ref_pairs_abs_margin_below_1e3=float(np.mean(np.abs(ref_margins) < 1e-3)), frac_ref_pairs_abs_margin_below_5e4=float(np.mean(np.abs(ref_margins) < 5e-4)),
                         p_pos_vs_imp_max_abs_diff=(float(max(pdiff)) if pdiff else None), n_p_identity_checks=len(pdiff),
                         acc_reference_full_condition=float(ref_sum[cond]["block_pairwise"]), n_blocks_reference_full=int(ref_sum[cond]["n_block"]), flips=flips)
        v = ver[cond]
        v["acc_within_0p5pt"] = bool(abs(v["acc_recomputed_same_blocks"] - v["acc_reference_same_blocks"]) <= 0.005)
        # pairing is consistent if (i) the reference per-video values recombined with MY pairing reproduce the reference per-block accuracy exactly,
        # (ii) every sign flip sits inside the per-video precision band of the cache, (iii) p is bit-identical across the matched pair
        v["pairing_consistent"] = bool(abs(v["acc_reference_per_video_recomputed_with_my_pairing"] - v["acc_reference_same_blocks"]) < 1e-9
                                       and v["max_abs_ref_margin_among_flips"] <= 4 * v["max_abs_diff_surprise"] and (v["p_pos_vs_imp_max_abs_diff"] in (None, 0.0)))
    rs["verification"] = ver
    if not all(v["pairing_consistent"] for v in ver.values()):
        log("!! verification FAILED — pairing or token layout is wrong; see rs.verification")
    if not all(v["acc_within_0p5pt"] for v in ver.values()):
        log("!! same-block accuracy differs from the reference by > 0.5 pt in some condition (near-tie flips from cache precision; see rs.verification)")
    rs["variants"] = VARIANTS
    rs["conditions"] = {}
    rng_s = np.random.default_rng(SEED + 20)
    for cond in RS_CONDS:
        rs["conditions"][cond] = summarize_condition([r for r in scored if r["condition"] == cond], rng_s)
    # pooled families (visible / late / early across families) for a compact view
    rs["pooled"] = {}
    for lab, i in (("visible", 0), ("late", 1), ("early", 2)):
        rs["pooled"][lab] = summarize_condition([r for r in scored if r["condition"] in {RS_FAM[f][i] for f in RS_FAM}], rng_s)
    rs["per_block"] = [dict(block_id=r["block_id"], condition=r["condition"], violation_type=r["violation_type"], sym_k=r["sym_k"],
                            pairs={pr: {v: r["pairs"][pr][v] for v in VARIANTS} for pr in "AB"}) for r in scored]
    res["d_rescoring"] = rs
    res["runtime_s"] = float(time.time() - t0)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res)
    log(f"done in {res['runtime_s']:.0f}s → {OUT}")


# ----------------------------------------------------------------------------- report
MAPCOLS = ("shift_only", "tokstd_only", "procrustes", "procrustes_scaled", "nm_procrustes", "ridge", "nm_ridge")
MAPLAB = {"shift_only": "shift", "tokstd_only": "tok-std", "procrustes": "Procrustes", "procrustes_scaled": "Proc+scale", "nm_procrustes": "norm-matched Proc", "ridge": "ridge", "nm_ridge": "norm-matched ridge"}
VLAB = {"raw": "raw (scorer)", "shift": "shift-only", "tokstd": "per-token std rescale (LN p)", "gscale": "global scalar rescale", "proc": "Procrustes", "proc_scaled": "Procrustes+scale",
        "nm_proc": "norm-matched Procrustes", "ridge": "ridge", "nm_ridge": "norm-matched ridge", "fam_proc": "per-family Procrustes", "fam_ridge": "per-family ridge",
        "ctrl_ridge_other": "CONTROL ridge p→h(other clip)", "z_raw": "CONTROL |z − h|", "z_ridge": "CONTROL ridge z→h"}


def ci(c, scale=1.0, fmt="{:.1f}"):
    return "[" + fmt.format(c[0] * scale) + ", " + fmt.format(c[1] * scale) + "]"


def write_md(res):
    L = []; A = L.append
    cfg = res["config"]
    A("# E3-v2 — p→h alignment maps and matched-pair re-scoring (v11_full, ViT-H; static / flat / ramp × visible / late / early)")
    A("")
    A(f"Generated by `z_research/scripts/analysis/ctxenc_subspace_v2.py` (seed {cfg['seed']}); every number below is read from `results.json` in this folder. Extends `../ctxenc_subspace_overlap/RESULTS.md` (v1).")
    A("")
    rs = res["d_rescoring"]
    A("## 0. Headline — matched-pair scoring accuracy before / after a global linear p→h map (fit on visible POSSIBLE clips, held-out blocks)")
    A("")
    A(f"Rescoring sample: {N_TEST} blocks per condition (2 matched pairs each; visible blocks are the held-out half), maps fit on {rs['fit_info']['pooled']['n_train_clips']} possible clips × 2048 tokens of the three visible conditions ({N_FIT} blocks each; λ chosen on {N_VAL} further blocks). Block-bootstrap 95 % CI, {cfg['n_boot']} resamples. Δ = paired difference vs raw on the same blocks.")
    A("")
    key = ["raw", "tokstd", "proc", "nm_proc", "ridge", "nm_ridge", "fam_ridge", "ctrl_ridge_other", "z_ridge"]
    A("| condition | n_blk | " + " | ".join(f"acc {VLAB[v]}" for v in key) + " |")
    A("|---|---|" + "---|" * len(key))
    for cond, c in rs["conditions"].items():
        cells = []
        for v in key:
            d = c["variants"][v]
            cells.append(f"**{100*d['acc']:.1f}** {ci(d['acc_ci95'], 100)}" if v == "raw" else f"{100*d['acc']:.1f} (Δ{100*d['d_acc']:+.1f} {ci(d['d_acc_ci95'], 100)})")
        A(f"| {cond} | {c['n_blocks']} | " + " | ".join(cells) + " |")
    for lab, c in rs["pooled"].items():
        cells = []
        for v in key:
            d = c["variants"][v]
            cells.append(f"**{100*d['acc']:.1f}** {ci(d['acc_ci95'], 100)}" if v == "raw" else f"{100*d['acc']:.1f} (Δ{100*d['d_acc']:+.1f} {ci(d['d_acc_ci95'], 100)})")
        A(f"| POOLED {lab} | {c['n_blocks']} | " + " | ".join(cells) + " |")
    A("")
    A("Mean margin = surprise(imp) − surprise(pos) (×1000), same layout:")
    A("")
    A("| condition | surprise(pos) raw | " + " | ".join(f"margin {VLAB[v]}" for v in key) + " |")
    A("|---|---|" + "---|" * len(key))
    for cond, c in list(rs["conditions"].items()) + [("POOLED " + k, v) for k, v in rs["pooled"].items()]:
        cells = []
        for v in key:
            d = c["variants"][v]
            cells.append(f"**{1000*d['margin']:.2f}** {ci(d['margin_ci95'], 1000, '{:.2f}')}" if v == "raw" else f"{1000*d['margin']:.2f} (Δ{1000*d['d_margin']:+.2f} {ci(d['d_margin_ci95'], 1000, '{:.2f}')})")
        A(f"| {cond} | {c['variants']['raw']['surprise_pos']:.4f} | " + " | ".join(cells) + " |")
    A("")
    A("## 1. Verification of the raw (unmapped) scorer against the established run")
    A("")
    A(f"Reference: `{cfg['reference_run']}` (per_video_surprise + per_block.json; scorer ran dtype float32 + autocast float16). The token cache used here was extracted by the probing protocol (bfloat16), so per-video surprise recomputed from the cache differs from the scorer's by a small precision offset; matched pairs whose reference margin lies inside that band flip sign. Columns: per-video agreement, accuracy on the SAME blocks (mine vs reference), the reference per-video values recombined with MY pairing (must equal the reference per-block accuracy exactly — this is the pairing check), number of flipped pairs and the largest |reference margin| among them, the share of reference pairs with |margin| < 1e-3 / 5e-4 on these blocks, p bit-identity across the matched pair (max |p_pos − p_imp| on every 25th block), and the established full-condition number.")
    A("")
    A("| condition | n_blk | n_pair | max \\|Δ\\| per video | mean \\|Δ\\| | corr | acc mine | acc ref (same blocks) | ref per-video + my pairing | within 0.5 pt | flipped pairs | max \\|ref margin\\| of flips | ref \\|m\\|<1e-3 / <5e-4 (%) | max \\|p_pos−p_imp\\| (n chk) | pairing consistent | acc ref full (n) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond, v in rs["verification"].items():
        pd_ = "n/a" if v["p_pos_vs_imp_max_abs_diff"] is None else f"{v['p_pos_vs_imp_max_abs_diff']:.1e} ({v['n_p_identity_checks']})"
        A(f"| {cond} | {v['n_blocks']} | {v['n_pairs']} | {v['max_abs_diff_surprise']:.1e} | {v['mean_abs_diff_surprise']:.1e} | {v['corr_per_video_surprise']:.4f} | {100*v['acc_recomputed_same_blocks']:.2f} | {100*v['acc_reference_same_blocks']:.2f} | {100*v['acc_reference_per_video_recomputed_with_my_pairing']:.2f} | {'yes' if v['acc_within_0p5pt'] else '**no**'} | {v['n_flipped_pairs']} | {v['max_abs_ref_margin_among_flips']:.1e} | {100*v['frac_ref_pairs_abs_margin_below_1e3']:.1f} / {100*v['frac_ref_pairs_abs_margin_below_5e4']:.1f} | {pd_} | {'OK' if v['pairing_consistent'] else '**FAIL**'} | {100*v['acc_reference_full_condition']:.2f} ({v['n_blocks_reference_full']}) |")
    A("")
    A("## 2. Re-scoring — every variant, per condition (accuracy % with CI; ΔA/ΔB = pair A (pos_a vs imp_ab) / pair B (pos_b vs imp_ba) accuracy; ties)")
    A("")
    fi = rs["fit_info"]["pooled"]
    A(f"Pooled-visible fit: train clips {fi['n_train_clips']} (tokens {fi['n_train_tokens']}), val tokens {fi['n_val_tokens']}; s_h = {fi['s_h']:.4f}, s_p = {fi['s_p']:.4f}, global scalar = {fi['gscale']:.3f}; Procrustes scale {fi['procrustes_scale']:.3f} (norm-matched {fi['nm_procrustes_scale']:.3f}); ridge λ_rel {fi['ridge_lambda_rel']} (val L1 {fi['ridge_val_l1']:.4f} vs raw {fi['val_l1_raw']:.4f}); norm-matched ridge λ {fi['nm_ridge_lambda_rel']} (val L1 {fi['nm_ridge_val_l1']:.4f}); control ridge p→h_other λ {fi['ridge_other_lambda_rel']} (val L1 {fi['ridge_other_val_l1']:.4f}); ridge z→h λ {fi['ridge_z_lambda_rel']} (val L1 {fi['ridge_z_val_l1']:.4f} vs raw |z−h| {fi['val_l1_z_raw']:.4f}).")
    for f in ("static", "movflat", "ramp"):
        g = rs["fit_info"][f]
        A(f"Per-family {f} fit: train clips {g['n_train_clips']}, Procrustes scale {g['procrustes_scale']:.3f}, ridge λ_rel {g['ridge_lambda_rel']} (val L1 {g['ridge_val_l1']:.4f} vs raw {g['val_l1_raw']:.4f}).")
    A("")
    for cond, c in rs["conditions"].items():
        A(f"### {cond} (n_blocks {c['n_blocks']}, n_pairs {c['n_pairs']})")
        A("")
        A("| variant | acc % [CI] | Δacc vs raw [CI] | acc A / B | margin ×1000 [CI] | Δmargin ×1000 [CI] | surprise(pos) | rel. margin % | ties |")
        A("|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            d = c["variants"][v]
            da = "—" if v == "raw" else f"{100*d['d_acc']:+.1f} {ci(d['d_acc_ci95'], 100)}"
            dm = "—" if v == "raw" else f"{1000*d['d_margin']:+.2f} {ci(d['d_margin_ci95'], 1000, '{:.2f}')}"
            A(f"| {VLAB[v]} | {100*d['acc']:.1f} {ci(d['acc_ci95'], 100)} | {da} | {100*d['acc_A']:.1f} / {100*d['acc_B']:.1f} | {1000*d['margin']:.2f} {ci(d['margin_ci95'], 1000, '{:.2f}')} | {dm} | {d['surprise_pos']:.4f} | {100*d['rel_margin']:.2f} | {d['n_ties']} |")
        A("")
        A("By violation type (acc %, raw / tokstd / Procrustes / norm-matched Proc / ridge / norm-matched ridge / per-family ridge; A/B for raw and ridge):")
        A("")
        A("| violation | n_blk | raw | tokstd | proc | nm_proc | ridge | nm_ridge | fam_ridge | raw A/B | ridge A/B |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for vt, bv in c["by_violation"].items():
            A(f"| {vt} | {bv['n_blocks']} | " + " | ".join(f"{100*bv['acc'][v]:.1f}" for v in ("raw", "tokstd", "proc", "nm_proc", "ridge", "nm_ridge", "fam_ridge")) + f" | {100*bv['acc_A']['raw']:.1f} / {100*bv['acc_B']['raw']:.1f} | {100*bv['acc_A']['ridge']:.1f} / {100*bv['acc_B']['ridge']:.1f} |")
        if len(c["by_k"]) > 1:
            A("")
            A("By sym_k (acc %):")
            A("")
            A("| k | n_blk | raw | tokstd | proc | nm_proc | ridge | nm_ridge | fam_ridge |")
            A("|---|---|---|---|---|---|---|---|---|")
            for k, bk in c["by_k"].items():
                A(f"| {k} | {bk['n_blocks']} | " + " | ".join(f"{100*bk['acc'][v]:.1f}" for v in ("raw", "tokstd", "proc", "nm_proc", "ridge", "nm_ridge", "fam_ridge")) + " |")
        A("")
    # ---- Part A
    A("## 3. Samples for the subspace part (Part A; one obj clip per block)")
    A("")
    A("| set | index condition | sym_k | n_blocks | tokens/clip | n_tokens | k counts | violation counts |")
    A("|---|---|---|---|---|---|---|---|")
    for n, m in res["sets"].items():
        A(f"| {n} | {m['condition']} | {m['ks']} | {m['n_blocks']} | {m['n_tok_per_clip']} | {m['n_tokens']} | {m['k_counts']} | {m['violation_counts']} |")
    A("")
    A("LN sanity (per-token mean |mean| / std / L2):")
    A("")
    A("| set | z | p | h |")
    A("|---|---|---|---|")
    for n, m in res["ln_check"].items():
        A("| " + n + " | " + " | ".join(f"{m[r]['token_mean_abs']:.3f} / {m[r]['token_std']:.3f} / {m[r]['token_l2']:.1f}" for r in REPS) + " |")
    A("")
    a = res["a_subspace"]
    A("## 4. (a) Top-k PCA subspace overlap (mean cos² of principal angles)")
    A("")
    A("Random Gaussian baseline: " + ", ".join(f"k={k}: {v['mean']:.4f} ± {v['std']:.4f}" for k, v in a["random_baseline"].items()))
    A("")
    A("| rep | set | EV top-16 | EV top-64 | participation ratio | split-half k=16 | split-half k=64 |")
    A("|---|---|---|---|---|---|---|")
    for key, v in a["spectra"].items():
        r, s = key.split("|"); sh = a["split_half"][key]
        A(f"| {r} | {s} | {v['explained_var']['16']:.3f} | {v['explained_var']['64']:.3f} | {v['participation_ratio']:.1f} | {sh['16']:.3f} | {sh['64']:.3f} |")
    A("")
    A("Within a set:")
    A("")
    A("| set | p-vs-h k16 | p-vs-h k64 | z-vs-h k16 | z-vs-h k64 | z-vs-p k16 | z-vs-p k64 | CKA(p,h) matched | CKA other clip | CKA perm token | CKA(h,h_other) | CKA(z,h) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in res["sets"]:
        w = a["within_set"]; b = res["b_cka"][s]
        A(f"| {s} | {w[f'p-vs-h|{s}']['16']:.3f} | {w[f'p-vs-h|{s}']['64']:.3f} | {w[f'z-vs-h|{s}']['16']:.3f} | {w[f'z-vs-h|{s}']['64']:.3f} | {w[f'z-vs-p|{s}']['16']:.3f} | {w[f'z-vs-p|{s}']['64']:.3f} | {b['cka_matched']:.3f} | {b['cka_other_clip_same_token']:.3f} | {b['cka_same_clip_perm_token']:.3f} | {b['cka_h_vs_h_other']:.3f} | {b['cka_z_vs_h_sametokidx']:.3f} |")
    A("")
    A("Across sets (same rep) and p(set1) vs h(set2):")
    A("")
    A("| rep | sets | k=16 | k=64 |")
    A("|---|---|---|---|")
    for key, v in a["across_sets"].items():
        r, s = key.split("|") if "|" in key else (key, "")
        A(f"| {r} | {s} | {v['16']:.3f} | {v['64']:.3f} |")
    A("")
    A("Asymmetric containment (A's top-16 inside B's top-kB; SPLIT = same rep, disjoint halves):")
    A("")
    A("| A top-16 in B | set | kB=64 | kB=256 |")
    A("|---|---|---|---|")
    seen = {}
    for key, v in a["containment"].items():
        lab, s = key.split("|"); base = lab.rsplit("top", 1)[0]; kb = lab.rsplit("top", 1)[1]
        seen.setdefault((base, s), {})[kb] = v
    for (base, s), v in seen.items():
        A(f"| {base} | {s} | {v['64']:.3f} | {v['256']:.3f} |")
    A("")
    A("Mean vectors: cos(μp, μh), cos(μz, μh), ‖μp‖, ‖μh‖, mean|μp−μh|:")
    A("")
    A("| set | cos(μp, μh) | cos(μz, μh) | ‖μp‖ | ‖μh‖ | mean\\|μp−μh\\| |")
    A("|---|---|---|---|---|---|")
    for s, v in a["mean_vectors"].items():
        A(f"| {s} | {v['cos_p_h']:.3f} | {v['cos_z_h']:.3f} | {v['norm_p']:.2f} | {v['norm_h']:.2f} | {v['l1_mean_gap']:.3f} |")
    A("")
    A(f"## 5. Per-k rows for the late sets ({N_PERK} blocks per k; block-bootstrap 95 % CI, {cfg['n_boot_sub']} resamples; L1 after = the family's visible-fit map of §6 applied)")
    A("")
    A("| set | k | n_blk | p-vs-h k16 [CI] | p-vs-h k64 [CI] | z-vs-h k16 [CI] | z-vs-h k64 [CI] | CKA(p,h) | CKA(z,h) | L1 before | " + " | ".join(f"L1 {MAPLAB[m]} (%)" for m in ("procrustes", "nm_procrustes", "ridge", "nm_ridge")) + " |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["a_perk_late"]:
        o, c_ = r["overlap"], r["overlap_ci95"]
        A(f"| {r['set']} | {r['k']} | {r['n_blocks']} | {o['p-vs-h|16']:.3f} {ci(c_['p-vs-h|16'], 1, '{:.3f}')} | {o['p-vs-h|64']:.3f} {ci(c_['p-vs-h|64'], 1, '{:.3f}')} | {o['z-vs-h|16']:.3f} {ci(c_['z-vs-h|16'], 1, '{:.3f}')} | {o['z-vs-h|64']:.3f} {ci(c_['z-vs-h|64'], 1, '{:.3f}')} | {r['cka_p_h']:.3f} | {r['cka_z_h']:.3f} | {r['l1_before']:.4f} | "
          + " | ".join(f"{r['l1_after'][m]:.4f} ({100*r['frac_removed'][m]:.1f})" for m in ("procrustes", "nm_procrustes", "ridge", "nm_ridge")) + " |")
    A("")
    A("## 6. (c) Alignment maps p→h on the subspace samples: absolute L1 after each map (% of the p−h gap removed) and the z→h control on the same rows")
    A("")
    A("Block-disjoint train / val / test = 60 / 20 / 20 % of the source set. Norm-matched = p rescaled to h's per-token std (s_h) before fitting and applying. z→h control: same map family fit from the context token at the same index.")
    A("")
    for fit, v in res["c_alignment"]["fits"].items():
        i = v["info"]
        A(f"### {fit}")
        A("")
        line = f"train tokens {i['n_train_tokens']} (blocks {i['n_train_blocks']}), val tokens {i['n_val_tokens']}, ridge λ_rel {i['ridge_lambda_rel']}, Procrustes scale {i['procrustes_scale']:.3f}, s_h {i['s_h']:.3f}"
        if "z_control" in i:
            line += f"; z→h: ridge λ_rel {i['z_control']['ridge_lambda_rel']}, Procrustes scale {i['z_control']['procrustes_scale']:.3f}"
        A(line)
        A("")
        if "SANITY" in fit:
            A("| eval set | n_tokens | L1 before | " + " | ".join(f"{MAPLAB[m]} L1 (%)" for m in MAPCOLS) + " |")
            A("|---|---|---|" + "---|" * len(MAPCOLS))
            for s, e in v["evals"].items():
                A(f"| {s} | {e['n_tokens']} | {e['l1_before']:.4f} | " + " | ".join(f"{e[m]['l1']:.4f} ({100*e[m]['frac_removed']:.1f})" for m in MAPCOLS) + " |")
        else:
            A("| eval set | n_blk | n_tokens | p−h L1 before | " + " | ".join(f"{MAPLAB[m]} L1 (%)" for m in MAPCOLS) + " | z−h L1 before | z→h Proc L1 (%) | z→h Proc+scale L1 (%) | z→h ridge L1 (%) |")
            A("|---|---|---|---|" + "---|" * (len(MAPCOLS) + 4))
            for s, e in v["evals"].items():
                zc = e["z_control"]
                A(f"| {s} | {e['n_blocks']} | {e['n_tokens']} | {e['l1_before']:.4f} | " + " | ".join(f"{e[m]['l1']:.4f} ({100*e[m]['frac_removed']:.1f})" for m in MAPCOLS)
                  + f" | {zc['l1_before']:.4f} | {zc['procrustes']['l1']:.4f} ({100*zc['procrustes']['frac_removed']:.1f}) | {zc['procrustes_scaled']['l1']:.4f} ({100*zc['procrustes_scaled']['frac_removed']:.1f}) | {zc['ridge']['l1']:.4f} ({100*zc['ridge']['frac_removed']:.1f}) |")
        A("")
    A(f"runtime {res['runtime_s']:.0f} s")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
