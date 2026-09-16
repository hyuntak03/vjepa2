#!/usr/bin/env python3
"""E3 — subspace geometry of z / p / h by condition (IntPhysGen v11_full, ViT-H cache): do p and h share a subspace, and does occlusion rotate the encoder's subspace?

CLAUDE.md §10 item 1 (부분공간 겹침 / Procrustes). All numbers are computed from the token cache only (no model, no GPU), deterministic (seed 0, closed-form).

Data  : /data_csv/intphysgen_v11_full/index_probe.csv, probe_type == 'obj' (possible clips with an object), one obj clip per block.
        6 condition sets: static_visible | static_occlusion (late, k=1..4 pooled, 30 blocks per k) | static_occlusion_early (k=4)
                          moving_visible_flat | moving_occlusion_flat (late, k=1..4 pooled) | moving_occlusion_flat_early (k=4)
        120 blocks per set (stratified by violation_type × sym_k), 256 random token indices per clip (of the 2048 half-clip tokens)
        → 30,720 tokens × 1280 per (representation, set).
Reps  : z = ctx_masked.npy (context tubelets 0–7, already affine-free LN'd at extraction)
        p = predictor.npy  (future slots 0–7, raw predictor output — lives in LN space by training objective)
        h = target.npy[2048:] (future tubelets 8–15, LN'd at extraction)
        p and h are token-matched (same clip, same token index); z uses the same index set on the context half.
        h_other = h of a DIFFERENT clip (same set) at the SAME token indices (baseline for CKA).
Measures (all deterministic):
  (a) top-k PCA subspaces (k = 16, 64); overlap = mean cos² of principal angles. Baselines: split-half of the same (rep, set) by block (upper bound),
      two random Gaussian k-subspaces in R^1280 (lower bound, expectation k/1280). Also explained-variance fraction of top-k and participation ratio.
  (b) linear CKA (feature-space form, exact) between token-matched p and h; baselines: different clip / same token index, same clip / permuted token.
  (c) orthogonal Procrustes W (1280×1280, after centering; also a scaled variant) and ridge (λ chosen on a block-disjoint validation fold) fit p→h on
      visible-set train blocks; report the fraction of mean|p−h| removed on visible held-out blocks and on late / early (transfer). Controls: mean-shift only,
      h→h sanity (must be ~100 %), map fit on late itself (late train → late test = linear-alignability ceiling for late), pooled-visible fit.
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_subspace_overlap/{results.json, RESULTS.md}
Run (CPU only, ~5–10 min; N_CLIPS / N_TOK env vars override 120 / 256):
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2 && /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_subspace_overlap.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
OUT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_subspace_overlap"
SEED = 0
N_CLIPS = int(os.environ.get("N_CLIPS", 120))
N_TOK = int(os.environ.get("N_TOK", 256))
KS = (16, 64)
D = 1280
HALF = 2048

# name -> (index condition, allowed sym_k)
SETS = {
    "static_visible": ("static_visible", [0]),
    "static_late": ("static_occlusion", [1, 2, 3, 4]),
    "static_early": ("static_occlusion_early", [4]),
    "movflat_visible": ("moving_visible_flat", [0]),
    "movflat_late": ("moving_occlusion_flat", [1, 2, 3, 4]),
    "movflat_early": ("moving_occlusion_flat_early", [4]),
}
FAMILIES = {"static": ("static_visible", "static_late", "static_early"),
            "movflat": ("movflat_visible", "movflat_late", "movflat_early")}
REPS = ("z", "p", "h")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- sampling
def sample_blocks(df, cond, ks, rng):
    """One obj clip per block; N_CLIPS blocks stratified by (violation_type, sym_k)."""
    sub = df[(df.probe_type == "obj") & (df.condition == cond) & (df.sym_k.isin(ks))]
    # one obj clip per block (random)
    sub = sub.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).drop_duplicates("block_id")
    strata = sub.groupby(["violation_type", "sym_k"])
    per = N_CLIPS // strata.ngroups
    picks = [g.sample(n=min(per, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in strata]
    out = pd.concat(picks)
    if len(out) < N_CLIPS:  # top up
        rest = sub.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(N_CLIPS - len(out), len(rest)), random_state=int(rng.integers(1 << 31)))])
    return out.sort_values("block_id").reset_index(drop=True)


def load_set(name, df, row_of, Z, P, H, rng):
    cond, ks = SETS[name]
    clips = sample_blocks(df, cond, ks, rng)
    n = len(clips)
    tok = np.stack([np.sort(rng.choice(HALF, N_TOK, replace=False)) for _ in range(n)])  # (n, N_TOK)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):  # derangement for the different-clip baseline
        perm = rng.permutation(n)
    z = np.empty((n, N_TOK, D), np.float32); p = np.empty_like(z); h = np.empty_like(z); h_other = np.empty_like(z)
    for i, vid in enumerate(clips.video_id):
        r = row_of[vid]; idx = tok[i]
        z[i] = Z[r][idx]; p[i] = P[r][idx]; h[i] = H[r][HALF + idx]
        h_other[i] = H[row_of[clips.video_id.iloc[perm[i]]]][HALF + idx]
    meta = dict(n_clips=n, n_tok_per_clip=N_TOK, n_tokens=n * N_TOK, condition=cond, ks=ks,
                k_counts={int(k): int(v) for k, v in clips.sym_k.value_counts().sort_index().items()},
                violation_counts={k: int(v) for k, v in clips.violation_type.value_counts().sort_index().items()},
                n_blocks=int(clips.block_id.nunique()))
    return dict(z=z, p=p, h=h, h_other=h_other, clips=clips, tok=tok, meta=meta)


# ----------------------------------------------------------------------------- (a) subspaces
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
    s = np.clip(s, 0, 1)
    return float(np.mean(s ** 2))


def random_overlap(k, rng, n_draw=10):
    vals = []
    for _ in range(n_draw):
        A = np.linalg.qr(rng.standard_normal((D, k)))[0]; B = np.linalg.qr(rng.standard_normal((D, k)))[0]
        vals.append(overlap(A, B, k))
    return float(np.mean(vals)), float(np.std(vals))


# ----------------------------------------------------------------------------- (b) CKA
def linear_cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Xc.T @ Yc) ** 2
    return float(hsic / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)))


# ----------------------------------------------------------------------------- (c) maps
def fit_procrustes(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    W = U @ Vt
    scale = float(S.sum() / (Xc ** 2).sum())
    return dict(kind="procrustes", W=W, mx=mx, my=my, scale=scale)


def fit_ridge(X, Y, Xv, Yv):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc; B = Xc.T @ Yc
    tr = np.trace(G) / D
    best = None
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        W = np.linalg.solve(G + lam * tr * np.eye(D), B)
        l1 = np.abs((Xv - mx) @ W + my - Yv).mean()
        if best is None or l1 < best[0]:
            best = (l1, lam, W)
    return dict(kind="ridge", W=best[2], mx=mx, my=my, lam=best[1], val_l1=float(best[0]))


def apply_map(m, X, variant="plain"):
    if variant == "shift":
        return X - m["mx"] + m["my"]
    if variant == "scaled":
        return m["scale"] * (X - m["mx"]) @ m["W"] + m["my"]
    return (X - m["mx"]) @ m["W"] + m["my"]


def l1(A, B):
    return float(np.abs(A - B).mean())


def eval_maps(maps, X, Y):
    base = l1(X, Y)
    out = dict(l1_before=base, n_tokens=int(len(X)))
    out["shift_only"] = dict(l1=l1(apply_map(maps["proc"], X, "shift"), Y))
    out["procrustes"] = dict(l1=l1(apply_map(maps["proc"], X), Y))
    out["procrustes_scaled"] = dict(l1=l1(apply_map(maps["proc"], X, "scaled"), Y))
    out["ridge"] = dict(l1=l1(apply_map(maps["ridge"], X), Y))
    ynorm = float(np.abs(Y).mean())
    out["mean_abs_target"] = ynorm
    for k in ("shift_only", "procrustes", "procrustes_scaled", "ridge"):
        # fraction of the p-h gap removed; when X == Y (h->h sanity) the gap is 0 and we report 1 - residual/|Y| instead
        out[k]["frac_removed"] = float(1 - out[k]["l1"] / base) if base > 0 else float(1 - out[k]["l1"] / ynorm)
        out[k]["rel_residual"] = float(out[k]["l1"] / ynorm)
    return out


def split_blocks(clips, rng, fr=(0.6, 0.2, 0.2)):
    """block-disjoint train/val/test, stratified by violation_type × sym_k."""
    tr, va, te = [], [], []
    for _, g in clips.groupby(["violation_type", "sym_k"]):
        ids = rng.permutation(g.index.values)
        n = len(ids); a = int(round(fr[0] * n)); b = int(round((fr[0] + fr[1]) * n))
        tr += list(ids[:a]); va += list(ids[a:b]); te += list(ids[b:])
    return np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int), np.array(sorted(te), dtype=int)


def flat(a, sel=None):
    a = a if sel is None else a[sel]
    return a.reshape(-1, D)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INDEX)
    row_of = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r")
    assert Z.shape[1] == HALF and P.shape[1] == HALF and H.shape[1] == 2 * HALF and Z.shape[2] == D

    rng = np.random.default_rng(SEED)
    data = {}
    for name in SETS:
        data[name] = load_set(name, df, row_of, Z, P, H, rng)
        log(f"loaded {name}: {data[name]['meta']}")

    res = dict(config=dict(seed=SEED, n_clips=N_CLIPS, n_tok=N_TOK, ks=list(KS), sets={k: dict(condition=v[0], ks=v[1]) for k, v in SETS.items()},
                           cache=str(CACHE), index=str(INDEX)),
               sets={n: d["meta"] for n, d in data.items()})

    # LN sanity: per-token mean ~0 / std ~1 for z and h (LN'd at extraction); p is raw predictor output
    ln = {}
    for name, d in data.items():
        ln[name] = {r: dict(token_mean_abs=float(np.abs(flat(d[r]).mean(1)).mean()), token_std=float(flat(d[r]).std(1).mean()),
                            token_l2=float(np.linalg.norm(flat(d[r]), axis=1).mean())) for r in REPS}
    res["ln_check"] = ln

    # ---------------- (a)
    log("(a) PCA subspaces")
    bases, spectra = {}, {}
    for name, d in data.items():
        for r in REPS:
            U, ev, pr = pca_basis(flat(d[r]))
            bases[(name, r)] = U; spectra[f"{r}|{name}"] = dict(explained_var={str(k): v for k, v in ev.items()}, participation_ratio=pr, n_tokens=int(flat(d[r]).shape[0]))
    a = dict(spectra=spectra, random_baseline={}, split_half={}, within_set={}, across_sets={})
    rr = np.random.default_rng(SEED + 1)
    for k in KS:
        m, s = random_overlap(k, rr); a["random_baseline"][str(k)] = dict(mean=m, std=s, expectation=k / D)
    # split-half by block (upper bound)
    for name, d in data.items():
        blocks = rng.permutation(len(d["clips"]))
        h1, h2 = np.sort(blocks[: len(blocks) // 2]), np.sort(blocks[len(blocks) // 2:])
        for r in REPS:
            U1, _, _ = pca_basis(flat(d[r], h1)); U2, _, _ = pca_basis(flat(d[r], h2))
            a["split_half"][f"{r}|{name}"] = {str(k): overlap(U1, U2, k) for k in KS} | dict(n_tokens_half=int(len(h1) * N_TOK))
    # within a set: p vs h, z vs h, z vs p
    for name in data:
        for r1, r2 in (("p", "h"), ("z", "h"), ("z", "p")):
            a["within_set"][f"{r1}-vs-{r2}|{name}"] = {str(k): overlap(bases[(name, r1)], bases[(name, r2)], k) for k in KS}
    # across sets, same rep
    pairs = []
    for fam, (vis, late, early) in FAMILIES.items():
        pairs += [(vis, late), (vis, early), (late, early)]
    pairs += [("static_visible", "movflat_visible"), ("static_late", "movflat_late"), ("static_early", "movflat_early")]
    for s1, s2 in pairs:
        for r in REPS:
            a["across_sets"][f"{r}|{s1}-vs-{s2}"] = {str(k): overlap(bases[(s1, r)], bases[(s2, r)], k) for k in KS}
    # extra: p(set) vs h(other set) — is the p↔h gap bigger than the condition gap?
    for fam, (vis, late, early) in FAMILIES.items():
        for s2 in (late, early):
            a["across_sets"][f"p({vis})-vs-h({s2})"] = {str(k): overlap(bases[(vis, "p")], bases[(s2, "h")], k) for k in KS}
            a["across_sets"][f"p({s2})-vs-h({vis})"] = {str(k): overlap(bases[(s2, "p")], bases[(vis, "h")], k) for k in KS}
    # asymmetric containment: mean squared projection of A's top-kA directions onto B's top-kB subspace
    # (1.0 = A's top-kA lies inside B's top-kB; distinguishes "p is a low-dim subset of h's space" from "p is rotated out of it")
    a["containment"] = {}
    for name, d in data.items():
        big = {r: pca_basis(flat(d[r]), kmax=256)[0] for r in REPS}
        for ra, rb in (("p", "h"), ("h", "p"), ("z", "h"), ("p", "z")):
            for kb in (64, 256):
                a["containment"][f"{ra}top16-in-{rb}top{kb}|{name}"] = float(np.mean(np.linalg.norm(big[rb][:, :kb].T @ big[ra][:, :16], axis=0) ** 2))
        # split-half reference for the same asymmetric quantity (top16 of half A inside top-kB of half B, same rep)
        blocks = np.random.default_rng(SEED + 3).permutation(len(d["clips"]))
        h1, h2 = np.sort(blocks[: len(blocks) // 2]), np.sort(blocks[len(blocks) // 2:])
        for r in ("p", "h"):
            U1 = pca_basis(flat(d[r], h1), kmax=16)[0]; U2 = pca_basis(flat(d[r], h2), kmax=256)[0]
            for kb in (64, 256):
                a["containment"][f"SPLIT {r}top16-in-{r}top{kb}|{name}"] = float(np.mean(np.linalg.norm(U2[:, :kb].T @ U1, axis=0) ** 2))
    # mean-vector geometry (PCA removes the mean; report it separately)
    a["mean_vectors"] = {}
    for name, d in data.items():
        mp, mh, mz = flat(d["p"]).mean(0), flat(d["h"]).mean(0), flat(d["z"]).mean(0)
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        a["mean_vectors"][name] = dict(cos_p_h=cos(mp, mh), cos_z_h=cos(mz, mh), norm_p=float(np.linalg.norm(mp)), norm_h=float(np.linalg.norm(mh)),
                                       l1_mean_gap=float(np.abs(mp - mh).mean()))
    res["a_subspace"] = a

    # ---------------- (b)
    log("(b) CKA")
    b = {}
    for name, d in data.items():
        p, h, ho = flat(d["p"]), flat(d["h"]), flat(d["h_other"])
        n_clip = len(d["clips"])
        # same clip, permuted token: permute tokens within each clip
        perm_tok = np.stack([rng.permutation(N_TOK) for _ in range(n_clip)])
        h_perm = np.take_along_axis(d["h"], perm_tok[:, :, None], axis=1).reshape(-1, D)
        b[name] = dict(n_tokens=int(len(p)), cka_matched=linear_cka(p, h), cka_other_clip_same_token=linear_cka(p, ho),
                       cka_same_clip_perm_token=linear_cka(p, h_perm), cka_h_vs_h_other=linear_cka(h, ho),
                       cka_z_vs_h_sametokidx=linear_cka(flat(d["z"]), h))
    res["b_cka"] = b

    # ---------------- (c)
    log("(c) Procrustes / ridge")
    c = dict(fits={})
    splits = {name: split_blocks(d["clips"], np.random.default_rng(SEED + 2)) for name, d in data.items()}

    def fit_on(src_sets, src="p", dst="h"):
        Xtr = np.concatenate([flat(data[s][src], splits[s][0]) for s in src_sets]); Ytr = np.concatenate([flat(data[s][dst], splits[s][0]) for s in src_sets])
        Xva = np.concatenate([flat(data[s][src], splits[s][1]) for s in src_sets]); Yva = np.concatenate([flat(data[s][dst], splits[s][1]) for s in src_sets])
        maps = dict(proc=fit_procrustes(Xtr, Ytr), ridge=fit_ridge(Xtr, Ytr, Xva, Yva))
        info = dict(n_train_tokens=int(len(Xtr)), n_train_blocks=int(sum(len(splits[s][0]) for s in src_sets)), n_val_tokens=int(len(Xva)),
                    ridge_lambda_rel=maps["ridge"]["lam"], ridge_val_l1=maps["ridge"]["val_l1"], procrustes_scale=maps["proc"]["scale"])
        return maps, info

    def evals_for(maps, targets, src="p", dst="h"):
        out = {}
        for label, sel in targets:
            s = label.split(" ")[0]
            out[label] = eval_maps(maps, flat(data[s][src], sel), flat(data[s][dst], sel)); out[label]["n_blocks"] = int(len(sel) if sel is not None else len(data[s]["clips"]))
        return out

    for fam, (vis, late, early) in FAMILIES.items():
        # p->h fit on visible
        maps, info = fit_on([vis])
        c["fits"][f"{fam}: p->h fit on {vis}"] = dict(info=info, evals=evals_for(maps, [(f"{vis} (train)", splits[vis][0]), (f"{vis} (held-out test)", splits[vis][2]), (late, None), (early, None)]))
        # late-fit ceiling
        maps_l, info_l = fit_on([late])
        c["fits"][f"{fam}: p->h fit on {late}"] = dict(info=info_l, evals=evals_for(maps_l, [(f"{late} (train)", splits[late][0]), (f"{late} (held-out test)", splits[late][2]), (f"{vis} (all)", None), (early, None)]))
        maps_e, info_e = fit_on([early])
        c["fits"][f"{fam}: p->h fit on {early}"] = dict(info=info_e, evals=evals_for(maps_e, [(f"{early} (held-out test)", splits[early][2]), (f"{vis} (all)", None), (late, None)]))
        # sanity h->h and p->p
        maps_hh, info_hh = fit_on([vis], src="h", dst="h")
        c["fits"][f"{fam}: SANITY h->h fit on {vis}"] = dict(info=info_hh, evals=evals_for(maps_hh, [(f"{vis} (held-out test)", splits[vis][2]), (late, None)], src="h", dst="h"))
        # z->h (context encoder token at same index -> future target token): a control for "what a map can do without prediction"
        maps_zh, info_zh = fit_on([vis], src="z", dst="h")
        c["fits"][f"{fam}: CONTROL z->h fit on {vis}"] = dict(info=info_zh, evals=evals_for(maps_zh, [(f"{vis} (held-out test)", splits[vis][2]), (late, None), (early, None)], src="z", dst="h"))
    # pooled visible
    maps, info = fit_on(["static_visible", "movflat_visible"])
    c["fits"]["pooled: p->h fit on static_visible+movflat_visible"] = dict(info=info, evals=evals_for(maps, [("static_visible (held-out test)", splits["static_visible"][2]), ("movflat_visible (held-out test)", splits["movflat_visible"][2]),
                                                                                                             ("static_late", None), ("static_early", None), ("movflat_late", None), ("movflat_early", None)]))
    res["c_alignment"] = c
    res["runtime_s"] = float(time.time() - t0)

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_md(res)
    log(f"done in {res['runtime_s']:.0f}s → {OUT}")


# ----------------------------------------------------------------------------- report
def write_md(res):
    L = []
    A = L.append
    A("# E3 — subspace overlap / CKA / Procrustes of z, p, h by condition (v11_full, ViT-H)")
    A("")
    A(f"Generated by `z_research/scripts/analysis/ctxenc_subspace_overlap.py` (seed {res['config']['seed']}); every number below is read from `results.json` in this folder.")
    A("")
    A("## Samples")
    A("")
    A("| set | index condition | sym_k | n_blocks (=clips) | tokens/clip | n_tokens | k counts | violation counts |")
    A("|---|---|---|---|---|---|---|---|")
    for n, m in res["sets"].items():
        A(f"| {n} | {m['condition']} | {m['ks']} | {m['n_blocks']} | {m['n_tok_per_clip']} | {m['n_tokens']} | {m['k_counts']} | {m['violation_counts']} |")
    A("")
    A("LN sanity (per-token mean |mean| / std / L2; z, h are affine-free LN'd at extraction, p is raw predictor output):")
    A("")
    A("| set | z mean\\|μ\\| / σ / L2 | p | h |")
    A("|---|---|---|---|")
    for n, m in res["ln_check"].items():
        A("| " + n + " | " + " | ".join(f"{m[r]['token_mean_abs']:.3f} / {m[r]['token_std']:.3f} / {m[r]['token_l2']:.1f}" for r in REPS) + " |")
    A("")
    a = res["a_subspace"]
    A("## (a) Top-k PCA subspace overlap (mean cos² of principal angles; 1.0 = identical)")
    A("")
    rb = a["random_baseline"]
    A("Random Gaussian baseline (lower bound): " + ", ".join(f"k={k}: {v['mean']:.4f} ± {v['std']:.4f} (expectation {v['expectation']:.4f})" for k, v in rb.items()))
    A("")
    A("Explained-variance fraction of top-k and participation ratio (effective dim):")
    A("")
    A("| rep | set | EV top-16 | EV top-64 | participation ratio | n_tokens |")
    A("|---|---|---|---|---|---|")
    for key, v in a["spectra"].items():
        r, s = key.split("|")
        A(f"| {r} | {s} | {v['explained_var']['16']:.3f} | {v['explained_var']['64']:.3f} | {v['participation_ratio']:.1f} | {v['n_tokens']} |")
    A("")
    A("Split-half baseline (same rep, same set, disjoint block halves — upper bound for what 'same subspace' looks like at this sample size):")
    A("")
    A("| rep | set | k=16 | k=64 | n_tokens per half |")
    A("|---|---|---|---|---|")
    for key, v in a["split_half"].items():
        r, s = key.split("|")
        A(f"| {r} | {s} | {v['16']:.3f} | {v['64']:.3f} | {v['n_tokens_half']} |")
    A("")
    A("Within a set — p vs h (token-matched), z vs h, z vs p:")
    A("")
    A("| pair | set | k=16 | k=64 |")
    A("|---|---|---|---|")
    for key, v in a["within_set"].items():
        pr, s = key.split("|")
        A(f"| {pr} | {s} | {v['16']:.3f} | {v['64']:.3f} |")
    A("")
    A("Across sets — same rep (does occlusion rotate the subspace?), plus p(set1) vs h(set2):")
    A("")
    A("| rep | sets | k=16 | k=64 |")
    A("|---|---|---|---|")
    for key, v in a["across_sets"].items():
        if "|" in key:
            r, s = key.split("|")
        else:
            r, s = key, ""
        A(f"| {r} | {s} | {v['16']:.3f} | {v['64']:.3f} |")
    A("")
    A("Asymmetric containment — mean squared projection of A's top-16 PCA directions onto B's top-kB subspace (1.0 = inside; SPLIT rows = same rep, disjoint block halves, reference):")
    A("")
    A("| A top-16 in B top-kB | set | kB=64 | kB=256 |")
    A("|---|---|---|---|")
    seen = {}
    for key, v in a["containment"].items():
        lab, s = key.split("|"); base = lab.rsplit("top", 1)[0]; kb = lab.rsplit("top", 1)[1]
        seen.setdefault((base, s), {})[kb] = v
    for (base, s), v in seen.items():
        A(f"| {base} | {s} | {v['64']:.3f} | {v['256']:.3f} |")
    A("")
    A("Mean vectors (PCA removes them): cosine(mean p, mean h), cosine(mean z, mean h), norms, mean |μp − μh|:")
    A("")
    A("| set | cos(μp, μh) | cos(μz, μh) | ‖μp‖ | ‖μh‖ | mean\\|μp−μh\\| |")
    A("|---|---|---|---|---|---|")
    for s, v in a["mean_vectors"].items():
        A(f"| {s} | {v['cos_p_h']:.3f} | {v['cos_z_h']:.3f} | {v['norm_p']:.2f} | {v['norm_h']:.2f} | {v['l1_mean_gap']:.3f} |")
    A("")
    A("## (b) Linear CKA, token-matched p vs h")
    A("")
    A("| set | n_tokens | CKA(p,h) same clip+token | baseline: other clip, same token idx | baseline: same clip, permuted token | control: CKA(h, h_other) | CKA(z, h) same tok idx |")
    A("|---|---|---|---|---|---|---|")
    for s, v in res["b_cka"].items():
        A(f"| {s} | {v['n_tokens']} | {v['cka_matched']:.3f} | {v['cka_other_clip_same_token']:.3f} | {v['cka_same_clip_perm_token']:.3f} | {v['cka_h_vs_h_other']:.3f} | {v['cka_z_vs_h_sametokidx']:.3f} |")
    A("")
    A("## (c) Alignment maps p→h: fraction of mean|p−h| removed (block-disjoint train / val / test = 60 / 20 / 20 % of the source set)")
    A("")
    A("Map fit: orthogonal Procrustes (centered; 'scaled' adds one scalar), ridge (λ·mean-diag chosen on the validation fold, grid 1e-4…100), shift-only = mean offset. Transfer rows apply the map to ALL blocks of the target set.")
    A("")
    for fit, v in res["c_alignment"]["fits"].items():
        i = v["info"]
        A(f"### {fit}")
        A("")
        A(f"train tokens {i['n_train_tokens']} (blocks {i['n_train_blocks']}), val tokens {i['n_val_tokens']}, ridge λ_rel = {i['ridge_lambda_rel']}, Procrustes scale = {i['procrustes_scale']:.3f}")
        A("")
        A("| eval set | n_blocks | n_tokens | L1 before | shift-only % | Procrustes % | Procrustes+scale % | ridge % |")
        A("|---|---|---|---|---|---|---|---|")
        for s, e in v["evals"].items():
            A(f"| {s} | {e['n_blocks']} | {e['n_tokens']} | {e['l1_before']:.4f} | {100*e['shift_only']['frac_removed']:.1f} | {100*e['procrustes']['frac_removed']:.1f} | {100*e['procrustes_scaled']['frac_removed']:.1f} | {100*e['ridge']['frac_removed']:.1f} |")
        A("")
    A(f"runtime {res['runtime_s']:.0f} s")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
