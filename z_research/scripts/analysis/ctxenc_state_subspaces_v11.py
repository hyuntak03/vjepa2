#!/usr/bin/env python3
"""S1 — factor subspaces (identity / position / env / motion / direction / occluder / hidden / time-signature) in z, p and h on
IntPhysGen v11_full: are the state factors entangled with each other and with nuisance in the encoder, does the predictor keep
each factor's geometry, and which subspaces actually carry the latent-L1 scoring distance |p - h|?

Data   : data_csv/intphysgen_v11_full/index_probe.csv, probe_type == 'obj' (possible clips with an object, CLAUDE.md §1-4).
         9 cells: {static, moving_flat, moving(ramp)} × {visible k=0, occlusion_early k=4, occlusion(late) k=4}; --blocks blocks per cell
         (default 160, seed 0; a block contributes its pos_a and, for color/shape games, pos_b obj clip).
Cache  : /local_datasets/world/world_analysis/cache/v11_full_vith  (float16; ctx_masked = z, predictor = p (raw), target = h (LN'd)).
         Token index = tubelet*256 + row*16 + col.  Object 3x3 window = n3() copied verbatim from v11_token_object_test.py,
         positions = metadata object_px_{x,y}_by_sample averaged over the tubelet's 2 samples.
Reps   : z7 = z tubelet 7 object 3x3 mean (predictor input at the boundary)   p0 = p slot 0 (tubelet 8) 3x3 mean at the true t8 position
         h8 = h tubelet 8, same window.  Time signature uses the 3x3 means of all 8 tubelets (z: ctx 0..7; p: slots 0..7 = t8..15;
         h: future t8..15, plus h ctx t0..7 as a reference) with the tubelet index as an 8-class label (8 rows per clip).
Factors: shape_pre (7) color_pre (8) env (4) motion (3) direction (2, moving cells) position (x,y cells @t7 for z, @t8 for p/h; continuous)
         occluder (visible k0 vs early k4, object visible in both) hidden (early k4 vs late k4) time (8).
Subspace definition (used consistently): closed-form ridge readout W (D×k, k = #classes one-hot / #dims) fitted on centered features,
         λ = rel · tr(XᵀX)/D with rel chosen on a block-held-out validation fold; subspace = orthonormalised column span of W (effective
         dim k_eff = rank, k−1 for centered one-hot).  Categorical factors also get the regularised-LDA top-(k−1) variant and the
         between-class-scatter (class-mean span) variant.  Overlap(S1,S2) = mean cos² of principal angles (min(k1,k2) angles).
         Variance share of S in rep R = tr(P_S C_R)/tr(C_R) with C_R the centered token covariance (object-window rows, and all tokens).
         Scoring-distance share = Σ_tok ||P_S (p−h)||² / Σ_tok ||p−h||² (exact from accumulated Σ ddᵀ over all 2048 future tokens, the
         slot-0 object window and all-slot object windows) and the L1 analogue mean|P_S d| / mean|d| (token subsample: 16 random future
         tokens per clip + object windows).  p is raw, h is the LN'd target — the same tokens and distance the scorer uses.
Baselines: isotropic random subspace (analytic max(k1,k2)/D + 200 draws), label-shuffled ridge subspace (5 permutations; data-aligned
         lower bound), split-half (disjoint block halves) reliability = upper bound; block-bootstrap 95 % CIs (--boot, default 200,
         λ fixed at the selected value) for readout, overlaps, variance and scoring shares.  Held-out readout = 4-fold CV by block.
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_state_subspaces_v11/{results.json, RESULTS.md, features.npz, accum.npz}
         (refuses to overwrite an existing results.json).  CPU only, no model, deterministic.  ~15–25 min at defaults.
Run:
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_state_subspaces_v11.py --blocks 160 --boot 200
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "32"); os.environ.setdefault("MKL_NUM_THREADS", "32"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
import argparse, csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
OUT_DEFAULT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_state_subspaces_v11"
SEED, D, CELL_PX = 0, 1280, 18.0
# cell name, condition, k, motion, occ_state
CELLS = [("static_visible", "static_visible", "0", "static", "visible"),
         ("static_early", "static_occlusion_early", "4", "static", "early"),
         ("static_late", "static_occlusion", "4", "static", "late"),
         ("flat_visible", "moving_visible_flat", "0", "moving_flat", "visible"),
         ("flat_early", "moving_occlusion_flat_early", "4", "moving_flat", "early"),
         ("flat_late", "moving_occlusion_flat", "4", "moving_flat", "late"),
         ("ramp_visible", "moving_visible", "0", "moving", "visible"),
         ("ramp_early", "moving_occlusion_early", "4", "moving", "early"),
         ("ramp_late", "moving_occlusion", "4", "moving", "late")]
REPS = ["z7", "p0", "h8"]
TIME_REP = {"z7": "z_ctx", "p0": "p_fut", "h8": "h_fut"}        # which 8-tubelet stack gives the time-signature rows of a rep
FACTORS = ["shape", "color", "env", "motion", "direction", "position", "occluder", "hidden", "time"]
CAT = [f for f in FACTORS if f != "position"]
LAM_REL = [1e-3, 1e-2, 1e-1, 1, 10, 100]
GAMMAS = [1e-2, 1e-1, 1]
N_RAND_TOK, N_SHUF, N_RANDSUB, K_PCA = 16, 5, 200, 16
GROUPS = {"visible": ["static_visible", "flat_visible", "ramp_visible"], "early": ["static_early", "flat_early", "ramp_early"],
          "late": ["static_late", "flat_late", "ramp_late"]}


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def n3(x, y):  # copied verbatim from v11_token_object_test.py
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- selection + extraction
def select_clips(n_blocks):
    idx = [r for r in csv.DictReader(INDEX.open()) if r["probe_type"] == "obj"]
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    rng = np.random.RandomState(SEED); sel = []
    for cell, cond, k, motion, occ in CELLS:
        by_block = defaultdict(list)
        for r in idx:
            if r["condition"] == cond and r["sym_k"] == k:
                by_block[r["block_id"]].append(r)
        blocks = sorted(by_block); take = list(rng.permutation(blocks)[:n_blocks])
        for b in take:
            for r in by_block[b]:
                m = meta[r["video_id"]]
                tub = np.stack([arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])], -1).reshape(16, 2, 2).mean(1)
                sel.append(dict(video_id=r["video_id"], cell=cell, cond=cond, k=k, motion=motion, occ=occ, block=f"{cond}_k{k}_{b}",
                                violation_type=r["violation_type"], variant=r["variant"], shape=r["shape_pre"], color=r["color_pre"],
                                env=r["env"], direction=m["travel_direction"], hidden=(m["hidden_start"], m["hidden_end"]),
                                tub=tub.tolist()))
    return sel


def extract(sel):
    vid_row = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    assert Z.shape[1] == 2048 and H.shape[1] == 4096 and P.shape[1] == 2048 and Z.shape[2] == D
    n = len(sel); order = sorted(range(n), key=lambda i: vid_row[sel[i]["video_id"]])
    F = dict(Fz=np.zeros((n, 8, D), np.float32), Fh=np.zeros((n, 16, D), np.float32), Fp=np.zeros((n, 8, D), np.float32),
             Drand=np.zeros((n, N_RAND_TOK, D), np.float16), Dwin=np.zeros((n, 8, 9, D), np.float16), Dwin_mask=np.zeros((n, 8, 9), bool),
             win_n=np.zeros((n, 16), np.int8), l1_all=np.zeros(n), l1_win0=np.zeros(n), rand_tok=np.zeros((n, N_RAND_TOK), np.int32))
    acc = {c[0]: dict(n_clip=0, z_S=np.zeros(D), z_SS=np.zeros((D, D)), p_S=np.zeros(D), p_SS=np.zeros((D, D)), h_S=np.zeros(D), h_SS=np.zeros((D, D)),
                      dd_all=np.zeros((D, D)), dd_win0=np.zeros((D, D)), dd_win8=np.zeros((D, D)), l1_all=0.0, l1_win0=0.0, l1_win8=0.0) for c in CELLS}
    rng = np.random.RandomState(SEED + 7); t0 = time.time()
    for j, i in enumerate(order):
        s = sel[i]; tub = np.asarray(s["tub"]); wins = [n3(*tub[t]) for t in range(16)]
        assert all(len(w) == 9 for w in wins[:9]), (s["video_id"], [len(w) for w in wins])   # t0..t8 always full; later future tubelets may be partial (object leaves the frame)
        F["win_n"][i] = [len(w) for w in wins]
        mean_or_nan = lambda A: A.mean(0) if len(A) else np.full(D, np.nan, np.float32)
        r = vid_row[s["video_id"]]
        z = np.asarray(Z[r], np.float32); h = np.asarray(H[r], np.float32); p = np.asarray(P[r], np.float32)
        zt, ht, pt = z.reshape(8, 256, D), h.reshape(16, 256, D), p.reshape(8, 256, D)
        for t in range(8):
            F["Fz"][i, t] = mean_or_nan(zt[t, wins[t]]); F["Fp"][i, t] = mean_or_nan(pt[t, wins[8 + t]])
        for t in range(16):
            F["Fh"][i, t] = mean_or_nan(ht[t, wins[t]])
        hf = h[2048:]; d = p - hf; dt = d.reshape(8, 256, D)
        a = acc[s["cell"]]; a["n_clip"] += 1
        a["z_S"] += z.sum(0); a["z_SS"] += z.T @ z; a["p_S"] += p.sum(0); a["p_SS"] += p.T @ p; a["h_S"] += hf.sum(0); a["h_SS"] += hf.T @ hf
        a["dd_all"] += d.T @ d
        dw0 = dt[0, wins[8]]; a["dd_win0"] += dw0.T @ dw0
        dw8f = np.concatenate([dt[t, wins[8 + t]] for t in range(8)]); a["dd_win8"] += dw8f.T @ dw8f
        a["l1_all"] += float(np.abs(d).mean()); a["l1_win0"] += float(np.abs(dw0).mean()); a["l1_win8"] += float(np.abs(dw8f).mean())
        for t in range(8):
            F["Dwin"][i, t, :len(wins[8 + t])] = dt[t, wins[8 + t]]; F["Dwin_mask"][i, t, :len(wins[8 + t])] = True
        ri = np.sort(rng.choice(2048, N_RAND_TOK, replace=False)); F["rand_tok"][i] = ri; F["Drand"][i] = d[ri]
        F["l1_all"][i] = np.abs(d).mean(); F["l1_win0"][i] = np.abs(dw0).mean()
        if j % 200 == 0:
            log(f"  extract {j}/{n}  {time.time() - t0:.0f}s")
    return F, acc


# ----------------------------------------------------------------------------------------------- linear algebra
def orth(W, tol=1e-6):
    U, s, _ = np.linalg.svd(np.asarray(W, np.float64), full_matrices=False); k = int((s > tol * s[0]).sum()); return U[:, :k]


def overlap(Q1, Q2):
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False); return float((s ** 2).mean())


def wstats(X, w):
    sw = w.sum(); mu = (w @ X) / sw; Xc = X - mu; return mu, Xc, (Xc * w[:, None]).T @ Xc / sw


def wridge(X, Y, rel, w=None):
    """closed-form ridge on centered features; λ = rel·tr(XᵀX)/D.  w = row weights (bootstrap multiplicities)."""
    w = np.ones(len(X)) if w is None else w
    sw = w.sum(); mu = (w @ X) / sw; ym = (w @ Y) / sw; Xc = X - mu; Yc = Y - ym; Xw = Xc * w[:, None]
    G = Xw.T @ Xc; lam = rel * np.trace(G) / D
    W = np.linalg.solve(G + lam * np.eye(D), Xw.T @ Yc); return W, mu, ym


def ridge_predict(X, W, mu, ym):
    return (X - mu) @ W + ym


def lda_fit(X, y, classes, gamma, w=None):
    """regularised LDA: directions = top-(k−1) eigvecs of (Sw + γ tr(Sw)/D I)^{-1} Sb; returns directions (D×k−1), mu, class centroids (projected)."""
    w = np.ones(len(X)) if w is None else w
    sw = w.sum(); mu = (w @ X) / sw; Xc = X - mu; St = (Xc * w[:, None]).T @ Xc
    Sb = np.zeros((D, D)); mus = []
    for c in classes:
        m = y == c; wc = w[m].sum(); mc = (w[m] @ Xc[m]) / max(wc, 1e-12); mus.append(mc); Sb += wc * np.outer(mc, mc)
    Sw = St - Sb; R = Sw + gamma * np.trace(Sw) / D * np.eye(D)
    L = np.linalg.cholesky(R); A = np.linalg.solve(L, Sb); M = np.linalg.solve(L, A.T).T; M = (M + M.T) / 2
    ev, V = np.linalg.eigh(M); V = V[:, ::-1][:, :len(classes) - 1]
    Wd = np.linalg.solve(L.T, V); cent = np.stack(mus) @ Wd; return Wd, mu, cent, np.stack(mus)


def nearest_centroid(X, Wd, mu, cent, classes):
    pr = (X - mu) @ Wd; d2 = ((pr[:, None, :] - cent[None]) ** 2).sum(-1); return np.array(classes)[d2.argmin(1)]


def stratified_block_folds(blocks, strata, n_folds, seed):
    rng = np.random.RandomState(seed); b2s = {}
    for b, s in zip(blocks, strata):
        b2s.setdefault(b, s)
    fold_of = {}
    for s in sorted(set(b2s.values())):
        bs = sorted(b for b in b2s if b2s[b] == s); rng.shuffle(bs)
        for i, b in enumerate(bs):
            fold_of[b] = i % n_folds
    return np.array([fold_of[b] for b in blocks])


def boot_ci(vals):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


# ----------------------------------------------------------------------------------------------- factor datasets
class FD:
    """rows of one (rep, factor): X (rows×D, float64), y labels / Y targets, clip index per row, block per row, stratum per row."""

    def __init__(self, X, y, clip, block, strata, kind, classes=None):
        self.X = np.asarray(X, np.float64); self.y = y; self.clip = clip; self.block = block; self.strata = strata; self.kind = kind
        self.classes = classes
        self.Y = (np.eye(len(classes))[[classes.index(v) for v in y]] if kind == "cat" else np.asarray(y, np.float64))
        self.k = len(classes) if kind == "cat" else self.Y.shape[1]

    def metric(self, pred, m=None):
        """pred aligned with all rows; m = boolean mask / index of rows to score."""
        m = np.ones(len(self.y), bool) if m is None else m
        return self.metric_sub(pred[m], m)

    def metric_sub(self, pred_sub, idx):
        """pred_sub aligned with rows idx (accuracy for cat, mean R² over dims for cont)."""
        if self.kind == "cat":
            return float(np.mean(pred_sub == self.y[idx]))
        yy = self.Y[idx]; sst = ((yy - yy.mean(0)) ** 2).sum(0); return float(np.mean(1 - ((pred_sub - yy) ** 2).sum(0) / np.maximum(sst, 1e-12)))

    def predict_ridge(self, X, W, mu, ym):
        out = ridge_predict(X, W, mu, ym); return np.array(self.classes)[out.argmax(1)] if self.kind == "cat" else out


def build_fds(sel, F):
    n = len(sel); cell = np.array([s["cell"] for s in sel]); blk = np.array([s["block"] for s in sel]); occ = np.array([s["occ"] for s in sel])
    tub = np.array([s["tub"] for s in sel])  # (n,16,2) px
    stacks = {"z_ctx": F["Fz"], "p_fut": F["Fp"], "h_fut": F["Fh"][:, 8:16], "h_ctx": F["Fh"][:, 0:8]}
    R = {"z7": F["Fz"][:, 7], "p0": F["Fp"][:, 0], "h8": F["Fh"][:, 8]}
    labs = {"shape": np.array([s["shape"] for s in sel]), "color": np.array([s["color"] for s in sel]), "env": np.array([s["env"] for s in sel]),
            "motion": np.array([s["motion"] for s in sel]), "direction": np.array([s["direction"] for s in sel]),
            "occluder": np.where(occ == "visible", "no_occluder", "occluder"), "hidden": np.where(occ == "late", "hidden", "visible_at_boundary")}
    masks = {"direction": np.array([s["motion"] != "static" for s in sel]), "occluder": np.isin(occ, ["visible", "early"]), "hidden": np.isin(occ, ["early", "late"])}
    fds = {}
    for rep in REPS:
        for f in FACTORS:
            if f == "time":
                S = stacks[TIME_REP[rep]]; X = S.reshape(n * 8, D); y = np.tile(np.arange(8).astype(str), n); clip = np.repeat(np.arange(n), 8)
                ok = np.isfinite(X).all(1)   # drop tubelets whose object window is fully outside the frame
                fds[(rep, f)] = FD(X[ok], y[ok], clip[ok], blk[clip[ok]], cell[clip[ok]], "cat", sorted(set(y), key=int))
            elif f == "position":
                t = 7 if rep == "z7" else 8; Y = tub[:, t] / CELL_PX
                fds[(rep, f)] = FD(R[rep], Y, np.arange(n), blk, cell, "cont")
            else:
                m = masks.get(f, np.ones(n, bool)); idx = np.where(m)[0]; y = labs[f][idx]
                fds[(rep, f)] = FD(R[rep][idx], y, idx, blk[idx], np.array([f"{a}|{b}" for a, b in zip(cell[idx], y)]), "cat", sorted(set(y)))
    # h context-time reference (not a REPS member; used only for the time cross-rep table)
    X = stacks["h_ctx"].reshape(n * 8, D); y = np.tile(np.arange(8).astype(str), n); clip = np.repeat(np.arange(n), 8); ok = np.isfinite(X).all(1)
    fds[("h_ctx", "time")] = FD(X[ok], y[ok], clip[ok], blk[clip[ok]], cell[clip[ok]], "cat", sorted(set(y), key=int))
    return fds, R, cell, blk


# ----------------------------------------------------------------------------------------------- readout (held-out) + λ selection
def select_rel(fd, tr, seed):
    f = stratified_block_folds(fd.block[tr], fd.strata[tr], 4, seed + 1); itr, iva = tr[f != 0], tr[f == 0]
    best = None
    for rel in LAM_REL:
        W, mu, ym = wridge(fd.X[itr], fd.Y[itr], rel); sc = fd.metric_sub(fd.predict_ridge(fd.X[iva], W, mu, ym), iva)
        if best is None or sc >= best[0] - 1e-12:
            best = (sc, rel)
    return best[1], best[0]


def select_gamma(fd, tr, seed):
    f = stratified_block_folds(fd.block[tr], fd.strata[tr], 4, seed + 1); itr, iva = tr[f != 0], tr[f == 0]
    best = None
    for g in GAMMAS:
        Wd, mu, cent, _ = lda_fit(fd.X[itr], fd.y[itr], fd.classes, g); sc = float(np.mean(nearest_centroid(fd.X[iva], Wd, mu, cent, fd.classes) == fd.y[iva]))
        if best is None or sc >= best[0] - 1e-12:
            best = (sc, g)
    return best[1], best[0]


def cv_readout(fd, rng_boot, n_boot=1000):
    """4-fold CV by block; returns held-out metric (+ block bootstrap CI), chosen rel / gamma, out-of-fold LDA / BCS accuracy for cat."""
    folds = stratified_block_folds(fd.block, fd.strata, 4, SEED); n = len(fd.y)
    pred_r = np.zeros(n, object) if fd.kind == "cat" else np.zeros_like(fd.Y); pred_l = np.zeros(n, object); pred_b = np.zeros(n, object)
    rels, gams = [], []
    for fo in range(4):
        tr, te = np.where(folds != fo)[0], np.where(folds == fo)[0]
        rel, _ = select_rel(fd, tr, SEED); rels.append(rel); W, mu, ym = wridge(fd.X[tr], fd.Y[tr], rel); pred_r[te] = fd.predict_ridge(fd.X[te], W, mu, ym)
        if fd.kind == "cat":
            g, _ = select_gamma(fd, tr, SEED); gams.append(g); Wd, mu2, cent, mus = lda_fit(fd.X[tr], fd.y[tr], fd.classes, g)
            pred_l[te] = nearest_centroid(fd.X[te], Wd, mu2, cent, fd.classes)
            Qb = orth(mus.T); pred_b[te] = nearest_centroid(fd.X[te], Qb, mu2, mus @ Qb, fd.classes)
    rel_final = Counter(rels).most_common()[0][0]; g_final = Counter(gams).most_common()[0][0] if gams else None
    # block bootstrap of the held-out metric
    ub = np.unique(fd.block); b2rows = {b: np.where(fd.block == b)[0] for b in ub}; boots_r, boots_l = [], []
    for _ in range(n_boot):
        rows = np.concatenate([b2rows[b] for b in rng_boot.choice(ub, len(ub))])
        m = np.zeros(n, bool); m[rows] = True  # membership (weights ignored for the metric; blocks are near-equal size)
        boots_r.append(fd.metric(pred_r, m))
        if fd.kind == "cat":
            boots_l.append(float(np.mean(pred_l[rows] == fd.y[rows])))
    out = dict(kind=fd.kind, k=fd.k, n_rows=int(n), n_clips=int(len(np.unique(fd.clip))), n_blocks=int(len(ub)), rel_per_fold=rels, rel_final=rel_final,
               ridge=fd.metric(pred_r), ridge_ci=boot_ci(boots_r), chance=(1 / fd.k if fd.kind == "cat" else 0.0),
               per_cell={c: fd.metric(pred_r, np.array([s.split("|")[0] == c for s in fd.strata])) for c in sorted(set(s.split("|")[0] for s in fd.strata))})
    if fd.kind == "cat":
        out.update(gamma_per_fold=gams, gamma_final=g_final, lda=float(np.mean(pred_l == fd.y)), lda_ci=boot_ci(boots_l), bcs=float(np.mean(pred_b == fd.y)),
                   class_counts=dict(Counter(fd.y)))
    else:
        out.update(r2_per_dim=[float(1 - ((pred_r[:, j] - fd.Y[:, j]) ** 2).sum() / max(((fd.Y[:, j] - fd.Y[:, j].mean()) ** 2).sum(), 1e-12)) for j in range(fd.k)],
                   label_std=fd.Y.std(0).tolist())
    return out


def fit_subspaces(fd, rel, gamma, w=None):
    """returns {ridge: Q, lda: Q, bcs: Q} (orthonormal D×k_eff) on the (weighted) rows."""
    W, mu, ym = wridge(fd.X, fd.Y, rel, w); out = {"ridge": orth(W)}
    if fd.kind == "cat":
        Wd, _, _, mus = lda_fit(fd.X, fd.y, fd.classes, gamma, w); out["lda"] = orth(Wd); out["bcs"] = orth(mus.T)
    return out


# ----------------------------------------------------------------------------------------------- shares
def var_share(Q, C):
    return float(np.trace(Q.T @ C @ Q) / np.trace(C))


def l2_share_tokens(Q, Dt, w):
    """Dt (N×D) token residuals, w token weights."""
    num = (w * ((Dt @ Q) ** 2).sum(1)).sum(); den = (w * (Dt ** 2).sum(1)).sum(); return float(num / den)


def l1_share_tokens(Q, Dt, w):
    proj = (Dt @ Q) @ Q.T; a = (w * np.abs(proj).mean(1)).sum(); b = (w * np.abs(Dt - proj).mean(1)).sum(); c = (w * np.abs(Dt).mean(1)).sum()
    return float(a / c), float(b / c)


def pca_from_acc(S, SS, n, k=K_PCA):
    mu = S / n; C = SS / n - np.outer(mu, mu); ev, V = np.linalg.eigh(C); V = V[:, ::-1]; ev = ev[::-1]
    return V[:, :k], float(ev[:k].sum() / ev.sum()), C


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--blocks", type=int, default=160); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--reuse-features", action="store_true"); ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.time()
    if (OUT / "results.json").exists():
        sys.exit(f"refusing to overwrite {OUT}/results.json — move it away or pass --out")
    fpath, apath = OUT / "features.npz", OUT / "accum.npz"
    sel = select_clips(a.blocks); log(f"selected {len(sel)} clips in {len(CELLS)} cells")
    if a.reuse_features and fpath.exists() and apath.exists() and [str(v) for v in np.load(fpath, allow_pickle=True)["video_id"]] == [s["video_id"] for s in sel]:
        npz = np.load(fpath, allow_pickle=True); F = {k: npz[k] for k in npz if k != "video_id"}
        an = np.load(apath); acc = defaultdict(dict)
        for key in an:
            c, name = key.split("__"); acc[c][name] = an[key] if an[key].ndim else an[key].item()
        log("features + accumulators reused")
    else:
        F, acc = extract(sel)
        np.savez(fpath, video_id=np.array([s["video_id"] for s in sel]), **F)
        np.savez(apath, **{f"{c}__{k}": np.asarray(v) for c, d in acc.items() for k, v in d.items()})
        log(f"features saved ({time.time() - T0:.0f}s)")
    n = len(sel); fds, R, cell, blk = build_fds(sel, F)
    cells_info = {c[0]: dict(condition=c[1], k=c[2], motion=c[3], occ=c[4], n_clips=int((cell == c[0]).sum()), n_blocks=len(set(blk[cell == c[0]])),
                             win_n_mean_per_tubelet=[float(v) for v in F["win_n"][cell == c[0]].mean(0)], n_empty_windows_t8_15=int((F["win_n"][cell == c[0], 8:] == 0).sum()),
                             hidden_frames=Counter(str(s["hidden"]) for s in sel if s["cell"] == c[0]).most_common(1)[0][0],
                             scorer_mean_abs_p_minus_h_all=float(acc[c[0]]["l1_all"] / acc[c[0]]["n_clip"]), scorer_mean_abs_win0=float(acc[c[0]]["l1_win0"] / acc[c[0]]["n_clip"]),
                             violation_counts=dict(Counter(s["violation_type"] for s in sel if s["cell"] == c[0]))) for c in CELLS}

    # ---- (a) held-out readout + λ/γ selection
    log("(a) held-out readout"); rng_b = np.random.RandomState(SEED + 3); readout = {}
    for key, fd in fds.items():
        t1 = time.time(); readout[f"{key[0]}|{key[1]}"] = cv_readout(fd, rng_b); r = readout[f"{key[0]}|{key[1]}"]
        log(f"   {key[0]:5s} {key[1]:9s} n={r['n_rows']:6d} ridge {r['ridge']:.3f} {'lda %.3f' % r['lda'] if 'lda' in r else ''} rel={r['rel_final']} ({time.time() - t1:.0f}s)")
    REL = {k: readout[f"{k[0]}|{k[1]}"]["rel_final"] for k in fds}; GAM = {k: readout[f"{k[0]}|{k[1]}"].get("gamma_final") for k in fds}

    # ---- full-sample subspaces (+ split-half + shuffled)
    log("subspaces: full, split-half, shuffled")
    Q = {k: fit_subspaces(fd, REL[k], GAM[k]) for k, fd in fds.items()}
    half = stratified_block_folds(blk, cell, 2, SEED + 11)  # clip-level half id by block
    Qh = {k: [fit_subspaces(fd, REL[k], GAM[k], w=(half[fd.clip] == hh).astype(float)) for hh in (0, 1)] for k, fd in fds.items()}
    rng_s = np.random.RandomState(SEED + 5); Qs = {}
    for k, fd in fds.items():
        lst = []
        for _ in range(N_SHUF):
            y = fd.y.copy()
            if k[1] == "time":   # permute tubelet labels within clip
                for c in np.unique(fd.clip):
                    m = np.where(fd.clip == c)[0]; y[m] = y[rng_s.permutation(m)]
            else:                # permute labels among rows of the same cell
                cl = np.array([s.split("|")[0] for s in fd.strata])
                for c in np.unique(cl):
                    m = np.where(cl == c)[0]; y[m] = y[rng_s.permutation(m)]
            Yp = np.eye(fd.k)[[fd.classes.index(v) for v in y]] if fd.kind == "cat" else y
            W, _, _ = wridge(fd.X, Yp, REL[k]); lst.append(orth(W))
        Qs[k] = lst
    kdim = {f"{k[0]}|{k[1]}": {m: int(q.shape[1]) for m, q in Q[k].items()} for k in Q}
    rng_r = np.random.RandomState(SEED + 9)

    def rand_overlap(k1, k2):
        v = [overlap(orth(rng_r.randn(D, k1)), orth(rng_r.randn(D, k2))) for _ in range(N_RANDSUB)]; return dict(analytic=max(k1, k2) / D, empirical_mean=float(np.mean(v)), empirical_p97_5=float(np.percentile(v, 97.5)))

    # ---- (b) pairwise overlap within each rep
    log("(b) pairwise overlaps"); pairwise = {}
    for rep in REPS:
        for meth in ("ridge", "lda"):
            M = {}
            for i, f1 in enumerate(FACTORS):
                for f2 in FACTORS[i + 1:]:
                    k1, k2 = (rep, f1), (rep, f2)
                    if meth == "lda" and (f1 not in CAT or f2 not in CAT):
                        continue
                    q1, q2 = Q[k1][meth], Q[k2][meth]
                    M[f"{f1}|{f2}"] = dict(k1=int(q1.shape[1]), k2=int(q2.shape[1]), full=overlap(q1, q2),
                                           cross_half=float(np.mean([overlap(Qh[k1][0][meth], Qh[k2][1][meth]), overlap(Qh[k1][1][meth], Qh[k2][0][meth])])),
                                           random=rand_overlap(q1.shape[1], q2.shape[1]),
                                           shuffle_real_vs_shuf=float(np.mean([overlap(q1, s) for s in Qs[k2]] + [overlap(s, q2) for s in Qs[k1]])) if meth == "ridge" else None,
                                           shuffle_shuf_vs_shuf=float(np.mean([overlap(s1, s2) for s1, s2 in zip(Qs[k1], Qs[k2])])) if meth == "ridge" else None)
            pairwise[f"{rep}|{meth}"] = M
    reliability = {f"{k[0]}|{k[1]}": {m: overlap(Qh[k][0][m], Qh[k][1][m]) for m in Q[k]} for k in Q}
    # ridge vs lda vs bcs agreement within (rep, factor)
    method_agreement = {f"{k[0]}|{k[1]}": dict(ridge_lda=overlap(Q[k]["ridge"], Q[k]["lda"]), ridge_bcs=overlap(Q[k]["ridge"], Q[k]["bcs"]), lda_bcs=overlap(Q[k]["lda"], Q[k]["bcs"])) for k in Q if "lda" in Q[k]}

    # ---- covariances + PCA
    log("(c) variance shares + PCA"); tot = {}
    for r_ in ("z", "p", "h"):
        S = sum(acc[c]["%s_S" % r_] for c in acc); SS = sum(acc[c]["%s_SS" % r_] for c in acc); nt = sum(acc[c]["n_clip"] for c in acc) * 2048
        tot[r_] = (S, SS, nt)
    PCA, C_all, evfrac = {}, {}, {}
    for r_, rep in (("z", "z7"), ("p", "p0"), ("h", "h8")):
        PCA[rep + "|all"], evfrac[rep + "|all"], C_all[rep] = pca_from_acc(*tot[r_])
        Xw = np.asarray(R[rep], np.float64); _, Xc, Cw = wstats(Xw, np.ones(n)); ev, V = np.linalg.eigh(Cw); PCA[rep + "|win"] = V[:, ::-1][:, :K_PCA]; evfrac[rep + "|win"] = float(ev[::-1][:K_PCA].sum() / ev.sum())
    C_win = {rep: wstats(np.asarray(R[rep], np.float64), np.ones(n))[2] for rep in REPS}
    varshare = {}
    for src in REPS:
        for f in FACTORS:
            for meth, q in Q[(src, f)].items():
                varshare[f"{src}|{f}|{meth}"] = {tgt: dict(win=var_share(q, C_win[tgt]), all=var_share(q, C_all[tgt])) for tgt in REPS}
    varshare_ref = {f"{rep}|pca16_{sp}": {tgt: dict(win=var_share(PCA[f"{rep}|{sp}"], C_win[tgt]), all=var_share(PCA[f"{rep}|{sp}"], C_all[tgt])) for tgt in REPS} for rep in REPS for sp in ("all", "win")}
    varshare_ref.update({f"{src}|{f}|shuffle": {tgt: dict(win=float(np.mean([var_share(q, C_win[tgt]) for q in Qs[(src, f)]])), all=float(np.mean([var_share(q, C_all[tgt]) for q in Qs[(src, f)]]))) for tgt in REPS} for src in REPS for f in FACTORS})

    # ---- (d) cross-representation
    log("(d) cross-rep overlaps"); cross = {}
    for f in FACTORS:
        for meth in (("ridge", "lda") if f in CAT else ("ridge",)):
            rec = {}
            for r1, r2 in (("z7", "p0"), ("z7", "h8"), ("p0", "h8")):
                q1, q2 = Q[(r1, f)][meth], Q[(r2, f)][meth]
                rec[f"{r1}-{r2}"] = dict(k1=int(q1.shape[1]), k2=int(q2.shape[1]), full=overlap(q1, q2), cross_half=float(np.mean([overlap(Qh[(r1, f)][0][meth], Qh[(r2, f)][1][meth]), overlap(Qh[(r1, f)][1][meth], Qh[(r2, f)][0][meth])])),
                                          reliability=[reliability[f"{r1}|{f}"][meth], reliability[f"{r2}|{f}"][meth]], random=rand_overlap(q1.shape[1], q2.shape[1]),
                                          shuffle=float(np.mean([overlap(q1, s) for s in Qs[(r2, f)]])) if meth == "ridge" else None)
            if f == "time":
                q1, q2 = Q[("z7", f)][meth], Q[("h_ctx", f)][meth]; rec["z_ctx-h_ctx"] = dict(k1=int(q1.shape[1]), k2=int(q2.shape[1]), full=overlap(q1, q2), random=rand_overlap(q1.shape[1], q2.shape[1]), reliability=[reliability["z7|time"][meth], reliability["h_ctx|time"][meth]])
                q1, q2 = Q[("h_ctx", f)][meth], Q[("h8", f)][meth]; rec["h_ctx-h_fut"] = dict(k1=int(q1.shape[1]), k2=int(q2.shape[1]), full=overlap(q1, q2), random=rand_overlap(q1.shape[1], q2.shape[1]))
            cross[f"{f}|{meth}"] = rec
    pca_overlap = {}
    for src in REPS:
        for f in FACTORS:
            q = Q[(src, f)]["ridge"]
            pca_overlap[f"{src}|{f}"] = {f"{rep}|{sp}": overlap(q, PCA[f"{rep}|{sp}"]) for rep in REPS for sp in ("all", "win")}
            pca_overlap[f"{src}|{f}"]["random_analytic"] = K_PCA / D
            pca_overlap[f"{src}|{f}"]["shuffle_vs_p0_all"] = float(np.mean([overlap(s, PCA["p0|all"]) for s in Qs[(src, f)]]))
    pca_overlap["pca16 z-p-h (all tokens)"] = {"z7-p0": overlap(PCA["z7|all"], PCA["p0|all"]), "z7-h8": overlap(PCA["z7|all"], PCA["h8|all"]), "p0-h8": overlap(PCA["p0|all"], PCA["h8|all"]), "explained_var_top16": evfrac}

    # ---- (e) scoring-distance share
    log("(e) scoring-distance shares"); scoring = {}
    union = {src: orth(np.concatenate([Q[(src, f)]["ridge"] for f in FACTORS], 1)) for src in REPS}
    ident = {src: orth(np.concatenate([Q[(src, f)]["ridge"] for f in ("shape", "color")], 1)) for src in REPS}
    subs_for_scoring = lambda src: dict(**{f: Q[(src, f)]["ridge"] for f in FACTORS}, identity=ident[src], all_factors_union=union[src])
    Drand = F["Drand"]; Dwin = F["Dwin"]
    grp_cells = dict(GROUPS); grp_cells.update({c[0]: [c[0]] for c in CELLS})
    for g, cl in grp_cells.items():
        m = np.isin(cell, cl); idx = np.where(m)[0]
        Cd = {t: sum(acc[c][f"dd_{t}"] for c in cl) for t in ("all", "win0", "win8")}; wm8 = F["Dwin_mask"][idx].reshape(-1).astype(float)
        ntok = {"all": len(idx) * 2048, "win0": len(idx) * 9, "win8": int(wm8.sum())}
        Dr = np.asarray(Drand[idx], np.float32).reshape(-1, D); Dw0 = np.asarray(Dwin[idx, 0], np.float32).reshape(-1, D); Dw8 = np.asarray(Dwin[idx], np.float32).reshape(-1, D)
        w1 = {"all": np.ones(len(Dr)), "win0": np.ones(len(Dw0)), "win8": wm8}; Dt = {"all": Dr, "win0": Dw0, "win8": Dw8}
        rec = dict(n_clips=int(len(idx)), n_blocks=int(len(set(blk[idx]))), n_tokens_exact=ntok, n_tokens_subsample={t: int(w1[t].sum()) for t in Dt},
                   mean_abs_d={t: float((w1[t] * np.abs(Dt[t]).mean(1)).sum() / w1[t].sum()) for t in Dt}, sum_sq_d={t: float(np.trace(Cd[t])) for t in Cd}, subspaces={})
        for src in REPS:
            for name, q in subs_for_scoring(src).items():
                e = dict(k=int(q.shape[1]), random_analytic=q.shape[1] / D)
                for t in ("all", "win0", "win8"):
                    l1s, l1c = l1_share_tokens(q, Dt[t], w1[t]); e[t] = dict(l2_exact=var_share(q, Cd[t]), l2_sub=l2_share_tokens(q, Dt[t], w1[t]), l1=l1s, l1_complement=l1c)
                e["complement_l2_exact"] = {t: 1 - e[t]["l2_exact"] for t in ("all", "win0", "win8")}
                rec["subspaces"][f"{src}|{name}"] = e
            for f in ("shape", "color", "position", "time"):
                e = {t: dict(l2_exact=float(np.mean([var_share(s, Cd[t]) for s in Qs[(src, f)]])), l1=float(np.mean([l1_share_tokens(s, Dt[t], w1[t])[0] for s in Qs[(src, f)]]))) for t in ("all", "win0", "win8")}
                rec["subspaces"][f"{src}|{f}|shuffle"] = dict(k=int(Qs[(src, f)][0].shape[1]), **e)
        for rep in REPS:
            for sp in ("all", "win"):
                q = PCA[f"{rep}|{sp}"]; rec["subspaces"][f"pca16_{rep}_{sp}"] = dict(k=K_PCA, random_analytic=K_PCA / D, **{t: dict(l2_exact=var_share(q, Cd[t]), l1=l1_share_tokens(q, Dt[t], w1[t])[0]) for t in ("all", "win0", "win8")})
        scoring[g] = rec

    # ---- block bootstrap (λ fixed) for headline numbers
    log(f"bootstrap B={a.boot}"); rng_bb = np.random.RandomState(SEED + 21); ub = {c: np.unique(blk[cell == c]) for c in cells_info}
    bs = defaultdict(list); t1 = time.time()
    grp_idx = {g: np.where(np.isin(cell, cl))[0] for g, cl in GROUPS.items()}
    Dr_g = {g: np.asarray(Drand[grp_idx[g]], np.float32).reshape(-1, D) for g in GROUPS}; Dw_g = {g: np.asarray(Dwin[grp_idx[g], 0], np.float32).reshape(-1, D) for g in GROUPS}
    for b in range(a.boot):
        wclip = np.zeros(n)
        for c, bl in ub.items():
            cnt = Counter(rng_bb.choice(bl, len(bl)))
            for bb, k_ in cnt.items():
                wclip[(blk == bb)] += k_
        Qb = {}
        for k, fd in fds.items():
            if k[0] == "h_ctx":
                continue
            W, _, _ = wridge(fd.X, fd.Y, REL[k], wclip[fd.clip]); Qb[k] = orth(W)
        Cb = {rep: wstats(np.asarray(R[rep], np.float64), wclip)[2] for rep in REPS}
        for i, f1 in enumerate(FACTORS):
            for f2 in FACTORS[i + 1:]:
                for rep in REPS:
                    bs[f"pair|{rep}|{f1}|{f2}"].append(overlap(Qb[(rep, f1)], Qb[(rep, f2)]))
        for f in FACTORS:
            for r1, r2 in (("z7", "p0"), ("z7", "h8"), ("p0", "h8")):
                bs[f"cross|{f}|{r1}-{r2}"].append(overlap(Qb[(r1, f)], Qb[(r2, f)]))
            for src in REPS:
                bs[f"pca|{src}|{f}|p0_all"].append(overlap(Qb[(src, f)], PCA["p0|all"]))
                for tgt in REPS:
                    bs[f"var|{src}|{f}|{tgt}|win"].append(var_share(Qb[(src, f)], Cb[tgt]))
        for g in GROUPS:
            wt_r = np.repeat(wclip[grp_idx[g]], N_RAND_TOK); wt_w = np.repeat(wclip[grp_idx[g]], 9)
            for src in REPS:
                subs = dict(**{f: Qb[(src, f)] for f in FACTORS}, identity=orth(np.concatenate([Qb[(src, "shape")], Qb[(src, "color")]], 1)), all_factors_union=orth(np.concatenate([Qb[(src, f)] for f in FACTORS], 1)))
                for name, q in subs.items():
                    bs[f"score|{g}|{src}|{name}|all|l2"].append(l2_share_tokens(q, Dr_g[g], wt_r)); bs[f"score|{g}|{src}|{name}|all|l1"].append(l1_share_tokens(q, Dr_g[g], wt_r)[0])
                    bs[f"score|{g}|{src}|{name}|win0|l2"].append(l2_share_tokens(q, Dw_g[g], wt_w)); bs[f"score|{g}|{src}|{name}|win0|l1"].append(l1_share_tokens(q, Dw_g[g], wt_w)[0])
        if b % 20 == 0:
            log(f"   boot {b}/{a.boot}  {time.time() - t1:.0f}s")
    CI = {k: boot_ci(v) for k, v in bs.items()}

    res = dict(script=str(Path(__file__).relative_to(ROOT)), seed=SEED, n_blocks_per_cell=a.blocks, n_boot=a.boot, n_clips=int(n), n_blocks=int(len(set(blk))),
               cache=str(CACHE), index=str(INDEX), lam_rel_grid=LAM_REL, gamma_grid=GAMMAS, n_rand_tok_per_clip=N_RAND_TOK, n_shuffle=N_SHUF, k_pca=K_PCA,
               definitions=dict(subspace="orthonormal span of closed-form ridge W (centered features, λ=rel·tr(XᵀX)/D); lda = regularised LDA top-(k−1); bcs = class-mean span",
                                overlap="mean cos² of principal angles over min(k1,k2) angles", variance_share="tr(P_S C)/tr(C), C = centered token covariance (win = object-window rows; all = all 2048 tokens of the half-clip)",
                                scoring_share="Σ||P_S d||²/Σ||d||² (l2_exact: all tokens of the cell from accumulated Σddᵀ; l2_sub: token subsample) and mean|P_S d|/mean|d| (l1), d = p − h on the scorer's future tokens; all = all 2048 future tokens (subsample 16/clip), win0 = slot-0 object 3x3, win8 = object 3x3 at all 8 slots",
                                bootstrap="block bootstrap within cell, λ fixed at the selected rel, B resamples; CIs are percentile 2.5/97.5; the all-token exact l2 shares have no CI (subsample l2/l1 do)",
                                position_label="cells (px/18) at tubelet 7 for z7, tubelet 8 for p0/h8", time_rows="3x3-mean token per tubelet (8 rows per clip); z: ctx t0..7, p: slots 0..7 (t8..15), h8: future t8..15, h_ctx: t0..7"),
               cells=cells_info, k_dims=kdim, readout=readout, reliability_split_half=reliability, method_agreement=method_agreement, pairwise=pairwise,
               variance_share=varshare, variance_share_ref=varshare_ref, cross_rep=cross, pca_overlap=pca_overlap, scoring=scoring, boot_ci=CI, elapsed_s=float(time.time() - T0))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_md(res, OUT); log(f"-> {OUT}/results.json, RESULTS.md  ({time.time() - T0:.0f}s)")


# ----------------------------------------------------------------------------------------------- markdown (generated from results.json only)
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def ci(c, nd=2):
    return f"[{c[0]:.{nd}f},{c[1]:.{nd}f}]" if c and c[0] is not None else "—"


def write_md(res, OUT):
    L = [f"# S1 — factor subspaces in z / p / h and what the latent-L1 score is sensitive to (IntPhysGen v11_full, ViT-H)", "",
         f"Every number is recomputed from `results.json` written by `{res['script']}` (seed {res['seed']}, {res['n_blocks_per_cell']} blocks per cell, {res['n_clips']} clips / {res['n_blocks']} blocks, bootstrap B={res['n_boot']}).",
         "Subspace = orthonormalised span of the closed-form ridge readout W (centered features, λ = rel·tr(XᵀX)/D, rel by block-held-out validation); `lda` = regularised LDA top-(k−1); `bcs` = class-mean span. "
         "Overlap = mean cos² of principal angles over min(k1,k2) angles (1 = nested). Random baseline = isotropic random subspaces (analytic max(k1,k2)/D). Split-half = same factor fitted on disjoint block halves (upper bound). "
         "`shuf` = ridge subspace fitted on permuted labels (data-aligned random lower bound). CIs = block bootstrap (λ fixed).",
         f"Reps: z7 = ctx_masked tubelet 7 object 3×3 mean (predictor input at the boundary); p0 = predictor slot 0 (t8) at the true t8 object window; h8 = target t8 same window. Time signature: 8 rows per clip ({res['definitions']['time_rows']}).", ""]
    L += ["## Sample", "", "| cell | condition | k | n clips | n blocks | hidden frames | scorer mean|p−h| all tokens | mean|p−h| slot-0 window |", "|---|---|---|---:|---:|---|---:|---:|"]
    for c, v in res["cells"].items():
        L.append(f"| {c} | {v['condition']} | {v['k']} | {v['n_clips']} | {v['n_blocks']} | {v['hidden_frames']} | {v['scorer_mean_abs_p_minus_h_all']:.3f} | {v['scorer_mean_abs_win0']:.3f} |")
    L += ["", "## (a) Are the factor subspaces real? Held-out readout (4-fold CV by block) — ridge accuracy / R² [95 % CI], LDA accuracy, k_eff", "",
          "| factor | rows / clips / blocks | chance | " + " | ".join(f"{r} ridge [CI] · lda · k_eff" for r in REPS) + " |", "|---|---|---:|" + "---|" * len(REPS)]
    for f in FACTORS:
        cells_ = []
        for r in REPS:
            o = res["readout"][f"{r}|{f}"]; kd = res["k_dims"][f"{r}|{f}"]["ridge"]
            cells_.append(f"{o['ridge']:.3f} {ci(o['ridge_ci'])} · {fmt(o.get('lda'))} · {kd}")
        o = res["readout"][f"z7|{f}"]; L.append(f"| {f} ({'R²' if o['kind'] == 'cont' else 'acc'}) | {o['n_rows']} / {o['n_clips']} / {o['n_blocks']} | {o['chance']:.3f} | " + " | ".join(cells_) + " |")
    L += ["", "Per-cell ridge readout (z7 / p0 / h8):", "", "| factor | " + " | ".join(res["cells"]) + " |", "|---|" + "---|" * len(res["cells"])]
    for f in FACTORS:
        row = []
        for c in res["cells"]:
            v = [res["readout"][f"{r}|{f}"]["per_cell"].get(c) for r in REPS]; row.append("/".join("—" if x is None else f"{x:.2f}" for x in v))
        L.append(f"| {f} | " + " | ".join(row) + " |")
    L += ["", "Split-half reliability of each subspace (overlap of the two half fits; ridge / lda / bcs) and ridge–LDA agreement:", "", "| factor | " + " | ".join(f"{r} rel(ridge/lda/bcs) · ridge∩lda" for r in REPS) + " |", "|---|" + "---|" * len(REPS)]
    for f in FACTORS:
        row = []
        for r in REPS:
            rl = res["reliability_split_half"][f"{r}|{f}"]; ma = res["method_agreement"].get(f"{r}|{f}", {})
            row.append(f"{rl['ridge']:.2f}/{fmt(rl.get('lda'), 2)}/{fmt(rl.get('bcs'), 2)} · {fmt(ma.get('ridge_lda'), 2)}")
        L.append(f"| {f} | " + " | ".join(row) + " |")
    for rep in REPS:
        for meth in ("ridge", "lda"):
            M = res["pairwise"][f"{rep}|{meth}"]; fs = FACTORS if meth == "ridge" else CAT
            L += ["", f"## (b) Pairwise subspace overlap within {rep} — {meth} (upper triangle: full fit [bootstrap CI]; lower triangle: cross-half / random analytic{' / shuf' if meth == 'ridge' else ''})", "",
                  "| | " + " | ".join(fs) + " |", "|---|" + "---|" * len(fs)]
            for i, f1 in enumerate(fs):
                row = []
                for j, f2 in enumerate(fs):
                    if i == j:
                        row.append(f"k={res['k_dims'][f'{rep}|{f1}'][meth]}")
                    elif j > i:
                        e = M[f"{f1}|{f2}"]; c = res["boot_ci"].get(f"pair|{rep}|{f1}|{f2}") if meth == "ridge" else None
                        row.append(f"**{e['full']:.2f}** {ci(c) if c else ''}")
                    else:
                        e = M[f"{f2}|{f1}"]; row.append(f"{e['cross_half']:.2f} / {e['random']['analytic']:.3f}" + (f" / {e['shuffle_real_vs_shuf']:.2f}" if meth == "ridge" else ""))
                L.append(f"| {f1} | " + " | ".join(row) + " |")
    L += ["", "## (c) Variance share of each factor subspace (ridge) — tr(P_S C)/tr(C), C from object-window rows (win) and from all 2048 half-clip tokens (all)", "",
          "Rows: subspace fitted on `src`; columns: variance share inside z7 / p0 / h8 (win | all). Random k-dim subspace expectation = k/1280. Reference rows: top-16 PCA of each rep, and label-shuffled ridge subspaces.", "",
          "| src | factor | k | " + " | ".join(f"in {t} win [CI] · all" for t in REPS) + " |", "|---|---|---:|" + "---|" * len(REPS)]
    for src in REPS:
        for f in FACTORS:
            e = res["variance_share"][f"{src}|{f}|ridge"]; k = res["k_dims"][f"{src}|{f}"]["ridge"]
            L.append(f"| {src} | {f} | {k} (rand {k / 1280:.3f}) | " + " | ".join(f"{e[t]['win']:.3f} {ci(res['boot_ci'].get(f'var|{src}|{f}|{t}|win'), 3)} · {e[t]['all']:.3f}" for t in REPS) + " |")
        for f in FACTORS:
            e = res["variance_share_ref"][f"{src}|{f}|shuffle"]; L.append(f"| {src} | {f} (shuf) | | " + " | ".join(f"{e[t]['win']:.3f} · {e[t]['all']:.3f}" for t in REPS) + " |")
    for rep in REPS:
        for sp in ("all", "win"):
            e = res["variance_share_ref"][f"{rep}|pca16_{sp}"]; L.append(f"| {rep} | pca16 ({sp} tokens) | 16 | " + " | ".join(f"{e[t]['win']:.3f} · {e[t]['all']:.3f}" for t in REPS) + " |")
    L += ["", "## (d) Cross-representation overlap of the same factor — does p keep the factor geometry of z?", "",
          "| factor | meth | " + " | ".join(f"{pr} full [CI] · cross-half · rel(a,b) · rand · shuf" for pr in ("z7-p0", "z7-h8", "p0-h8")) + " |", "|---|---|" + "---|" * 3]
    for key, rec in res["cross_rep"].items():
        f, meth = key.split("|"); row = []
        for pr in ("z7-p0", "z7-h8", "p0-h8"):
            e = rec[pr]; c = res["boot_ci"].get(f"cross|{f}|{pr}") if meth == "ridge" else None
            row.append(f"{e['full']:.2f} {ci(c) if c else ''} · {e['cross_half']:.2f} · ({e['reliability'][0]:.2f},{e['reliability'][1]:.2f}) · {e['random']['analytic']:.3f} · {fmt(e.get('shuffle'), 2)}")
        L.append(f"| {f} | {meth} | " + " | ".join(row) + " |")
        if f == "time":
            L.append(f"| time (extra) | {meth} | z_ctx–h_ctx {rec['z_ctx-h_ctx']['full']:.2f} (rel {rec['z_ctx-h_ctx']['reliability'][0]:.2f},{rec['z_ctx-h_ctx']['reliability'][1]:.2f}) | h_ctx–h_fut {rec['h_ctx-h_fut']['full']:.2f} | |")
    L += ["", "Overlap of factor subspaces (ridge) with top-16 PCA subspaces (random analytic 16/1280 = 0.0125):", "",
          "| src | factor | k | vs pca16 p0 (all tokens) [CI] | vs pca16 p0 (win) | vs pca16 z7 (all) | vs pca16 h8 (all) | shuf vs pca16 p0 |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for src in REPS:
        for f in FACTORS:
            e = res["pca_overlap"][f"{src}|{f}"]; L.append(f"| {src} | {f} | {res['k_dims'][f'{src}|{f}']['ridge']} | {e['p0|all']:.3f} {ci(res['boot_ci'].get(f'pca|{src}|{f}|p0_all'), 3)} | {e['p0|win']:.3f} | {e['z7|all']:.3f} | {e['h8|all']:.3f} | {e['shuffle_vs_p0_all']:.3f} |")
    pz = res["pca_overlap"]["pca16 z-p-h (all tokens)"]
    L.append(f"\nPCA16 overlaps (all tokens): z7–p0 {pz['z7-p0']:.3f}, z7–h8 {pz['z7-h8']:.3f}, p0–h8 {pz['p0-h8']:.3f}; explained variance of top-16: " + ", ".join(f"{k} {v:.2f}" for k, v in pz["explained_var_top16"].items()))
    L += ["", "## (e) Scoring-distance share — what fraction of the scorer's |p − h| lies in each subspace", "",
          "d = p − h on the future tokens the scorer uses. `L2` = Σ||P_S d||²/Σ||d||² over ALL tokens of the cell (exact, from Σddᵀ); `L1` = mean|P_S d| / mean|d| on the token subsample [bootstrap CI]. "
          "all = all 2048 future tokens; win0 = slot-0 object 3×3; win8 = object 3×3 at all 8 slots. Random k-dim expectation of the L2 share = k/1280. `complement` of the union = 1 − L2(union)."]
    for g in list(GROUPS) + [c for c in res["cells"]]:
        rec = res["scoring"][g]
        L += ["", f"### {g} — n clips {rec['n_clips']}, n blocks {rec['n_blocks']}, tokens all {rec['n_tokens_exact']['all']} (subsample {rec['n_tokens_subsample']['all']}), win0 {rec['n_tokens_exact']['win0']}; mean|d| all {rec['mean_abs_d']['all']:.3f} / win0 {rec['mean_abs_d']['win0']:.3f}", "",
              "| subspace (src|factor) | k | rand | all L2 · L1 [CI] | win0 L2 · L1 [CI] | win8 L2 · L1 |", "|---|---:|---:|---|---|---|"]
        for name, e in rec["subspaces"].items():
            if "shuffle" in name:
                L.append(f"| {name} | {e['k']} | | {e['all']['l2_exact']:.3f} · {e['all']['l1']:.3f} | {e['win0']['l2_exact']:.3f} · {e['win0']['l1']:.3f} | {e['win8']['l2_exact']:.3f} · {e['win8']['l1']:.3f} |"); continue
            if name.startswith("pca16"):
                L.append(f"| {name} | {e['k']} | {e['random_analytic']:.3f} | {e['all']['l2_exact']:.3f} · {e['all']['l1']:.3f} | {e['win0']['l2_exact']:.3f} · {e['win0']['l1']:.3f} | {e['win8']['l2_exact']:.3f} · {e['win8']['l1']:.3f} |"); continue
            src, fac = name.split("|"); c_all = res["boot_ci"].get(f"score|{g}|{src}|{fac}|all|l1") if g in GROUPS else None; c_w = res["boot_ci"].get(f"score|{g}|{src}|{fac}|win0|l1") if g in GROUPS else None
            L.append(f"| {name} | {e['k']} | {e['random_analytic']:.3f} | {e['all']['l2_exact']:.3f} · {e['all']['l1']:.3f} {ci(c_all) if c_all else ''} | {e['win0']['l2_exact']:.3f} · {e['win0']['l1']:.3f} {ci(c_w) if c_w else ''} | {e['win8']['l2_exact']:.3f} · {e['win8']['l1']:.3f} |")
            if fac == "all_factors_union":
                L.append(f"| {src}|complement of union | {1280 - e['k']} | | {e['complement_l2_exact']['all']:.3f} · {e['all']['l1_complement']:.3f} | {e['complement_l2_exact']['win0']:.3f} · {e['win0']['l1_complement']:.3f} | {e['complement_l2_exact']['win8']:.3f} · {e['win8']['l1_complement']:.3f} |")
    # ---- auto summary
    L += ["", "## Summary (auto-generated from the numbers above)", ""]
    M = res["pairwise"]["z7|ridge"]; pairs = sorted(((e["full"], k, e) for k, e in M.items()), reverse=True)
    L.append("Within z7 (ridge), pairs ranked by overlap (full · cross-half · random · shuf):")
    for v, k, e in pairs:
        L.append(f"- {k}: {v:.2f} · {e['cross_half']:.2f} · {e['random']['analytic']:.3f} · {e['shuffle_real_vs_shuf']:.2f}")
    L.append("\nSame factor across reps (ridge; full overlap, split-half reliabilities in parentheses):")
    for f in FACTORS:
        e = res["cross_rep"][f"{f}|ridge"]; L.append(f"- {f}: z7–p0 {e['z7-p0']['full']:.2f} ({e['z7-p0']['reliability'][0]:.2f},{e['z7-p0']['reliability'][1]:.2f}), z7–h8 {e['z7-h8']['full']:.2f}, p0–h8 {e['p0-h8']['full']:.2f}; random {e['z7-p0']['random']['analytic']:.3f}")
    L.append("\nScoring-distance L2 share (all future tokens, exact) of h8-fitted subspaces, visible / early / late:")
    for name in ["identity", "position", "env", "motion", "occluder", "hidden", "time", "all_factors_union"]:
        vals = [res["scoring"][g]["subspaces"][f"h8|{name}"] for g in GROUPS]; L.append(f"- {name} (k={vals[0]['k']}, rand {vals[0]['random_analytic']:.3f}): " + " / ".join(f"{v['all']['l2_exact']:.3f}" for v in vals) + "   (win0: " + " / ".join(f"{v['win0']['l2_exact']:.3f}" for v in vals) + ")")
    L += ["", "## What cannot be concluded", "",
          "- Linear subspaces only: a factor that is read out non-linearly, or spread over many low-variance directions, is under-estimated; ridge subspaces are lower bounds on 'where the factor lives'.",
          "- The ridge subspace of a k-class factor has k−1 dimensions by construction; overlap between subspaces of very different k is bounded by the smaller one and is compared to its own random baseline, not across pairs.",
          "- Object-window features (3×3 means) at the true position: the predictor may place the object elsewhere in late cells (§5-5), so p0/h8 rows in late cells are 'the scorer's window', not 'the object'.",
          "- Position variance comes almost entirely from the static cells (moving objects cross the centre at the boundary, x7 range 112–176 px); the position subspace is a static-cell subspace.",
          "- motion is confounded with scene layout (ramp present) and occluder/hidden with occluder presence by design (CLAUDE.md §8-5).",
          "- Scoring shares are descriptive decompositions of |p−h| over a fixed basis; they do not say which directions decide a pair's sign (that needs the matched-pair difference h_pos − h_imp, not p − h).",
          "- Bootstrap CIs resample blocks with λ fixed; λ selection noise is not included. The exact all-token L2 shares carry no CI (subsample L1/L2 do).", "",
          "## 재현", "", "```", f"cd {ROOT}", f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python {res['script']} --blocks {res['n_blocks_per_cell']} --boot {res['n_boot']}", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
