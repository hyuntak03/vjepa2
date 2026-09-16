#!/usr/bin/env python3
"""S2 — kinematic factor SUBSPACES (position / velocity / acceleration / scenario / time signature / object-vs-background) in the
predictor's input token (z16 = ctx_masked, tubelet 7, 3x3 mean at the true object position), how far they are separable from each
other, whether the background-window velocity code is the same subspace as the object-window one, whether p / h / z32 keep the same
subspaces, and how much of the scoring distance ||p - h||^2 (and its L1 analogue) on the future tokens lies in them.  RollOut_v2 plausible.

Data      : data_csv/rollout_v2/index.csv, plausible == 1, scenarios flat_v flat_a fall ledge arc ramp_a; N_PER clips per scenario, seed 0
            (validity: tubelets 5..15 in frame and inside the 288-px render; velocity is confounded with position inside ramp_a / arc / ledge).
Caches    : rollout_v2_z16_vith/ctx_masked.npy (z16, 8 tubelets, predictor input, LN'd) · rollout_v2_vith/{predictor,target}.npy (p raw, h LN'd,
            16 tubelets) · rollout_v2_ctx32_vith/isolated_ctx_0_32.npy (z32, 16 tubelets, LN'd).  Token index = tubelet*256 + row*16 + col.
Labels    : cells (1 cell = 18 px); tubelet position = mean of its 2 samples; pos_T = (x_T, y_T), v_T = pos_T - pos_{T-1}, a_T = v_T - v_{T-1};
            vel_resid = v_T minus its within-scenario quadratic-in-position fit (velocity variation not explained by position);
            scenario = 6 classes; time = tubelet index (8 classes) of the window; objbg = object 3x3 vs far (Chebyshev >= 4 cells from the
            object at every context tubelet) 3x3 at tubelet 7.  T = 7 for z16 reps, T = 8 for p / h / z32 reps.
Reps      : z16_obj (t7, obj 3x3) · z16_bg (t7, far 3x3) · p0 (slot 0 = t8, window at pos8) · h8 (t8 at pos8) · z32_8 (t8 at pos8) ·
            extra rows h7 / z32_7 (t7 at pos7).  Time reps: the same windows at every tubelet (z16: t0..7; p/h/z32: t8..15).
Subspace  : column span (SVD-orthonormalised, rank-revealing) of the closed-form ridge readout W (D x k) fitted on the centered raw features
            (no per-dim scaling, so W lives in the token coordinates used for variance / scoring), lambda = rel * tr(X'X)/D, rel chosen per
            factor on a block-held-out inner validation fold (25 %) inside each of 5 outer block folds; the final W uses the median rel.
            Categorical factors also get the between-class-scatter (S_b top k-1) and LDA ((S_w + gI)^-1 S_b top k-1) variants.
Overlap   : mean cos^2 of the principal angles between two subspaces (= ||Qa'Qb||_F^2 / min(ka, kb); k1, k2 always stated).
            Lower bound: random subspaces of the same (k1, k2) in D = 1280 (analytic max(k1,k2)/D and 200 empirical draws).
            Upper bound: split-half (two disjoint block halves, 10 random splits) of each factor; ceiling of a pair = sqrt(sh_a * sh_b).
Shares    : variance share = tr(Q' C Q)/tr(C) on the centered clip-level features of a rep (random-k expectation k/D; PCA top-k ceiling);
            scoring-distance share = tr(Q' G Q)/tr(G) with G = sum over the scorer's future tokens of d d', d = p - h[8:16] (exact, all 2048
            tokens per clip and the 3x3 object windows), and the L1 analogue mean|Q Q' d| / mean|d| on stored tokens (object windows exact;
            'all' = 48 random future tokens per clip).  Random-k and PCA-top-k baselines per (scenario, window).
CIs       : block bootstrap (clips resampled with replacement within scenario; block_id = clip for plausible-only RollOut_v2), B = --boot,
            refitting every readout with its fixed lambda; R^2 / accuracy CIs resample the out-of-fold predictions.  No GPU, no model.
Outputs   : z_research/context_encoder_analysis/exp_results/ctxenc_state_subspaces_kinematics/{results.json, RESULTS.md, features.npz,
            scoring_tokens.npz}

  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2 && \
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_state_subspaces_kinematics.py --n-per 300 --boot 200
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "16"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "16"); os.environ.setdefault("MKL_NUM_THREADS", "16")
import argparse, csv, json, re, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
INDEX = ROOT / "data_csv/rollout_v2/index.csv"
CACHE_Z16 = Path("/local_datasets/world/world_analysis/cache/rollout_v2_z16_vith")
CACHE_HP = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
CACHE_Z32 = Path("/local_datasets/world/world_analysis/cache/rollout_v2_ctx32_vith")
OUT_DEFAULT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_state_subspaces_kinematics"
SCEN = ["flat_v", "flat_a", "fall", "ledge", "arc", "ramp_a"]
SEED, K_FOLD, N_INNER = 0, 5, 4
LAM_REL = [float(v) for v in np.logspace(-3, 3, 13)]
CELL, RES, D = 18.0, 288, 1280
N_SUB_TOK = 48              # random future tokens per clip stored for the 'all' L1 / bootstrap shares
N_SPLIT_HALF = 10
N_RANDOM = 200
DEGEN_STD = 0.005           # a_T std (cells/tubelet^2, 2-D norm) below which a scenario's acceleration label is marked degenerate
CONT = {"position": ["x", "y"], "velocity": ["vx", "vy"], "accel": ["ax", "ay"], "vel_resid": ["vx_r", "vy_r"]}
CAT = {"scenario": SCEN, "time": [f"t{t}" for t in range(8)], "objbg": ["object", "background"]}
CLIP_REPS = {"z16_obj": 7, "z16_bg": 7, "p0": 8, "h8": 8, "z32_8": 8, "h7": 7, "z32_7": 7}          # rep -> label time T
MAIN_REPS = ["z16_obj", "z16_bg", "p0", "h8", "z32_8"]
TIME_REPS = {"z16_obj": "Zobj", "z16_bg": "Zbg", "p": "Pobj", "h": "Hobj", "z32": "Z32obj"}


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def n3(x, y):
    """copied verbatim from z_research/scripts/analysis/v11_token_object_test.py — 3x3 token window around px (x, y)."""
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------- clips, labels
def load_clips(n_per):
    rows = [r for r in csv.DictReader(INDEX.open()) if r["plausible"] == "1" and r["scenario"] in SCEN]
    rng = np.random.default_rng(SEED); keep = []; counts = {}
    for scn in SCEN:
        rs = [r for r in rows if r["scenario"] == scn]; ok = []
        for r in rs:
            px, py, inf = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"]), arr(r["in_frame_by_sample"])
            tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)
            if inf[10:32].min() > 0 and (tx[5:16] >= 0).all() and (tx[5:16] < RES).all() and (ty[5:16] >= 0).all() and (ty[5:16] < RES).all():
                ok.append(r)
        sel = rng.choice(len(ok), min(n_per, len(ok)), replace=False); keep += [ok[i] for i in sorted(sel)]
        counts[scn] = dict(plausible=len(rs), valid=len(ok), sampled=int(len(sel))); log(f"  {scn}: {counts[scn]}")
    return keep, counts


def positions_of(r):
    px, py = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])
    tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)
    return np.stack([tx, ty], 1)                                          # (16, 2) px per tubelet


def bg_cell(pos_px, rng):
    """token cell (c, r) with Chebyshev distance >= 4 from the object at every context tubelet 0..7 (else the farthest); centre px."""
    cells = pos_px[:8] // CELL
    cand = [(c, r) for r in range(1, 15) for c in range(1, 15)]
    dist = np.array([min(max(abs(c - cx), abs(r - cy)) for cx, cy in cells) for c, r in cand])
    good = np.where(dist >= 4)[0] if (dist >= 4).any() else np.where(dist == dist.max())[0]
    c, r = cand[good[rng.integers(len(good))]]; return c * CELL + 9, r * CELL + 9, float(dist[good[0]] if (dist >= 4).any() else dist.max())


def make_labels(POS):
    """POS (n, 16, 2) cells. returns dict T -> {position, velocity, accel} arrays (n, 2)."""
    out = {}
    for T in (7, 8):
        v = POS[:, T] - POS[:, T - 1]; a = v - (POS[:, T - 1] - POS[:, T - 2])
        out[T] = {"position": POS[:, T].copy(), "velocity": v, "accel": a}
    return out


def vel_resid(pos, vel, scen):
    """velocity minus its within-scenario least-squares fit on [1, x, y, x^2, y^2, xy]."""
    res = np.zeros_like(vel)
    for scn in SCEN:
        m = scen == scn; x, y = pos[m, 0], pos[m, 1]; A = np.stack([np.ones(m.sum()), x, y, x ** 2, y ** 2, x * y], 1)
        coef, *_ = np.linalg.lstsq(A, vel[m], rcond=None); res[m] = vel[m] - A @ coef
    return res


# ----------------------------------------------------------------------------------------------- extraction
def extract(clips, rng):
    metas = {k: json.loads((c / "meta.json").read_text()) for k, c in (("z16", CACHE_Z16), ("hp", CACHE_HP), ("z32", CACHE_Z32))}
    assert metas["z16"]["base_counts"]["ctx_masked"] == 2048 and metas["hp"]["base_counts"]["target"] == 4096 and metas["hp"]["base_counts"]["predictor"] == 2048
    assert metas["z32"]["base_counts"]["isolated_ctx:0_32"] == 4096
    row = {k: {v: i for i, v in enumerate(m["video_ids"])} for k, m in metas.items()}
    Z = np.load(CACHE_Z16 / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE_HP / "target.npy", mmap_mode="r")
    P = np.load(CACHE_HP / "predictor.npy", mmap_mode="r"); Z32 = np.load(CACHE_Z32 / "isolated_ctx_0_32.npy", mmap_mode="r")
    n = len(clips); f32 = np.float32
    F = {k: np.zeros((n, 8, D), f32) for k in ("Zobj", "Zbg", "Pobj", "Hobj", "Z32obj")}
    F.update({k: np.zeros((n, D), f32) for k in ("Ht7", "Z32t7")})
    POS = np.zeros((n, 16, 2)); win_n = np.zeros((n, 16), np.int8); bg_info = np.zeros((n, 3))
    Dobj = np.zeros((n, 8, 9, D), np.float16); Dobj_mask = np.zeros((n, 8, 9), bool); Dsub = np.zeros((n, N_SUB_TOK, D), np.float16); sub_idx = np.zeros((n, N_SUB_TOK), np.int16)
    G = {w: {s: np.zeros((D, D)) for s in SCEN} for w in ("all", "obj")}; L1 = {w: {s: 0.0 for s in SCEN} for w in ("all", "obj")}; NT = {w: {s: 0 for s in SCEN} for w in ("all", "obj")}
    scen = np.array([r["scenario"] for r in clips])
    order = np.argsort([row["z16"][r["video_id"]] for r in clips]); t0 = time.time()
    for j, i in enumerate(order):
        r = clips[i]; pos_px = positions_of(r); POS[i] = pos_px / CELL; wins = [n3(*pos_px[t]) for t in range(16)]; win_n[i] = [len(w) for w in wins]
        bx, by, bd = bg_cell(pos_px, rng); bwin = n3(bx, by); bg_info[i] = (bx, by, bd)
        z = np.asarray(Z[row["z16"][r["video_id"]]], f32).reshape(8, 256, D)
        for t in range(8):
            F["Zobj"][i, t] = z[t, wins[t]].mean(0) if wins[t] else 0.0; F["Zbg"][i, t] = z[t, bwin].mean(0)   # empty window (object off-screen before t5) -> 0, masked out below
        z32 = np.asarray(Z32[row["z32"][r["video_id"]]], f32).reshape(16, 256, D); F["Z32t7"][i] = z32[7, wins[7]].mean(0)
        h = np.asarray(H[row["hp"][r["video_id"]]], f32).reshape(16, 256, D); F["Ht7"][i] = h[7, wins[7]].mean(0)
        p = np.asarray(P[row["hp"][r["video_id"]]], f32).reshape(8, 256, D)
        for s in range(8):
            w = wins[8 + s]; F["Z32obj"][i, s] = z32[8 + s, w].mean(0); F["Hobj"][i, s] = h[8 + s, w].mean(0); F["Pobj"][i, s] = p[s, w].mean(0)
        d = p - h[8:]                                                   # (8, 256, D): exactly the scorer's difference (h already LN'd)
        dobj = []
        for s in range(8):
            w = wins[8 + s]; Dobj[i, s, :len(w)] = d[s, w]; Dobj_mask[i, s, :len(w)] = True; dobj.append(d[s, w])
        dobj = np.concatenate(dobj, 0); dall = d.reshape(-1, D)
        si = np.sort(rng.choice(2048, N_SUB_TOK, replace=False)); sub_idx[i] = si; Dsub[i] = dall[si]
        s_ = scen[i]; G["all"][s_] += (dall.T @ dall).astype(np.float64); G["obj"][s_] += (dobj.T @ dobj).astype(np.float64)
        L1["all"][s_] += float(np.abs(dall).sum()); L1["obj"][s_] += float(np.abs(dobj).sum()); NT["all"][s_] += len(dall); NT["obj"][s_] += len(dobj)
        if j % 200 == 0:
            log(f"    extract {j}/{n} {time.time() - t0:.0f}s")
    return F, POS, win_n, bg_info, dict(Dobj=Dobj, Dobj_mask=Dobj_mask, Dsub=Dsub, sub_idx=sub_idx), G, L1, NT


# ----------------------------------------------------------------------------------------------- ridge machinery
def folds_by_block(blocks, strata, k, rng):
    b2s = {}
    for b, s in zip(blocks, strata):
        b2s.setdefault(b, s)
    fold_of = {}
    for s in sorted(set(b2s.values())):
        bs = sorted(b for b in b2s if b2s[b] == s); rng.shuffle(bs)
        for i, b in enumerate(bs):
            fold_of[b] = i % k
    return np.array([fold_of[b] for b in blocks])


class Eig:
    """one eigendecomposition of the centered Gram X'X, reused for every lambda and every label set on the same rows."""
    def __init__(self, X):
        self.mu = X.mean(0); Xc = (X - self.mu).astype(np.float32); C = (Xc.T @ Xc).astype(np.float64)
        self.tr = float(np.trace(C)); self.s, self.V = eigh(C, driver="evd"); self.Xc = Xc

    def W(self, Yc, rel):
        XtY = (self.Xc.T @ Yc.astype(np.float32)).astype(np.float64); lam = rel * self.tr / D
        return self.V @ ((self.V.T @ XtY) / (self.s + lam)[:, None])


class Readout:
    def __init__(self, mu, ym, W, rel): self.mu, self.ym, self.W, self.rel = mu, ym, W, rel

    def predict(self, X, mu=None): return (X - (self.mu if mu is None else mu)) @ self.W + self.ym


def fit_fixed(X, Y, rel):
    """closed-form ridge with a fixed relative lambda (Cholesky); used for final subspaces, split-half and bootstrap."""
    mu = X.mean(0); Xc = (X - mu).astype(np.float32); C = (Xc.T @ Xc).astype(np.float64); lam = rel * np.trace(C) / D
    ym = Y.mean(0); XtY = (Xc.T @ (Y - ym).astype(np.float32)).astype(np.float64)
    W = cho_solve(cho_factor(C + lam * np.eye(D)), XtY); return Readout(mu, ym, W, rel)


def select_rel(E, Y_tr, X_va, Y_va, kind):
    """one rel per factor: continuous -> sum over columns of MSE / var; categorical -> accuracy (ties -> larger lambda)."""
    ym = Y_tr.mean(0); best = None
    for rel in LAM_REL:
        pred = (X_va - E.mu) @ E.W(Y_tr - ym, rel) + ym
        crit = -(((pred - Y_va) ** 2).mean(0) / (Y_va.var(0) + 1e-12)).sum() if kind == "cont" else float((pred.argmax(1) == Y_va.argmax(1)).mean())
        if best is None or crit >= best[0] - 1e-12:
            best = (crit, rel)
    return best[1]


def cv_readouts(X, factors, blocks, strata, seed):
    """5 outer block folds; per fold one Eig on the inner-train rows (rel selection) and one on the full train rows (refit).
    factors: name -> (Y, kind).  returns oof predictions per factor, fold Readouts per factor, fold ids, chosen rels."""
    rng = np.random.default_rng(seed); fold = folds_by_block(blocks, strata, K_FOLD, rng)
    oof = {f: np.zeros_like(Y) for f, (Y, _) in factors.items()}; models = {f: [] for f in factors}; rels = {f: [] for f in factors}
    for k in range(K_FOLD):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        inner = folds_by_block(blocks[tr], strata[tr], N_INNER, rng); itr, iva = tr[inner != 0], tr[inner == 0]
        E_in = Eig(X[itr]); E_tr = Eig(X[tr])
        for f, (Y, kind) in factors.items():
            rel = select_rel(E_in, Y[itr], X[iva], Y[iva], kind); rels[f].append(rel)
            ym = Y[tr].mean(0); m = Readout(E_tr.mu, ym, E_tr.W(Y[tr] - ym, rel), rel); models[f].append(m); oof[f][te] = m.predict(X[te])
    return oof, models, fold, rels


def median_rel(rs):
    lr = np.log(LAM_REL); return float(LAM_REL[int(np.argmin(np.abs(lr - np.median(np.log(rs)))))])


def r2(p, y, min_std=1e-6):
    return None if y.std() < min_std else float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


R2_SCEN_MIN_STD = 0.05      # per-scenario R^2 is reported only where the scenario's own label std >= this (cells); below it R^2 is noise


def cont_metrics(pred, Y, scen, cols):
    rec = {"pooled": {c: {"r2": r2(pred[:, j], Y[:, j]), "mae": float(np.abs(pred[:, j] - Y[:, j]).mean()), "label_std": float(Y[:, j].std())} for j, c in enumerate(cols)}, "n": int(len(Y)), "per_scenario": {}}
    for scn in SCEN:
        m = scen == scn
        rec["per_scenario"][scn] = {c: {"r2": r2(pred[m, j], Y[m, j], R2_SCEN_MIN_STD), "mae": float(np.abs(pred[m, j] - Y[m, j]).mean()), "label_std": float(Y[m, j].std())} for j, c in enumerate(cols)} | {"n": int(m.sum())}
    return rec


def cat_metrics(pred, Y, scen=None):
    yp, yt = pred.argmax(1), Y.argmax(1); k = Y.shape[1]
    bacc = float(np.mean([(yp[yt == c] == c).mean() for c in range(k) if (yt == c).any()]))
    rec = {"acc": float((yp == yt).mean()), "bacc": bacc, "chance": 1.0 / k, "n": int(len(yt))}
    if scen is not None:
        rec["per_scenario"] = {scn: {"acc": float((yp[scen == scn] == yt[scen == scn]).mean()), "n": int((scen == scn).sum())} for scn in SCEN}
    return rec


# ----------------------------------------------------------------------------------------------- subspaces and overlaps
def subspace(W, tol=1e-4):
    U, s, _ = np.linalg.svd(W, full_matrices=False); return U[:, s > tol * s[0]]


def scatter_lda(X, y, k):
    """between-class scatter top (k-1) and LDA (S_w shrinkage 0.1 tr/D) top (k-1) subspaces for integer classes y."""
    mu = X.mean(0); Sb = np.zeros((D, D)); Sw = np.zeros((D, D))
    for c in range(k):
        m = y == c
        if not m.any():
            continue
        mc = X[m].mean(0); Sb += m.sum() * np.outer(mc - mu, mc - mu); Xc = (X[m] - mc).astype(np.float32); Sw += (Xc.T @ Xc).astype(np.float64)
    s, V = eigh(Sb, driver="evd"); Q_sb = V[:, ::-1][:, :k - 1]
    g = 0.1 * np.trace(Sw) / D; s2, V2 = eigh(Sb, Sw + g * np.eye(D)); Q_lda = V2[:, ::-1][:, :k - 1]
    Q_lda, _ = np.linalg.qr(Q_lda); return Q_sb, Q_lda


def overlap(Qa, Qb):
    ka, kb = Qa.shape[1], Qb.shape[1]; return float((np.linalg.norm(Qa.T @ Qb) ** 2) / min(ka, kb))


def random_overlap(ka, kb, rng, n=N_RANDOM):
    v = []
    for _ in range(n):
        Qa, _ = np.linalg.qr(rng.standard_normal((D, ka))); Qb, _ = np.linalg.qr(rng.standard_normal((D, kb))); v.append(overlap(Qa, Qb))
    return {"mean": float(np.mean(v)), "p97.5": float(np.percentile(v, 97.5)), "analytic": max(ka, kb) / D, "n_draws": n}


def orth_union(*Qs):
    Q, _ = np.linalg.qr(np.concatenate(Qs, 1)); return Q


def var_share(Q, C):
    return float(np.trace(Q.T @ C @ Q) / np.trace(C))


def pca_ceiling(C, k):
    s = np.linalg.eigvalsh(C)[::-1]; return float(s[:k].sum() / s.sum())


def l2_share_tokens(Q, T):
    """T (m, D) float32 tokens; share of sum ||d||^2 inside span(Q)."""
    return float(((T @ Q.astype(np.float32)) ** 2).sum() / (T ** 2).sum())


def l1_share_tokens(Q, T):
    return float(np.abs((T @ Q.astype(np.float32)) @ Q.T.astype(np.float32)).mean() / np.abs(T).mean())


def boot_ci(v):
    v = np.asarray([x for x in v if x is not None], float); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per", type=int, default=300); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--reuse-features", action="store_true"); ap.add_argument("--out", default=str(OUT_DEFAULT)); a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(SEED)
    log("loading clips"); clips, counts = load_clips(a.n_per); n = len(clips)
    scen = np.array([r["scenario"] for r in clips]); vids = np.array([r["video_id"] for r in clips]); blocks = np.array([r["block_id"] for r in clips])
    fpath, tpath = OUT / "features.npz", OUT / "scoring_tokens.npz"
    if a.reuse_features and fpath.is_file() and tpath.is_file() and list(np.load(fpath, allow_pickle=True)["video_id"]) == list(vids):
        z = np.load(fpath, allow_pickle=True); F = {k: z[k] for k in ("Zobj", "Zbg", "Pobj", "Hobj", "Z32obj", "Ht7", "Z32t7")}; POS, win_n, bg_info = z["POS"], z["win_n"], z["bg_info"]
        G = {w: {s: z[f"G_{w}_{s}"] for s in SCEN} for w in ("all", "obj")}; L1 = json.loads(str(z["L1"])); NT = json.loads(str(z["NT"]))
        tz = np.load(tpath); TOK = {k: tz[k] for k in tz.files}; log("features + tokens from cache")
    else:
        log("extracting features"); F, POS, win_n, bg_info, TOK, G, L1, NT = extract(clips, rng)
        np.savez(fpath, video_id=vids, scenario=scen, block_id=blocks, POS=POS, win_n=win_n, bg_info=bg_info, L1=json.dumps(L1), NT=json.dumps(NT),
                 **F, **{f"G_{w}_{s}": G[w][s] for w in G for s in SCEN})
        np.savez(tpath, **TOK)
    log(f"n={n}  features ready {time.time() - t0:.0f}s")
    Dobj = TOK["Dobj"].astype(np.float32)[TOK["Dobj_mask"]]                                   # (m_obj, D) valid object-window tokens
    obj_clip = np.repeat(np.arange(n), TOK["Dobj_mask"].reshape(n, -1).sum(1)); Dsub = TOK["Dsub"].astype(np.float32).reshape(-1, D); sub_clip = np.repeat(np.arange(n), N_SUB_TOK)

    # ---- labels
    LAB = make_labels(POS)
    for T in (7, 8):
        LAB[T]["vel_resid"] = vel_resid(LAB[T]["position"], LAB[T]["velocity"], scen)
    onehot_scen = (scen[:, None] == np.array(SCEN)[None]).astype(float)
    label_stats = {}
    for T in (7, 8):
        label_stats[f"T{T}"] = {}
        for scn in SCEN + ["pooled"]:
            m = scen == scn if scn != "pooled" else np.ones(n, bool); rec = {"n": int(m.sum())}
            for f, cols in CONT.items():
                for j, c in enumerate(cols):
                    rec[f"{c}_std"] = float(LAB[T][f][m, j].std())
            a2 = LAB[T]["accel"][m]; rec["accel_norm_std"] = float(np.sqrt(a2.var(0).sum())); rec["accel_degenerate"] = bool(rec["accel_norm_std"] < DEGEN_STD)
            pv = LAB[T]["position"][m], LAB[T]["velocity"][m]
            rec["corr_pos_vel"] = {f"{pc}-{vc}": (float(np.corrcoef(pv[0][:, i], pv[1][:, j])[0, 1]) if pv[0][:, i].std() > 1e-6 and pv[1][:, j].std() > 1e-6 else None) for i, pc in enumerate("xy") for j, vc in enumerate(["vx", "vy"])}
            rec["vel_resid_frac_var"] = float(LAB[T]["vel_resid"][m].var(0).sum() / (LAB[T]["velocity"][m].var(0).sum() + 1e-12))
            label_stats[f"T{T}"][scn] = rec

    # ---- clip-level reps
    X_rep = {"z16_obj": F["Zobj"][:, 7], "z16_bg": F["Zbg"][:, 7], "p0": F["Pobj"][:, 0], "h8": F["Hobj"][:, 0], "z32_8": F["Z32obj"][:, 0], "h7": F["Ht7"], "z32_7": F["Z32t7"]}
    X_rep = {k: v.astype(np.float64) for k, v in X_rep.items()}
    strata = scen.copy()

    # ---- (a) readout quality, CV by block
    log("(a) readouts"); quality = {}; final = {}; oofs = {}; models = {}; folds = {}
    for rep, T in CLIP_REPS.items():
        factors = {f: (LAB[T][f], "cont") for f in CONT} | {"scenario": (onehot_scen, "cat")}
        oof, mdl, fold, rels = cv_readouts(X_rep[rep], factors, blocks, strata, SEED); oofs[rep], models[rep], folds[rep] = oof, mdl, fold
        quality[rep] = {}; final[rep] = {}
        for f, (Y, kind) in factors.items():
            q = cont_metrics(oof[f], Y, scen, CONT[f]) if kind == "cont" else cat_metrics(oof[f], Y, scen)
            q["rel_per_fold"] = rels[f]; q["rel_final"] = median_rel(rels[f]); quality[rep][f] = q
            final[rep][f] = fit_fixed(X_rep[rep], Y, q["rel_final"])
        log(f"    {rep} done {time.time() - t0:.0f}s")
    # time factor (rows = clip x tubelet) and objbg
    # rows = clip x tubelet (clip-major); z16_obj rows with an empty window (object off-screen at t < 5) are dropped
    t_full = np.tile(np.arange(8), n); onehot_full = np.eye(8)[t_full]
    tmask = {rep: ((win_n[:, :8] > 0).reshape(-1) if rep == "z16_obj" else np.ones(n * 8, bool)) for rep in TIME_REPS}
    comp = {rep: np.cumsum(tmask[rep]) - 1 for rep in TIME_REPS}                       # full row id -> compressed row id
    X_time = {rep: F[key].reshape(n * 8, D).astype(np.float64)[tmask[rep]] for rep, key in TIME_REPS.items()}
    onehot_t = {rep: onehot_full[tmask[rep]] for rep in TIME_REPS}; t_idx = {rep: t_full[tmask[rep]] for rep in TIME_REPS}
    blocks_t = {rep: np.repeat(blocks, 8)[tmask[rep]] for rep in TIME_REPS}; strata_t = {rep: np.repeat(scen, 8)[tmask[rep]] for rep in TIME_REPS}
    for rep in TIME_REPS:
        oof, mdl, fold, rels = cv_readouts(X_time[rep], {"time": (onehot_t[rep], "cat")}, blocks_t[rep], strata_t[rep], SEED)
        q = cat_metrics(oof["time"], onehot_t[rep], strata_t[rep]); q["rel_per_fold"] = rels["time"]; q["rel_final"] = median_rel(rels["time"])
        q["per_tubelet_acc"] = {f"t{t}": float((oof["time"].argmax(1)[t_idx[rep] == t] == t).mean()) for t in range(8)}
        q["n_per_tubelet"] = {f"t{t}": int((t_idx[rep] == t).sum()) for t in range(8)}
        quality.setdefault(f"time:{rep}", {})["time"] = q; final.setdefault(f"time:{rep}", {})["time"] = fit_fixed(X_time[rep], onehot_t[rep], q["rel_final"])
    X_ob = np.concatenate([X_rep["z16_obj"], X_rep["z16_bg"]], 0); Y_ob = np.eye(2)[np.r_[np.zeros(n, int), np.ones(n, int)]]; blocks_ob = np.tile(blocks, 2); strata_ob = np.tile(scen, 2)
    oof, mdl, fold, rels = cv_readouts(X_ob, {"objbg": (Y_ob, "cat")}, blocks_ob, strata_ob, SEED)
    q = cat_metrics(oof["objbg"], Y_ob, np.tile(scen, 2)); q["rel_per_fold"] = rels["objbg"]; q["rel_final"] = median_rel(rels["objbg"])
    quality["objbg:z16"] = {"objbg": q}; final["objbg:z16"] = {"objbg": fit_fixed(X_ob, Y_ob, q["rel_final"])}
    log(f"    time / objbg done {time.time() - t0:.0f}s")

    # ---- cross-application of fold readouts (rep A train fold -> rep B test fold), recentered with B's train-fold mean; also raw
    log("cross-application"); cross = {}
    for f in ("position", "velocity"):
        cross[f] = {}
        for A in CLIP_REPS:
            for B in CLIP_REPS:
                if A == B:
                    continue
                Y = LAB[CLIP_REPS[B]][f]; pr_rc, pr_raw = np.zeros_like(Y), np.zeros_like(Y)
                for k in range(K_FOLD):
                    tr, te = folds[A] != k, folds[A] == k; m = models[A][f][k]
                    pr_rc[te] = m.predict(X_rep[B][te], mu=X_rep[B][tr].mean(0)); pr_raw[te] = m.predict(X_rep[B][te])
                cross[f][f"{A}->{B}"] = {"recentered": cont_metrics(pr_rc, Y, scen, CONT[f]), "raw": {c: {"r2": r2(pr_raw[:, j], Y[:, j])} for j, c in enumerate(CONT[f])},
                                         "label_time_of_B": CLIP_REPS[B], "n": int(n)}
    # objbg cross-application for the time factor (object-window time readout applied to background windows and vice versa)
    cross["time"] = {}
    cm = tmask["z16_obj"]; Xt_c = {rep: F[TIME_REPS[rep]].reshape(n * 8, D).astype(np.float64)[cm] for rep in ("z16_obj", "z16_bg")}   # rows valid in both
    for A, B in (("z16_obj", "z16_bg"), ("z16_bg", "z16_obj")):
        oofA, mdlA, foldA, _ = cv_readouts(Xt_c[A], {"time": (onehot_full[cm], "cat")}, np.repeat(blocks, 8)[cm], np.repeat(scen, 8)[cm], SEED)
        pr = np.zeros_like(onehot_full[cm])
        for k in range(K_FOLD):
            tr, te = foldA != k, foldA == k; pr[te] = mdlA["time"][k].predict(Xt_c[B][te], mu=Xt_c[B][tr].mean(0))
        cross["time"][f"{A}->{B}"] = cat_metrics(pr, onehot_full[cm]) | {"recentered": True, "self_acc_same_rows": cat_metrics(oofA["time"], onehot_full[cm])["acc"]}

    # ---- subspaces (ridge) + categorical variants
    Q = {}
    for rep in CLIP_REPS:
        for f in list(CONT) + ["scenario"]:
            Q[(rep, f)] = subspace(final[rep][f].W)
    for rep in TIME_REPS:
        Q[(f"time:{rep}", "time")] = subspace(final[f"time:{rep}"]["time"].W)
    Q[("objbg:z16", "objbg")] = subspace(final["objbg:z16"]["objbg"].W)
    Qsb, Qlda = {}, {}
    for rep in MAIN_REPS:
        Qsb[(rep, "scenario")], Qlda[(rep, "scenario")] = scatter_lda(X_rep[rep], onehot_scen.argmax(1), 6)
    for rep in ("z16_obj", "h"):
        Qsb[(f"time:{rep}", "time")], Qlda[(f"time:{rep}", "time")] = scatter_lda(X_time[rep], t_idx[rep], 8)
    Qsb[("objbg:z16", "objbg")], Qlda[("objbg:z16", "objbg")] = scatter_lda(X_ob, Y_ob.argmax(1), 2)
    ks = {f"{k[0]}|{k[1]}": int(v.shape[1]) for k, v in Q.items()}
    variant_agreement = {f"{k[0]}|{k[1]}": {"ridge_vs_scatter": overlap(Q[k], Qsb[k]), "ridge_vs_lda": overlap(Q[k], Qlda[k]), "scatter_vs_lda": overlap(Qsb[k], Qlda[k]), "k": [int(Q[k].shape[1]), int(Qsb[k].shape[1])]} for k in Qsb}

    # ---- split-half (upper bound) for every factor subspace used in an overlap
    log("split-half"); rs = np.random.default_rng(SEED + 1); split_half = {}

    def sh(fit_fn, key, blocks_, strata_):
        vals = []
        for _ in range(N_SPLIT_HALF):
            f2 = folds_by_block(blocks_, strata_, 2, rs); vals.append(overlap(fit_fn(f2 == 0), fit_fn(f2 == 1)))
        split_half[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n_splits": N_SPLIT_HALF}

    for rep in CLIP_REPS:
        for f in list(CONT) + ["scenario"]:
            Y = LAB[CLIP_REPS[rep]][f] if f in CONT else onehot_scen; rel = quality[rep][f]["rel_final"]
            sh(lambda m, X=X_rep[rep], Y=Y, rel=rel: subspace(fit_fixed(X[m], Y[m], rel).W), f"{rep}|{f}", blocks, strata)
    for rep in TIME_REPS:
        rel = quality[f"time:{rep}"]["time"]["rel_final"]
        sh(lambda m, X=X_time[rep], Y=onehot_t[rep], rel=rel: subspace(fit_fixed(X[m], Y[m], rel).W), f"time:{rep}|time", blocks_t[rep], strata_t[rep])
    rel = quality["objbg:z16"]["objbg"]["rel_final"]
    sh(lambda m: subspace(fit_fixed(X_ob[m], Y_ob[m], rel).W), "objbg:z16|objbg", blocks_ob, strata_ob)
    rand_cache = {}

    def rand(ka, kb):
        key = (min(ka, kb), max(ka, kb))
        if key not in rand_cache:
            rand_cache[key] = random_overlap(key[0], key[1], np.random.default_rng(SEED + 2))
        return rand_cache[key]

    def pair(qa, qb, ka_key, kb_key, variant="ridge"):
        return {"overlap": overlap(qa, qb), "k": [int(qa.shape[1]), int(qb.shape[1])], "random": rand(qa.shape[1], qb.shape[1]),
                "split_half_a": split_half.get(ka_key, {}).get("mean"), "split_half_b": split_half.get(kb_key, {}).get("mean"),
                "ceiling_sqrt": (float(np.sqrt(split_half[ka_key]["mean"] * split_half[kb_key]["mean"])) if ka_key in split_half and kb_key in split_half else None), "variant": variant}

    # ---- (b) pairwise within z16_obj
    log("(b)(c)(d) overlaps")
    fac_b = {"position": Q[("z16_obj", "position")], "velocity": Q[("z16_obj", "velocity")], "vel_resid": Q[("z16_obj", "vel_resid")], "accel": Q[("z16_obj", "accel")],
             "scenario": Q[("z16_obj", "scenario")], "time": Q[("time:z16_obj", "time")], "objbg": Q[("objbg:z16", "objbg")]}
    key_b = {"position": "z16_obj|position", "velocity": "z16_obj|velocity", "vel_resid": "z16_obj|vel_resid", "accel": "z16_obj|accel", "scenario": "z16_obj|scenario", "time": "time:z16_obj|time", "objbg": "objbg:z16|objbg"}
    names = list(fac_b); within = {}
    for i, fa in enumerate(names):
        for fb in names[i + 1:]:
            within[f"{fa}~{fb}"] = pair(fac_b[fa], fac_b[fb], key_b[fa], key_b[fb])
    fac_sb = {"scenario": Qsb[("z16_obj", "scenario")], "time": Qsb[("time:z16_obj", "time")], "objbg": Qsb[("objbg:z16", "objbg")]}
    for fa in ("position", "velocity"):
        for fb, qb in fac_sb.items():
            within[f"{fa}~{fb}(scatter)"] = pair(fac_b[fa], qb, key_b[fa], key_b[fb], "scatter")
    for i, fa in enumerate(list(fac_sb)):
        for fb in list(fac_sb)[i + 1:]:
            within[f"{fa}~{fb}(scatter)"] = pair(fac_sb[fa], fac_sb[fb], key_b[fa], key_b[fb], "scatter")
    # same pairs in h8 (the encoder that saw the future) for contrast
    within_h = {}
    fac_h = {"position": Q[("h8", "position")], "velocity": Q[("h8", "velocity")], "scenario": Q[("h8", "scenario")], "time": Q[("time:h", "time")]}
    key_h = {"position": "h8|position", "velocity": "h8|velocity", "scenario": "h8|scenario", "time": "time:h|time"}
    for i, fa in enumerate(list(fac_h)):
        for fb in list(fac_h)[i + 1:]:
            within_h[f"{fa}~{fb}"] = pair(fac_h[fa], fac_h[fb], key_h[fa], key_h[fb])

    # ---- (c) object-window vs background-window subspaces
    objbg = {f: pair(Q[("z16_obj", f)], Q[("z16_bg", f)], f"z16_obj|{f}", f"z16_bg|{f}") for f in list(CONT) + ["scenario"]}
    objbg["time"] = pair(Q[("time:z16_obj", "time")], Q[("time:z16_bg", "time")], "time:z16_obj|time", "time:z16_bg|time")
    objbg["velocity(bg)~position(bg)"] = pair(Q[("z16_bg", "velocity")], Q[("z16_bg", "position")], "z16_bg|velocity", "z16_bg|position")
    objbg["velocity(bg)~scenario(bg)"] = pair(Q[("z16_bg", "velocity")], Q[("z16_bg", "scenario")], "z16_bg|velocity", "z16_bg|scenario")
    objbg["velocity(bg)~time(bg)"] = pair(Q[("z16_bg", "velocity")], Q[("time:z16_bg", "time")], "z16_bg|velocity", "time:z16_bg|time")
    objbg["velocity(obj)~scenario(bg)"] = pair(Q[("z16_obj", "velocity")], Q[("z16_bg", "scenario")], "z16_obj|velocity", "z16_bg|scenario")
    objbg["velocity(bg)~scenario(obj)"] = pair(Q[("z16_bg", "velocity")], Q[("z16_obj", "scenario")], "z16_bg|velocity", "z16_obj|scenario")

    # ---- (d) cross-representation
    cross_rep = {}
    for f in ("position", "velocity", "vel_resid", "scenario"):
        cross_rep[f] = {}
        for A in ("z16_obj",):
            for B in ("p0", "h8", "z32_8", "h7", "z32_7", "z16_bg"):
                cross_rep[f][f"{A}~{B}"] = pair(Q[(A, f)], Q[(B, f)], f"{A}|{f}", f"{B}|{f}")
        for A, B in (("p0", "h8"), ("p0", "z32_8"), ("h8", "z32_8"), ("h7", "h8"), ("z32_7", "z32_8"), ("h7", "z32_7")):
            cross_rep[f][f"{A}~{B}"] = pair(Q[(A, f)], Q[(B, f)], f"{A}|{f}", f"{B}|{f}")
    cross_rep["time"] = {f"{A}~{B}": pair(Q[(f"time:{A}", "time")], Q[(f"time:{B}", "time")], f"time:{A}|time", f"time:{B}|time") for A, B in (("z16_obj", "p"), ("z16_obj", "h"), ("z16_obj", "z32"), ("p", "h"), ("p", "z32"), ("h", "z32"))}

    # ---- lambda sensitivity of the headline overlaps (subspace directions of an under-determined ridge depend on lambda)
    log("lambda sensitivity"); lam_sens = {}
    rs2 = np.random.default_rng(SEED + 5)
    for rel in (0.01, 1.0, 100.0):
        Qr = {}; shr = {}
        for rep, fs in (("z16_obj", ["position", "velocity", "vel_resid"]), ("z16_bg", ["velocity"]), ("p0", ["position", "velocity"]), ("h8", ["position", "velocity"]), ("z32_8", ["velocity"])):
            for f in fs:
                Y = LAB[CLIP_REPS[rep]][f]; Qr[(rep, f)] = subspace(fit_fixed(X_rep[rep], Y, rel).W)
                vals = []
                for _ in range(4):
                    f2 = folds_by_block(blocks, strata, 2, rs2); vals.append(overlap(subspace(fit_fixed(X_rep[rep][f2 == 0], Y[f2 == 0], rel).W), subspace(fit_fixed(X_rep[rep][f2 == 1], Y[f2 == 1], rel).W)))
                shr[(rep, f)] = float(np.mean(vals))
        pairs = {"z16_obj: position~velocity": (("z16_obj", "position"), ("z16_obj", "velocity")), "z16_obj: position~vel_resid": (("z16_obj", "position"), ("z16_obj", "vel_resid")),
                 "velocity: z16_obj~z16_bg": (("z16_obj", "velocity"), ("z16_bg", "velocity")), "velocity: z16_obj~p0": (("z16_obj", "velocity"), ("p0", "velocity")),
                 "velocity: z16_obj~h8": (("z16_obj", "velocity"), ("h8", "velocity")), "velocity: p0~h8": (("p0", "velocity"), ("h8", "velocity")), "velocity: z16_obj~z32_8": (("z16_obj", "velocity"), ("z32_8", "velocity")),
                 "position: z16_obj~p0": (("z16_obj", "position"), ("p0", "position")), "position: z16_obj~h8": (("z16_obj", "position"), ("h8", "position")), "position: p0~h8": (("p0", "position"), ("h8", "position"))}
        lam_sens[str(rel)] = {k: {"overlap": overlap(Qr[a], Qr[b]), "split_half_a": shr[a], "split_half_b": shr[b], "ceiling_sqrt": float(np.sqrt(shr[a] * shr[b])), "k": [int(Qr[a].shape[1]), int(Qr[b].shape[1])]} for k, (a, b) in pairs.items()}

    # ---- variance shares
    log("variance / scoring shares"); vshare = {}
    for rep in MAIN_REPS:
        Xc = X_rep[rep] - X_rep[rep].mean(0); C = Xc.T @ Xc; vshare[rep] = {"total_var": float(np.trace(C) / n)}
        for f in list(CONT) + ["scenario"]:
            k = Q[(rep, f)].shape[1]; vshare[rep][f] = {"share": var_share(Q[(rep, f)], C), "k": k, "random_expect": k / D, "pca_topk": pca_ceiling(C, k)}
        qt = Q[(f"time:{TIME_REPS_INV[rep]}", "time")] if rep in TIME_REPS_INV else None
        if qt is not None:
            vshare[rep]["time"] = {"share": var_share(qt, C), "k": int(qt.shape[1]), "random_expect": qt.shape[1] / D, "pca_topk": pca_ceiling(C, qt.shape[1])}
        if rep == "z16_obj":
            qo = Q[("objbg:z16", "objbg")]; vshare[rep]["objbg"] = {"share": var_share(qo, C), "k": 1, "random_expect": 1 / D, "pca_topk": pca_ceiling(C, 1)}
            qu = orth_union(Q[(rep, "position")], Q[(rep, "velocity")]); vshare[rep]["union_pos_vel"] = {"share": var_share(qu, C), "k": int(qu.shape[1]), "random_expect": qu.shape[1] / D, "pca_topk": pca_ceiling(C, qu.shape[1])}

    # ---- (e) scoring-distance shares.  subspaces fitted on h8 / p0 / z16_obj; time from the corresponding time rep
    def share_sets(src):
        trep = {"h8": "h", "p0": "p", "z16_obj": "z16_obj"}[src]
        S = {"position": Q[(src, "position")], "velocity": Q[(src, "velocity")], "accel": Q[(src, "accel")], "scenario": Q[(src, "scenario")], "time": Q[(f"time:{trep}", "time")]}
        S["union_pos_vel"] = orth_union(S["position"], S["velocity"]); S["union_kin_time_scen"] = orth_union(S["position"], S["velocity"], S["scenario"], S["time"])
        return S

    Gpool = {w: sum(G[w][s] for s in SCEN) for w in G}; scoring = {"exact_gram": {}, "magnitude": {}}
    rr = np.random.default_rng(SEED + 3)
    TOKS = {"obj": (Dobj, obj_clip), "all": (Dsub, sub_clip)}
    masks = {w: {scn: (np.ones(len(TOKS[w][0]), bool) if scn == "pooled" else (scen[TOKS[w][1]] == scn)) for scn in SCEN + ["pooled"]} for w in TOKS}
    ev_all = {}
    for w in ("all", "obj"):
        for scn in SCEN + ["pooled"]:
            Gw = Gpool[w] if scn == "pooled" else G[w][scn]; nt = sum(NT[w].values()) if scn == "pooled" else NT[w][scn]; l1 = sum(L1[w].values()) if scn == "pooled" else L1[w][scn]
            scoring["magnitude"][f"{w}|{scn}"] = {"n_tokens": int(nt), "mean_abs_d": l1 / nt / D, "mean_sq_norm_d": float(np.trace(Gw) / nt), "n_clips": int(n if scn == "pooled" else (scen == scn).sum())}
            ev_all[(w, scn)] = np.linalg.eigvalsh(Gw)[::-1]

    def per_token(q, T):
        """per-token squared projection norm and per-token mean |Q Q' d| (one pass over the stored tokens)."""
        qf = q.astype(np.float32); pq = T @ qf; return (pq ** 2).sum(1), np.abs(pq @ qf.T).mean(1)

    tok_sq = {w: (TOKS[w][0] ** 2).sum(1) for w in TOKS}; tok_l1 = {w: np.abs(TOKS[w][0]).mean(1) for w in TOKS}
    rand_l1 = {}                                           # (w, k) -> per-token mean |random-k projection| averaged over 3 draws

    def rand_l1_tok(w, k):
        if (w, k) not in rand_l1:
            rand_l1[(w, k)] = np.mean([per_token(np.linalg.qr(rr.standard_normal((D, k)))[0], TOKS[w][0])[1] for _ in range(3)], 0)
        return rand_l1[(w, k)]

    for src in ("h8", "p0", "z16_obj"):
        S = share_sets(src)
        for w in ("all", "obj"):
            T = TOKS[w][0]; pt = {f: per_token(q, T) for f, q in S.items()}
            for scn in SCEN + ["pooled"]:
                Gw = Gpool[w] if scn == "pooled" else G[w][scn]; ev = ev_all[(w, scn)]; tr = ev.sum(); mt = masks[w][scn]; rec = {}
                for f, q in S.items():
                    k = q.shape[1]; sq, l1t = pt[f]
                    rec[f] = {"k": k, "l2_share_exact": float(np.trace(q.T @ Gw @ q) / tr), "l2_random_expect": k / D, "l2_pca_topk": float(ev[:k].sum() / tr),
                              "l2_share_tokens": float(sq[mt].sum() / tok_sq[w][mt].sum()), "l1_share_tokens": float(l1t[mt].mean() / tok_l1[w][mt].mean()),
                              "l1_random_tokens": float(rand_l1_tok(w, k)[mt].mean() / tok_l1[w][mt].mean()), "n_tokens_stored": int(mt.sum())}
                scoring["exact_gram"][f"{src}|{w}|{scn}"] = rec
        log(f"    shares {src} done {time.time() - t0:.0f}s")
    # cross-application predictions used by the bootstrap R² CIs (computed once)
    cross_pr = {}
    for A, B in (("z16_obj", "z16_bg"), ("z16_bg", "z16_obj"), ("z16_obj", "p0"), ("h8", "p0"), ("z16_obj", "h8")):
        for f in ("position", "velocity"):
            Y = LAB[CLIP_REPS[B]][f]; pr = np.zeros_like(Y)
            for k in range(K_FOLD):
                tr, te = folds[A] != k, folds[A] == k; pr[te] = models[A][f][k].predict(X_rep[B][te], mu=X_rep[B][tr].mean(0))
            cross_pr[(A, B, f)] = pr
    # bootstrap token subsample (16 stored tokens per clip and window) so that each iteration stays cheap; the point estimates above use all stored tokens
    BT = {}
    for w, (T, cl) in TOKS.items():
        pick = np.concatenate([rr.choice(np.where(cl == i)[0], min(16, (cl == i).sum()), replace=False) for i in range(n)]); BT[w] = (T[pick], cl[pick])

    # ---- bootstrap (block = clip; resample within scenario)
    log(f"bootstrap B={a.boot}"); rb = np.random.default_rng(SEED + 4); boot = {"within": {k: [] for k in ["position~velocity", "position~scenario", "velocity~scenario", "position~time", "velocity~time", "scenario~time", "position~objbg", "velocity~objbg", "vel_resid~position", "vel_resid~time"]},
                                                                                "objbg_vel": [], "objbg_pos": [], "objbg_time": [], "cross": {k: [] for k in ["position:z16_obj~p0", "position:z16_obj~h8", "position:z16_obj~z32_8", "velocity:z16_obj~p0", "velocity:z16_obj~h8", "velocity:z16_obj~z32_8", "velocity:p0~h8", "position:p0~h8", "time:z16_obj~h", "time:p~h"]},
                                                                                "share": {}, "r2": {}, "acc": {}, "t_iter": []}
    rel_of = lambda rep, f: quality[rep][f]["rel_final"]
    idx_scn = {scn: np.where(scen == scn)[0] for scn in SCEN}
    for b in range(a.boot):
        tb = time.time(); idx = np.concatenate([rb.choice(idx_scn[s], len(idx_scn[s]), replace=True) for s in SCEN]); idx_tf = (idx[:, None] * 8 + np.arange(8)[None]).reshape(-1)
        Qb = {}
        for rep, fs in (("z16_obj", ["position", "velocity", "vel_resid", "scenario"]), ("z16_bg", ["position", "velocity"]), ("p0", ["position", "velocity", "scenario", "time"]), ("h8", ["position", "velocity", "scenario", "time"]), ("z32_8", ["position", "velocity"])):
            T = CLIP_REPS[rep]
            for f in fs:
                if f == "time":
                    continue
                Y = LAB[T][f][idx] if f in CONT else onehot_scen[idx]; Qb[(rep, f)] = subspace(fit_fixed(X_rep[rep][idx], Y, rel_of(rep, f)).W)
        for rep in ("z16_obj", "z16_bg", "p", "h"):
            rows = comp[rep][idx_tf[tmask[rep][idx_tf]]]
            Qb[(f"time:{rep}", "time")] = subspace(fit_fixed(X_time[rep][rows], onehot_t[rep][rows], rel_of(f"time:{rep}", "time")).W)
        idx_ob = np.concatenate([idx, idx + n]); Qb[("objbg:z16", "objbg")] = subspace(fit_fixed(X_ob[idx_ob], Y_ob[idx_ob], rel_of("objbg:z16", "objbg")).W)
        g = lambda rep, f: Qb[(rep, f)] if f not in ("time", "objbg") else (Qb[(f"time:{rep}", "time")] if f == "time" else Qb[("objbg:z16", "objbg")])
        for key in boot["within"]:
            fa, fb = key.split("~"); boot["within"][key].append(overlap(g("z16_obj", fa), g("z16_obj", fb)))
        boot["objbg_vel"].append(overlap(Qb[("z16_obj", "velocity")], Qb[("z16_bg", "velocity")])); boot["objbg_pos"].append(overlap(Qb[("z16_obj", "position")], Qb[("z16_bg", "position")]))
        boot["objbg_time"].append(overlap(Qb[("time:z16_obj", "time")], Qb[("time:z16_bg", "time")]))
        for key in boot["cross"]:
            f, ab = key.split(":"); A, B = ab.split("~"); boot["cross"][key].append(overlap(g(A, f), g(B, f)))
        # scoring shares with h8-fitted subspaces on the resampled clips' stored tokens (obj exact windows; all = 48-token subsample)
        Sb = {"position": Qb[("h8", "position")], "velocity": Qb[("h8", "velocity")], "scenario": Qb[("h8", "scenario")], "time": Qb[("time:h", "time")]}
        Sb["union_pos_vel"] = orth_union(Sb["position"], Sb["velocity"]); Sb["union_kin_time_scen"] = orth_union(Sb["position"], Sb["velocity"], Sb["scenario"], Sb["time"])
        cnt = np.bincount(idx, minlength=n)
        for w, (T, cl) in BT.items():
            wt = cnt[cl].astype(np.float32)                                    # bootstrap weights per stored token
            sq0, l10 = (T ** 2).sum(1), np.abs(T).mean(1); pt = {f: per_token(q, T) for f, q in Sb.items()}
            for scn in ["pooled"] + SCEN:
                m = np.ones(len(T), bool) if scn == "pooled" else (scen[cl] == scn); wm = wt[m]
                den2 = float((wm * sq0[m]).sum()); den1 = float((wm * l10[m]).sum())
                for f in Sb:
                    sq, l1t = pt[f]; l2 = float((wm * sq[m]).sum() / den2); l1 = float((wm * l1t[m]).sum() / den1)
                    boot["share"].setdefault(f"h8|{w}|{scn}|{f}", {"l2": [], "l1": []}); boot["share"][f"h8|{w}|{scn}|{f}"]["l2"].append(l2); boot["share"][f"h8|{w}|{scn}|{f}"]["l1"].append(l1)
        # R^2 / acc CIs from out-of-fold predictions
        for rep in MAIN_REPS:
            for f in ("position", "velocity", "vel_resid"):
                Y = LAB[CLIP_REPS[rep]][f]; boot["r2"].setdefault(f"{rep}|{f}", []).append(float(np.mean([r2(oofs[rep][f][idx, j], Y[idx, j]) or 0 for j in range(2)])))
            boot["acc"].setdefault(f"{rep}|scenario", []).append(float((oofs[rep]["scenario"][idx].argmax(1) == onehot_scen[idx].argmax(1)).mean()))
        for (A, B, f), pr in cross_pr.items():
            Y = LAB[CLIP_REPS[B]][f]; boot["r2"].setdefault(f"cross:{A}->{B}|{f}", []).append(float(np.mean([r2(pr[idx, j], Y[idx, j]) or 0 for j in range(2)])))
        boot["t_iter"].append(time.time() - tb)
        if b % 20 == 0:
            log(f"    boot {b}/{a.boot}  {boot['t_iter'][-1]:.1f}s/iter  {time.time() - t0:.0f}s")
    CI = {"within": {k: boot_ci(v) for k, v in boot["within"].items()}, "objbg_vel": boot_ci(boot["objbg_vel"]), "objbg_pos": boot_ci(boot["objbg_pos"]), "objbg_time": boot_ci(boot["objbg_time"]),
          "cross": {k: boot_ci(v) for k, v in boot["cross"].items()}, "share": {k: {"l2": boot_ci(v["l2"]), "l1": boot_ci(v["l1"])} for k, v in boot["share"].items()},
          "r2": {k: boot_ci(v) for k, v in boot["r2"].items()}, "acc": {k: boot_ci(v) for k, v in boot["acc"].items()}, "B": a.boot, "mean_s_per_iter": float(np.mean(boot["t_iter"])) if boot["t_iter"] else None}

    res = {"script": str(Path(__file__).resolve().relative_to(ROOT)), "config": {"index": str(INDEX), "caches": {"z16": str(CACHE_Z16), "hp": str(CACHE_HP), "z32": str(CACHE_Z32)}, "scenarios": SCEN, "n_per_scenario_max": a.n_per, "seed": SEED, "k_fold": K_FOLD, "n_inner": N_INNER,
                                                                                    "lambda_rel_grid": LAM_REL, "n_sub_tok": N_SUB_TOK, "n_split_half": N_SPLIT_HALF, "n_random": N_RANDOM, "degenerate_accel_std": DEGEN_STD, "D": D, "cell_px": CELL, "boot": a.boot,
                                                                                    "rep_label_time": CLIP_REPS, "units": "cells (1 cell = 18 px); velocity cells/tubelet; acceleration cells/tubelet^2"},
           "n_clips": int(n), "n_per_scenario": {s: int((scen == s).sum()) for s in SCEN}, "selection": counts, "n_blocks": int(len(set(blocks))),
           "window_size_counts": {f"t{t}": {str(k): int(v) for k, v in zip(*np.unique(win_n[:, t], return_counts=True))} for t in range(16)},
           "bg_min_chebyshev_dist": {"mean": float(bg_info[:, 2].mean()), "min": float(bg_info[:, 2].min()), "frac_ge4": float((bg_info[:, 2] >= 4).mean())},
           "label_stats": label_stats, "readout_quality": quality, "cross_application": cross, "subspace_k": ks, "variant_agreement": variant_agreement, "split_half": split_half,
           "overlap_within_z16_obj": within, "overlap_within_h8": within_h, "overlap_obj_vs_bg": objbg, "overlap_cross_rep": cross_rep, "lambda_sensitivity": lam_sens, "variance_share": vshare, "scoring_share": scoring, "bootstrap_ci": CI, "elapsed_s": float(time.time() - t0)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1)); write_md(res, OUT); log(f"-> {OUT}/results.json, RESULTS.md  ({time.time() - t0:.0f}s)")


TIME_REPS_INV = {"z16_obj": "z16_obj", "z16_bg": "z16_bg", "p0": "p", "h8": "h", "z32_8": "z32"}


# ----------------------------------------------------------------------------------------------- markdown
def fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, (float, int)) else str(v))


def ci(c, nd=3):
    return "—" if not c or c[0] is None else f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def prow(name, p, cival=None):
    return f"| {name} | {p['k'][0]}, {p['k'][1]} | {fmt(p['overlap'])} | {ci(cival) if cival else '—'} | {fmt(p['random']['analytic'])} / {fmt(p['random']['p97.5'])} | {fmt(p['split_half_a'])} / {fmt(p['split_half_b'])} | {fmt(p['ceiling_sqrt'])} |"


def write_md(res, OUT):
    c = res["config"]; n = res["n_clips"]; L = []
    A = L.append
    A("# S2 — kinematic factor subspaces at the predictor input (RollOut_v2 plausible, ViT-H)")
    A("")
    A(f"Every number below is recomputed from `results.json` written by `{res['script']}` (seed {c['seed']}, up to {c['n_per_scenario_max']} clips per scenario, "
      f"n = {n} clips = {res['n_blocks']} blocks; per scenario {res['n_per_scenario']}). Subsampled from the 4,312 plausible clips of the six scenarios for the compute budget.")
    A(f"Subspace = orthonormalised column span of the closed-form ridge readout (centered raw token coordinates, lambda = rel·tr(X'X)/D, rel chosen on an inner block fold). "
      f"Overlap = mean cos² of principal angles (k1, k2 stated). Lower bound = random subspaces of the same sizes in D = {c['D']} (analytic / empirical 97.5 %); upper bound = split-half of each factor ({c['n_split_half']} splits), pair ceiling = sqrt of the two. "
      f"CIs = block bootstrap over clips within scenario, B = {res['bootstrap_ci']['B']} (mean {fmt(res['bootstrap_ci']['mean_s_per_iter'], 1)} s/iter). Reps: z16_obj / z16_bg = ctx_masked t7 (3x3 at the object / far background), p0 = predictor slot 0 at pos8, h8 / z32_8 = target encoder / 32-frame context encoder t8 at pos8; h7 / z32_7 at pos7.")
    A(f"Units: cells (1 cell = 18 px). z16 reps are labelled at T = 7, p / h / z32 at T = 8. Background window: Chebyshev distance ≥ 4 from the object at every context tubelet for {100 * res['bg_min_chebyshev_dist']['frac_ge4']:.1f} % of clips (mean min-dist {res['bg_min_chebyshev_dist']['mean']:.2f}).")
    A("")
    # label stats
    A("## Labels — std per scenario (cells), acceleration degeneracy, position–velocity confound")
    A("")
    for T in ("T7", "T8"):
        A(f"### {T}")
        A("")
        A("| scenario | n | x std | y std | vx std | vy std | a‖ std | accel degenerate | corr x–vx | corr y–vy | corr x–vy | corr y–vx | vel_resid var / vel var |")
        A("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for scn, s in res["label_stats"][T].items():
            cc = s["corr_pos_vel"]
            A(f"| {scn} | {s['n']} | {fmt(s['x_std'])} | {fmt(s['y_std'])} | {fmt(s['vx_std'])} | {fmt(s['vy_std'])} | {s['accel_norm_std']:.4f} | {'**yes**' if s['accel_degenerate'] else 'no'} | {fmt(cc['x-vx'], 2)} | {fmt(cc['y-vy'], 2)} | {fmt(cc['x-vy'], 2)} | {fmt(cc['y-vx'], 2)} | {fmt(s['vel_resid_frac_var'], 2)} |")
        A("")
    A("`accel degenerate` = 2-D std of a_T below 0.005 cells/tubelet² inside the scenario; the pooled acceleration label is then mostly scenario identity. `vel_resid` = velocity minus its within-scenario quadratic fit on position — the velocity variation that position does not explain (small in ramp_a / arc / ledge, where velocity is a function of position).")
    A("")
    # (a) readout quality
    A("## (a) Readout quality — out-of-fold R² (continuous, per column; pooled over clips and per scenario relative to the scenario's own variance) / accuracy (categorical)")
    A("")
    A("| rep (T) | factor | rel λ | pooled | " + " | ".join(SCEN) + " | 95 % CI (pooled mean of the two columns) |")
    A("|---|---|---:|---|" + "---|" * len(SCEN) + "---|")
    CIr = res["bootstrap_ci"]["r2"]; CIa = res["bootstrap_ci"]["acc"]
    for rep, T in c["rep_label_time"].items():
        q = res["readout_quality"][rep]
        for f, cols in CONT.items():
            po = " / ".join(f"{cc} {fmt(q[f]['pooled'][cc]['r2'], 2)}" for cc in cols)
            ps = [" / ".join(fmt(q[f]["per_scenario"][s][cc]["r2"], 2) for cc in cols) for s in SCEN]
            A(f"| {rep} ({T}) | {f} | {q[f]['rel_final']:g} | {po} | " + " | ".join(ps) + f" | {ci(CIr.get(f'{rep}|{f}'), 2)} |")
        s = q["scenario"]; A(f"| {rep} ({T}) | scenario (acc, chance {s['chance']:.2f}) | {s['rel_final']:g} | acc {fmt(s['acc'], 3)} bacc {fmt(s['bacc'], 3)} | " + " | ".join(fmt(s["per_scenario"][sc]["acc"], 2) for sc in SCEN) + f" | {ci(CIa.get(f'{rep}|scenario'), 3)} |")
    A("")
    A("| time rep | n rows | acc (chance 0.125) | bacc | rel λ | per-tubelet acc (n) |")
    A("|---|---:|---:|---:|---:|---|")
    for rep in TIME_REPS:
        q = res["readout_quality"][f"time:{rep}"]["time"]; A(f"| time:{rep} | {q['n']} | {fmt(q['acc'])} | {fmt(q['bacc'])} | {q['rel_final']:g} | " + ", ".join(f"{k} {v:.2f} ({q['n_per_tubelet'][k]})" for k, v in q["per_tubelet_acc"].items()) + " |")
    q = res["readout_quality"]["objbg:z16"]["objbg"]; A(f"| objbg:z16 (object vs background 3x3 @t7) | {q['n']} | {fmt(q['acc'])} | {fmt(q['bacc'])} | {q['rel_final']:g} | chance 0.5 |")
    A("")
    A("z16_obj time rows drop tubelets whose 3x3 window is empty (object off-screen before t5); the other time reps use every row.")
    A("")
    A("Subspace ranks k: " + ", ".join(f"{k} {v}" for k, v in res["subspace_k"].items()))
    A("")
    A("Ridge vs between-class-scatter vs LDA subspaces of the same categorical factor (overlap):")
    A("")
    A("| factor | k (ridge, scatter) | ridge~scatter | ridge~LDA | scatter~LDA |")
    A("|---|---|---:|---:|---:|")
    for k, v in res["variant_agreement"].items():
        A(f"| {k} | {v['k'][0]}, {v['k'][1]} | {fmt(v['ridge_vs_scatter'])} | {fmt(v['ridge_vs_lda'])} | {fmt(v['scatter_vs_lda'])} |")
    A("")
    # cross application
    A("## Cross-application of readouts (fit on rep A's train fold, applied to rep B's test fold; recentered with B's train-fold mean; R² mean of x,y / raw = no recentering)")
    A("")
    for f in ("position", "velocity"):
        A(f"### {f}")
        A("")
        A("| A → B | label T of B | R² recentered (x / y) | R² raw (x / y) | 95 % CI (recentered mean) | " + " | ".join(f"{s}" for s in SCEN) + " |")
        A("|---|---|---|---|---|" + "---|" * len(SCEN))
        for k, v in res["cross_application"][f].items():
            cols = CONT[f]; rc = v["recentered"]
            A(f"| {k} | {v['label_time_of_B']} | " + " / ".join(fmt(rc["pooled"][cc]["r2"], 2) for cc in cols) + " | " + " / ".join(fmt(v["raw"][cc]["r2"], 2) for cc in cols) + f" | {ci(CIr.get(f'cross:{k}|{f}'), 2)} | " + " | ".join(" / ".join(fmt(rc["per_scenario"][s][cc]["r2"], 2) for cc in cols) for s in SCEN) + " |")
        A("")
    A("Time readout cross-application (object window ↔ background window, z16): " + "; ".join(f"{k}: acc {fmt(v['acc'])} (bacc {fmt(v['bacc'])})" for k, v in res["cross_application"]["time"].items()))
    A("")
    hdr = ["| pair | k1, k2 | overlap | 95 % CI | random (analytic / p97.5) | split-half a / b | ceiling √(a·b) |", "|---|---|---:|---|---|---|---:|"]
    A("## (b) Pairwise factor-subspace overlap inside z16_obj (predictor input, boundary token)")
    A("")
    L += hdr
    CIw = res["bootstrap_ci"]["within"]
    for k, p in res["overlap_within_z16_obj"].items():
        A(prow(k, p, CIw.get(k)))
    A("")
    A("Same pairs inside h8 (target encoder at t8, saw the future) for contrast:")
    A("")
    L += hdr
    for k, p in res["overlap_within_h8"].items():
        A(prow(k, p))
    A("")
    A("## (c) Object-window vs background-window subspaces (z16, t7)")
    A("")
    L += hdr
    CIo = res["bootstrap_ci"]
    for k, p in res["overlap_obj_vs_bg"].items():
        A(prow(k, p, {"velocity": CIo["objbg_vel"], "position": CIo["objbg_pos"], "time": CIo["objbg_time"]}.get(k)))
    A("")
    A("Read with the cross-application table above (z16_obj→z16_bg and z16_bg→z16_obj R²): same subspace AND transferable readout = one global code; low overlap or non-transferable = a different (scene / camera) code in the background.")
    A("")
    A("## (d) Cross-representation overlap of the same factor subspace")
    A("")
    CIc = res["bootstrap_ci"]["cross"]
    for f, dd in res["overlap_cross_rep"].items():
        A(f"### {f}")
        A("")
        L += hdr
        for k, p in dd.items():
            A(prow(k, p, CIc.get(f"{f}:{k}")))
        A("")
    A("## λ sensitivity of the headline overlaps (same subspaces re-fitted with a fixed rel λ; split-half from 4 splits)")
    A("")
    A("A ridge readout in D = 1280 with n < D clips has an under-determined weight direction; a larger λ pulls it toward the high-variance directions and makes it more reproducible (higher split-half). Overlaps must be read against the ceiling in the same row.")
    A("")
    A("| pair | " + " | ".join(f"rel λ = {r}: overlap / ceiling" for r in res["lambda_sensitivity"]) + " |")
    A("|---|" + "---|" * len(res["lambda_sensitivity"]))
    keys = list(next(iter(res["lambda_sensitivity"].values())))
    for k in keys:
        A(f"| {k} | " + " | ".join(f"{fmt(v[k]['overlap'])} / {fmt(v[k]['ceiling_sqrt'])}" for v in res["lambda_sensitivity"].values()) + " |")
    A("")
    A("## Variance share of each factor subspace in the rep's centered token variance (random-k expectation = k/D; PCA top-k = the most any k-dim subspace can hold)")
    A("")
    A("| rep | factor | k | share | random k/D | PCA top-k |")
    A("|---|---|---:|---:|---:|---:|")
    for rep, dd in res["variance_share"].items():
        for f, v in dd.items():
            if f == "total_var":
                continue
            A(f"| {rep} | {f} | {v['k']} | {fmt(v['share'], 4)} | {fmt(v['random_expect'], 4)} | {fmt(v['pca_topk'], 4)} |")
    A("")
    A("## (e) Scoring-distance share — fraction of Σ‖P_S(p−h)‖² (exact, all future tokens / 3x3 object windows) and of the L1 analogue mean|P_S(p−h)| / mean|p−h| inside each factor subspace")
    A("")
    A("Subspaces fitted on h8 (headline; also p0 and z16_obj below). `random` = k/D (L2) or 5 random k-dim subspaces (L1); `PCA top-k` = ceiling for any k-dim subspace. L1 'all' uses 48 stored random future tokens per clip; L2 exact uses every token.")
    A("")
    mg = res["scoring_share"]["magnitude"]
    A("| window | scenario | n clips | n tokens | mean |p−h| (per element) | mean ‖p−h‖² (per token) |")
    A("|---|---|---:|---:|---:|---:|")
    for k, v in mg.items():
        w, s = k.split("|"); A(f"| {w} | {s} | {v['n_clips']} | {v['n_tokens']} | {v['mean_abs_d']:.4f} | {v['mean_sq_norm_d']:.3f} |")
    A("")
    CIs = res["bootstrap_ci"]["share"]
    for src in ("h8", "p0", "z16_obj"):
        A(f"### subspaces fitted on {src}")
        A("")
        A("| window | scenario | factor | k | L2 share exact | L2 CI (stored tokens) | L2 random | L2 PCA top-k | L1 share | L1 CI | L1 random |")
        A("|---|---|---|---:|---:|---|---:|---:|---:|---|---:|")
        for w in ("all", "obj"):
            for scn in ["pooled"] + SCEN:
                rec = res["scoring_share"]["exact_gram"][f"{src}|{w}|{scn}"]
                for f, v in rec.items():
                    cc = CIs.get(f"{src}|{w}|{scn}|{f}", {})
                    A(f"| {w} | {scn} | {f} | {v['k']} | {fmt(v['l2_share_exact'], 4)} | {ci(cc.get('l2'), 4)} | {fmt(v['l2_random_expect'], 4)} | {fmt(v['l2_pca_topk'], 4)} | {fmt(v['l1_share_tokens'], 4)} | {ci(cc.get('l1'), 4)} | {fmt(v['l1_random_tokens'], 4)} |")
        A("")
    A("## Limits")
    A("")
    A("- Linear, ridge-defined subspaces: a factor that is read out non-linearly, or spread over many low-variance directions, is under-estimated (ridge is a lower bound on the information and its span is one of many k-dim spans that decode the factor).")
    A("- Acceleration is degenerate inside most scenarios (see label table); its pooled subspace is largely a scenario-identity subspace and must not be read as an acceleration code.")
    A("- Velocity is a function of position inside ramp_a / arc / ledge; separability claims rest on flat_v / flat_a / fall and on the `vel_resid` factor.")
    A("- Split-half is the ceiling for a subspace re-estimated from half the clips, so overlaps near it are 'as similar as the estimate allows', not proof of identity; overlaps near the random line are 'no more aligned than chance'. With n < D the ridge direction itself is only partly determined (see the λ-sensitivity table), so a low overlap between two well-decoded factors can mean 'different subspaces' or 'the subspace estimate is unstable' — the ceiling separates the two.")
    A("- Bootstrap replicates contain duplicated clips, which perturbs subspace estimates the same way a smaller sample does; percentile CIs can therefore sit beside the point estimate. Read the CI width as the sampling uncertainty and the point estimate as the value.")
    A(f"- Per-scenario R² is shown only where the scenario's label std ≥ {R2_SCEN_MIN_STD} cells; inside flat scenarios y (and inside ramp_a x, y) barely vary and R² there is noise.")
    A("- Scoring shares use the scorer's own future tokens and the same h (LN'd) and raw p, but the L1 share of a projection is not additive across subspaces and is only compared to its random and PCA baselines.")
    A(f"- Subsample: {c['n_per_scenario_max']} clips per scenario (seed 0) of the 784 (392 for ledge) plausible clips; bootstrap B = {res['bootstrap_ci']['B']}.")
    A("")
    A("## 재현")
    A("")
    A("```")
    A(f"cd {ROOT}")
    A(f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python {res['script']} --n-per {c['n_per_scenario_max']} --boot {res['bootstrap_ci']['B']}")
    A("```")
    A("")
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
