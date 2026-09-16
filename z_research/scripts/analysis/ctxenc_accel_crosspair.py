#!/usr/bin/env python3
"""Is acceleration integrated into the boundary tubelet (z16 tubelet 7 = the predictor's input at the context edge) beyond position
and velocity — tested WITHOUT a new render, by matching clips ACROSS RollOut_v2 flat_v (a = 0, v in {-160..160} cm/s) and flat_a
(v0 = 0, a in {-40..40} cm/s^2) on (pos7, v7) so that only the acceleration history (positions t0..t6) differs.

Design facts (from data_csv/rollout_v2/index.csv, verified in this script and written to results.json):
  * every (scenario, primary, anchor) cell holds 56 clips with an IDENTICAL trajectory (label std 0); clips differ only in shape x color x env.
  * both scenarios share the level scene (motion 'moving_flat'), the same object size / depth, anchor +-100 cm at raw frame 45.
  * cells that match at |d pos7| <= 0.3 cell and |d v7| <= 0.05 cell/tubelet: flat_a a = +-40 vs flat_v v = +-105 (d v7 = 0.015, d x7 = 0.004),
    and the physically identical stationary cells flat_a a = 0 vs flat_v v = 0 (d = 0) which serve as the cross-scenario SAME-a control.
    A looser |d v7| <= 0.06 adds flat_a a = +-27 vs flat_v v = +-55 (d v7 = 0.052) — reported as a sensitivity variant.
  * across the whole union, a7 is a deterministic function of (x7, v7) cell identity, so the position+velocity shortcut is expected to
    be high there; the union ridge is reported to SHOW the leak, the matched-subset ridge is the actual test.
Features (3x3 token mean around the true object position, n3() copied verbatim from v11_token_object_test.py):
  F1 z16 tubelet 7 (1280)   F2 z16 tubelets 7+6 (2560)   F3 z16 tubelets 0..7 (10240)   F1_bg z16 tubelet 7 at a window >= 4 cells away
  H7 target encoder tubelet 7 (saw all 32 frames)   P0 predictor slot 0 (= tubelet 8) at the true pos8.
Tests:
  (1) parameter-free pairs: cosine (raw / union-centered / scenario-centered) for cross-scenario matched pairs that DIFFER in |a| vs
      within-scenario same-cell pairs (same a) vs the stationary cross-scenario same-a control; only appearance-mismatched pairs in the
      main rows (within-cell pairs are always appearance-mismatched); clip-level (block) bootstrap 95 % CI, 300 resamples; AUC
      (within > cross); velocity-different within-scenario pairs as a yardstick.
  (2) closed-form ridge (SVD), 5-fold CV by block_id stratified by cell, lambda on an inner 25 % validation split, labels a7 (cells/
      tubelet^2) and nominal a (cm/s^2); features residualised on poly2(x7, y7, vx7, vy7) fitted on the training fold; baselines:
      position+velocity-only shortcut (poly2 and poly3), label shuffle; subsets: union / matched / matched+stationary; block bootstrap R^2 CI.
  (3) the same for the binary label 'has nonzero acceleration' (balanced by subsampling negatives, seed 0): balanced accuracy + AUC,
      plus the stationary control (flat_v v=0 vs flat_a a=0 scenario classification, must be chance).
No GPU, no model; caches read one clip at a time (mmap, float16). Deterministic (seed 0).
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_accel_crosspair/{results.json, RESULTS.md, features.npz}
Run:
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_accel_crosspair.py [--n-per-scenario 784] [--boot 300]
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):      # 96-core box: unbounded BLAS threads make the small SVDs 50x slower
    os.environ.setdefault(_v, "16")
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
INDEX = ROOT / "data_csv/rollout_v2/index.csv"
CACHE_Z = Path("/local_datasets/world/world_analysis/cache/rollout_v2_z16_vith")
CACHE_HP = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
OUT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_accel_crosspair"
SCEN = ["flat_v", "flat_a"]
SEED, K_FOLD, VAL_FRAC = 0, 5, 0.25
LAMBDAS = [float(v) for v in np.logspace(-2, 6, 17)]
CELL, RES, D = 18.0, 288, 1280
THR_POS, THR_V, THR_V_LOOSE = 0.3, 0.05, 0.06
FEATS = ["F1", "F2", "F3", "F1_bg", "H7", "P0"]
CENTERINGS = ["raw", "union_centered", "scenario_centered"]


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def n3(x, y):
    """copied verbatim from z_research/scripts/analysis/v11_token_object_test.py — 3x3 token window around px (x, y)."""
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


def bg_cell(cx, cy, rng):
    """a token cell at Chebyshev distance >= 4 from (cx, cy), chosen deterministically (same rule as ctxenc_boundary_kinematics.py)."""
    cand = [(c, r) for r in range(1, 15) for c in range(1, 15) if max(abs(c - cx), abs(r - cy)) >= 4]
    c, r = cand[rng.integers(len(cand))]; return c * CELL + 9, r * CELL + 9


# ----------------------------------------------------------------------------------------------- data + labels
def load_clips(n_per):
    rows = [r for r in csv.DictReader(INDEX.open()) if r["plausible"] == "1" and r["scenario"] in SCEN]
    rng = np.random.default_rng(SEED); keep = []
    for scn in SCEN:
        rs = [r for r in rows if r["scenario"] == scn]
        ok = []
        for r in rs:
            px, py, inf = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"]), arr(r["in_frame_by_sample"])
            tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)
            if inf[10:18].min() > 0 and (tx[5:9] >= 0).all() and (tx[5:9] < RES).all() and (ty[5:9] >= 0).all() and (ty[5:9] < RES).all():
                ok.append(r)
        if n_per < len(ok):      # stratified subsample: equal count per (primary, secondary) cell
            cells = {}
            for i, r in enumerate(ok):
                cells.setdefault((r["primary"], r["secondary"]), []).append(i)
            per = max(1, n_per // len(cells)); sel = sorted(sum([list(rng.choice(v, min(per, len(v)), replace=False)) for v in cells.values()], []))
        else:
            sel = list(range(len(ok)))
        keep += [ok[i] for i in sel]
        print(f"  {scn}: {len(rs)} plausible, {len(ok)} valid, {len(sel)} used", flush=True)
    return keep


def labels_of(r):
    px, py = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])
    tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)
    cx, cy = tx / CELL, ty / CELL
    vx7, vy7 = cx[7] - cx[6], cy[7] - cy[6]; ax7, ay7 = vx7 - (cx[6] - cx[5]), vy7 - (cy[6] - cy[5])
    a_nom = float(r["primary"]) if r["scenario"] == "flat_a" else 0.0
    v_nom = float(r["primary"]) if r["scenario"] == "flat_v" else 0.0
    return dict(x7=cx[7], y7=cy[7], vx7=vx7, vy7=vy7, ax7=ax7, ay7=ay7, x8=cx[8], y8=cy[8], a_nom=a_nom, v_nom=v_nom, anchor=float(r["secondary"]), x0=cx[0]), tx, ty


def extract(clips):
    meta_z = json.loads((CACHE_Z / "meta.json").read_text()); meta_hp = json.loads((CACHE_HP / "meta.json").read_text())
    row_z = {v: i for i, v in enumerate(meta_z["video_ids"])}; row_hp = {v: i for i, v in enumerate(meta_hp["video_ids"])}
    assert meta_z["base_counts"]["ctx_masked"] == 2048 and meta_hp["base_counts"]["target"] == 4096 and meta_hp["base_counts"]["predictor"] == 2048
    Z = np.load(CACHE_Z / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE_HP / "target.npy", mmap_mode="r"); P = np.load(CACHE_HP / "predictor.npy", mmap_mode="r")
    n = len(clips); Fz = np.zeros((n, 8, D), np.float32); Fbg = np.zeros((n, D), np.float32); Fh7 = np.zeros((n, D), np.float32); Fp0 = np.zeros((n, D), np.float32)
    rng = np.random.default_rng(SEED); order = np.argsort([row_z[r["video_id"]] for r in clips]); t0 = time.time(); win_ok = np.ones(n, bool)
    for j, i in enumerate(order):
        r = clips[i]; _, tx, ty = labels_of(r); rz, rh = row_z[r["video_id"]], row_hp[r["video_id"]]
        z = np.asarray(Z[rz], np.float32).reshape(8, 256, D)
        for t in range(8):
            w = n3(tx[t], ty[t]); win_ok[i] &= len(w) == 9; Fz[i, t] = z[t, w].mean(0) if w else 0.0
        bx, by = bg_cell(int(tx[7] // CELL), int(ty[7] // CELL), rng); Fbg[i] = z[7, n3(bx, by)].mean(0)
        h = np.asarray(H[rh], np.float32).reshape(16, 256, D); Fh7[i] = h[7, n3(tx[7], ty[7])].mean(0)
        w8 = n3(tx[8], ty[8]); win_ok[i] &= len(w8) == 9; p = np.asarray(P[rh], np.float32).reshape(8, 256, D); Fp0[i] = p[0, w8].mean(0)
        if j % 200 == 0:
            print(f"    extract {j}/{n} {time.time()-t0:.0f}s", flush=True)
    return Fz, Fbg, Fh7, Fp0, win_ok


# ----------------------------------------------------------------------------------------------- ridge (closed form, SVD)
def ridge_fit_predict(Xtr, Ytr, Xte, lams):
    """closed-form ridge, standardised on train; primal SVD when n >= d, dual (Gram eigendecomposition) when d > n — same solution:
    (X'X + lam I)^-1 X'Y = X' (XX' + lam I)^-1 Y."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xc = (Xtr - mu) / sd; ym = Ytr.mean(0); Yc = Ytr - ym; Xte_c = (Xte - mu) / sd
    if Xc.shape[1] > Xc.shape[0]:
        w, V = np.linalg.eigh(Xc @ Xc.T); w = np.clip(w, 0, None); VtY = V.T @ Yc; Kte = Xte_c @ Xc.T
        return {lam: Kte @ (V @ (VtY / (w + lam)[:, None])) + ym for lam in lams}
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); UtY = U.T @ Yc
    return {lam: Xte_c @ (Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)) + ym for lam in lams}


def strat_folds(strata, k, rng):
    fold = np.zeros(len(strata), int)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % k
    return fold


def poly_design(K, degree):
    """polynomial features (no intercept; constant columns dropped) of kinematic matrix K (n, 4) = (x7, y7, vx7, vy7)."""
    cols = [K[:, j] for j in range(K.shape[1])]
    out = list(cols)
    if degree >= 2:
        out += [cols[i] * cols[j] for i in range(len(cols)) for j in range(i, len(cols))]
    if degree >= 3:
        out += [cols[i] * cols[j] * cols[k] for i in range(len(cols)) for j in range(i, len(cols)) for k in range(j, len(cols))]
    X = np.stack(out, 1); X = X[:, X.std(0) > 1e-9]
    return X


def residualise(Xtr, Xte, Ktr, Kte):
    """regress every feature dim on [1, poly(K)] fitted on the training fold; return residuals for train and test."""
    A = np.concatenate([np.ones((len(Ktr), 1)), Ktr], 1); B = np.concatenate([np.ones((len(Kte), 1)), Kte], 1)
    coef, *_ = np.linalg.lstsq(A, Xtr, rcond=None)
    return Xtr - A @ coef, Xte - B @ coef


def cv_predict(X, Y, strata, seed, K=None, shuffle_labels=False):
    """out-of-fold predictions; if K (poly kinematics design) is given, X is residualised on K within each fold first."""
    rng = np.random.default_rng(seed); fold = strat_folds(strata, K_FOLD, rng); pred = np.zeros_like(Y); lam_used = []
    for f in range(K_FOLD):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        Ytr = Y[tr].copy()
        if shuffle_labels:
            Ytr = Ytr[rng.permutation(len(tr))]
        inner = strat_folds(strata[tr], int(round(1 / VAL_FRAC)), rng); itr, iva = tr[inner != 0], tr[inner == 0]
        Xitr, Xiva = (X[itr], X[iva]) if K is None else residualise(X[itr], X[iva], K[itr], K[iva])
        pv = ridge_fit_predict(Xitr, Ytr[inner != 0], Xiva, LAMBDAS); Yva = Ytr[inner == 0]
        best = [min(LAMBDAS, key=lambda l: ((pv[l][:, j] - Yva[:, j]) ** 2).mean()) for j in range(Y.shape[1])]
        Xtr_, Xte_ = (X[tr], X[te]) if K is None else residualise(X[tr], X[te], K[tr], K[te])
        pt = ridge_fit_predict(Xtr_, Ytr, Xte_, sorted(set(best)))
        for j, l in enumerate(best):
            pred[te, j] = pt[l][:, j]
        lam_used.append(best)
    return pred, lam_used


def r2(p, y):
    sst = ((y - y.mean()) ** 2).sum(); return float(1 - ((p - y) ** 2).sum() / sst) if sst > 0 else None


def boot_idx(n, B, rng):
    return rng.integers(n, size=(B, n))


def r2_boot(p, y, idx):
    vals = [r2(p[i], y[i]) for i in idx]; vals = [v for v in vals if v is not None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def bacc(score, y):
    pr = np.sign(score); pr[pr == 0] = 1; pos, neg = y > 0, y < 0
    return float(0.5 * ((pr[pos] > 0).mean() + (pr[neg] < 0).mean()))


def auc(score_pos, score_neg):
    """P(score_pos > score_neg) + 0.5 P(tie), via sorting."""
    sn = np.sort(score_neg); n = len(sn)
    lo = np.searchsorted(sn, score_pos, "left"); hi = np.searchsorted(sn, score_pos, "right")
    return float(((lo + hi) / 2).mean() / n)


def cls_boot(score, y, idx):
    b = [bacc(score[i], y[i]) for i in idx if (y[i] > 0).any() and (y[i] < 0).any()]; a = [auc(score[i][y[i] > 0], score[i][y[i] < 0]) for i in idx if (y[i] > 0).any() and (y[i] < 0).any()]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ----------------------------------------------------------------------------------------------- parameter-free pair test
def cell_key(lab, scn):
    return (scn, lab["a_nom"] if scn == "flat_a" else lab["v_nom"], lab["anchor"])


def pair_test(F_by_feat, cells, cell_lab, scen, app, thr_v, B, rng):
    """cells: {key: idx array}; cell_lab: {key: dict(x7, vx7, a_nom, v_nom)}; app: appearance triple per clip.
    Groups of clip pairs: cross_diff_a (flat_v cell x flat_a cell matched on pos7,v7 with a != 0), cross_same_a (stationary v=0 x a=0),
    within_flat_v / within_flat_a (same cell, cells that take part in cross_diff_a), within_stationary_* (same cell, v=0 / a=0),
    yardsticks: within-scenario pos-matched pairs with a clear velocity difference."""
    fv = [k for k in cells if k[0] == "flat_v"]; fa = [k for k in cells if k[0] == "flat_a"]
    cross = []
    for kv in fv:
        for ka in fa:
            dx = abs(cell_lab[kv]["x7"] - cell_lab[ka]["x7"]); dv = abs(cell_lab[kv]["vx7"] - cell_lab[ka]["vx7"])
            if dx <= THR_POS and dv <= thr_v:
                cross.append(dict(flat_v=kv, flat_a=ka, d_x7=dx, d_vx7=dv, d_x0=abs(cell_lab[kv]["x0"] - cell_lab[ka]["x0"]), a_nom=cell_lab[ka]["a_nom"], v_nom=cell_lab[kv]["v_nom"], anchor=kv[2],
                                  kind="cross_diff_a" if cell_lab[ka]["a_nom"] != 0 else "cross_same_a"))
    # yardstick pairs: same scenario, same anchor, different primary with |d v7| in [0.1, 0.25] (a clear velocity difference, pos matched)
    yard = []
    for ks in (fv, fa):
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                if k1[2] == k2[2] and abs(cell_lab[k1]["x7"] - cell_lab[k2]["x7"]) <= THR_POS and 0.1 <= abs(cell_lab[k1]["vx7"] - cell_lab[k2]["vx7"]) <= 0.25:
                    yard.append(dict(a=k1, b=k2, d_vx7=abs(cell_lab[k1]["vx7"] - cell_lab[k2]["vx7"]), scen=k1[0]))
    # clip-pair groups: list of (idxA, idxB, within?)
    groups = {"cross_diff_a": [(cells[c["flat_v"]], cells[c["flat_a"]], False) for c in cross if c["kind"] == "cross_diff_a"],
              "cross_same_a_stationary": [(cells[c["flat_v"]], cells[c["flat_a"]], False) for c in cross if c["kind"] == "cross_same_a"],
              "within_flat_v_same_cell": [(cells[c["flat_v"]], cells[c["flat_v"]], True) for c in cross if c["kind"] == "cross_diff_a"],
              "within_flat_a_same_cell": [(cells[c["flat_a"]], cells[c["flat_a"]], True) for c in cross if c["kind"] == "cross_diff_a"],
              "within_stationary_flat_v": [(cells[c["flat_v"]], cells[c["flat_v"]], True) for c in cross if c["kind"] == "cross_same_a"],
              "within_stationary_flat_a": [(cells[c["flat_a"]], cells[c["flat_a"]], True) for c in cross if c["kind"] == "cross_same_a"],
              "yardstick_within_flat_v_diff_v": [(cells[y["a"]], cells[y["b"]], False) for y in yard if y["scen"] == "flat_v"],
              "yardstick_within_flat_a_diff_v": [(cells[y["a"]], cells[y["b"]], False) for y in yard if y["scen"] == "flat_a"]}
    # per-cell bootstrap resamples shared across groups (paired differences)
    boot = {k: boot_idx(len(v), B, rng) for k, v in cells.items()}
    cell_of = {}
    for k, v in cells.items():
        for i in v:
            cell_of[int(i)] = k
    out = {"cell_pairs": [{**c, "flat_v": list(map(str, c["flat_v"])), "flat_a": list(map(str, c["flat_a"]))} for c in cross],
           "yardstick_pairs": [{"a": list(map(str, y["a"])), "b": list(map(str, y["b"])), "d_vx7": y["d_vx7"]} for y in yard], "n_boot": B, "features": {}}
    for fname, F in F_by_feat.items():
        Fs = {"raw": F, "union_centered": F - F.mean(0), "scenario_centered": F.copy()}
        for scn in SCEN:
            m = scen == scn; Fs["scenario_centered"][m] = F[m] - F[m].mean(0)
        rec = {}
        for cname, X in Fs.items():
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            gres = {}; pooled = {}
            for gname, plist in groups.items():
                if not plist:
                    gres[gname] = {"n_pairs": 0}; continue
                vals_all, vals_same_app, mats = [], [], []
                for ia, ib, within in plist:
                    M = Xn[ia] @ Xn[ib].T; same_app = app[ia][:, None] == app[ib][None]
                    if within:
                        mask = np.triu(np.ones_like(M, bool), 1)
                    else:
                        mask = ~same_app
                    vals_all.append(M[mask]); mats.append((ia, ib, within, M, same_app))
                    if not within and same_app.any():
                        vals_same_app.append(M[same_app])
                v = np.concatenate(vals_all)
                # block bootstrap: resample clips within every cell (shared draws)
                bm = np.zeros(B)
                for b in range(B):
                    s, c = 0.0, 0
                    for ia, ib, within, M, same_app in mats:
                        ra, rb = boot[cell_of[int(ia[0])]][b], boot[cell_of[int(ib[0])]][b]
                        Mb = M[ra][:, rb]
                        if within:
                            mask = ra[:, None] != ra[None]; mask = np.triu(mask, 1) | np.tril(mask, -1)   # exclude identical resampled clips
                        else:
                            mask = ~same_app[ra][:, rb]
                        s += Mb[mask].sum(); c += mask.sum()
                    bm[b] = s / max(c, 1)
                g = {"n_pairs": int(len(v)), "n_cell_pairs": len(plist), "cos_mean": float(v.mean()), "cos_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "cos_sd": float(v.std())}
                if vals_same_app:
                    vs = np.concatenate(vals_same_app); g["same_appearance"] = {"n_pairs": int(len(vs)), "cos_mean": float(vs.mean())}
                gres[gname] = g; pooled[gname] = (v, bm)
            # contrasts: within (mean of the two within groups) − cross_diff_a; AUC within > cross; same for stationary control
            def contrast(w1, w2, cr):
                if any(pooled.get(k) is None for k in (w1, w2, cr)):
                    return None
                vw = np.concatenate([pooled[w1][0], pooled[w2][0]]); vc = pooled[cr][0]
                bw = 0.5 * (pooled[w1][1] + pooled[w2][1]); d = bw - pooled[cr][1]
                return {"within_minus_cross": float(vw.mean() - vc.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "auc_within_gt_cross": auc(vw, vc), "n_within": int(len(vw)), "n_cross": int(len(vc))}
            rec[cname] = {"groups": gres, "contrast_diff_a": contrast("within_flat_v_same_cell", "within_flat_a_same_cell", "cross_diff_a"),
                          "contrast_same_a_control": contrast("within_stationary_flat_v", "within_stationary_flat_a", "cross_same_a_stationary")}
        out["features"][fname] = rec
        print(f"    pairs {fname:6s} thr_v={thr_v}  cross_diff_a n={rec['raw']['groups']['cross_diff_a']['n_pairs']}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- ridge tests
def ridge_tests(F_by_feat, Kin, Y, strata, subsets, B, rng, has_a_scores=None):
    """Y: dict label -> (n,) vector.  subsets: {name: idx}.  Returns nested results with bootstrap CIs."""
    res = {}
    for sname, sidx in subsets.items():
        srec = {"n": int(len(sidx)), "n_per_scenario": {}}
        st = strata[sidx]; idxb = boot_idx(len(sidx), B, rng)
        for lab, yfull in Y.items():
            y = yfull[sidx][:, None]; binary = lab == "has_a"
            if binary and not ((y > 0).any() and (y < 0).any()):
                continue
            lrec = {"label_std": float(y.std()), "n_pos": int((y > 0).sum()) if binary else None}
            def summarise(pred, name):
                p = pred[:, 0]
                if binary:
                    ba, au = bacc(p, y[:, 0]), auc(p[y[:, 0] > 0], p[y[:, 0] < 0]); cb, ca = cls_boot(p, y[:, 0], idxb)
                    lrec[name] = {"bacc": ba, "bacc_ci": cb, "auc": au, "auc_ci": ca}
                else:
                    lrec[name] = {"r2": r2(p, y[:, 0]), "r2_ci": r2_boot(p, y[:, 0], idxb), "mae": float(np.abs(p - y[:, 0]).mean())}
            for deg in (2, 3):
                Kp = poly_design(Kin[sidx], deg); pred, _ = cv_predict(Kp, y, st, SEED); summarise(pred, f"shortcut_pos_vel_poly{deg}")
            Kp2 = poly_design(Kin[sidx], 2)
            for fname, F in F_by_feat.items():
                X = F[sidx].astype(np.float64); t1 = time.time()
                pred, lam = cv_predict(X, y, st, SEED, K=Kp2); summarise(pred, f"{fname}_resid_poly2")
                pred, _ = cv_predict(X, y, st, SEED); summarise(pred, f"{fname}_no_resid")
                pred, _ = cv_predict(X, y, st, SEED, K=Kp2, shuffle_labels=True); summarise(pred, f"{fname}_resid_poly2__shuffle")
                print(f"    ridge {sname:18s} {lab:6s} {fname:6s} {time.time()-t1:.0f}s", flush=True)
            srec[lab] = lrec
        res[sname] = srec
    return res


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per-scenario", type=int, default=784); ap.add_argument("--boot", type=int, default=300); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = a.boot
    print("loading clips", flush=True); clips = load_clips(a.n_per_scenario); n = len(clips)
    scen = np.array([r["scenario"] for r in clips]); vids = [r["video_id"] for r in clips]
    labs = [labels_of(r)[0] for r in clips]
    feat_file = OUT / "features.npz"
    if feat_file.is_file() and list(np.load(feat_file, allow_pickle=True)["video_id"]) == vids:
        z = np.load(feat_file, allow_pickle=True); Fz, Fbg, Fh7, Fp0, win_ok = (z[k] for k in ["Fz", "Fbg", "Fh7", "Fp0", "win_ok"]); print("features from cache", flush=True)
    else:
        print("extracting features", flush=True); Fz, Fbg, Fh7, Fp0, win_ok = extract(clips)
        np.savez(feat_file, video_id=np.array(vids), scenario=scen, Fz=Fz, Fbg=Fbg, Fh7=Fh7, Fp0=Fp0, win_ok=win_ok)
    print(f"n={n}  extract done {time.time()-t0:.0f}s  full 3x3 windows: {int(win_ok.sum())}/{n}", flush=True)
    F_by_feat = {"F1": Fz[:, 7], "F2": np.concatenate([Fz[:, 7], Fz[:, 6]], 1), "F3": Fz.reshape(n, -1), "F1_bg": Fbg, "H7": Fh7, "P0": Fp0}
    Kin = np.stack([[l["x7"], l["y7"], l["vx7"], l["vy7"]] for l in labs]); app = np.array([f"{r['shape_pre']}|{r['color_pre']}|{r['env']}" for r in clips])
    Y = {"ax7": np.array([l["ax7"] for l in labs]), "a_nom": np.array([l["a_nom"] for l in labs]), "has_a": np.array([1.0 if l["a_nom"] != 0 else -1.0 for l in labs])}
    # cells
    keys = [cell_key(l, s) for l, s in zip(labs, scen)]; cells = {}
    for i, k in enumerate(keys):
        cells.setdefault(k, []).append(i)
    cells = {k: np.array(v) for k, v in cells.items()}
    cell_lab = {}; cell_table = []
    for k, v in cells.items():
        sub = {q: np.array([labs[i][q] for i in v]) for q in ("x7", "y7", "vx7", "vy7", "ax7", "x0", "a_nom", "v_nom")}
        cell_lab[k] = {q: float(sub[q].mean()) for q in sub}
        cell_table.append({"scenario": k[0], "primary": k[1], "anchor": k[2], "n": int(len(v)), **{q: float(sub[q].mean()) for q in ("x7", "vx7", "ax7", "x0")}, "label_std_max": float(max(sub[q].std() for q in ("x7", "vx7", "ax7")))})
    strata = np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in keys])
    rng = np.random.default_rng(SEED)

    # ---- (1) pair tests at the spec threshold and at the loose one
    print("pair test", flush=True)
    pairs = {"thr_v_0.05": pair_test(F_by_feat, cells, cell_lab, scen, app, THR_V, B, np.random.default_rng(SEED)),
             "thr_v_0.06_sensitivity": pair_test(F_by_feat, cells, cell_lab, scen, app, THR_V_LOOSE, B, np.random.default_rng(SEED))}

    # ---- subsets for ridge
    def cells_in(cp, kind):
        s = set()
        for c in cp:
            if c["kind"] == kind:
                s.add(tuple(c["flat_v"])); s.add(tuple(c["flat_a"]))
        return s
    def idx_of(keyset):
        return np.array(sorted(i for i, k in enumerate(keys) if (k[0], str(k[1]), str(k[2])) in keyset))
    strict_diff = cells_in(pairs["thr_v_0.05"]["cell_pairs"], "cross_diff_a"); strict_same = cells_in(pairs["thr_v_0.05"]["cell_pairs"], "cross_same_a")
    loose_diff = cells_in(pairs["thr_v_0.06_sensitivity"]["cell_pairs"], "cross_diff_a")
    subsets = {"union_all": np.arange(n), "matched_strict": idx_of(strict_diff), "matched_strict+stationary": idx_of(strict_diff | strict_same), "matched_loose": idx_of(loose_diff)}
    # balanced binary subset for the union: negatives (flat_v all + flat_a a=0) subsampled to the positive count, stratified by cell
    pos = np.where(Y["has_a"] > 0)[0]; neg = np.where(Y["has_a"] < 0)[0]; rs = np.random.default_rng(SEED)
    neg_cells = {}
    for i in neg:
        neg_cells.setdefault(strata[i], []).append(i)
    per = len(pos) // len(neg_cells); neg_sel = sorted(sum([list(rs.choice(v, min(per, len(v)), replace=False)) for v in neg_cells.values()], []))
    subsets["union_balanced_binary"] = np.array(sorted(list(pos) + neg_sel))
    print("ridge tests", flush=True)
    ridge = {}
    ridge["continuous"] = ridge_tests(F_by_feat, Kin, {"ax7": Y["ax7"], "a_nom": Y["a_nom"]}, strata, {k: subsets[k] for k in ("union_all", "matched_strict", "matched_strict+stationary", "matched_loose")}, B, rng)
    ridge["binary_has_a"] = ridge_tests(F_by_feat, Kin, {"has_a": Y["has_a"]}, strata, {k: subsets[k] for k in ("union_balanced_binary", "matched_strict", "matched_loose")}, B, rng)
    # stationary control: scenario classification flat_v v=0 vs flat_a a=0 (identical physics) — must be chance
    stat_idx = idx_of(strict_same); Ystat = np.where(scen == "flat_a", 1.0, -1.0)
    ridge["stationary_scenario_control"] = ridge_tests(F_by_feat, Kin, {"has_a": Ystat}, strata, {"stationary_v0_vs_a0": stat_idx}, B, rng) if len(stat_idx) else {"skipped": "no stationary cells"}

    rep = {"config": {"index": str(INDEX), "cache_z16": str(CACHE_Z), "cache_hp": str(CACHE_HP), "scenarios": SCEN, "n_per_scenario_requested": a.n_per_scenario, "seed": SEED, "k_fold": K_FOLD, "val_frac": VAL_FRAC, "lambdas": LAMBDAS,
                      "thr_pos_cells": THR_POS, "thr_v_cells_per_tubelet": THR_V, "thr_v_loose": THR_V_LOOSE, "n_boot": B, "cell_px": CELL,
                      "units": "cells (1 cell = 18 px); velocity = cells / tubelet (6 raw frames); acceleration = cells / tubelet^2; a_nom in cm/s^2 (flat_v: 0)",
                      "features": {"F1": "z16 t7 3x3 @pos7 (1280)", "F2": "z16 t7+t6 (2560)", "F3": "z16 t0..t7 (10240)", "F1_bg": "z16 t7 3x3 >= 4 cells from object", "H7": "target enc t7 3x3 @pos7", "P0": "predictor slot 0 3x3 @pos8"},
                      "feature_dims": {k: int(v.shape[1]) for k, v in F_by_feat.items()}, "residualiser": "OLS of each feature dim on [1, poly2(x7, y7, vx7, vy7)] fitted on the training fold (constant columns dropped)",
                      "bootstrap": "clip-level (block_id = clip) resampling, percentile 95 % CI; pair test resamples clips within every cell with shared draws so within-vs-cross differences are paired"},
           "n_clips": int(n), "n_per_scenario": {s: int((scen == s).sum()) for s in SCEN}, "n_full_3x3_windows": int(win_ok.sum()),
           "cells": sorted(cell_table, key=lambda c: (c["scenario"], c["primary"], c["anchor"])), "subset_sizes": {k: int(len(v)) for k, v in subsets.items()},
           "pair_test": pairs, "ridge": ridge, "elapsed_s": float(time.time() - t0)}
    (OUT / "results.json").write_text(json.dumps(rep, indent=1)); write_md(rep); print(f"→ {OUT}/results.json, RESULTS.md  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------------------------- markdown (generated from the json only)
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def ci(g, k):
    return f"{fmt(g[k])} [{fmt(g[k+'_ci'][0])}, {fmt(g[k+'_ci'][1])}]" if g.get(k + "_ci") else fmt(g.get(k))


def write_md(rep):
    c = rep["config"]
    L = [f"# Acceleration at the boundary tubelet — cross-scenario matched pairs (RollOut_v2 flat_v × flat_a, n={rep['n_clips']}, {rep['n_per_scenario']})", "",
         "Question: does z16 tubelet 7 (predictor input, F1) carry acceleration beyond position + velocity? Clips are matched ACROSS flat_v (a = 0) and flat_a (v0 = 0) on (pos7, v7); only the history t0..t6 differs.",
         f"Units: {c['units']}. Bootstrap: {c['bootstrap']} ({c['n_boot']} resamples). Feature dims: " + ", ".join(f"{k} {v}" for k, v in c["feature_dims"].items()) + ".", "",
         "## Cells (trajectory identical within a cell — label std is 0 — so a cell is one (pos7, v7, a) point with 56 appearance variants)", "",
         "| scenario | primary | anchor | n | x7 | vx7 | ax7 | x0 | max label std |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cl in rep["cells"]:
        L.append(f"| {cl['scenario']} | {cl['primary']:.0f} | {cl['anchor']:.0f} | {cl['n']} | {cl['x7']:.3f} | {cl['vx7']:.4f} | {cl['ax7']:.4f} | {cl['x0']:.2f} | {cl['label_std_max']:.1e} |")
    for tname, pt in rep["pair_test"].items():
        L += ["", f"## (1) Parameter-free pair test — {tname} (|Δpos7| ≤ {c['thr_pos_cells']}, |Δv7| ≤ {tname.split('_')[2]})", "", "Matched cell pairs:", "",
              "| kind | flat_v cell (v, anchor) | flat_a cell (a, anchor) | Δx7 | Δvx7 | Δx0 (history gap) |", "|---|---|---|---:|---:|---:|"]
        for cp in pt["cell_pairs"]:
            L.append(f"| {cp['kind']} | v={cp['v_nom']:.0f}, {cp['anchor']:.0f} | a={cp['a_nom']:.0f}, {cp['anchor']:.0f} | {cp['d_x7']:.4f} | {cp['d_vx7']:.4f} | {cp['d_x0']:.2f} |")
        L += ["", "Mean cosine per pair group (appearance-mismatched pairs only; within-cell pairs are always appearance-mismatched). `cross_diff_a` = matched across scenarios, differ in |a|; `within_*_same_cell` = same cell (same a, same trajectory); `cross_same_a_stationary` = v=0 vs a=0 (identical physics, cross-scenario control); yardsticks = same scenario, pos-matched, |Δv7| 0.10–0.25.", ""]
        for fname, rec in pt["features"].items():
            L += [f"### {fname}", "", "| centering | " + " | ".join(f"{g} (n)" for g in rec["raw"]["groups"]) + " | within − cross (diff-a) [CI], AUC | within − cross (same-a control) [CI], AUC |", "|---|" + "---|" * len(rec["raw"]["groups"]) + "---|---|"]
            for cname, r in rec.items():
                cells_ = " | ".join((f"{ci(g, 'cos')} ({g['n_pairs']})" if g.get("n_pairs") else "—") for g in r["groups"].values())
                cd, cs = r["contrast_diff_a"], r["contrast_same_a_control"]
                fc = lambda x: "—" if x is None else f"{x['within_minus_cross']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}], AUC {x['auc_within_gt_cross']:.3f}"
                L.append(f"| {cname} | {cells_} | {fc(cd)} | {fc(cs)} |")
            sa = [(g, r["same_appearance"]) for g, r in rec["raw"]["groups"].items() if r.get("same_appearance")]
            if sa:
                L.append(""); L.append("Same-appearance cross pairs (identical shape/color/env, raw cosine): " + "; ".join(f"{g} n={s['n_pairs']} cos {s['cos_mean']:.4f}" for g, s in sa))
            L.append("")
    L += ["## (2) Ridge — continuous acceleration labels (out-of-fold R² [block-bootstrap 95 % CI] / MAE)", "",
          "`*_resid_poly2` = features residualised on poly2(x7, y7, vx7, vy7) inside each training fold; `*_no_resid` = raw features; `shortcut_pos_vel_polyN` = position + velocity only. Label std per subset shows the scale.", ""]
    for sname, s in rep["ridge"]["continuous"].items():
        L += [f"### subset `{sname}` (n={s['n']})", "", "| model | ax7 R² [CI] | ax7 MAE | a_nom R² [CI] | a_nom MAE |", "|---|---|---:|---|---:|"]
        models = [m for m in s["ax7"] if m not in ("label_std", "n_pos")]
        L.append(f"| (label std) | {s['ax7']['label_std']:.4f} | | {s['a_nom']['label_std']:.2f} | |")
        for m in models:
            L.append(f"| {m} | {ci(s['ax7'][m], 'r2')} | {s['ax7'][m]['mae']:.4f} | {ci(s['a_nom'][m], 'r2')} | {s['a_nom'][m]['mae']:.2f} |")
        L.append("")
    L += ["## (3) Ridge — binary 'has nonzero acceleration' (balanced accuracy [CI] / AUC [CI]; chance 0.5)", ""]
    for sname, s in rep["ridge"]["binary_has_a"].items():
        L += [f"### subset `{sname}` (n={s['n']}, positives {s['has_a']['n_pos']})", "", "| model | bacc [CI] | AUC [CI] |", "|---|---|---|"]
        for m, r in s["has_a"].items():
            if m in ("label_std", "n_pos"):
                continue
            L.append(f"| {m} | {ci(r, 'bacc')} | {ci(r, 'auc')} |")
        L.append("")
    sc = rep["ridge"]["stationary_scenario_control"]
    L += ["### Stationary control — classify scenario (flat_a a=0 vs flat_v v=0; identical physics, must be chance)", ""]
    if "skipped" in sc:
        L.append(sc["skipped"])
    else:
        s = sc["stationary_v0_vs_a0"]; L += [f"n={s['n']}", "", "| model | bacc [CI] | AUC [CI] |", "|---|---|---|"]
        for m, r in s["has_a"].items():
            if m in ("label_std", "n_pos"):
                continue
            L.append(f"| {m} | {ci(r, 'bacc')} | {ci(r, 'auc')} |")
    L += ["", "## 재현", "", "```", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2", f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_accel_crosspair.py --n-per-scenario {c['n_per_scenario_requested']} --boot {c['n_boot']}", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
