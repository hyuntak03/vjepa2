#!/usr/bin/env python3
"""E1 — does the predictor's INPUT (context encoder, 16 context frames = z16) carry kinematic state (velocity, acceleration) at the
boundary token (tubelet 7), and does it pass into the predictor's output (p slot 0 = tubelet 8)?  RollOut_v2 plausible clips.

Labels (cell units, 1 cell = 18 px on the 288-px render; tubelet position = mean of its two samples):
  pos7 = (x7, y7); v7 = pos7 - pos6; a7 = v7 - (pos6 - pos5); v8 = pos8 - pos7.   x / y separately.
Feature sets (3x3 token mean around the TRUE object position — window code copied from v11_token_object_test.py::n3):
  F1  z16 tubelet 7 (1280)            F2  z16 tubelets 7+6 (2560)        F3  z16 tubelets 0..7 (10240)
  F1h h tubelet 7 (ceiling, saw 32 frames)   F1h8 h tubelet 8            P0  p slot 0 = tubelet 8, window at pos8
  F1_bg z16 tubelet 7 at a 3x3 window >= 4 cells away from the object (same clip; global/scene-context control)
Baselines in the same tables: label shuffle (train labels permuted), position shortcuts POS7 / POS7q (quadratic) / POS67 / POS0-7,
  scenario one-hot SCEN, SCEN+POS7q.
Ridge: closed form via SVD, per-dim standardisation, lambda picked per label on a validation fold (25 % of the training blocks),
  outer 5-fold CV by block_id stratified by scenario, seed 0.  Out-of-fold predictions -> R^2 / MAE pooled and per scenario.
  Two fit modes: 'pooled' (one model over all scenarios) and 'per_scenario' (model fit inside each scenario).
Parameter-free check: within scenario, clip pairs with |d pos7| < 0.3 cell, split |d v7| < 0.1 (similar) vs > 0.3 (different);
  mean cosine similarity of the F1 vectors with bootstrap 95 % CI over pairs.  Acceleration: pairs matched on pos7 AND v7,
  split by |d a7| relative to the scenario's a7 std (scenarios with a7 std > 0.005 only).
No GPU, no model; caches read one clip at a time (mmap).  Deterministic.
Outputs: z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/{results.json, RESULTS.md, features.npz}
Run:
  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_boundary_kinematics.py [n_per_scenario=300]
"""
import json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
INDEX = ROOT / "data_csv/rollout_v2/index.csv"
CACHE_Z = Path("/local_datasets/world/world_analysis/cache/rollout_v2_z16_vith")
CACHE_HP = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
OUT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics"
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge"]
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED, K_FOLD, VAL_FRAC = 0, 5, 0.25
LAMBDAS = [float(v) for v in np.logspace(-2, 6, 17)]
CELL, RES, D = 18.0, 288, 1280
LABELS = ["x7", "y7", "vx7", "vy7", "ax7", "ay7", "vx8", "vy8"]
DEGEN_STD = 1e-3          # label std (cells) below which per-scenario R^2 is reported as None (degenerate target)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def n3(x, y):
    """copied verbatim from z_research/scripts/analysis/v11_token_object_test.py — 3x3 token window around px (x, y)."""
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


def bg_cell(cx, cy, rng):
    """a token cell at Chebyshev distance >= 4 from (cx, cy), chosen deterministically; returns its centre px."""
    cand = [(c, r) for r in range(1, 15) for c in range(1, 15) if max(abs(c - cx), abs(r - cy)) >= 4]
    c, r = cand[rng.integers(len(cand))]; return c * CELL + 9, r * CELL + 9


# ----------------------------------------------------------------------------------------------- data + labels
def load_clips():
    import csv
    rows = [r for r in csv.DictReader(INDEX.open()) if r["plausible"] == "1" and r["scenario"] in SCEN]
    rng = np.random.default_rng(SEED); keep = []
    for scn in SCEN:
        rs = [r for r in rows if r["scenario"] == scn]
        # validity: tubelets 5..8 (samples 10..17) in frame and inside the render
        ok = []
        for r in rs:
            px, py, inf = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"]), arr(r["in_frame_by_sample"])
            tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)
            if inf[10:18].min() > 0 and (tx[5:9] >= 0).all() and (tx[5:9] < RES).all() and (ty[5:9] >= 0).all() and (ty[5:9] < RES).all():
                ok.append(r)
        sel = rng.choice(len(ok), min(N_PER, len(ok)), replace=False)
        keep += [ok[i] for i in sorted(sel)]
        print(f"  {scn}: {len(rs)} plausible, {len(ok)} valid, {len(sel)} sampled", flush=True)
    return keep


def labels_of(r):
    px, py = arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])
    tx, ty = px.reshape(16, 2).mean(1), py.reshape(16, 2).mean(1)          # px per tubelet
    cx, cy = tx / CELL, ty / CELL                                            # cells (continuous)
    vx7, vy7 = cx[7] - cx[6], cy[7] - cy[6]; ax7, ay7 = vx7 - (cx[6] - cx[5]), vy7 - (cy[6] - cy[5])
    return dict(x7=cx[7], y7=cy[7], vx7=vx7, vy7=vy7, ax7=ax7, ay7=ay7, vx8=cx[8] - cx[7], vy8=cy[8] - cy[7],
                x6=cx[6], y6=cy[6], x8=cx[8], y8=cy[8], pos_x=cx[:8], pos_y=cy[:8]), tx, ty


def extract(clips):
    """features per clip: z16 3x3 means at tubelets 0..7, bg window at t7, h at t7/t8, p slot 0 at t8."""
    meta_z = json.loads((CACHE_Z / "meta.json").read_text()); meta_hp = json.loads((CACHE_HP / "meta.json").read_text())
    row_z = {v: i for i, v in enumerate(meta_z["video_ids"])}; row_hp = {v: i for i, v in enumerate(meta_hp["video_ids"])}
    assert meta_z["base_counts"]["ctx_masked"] == 2048 and meta_hp["base_counts"]["target"] == 4096 and meta_hp["base_counts"]["predictor"] == 2048
    Z = np.load(CACHE_Z / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE_HP / "target.npy", mmap_mode="r"); P = np.load(CACHE_HP / "predictor.npy", mmap_mode="r")
    n = len(clips); Fz = np.zeros((n, 8, D), np.float32); Fbg = np.zeros((n, D), np.float32); Fh7 = np.zeros((n, D), np.float32); Fh8 = np.zeros((n, D), np.float32); Fp0 = np.zeros((n, D), np.float32)
    Y = np.zeros((n, len(LABELS))); POS = np.zeros((n, 8, 2)); rng = np.random.default_rng(SEED); win_sz = np.zeros((n, 9), np.int8)
    order = np.argsort([row_z[r["video_id"]] for r in clips]); t0 = time.time()
    for j, i in enumerate(order):
        r = clips[i]; lab, tx, ty = labels_of(r)
        Y[i] = [lab[k] for k in LABELS]; POS[i, :, 0] = lab["pos_x"]; POS[i, :, 1] = lab["pos_y"]
        rz, rh = row_z[r["video_id"]], row_hp[r["video_id"]]
        z = np.asarray(Z[rz], np.float32).reshape(8, 256, D)
        for t in range(8):
            w = n3(tx[t], ty[t]); win_sz[i, t] = len(w); Fz[i, t] = z[t, w].mean(0) if w else 0.0
        w8 = n3(tx[8], ty[8]); win_sz[i, 8] = len(w8)
        bx, by = bg_cell(int(tx[7] // CELL), int(ty[7] // CELL), rng); Fbg[i] = z[7, n3(bx, by)].mean(0)
        h = np.asarray(H[rh], np.float32).reshape(16, 256, D); Fh7[i] = h[7, n3(tx[7], ty[7])].mean(0); Fh8[i] = h[8, w8].mean(0)
        p = np.asarray(P[rh], np.float32).reshape(8, 256, D); Fp0[i] = p[0, w8].mean(0)
        if j % 200 == 0:
            print(f"    extract {j}/{n} {time.time()-t0:.0f}s", flush=True)
    return Fz, Fbg, Fh7, Fh8, Fp0, Y, POS, win_sz


# ----------------------------------------------------------------------------------------------- ridge (closed form, SVD)
def ridge_fit_predict(Xtr, Ytr, Xte, lams):
    """standardise on train, SVD once, return {lam: Yte_pred} for every lam."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xc = (Xtr - mu) / sd; ym = Ytr.mean(0); Yc = Ytr - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); UtY = U.T @ Yc; Xte_c = (Xte - mu) / sd
    out = {}
    for lam in lams:
        coef = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY); out[lam] = Xte_c @ coef + ym
    return out


def strat_folds(scen, k, rng):
    fold = np.zeros(len(scen), int)
    for scn in np.unique(scen):
        idx = np.where(scen == scn)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % k
    return fold


def cv_predict(X, Y, scen, seed, shuffle_labels=False):
    """out-of-fold predictions; lambda per label chosen on an inner validation split of each training fold."""
    rng = np.random.default_rng(seed); fold = strat_folds(scen, K_FOLD, rng); pred = np.zeros_like(Y); lam_used = []
    for f in range(K_FOLD):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        Ytr = Y[tr].copy()
        if shuffle_labels:
            Ytr = Ytr[rng.permutation(len(tr))]
        inner = strat_folds(scen[tr], int(round(1 / VAL_FRAC)), rng); itr, iva = tr[inner != 0], tr[inner == 0]
        Ytr_i = Ytr[inner != 0]; Yva_i = Ytr[inner == 0]
        pv = ridge_fit_predict(X[itr], Ytr_i, X[iva], LAMBDAS)
        best = [min(LAMBDAS, key=lambda l: ((pv[l][:, j] - Yva_i[:, j]) ** 2).mean()) for j in range(Y.shape[1])]
        pt = ridge_fit_predict(X[tr], Ytr, X[te], sorted(set(best)))
        for j, l in enumerate(best):
            pred[te, j] = pt[l][:, j]
        lam_used.append(best)
    return pred, lam_used


def r2(p, y):
    sst = ((y - y.mean()) ** 2).sum(); return None if y.std() < DEGEN_STD else float(1 - ((p - y) ** 2).sum() / sst)


def score(pred, Y, scen):
    res = {}
    for j, lab in enumerate(LABELS):
        rec = {"pooled": {"r2": r2(pred[:, j], Y[:, j]), "mae": float(np.abs(pred[:, j] - Y[:, j]).mean()), "n": int(len(Y)), "label_std": float(Y[:, j].std())}, "per_scenario": {}}
        for scn in SCEN:
            m = scen == scn
            if m.sum() == 0:
                continue
            rec["per_scenario"][scn] = {"r2": r2(pred[m, j], Y[m, j]), "mae": float(np.abs(pred[m, j] - Y[m, j]).mean()), "n": int(m.sum()), "label_std": float(Y[m, j].std())}
        res[lab] = rec
    return res


# ----------------------------------------------------------------------------------------------- parameter-free pair check
def boot_ci(v, rng, B=2000):
    if len(v) == 0:
        return [None, None]
    idx = rng.integers(len(v), size=(B, len(v))); m = v[idx].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pair_check(F, pos, vel, acc, scen, name, rng, a_std_min=0.005):
    """F (n, D); pos/vel/acc (n, 2) in cells.  Returns per-scenario cosine stats for velocity and (where defined) acceleration."""
    out = {}
    for scn in SCEN:
        m = np.where(scen == scn)[0]
        if len(m) < 2:
            continue
        X = F[m]; Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); C = Xn @ Xn.T
        Xc = X - X.mean(0); Xcn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8); Cc = Xcn @ Xcn.T
        iu = np.triu_indices(len(m), 1)
        dpos = np.linalg.norm(pos[m][:, None] - pos[m][None], axis=-1)[iu]; dv = np.linalg.norm(vel[m][:, None] - vel[m][None], axis=-1)[iu]
        da = np.linalg.norm(acc[m][:, None] - acc[m][None], axis=-1)[iu]; c, cc = C[iu], Cc[iu]
        near = dpos < 0.3; sim, dif = near & (dv < 0.1), near & (dv > 0.3)
        rec = {"n_clips": int(len(m)), "n_pairs_all": int(len(c)), "n_pairs_pos_matched": int(near.sum()),
               "velocity": {"thr": {"pos": 0.3, "sim_v": 0.1, "dif_v": 0.3},
                            "similar_v": {"n": int(sim.sum()), "cos_raw": float(c[sim].mean()) if sim.any() else None, "cos_raw_ci": boot_ci(c[sim], rng), "cos_centered": float(cc[sim].mean()) if sim.any() else None, "cos_centered_ci": boot_ci(cc[sim], rng)},
                            "different_v": {"n": int(dif.sum()), "cos_raw": float(c[dif].mean()) if dif.any() else None, "cos_raw_ci": boot_ci(c[dif], rng), "cos_centered": float(cc[dif].mean()) if dif.any() else None, "cos_centered_ci": boot_ci(cc[dif], rng)},
                            "all_pairs": {"n": int(len(c)), "cos_raw": float(c.mean()), "cos_centered": float(cc.mean())}}}
        a_std = float(np.sqrt((acc[m].var(0)).sum()))
        if a_std > a_std_min:
            matched = near & (dv < 0.1); s_a, d_a = matched & (da < 0.25 * a_std), matched & (da > 1.0 * a_std)
            rec["acceleration"] = {"a7_std_cells": a_std, "thr": {"pos": 0.3, "v": 0.1, "sim_a": 0.25 * a_std, "dif_a": 1.0 * a_std}, "n_pairs_pos_v_matched": int(matched.sum()),
                                   "similar_a": {"n": int(s_a.sum()), "cos_raw": float(c[s_a].mean()) if s_a.any() else None, "cos_raw_ci": boot_ci(c[s_a], rng), "cos_centered": float(cc[s_a].mean()) if s_a.any() else None, "cos_centered_ci": boot_ci(cc[s_a], rng)},
                                   "different_a": {"n": int(d_a.sum()), "cos_raw": float(c[d_a].mean()) if d_a.any() else None, "cos_raw_ci": boot_ci(c[d_a], rng), "cos_centered": float(cc[d_a].mean()) if d_a.any() else None, "cos_centered_ci": boot_ci(cc[d_a], rng)}}
        else:
            rec["acceleration"] = {"a7_std_cells": a_std, "skipped": "a7 std below threshold (degenerate)"}
        out[scn] = rec
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    print("loading clips", flush=True); clips = load_clips(); n = len(clips)
    scen = np.array([r["scenario"] for r in clips]); vids = [r["video_id"] for r in clips]
    feat_file = OUT / "features.npz"
    if feat_file.is_file() and list(np.load(feat_file, allow_pickle=True)["video_id"]) == vids:
        z = np.load(feat_file, allow_pickle=True); Fz, Fbg, Fh7, Fh8, Fp0, Y, POS, win_sz = (z[k] for k in ["Fz", "Fbg", "Fh7", "Fh8", "Fp0", "Y", "POS", "win_sz"]); print("features from cache", flush=True)
    else:
        print("extracting features", flush=True); Fz, Fbg, Fh7, Fh8, Fp0, Y, POS, win_sz = extract(clips)
        np.savez(feat_file, video_id=np.array(vids), scenario=scen, Fz=Fz, Fbg=Fbg, Fh7=Fh7, Fh8=Fh8, Fp0=Fp0, Y=Y, POS=POS, win_sz=win_sz)
    print(f"n={n}  extract done {time.time()-t0:.0f}s", flush=True)

    # ---- label stats per scenario
    label_stats = {scn: {lab: {"mean": float(Y[scen == scn, j].mean()), "std": float(Y[scen == scn, j].std()), "n": int((scen == scn).sum())} for j, lab in enumerate(LABELS)} for scn in SCEN}
    label_stats["pooled"] = {lab: {"mean": float(Y[:, j].mean()), "std": float(Y[:, j].std()), "n": int(n)} for j, lab in enumerate(LABELS)}

    # ---- feature sets and baselines
    onehot = (scen[:, None] == np.array(SCEN)[None]).astype(float)
    x7, y7, x6, y6 = POS[:, 7, 0], POS[:, 7, 1], POS[:, 6, 0], POS[:, 6, 1]
    pos7q = np.stack([x7, y7, x7 ** 2, y7 ** 2, x7 * y7], 1)
    sets = {"F1": Fz[:, 7].astype(np.float64), "F2": np.concatenate([Fz[:, 7], Fz[:, 6]], 1).astype(np.float64), "F3": Fz.reshape(n, -1).astype(np.float64),
            "F1h": Fh7.astype(np.float64), "F1h8": Fh8.astype(np.float64), "P0": Fp0.astype(np.float64), "F1_bg": Fbg.astype(np.float64),
            "POS7": np.stack([x7, y7], 1), "POS7q": pos7q, "POS67": np.stack([x7, y7, x6, y6], 1), "POS0-7": POS.reshape(n, -1), "SCEN": onehot, "SCEN+POS7q": np.concatenate([onehot, pos7q], 1)}
    dims = {k: int(v.shape[1]) for k, v in sets.items()}
    results = {"pooled_fit": {}, "per_scenario_fit": {}, "lambda": {}}
    for name, X in sets.items():
        t1 = time.time(); pred, lam = cv_predict(X, Y, scen, SEED); results["pooled_fit"][name] = score(pred, Y, scen); results["lambda"][name] = lam
        if name in ("F1", "F2", "F3", "F1h", "P0", "F1_bg", "SCEN+POS7q"):
            pred_s, _ = cv_predict(X, Y, scen, SEED, shuffle_labels=True); results["pooled_fit"][name + "__shuffle"] = score(pred_s, Y, scen)
        # per-scenario fit: model trained inside each scenario (removes scenario identity and cross-scenario variance)
        if name not in ("SCEN", "SCEN+POS7q"):
            pred_ps = np.zeros_like(Y)
            for scn in SCEN:
                m = np.where(scen == scn)[0]; pr, _ = cv_predict(X[m], Y[m], scen[m], SEED); pred_ps[m] = pr
            results["per_scenario_fit"][name] = score(pred_ps, Y, scen)
            if name in ("F1", "F2", "P0"):
                pred_ps = np.zeros_like(Y)
                for scn in SCEN:
                    m = np.where(scen == scn)[0]; pr, _ = cv_predict(X[m], Y[m], scen[m], SEED, shuffle_labels=True); pred_ps[m] = pr
                results["per_scenario_fit"][name + "__shuffle"] = score(pred_ps, Y, scen)
        print(f"  ridge {name:12s} d={dims[name]:6d}  {time.time()-t1:.0f}s", flush=True)

    # ---- parameter-free pair check
    rng = np.random.default_rng(SEED)
    vel7 = Y[:, [2, 3]]; acc7 = Y[:, [4, 5]]; vel8 = Y[:, [6, 7]]; pos7 = POS[:, 7]; pos8 = np.stack([Y[:, 0] + Y[:, 6], Y[:, 1] + Y[:, 7]], 1)
    pairs = {"F1": pair_check(Fz[:, 7], pos7, vel7, acc7, scen, "F1", rng), "F2": pair_check(np.concatenate([Fz[:, 7], Fz[:, 6]], 1), pos7, vel7, acc7, scen, "F2", rng),
             "F1h": pair_check(Fh7, pos7, vel7, acc7, scen, "F1h", rng), "F1_bg": pair_check(Fbg, pos7, vel7, acc7, scen, "F1_bg", rng),
             "P0 (matched on pos8, split on v8 / a7)": pair_check(Fp0, pos8, vel8, acc7, scen, "P0", rng)}

    rep = {"config": {"index": str(INDEX), "cache_z16": str(CACHE_Z), "cache_hp": str(CACHE_HP), "scenarios": SCEN, "n_per_scenario_max": N_PER, "seed": SEED, "k_fold": K_FOLD, "val_frac": VAL_FRAC,
                      "lambdas": LAMBDAS, "cell_px": CELL, "labels": LABELS, "units": "cells (1 cell = 18 px); velocity = cells / tubelet (2 sampled frames = 6 raw frames); acceleration = cells / tubelet^2",
                      "window": "3x3 token mean around true tubelet position, n3() copied from v11_token_object_test.py; bg window >= 4 cells (Chebyshev) from the object",
                      "feature_dims": dims, "degenerate_label_std": DEGEN_STD},
           "n_clips": int(n), "n_per_scenario": {scn: int((scen == scn).sum()) for scn in SCEN}, "window_size_counts": {f"t{t}": {str(k): int(v) for k, v in zip(*np.unique(win_sz[:, t], return_counts=True))} for t in range(9)},
           "label_stats": label_stats, "ridge": results, "pair_check": pairs, "elapsed_s": float(time.time() - t0)}
    (OUT / "results.json").write_text(json.dumps(rep, indent=1)); write_md(rep); print(f"→ {OUT}/results.json, RESULTS.md  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------------------------- markdown
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def write_md(rep):
    L = [f"# E1 — kinematic state at the boundary token (RollOut_v2 plausible, n={rep['n_clips']}, {rep['n_per_scenario']})", "",
         "Units: cells (1 cell = 18 px). Velocity = cells/tubelet, acceleration = cells/tubelet². R² = None (—) where the label std < 1e-3 (degenerate).",
         "Out-of-fold predictions, 5-fold CV by block (stratified by scenario, seed 0); λ per label on an inner 25 % validation split. Feature dims: " + ", ".join(f"{k} {v}" for k, v in rep["config"]["feature_dims"].items()), ""]
    L += ["## Label statistics (std in cells)", "", "| scenario | n | " + " | ".join(LABELS) + " |", "|---|---:|" + "---:|" * len(LABELS)]
    for scn in SCEN + ["pooled"]:
        s = rep["label_stats"][scn]; L.append(f"| {scn} | {s[LABELS[0]]['n']} | " + " | ".join(f"{s[l]['std']:.3f}" for l in LABELS) + " |")
    for mode in ("pooled_fit", "per_scenario_fit"):
        R = rep["ridge"][mode]
        L += ["", f"## Ridge — {mode.replace('_', ' ')} (R² pooled over all clips / MAE in cells)", "", "| feature set | " + " | ".join(f"{l} R² | {l} MAE" for l in LABELS) + " |", "|---|" + "---:|---:|" * len(LABELS)]
        for name, res in R.items():
            L.append(f"| {name} | " + " | ".join(f"{fmt(res[l]['pooled']['r2'])} | {fmt(res[l]['pooled']['mae'])}" for l in LABELS) + " |")
        for scn in SCEN:
            n = rep["n_per_scenario"][scn]
            L += ["", f"### {mode.replace('_', ' ')} — scenario {scn} (n={n}; R² relative to the scenario's own label variance)", "", "| feature set | " + " | ".join(f"{l} R² | {l} MAE" for l in LABELS) + " |", "|---|" + "---:|---:|" * len(LABELS)]
            for name, res in R.items():
                L.append(f"| {name} | " + " | ".join(f"{fmt(res[l]['per_scenario'][scn]['r2'])} | {fmt(res[l]['per_scenario'][scn]['mae'])}" for l in LABELS) + " |")
    L += ["", "## Parameter-free pair check — mean cosine similarity (raw / scenario-centered) with bootstrap 95 % CI over pairs", ""]
    ci = lambda g, k: f"{fmt(g[k])} [{fmt(g[k+'_ci'][0])}, {fmt(g[k+'_ci'][1])}]"
    for fname, PC in rep["pair_check"].items():
        L += [f"### {fname}", "", "| scenario | clips | pairs pos-matched | similar-v n | cos raw [CI] | cos centered [CI] | different-v n | cos raw [CI] | cos centered [CI] | all-pairs cos raw / centered |", "|---|---:|---:|---:|---|---|---:|---|---|---|"]
        for scn, r in PC.items():
            v = r["velocity"]; s, d = v["similar_v"], v["different_v"]
            L.append(f"| {scn} | {r['n_clips']} | {r['n_pairs_pos_matched']} | {s['n']} | {ci(s,'cos_raw')} | {ci(s,'cos_centered')} | {d['n']} | {ci(d,'cos_raw')} | {ci(d,'cos_centered')} | {fmt(v['all_pairs']['cos_raw'])} / {fmt(v['all_pairs']['cos_centered'])} |")
        L += ["", "| scenario | a7 std | pairs pos+v matched | similar-a n | cos raw [CI] | cos centered [CI] | different-a n | cos raw [CI] | cos centered [CI] |", "|---|---:|---:|---:|---|---|---:|---|---|"]
        for scn, r in PC.items():
            a = r["acceleration"]
            if "skipped" in a:
                L.append(f"| {scn} | {a['a7_std_cells']:.4f} | — | — | skipped (degenerate) | | | | |"); continue
            s, d = a["similar_a"], a["different_a"]
            L.append(f"| {scn} | {a['a7_std_cells']:.4f} | {a['n_pairs_pos_v_matched']} | {s['n']} | {ci(s,'cos_raw')} | {ci(s,'cos_centered')} | {d['n']} | {ci(d,'cos_raw')} | {ci(d,'cos_centered')} |")
        L.append("")
    L += ["## 재현", "", "```", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2", f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_boundary_kinematics.py {N_PER}", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
