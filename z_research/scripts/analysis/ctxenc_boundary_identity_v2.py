#!/usr/bin/env python3
"""E2-v2 — does the boundary tubelet (t7) of the predictor input (ctx_masked) carry a *readable, transferable* identity code when
the object is hidden: adds the accelerating (ramp) family, occluder-excluded mean readouts, a concatenated 3x3 readout, reverse
transfers (late -> early / visible) and block-bootstrap 95 % CIs to ctxenc_boundary_identity_occlusion.py.

Data      : IntPhysGen v11_full, index_probe.csv rows with probe_type == 'obj' (possible variants only, CLAUDE.md §1-4).
Cells     : the 6 old cells (static / moving_flat x visible k0 / late k1..4 / early k4; identical clip selection, checked against the old
            features.npz) + moving_ramp_visible (moving_visible k0) / moving_ramp_late (moving_occlusion k1..4) / moving_ramp_early_k4
            (moving_occlusion_early k4).  100 blocks (= up to 200 clips) per (condition, k) cell, seed 0.
Reps      : z = ctx_masked.npy (8 context tubelets, predictor input);  h = target.npy tubelets 0..7 (encoder that also saw the future).
Features  : B7    = mean of the 3x3 tokens around the true object position at tubelet 7 (mapping = v11_token_object_test.n3)
            B7cat = the same 9 tokens concatenated in row-major order (9 x 1280 = 11520-d; zero-padded if the window is clipped)
            B7all = mean of all 256 tokens of tubelet 7
            C     = mean of all 2048 context tokens
            BG    = mean of the 3x3 tokens at a fixed far-from-object location (rows 0..2, cols 13..15) at tubelet 7  [control]
            C_noocc, B7all_noocc = C / B7all with every token inside the occluder's token columns x rows removed, at every context
                    tubelet whose sample frames fall inside [occ_rise_start, occ_fall_end] (no occluder -> identical to C / B7all)
            C_noocc1 = C_noocc with the occluder token box dilated by one token on every side (robustness)
Occluder token box (documented derivation, validated per clip in results.json['projection_checks']):
            pinhole camera from metadata.csv: cam (0, -1150, 210) cm, yaw 90 deg (forward +Y, image-right = -X), pitch -3 deg,
            f_px = focal_mm / sensor_mm * 288 = 264.  For a world point P: rel = P - cam, d = rel.fwd, u = rel.up, r = rel.right,
            px = 144 + f r / d, py = 144 - f u / d.  This reproduces object_px_{x,y}_by_sample (max |err| reported).
            Occluder (trapdoor, rises from the floor): world x in [occluder_x_cm -/+ occ_width_cm/2], y = cam_y + occ_depth_cm,
            z in [0, occ_height_cm]; the four corners project to an axis-aligned pixel box [x_lo, x_hi] x [y_lo, y_hi]; its width and
            height are compared with occ_apparent_px and occ_frame_frac_h * 288.  Token column c (18 px wide) is excluded iff
            [18c, 18c+18) overlaps [x_lo, x_hi], likewise rows; the excluded set is rows x cols (token index = row*16 + col).
Readout   : closed-form one-vs-rest ridge on centred one-hot labels (argmax), features standardized on the train split, lambda chosen on a
            block-held-out validation fold (25 % of train blocks) inside the train split; primal or dual (n <= d) form via one
            eigendecomposition per training matrix shared by the lambda grid, both labels, and the label-shuffled control.
Splits    : always by block_id, stratified by k | violation_type, seed 0.  Self = 4-fold CV by block within a cell.
Fits      : self (per cell; late pooled k1..4 and per k) ; visible -> late (pooled, per k) ; visible -> early ; early -> late (pooled, per k) ;
            early -> visible ; late -> early ; late -> visible.
Controls  : label shuffle (train labels permuted, seed 0) for every fit; BG feature; h ceiling for everything; block-bootstrap 95 % CI
            (1000 resamples of test blocks, seed 0) on every accuracy.
Outputs   : z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/{results.json, RESULTS.md, features.npz}

  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2 && \
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_boundary_identity_v2.py [--blocks 100] [--reuse-features]
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import ctxenc_boundary_identity_occlusion as old   # noqa: E402  (arr, n3, stratified_block_folds reused verbatim)

CACHE = old.CACHE; INDEX = old.INDEX; META = old.META
OLD_OUT = old.OUT
OUT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2"
SEED = 0
T7 = 7; T7_FRAMES = (42, 45)
BG_TOKENS = old.BG_TOKENS
LABELS = {"shape": "shape_pre", "color": "color_pre"}
FEATS = ["B7", "B7cat", "B7all", "C", "BG", "C_noocc", "C_noocc1", "B7all_noocc"]
REPS = ["z", "h"]
CELLS = list(old.CELLS) + [
    ("moving_ramp_visible", "moving_visible", ["0"]),
    ("moving_ramp_late", "moving_occlusion", ["1", "2", "3", "4"]),
    ("moving_ramp_early_k4", "moving_occlusion_early", ["4"]),
]
FAMILIES = {"static": ("static_visible", "static_late", "static_early_k4"),
            "moving_flat": ("moving_flat_visible", "moving_flat_late", "moving_flat_early_k4"),
            "moving_ramp": ("moving_ramp_visible", "moving_ramp_late", "moving_ramp_early_k4")}
LAM_REL = old.LAM_REL
N_BOOT = 1000
RES = 288; F_PX = 22.0 / 24.0 * RES; CAM = np.array([0.0, -1150.0, 210.0]); PITCH = np.deg2rad(-3.0)
FWD = np.array([0.0, np.cos(PITCH), np.sin(PITCH)]); UP = np.array([0.0, -np.sin(PITCH), np.cos(PITCH)]); RIGHT = np.array([-1.0, 0.0, 0.0])
arr, n3, stratified_block_folds = old.arr, old.n3, old.stratified_block_folds


def project(P):
    """P: (..., 3) world cm -> (..., 2) pixel (px, py) in the 288 render.  See module docstring."""
    rel = np.asarray(P, float) - CAM
    d = rel @ FWD; u = rel @ UP; r = rel @ RIGHT
    return np.stack([RES / 2 + F_PX * r / d, RES / 2 - F_PX * u / d], -1)


def n3_ordered(x, y):
    """row-major 3x3 (r, c) slots around the object (None where clipped) -> for B7cat; the kept tokens equal n3(x, y)."""
    cx, cy = int(x // 18), int(y // 18)
    return [(r * 16 + c) if (0 <= r < 16 and 0 <= c < 16) else None for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2)]


def occluder_tokens(m, dilate=0):
    """token-index sets per context tubelet (t=0..7) covered by the occluder while it is on screen; [] where absent."""
    if m["has_occlusion"] != "1" or m["occluder_x_cm"] == "":
        return [[] for _ in range(8)], None
    ox, ow, oh, od = (float(m[k]) for k in ("occluder_x_cm", "occ_width_cm", "occ_height_cm", "occ_depth_cm"))
    ys = CAM[1] + od
    corners = np.array([[ox - ow / 2, ys, 0], [ox + ow / 2, ys, 0], [ox - ow / 2, ys, oh], [ox + ow / 2, ys, oh]])
    p = project(corners); x_lo, x_hi = p[:, 0].min(), p[:, 0].max(); y_lo, y_hi = p[:, 1].min(), p[:, 1].max()
    cols = [c for c in range(16) if 18 * c < x_hi and 18 * (c + 1) > x_lo]
    rows = [r for r in range(16) if 18 * r < y_hi and 18 * (r + 1) > y_lo]
    if dilate:
        cols = sorted({c for c0 in cols for c in range(c0 - dilate, c0 + dilate + 1) if 0 <= c < 16})
        rows = sorted({r for r0 in rows for r in range(r0 - dilate, r0 + dilate + 1) if 0 <= r < 16})
    toks = [r * 16 + c for r in rows for c in cols]
    rs, fe = int(float(m["occ_rise_start"])), int(float(m["occ_fall_end"]))
    per_t = [toks if any(rs <= f <= fe for f in (6 * t, 6 * t + 3)) else [] for t in range(8)]
    box = dict(x_lo=float(x_lo), x_hi=float(x_hi), y_lo=float(y_lo), y_hi=float(y_hi), cols=cols, rows=rows,
               width_err_px=float((x_hi - x_lo) - float(m["occ_apparent_px"])), height_err_px=float((y_hi - y_lo) - float(m["occ_frame_frac_h"]) * RES),
               on_frames=[rs, fe])
    return per_t, box


# ----------------------------------------------------------------------------------------------- selection (old order first)
def select_clips(n_blocks):
    idx = [r for r in csv.DictReader(INDEX.open()) if r["probe_type"] == "obj"]
    rng = np.random.RandomState(SEED)
    fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    sel = []
    for cell, cond, ks in CELLS:
        for k in ks:
            rows = [r for r in idx if r["condition"] == cond and r["sym_k"] == k]
            by_block = defaultdict(list)
            for r in rows:
                by_block[r["block_id"]].append(r)
            blocks = sorted(by_block)
            take = list(rng.permutation(blocks)[:n_blocks])
            for b in take:
                for r in by_block[b]:
                    sel.append(dict(video_id=r["video_id"], cell=cell, family=fam_of[cell], cond=cond, k=k, block=f"{cond}_k{k}_{b}",
                                    violation_type=r["violation_type"], variant=r["variant"], shape=r["shape_pre"], color=r["color_pre"]))
    return sel


def extract(sel):
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    vid_row = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r")
    assert Z.shape[1] == 2048 and H.shape[1] == 4096, (Z.shape, H.shape)
    order = sorted(range(len(sel)), key=lambda i: vid_row[sel[i]["video_id"]])
    n = len(sel); D = Z.shape[2]
    F = {f"{rep}_{ft}": np.zeros((n, 9 * D if ft == "B7cat" else D), np.float32) for rep in REPS for ft in FEATS}
    info = {k: np.zeros(n) for k in ("hidden_frac_t7", "win_n", "occ_on_t7", "n_occ_tok_t7", "n_occ_tok_ctx", "n_occ_tok1_ctx", "b7_in_occ_frac",
                                     "proj_err_px", "occ_width_err_px", "occ_height_err_px")}
    info["obj_px_t7"] = np.zeros((n, 2)); boxes = {}
    t0 = time.time()
    for j, i in enumerate(order):
        s = sel[i]; m = meta[s["video_id"]]
        t32 = np.stack([arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1)
        w3 = np.stack([arr(m["object_x_cm_by_sample"]), np.full(32, CAM[1] + float(m["obj_depth_cm"])), arr(m["object_z_cm_by_sample"])], -1)
        info["proj_err_px"][i] = np.abs(project(w3) - t32).max()
        win = n3(*tub[T7]); slots = n3_ordered(*tub[T7]); assert [t for t in slots if t is not None] == win
        info["win_n"][i] = len(win); info["obj_px_t7"][i] = tub[T7]
        if m["has_occlusion"] == "1" and m["hidden_start"] != "":
            hs, he = int(float(m["hidden_start"])), int(float(m["hidden_end"]))
            info["hidden_frac_t7"][i] = np.mean([(hs <= f <= he) for f in T7_FRAMES])
        occ_t, box = occluder_tokens(m); occ1_t, _ = occluder_tokens(m, dilate=1)
        if box is not None:
            boxes[s["cell"]] = boxes.get(s["cell"], []) + [box]
            info["occ_width_err_px"][i] = box["width_err_px"]; info["occ_height_err_px"][i] = box["height_err_px"]
        info["occ_on_t7"][i] = bool(occ_t[T7]); info["n_occ_tok_t7"][i] = len(occ_t[T7])
        info["n_occ_tok_ctx"][i] = sum(len(t) for t in occ_t); info["n_occ_tok1_ctx"][i] = sum(len(t) for t in occ1_t)
        info["b7_in_occ_frac"][i] = np.mean([t in set(occ_t[T7]) for t in win]) if occ_t[T7] else 0.0
        keep_ctx = np.ones(2048, bool); keep1_ctx = np.ones(2048, bool)
        for t in range(8):
            for tok in occ_t[t]: keep_ctx[t * 256 + tok] = False
            for tok in occ1_t[t]: keep1_ctx[t * 256 + tok] = False
        keep_t7 = keep_ctx[T7 * 256:(T7 + 1) * 256]
        r = vid_row[s["video_id"]]
        for rep, A in (("z", Z), ("h", H)):
            ctx = np.asarray(A[r, :2048], np.float32); t7 = ctx[T7 * 256:(T7 + 1) * 256]
            F[f"{rep}_B7"][i] = t7[win].mean(0); F[f"{rep}_B7all"][i] = t7.mean(0); F[f"{rep}_C"][i] = ctx.mean(0); F[f"{rep}_BG"][i] = t7[BG_TOKENS].mean(0)
            F[f"{rep}_B7cat"][i] = np.concatenate([t7[t] if t is not None else np.zeros(D, np.float32) for t in slots])
            F[f"{rep}_C_noocc"][i] = ctx[keep_ctx].mean(0); F[f"{rep}_C_noocc1"][i] = ctx[keep1_ctx].mean(0); F[f"{rep}_B7all_noocc"][i] = t7[keep_t7].mean(0)
        if j % 300 == 0:
            print(f"  extract {j}/{n}  {time.time() - t0:.0f}s", flush=True)
    return F, info, boxes


# ----------------------------------------------------------------------------------------------- ridge (eigen-cached, closed form)
class RidgeSolver:
    """standardize X on itself; one eigendecomposition; W(lambda) for any centred target matrix Y in O(n d c)."""

    def __init__(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xs = ((X - self.mu) / self.sd).astype(np.float64); n, d = Xs.shape
        self.primal = n > d
        if self.primal:
            S, V = np.linalg.eigh(Xs.T @ Xs); self.S = np.clip(S, 0, None); self.V = V; self.Xs = Xs
        else:
            S, U = np.linalg.eigh(Xs @ Xs.T); self.S = np.clip(S, 0, None); self.U = U; self.A = Xs.T @ U

    def solve(self, Y, lam):
        if self.primal:
            B = self.V.T @ (self.Xs.T @ Y); return self.V @ (B / (self.S + lam)[:, None])
        return self.A @ ((self.U.T @ Y) / (self.S + lam)[:, None])

    def scores(self, X, W): return ((X - self.mu) / self.sd) @ W


def onehot(y, classes):
    Y = np.eye(len(classes))[[classes.index(v) for v in y]]; return Y - Y.mean(0)


def fit_readouts(X, ys, blocks, strata, classes, rng):
    """ys: {label: y}. Returns {label: dict(W, W_shuf, lam_rel, val_acc)}; lambda chosen per (label, real/shuffled) on an inner block fold."""
    f = stratified_block_folds(blocks, strata, 4, SEED + 1); tr, va = f != 0, f == 0
    inner = RidgeSolver(X[tr]); full = RidgeSolver(X); n_tr, n_all = int(tr.sum()), len(blocks); out = {}
    for L, y in ys.items():
        cls = classes[L]; ysh = y.copy(); rng.shuffle(ysh); res = {}
        for tag, yy in (("real", y), ("shuf", ysh)):
            Yin = onehot(yy[tr], cls); best = None
            for rel in LAM_REL:
                pred = inner.scores(X[va], inner.solve(Yin, rel * n_tr)).argmax(1)
                acc = float(np.mean(np.array(cls)[pred] == yy[va]))
                if best is None or acc >= best[0] - 1e-12: best = (acc, rel)
            res[tag] = (full.solve(onehot(yy, cls), best[1] * n_all), best[1], best[0])
        out[L] = dict(W=res["real"][0], W_shuf=res["shuf"][0], lam_rel=res["real"][1], val_acc=res["real"][2], lam_rel_shuf=res["shuf"][1])
    return full, out


def predict(solver, W, X, cls): return np.array(cls)[solver.scores(X, W).argmax(1)]


def block_bootstrap(correct, blocks, n_boot=N_BOOT, seed=SEED):
    ub, inv = np.unique(blocks, return_inverse=True)
    nc = np.bincount(inv, weights=correct.astype(float), minlength=len(ub)); nn = np.bincount(inv, minlength=len(ub)).astype(float)
    rng = np.random.RandomState(seed); idx = rng.randint(0, len(ub), size=(n_boot, len(ub)))
    accs = nc[idx].sum(1) / nn[idx].sum(1)
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--blocks", type=int, default=100); ap.add_argument("--reuse-features", action="store_true")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); fpath = OUT / "features.npz"
    if a.reuse_features and fpath.exists():
        npz = np.load(fpath, allow_pickle=True); sel = list(npz["sel"]); F = {k: npz[k] for k in npz if k.startswith(("z_", "h_"))}
        info = {k: npz[k] for k in npz if not k.startswith(("z_", "h_", "sel", "boxes"))}; boxes = json.loads(str(npz["boxes"]))
    else:
        sel = select_clips(a.blocks); print(f"selected {len(sel)} clips", flush=True)
        F, info, boxes = extract(sel)
        np.savez(fpath, sel=np.array(sel, dtype=object), boxes=json.dumps(boxes), **F, **info)
    n = len(sel)
    cell = np.array([s["cell"] for s in sel]); k = np.array([s["k"] for s in sel]); blk = np.array([s["block"] for s in sel])
    lab = {L: np.array([s[L] for s in sel]) for L in LABELS}; strata = np.array([f"{s['k']}|{s['violation_type']}" for s in sel])
    classes = {L: sorted(set(lab[L])) for L in LABELS}; chance = {L: 1 / len(classes[L]) for L in LABELS}
    rng = np.random.RandomState(SEED)

    # -- old-cell reproduction check (same clips, same feature values as the old run)
    repro = {"checked": False}
    old_npz_path = OLD_OUT / "features.npz"
    if old_npz_path.exists():
        onpz = np.load(old_npz_path, allow_pickle=True); osel = list(onpz["sel"]); opos = {s["video_id"]: i for i, s in enumerate(osel)}
        common = [(i, opos[s["video_id"]]) for i, s in enumerate(sel) if s["video_id"] in opos]
        ii, oo = map(np.array, zip(*common))
        old_cell_names = {c[0] for c in old.CELLS}
        new_old_vids = {s["video_id"] for s in sel if s["cell"] in old_cell_names}
        repro = dict(checked=True, n_old=len(osel), n_common=len(common), same_video_set=bool(new_old_vids == set(opos)),
                     max_abs_diff={f"{r}_{ft}": float(np.abs(F[f"{r}_{ft}"][ii] - onpz[f"{r}_{ft}"][oo]).max()) for r in REPS for ft in ("B7", "B7all", "C", "BG")})

    # -- cell summary / sanity
    cells = {}
    for c, cond, ks in CELLS:
        for kk in ["all"] + ks:
            if kk != "all" and len(ks) == 1: continue
            m = (cell == c) & ((k == kk) if kk != "all" else True)
            cells[f"{c}|k={kk}"] = dict(condition=cond, k=kk, n_clips=int(m.sum()), n_blocks=len(set(blk[m])),
                                       hidden_frac_t7_mean=float(info["hidden_frac_t7"][m].mean()), win_n_mean=float(info["win_n"][m].mean()),
                                       occ_on_t7_frac=float(info["occ_on_t7"][m].mean()), n_occ_tok_t7_mean=float(info["n_occ_tok_t7"][m].mean()),
                                       n_occ_tok_ctx_mean=float(info["n_occ_tok_ctx"][m].mean()), n_occ_tok1_ctx_mean=float(info["n_occ_tok1_ctx"][m].mean()),
                                       b7_in_occ_frac_mean=float(info["b7_in_occ_frac"][m].mean()),
                                       obj_px_t7_x_range=[float(info["obj_px_t7"][m, 0].min()), float(info["obj_px_t7"][m, 0].max())],
                                       shape_counts=dict(Counter(lab["shape"][m])), color_counts=dict(Counter(lab["color"][m])))
    proj = dict(f_px=F_PX, cam=CAM.tolist(), pitch_deg=-3.0, object_reprojection_max_abs_err_px=float(info["proj_err_px"].max()),
                occluder_width_err_px_range=[float(info["occ_width_err_px"].min()), float(info["occ_width_err_px"].max())],
                occluder_height_err_px_range=[float(info["occ_height_err_px"].min()), float(info["occ_height_err_px"].max())],
                floor_line_py_at_occluder_depth=float(project(np.array([[0, CAM[1] + 1190, 0]]))[0, 1]),
                boxes_per_cell={c: dict(n=len(b), x_lo_mean=float(np.mean([x["x_lo"] for x in b])), x_hi_mean=float(np.mean([x["x_hi"] for x in b])),
                                        y_lo_mean=float(np.mean([x["y_lo"] for x in b])), y_hi_mean=float(np.mean([x["y_hi"] for x in b])),
                                        n_cols_mean=float(np.mean([len(x["cols"]) for x in b])), n_rows_mean=float(np.mean([len(x["rows"]) for x in b])),
                                        on_frames=sorted({tuple(x["on_frames"]) for x in b})) for c, b in boxes.items()})
    fits = []

    def sub(c, kk=None):
        m = cell == c
        return m & (k == kk) if kk is not None else m

    def record(fit, fam, rep, ft, train, test, kk, mt, preds, n_train, n_train_blocks, extra):
        for L in LABELS:
            corr = preds[L]["pred"] == lab[L][mt]; corr_s = preds[L]["pred_shuf"] == lab[L][mt]
            fits.append(dict(fit=fit, family=fam, label=L, rep=rep, feat=ft, train=train, test=test, k=kk, n_train=n_train, n_train_blocks=n_train_blocks,
                             n_test=int(mt.sum()), n_test_blocks=len(set(blk[mt])), acc=float(corr.mean()), ci95=block_bootstrap(corr, blk[mt]),
                             acc_shuffle=float(corr_s.mean()), chance=chance[L], **extra.get(L, {})))

    t0 = time.time(); fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    for rep in REPS:
        for ft in FEATS:
            X = F[f"{rep}_{ft}"]
            # (1) self: 4-fold CV by block (folds stratified by k|violation_type; label-independent so both labels share the solver)
            for c, cond, ks in CELLS:
                for kk in ([None] + ks if len(ks) > 1 else [None]):
                    m = sub(c, kk); idx = np.where(m)[0]; Xm, bm, sm = X[m], blk[m], strata[m]
                    folds = stratified_block_folds(bm, sm, 4, SEED)
                    pred = {L: np.empty(len(idx), object) for L in LABELS}; pred_s = {L: np.empty(len(idx), object) for L in LABELS}; lams = {L: [] for L in LABELS}
                    for fo in range(4):
                        tr, te = folds != fo, folds == fo
                        solver, ro = fit_readouts(Xm[tr], {L: lab[L][idx][tr] for L in LABELS}, bm[tr], sm[tr], classes, rng)
                        for L in LABELS:
                            pred[L][te] = predict(solver, ro[L]["W"], Xm[te], classes[L]); pred_s[L][te] = predict(solver, ro[L]["W_shuf"], Xm[te], classes[L]); lams[L].append(ro[L]["lam_rel"])
                    record("self", fam_of[c], rep, ft, c, c, kk or "all", m, {L: dict(pred=pred[L], pred_shuf=pred_s[L]) for L in LABELS},
                           int(m.sum()), len(set(bm)), {L: dict(lam_rel=lams[L], note="4-fold CV by block") for L in LABELS})
            # (2) transfers within family: visible / early / late(pooled) as sources
            for fam, (vis, late, early) in FAMILIES.items():
                names = {vis: "visible", late: "late", early: "early"}
                targets = {vis: [(late, None)] + [(late, kk) for kk in "1234"] + [(early, None)],
                           early: [(late, None)] + [(late, kk) for kk in "1234"] + [(vis, None)],
                           late: [(early, None), (vis, None)]}
                for src, tgts in targets.items():
                    ms = sub(src); solver, ro = fit_readouts(X[ms], {L: lab[L][ms] for L in LABELS}, blk[ms], strata[ms], classes, rng)
                    for tc, kk in tgts:
                        mt = sub(tc, kk)
                        record(f"transfer_{names[src]}->{names[tc]}", fam, rep, ft, src, tc, kk or "all", mt,
                               {L: dict(pred=predict(solver, ro[L]["W"], X[mt], classes[L]), pred_shuf=predict(solver, ro[L]["W_shuf"], X[mt], classes[L])) for L in LABELS},
                               int(ms.sum()), len(set(blk[ms])), {L: dict(lam_rel=ro[L]["lam_rel"], val_acc=ro[L]["val_acc"]) for L in LABELS})
            print(f"  fits done: {rep} {ft}  ({time.time() - t0:.0f}s)", flush=True)

    res = dict(script=str(Path(__file__).relative_to(ROOT)), seed=SEED, n_blocks_per_cell=a.blocks, n_clips_total=n, n_boot=N_BOOT, cache=str(CACHE), index=str(INDEX),
               bg_tokens=BG_TOKENS, lam_rel_grid=LAM_REL, feats=FEATS, classes=classes, old_cells_reproduced=repro, projection_checks=proj, cells=cells, fits=fits)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_md(res); print(f"-> {OUT}/results.json, RESULTS.md")


# ----------------------------------------------------------------------------------------------- report
def _get(fits, **kw):
    r = [f for f in fits if all(f[a] == b for a, b in kw.items())]; return r[0] if r else None


def _cell(f):
    if f is None: return "—"
    lo, hi = f["ci95"]; return f"{100 * f['acc']:.1f} [{100 * lo:.0f},{100 * hi:.0f}] ({100 * f['acc_shuffle']:.0f})"


def _flag(f):  # CI lower bound above chance?
    return "—" if f is None else ("above" if f["ci95"][0] > f["chance"] else "chance")


def write_md(res):
    fits = res["fits"]; L = []
    L += ["# E2-v2 — boundary-tubelet identity at the predictor input under occlusion: ramp family, occluder-excluded means, concatenated 3x3, reverse transfers, bootstrap CIs (IntPhysGen v11_full, ViT-H)", "",
          f"Every number below is recomputed from `results.json` written by `{res['script']}` (seed {res['seed']}, {res['n_blocks_per_cell']} blocks = up to {2 * res['n_blocks_per_cell']} clips per (condition, k) cell, {res['n_clips_total']} clips total).",
          f"Entry format: **acc [95 % block-bootstrap CI, {res['n_boot']} resamples] (label-shuffle control)**, all in %. Readout = closed-form one-vs-rest ridge (argmax), block-held-out, lambda on an inner block-held-out fold. Splits by block_id, stratified by k|violation_type, seed 0.",
          "Reps: `z` = ctx_masked (predictor input), `h` = target encoder (ceiling). Feats: B7 = 3x3 mean at the true object position @t7; B7cat = the same 9 tokens concatenated (11520-d); B7all = all 256 tokens @t7; C = all 2048 context tokens; BG = fixed far 3x3 @t7 (control); `_noocc` = occluder token box removed at every tubelet where the occluder is on screen (`_noocc1` = box dilated by one token).", ""]
    rp = res["old_cells_reproduced"]; pj = res["projection_checks"]
    L += ["## Controls on the pipeline", "",
          f"- Old six cells: {'same clip set as the old run and identical feature values (max |diff| ' + ', '.join(f'{k} {v:.1e}' for k, v in rp['max_abs_diff'].items()) + ')' if rp.get('checked') and rp.get('same_video_set') else 'NOT reproduced — see results.json[old_cells_reproduced]'}.",
          f"- Pinhole re-projection of `object_{{x,z}}_cm_by_sample` vs metadata `object_px_*_by_sample`: max |err| = {pj['object_reprojection_max_abs_err_px']:.2f} px (f = {pj['f_px']:.0f} px, cam {pj['cam']}, pitch {pj['pitch_deg']} deg).",
          f"- Occluder box vs metadata: width − `occ_apparent_px` in [{pj['occluder_width_err_px_range'][0]:.2f}, {pj['occluder_width_err_px_range'][1]:.2f}] px; height − `occ_frame_frac_h`·288 in [{pj['occluder_height_err_px_range'][0]:.2f}, {pj['occluder_height_err_px_range'][1]:.2f}] px; floor line at occluder depth py = {pj['floor_line_py_at_occluder_depth']:.1f}.",
          "", "| cell | n boxes | x_lo–x_hi (px, mean) | y_lo–y_hi (px, mean) | token cols | token rows | occluder on frames |", "|---|---:|---|---|---:|---:|---|"]
    for c, b in pj["boxes_per_cell"].items():
        L.append(f"| {c} | {b['n']} | {b['x_lo_mean']:.0f}–{b['x_hi_mean']:.0f} | {b['y_lo_mean']:.0f}–{b['y_hi_mean']:.0f} | {b['n_cols_mean']:.1f} | {b['n_rows_mean']:.1f} | {b['on_frames']} |")
    L += ["", "## Cells", "", "| cell | condition | k | n clips | n blocks | hidden frac @t7 (42,45) | occluder on @t7 | occ tokens @t7 | occ tokens / 2048 ctx (dilated) | B7 window inside occluder | obj x @t7 (px) |", "|---|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for name, c in res["cells"].items():
        L.append(f"| {name.split('|')[0]} | {c['condition']} | {c['k']} | {c['n_clips']} | {c['n_blocks']} | {c['hidden_frac_t7_mean']:.2f} | {c['occ_on_t7_frac']:.2f} | {c['n_occ_tok_t7_mean']:.1f} | {c['n_occ_tok_ctx_mean']:.0f} ({c['n_occ_tok1_ctx_mean']:.0f}) | {c['b7_in_occ_frac_mean']:.2f} | {c['obj_px_t7_x_range'][0]:.0f}–{c['obj_px_t7_x_range'][1]:.0f} |")
    fams = list(FAMILIES)
    # ---- headline tables
    L += ["", "## Headline tables (z = predictor input unless stated; `above` = CI lower bound > chance)", ""]
    for lab in LABELS:
        ch = _get(fits, label=lab)["chance"]
        L += [f"### {lab}_pre, chance {100 * ch:.1f}%", "", "**(i) Does ramp behave like flat / static? — by family, feature B7 (z and h)**", "",
              "| family | rep | self visible | self late (k1..4) | visible→late | visible→early | early→late | late→early | late→visible |", "|---|---|---|---|---|---|---|---|---|"]
        for fam in fams:
            vis, late, early = FAMILIES[fam]
            for rep in REPS:
                L.append(f"| {fam} | {rep} | " + " | ".join(_cell(_get(fits, label=lab, rep=rep, feat="B7", family=fam, **q)) for q in [
                    dict(fit="self", train=vis, k="all"), dict(fit="self", train=late, k="all"), dict(fit="transfer_visible->late", k="all"), dict(fit="transfer_visible->early"),
                    dict(fit="transfer_early->late", k="all"), dict(fit="transfer_late->early"), dict(fit="transfer_late->visible")]) + " |")
        L += ["", "**(ii) Is the C transfer drop composition? — visible→early, by feature (z / h)**", "",
              "| family | rep | B7 | B7all | B7all_noocc | C | C_noocc | C_noocc1 | BG |", "|---|---|---|---|---|---|---|---|---|"]
        for fam in fams:
            for rep in REPS:
                L.append(f"| {fam} | {rep} | " + " | ".join(_cell(_get(fits, label=lab, rep=rep, feat=ft, fit="transfer_visible->early", family=fam)) for ft in ["B7", "B7all", "B7all_noocc", "C", "C_noocc", "C_noocc1", "BG"]) + " |")
        L += ["", "**(ii-b) same features, visible→late (pooled k1..4) and self late**", "",
              "| family | rep | fit | B7 | B7all | B7all_noocc | C | C_noocc | C_noocc1 | BG |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in fams:
            vis, late, early = FAMILIES[fam]
            for rep in REPS:
                for fit, q in (("visible→late", dict(fit="transfer_visible->late", family=fam, k="all")), ("self late", dict(fit="self", train=late, k="all"))):
                    L.append(f"| {fam} | {rep} | {fit} | " + " | ".join(_cell(_get(fits, label=lab, rep=rep, feat=ft, **q)) for ft in ["B7", "B7all", "B7all_noocc", "C", "C_noocc", "C_noocc1", "BG"]) + " |")
        L += ["", "**(iii) Does concatenation change the visible→late verdict? — B7 (mean) vs B7cat (9 tokens concatenated)**", "",
              "| family | rep | self visible B7 | self visible B7cat | self late B7 | self late B7cat | visible→late B7 | visible→late B7cat | verdict B7cat | visible→early B7 | visible→early B7cat |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam in fams:
            vis, late, early = FAMILIES[fam]
            for rep in REPS:
                g = lambda ft, **q: _get(fits, label=lab, rep=rep, feat=ft, **q)
                vl = g("B7cat", fit="transfer_visible->late", family=fam, k="all")
                L.append(f"| {fam} | {rep} | {_cell(g('B7', fit='self', train=vis, k='all'))} | {_cell(g('B7cat', fit='self', train=vis, k='all'))} | {_cell(g('B7', fit='self', train=late, k='all'))} | {_cell(g('B7cat', fit='self', train=late, k='all'))} | "
                         f"{_cell(g('B7', fit='transfer_visible->late', family=fam, k='all'))} | {_cell(vl)} | {_flag(vl)} | {_cell(g('B7', fit='transfer_visible->early', family=fam))} | {_cell(g('B7cat', fit='transfer_visible->early', family=fam))} |")
        L += ["", "**(iv) Is the hidden-window code a readable identity code? — reverse transfers late→early / late→visible, and per-k self late**", "",
              "| family | rep | feat | self late (pooled) | self late k1 | k2 | k3 | k4 | late→early | verdict | late→visible | verdict | early→late | early→visible |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam in fams:
            vis, late, early = FAMILIES[fam]
            for rep in REPS:
                for ft in ["B7", "B7cat", "B7all", "C", "C_noocc"]:
                    g = lambda **q: _get(fits, label=lab, rep=rep, feat=ft, **q)
                    le, lv = g(fit="transfer_late->early", family=fam), g(fit="transfer_late->visible", family=fam)
                    L.append(f"| {fam} | {rep} | {ft} | {_cell(g(fit='self', train=late, k='all'))} | " + " | ".join(_cell(g(fit="self", train=late, k=kk)) for kk in "1234") +
                             f" | {_cell(le)} | {_flag(le)} | {_cell(lv)} | {_flag(lv)} | {_cell(g(fit='transfer_early->late', family=fam, k='all'))} | {_cell(g(fit='transfer_early->visible', family=fam))} |")
        L.append("")
    # ---- full tables
    L += ["## Full tables", ""]
    for lab in LABELS:
        ch = _get(fits, label=lab)["chance"]
        L += [f"### {lab}_pre — chance {100 * ch:.1f}%", ""]
        for fam in fams:
            vis, late, early = FAMILIES[fam]
            spec = [("self", vis, vis, "all"), ("self", early, early, "all"), ("self", late, late, "all")] + [("self", late, late, kk) for kk in "1234"] + \
                   [("transfer_visible->late", vis, late, "all")] + [("transfer_visible->late", vis, late, kk) for kk in "1234"] + [("transfer_visible->early", vis, early, "all")] + \
                   [("transfer_early->late", early, late, "all")] + [("transfer_early->late", early, late, kk) for kk in "1234"] + [("transfer_early->visible", early, vis, "all")] + \
                   [("transfer_late->early", late, early, "all"), ("transfer_late->visible", late, vis, "all")]
            for rep in REPS:
                L += [f"#### {fam} — `{rep}`", "", "| fit | train → test | k | n_train (blocks) | n_test (blocks) | " + " | ".join(FEATS) + " |", "|---|---|---|---:|---:|" + "---|" * len(FEATS)]
                for fit, tr, te, kk in spec:
                    f0 = _get(fits, label=lab, rep=rep, feat=FEATS[0], fit=fit, train=tr, test=te, k=kk)
                    if f0 is None: continue
                    L.append(f"| {fit} | {tr} → {te} | {kk} | {f0['n_train']} ({f0['n_train_blocks']}) | {f0['n_test']} ({f0['n_test_blocks']}) | " +
                             " | ".join(_cell(_get(fits, label=lab, rep=rep, feat=ft, fit=fit, train=tr, test=te, k=kk)) for ft in FEATS) + " |")
                L.append("")
    L += ["## 재현", "", "```", f"cd {ROOT}", f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python {res['script']} --blocks {res['n_blocks_per_cell']}", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
