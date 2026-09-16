#!/usr/bin/env python3
"""E2 — at the predictor's input (ctx_masked, context-encoder output after LN), does the boundary tubelet (t7) still carry the
object's identity (shape_pre / color_pre) when the object is hidden behind the occluder, and does the mere presence of an
occluder (object visible at the boundary, *_early k=4) shift the identity code?

Data      : IntPhysGen v11_full, index_probe.csv rows with probe_type == 'obj' (possible variants only, CLAUDE.md §1-4).
Cells     : static_visible k0 | static_occlusion k1..4 (late) | static_occlusion_early k4 | moving_visible_flat k0 |
            moving_occlusion_flat k1..4 (late) | moving_occlusion_flat_early k4.  Subsample 100 blocks (= 200 clips) per cell, seed 0.
Reps      : z = ctx_masked.npy (8 context tubelets, predictor input);  h = target.npy tubelets 0..7 (encoder that also saw the future).
Features  : B7   = mean of the 3x3 tokens around the true object position at tubelet 7 (mapping copied from v11_token_object_test.py)
            B7all= mean of all 256 tokens of tubelet 7
            C    = mean of all 2048 context tokens
            BG   = mean of the 3x3 tokens at a fixed far-from-object location (rows 0..2, cols 13..15) at tubelet 7  [control]
Readout   : closed-form one-vs-rest ridge on one-hot labels (argmax), features standardized on the train split, lambda chosen on a
            block-held-out validation fold inside the train split. Splits by block_id, stratified, seed 0. No SGD -> deterministic.
Fits      : (1) self  = 4-fold CV by block within a cell (late pooled k1..4 and per k)
            (2) transfer visible k0 -> late k1..4 (pooled and per k)
            (3) transfer visible k0 -> early k4 (object visible at boundary, occluder in scene)
            (2b) transfer early k4 -> late k1..4 (occluder present in both; only hiddenness differs)
            controls: label shuffle (train labels permuted, seed 0) for every fit; BG feature; h ceiling for everything.
Sanity    : per cell, fraction of tubelet-7 frames (raw 42, 45) inside [hidden_start, hidden_end] from metadata.csv (k=1 hides only frame 45).
Outputs   : z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_occlusion/{results.json, RESULTS.md, features.npz}

  cd /data/hyuntak/project/2026/2027_cvpr/vjepa2 && \
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_boundary_identity_occlusion.py [--blocks 100] [--reuse-features]
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
OUT = ROOT / "z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_occlusion"
SEED = 0
T7 = 7                                   # boundary context tubelet (raw frames 42, 45)
T7_FRAMES = (42, 45)
BG_TOKENS = [r * 16 + c for r in range(0, 3) for c in range(13, 16)]   # fixed far-from-object 3x3 (top-right)
LABELS = {"shape": "shape_pre", "color": "color_pre"}
FEATS = ["B7", "B7all", "C", "BG"]
REPS = ["z", "h"]
CELLS = [  # name, condition, ks
    ("static_visible", "static_visible", ["0"]),
    ("static_late", "static_occlusion", ["1", "2", "3", "4"]),
    ("static_early_k4", "static_occlusion_early", ["4"]),
    ("moving_flat_visible", "moving_visible_flat", ["0"]),
    ("moving_flat_late", "moving_occlusion_flat", ["1", "2", "3", "4"]),
    ("moving_flat_early_k4", "moving_occlusion_flat_early", ["4"]),
]
FAMILIES = {"static": ("static_visible", "static_late", "static_early_k4"),
            "moving_flat": ("moving_flat_visible", "moving_flat_late", "moving_flat_early_k4")}
LAM_REL = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]      # lambda = rel * n_train (features standardized -> diag(X'X) ~ n)


def arr(s):  # same parser as rollout2_test_readout.arr
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def n3(x, y):  # copied verbatim from v11_token_object_test.py
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


# ----------------------------------------------------------------------------------------------- selection
def select_clips(n_blocks):
    idx = [r for r in csv.DictReader(INDEX.open()) if r["probe_type"] == "obj"]
    rng = np.random.RandomState(SEED)
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
                    sel.append(dict(video_id=r["video_id"], cell=cell, cond=cond, k=k, block=f"{cond}_k{k}_{b}",
                                    violation_type=r["violation_type"], variant=r["variant"],
                                    shape=r["shape_pre"], color=r["color_pre"]))
    return sel


def extract(sel):
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    vid_row = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r")
    assert Z.shape[1] == 2048 and H.shape[1] == 4096, (Z.shape, H.shape)
    order = sorted(range(len(sel)), key=lambda i: vid_row[sel[i]["video_id"]])   # sequential disk reads
    n = len(sel); D = Z.shape[2]
    F = {f"{rep}_{ft}": np.zeros((n, D), np.float32) for rep in REPS for ft in FEATS}
    info = {"hidden_frac_t7": np.zeros(n), "win_n": np.zeros(n, int), "obj_px_t7": np.zeros((n, 2))}
    t0 = time.time()
    for j, i in enumerate(order):
        s = sel[i]; m = meta[s["video_id"]]
        t32 = np.stack([arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1)
        win = n3(*tub[T7]); info["win_n"][i] = len(win); info["obj_px_t7"][i] = tub[T7]
        if m["has_occlusion"] == "1" and m["hidden_start"] != "":
            hs, he = int(float(m["hidden_start"])), int(float(m["hidden_end"]))
            info["hidden_frac_t7"][i] = np.mean([(hs <= f <= he) for f in T7_FRAMES])
        r = vid_row[s["video_id"]]
        for rep, A in (("z", Z), ("h", H)):
            ctx = np.asarray(A[r, :2048], np.float32)          # 8 context tubelets x 256 tokens
            t7 = ctx[T7 * 256:(T7 + 1) * 256]
            F[f"{rep}_B7"][i] = t7[win].mean(0); F[f"{rep}_B7all"][i] = t7.mean(0)
            F[f"{rep}_C"][i] = ctx.mean(0); F[f"{rep}_BG"][i] = t7[BG_TOKENS].mean(0)
        if j % 200 == 0:
            print(f"  extract {j}/{n}  {time.time() - t0:.0f}s", flush=True)
    return F, info


# ----------------------------------------------------------------------------------------------- ridge
def _fit(X, Y, lam):
    n, d = X.shape
    if n <= d:
        K = X @ X.T; return X.T @ np.linalg.solve(K + lam * np.eye(n), Y)
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


class Ridge:
    def __init__(self, classes): self.classes = list(classes)

    def fit(self, X, y, lam):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        Y = np.eye(len(self.classes))[[self.classes.index(v) for v in y]]; Y = Y - Y.mean(0)
        self.W = _fit(Xs, Y, lam); self.b = -((Xs.mean(0)) @ self.W); return self

    def predict(self, X):
        return [self.classes[i] for i in (((X - self.mu) / self.sd) @ self.W + self.b).argmax(1)]


def stratified_block_folds(blocks, strata, n_folds, seed):
    """blocks: array of block ids per clip; strata: per clip stratum; returns fold id per clip (blocks kept together)."""
    rng = np.random.RandomState(seed)
    b2s = {}
    for b, s in zip(blocks, strata): b2s.setdefault(b, s)
    fold_of = {}
    for s in sorted(set(b2s.values())):
        bs = sorted(b for b in b2s if b2s[b] == s); rng.shuffle(bs)
        for i, b in enumerate(bs): fold_of[b] = i % n_folds
    return np.array([fold_of[b] for b in blocks])


def train_with_lambda_selection(X, y, blocks, strata, classes, seed):
    """pick lambda on one block-held-out validation fold (25%) inside the train set, refit on all."""
    f = stratified_block_folds(blocks, strata, 4, seed + 1); tr, va = f != 0, f == 0
    n_tr = tr.sum(); best = None
    for rel in LAM_REL:
        acc = float(np.mean(np.array(Ridge(classes).fit(X[tr], y[tr], rel * n_tr).predict(X[va])) == y[va]))
        if best is None or acc >= best[0] - 1e-12: best = (acc, rel)   # ties -> larger lambda
    lam = best[1] * len(y)
    return Ridge(classes).fit(X, y, lam), best[1], best[0]


def acc(model, X, y): return float(np.mean(np.array(model.predict(X)) == y))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--blocks", type=int, default=100); ap.add_argument("--reuse-features", action="store_true")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    fpath = OUT / "features.npz"
    if a.reuse_features and fpath.exists():
        npz = np.load(fpath, allow_pickle=True); sel = list(npz["sel"]); F = {k: npz[k] for k in npz if k.startswith(("z_", "h_"))}
        info = {k: npz[k] for k in ("hidden_frac_t7", "win_n", "obj_px_t7")}
    else:
        sel = select_clips(a.blocks); print(f"selected {len(sel)} clips", flush=True)
        F, info = extract(sel)
        np.savez(fpath, sel=np.array(sel, dtype=object), **F, **info)
    cell = np.array([s["cell"] for s in sel]); k = np.array([s["k"] for s in sel]); blk = np.array([s["block"] for s in sel])
    lab = {L: np.array([s[L] for s in sel]) for L in LABELS}
    strata = np.array([f"{s['k']}|{s['violation_type']}" for s in sel])
    classes = {L: sorted(set(lab[L])) for L in LABELS}
    rng = np.random.RandomState(SEED)

    # -- cell summary / sanity
    cells = {}
    for c, cond, ks in CELLS:
        for kk in ["all"] + ks:
            m = (cell == c) & ((k == kk) if kk != "all" else True)
            if kk != "all" and len(ks) == 1: continue
            cells[f"{c}|k={kk}"] = dict(condition=cond, k=kk, n_clips=int(m.sum()), n_blocks=len(set(blk[m])),
                                       hidden_frac_t7_mean=float(info["hidden_frac_t7"][m].mean()),
                                       win_n_mean=float(info["win_n"][m].mean()),
                                       obj_px_t7_x_range=[float(info["obj_px_t7"][m, 0].min()), float(info["obj_px_t7"][m, 0].max())],
                                       shape_counts=dict(Counter(lab["shape"][m])), color_counts=dict(Counter(lab["color"][m])))
    fits = []

    def rec(**kw): fits.append(kw); return kw

    def sub(c, kk=None):
        m = cell == c
        return m & (k == kk) if kk is not None else m

    for L in LABELS:
        y = lab[L]; ch = 1 / len(classes[L]); cls = classes[L]
        for rep in REPS:
            for ft in FEATS:
                X = F[f"{rep}_{ft}"]
                # (1) self, 4-fold CV by block
                for c, cond, ks in CELLS:
                    for kk in ([None] + ks if len(ks) > 1 else [None]):
                        m = sub(c, kk); Xm, ym, bm, sm = X[m], y[m], blk[m], strata[m]
                        folds = stratified_block_folds(bm, np.array([f"{s}|{l}" for s, l in zip(sm, ym)]), 4, SEED)
                        corr = corr_sh = 0; lams = []
                        for fo in range(4):
                            tr, te = folds != fo, folds == fo
                            mdl, rel, va = train_with_lambda_selection(Xm[tr], ym[tr], bm[tr], sm[tr], cls, SEED); lams.append(rel)
                            corr += int(np.sum(np.array(mdl.predict(Xm[te])) == ym[te]))
                            ysh = ym[tr].copy(); rng.shuffle(ysh)
                            mdl_s, _, _ = train_with_lambda_selection(Xm[tr], ysh, bm[tr], sm[tr], cls, SEED)
                            corr_sh += int(np.sum(np.array(mdl_s.predict(Xm[te])) == ym[te]))
                        rec(fit="self", label=L, rep=rep, feat=ft, train=c, test=c, k=kk or "all", n_train=int(m.sum()), n_test=int(m.sum()),
                            acc=corr / m.sum(), acc_shuffle=corr_sh / m.sum(), chance=ch, lam_rel=lams, note="4-fold CV by block")
                # (2)(3)(2b) transfer within family
                for fam, (vis, late, early) in FAMILIES.items():
                    for src_name, src in (("visible", vis), ("early", early)):
                        ms = sub(src); mdl, rel, va = train_with_lambda_selection(X[ms], y[ms], blk[ms], strata[ms], cls, SEED)
                        ysh = y[ms].copy(); rng.shuffle(ysh); mdl_s, _, _ = train_with_lambda_selection(X[ms], ysh, blk[ms], strata[ms], cls, SEED)
                        targets = [(late, None)] + [(late, kk) for kk in ["1", "2", "3", "4"]] + ([(early, None)] if src_name == "visible" else [])
                        for tc, kk in targets:
                            mt = sub(tc, kk)
                            rec(fit=f"transfer_{src_name}->{'early' if tc == early else 'late'}", label=L, rep=rep, feat=ft, train=src, test=tc,
                                k=kk or "all", n_train=int(ms.sum()), n_test=int(mt.sum()), acc=acc(mdl, X[mt], y[mt]),
                                acc_shuffle=acc(mdl_s, X[mt], y[mt]), chance=ch, lam_rel=rel, val_acc=va)
                print(f"  fits done: {L} {rep} {ft}", flush=True)

    res = dict(script=str(Path(__file__).relative_to(ROOT)), seed=SEED, n_blocks_per_cell=a.blocks, n_clips_total=len(sel),
               cache=str(CACHE), index=str(INDEX), bg_tokens=BG_TOKENS, lam_rel_grid=LAM_REL, classes=classes, cells=cells, fits=fits)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_md(res); print(f"-> {OUT}/results.json, RESULTS.md")


def write_md(res):
    fits = res["fits"]; L = []
    L += ["# E2 — boundary tubelet identity at the predictor input under occlusion (IntPhysGen v11_full, ViT-H)", "",
          f"Every number below is recomputed from `results.json` written by `{res['script']}` (seed {res['seed']}, {res['n_blocks_per_cell']} blocks = up to {2 * res['n_blocks_per_cell']} clips per (condition, k) cell, {res['n_clips_total']} clips total).",
          "Readout = closed-form one-vs-rest ridge (argmax), block-held-out, lambda chosen on an inner block-held-out validation fold. `shuf` = same fit with train labels permuted.",
          "Reps: `z` = ctx_masked (predictor input), `h` = target encoder (ceiling). Feats: B7 = 3x3 at true object position @t7, B7all = all 256 tokens @t7, C = all 2048 context tokens, BG = fixed far 3x3 @t7 (control).", ""]
    L += ["## Cells", "", "| cell | condition | k | n clips | n blocks | hidden frac of t7 frames (42,45) | 3x3 window size | obj x @t7 (px) |", "|---|---|---|---:|---:|---:|---:|---|"]
    for name, c in res["cells"].items():
        L.append(f"| {name.split('|')[0]} | {c['condition']} | {c['k']} | {c['n_clips']} | {c['n_blocks']} | {c['hidden_frac_t7_mean']:.2f} | {c['win_n_mean']:.1f} | {c['obj_px_t7_x_range'][0]:.0f}–{c['obj_px_t7_x_range'][1]:.0f} |")
    L.append("")

    def row(f): return f"{100 * f['acc']:.1f} ({100 * f['acc_shuffle']:.1f})"

    for lab in ("shape", "color"):
        ch = [f for f in fits if f["label"] == lab][0]["chance"]
        L += [f"## {lab}_pre — accuracy % (shuffle control in parentheses), chance {100 * ch:.1f}%", ""]
        for fam, (vis, late, early) in FAMILIES.items():
            L += [f"### {fam}", "", "| fit | train → test | k | n_train | n_test | " + " | ".join(f"{r}:{ft}" for r in REPS for ft in FEATS) + " |",
                  "|---|---|---|---:|---:|" + "---:|" * (len(REPS) * len(FEATS))]
            spec = [("self", vis, vis, "all"), ("self", early, early, "all"), ("self", late, late, "all")] + [("self", late, late, kk) for kk in "1234"] + \
                   [("transfer_visible->late", vis, late, "all")] + [("transfer_visible->late", vis, late, kk) for kk in "1234"] + \
                   [("transfer_visible->early", vis, early, "all"), ("transfer_early->late", early, late, "all")] + [("transfer_early->late", early, late, kk) for kk in "1234"]
            for fit, tr, te, kk in spec:
                cells_ = {(f["rep"], f["feat"]): f for f in fits if f["label"] == lab and f["fit"] == fit and f["train"] == tr and f["test"] == te and f["k"] == kk}
                if not cells_: continue
                f0 = next(iter(cells_.values()))
                L.append(f"| {fit} | {tr} → {te} | {kk} | {f0['n_train']} | {f0['n_test']} | " + " | ".join(row(cells_[(r, ft)]) for r in REPS for ft in FEATS) + " |")
            L.append("")
    L += ["## 재현", "", "```", f"cd {ROOT}", f"/data/hyuntak/anaconda3/envs/vjepa2/bin/python {res['script']} --blocks {res['n_blocks_per_cell']}", "```", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
