#!/usr/bin/env python3
"""정보가 주성분 스펙트럼의 어디에 퍼져 있나 — PCA/t-SNE 그림이 쓸모 있을지 먼저 잰다.

t-SNE 나 PCA 산점도는 **분산이 큰 축**을 먼저 보여 준다. 그런데 우리가 쫓는 신호는
두 후보 미래를 가르는 성분이고, 그건 걸음 전체의 0.1~1% 다(step_direction.py 실측).
그래서 "상위 k 개 주성분만으로 shape/color/env 를 얼마나 읽을 수 있나" 를 재면
그림을 그릴 가치가 있는지 바로 판단된다.

  top-2/top-10 에서 env 만 나오고 shape 이 chance 면 -> 산점도는 배경만 보여 준다
  shape 이 상위에서 이미 나오면                      -> 그림이 실제로 쓸모 있다

  python z_research/scripts/analysis/pca_spectrum.py --bases predictor target
"""
from __future__ import annotations
import argparse, collections, csv, json
from pathlib import Path
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
COL = {"shape": "shape_pre", "color": "color_pre", "env": "env"}


def pooled(cache: Path, base: str, rows_idx, half=None, chunk=64):
    m = json.loads((cache / "meta.json").read_text())
    n, t, d = len(m["video_ids"]), m["base_counts"][base], m["embed_dim"]
    mm = np.memmap(cache / f"{base}.npy", dtype=np.float16, mode="r", shape=(n, t, d))
    sl = slice(t // 2, t) if half == "future" else slice(0, t)
    out = np.empty((len(rows_idx), d), np.float32)
    for s in range(0, len(rows_idx), chunk):
        out[s:s + chunk] = np.asarray(mm[rows_idx[s:s + chunk], sl], np.float32).mean(1)
    return out


def ridge_cv(X, y, K, groups, lam=1e2, folds=5):
    g = np.array(groups); uq = np.unique(g)
    rng = np.random.default_rng(0)
    part = {b: i % folds for i, b in enumerate(rng.permutation(uq))}
    fold = np.array([part[b] for b in g])
    acc = []
    for f in range(folds):
        tr, te = fold != f, fold == f
        Xtr = np.c_[X[tr], np.ones(tr.sum())]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ np.eye(K)[y[tr]])
        acc.append(float(((np.c_[X[te], np.ones(te.sum())] @ W).argmax(1) == y[te]).mean()))
    return 100 * float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/local_datasets/world/world_analysis/cache/v10_vith"))
    ap.add_argument("--index", type=Path, default=Path("data_csv/intphysgen_v10/index_probe.csv"))
    ap.add_argument("--bases", nargs="*", default=["predictor", "target"])
    a = ap.parse_args()

    vids = json.loads((a.cache / "meta.json").read_text())["video_ids"]
    pos = {v: i for i, v in enumerate(vids)}
    rows = [r for r in csv.DictReader(a.index.open()) if r["probe_type"] == "obj"]
    idx = np.array([pos[r["video_id"]] for r in rows])
    block = [r["block_id"] for r in rows]
    lab, K = {}, {}
    for t, c in COL.items():
        cls = sorted({r[c] for r in rows}); K[t] = len(cls)
        lab[t] = np.array([cls.index(r[c]) for r in rows])
    lab["condition"] = np.array([CONDS.index(r["condition"]) for r in rows]); K["condition"] = 4

    KS = [2, 5, 10, 25, 50, 100, 300, 1280]
    for base in a.bases:
        X = pooled(a.cache, base, idx, half="future" if base == "target" else None)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        Z = U * S                                    # 주성분 좌표
        print(f"\n### {base}   (obj {len(rows)}행, 차원별 표준화 후 PCA)")
        print(f"  {'top-k':>7}{'누적분산':>10}" + "".join(f"{t:>10}" for t in ["shape", "color", "env", "condition"])
              + "     chance 12.5 / 12.5 / 25.0 / 25.0")
        for k in KS:
            accs = [ridge_cv(Z[:, :k], lab[t], K[t], block) for t in ["shape", "color", "env", "condition"]]
            print(f"  {k:>7}{100*var[:k].sum():>9.1f}%" + "".join(f"{x:>10.1f}" for x in accs))


if __name__ == "__main__":
    main()
