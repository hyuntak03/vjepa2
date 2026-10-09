#!/usr/bin/env python3
"""H1 분석 — 가속은 z 에 있는가 (층별 ridge, CPU).

표적 (RollOut_v3 가능 clip, 자유 운동 6 법칙 — flat_v/a/d, ramp_a/d, arc; px, 튜블릿 단위):
  x_T         문맥 마지막 튜블릿 물체 위치 (2 프레임 평균)
  v_T = x_T − x_{T−1},   a_T = x_T − 2x_{T−1} + x_{T−2}
  r_k = x_{T+k} − x_T − k·v_T      k = 2, 4, 8 튜블릿 — 등속 외삽이 못 맞히는 미래 (= 가속이 만드는 몫)
특징 (층마다): 문맥 마지막 3 튜블릿 물체 3×3 창 평균을 이어 붙임 (3×D).  encoder l = 0..31, encF (= z), tgt (문맥만 본 target encoder), pctx l = 0..11.
ridge (dual, λ 는 train 안 5-fold), clip 단위 반반 split × 5 seed, 법칙으로 층화.
R² 두 가지:  pooled (전체 분산) · within (법칙마다 평균을 뺀 분산 — 법칙 정체만 맞혀서 얻는 몫을 뺀다).
대조:  state = [x_T, v_T] (px, 2 축씩)  — 위치·속도를 알면 a 가 얼마나 풀리나 (지름길 크기)
        shuffle = 특징 행을 섞음 (0 근처여야)

  srun -w vll6 -c 16 --mem 64G python auto_research/scripts/h1_analyze_v3.py
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h1"
OUT = ROOT / "auto_research/exp_results/h1"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
LAMS = 10.0 ** np.arange(-2, 6)


def arr(s):
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def tub(X, f0):
    return (X[:, f0] + X[:, f0 + 1]) / 2


def ridge_dual(Xtr, Ytr, Xte, lam):
    K = Xtr @ Xtr.T
    A = np.linalg.solve(K + lam * np.eye(len(K)), Ytr)
    return Xte @ (Xtr.T @ A)


def fit_predict(Xtr, Ytr, Xte, rng):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0); Ytr = Ytr - ym
    folds = rng.permutation(len(Xtr)) % 5
    best, bl = None, None
    for lam in LAMS:
        err = 0.0
        for f in range(5):
            tr, va = folds != f, folds == f
            P = ridge_dual(Xtr[tr], Ytr[tr], Xtr[va], lam)
            err += ((P - Ytr[va]) ** 2).sum()
        if best is None or err < best:
            best, bl = err, lam
    return ridge_dual(Xtr, Ytr, Xte, bl) + ym, bl


def r2(P, Y, groups=None):
    if groups is not None:                               # 법칙 안 분산
        Yc, Pc = Y.copy(), P.copy()
        for g in np.unique(groups):
            m = groups == g
            mu = Y[m].mean(0); Yc[m] -= mu; Pc[m] -= mu
        Y, P = Yc, Pc
        return 1 - ((P - Y) ** 2).sum(0) / (Y ** 2).sum(0)
    return 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=IN_DEFAULT)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    I = Path(a.inp); OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(I / "meta.json"))
    ids, split = meta["video_ids"], meta["split"]
    rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    R = [rows[v] for v in ids]
    scen = np.array([r["scenario"] for r in R]); plaus = np.array([r["plausible"] == "1" for r in R])
    keep = plaus & np.isin(scen, FREE)
    X = np.stack([arr(r["px_x_by_sample"]) for r in R]); Y = np.stack([arr(r["px_y_by_sample"]) for r in R])
    ok = np.stack([arr(r["in_frame_by_sample"]) for r in R]) > 0
    xT = np.stack([tub(X, split - 2), tub(Y, split - 2)], 1)
    xT1 = np.stack([tub(X, split - 4), tub(Y, split - 4)], 1)
    xT2 = np.stack([tub(X, split - 6), tub(Y, split - 6)], 1)
    v = xT - xT1; acc = xT - 2 * xT1 + xT2
    targets = {"v": v, "a": acc}
    for k in (2, 4, 8):
        f0 = split - 2 + 2 * k
        xk = np.stack([tub(X, f0), tub(Y, f0)], 1)
        targets[f"r{k}"] = xk - xT - k * v
        keep &= ok[:, f0] & ok[:, f0 + 1]
    keep &= ok[:, split - 6:split].all(1)
    idx = np.where(keep)[0]; g = scen[idx]
    print("clips", len(idx), {s: int((g == s).sum()) for s in FREE})
    state = np.concatenate([xT, v], 1)[idx]
    res = {}
    for C in meta["Cs"]:
        enc = np.load(I / f"enc_C{C}.npy", mmap_mode="r"); encF = np.load(I / f"encF_C{C}.npy")
        tgt = np.load(I / f"tgt_C{C}.npy"); pctx = np.load(I / f"pctx_C{C}.npy", mmap_mode="r")
        feats = {f"enc{l}": lambda l=l: np.asarray(enc[idx, :, l], np.float64).reshape(len(idx), -1) for l in range(32)}
        feats["encF"] = lambda: encF[idx].reshape(len(idx), -1).astype(np.float64)
        feats["tgt"] = lambda: tgt[idx].reshape(len(idx), -1).astype(np.float64)
        for l in range(12):
            feats[f"pctx{l}"] = lambda l=l: np.asarray(pctx[idx, :, l], np.float64).reshape(len(idx), -1)
        feats["state"] = lambda: state
        feats["shuffle_encF"] = lambda: encF[idx].reshape(len(idx), -1)[np.random.default_rng(1).permutation(len(idx))].astype(np.float64)
        out = {}
        for name, fx in feats.items():
            F = fx()
            acc_r = {t: {"pooled": [], "within": []} for t in targets}
            lam_used = []
            for sd in range(a.seeds):
                rng = np.random.default_rng(sd)
                tr = np.zeros(len(idx), bool)
                for s in FREE:
                    m = np.where(g == s)[0]; tr[rng.permutation(m)[: len(m) // 2]] = True
                Yall = np.concatenate([targets[t][idx] for t in targets], 1)
                P, lam = fit_predict(F[tr], Yall[tr], F[~tr], rng); lam_used.append(float(lam))
                c = 0
                for t in targets:
                    d = targets[t].shape[1]
                    acc_r[t]["pooled"].append(r2(P[:, c:c + d], Yall[~tr, c:c + d]).tolist())
                    acc_r[t]["within"].append(r2(P[:, c:c + d], Yall[~tr, c:c + d], g[~tr]).tolist())
                    c += d
            out[name] = {t: {k: np.mean(v_, 0).round(4).tolist() for k, v_ in d_.items()} for t, d_ in acc_r.items()}
            out[name]["lam"] = lam_used
            print(f"C{C} {name:12s} a within (x,y) {out[name]['a']['within']}  r8 within {out[name]['r8']['within']}  v pooled {out[name]['v']['pooled']}", flush=True)
        res[f"C{C}"] = out
    json.dump(res, open(OUT / "h1_v3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
