#!/usr/bin/env python3
"""R3-(B): encoder 운동 판독 독립 재계산. ego_{pan,ssv2,ek100}. 자체 ridge (닫힌 해, Gram 공유), fold seed 다름, 5/10-fold, EK100 은 영상 단위 split 도.
추가: 64 토큰 부표본 평균, per-clip 상수 baseline (leak 검사), 전역 몫 (median vs mean, out-of-frame 제외), flow 자기 일관성."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); R = ROOT / "auto_research/_stage/results_ar"; OUT = ROOT / "auto_research/_verify_scratch/R3/out"; OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def r2(Y, P):
    ss = ((Y - Y.mean(0)) ** 2).sum(0); return 1 - ((Y - P) ** 2).sum(0) / np.maximum(ss, 1e-9)


def ridge_cv(X, Y, groups, k, seed, alpha_by_inner=True):
    """X (N,d) float32, Y (N,m). groups (N,) 정수 — fold 는 group 단위. α 는 열별 inner 3-fold (Gram 공유)."""
    rng = np.random.default_rng(seed); N, d = X.shape; m = Y.shape[1]
    ug = np.unique(groups); rng.shuffle(ug); folds = np.array_split(ug, k); P = np.zeros((N, m))
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = ((X - mu) / sd).astype(np.float64); ym = Y[tr].mean(0); Yc = (Y - ym).astype(np.float64)
        if alpha_by_inner:
            utr = np.unique(groups[tr]); inner = np.array_split(rng.permutation(utr), 3); sc = np.zeros((len(ALPHAS), m))
            for g in inner:
                ite = tr & np.isin(groups, g); itr = tr & ~ite; Gm = Xs[itr].T @ Xs[itr]; XY = Xs[itr].T @ Yc[itr]
                for ai, a in enumerate(ALPHAS):
                    W = np.linalg.solve(Gm + a * itr.sum() * np.eye(d), XY); sc[ai] += r2(Yc[ite], Xs[ite] @ W)
            best = sc.argmax(0)
        else:
            best = np.full(m, ALPHAS.index(1))
        Gm = Xs[tr].T @ Xs[tr]; XY = Xs[tr].T @ Yc[tr]
        for ai in np.unique(best):
            cols = np.where(best == ai)[0]; W = np.linalg.solve(Gm + ALPHAS[ai] * tr.sum() * np.eye(d), XY[:, cols]); P[np.ix_(te, cols)] = Xs[te] @ W + ym[cols]
    return r2(Y, P)


def run(src):
    I = R / f"ego_{src}"; meta = json.load(open(I / "meta.json")); n = meta["n"]
    Z = np.load(I / "z_last.npy", mmap_mode="r"); FL = np.load(I / "flow.npy"); GF = np.load(I / "gflow.npy"); FU = np.load(I / "fut.npy")
    done = np.isfinite(GF[:, 0]) & np.isfinite(np.asarray(Z[:, 0, 0], dtype=np.float32)); idx = np.where(done)[0]
    rng = np.random.default_rng(7); res = dict(n=int(len(idx)))
    # 영상 id (EK100: file), clip 인덱스
    vid = np.array([meta["items"][c].get("file", str(c)) for c in idx]); _, vgrp = np.unique(vid, return_inverse=True)
    res["n_videos"] = int(len(np.unique(vid)))
    Xg = np.stack([np.asarray(Z[c], dtype=np.float32).mean(0) for c in idx]); Yg = GF[idx].astype(np.float64)
    sub = rng.choice(256, 64, replace=False); Xg64 = np.stack([np.asarray(Z[c][sub], dtype=np.float32).mean(0) for c in idx])
    Ysh = Yg[rng.permutation(len(idx))]
    res["global_R2_5fold_clip"] = ridge_cv(Xg, Yg, idx, 5, 11).round(3).tolist()
    res["global_R2_10fold_clip"] = ridge_cv(Xg, Yg, idx, 10, 12).round(3).tolist()
    res["global_R2_5fold_video"] = ridge_cv(Xg, Yg, vgrp, 5, 13).round(3).tolist()
    res["global_R2_64tok_mean_5fold_clip"] = ridge_cv(Xg64, Yg, idx, 5, 11).round(3).tolist()
    res["global_R2_shuffled"] = ridge_cv(Xg, Ysh, idx, 5, 11).round(3).tolist()
    # per-clip constant baseline (예측 = 학습 fold 평균) → R² 는 정의상 ≈ 0; 대신 "clip 구조 leak" 검사 = fold 를 clip 이 아니라 토큰으로 나누면 얼마나 오르나 (국소 판)
    # 국소
    Xl, Yl, Cl, Yf, Vl = [], [], [], [], []
    for j, c in enumerate(idx):
        s = rng.choice(256, 64, replace=False); Xl.append(np.asarray(Z[c][s], dtype=np.float32)); Yl.append(FL[c][s]); Cl.append(np.full(64, c)); Yf.append(FU[c][:, s]); Vl.append(np.full(64, vgrp[j]))
    Xl, Yl, Cl, Vl = np.concatenate(Xl), np.concatenate(Yl).astype(np.float64), np.concatenate(Cl), np.concatenate(Vl); Yf = np.concatenate(Yf, 1)
    Yres = Yl - GF[Cl]; Yfut = np.nan_to_num(np.transpose(Yf, (1, 0, 2)).reshape(len(Yl), 16), nan=0.0)
    Yall = np.concatenate([Yl, Yres, Yl[rng.permutation(len(Yl))], Yfut], 1)
    rl = ridge_cv(Xl, Yall, Cl, 5, 21).round(3)
    res["local_R2_5fold_clip"] = rl[:2].tolist(); res["local_residual_R2"] = rl[2:4].tolist(); res["local_shuffled"] = rl[4:6].tolist(); res["future_R2_by_slot"] = rl[6:].reshape(8, 2).tolist()
    rlv = ridge_cv(Xl, Yall[:, :2], Vl, 5, 22).round(3); res["local_R2_5fold_video"] = rlv.tolist()
    # token-level random folds (clip leak 검사): 같은 clip 의 다른 토큰이 train 에 있으면 얼마나 오르나
    rlt = ridge_cv(Xl, Yall[:, :2], np.arange(len(Yl)), 5, 23).round(3); res["local_R2_tokenfolds(leak_check)"] = rlt.tolist()
    # flow 자기 일관성 (표적 신뢰도 상한 대리): 문맥 끝 flow → 슬롯0 1걸음 변위 (토큰 / 전역), clip-CV 선형
    X1 = FL[idx].reshape(-1, 2); Y1 = -FU[idx, 0].reshape(-1, 2); ok = np.isfinite(Y1).all(1); C1 = np.repeat(idx, 256)
    res["flow_selfconsist_token_R2_cv"] = ridge_cv(X1[ok].astype(np.float32), Y1[ok].astype(np.float64), C1[ok], 5, 31, alpha_by_inner=False).round(3).tolist()
    Yg1 = -np.nanmedian(FU[idx, 0], 1); res["flow_selfconsist_global_R2_cv"] = ridge_cv(GF[idx].astype(np.float32), Yg1.astype(np.float64), idx, 5, 32, alpha_by_inner=False).round(3).tolist()
    # 전역 몫: median vs mean, out-of-frame 제외 (역추적 위치가 화면 밖 = |fut|>256 로 근사 못 함 → fut 가 nan 인 토큰 제외 + |fut| 큰 값 제외)
    share = {}
    for i in (0, 3, 7):
        F = FU[idx, i]; fin = np.isfinite(F).all(2)
        for nm, agg in (("median", np.nanmedian), ("mean", np.nanmean)):
            m = agg(np.where(fin[..., None], F, np.nan), 1, keepdims=True); resid = np.where(fin[..., None], F - m, np.nan)
            share[f"slot{i}|{nm}"] = (1 - np.nanvar(resid, axis=(0, 1)) / np.nanvar(np.where(fin[..., None], F, np.nan), axis=(0, 1))).round(3).tolist()
        # in-frame only: 출처가 화면 안 (|x + fut| in [0,256))
        cx = (np.arange(256) % 16) * 16 + 8; cy = (np.arange(256) // 16) * 16 + 8; px = cx[None] + F[..., 0]; py = cy[None] + F[..., 1]
        inf_ = fin & (px >= 0) & (px < 256) & (py >= 0) & (py < 256)
        m = np.nanmedian(np.where(inf_[..., None], F, np.nan), 1, keepdims=True); resid = np.where(inf_[..., None], F - m, np.nan)
        share[f"slot{i}|median_inframe"] = (1 - np.nanvar(resid, axis=(0, 1)) / np.nanvar(np.where(inf_[..., None], F, np.nan), axis=(0, 1))).round(3).tolist(); share[f"slot{i}|inframe_frac"] = round(float(inf_.mean()), 3)
    res["global_share"] = share
    json.dump(res, open(OUT / f"b_{src}.json", "w"), indent=1)
    print("==", src); [print(" ", k, v) for k, v in res.items()]
    return res


if __name__ == "__main__":
    for s in sys.argv[1:] or ("pan", "ssv2", "ek100"): run(s)
