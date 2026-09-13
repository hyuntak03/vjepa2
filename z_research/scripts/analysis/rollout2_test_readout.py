#!/usr/bin/env python3
"""RollOut_v2 (test) — 학습셋(rollout_v2_training) 의 p 로 정한 위치 readout w_p 를 v2 의 p 에 test 로만 건다.

자: `rollout2_fit_readout.py` 가 저장한 w_p.npy (1281, 2). 미래 8 슬롯 × 256 토큰 공간평균 → (8, 1280) → x̂, ŷ.
v2 의 어떤 표현도 fitting 에 쓰지 않는다. 가능(possible) 클립만 채점한다 (불가능 클립은 정답 궤적이 둘이라 별도).
지표 (시나리오별): R² (슬롯 전체), MAE px, β = 읽은 슬롯 기울기 vs 실제 기울기의 회귀 계수 (움직이는 축만; 1 이면 속도까지 맞음).
저장: exp_results/<train>/spatial_pooling/p/test/{test.json, preds.npz}   (<train> = ROLLOUT2_TRAIN, 기본 v5)   (preds: video_id, pred (n,8,2), truth (n,8,2), 정규화 좌표 px/144−1)

  python z_research/scripts/analysis/rollout2_test_readout.py
"""
from __future__ import annotations
import csv, json, os, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
# ── 학습셋 (환경변수 ROLLOUT2_TRAIN, 기본 v5). 모든 readout 스크립트가 여기서 경로를 가져간다 ──
#   v5: RollOut_v2_training (09-11 19:34) = 사물 없음 4,480 (line_x/line_z/line_xz/still) + 사물 1개 3,584 (prop_x/prop_still, 절반은 가림) = 8,064 clip
#       index/캐시 이름 rollout_v2_training_v5 → exp_results/v5, figures/v5
TRAIN = os.environ.get("ROLLOUT2_TRAIN", "v5")                                  # 학습셋 버전. v1~v4 는 2026-09-11 삭제 — v5 만 남았다
TR_NAME = {"v5": "rollout_v2_training_v5"}[TRAIN]
TR_CACHE = Path("/local_datasets/world/world_analysis/cache") / f"{TR_NAME}_vith"; TR_CACHE_CTX32 = Path("/local_datasets/world/world_analysis/cache") / f"{TR_NAME}_ctx32_vith"
TR_INDEX = ROOT / f"data_csv/{TR_NAME}/index_probe.csv"
RES_ROOT = ROOT / "z_research/RollOutV2/exp_results" / TRAIN; FIG_ROOT = ROOT / "z_research/RollOutV2/figures" / TRAIN
W = RES_ROOT / "spatial_pooling/p/fit/w_p.npy"
OUT = RES_ROOT / "spatial_pooling/p/test"
POOLED = CACHE / "_pooled_p8.npz"
S, T, D, RES = 256, 8, 1280, 144.0
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
MOVING = {"flat_v": "x", "flat_a": "x", "ramp_a": "x", "arc": "xy", "fall": "y", "ledge": "xy", "wall": ""}   # 미래 8 슬롯에서 움직이는 축 (wall 가능 = 정지)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def slope(x):
    t = np.arange(T) - (T - 1) / 2; return (x * t).sum(1) / (t ** 2).sum()


def beta(b_hat, b_true):
    if len(np.unique(np.round(b_true, 6))) < 3:
        return None
    A = np.vstack([b_true, np.ones_like(b_true)]).T; return float(np.linalg.lstsq(A, b_hat, rcond=None)[0][0])


def pool_p(n):
    if POOLED.is_file():
        z = np.load(POOLED)
        if z["n"] == n:
            return z["p"]
    P = np.load(CACHE / "predictor.npy", mmap_mode="r"); X = np.empty((n, T, D), np.float32); t0 = time.time()
    for k in range(0, n, 64):
        X[k:k + 64] = np.asarray(P[k:k + 64]).reshape(-1, T, S, D).mean(2)
        if k % 1280 == 0:
            print(f"    풀링 {k}/{n} {time.time()-t0:.0f}s", flush=True)
    np.savez(POOLED, n=n, p=X); return X


def main():
    idx = list(csv.DictReader(INDEX.open())); n = len(idx)
    meta = json.loads((CACHE / "meta.json").read_text())
    if meta["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("캐시 video_ids 가 index 와 다르다")
    w = np.load(W); print(f"w_p {w.shape}  ← {W}")
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(n, 16, 2).mean(2) / RES - 1, py.reshape(n, 16, 2).mean(2) / RES - 1], -1)[:, T:]
    inf = (np.stack([arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(n, 16, 2).all(2)[:, T:]
    scen = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])

    X = pool_p(n); pred = (X.reshape(-1, D).astype(np.float64) @ w[:-1] + w[-1]).reshape(n, T, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "preds.npz", video_id=np.array([r["video_id"] for r in idx]), pred=pred, truth=L, in_frame=inf, scenario=scen, plausible=plaus)

    rep = {"w": str(W), "cache": str(CACHE), "n_clip": n, "per_scenario": {}}
    print(f"\n{'scenario':<8} {'n':>4}  {'R²x':>6} {'R²y':>6}  {'MAEx':>6} {'MAEy':>6}   {'βx':>5} {'βy':>5}")
    for scn in SCEN:
        k = (scen == scn) & (plaus == 1); m = inf[k]; p = pred[k]; y = L[k]
        rec = {"n": int(k.sum()), "r2_x": r2(p[:, :, 0][m], y[:, :, 0][m]) if "x" in MOVING[scn] else None,
               "r2_y": r2(p[:, :, 1][m], y[:, :, 1][m]) if "y" in MOVING[scn] else None,
               "mae_px_x": float(np.abs(p[:, :, 0] - y[:, :, 0])[m].mean() * RES), "mae_px_y": float(np.abs(p[:, :, 1] - y[:, :, 1])[m].mean() * RES)}
        for c, cn in ((0, "x"), (1, "y")):
            rec[f"beta_{cn}"] = beta(slope(p[:, :, c]), slope(y[:, :, c])) if cn in MOVING[scn] else None
        rep["per_scenario"][scn] = rec
        f = lambda v: "   -  " if v is None else f"{v:6.2f}"
        print(f"{scn:<8} {rec['n']:>4}  {f(rec['r2_x'])} {f(rec['r2_y'])}  {rec['mae_px_x']:6.1f} {rec['mae_px_y']:6.1f}   {f(rec['beta_x'])[1:]} {f(rec['beta_y'])[1:]}")
    k = plaus == 1; m = inf[k]
    rep["all_possible"] = {"r2_x": r2(pred[k][:, :, 0][m], L[k][:, :, 0][m]), "r2_y": r2(pred[k][:, :, 1][m], L[k][:, :, 1][m]),
                           "mae_px_x": float(np.abs(pred[k] - L[k])[:, :, 0][m].mean() * RES), "mae_px_y": float(np.abs(pred[k] - L[k])[:, :, 1][m].mean() * RES)}
    a = rep["all_possible"]; print(f"{'all':<8} {int(k.sum()):>4}  {a['r2_x']:6.2f} {a['r2_y']:6.2f}  {a['mae_px_x']:6.1f} {a['mae_px_y']:6.1f}")
    print("R²·β 는 움직이는 축만 (정지 축은 MAE 만). wall 은 미래에 x 도 정지라 MAE 만 본다.")
    (OUT / "test.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(f"→ {OUT}")


if __name__ == "__main__":
    main()
