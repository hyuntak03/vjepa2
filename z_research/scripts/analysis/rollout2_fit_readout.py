#!/usr/bin/env python3
"""RollOut_v2_training (무중력 직선 896 clip) 의 p 로 위치 readout w 를 정하고 저장한다. test 는 하지 않는다.

자: 미래 8 슬롯 × 256 토큰 공간평균 → (8, 1280); 좌표(x, y)마다 w 하나 (1280 + bias) 를 슬롯·clip 전부가 공유.
라벨: 튜블릿(2 샘플) 평균 픽셀 / 144 − 1 ∈ [−1, 1].  행 = 896 × 8 = 7,168, 파라미터 1,281 (좌표당).
순수 최소제곱 (ridge 없음). 896 clip 전부로 정한다 — test 는 rollout_v2 이므로 학습셋 안 held-out 은 두지 않는다
(2026-09-10 50/50 확인: held-out R² x 0.91 / y 0.90, fit 대비 −0.05 → 표본 충분).

저장: z_research/RollOutV2/exp_results/readout_training/{w_p.npy, fit.json}
  w_p.npy  (1281, 2) float64 — [:1280] 가중치, [1280] bias; 열 0 = x, 열 1 = y.  적용: X @ w[:-1] + w[-1]

  python z_research/scripts/analysis/rollout2_fit_readout.py
"""
from __future__ import annotations
import csv, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v2_training_vith")
INDEX = ROOT / "data_csv/rollout_v2_training/index_probe.csv"
OUT = ROOT / "z_research/RollOutV2/exp_results/readout_training"
S, T, D, RES = 256, 8, 1280, 144.0


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def fit(X, Y):
    """중심화 최소제곱. 반환 (D+1, 2): 마지막 행이 bias."""
    mu = X.mean(0); ym = Y.mean(0)
    w = np.linalg.lstsq(X - mu, Y - ym, rcond=None)[0]
    return np.vstack([w, ym - mu @ w])


def apply(X, w):
    return X @ w[:-1] + w[-1]


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    idx = list(csv.DictReader(INDEX.open())); n = len(idx)
    meta = json.loads((CACHE / "meta.json").read_text())
    if meta["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("캐시 video_ids 가 index 와 다르다")
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(n, 16, 2).mean(2) / RES - 1, py.reshape(n, 16, 2).mean(2) / RES - 1], -1)[:, T:]   # (n, 8, 2) 미래
    inf = (np.stack([arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(n, 16, 2).all(2)[:, T:]
    scen = np.array([r["scenario"] for r in idx])
    assert inf.all(), "학습셋 미래 슬롯에 off-screen 이 있다"

    P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    X = np.empty((n, T, D), np.float32); t0 = time.time()
    for k in range(0, n, 64):
        X[k:k + 64] = np.asarray(P[k:k + 64]).reshape(-1, T, S, D).mean(2)
    print(f"풀링 {n} clip  {time.time()-t0:.0f}s   행 {n*T}, 파라미터 {D+1}/좌표")

    Xf = X.reshape(-1, D).astype(np.float64); Yf = L.reshape(-1, 2)
    w = fit(Xf, Yf); OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "w_p.npy", w)
    pr = apply(Xf, w); rep = {"n_clip": n, "rows": n * T, "params_per_coord": D + 1, "cache": str(CACHE), "source": "predictor future 8 tubelets, spatial mean",
                              "label": "tubelet-mean px / 144 - 1", "r2_x_fit": r2(pr[:, 0], Yf[:, 0]), "r2_y_fit": r2(pr[:, 1], Yf[:, 1]),
                              "mae_px_x": float(np.abs(pr[:, 0] - Yf[:, 0]).mean() * RES), "mae_px_y": float(np.abs(pr[:, 1] - Yf[:, 1]).mean() * RES)}
    print(f"p: fit R² x {rep['r2_x_fit']:.3f} y {rep['r2_y_fit']:.3f}   MAE {rep['mae_px_x']:.1f} / {rep['mae_px_y']:.1f} px   → {OUT / 'w_p.npy'}  shape {w.shape}")
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"→ {OUT / 'fit.json'}")


if __name__ == "__main__":
    main()
