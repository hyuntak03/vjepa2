#!/usr/bin/env python3
"""RollOut_v2 — p-fit 천장. v2 가능 클립의 30% (시나리오 층화, clip 단위) 로 같은 자 (풀링 → (1281, 2) 최소제곱) 를 v2 의 p 에
정하고 나머지 70% 에서 잰다. 같은 70% 에 학습셋 w (`p/fit/w_p.npy`) 도 걸어 나란히 놓는다.
  천장 ≫ 학습셋-w : 그 시나리오의 p 에 슬롯별 위치가 있지만 학습셋 코드와 다른 방향
  천장 ≈ 학습셋-w (둘 다 낮음): p 에 슬롯별 위치가 없다 (predictor 가 그 운동을 못 그린다)
학습셋 w 의 결론(이식)은 이 스크립트가 바꾸지 않는다 — 대조용.
저장: exp_results/<train>/spatial_pooling/p/ceiling/{ceiling.json, preds.npz}

  python z_research/scripts/analysis/rollout2_ceiling.py [--seed 0] [--frac 0.3]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout2_test_readout as rt                         # noqa: E402
from rollout2_fit_readout import fit, apply                # noqa: E402

OUT = rt.RES_ROOT / "spatial_pooling/p/ceiling"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--frac", type=float, default=0.3); a = ap.parse_args()
    idx = list(csv.DictReader(rt.INDEX.open())); n = len(idx)
    meta = json.loads((rt.CACHE / "meta.json").read_text())
    if meta["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("캐시 video_ids 가 index 와 다르다")
    px = np.stack([rt.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([rt.arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(n, 16, 2).mean(2) / rt.RES - 1, py.reshape(n, 16, 2).mean(2) / rt.RES - 1], -1)[:, rt.T:]
    inf = (np.stack([rt.arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(n, 16, 2).all(2)[:, rt.T:]
    scen = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    X = rt.pool_p(n); w_tr = np.load(rt.W)

    rng = np.random.RandomState(a.seed); tr, te = [], []
    for s in rt.SCEN:
        ids = np.where((scen == s) & (plaus == 1))[0]; rng.shuffle(ids); k = int(round(a.frac * len(ids))); tr += list(ids[:k]); te += list(ids[k:])
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    m = inf[tr].ravel(); w_v2 = fit(X[tr].reshape(-1, rt.D).astype(np.float64)[m], L[tr].reshape(-1, 2)[m])
    print(f"v2 p-fit: train {len(tr)} clip ({len(tr)*rt.T} 행) / test {len(te)} clip   학습셋 w ← {rt.W.name}")
    pred = {"v2_fit": apply(X[te].reshape(-1, rt.D).astype(np.float64), w_v2).reshape(len(te), rt.T, 2),
            "train_w": apply(X[te].reshape(-1, rt.D).astype(np.float64), w_tr).reshape(len(te), rt.T, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "preds.npz", video_id=np.array([idx[i]["video_id"] for i in te]), truth=L[te], in_frame=inf[te], scenario=scen[te], **{f"pred_{k}": v for k, v in pred.items()})
    rep = {"seed": a.seed, "frac": a.frac, "n_train": int(len(tr)), "n_test": int(len(te)), "per_scenario": {}}
    print(f"\n{'scenario':<8} {'n':>4} | {'v2 p-fit (천장)':^28} | {'학습셋 w':^28}\n{'':<8} {'':>4} | {'R²x':>6} {'R²y':>6} {'MAEx':>5} {'MAEy':>5} {'βx':>5} {'βy':>5} | {'R²x':>6} {'R²y':>6} {'MAEx':>5} {'MAEy':>5} {'βx':>5} {'βy':>5}")
    f = lambda v: "  -  " if v is None else f"{v:5.2f}"
    for s in rt.SCEN:
        k = scen[te] == s; mm = inf[te][k]; y = L[te][k]; row = {}
        for name in ("v2_fit", "train_w"):
            p = pred[name][k]; mv = rt.MOVING[s]
            row[name] = {"r2_x": rt.r2(p[:, :, 0][mm], y[:, :, 0][mm]) if "x" in mv else None, "r2_y": rt.r2(p[:, :, 1][mm], y[:, :, 1][mm]) if "y" in mv else None,
                         "mae_px_x": float(np.abs(p - y)[:, :, 0][mm].mean() * rt.RES), "mae_px_y": float(np.abs(p - y)[:, :, 1][mm].mean() * rt.RES),
                         "beta_x": rt.beta(rt.slope(p[:, :, 0]), rt.slope(y[:, :, 0])) if "x" in mv else None,
                         "beta_y": rt.beta(rt.slope(p[:, :, 1]), rt.slope(y[:, :, 1])) if "y" in mv else None}
        rep["per_scenario"][s] = {"n_test": int(k.sum()), **row}
        c = lambda r: f"{f(r['r2_x']):>6} {f(r['r2_y']):>6} {r['mae_px_x']:5.1f} {r['mae_px_y']:5.1f} {f(r['beta_x'])} {f(r['beta_y'])}"
        print(f"{s:<8} {int(k.sum()):>4} | {c(row['v2_fit'])} | {c(row['train_w'])}")
    print("R²·β 는 움직이는 축만. 같은 70% test 클립, 같은 자 형태 (풀링 → 1281 × 2 최소제곱).")
    (OUT / "ceiling.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(f"→ {OUT}")


if __name__ == "__main__":
    main()
