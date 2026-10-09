# 판독 허용폭과 무관한 직접 척도: 외형 템플릿 argmax (loc_app) 의 진행 방향 변위 (칸) 곡선 — 멈춘 자리 (plateau) 가 팔 사이에서 같은 칸인가 (거리) · K·δ 인가 (슬롯).
import csv, json, sys, numpy as np
from pathlib import Path
C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research"); CELL = 18.0
IDX = {r["video_id"]: r for r in csv.DictReader(open("data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
for tag, suf in (("release", ""), ("pv1", "_pv1"), ("ariel", "_ariel"), ("v11ft", "_v11ft")):
    print(f"== {tag}   arm: 예측 변위 (칸, 진행 방향) 슬롯별 평균 | 진실 변위 | plateau = 예측 최대   δ (칸/슬롯)")
    for d in (C / f"v3_p2{suf}", C / f"v3_p2s16_{tag}"):
        meta = json.load(open(d / "meta.json")); ids = meta["video_ids"]; LA = np.load(d / "loc_app.npy")
        X = np.stack([f(IDX[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(IDX[v]["px_y_by_sample"]) for v in ids])
        INF = np.stack([f(IDX[v]["in_frame_by_sample"]) for v in ids]) > 0.5
        for ai, A in enumerate(meta["arms"]):
            fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]; fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
            cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
            P, T = [], []
            for t in range(nf):
                g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
                ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
                ok = INF[:, fa] & INF[:, g0] & INF[:, g1] & np.isfinite(LA[:, ai, t, 0])
                dv = ct - cl; dn = np.linalg.norm(dv, axis=-1); u = dv / np.maximum(dn, 1e-6)[:, None]
                ap = LA[:, ai, t, [1, 0]] + 0.5
                pd = ((ap - cl) * u).sum(-1)
                m = ok & (dn >= 0.5)
                P.append(np.nanmean(pd[m])); T.append(np.nanmean(dn[m]))
            P, T = np.array(P), np.array(T)
            print(f"  {A['name']:7s} pred " + " ".join(f"{x:4.1f}" for x in P) + f"\n  {'':7s} true " + " ".join(f"{x:4.1f}" for x in T) + f"   plateau {P.max():.2f} @t{P.argmax()}  δ {T[1]-T[0]:.2f}")
