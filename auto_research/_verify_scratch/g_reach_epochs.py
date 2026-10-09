# 묶음 G (B1): 같은 학습 run 안에서 epoch 에 따라 reach (외형 템플릿 예측 변위 plateau, 칸) 가 느는가. + t10 적중−귀무는 p2_analyze 로 따로.
import csv, json, numpy as np
from pathlib import Path
C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research"); CELL = 18.0
IDX = {r["video_id"]: r for r in csv.DictReader(open("data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
runs = [("release", "v3_p2c_release"), ("ariel ep5", "v3_p2c_ariel5"), ("ariel ep19", "v3_p2c_ariel19"), ("ariel ep30", "v3_p2c_ariel30"), ("ariel ep43", "v3_p2c_ariel"),
        ("pv1 e1", "v3_p2c_pv1e1"), ("pv1 e5", "v3_p2c_pv1e5"), ("pv1 e15", "v3_p2c_pv1"), ("v11ft e1", "v3_p2c_v11e1"), ("v11ft e5", "v3_p2c_v11e5"), ("v11ft e10", "v3_p2c_v11ft")]
out = {}
for name, d in runs:
    d = C / d; meta = json.load(open(d / "meta.json")); ids = meta["video_ids"]; LA = np.load(d / "loc_app.npy"); LT = np.load(d / "loc_tru.npy")
    X = np.stack([f(IDX[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(IDX[v]["px_y_by_sample"]) for v in ids])
    INF = np.stack([f(IDX[v]["in_frame_by_sample"]) for v in ids]) > 0.5
    row = {}
    for ai, A in enumerate(meta["arms"]):
        fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]; fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
        cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
        P = []
        for t in range(nf):
            g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
            ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
            ok = INF[:, fa] & INF[:, g0] & INF[:, g1] & np.isfinite(LA[:, ai, t, 0])
            dv = ct - cl; dn = np.linalg.norm(dv, axis=-1); u = dv / np.maximum(dn, 1e-6)[:, None]
            pd = ((LA[:, ai, t, [1, 0]] + 0.5 - cl) * u).sum(-1); m = ok & (dn >= 0.5)
            P.append(float(np.nanmean(pd[m])))
        row[A["name"]] = dict(plateau=round(max(P), 2), at=int(np.argmax(P)), last=round(P[-1], 2), nf=nf)
    out[name] = row
    print(f"{name:11s} " + "  ".join(f"{k} {v['plateau']:.2f}@t{v['at']}{'*' if v['at'] == v['nf'] - 1 else ''}" for k, v in row.items()))
json.dump(out, open("auto_research/exp_results/p2/g_reach_epochs.json", "w"), indent=1)
