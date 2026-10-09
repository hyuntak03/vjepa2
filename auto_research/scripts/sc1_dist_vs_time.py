#!/usr/bin/env python3
"""E — v3 에서 거리 (칸) 와 경과 시간 (raw 프레임) 을 가른다 (SC1 단서 1). python sc1_dist_vs_time.py release= pv1=_pv1 ariel=_ariel

같은 clip 을 stride 1/2/4 · 분할 16/32 로 물은 7 팔을 합치면 (거리, t_since) 가 팔 · 속도 분위에 따라 갈린다.
(1) 거리 구간 (1 칸 폭) 안에서 t_since 삼분위별 hit−null_ex 평균 — 거리를 고정하면 시간이 효과가 있나.
(2) t_since 구간 (8 raw 프레임 폭) 안에서 거리 삼분위별 — 시간을 고정하면 거리가 효과가 있나.
(3) 회귀 hit−null_ex ~ 1 + dist + t_since/8 + slot (궤적 군집 bootstrap 300) 과 부분 R² (각 변수를 뺐을 때 R² 감소).
토큰 = (clip, 팔, 슬롯). 궤적 군집 CI.
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc1_horizon_units as U

OUT = U.OUT; res = {}
for arg in sys.argv[1:]:
    tag, suf = arg.split("=", 1)
    a1, tr = U.load(U.C / f"v3_p2{suf}"); a2, _ = U.load(U.C / f"v3_p2s16_{tag}")
    arms = a1 + a2; n = len(tr)
    Y, D, T, S, CL = [], [], [], [], []
    for A in arms:
        HN, DI, TS = A["HN"], A["DI"], A["TS"]
        ok = np.isfinite(HN) & np.isfinite(DI)
        Y.append(HN[ok]); D.append(DI[ok]); T.append(np.broadcast_to(TS[None], HN.shape)[ok]); S.append(np.broadcast_to(np.arange(HN.shape[1])[None], HN.shape)[ok])
        CL.append(np.broadcast_to(tr[:, None], HN.shape)[ok])
    y, d, t, s, cl = map(np.concatenate, (Y, D, T, S, CL)); t8 = t / 8.0
    rec = dict(n_tokens=int(len(y)))
    # (1) 거리 고정 → 시간 효과
    rows = []
    for lo in range(1, 8):
        m = (d >= lo) & (d < lo + 1)
        if m.sum() < 300: continue
        q = np.quantile(t[m], [1 / 3, 2 / 3]); b = np.digitize(t[m], q)
        rows.append(dict(dist=[lo, lo + 1], n=int(m.sum()), t_terciles_mean=[round(float(t[m][b == j].mean()), 1) for j in range(3)],
                         hitnull_by_t_tercile=[round(float(y[m][b == j].mean()), 3) for j in range(3)]))
    rec["fixed_dist_vary_time"] = rows
    rows = []
    for lo in range(0, 40, 8):
        m = (t >= lo) & (t < lo + 8)
        if m.sum() < 300: continue
        q = np.quantile(d[m], [1 / 3, 2 / 3]); b = np.digitize(d[m], q)
        rows.append(dict(t_since=[lo, lo + 8], n=int(m.sum()), d_terciles_mean=[round(float(d[m][b == j].mean()), 2) for j in range(3)],
                         hitnull_by_d_tercile=[round(float(y[m][b == j].mean()), 3) for j in range(3)]))
    rec["fixed_time_vary_dist"] = rows
    # (3) 회귀 + 부분 R²
    X = np.stack([np.ones_like(d), d, t8, s.astype(float)], 1); fit = lambda X_, y_: np.linalg.lstsq(X_, y_, rcond=None)[0]
    r2 = lambda X_, y_: 1 - ((y_ - X_ @ fit(X_, y_)) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
    full = r2(X, y); part = {nm: round(float(full - r2(np.delete(X, j, 1), y)), 4) for j, nm in ((1, "dist"), (2, "t_since8"), (3, "slot"))}
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        ss = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(fit(X[ss], y[ss]))
    bs = np.array(bs); b0 = fit(X, y)
    rec["regression"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_cell", "per_8raw", "per_slot"))}
    rec["R2_full"] = round(float(full), 4); rec["partial_R2_drop"] = part
    rec["corr_dist_time"] = round(float(np.corrcoef(d, t)[0, 1]), 3)
    res[tag] = rec
    print(f"=== {tag}  n {len(y)}  corr(dist, t) {rec['corr_dist_time']}  R² {rec['R2_full']}  partial drop {part}")
    print("  회귀:", json.dumps(rec["regression"]))
    for r in rec["fixed_dist_vary_time"]: print(f"  dist {r['dist']} n{r['n']:6d}: t {r['t_terciles_mean']} → hit−null {r['hitnull_by_t_tercile']}")
    for r in rec["fixed_time_vary_dist"]: print(f"  t {r['t_since']} n{r['n']:6d}: d {r['d_terciles_mean']} → hit−null {r['hitnull_by_d_tercile']}")
json.dump(res, open(OUT / f"sc1_dist_vs_time{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
