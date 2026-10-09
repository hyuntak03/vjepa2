#!/usr/bin/env python3
"""H5 분석 — 되먹임 팔별로 물체가 얼마나 오래 · 어디에 남나 (파라미터 없음).
팔: os (한 번에) · single · ar_p (자기 예측 되먹임) · ar_z (참 관측 되먹임, 상한) · ar_zA (참 관측 + 어댑터) · ar_h (target 참 미래 + 어댑터).
물체다움 기준 = os 팔 j ≤ 1 의 contrast 10 백분위 (모든 팔에 같은 기준). 진행률은 H4 와 같은 식 (거리 ≥ 3 칸)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h5")
OUT = Path(__file__).resolve().parents[2] / "auto_research/exp_results/h5"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; arms = meta["arms"]
H = np.load(I / "h5.npy")                                   # (n, 2C, 8j, 6arm, 9)
R, X, Y, OK = load_index(ids)
scen = np.array([r["scenario"] for r in R]); pl = np.array([r["plausible"] == "1" for r in R]); free = np.isin(scen, FREE) & pl
last = cell(tub_xy(X, Y, 30)).astype(float)
res = {}; lines = []
for ci, C in enumerate(meta["Cs"]):
    con = H[:, ci, :, :, 2] - H[:, ci, :, :, 3]
    ref = np.nanpercentile(con[free][:, :2, arms.index("os")], 10)
    lines.append(f"== C{C}  (물체다움 기준 {ref:.3f});  행 = 팔, 열 = j 0..7")
    for ai, arm in enumerate(arms):
        ob, ht, pg, l1 = [], [], [], []
        for j in range(meta["J"]):
            f0 = 32 + 2 * j; Tc = cell(tub_xy(X, Y, f0)).astype(float); d = Tc - last; dd = (d ** 2).sum(1)
            ok = free & OK[:, f0] & OK[:, f0 + 1]
            a = H[:, ci, j, ai, :2][:, ::-1]; o = con[:, j, ai] >= ref
            ob.append(o[ok].mean()); ht.append((np.abs(a - Tc).max(1) <= 1)[ok].mean())
            mv = ok & o & (np.sqrt(dd) >= 3)
            pg.append(np.median((((a - last) * d).sum(1) / np.maximum(dd, 1))[mv]) if mv.sum() >= 20 else np.nan)
            l1.append(np.nanmean(H[ok, ci, j, ai, 8]))
        res[f"C{C}|{arm}"] = dict(objlike=np.round(ob, 3).tolist(), hit=np.round(ht, 3).tolist(), prog=np.round(pg, 2).tolist(), l1=np.round(l1, 4).tolist())
        lines.append(f"  {arm:7s} 물체다움 " + " ".join(f"{x:.2f}" for x in ob) + " | 진실적중 " + " ".join(f"{x:.2f}" for x in ht)
                     + " | 진행률 " + " ".join("  – " if np.isnan(x) else f"{x:.2f}" for x in pg) + " | L1 " + " ".join(f"{x:.3f}" for x in l1))
json.dump(res, open(OUT / "h5_v3.json", "w"), indent=1)
print("\n".join(lines)); (OUT / "h5_summary.txt").write_text("\n".join(lines))
