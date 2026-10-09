#!/usr/bin/env python3
"""먼 슬롯에서 p 의 '물체다운 칸' 은 어디에 있나 — 파라미터 없는 진행률.
  argmax 칸 a (템플릿 = h 진실 토큰), 마지막 본 칸 L, 진실 칸 T (칸 중심 좌표):
  progress = (a − L)·(T − L) / |T − L|²     0 = 마지막 본 자리, 1 = 진실,  0~1 = 뒤처짐 (lag)
  물체다움 = contrast (max cos − median cos). 기준 = 같은 창 t0~t2 의 contrast 10 번째 백분위 (물체가 확실히 있는 슬롯).
  |T − L| ≥ 3 칸인 슬롯만 (진행률이 정의되는 곳). C16_P32 · C32_P32 · C8_P32, 자유 운동, 층 L11 과 L6."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23")
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; scen = np.array(meta["scenario"]); pl = np.array(meta["plausible"]) == "1"
R, X, Y, OK = load_index(ids); out = {}
last = cell(tub_xy(X, Y, 30)).astype(float)
for C, P in ((16, 32), (32, 32), (8, 32)):
    loc = np.load(I / f"loc_C{C}_P{P}.npy"); S = P // 2
    con = loc[..., 2] - loc[..., 3]
    for L in (11, 6):
        ref = np.percentile(con[np.isin(scen, FREE) & pl][:, :3, L], 10)
        rows = []
        for t in range(S):
            f0 = 32 + 2 * t; T = cell(tub_xy(X, Y, f0)).astype(float)
            d = T - last; dd = (d ** 2).sum(1)
            ok = np.isin(scen, FREE) & pl & OK[:, f0] & OK[:, f0 + 1] & (np.sqrt(dd) >= 3)
            a = loc[:, t, L, :2][:, ::-1]
            prog = ((a - last) * d).sum(1) / np.maximum(dd, 1)
            obj = con[:, t, L] >= ref
            if ok.sum() < 30:
                continue
            sel = ok & obj
            q = np.percentile(prog[sel], [25, 50, 75]) if sel.sum() >= 20 else [np.nan] * 3
            rows.append(dict(t=t, n=int(ok.sum()), objlike=round(float(obj[ok].mean()), 3),
                             prog_q=[round(float(x), 2) for x in q],
                             frac_near_last=round(float((prog[sel] < 0.25).mean()), 3) if sel.sum() else None,
                             frac_lag=round(float(((prog[sel] >= 0.25) & (prog[sel] < 0.75)).mean()), 3) if sel.sum() else None,
                             frac_truth=round(float((prog[sel] >= 0.75).mean()), 3) if sel.sum() else None))
        out[f"C{C}_P{P}_L{L}"] = dict(ref_contrast=float(ref), rows=rows)
        print(f"== C{C}_P{P} L{L}  (물체다움 기준 contrast ≥ {ref:.3f})")
        for r in rows:
            print(f"  t{r['t']:2d} n{r['n']:4d} 물체다움 {r['objlike']:.2f} | 진행률 25/50/75% {r['prog_q']} | 마지막자리 {r['frac_near_last']} 뒤처짐 {r['frac_lag']} 진실 {r['frac_truth']}")
json.dump(out, open(Path(__file__).resolve().parents[2] / "auto_research/exp_results/h23/progress.json", "w"), indent=1)
