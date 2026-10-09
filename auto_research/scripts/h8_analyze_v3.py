#!/usr/bin/env python3
"""H8 분석 — knockout 명세마다 슬롯별 물체다움 · 진실 적중 · 진행률 (파라미터 없음). 기준 = clean_null (bool mask 경로).
물체다움 기준 = clean_null 의 k ≤ 2 contrast 10 백분위 (모든 명세 공통)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell
I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h8")
OUT = Path(__file__).resolve().parents[2] / "auto_research/exp_results/h8"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; SP = meta["specs"]; K = meta["K"]
H = np.load(I / "h8.npy")
R, X, Y, OK = load_index(ids)
last = cell(tub_xy(X, Y, 30)).astype(float)
con = H[..., 2] - H[..., 3]
ref = np.nanpercentile(con[:, SP.index("clean_null"), :3], 10)
res = {}; lines = [f"물체다움 기준 {ref:.3f}.  열 = 슬롯 " + " ".join(f"{k:>4d}" for k in range(0, K, 2))]
for si, sp in enumerate(SP):
    ob, ht, pg = [], [], []
    for k in range(K):
        f0 = 32 + 2 * k; T = cell(tub_xy(X, Y, f0)).astype(float); d = T - last; dd = (d ** 2).sum(1)
        ok = OK[:, f0] & OK[:, f0 + 1]
        a = H[:, si, k, :2][:, ::-1]; o = con[:, si, k] >= ref
        ob.append(o[ok].mean()); ht.append((np.abs(a - T).max(1) <= 1)[ok].mean())
        mv = ok & o & (np.sqrt(dd) >= 3)
        pg.append(np.median((((a - last) * d).sum(1) / np.maximum(dd, 1))[mv]) if mv.sum() >= 20 else np.nan)
    res[sp] = dict(objlike=np.round(ob, 3).tolist(), hit=np.round(ht, 3).tolist(), prog=np.round(pg, 2).tolist())
    lines.append(f"{sp:11s} 물체다움 " + " ".join(f"{ob[k]:.2f}" for k in range(0, K, 2)))
    lines.append(f"{'':11s} 진실적중 " + " ".join(f"{ht[k]:.2f}" for k in range(0, K, 2)))
    lines.append(f"{'':11s} 진행률   " + " ".join("  – " if np.isnan(pg[k]) else f"{pg[k]:.2f}" for k in range(0, K, 2)))
json.dump(res, open(OUT / "h8_v3.json", "w"), indent=1)
print("\n".join(lines)); (OUT / "h8_summary.txt").write_text("\n".join(lines))
