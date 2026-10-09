#!/usr/bin/env python3
"""자 없는 위치 읽기의 귀무 적중: argmax 칸이 **같은 시나리오 다른 clip** 의 진실 칸 1 칸 안에 들 확률 (위치 사전확률).
적중 − 귀무 = 그 clip 의 물체를 짚은 몫.  C16_P32 · C32_P32, 층별 · 슬롯별, 자유 운동 6 법칙."""
import json, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23")
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; scen = np.array(meta["scenario"]); pl = np.array(meta["plausible"]) == "1"
R, X, Y, OK = load_index(ids)
rng = np.random.default_rng(0); out = {}
for C, P in ((16, 32), (32, 32)):
    loc = np.load(I / f"loc_C{C}_P{P}.npy"); S = P // 2
    m0 = np.isin(scen, FREE) & pl
    tab = {}
    for t in range(S):
        f0 = 32 + 2 * t; fc = cell(tub_xy(X, Y, f0)); ok = m0 & OK[:, f0] & OK[:, f0 + 1]
        perm = np.arange(len(ids))
        for s in FREE:
            ix = np.where(scen == s)[0]; perm[ix] = rng.permutation(ix)
        am = loc[:, t, :, :2][..., ::-1]                             # (n,12,2) x,y
        hit = (np.abs(am - fc[:, None]).max(-1) <= 1)[ok].mean(0)
        null = (np.abs(am - fc[perm][:, None]).max(-1) <= 1)[ok].mean(0)
        con = (loc[:, t, :, 2] - loc[:, t, :, 3])[ok].mean(0)
        tab[t] = dict(hit=hit.round(3).tolist(), null=null.round(3).tolist(), contrast=con.round(3).tolist(), n=int(ok.sum()))
    out[f"C{C}_P{P}"] = tab
    print(f"== C{C}_P{P}  (행: 슬롯 / 열: L0 L2 L4 L6 L8 L10 L11 — hit / null)")
    for t in range(S):
        v = tab[t]
        print(f"t{t:2d} n{v['n']:4d} " + "  ".join(f"{v['hit'][l]:.2f}/{v['null'][l]:.2f}" for l in (0, 2, 4, 6, 8, 10, 11)))
json.dump(out, open(Path(__file__).resolve().parents[2] / "auto_research/exp_results/h23/null_hit.json", "w"))
