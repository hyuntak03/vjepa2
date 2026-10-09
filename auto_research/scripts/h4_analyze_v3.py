#!/usr/bin/env python3
"""H4 분석 — 단일 질의 · joint 질의 · 위치 이동 (s = 4, 8) 에서 p 의 물체가 어디에 있나 (파라미터 없음).

(clip, C, k, mode) 마다 loc_tru (템플릿 = 진실 h 토큰) · loc_app (템플릿 = 마지막으로 본 물체 h 토큰):
  hit_truth   argmax 가 진실 칸 1 칸 안
  objlike     contrast (max − median cos) 가 같은 모드 · 창의 k ≤ 2 의 10 번째 백분위 이상
  progress    (argmax − 마지막 칸)·(진실 − 마지막) / |진실 − 마지막|²    (|진실 − 마지막| ≥ 3 칸일 때)
속도 3 분위 (문맥 끝 속도) 로 나눠 "물체가 사라지는 k" 가 속도에 따라 움직이나 (거리) / 그대로인가 (시간) 를 본다.
위치 이동: 내용은 같고 RoPE 위치 인덱스만 s 튜블릿 민 것 — 사라지는 k 가 s 만큼 당겨지면 절대 위치 효과.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE

I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h4")
OUT = Path(__file__).resolve().parents[2] / "auto_research/exp_results/h4"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; Cs, K = meta["Cs"], meta["K"]
H = np.load(I / "h4.npy")                                   # (n, 4C, K, 4 mode, 9)
R, X, Y, OK = load_index(ids)
scen = np.array([r["scenario"] for r in R]); pl = np.array([r["plausible"] == "1" for r in R])
last_xy = tub_xy(X, Y, 30); last = cell(last_xy).astype(float)
v_end = np.linalg.norm(last_xy - tub_xy(X, Y, 28), axis=1)
free = np.isin(scen, FREE) & pl
q1, q2 = np.percentile(v_end[free], [33, 67]); spd = np.digitize(v_end, [q1, q2])
res = {}; lines = []
for ci, C in enumerate(Cs):
    for m, mode in enumerate(meta["modes"]):
        if np.isnan(H[:, ci, :, m, 0]).all():
            continue
        con = H[:, ci, :, m, 2] - H[:, ci, :, m, 3]
        ref = np.nanpercentile(con[free][:, :3], 10)
        tab = {}
        for sb, nm in enumerate(("slow", "mid", "fast")):
            row = []
            for k in range(K):
                f0 = 32 + 2 * k; T = cell(tub_xy(X, Y, f0)).astype(float); d = T - last; dd = (d ** 2).sum(1)
                ok = free & (spd == sb) & OK[:, f0] & OK[:, f0 + 1]
                a = H[:, ci, k, m, :2][:, ::-1]
                hit = (np.abs(a - T).max(1) <= 1)
                obj = con[:, k] >= ref
                prog = ((a - last) * d).sum(1) / np.maximum(dd, 1)
                mv = ok & (np.sqrt(dd) >= 3) & obj
                row.append(dict(k=k, n=int(ok.sum()), hit=round(float(hit[ok].mean()), 3), objlike=round(float(obj[ok].mean()), 3),
                                dist_cells=round(float(np.sqrt(dd[ok]).mean()), 2),
                                prog_med=round(float(np.median(prog[mv])), 2) if mv.sum() >= 20 else None))
            tab[nm] = row
        res[f"C{C}|{mode}"] = tab
        lines.append(f"== C{C} {mode}  (물체다움 기준 {ref:.3f})")
        for nm in ("slow", "mid", "fast"):
            lines.append(f"  {nm:4s} objlike " + " ".join(f"{r['objlike']:.2f}" for r in tab[nm]))
            lines.append(f"  {nm:4s} dist칸  " + " ".join(f"{r['dist_cells']:.1f}" for r in tab[nm]))
            lines.append(f"  {nm:4s} prog    " + " ".join("  – " if r["prog_med"] is None else f"{r['prog_med']:.2f}" for r in tab[nm]))
json.dump(res, open(OUT / "h4_v3.json", "w"), indent=1)
print("\n".join(lines)); (OUT / "h4_summary.txt").write_text("\n".join(lines))
