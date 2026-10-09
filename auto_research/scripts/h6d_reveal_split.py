#!/usr/bin/env python3
"""H6d — IntPhys1 이동+가림: 재등장 (보이는 분기) 튜블릿의 판별을 '문맥 끝에서 물체가 보였나' · 방향 · 거리 d 로 쪼갠다.
문맥 끝 가시 = pos 영상의 문맥 마지막 두 샘플 프레임 중 하나라도 가시 물체 수 > 0 (obj_vis.npz, 프레임별 가시 물체 수).
방향: imp 가 물체 보이는 프레임이 적으면 disappear (intphys1_direction_audit 와 같은 정의).  방법 p / copyz / copyh."""
import collections, csv, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; H6 = ROOT / "auto_research/exp_results/h6"; AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
parts = sorted(H6.glob("tubelet_l1_vith_r*.npz")); z = np.load(parts[0]); combos, vids = list(z["combos"]), list(z["video_ids"])
R = np.concatenate([np.load(f)["rows"] for f in parts])
L = {(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]), int(r[4])): r[5:9] for r in R}
byv = collections.defaultdict(list)
for k, m in L.items(): byv[k[0]].append((k, m))
ov = np.load(AUX / "obj_vis.npz"); V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
out = collections.defaultdict(lambda: collections.defaultdict(list))
for p in csv.DictReader((AUX / "pairs.csv").open()):
    if f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}' != "moving/occluded": continue
    a_, b_ = int(V[p["imp"]]["obj_visible_frames"]), int(V[p["pos"]]["obj_visible_frames"])
    dr = "disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal")
    vis = ov[p["pos"]]
    for (v, cb, s, C, j), mpos in byv[p["pos"]]:
        key = (p["imp"], cb, s, C, j)
        if key not in L or cb not in ("skip2_w16", "skip2_w32"): continue
        k_ = 2; e0 = s + k_ * C
        jev = int(np.floor(((int(p["first_div_strict"]) - 1) - e0) / (2 * k_)))
        if jev < 0 or j != jev: continue                                     # 재등장 튜블릿만
        ctxvis = "ctxend_visible" if (vis[max(0, e0 - 2)] > 0 or vis[max(0, e0 - 4)] > 0) else "ctxend_hidden"
        db = "d0-2" if j <= 2 else ("d3-5" if j <= 5 else "d6+")
        for m, nm in ((0, "p"), (2, "copyz"), (1, "copyh")):
            ok = float(L[key][m] - mpos[m] > 0)
            for g in (f"{ctxvis}|{dr}|{db}", f"{ctxvis}|{dr}|all", f"{ctxvis}|all|{db}", f"all|{dr}|{db}"):
                out[g][nm].append(ok)
res = {g: {m: [round(100 * np.mean(v), 1), len(v)] for m, v in d.items()} for g, d in sorted(out.items())}
json.dump(res, open(H6 / "h6d_reveal_split.json", "w"), indent=1)
for g, d in res.items():
    if d["p"][1] >= 20: print(f"{g:40s} p {d['p'][0]:5.1f}  copyz {d['copyz'][0]:5.1f}  copyh {d['copyh'][0]:5.1f}  n {d['p'][1]}")
