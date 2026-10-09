#!/usr/bin/env python3
"""H9b 분석 — IntPhys 1 Garrido 규칙으로 변형별 채점 (시작점마다 C 최솟값 → 시작점 평균 → matched pair → O1/O2/O3 macro) + 운동×가림 + 사라짐 방향.
검증: clean 창 평균 = per_window.json, clean skip2_w32 macro = 88.89."""
import collections, csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; H9 = ROOT / "auto_research/exp_results/h9"; AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
BENCH = ROOT / "z_research/Benchmarks/exp_results"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
parts = sorted(H9.glob("ip1_intervene_r*.npz")); z = np.load(parts[0]); combos, vids, VAR = list(z["combos"]), list(z["video_ids"]), list(z["variants"])
R = np.concatenate([np.load(f)["rows"] for f in parts])
T = collections.defaultdict(list)
for r in R:
    T[(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]), int(r[5]))].append(r[6])
pw = {}
for w in (16, 32):
    for vid, rows in json.load(open(BENCH / f"intphys1_sliding__intphys1_dev_vith_w{w}/per_window.json"))["windows"].items():
        for cb, s, C, sur in rows: pw[(vid, cb, int(s), int(C))] = sur
d = [abs(np.mean(v) - pw[k[:4]]) for k, v in T.items() if k[4] == 0 and k[:4] in pw]
print(f"검증 clean 창 평균 vs per_window: max {max(d):.2e} median {np.median(d):.2e}  (n={len(d)})")
V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
pairs = list(csv.DictReader((AUX / "pairs.csv").open())); out = {}
for cb in combos:
    for vi, vn in enumerate(VAR):
        vs = {}
        for (v, c, s, C, x), l in T.items():
            if c == cb and x == vi: vs.setdefault(v, {}).setdefault(s, []).append(np.mean(l))
        sc = {v: np.mean([min(cs) for cs in st.values()]) for v, st in vs.items()}
        by = collections.defaultdict(list)
        for p in pairs:
            ok = 1.0 if sc[p["imp"]] > sc[p["pos"]] else (0.5 if sc[p["imp"]] == sc[p["pos"]] else 0.0)
            by[p["principle"][:2]].append(ok); by[f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'].append(ok)
            if p["principle"][:2] in ("O1", "O3"):
                a_, b_ = int(V[p["imp"]]["obj_visible_frames"]), int(V[p["pos"]]["obj_visible_frames"])
                by["dir:" + ("disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal"))].append(ok)
        macro = 100 * np.mean([np.mean(by[k]) for k in ("O1", "O2", "O3")])
        out[f"{cb}|{vn}"] = {"macro": round(macro, 2), **{k: round(100 * np.mean(v), 1) for k, v in sorted(by.items())}}
        print(f"{cb:10s} {vn:8s} macro {macro:6.2f}  " + "  ".join(f"{k} {100*np.mean(v):5.1f}" for k, v in sorted(by.items())))
json.dump(out, open(H9 / "h9b_intphys1.json", "w"), indent=1)
