#!/usr/bin/env python3
"""H6e 분석 — 인과 표적 vs 표준 표적으로 IntPhys 1 채점 (Garrido 규칙 그대로) + 튜블릿 단위 사건 전/후.

Garrido 규칙 (PROTOCOLS.md): 창 surprise = 튜블릿 평균 L1 → 시작점마다 C 최솟값 (Filtered) → 시작점 평균 (AvgSurprise) →
matched pair 비교 → property (O1/O2/O3) macro.  combo 마다 따로 (skip2_w32 가 88.89 칸).
검증: 표준 표적 (p_full) 으로 skip2_w32 가 88.89 를 재현해야 한다.
"""
import collections, csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; H6 = ROOT / "auto_research/exp_results/h6"; AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
parts = sorted(H6.glob("causal_l1_vith_r*.npz")); z = np.load(parts[0]); combos, vids = list(z["combos"]), list(z["video_ids"])
R = np.concatenate([np.load(f)["rows"] for f in parts]); cols = list(z["cols"])
MET = ("p_causal", "copyz_causal", "p_full", "copyz_full")
T = collections.defaultdict(dict)                                        # (vid, combo, start, C) -> {j: vals}
for r in R:
    T[(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]))][int(r[4])] = r[5:9]
pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
out = {}
# 1) Garrido 점수
for cb in combos:
    for mi, m in enumerate(MET):
        vs = {}
        for (v, c, s, C), d in T.items():
            if c != cb: continue
            vs.setdefault(v, {}).setdefault(s, []).append(np.mean([x[mi] for x in d.values()]))
        score = {v: np.mean([min(cs) for cs in st.values()]) for v, st in vs.items()}
        by = collections.defaultdict(list)
        for p in pairs:
            if p["pos"] in score and p["imp"] in score:
                a, b = score[p["imp"]], score[p["pos"]]
                by[p["principle"][:2]].append(1.0 if a > b else (0.5 if a == b else 0.0))
                by[f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'].append(1.0 if a > b else (0.5 if a == b else 0.0))
        macro = np.mean([np.mean(by[k]) for k in ("O1", "O2", "O3")]) * 100
        out[f"garrido|{cb}|{m}"] = {"macro": round(macro, 2), **{k: round(100 * np.mean(v), 1) for k, v in sorted(by.items())}}
        print(f"{cb:10s} {m:13s} macro {macro:6.2f}  " + "  ".join(f"{k} {100*np.mean(v):5.1f}" for k, v in sorted(by.items())))
# 2) 튜블릿 단위 사건 전/분기/후 (보이는 분기)
acc = collections.defaultdict(list)
for p in pairs:
    g = f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'
    for (v, cb, s, C), d in T.items():
        if v != p["pos"] or cb not in ("skip2_w16", "skip2_w32"): continue
        di = T.get((p["imp"], cb, s, C))
        if di is None: continue
        e0 = s + 2 * C; jev = int(np.floor(((int(p["first_div_strict"]) - 1) - e0) / 4)); ntub = len(d)
        if not (0 <= jev < ntub): continue
        for j, vp in d.items():
            e = j - jev; eb = "pre" if e < 0 else ("e0" if e == 0 else "post")
            for mi, m in enumerate(MET):
                dl = di[j][mi] - vp[mi]
                acc[(g, eb, m)].append(1.0 if dl > 0 else (0.5 if dl == 0 else 0.0))
for k in sorted(acc):
    out[f"tub|{k[0]}|{k[1]}|{k[2]}"] = [round(100 * np.mean(acc[k]), 1), len(acc[k])]
print("\n튜블릿 (보이는 분기 기준): 그룹 · 구간 · 방법 → 정답률")
for g in sorted({k[0] for k in acc}):
    for eb in ("pre", "e0", "post"):
        print(f"  {g:18s} {eb:5s} " + "  ".join(f"{m} {100*np.mean(acc[(g, eb, m)]):5.1f}" for m in MET if (g, eb, m) in acc))
json.dump(out, open(H6 / "h6e_causal.json", "w"), indent=1)
