#!/usr/bin/env python3
"""v11 surprise_c16t32 결과를 조건 × 위반 (vanish/shape/color) × 방향 (A/B) 으로 — per_block.json 의 per_video_surprise 와 index.csv 에서 다시 계산 (2026-10-07).
    python auto_research/scripts/v11_cond_viol_table.py --runs surprise_c16t32__v11_vith surprise_c16t32__v11_dinof_highres surprise_c16t32__v11_dinof_highres_copy
matched 쌍: (pos_a, imp_ab) = 문맥 A · (pos_b, imp_ba) = 문맥 B. 정답 = surprise(imp) > surprise(pos). 동점은 0.5.
"""
import argparse, csv, json, collections
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--root", default="z_research/IntPhysGenV11/exp_results"); ap.add_argument("--index", default="data_csv/intphysgen_v11/index.csv")
a = ap.parse_args()
idx = list(csv.DictReader(open(a.index))); byblk = collections.defaultdict(dict)
for r in idx: byblk[r["source_block"]][r["variant"]] = r
COND = ["static_visible", "static_occlusion", "moving_visible_flat", "moving_occlusion_flat", "moving_visible", "moving_occlusion"]; VIOL = ["vanish", "shape", "color"]
def acc_table(pv):
    cell = collections.defaultdict(list); d = collections.defaultdict(list)
    for sb, v in byblk.items():
        if not all(k in v for k in ("pos_a", "pos_b", "imp_ab", "imp_ba")): continue
        c, vt = v["pos_a"]["condition"], v["pos_a"]["violation_type"]
        for pos, imp, dirn in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
            sp, si = pv.get(v[pos]["video_id"]), pv.get(v[imp]["video_id"])
            if sp is None or si is None: continue
            ok = 1.0 if si > sp else (0.5 if si == sp else 0.0); cell[(c, vt)].append(ok); d[(c, vt, dirn)].append(ok)
    return cell, d
res = {}
for run in a.runs:
    P = json.load(open(f"{a.root}/{run}/per_block.json")); pv = P["per_video_surprise"]; pv = pv.get("all", pv) if isinstance(next(iter(pv.values())), dict) else pv
    cell, d = acc_table(pv); res[run] = (cell, d)
    allv = [x for c in cell.values() for x in c]; print(f"\n=== {run}  overall {100*np.mean(allv):.2f} (n_pair {len(allv)})")
    print(f"{'condition':24s}" + "".join(f"{vt:>16s}" for vt in VIOL) + "     (A / B 방향)")
    for c in COND:
        row = "".join(f"{100*np.mean(cell[(c,vt)]):8.1f} ({len(cell[(c,vt)]):3d})" for vt in VIOL)
        dirs = "  ".join(f"{vt[:2]} {100*np.mean(d[(c,vt,'A')]):.0f}/{100*np.mean(d[(c,vt,'B')]):.0f}" for vt in VIOL)
        print(f"{c:24s}{row}     {dirs}")
if len(a.runs) >= 2:
    print("\n=== 차이 (두 번째 − 첫 번째 run) 조건 × 위반")
    c0, _ = res[a.runs[0]]; c1, _ = res[a.runs[1]]
    for c in COND: print(f"{c:24s}" + "".join(f"{100*(np.mean(c1[(c,vt)])-np.mean(c0[(c,vt)])):+8.1f}" for vt in VIOL))
