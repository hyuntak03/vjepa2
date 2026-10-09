#!/usr/bin/env python3
"""DINO-Foresight × IntPhys 1 dev (Garrido sliding) 결과 표 — summary.json 에서 다시 계산 (2026-10-02).
    python auto_research/scripts/dinof_intphys1_table.py [--models dinof_highres,dinof_highres_copy,dinof_lowres]
"""
import argparse, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="z_research/IntPhys/exp_results"); a = ap.parse_args()
rows = []
for d in sorted(glob.glob(f"{a.root}/intphys1_sliding__intphys1_dev_dinof_*")):
    f = f"{d}/summary.json"
    if "smoke" in d or not os.path.exists(f): continue
    name = os.path.basename(d).replace("intphys1_sliding__intphys1_dev_", "")
    for cell, v in json.load(open(f))["surprise"].items():
        o = v["overall"]; bt = v["by_block_type"]; props = {t: 100 * bt[t]["block_pairwise"] for t in ("O1", "O2", "O3")}
        rows.append((name, cell.replace("/avg", ""), 100 * o["block_pairwise"], o["n_ties"], props, sum(props.values()) / 3))
print(f"{'model':24s} {'cell':12s} {'pair acc':>9s} {'ties':>5s} {'O1':>6s} {'O2':>6s} {'O3':>6s} {'macro':>6s}")
for name, cell, acc, ties, p, mac in rows:
    print(f"{name:24s} {cell:12s} {acc:9.2f} {ties:5d} {p['O1']:6.1f} {p['O2']:6.1f} {p['O3']:6.1f} {mac:6.2f}")
best = {}
for name, cell, acc, ties, p, mac in rows:
    if name not in best or mac > best[name][2]: best[name] = (cell, acc, mac, p)
print("\n모델별 최고 칸 (property macro 기준, A.8 탐색 → descriptive):")
for n, (c, acc, mac, p) in best.items(): print(f"  {n:24s} {c:12s} macro {mac:.2f}  (O1 {p['O1']:.1f} / O2 {p['O2']:.1f} / O3 {p['O3']:.1f})")
