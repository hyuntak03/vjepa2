#!/usr/bin/env python3
"""IntPhys1_surrogate 결과 → 시트 Object_permanence 의 "IntPhys1 Surrogate" 5 칸 (2026-11-03). GPU 불필요.

열 순서 (시트 그대로): 물체 움직임 A, B · 물체 정지 A, B · overall.
  A = 사라짐 짝 (pos_a, imp_ab) · B = 나타남 짝 (pos_b, imp_ba). 물체 움직임 = motion ip1_throw, 정지 = flat_static. overall = 400 짝 전부.
판정: s(불가능) > s(가능) 정답, 같으면 0.5 (op_cells_vob3.py 와 같다). 결과 = cvpr/results/eval/testbed/intphys1_surrogate_<모델>/per_block.json
  /venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py <모델> ... [--tsv] [--n]
"""
import csv, json, math, os, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
IDX = f"{REPO}/data_csv/intphys1_surrogate/index.csv"
COLS = ["moving/A", "moving/B", "static/A", "static/B", "overall"]


def pairs():
    by = defaultdict(dict)
    for r in csv.DictReader(open(IDX)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    return [(v["pos"], v["imp"]) for v in by.values()]


def sel(cell):
    def f(r):
        if cell == "overall":
            return True
        m, ab = cell.split("/")
        return r["condition"] == m and r["pair_id"] == ab
    return f


def main():
    args = sys.argv[1:]
    models = [a for a in args if not a.startswith("--")]
    P = pairs(); sep = "\t" if "--tsv" in args else " | "
    print(sep.join(["model"] + [f"sur:{c}" for c in COLS]))
    for m in models:
        f = f"{REPO}/cvpr/results/eval/testbed/intphys1_surrogate_{m}/per_block.json"
        if not os.path.exists(f):
            print(sep.join([m] + ["-"] * len(COLS))); continue
        pv = json.load(open(f))["per_video_surprise"]
        row = []
        for c in COLS:
            s = [(p["video_id"], i["video_id"]) for p, i in P if sel(c)(p)]
            acc = [1.0 if pv[b] > pv[a] else (0.5 if pv[b] == pv[a] else 0.0) for a, b in s]
            row.append(f"{100 * sum(acc) / len(acc):.1f}")
        print(sep.join([m] + row))
    if "--n" in args:
        n = [sum(1 for p, i in P if sel(c)(p)) for c in COLS]
        print(sep.join(["n pairs"] + [str(x) for x in n]))
        print(sep.join(["95% CI (±)"] + [f"±{100 * 1.96 * math.sqrt(0.25 / x):.1f}" for x in n]))


if __name__ == "__main__":
    main()
