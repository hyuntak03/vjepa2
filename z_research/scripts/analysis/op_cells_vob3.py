#!/usr/bin/env python3
"""VOBench_v3 결과 → 시트 Object_permanence 의 D:W 열 (2026-11-02). GPU 불필요.

열 순서 (시트 그대로): [가림막 panel | mover] × [primitive | realistic] × [물체 움직임 A, B · 물체 정지 A, B · overall] = 20 칸.
  panel = 서 있는 판 (시트 "VOBench late" 묶음) · mover = 지나가는 van / 상자 (시트 "moving occluder" 묶음).
  A = 사라짐 짝 (pos_a, imp_ab) · B = 나타남 짝 (pos_b, imp_ba). 물체 정지 = motion flat_static. overall = 그 objset 짝 전부 (짝 가중).
판정: s(불가능) > s(가능) 정답, 같으면 0.5. 결과 = cvpr/results/eval/testbed/vobench_v3_<모델>[<접미>]/per_block.json
  /venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py <모델> ... [--suffix _bs4] [--tsv] [--n]
"""
import csv, json, math, os, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
IDX = f"{REPO}/data_csv/vobench_v3/index.csv"
COLS = [(occ, obj, cell) for occ in ("panel", "mover") for obj in ("primitive", "realistic")
        for cell in ("moving/A", "moving/B", "static/A", "static/B", "overall")]


def pairs():
    by = defaultdict(dict)
    for r in csv.DictReader(open(IDX)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    return [(v["pos"], v["imp"]) for v in by.values()]


def sel(occ, obj, cell):
    def f(r):
        if r["occluder_kind"] != occ or r["objset"] != obj:
            return False
        if cell == "overall":
            return True
        m, ab = cell.split("/")
        return ("static" if r["motion"] == "flat_static" else "moving") == m and r["pair_id"] == ab
    return f


def main():
    args = sys.argv[1:]
    suf = args[args.index("--suffix") + 1] if "--suffix" in args else ""
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--suffix")]
    P = pairs(); sep = "\t" if "--tsv" in args else " | "
    print(sep.join(["model"] + [f"{o}:{b[:4]}:{c}" for o, b, c in COLS]))
    for m in models:
        f = f"{REPO}/cvpr/results/eval/testbed/vobench_v3_{m}{suf}/per_block.json"
        if not os.path.exists(f):
            print(sep.join([m] + ["-"] * len(COLS))); continue
        pv = json.load(open(f))["per_video_surprise"]
        row = []
        for c in COLS:
            s = [(p["video_id"], i["video_id"]) for p, i in P if sel(*c)(p)]
            acc = [1.0 if pv[b] > pv[a] else (0.5 if pv[b] == pv[a] else 0.0) for a, b in s]
            row.append(f"{100 * sum(acc) / len(acc):.1f}")
        print(sep.join([m] + row))
    if "--n" in args:
        n = [sum(1 for p, i in P if sel(*c)(p)) for c in COLS]
        print(sep.join(["n pairs"] + [str(x) for x in n]))
        print(sep.join(["95% CI (±)"] + [f"±{100 * 1.96 * math.sqrt(0.25 / x):.1f}" for x in n]))


if __name__ == "__main__":
    main()
