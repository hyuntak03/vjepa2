#!/usr/bin/env python3
"""VOBench_v3 visible · early · mid (2026-11-09) 칸 계산. GPU 불필요. op_cells_vob3.py 와 같은 판정 (s(불가능) > s(가능) 정답, 같으면 0.5).

행 = 모델, 열 = [visible (가림막 없음) | early · 판 | early · 지나감 | mid · 판 | mid · 지나감] × [움직임 A · B, 정지 A · B, 전체].
A = 사라짐 짝 (pos_a, imp_ab), B = 나타남 짝 (pos_b, imp_ba). 이음 48 은 세 세트 모두 보이는 곳에서 일어난다.
  /venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3_vem.py <모델> ... [--objset primitive|realistic] [--by-k] [--n] [--late]
  --late: 같은 모델의 VOBench_v3 (late) 칸을 같은 꼴로 뒤에 붙인다 (판 · 지나감).
"""
import csv, json, math, os, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = f"{REPO}/cvpr/results/eval/testbed"
CELLS = ("moving/A", "moving/B", "static/A", "static/B", "overall")


def pairs(idx):
    by = defaultdict(dict)
    for r in csv.DictReader(open(idx)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    return [(v["pos"], v["imp"]) for v in by.values()]


def keep(r, occ, kind, cell, objset, k):
    if r["occluder_kind"] != kind or (occ and not r["occlusion"].startswith(occ)) or (objset and r["objset"] != objset) or (k and r["k"] != k):
        return False
    if cell == "overall":
        return True
    m, ab = cell.split("/")
    return ("static" if r["motion"] == "flat_static" else "moving") == m and r["pair_id"] == ab


def score(pv, P, f):
    s = [(p["video_id"], i["video_id"]) for p, i in P if f(p)]
    if not s:
        return None, 0
    acc = [1.0 if pv[b] > pv[a] else (0.5 if pv[b] == pv[a] else 0.0) for a, b in s]
    return 100 * sum(acc) / len(acc), len(s)


def main():
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    objset = opt("--objset")
    models = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--objset",))]
    groups = [("visible", "none"), ("early", "panel"), ("early", "mover"), ("mid", "panel"), ("mid", "mover")]
    ks = [None] if "--by-k" not in a else [None, "1", "2", "3"]
    P = pairs(f"{REPO}/data_csv/vobench_v3_vem/index.csv")
    PL = pairs(f"{REPO}/data_csv/vobench_v3/index.csv") if "--late" in a else None
    cols = [(o, kd, c, k) for o, kd in groups for k in (ks if o != "visible" else [None]) for c in CELLS]
    if PL:
        cols += [("late", kd, c, None) for kd in ("panel", "mover") for c in CELLS]
    head = ["model"] + [f"{o}:{kd}{':k' + k if k else ''}:{c}" for o, kd, c, k in cols]
    print(" | ".join(head))
    for m in models:
        f = f"{RES}/vobench_v3_vem_{m}/per_block.json"
        if not os.path.exists(f):
            print(" | ".join([m] + ["-"] * len(cols))); continue
        pv = json.load(open(f))["per_video_surprise"]
        pvl = json.load(open(f"{RES}/vobench_v3_{m}/per_block.json"))["per_video_surprise"] if PL and os.path.exists(f"{RES}/vobench_v3_{m}/per_block.json") else None
        row, ns = [], []
        for o, kd, c, k in cols:
            if o == "late":
                v, n = score(pvl, PL, lambda r: keep(r, None, kd, c, objset, None)) if pvl else (None, 0)
            else:
                v, n = score(pv, P, lambda r: keep(r, o, kd, c, objset, k))
            row.append("-" if v is None else f"{v:.1f}"); ns.append(n)
        print(" | ".join([m] + row))
    if "--n" in a:
        print(" | ".join(["n pairs"] + [str(x) for x in ns]))
        print(" | ".join(["95% CI (±)"] + [f"±{100 * 1.96 * math.sqrt(0.25 / x):.1f}" if x else "-" for x in ns]))


if __name__ == "__main__":
    main()
