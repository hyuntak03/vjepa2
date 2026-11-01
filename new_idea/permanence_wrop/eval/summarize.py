#!/usr/bin/env python3
"""permanence_wrop 결과 표 — 릴리즈(vith) · B · C · D 를 같은 쌍에서 나란히 (2026-11-01).

입력: cvpr/results/eval/{testbed,intphys1_late}/<ds>_<모델>/per_block.json (채점 체인 산출물)
출력: results/summary_tables.md (그룹별 정확도 + 각 판 대 B 의 같은 쌍 McNemar)
"""
import csv, json, math, os
from collections import defaultdict
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
M = {"A": "vith", "B": "vast_permwrop_B", "C": "vast_permwrop_C", "D": "vast_permwrop_D"}
SETS = [  # (ds, task, index, 그룹 함수)
    ("vobench_v1_split_v2_test", "testbed", "data_csv/vobench_v1_split_v2/index_test.csv",
     lambda r: [f"law={r['condition'].split('/')[1]}", f"law={r['condition'].split('/')[1]}·timing={r['condition'].split('/')[2]}"]),
    ("real_bench_quads", "testbed", None, lambda r: [f"cond={r.get('condition','')}"]),
    ("intphys1_late_uni", "intphys1_late", None, lambda r: [f"type={r['block_type']}", f"type={r['block_type']}·{r['group']}"]),
    ("intphys2_violation_uni", "intphys1_late", None, lambda r: [f"type={r['block_type']}"]),
]

def pairs_of(ds, task, index):
    R = os.path.join(REPO, "cvpr/results/eval", task)
    S = {k: json.load(open(os.path.join(R, f"{ds}_{m}", "per_block.json")))["per_video_surprise"] for k, m in M.items()}
    rows = list(csv.DictReader(open(os.path.join(REPO, index or f"data_csv/{ds}/index.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["block_id"], r.get("pair_id", ""))][str(r.get("plausible", ""))] = r
    out = []
    for (b, pid), d in by.items():
        if "1" not in d or "0" not in d: continue
        vp, vi = d["1"]["video_id"], d["0"]["video_id"]
        if not all(vp in S[k] and vi in S[k] for k in M): continue
        out.append((pid, d["1"], {k: S[k][vi] - S[k][vp] > 0 for k in M}))
    return out

def mc(pp, x, y):
    b = sum(p[2][x] and not p[2][y] for p in pp); c = sum(p[2][y] and not p[2][x] for p in pp)
    return c, b, (math.erfc(abs(c - b) / math.sqrt(2 * (b + c))) if b + c else 1.0)

lines = ["# permanence_wrop 결과 표 (자동 생성: eval/summarize.py)", "",
         "정확도 % (같은 쌍). `C−B` 열 = C 대 B 같은 쌍 비교 (틀림→맞음 / 맞음→틀림, McNemar p). 방향 = pair_id.", ""]
for ds, task, index, gf in SETS:
    pp = pairs_of(ds, task, index)
    groups = defaultdict(list)
    groups["전체"] = pp
    for p in pp:
        groups[f"방향 {p[0]}"].append(p)
        for g in gf(p[1]):
            groups[g].append(p); groups[f"{g}·방향 {p[0]}"].append(p)
    lines += [f"## {ds}", "", "| 그룹 | n | A 릴리즈 | B | C | D | C−B | D−B | C−D |", "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for g in sorted(groups, key=lambda s: (s != "전체", not s.startswith("방향"), s)):
        q = groups[g]; n = len(q)
        acc = {k: 100 * sum(p[2][k] for p in q) / n for k in M}
        cmp = []
        for x, y in (("B", "C"), ("B", "D"), ("D", "C")):
            c, b, pv = mc(q, x, y); cmp.append(f"{c}/{b} p{pv:.2f}")
        lines.append(f"| {g} | {n} | {acc['A']:.1f} | {acc['B']:.1f} | {acc['C']:.1f} | {acc['D']:.1f} | " + " | ".join(cmp) + " |")
    lines.append("")
    if ds == "real_bench_quads":
        same = sum(p[2]["B"] == p[2]["C"] == p[2]["D"] for p in pp)
        lines.append(f"real_bench: B · C · D 정오가 같은 쌍 {same}/{len(pp)}"); lines.append("")
open(os.path.join(os.path.dirname(__file__), "../results/summary_tables.md"), "w").write("\n".join(lines))
print("\n".join(lines))
