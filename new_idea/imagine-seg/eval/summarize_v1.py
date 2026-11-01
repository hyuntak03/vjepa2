#!/usr/bin/env python3
"""imagine-seg 공정 비교 v1 결과 표 — 릴리즈(vith) · ① ctrl · ② head · ③ inpath 를 같은 쌍에서 (2026-11-01).

입력: cvpr/results/eval/{testbed,intphys1_late}/<ds>_<모델>/per_block.json (채점 산출물)
출력: new_idea/imagine-seg/results/summary_v1.md + summary_v1.json
같은 쌍 비교 = (틀림→맞음 / 맞음→틀림, McNemar 정규근사 p). permanence_wrop/eval/summarize.py 와 같은 규칙.
"""
import csv, json, math, os
from collections import defaultdict
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
M = {"R": "vith", "C": "vast_imgseg_ctrl", "H": "vast_imgseg_head", "P": "vast_imgseg_inpath"}
SETS = [
    ("vobench_v1_split_v2_test", "testbed", "data_csv/vobench_v1_split_v2/index_test.csv",
     lambda r: [f"law={r['condition'].split('/')[1]}", f"law={r['condition'].split('/')[1]}·timing={r['condition'].split('/')[2]}"]),
    ("real_bench_quads", "testbed", None, lambda r: [f"cond={r.get('condition','')}"]),
    ("intphys1_late_uni", "intphys1_late", None, lambda r: [f"type={r['block_type']}"]),
    ("intphys2_violation_uni", "intphys1_late", None, lambda r: [f"type={r['block_type']}"]),
]
CMP = (("C", "R"), ("H", "C"), ("P", "C"), ("P", "H"))

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

def mc(pp, new, ref):
    up = sum(p[2][new] and not p[2][ref] for p in pp); dn = sum(p[2][ref] and not p[2][new] for p in pp)
    return up, dn, (math.erfc(abs(up - dn) / math.sqrt(2 * (up + dn))) if up + dn else 1.0)

res, lines = {}, ["# imagine-seg v1 결과 표 (자동 생성: eval/summarize_v1.py)", "",
                  "정확도 % (같은 쌍). 비교 열 = 새 판 대 기준 판 (틀림→맞음 / 맞음→틀림, McNemar p). R 릴리즈 · C 대조군 · H 보조 head · P in-path. 방향 = pair_id.", ""]
for ds, task, index, gf in SETS:
    pp = pairs_of(ds, task, index)
    groups = defaultdict(list); groups["전체"] = pp
    for p in pp:
        groups[f"방향 {p[0]}"].append(p)
        for g in gf(p[1]):
            groups[g].append(p); groups[f"{g}·방향 {p[0]}"].append(p)
    lines += [f"## {ds}", "", "| 그룹 | n | R 릴리즈 | C 대조 | H head | P in-path | C−R | H−C | P−C | P−H |", "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    res[ds] = {}
    for g in sorted(groups, key=lambda s: (s != "전체", not s.startswith("방향"), s)):
        q = groups[g]; n = len(q)
        acc = {k: 100 * sum(p[2][k] for p in q) / n for k in M}
        cm = {f"{a}-{b}": mc(q, a, b) for a, b in CMP}
        res[ds][g] = {"n": n, "acc": acc, "cmp": cm}
        lines.append(f"| {g} | {n} | " + " | ".join(f"{acc[k]:.1f}" for k in M) + " | " +
                     " | ".join(f"{u}/{d} p{pv:.2f}" for u, d, pv in cm.values()) + " |")
    lines.append("")
os.makedirs(os.path.join(os.path.dirname(__file__), "../results"), exist_ok=True)
open(os.path.join(os.path.dirname(__file__), "../results/summary_v1.md"), "w").write("\n".join(lines))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/summary_v1.json"), "w"), indent=1, ensure_ascii=False)
print("\n".join(lines))
