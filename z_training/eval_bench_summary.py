#!/usr/bin/env python3
"""eval_bench.sh 결과 요약 (2026-10-11). python z_training/eval_bench_summary.py <run 폴더> [ckpt stem=latest]

IntPhys1 A.8 : 창별 summary.json 의 best_avg.block_pairwise (Filtered 고정, 하네스 자체 채점) → 창 최고 = 격자 최고
IntPhys 2    : 창별 summary.json 의 overall / breakdown.difficulty → **열별 최고** (Table 2 캡션 규칙)
값은 전부 산출물에서 다시 읽는다.
"""
import glob
import json
import os
import re
import sys

run = sys.argv[1]
stem = sys.argv[2] if len(sys.argv) > 2 else "latest"
E = os.path.join(run, "eval")
print(f"== {os.path.basename(run)} / {stem} ==")

best = None
for d in sorted(glob.glob(os.path.join(E, f"intphys1_sliding__intphys1_dev_{stem}_w*"))):
    f = os.path.join(d, "summary.json")
    if not os.path.isfile(f):
        print(f"IntPhys1 {os.path.basename(d)}: summary 없음"); continue
    s = json.load(open(f))
    ba = s.get("best_avg") or {}
    acc = 100 * float(ba.get("block_pairwise", float("nan")))
    combo = (ba.get("key") or "?").split("/")[0]
    print(f"IntPhys1 A.8 {d.rsplit('_', 1)[-1]}: {acc:.2f}%  (칸 {combo}, ties {ba.get('n_ties', 0)})")
    if best is None or acc > best[0]:
        best = (acc, d.rsplit('_', 1)[-1])
if best:
    print(f"IntPhys1 A.8 격자 최고: {best[0]:.2f}% ({best[1]})   [릴리즈 88.89 · tube 기준선 82.22]")

cols = {}
for d in sorted(glob.glob(os.path.join(E, f"intphys2_main_{stem}_w*"))):
    f = os.path.join(d, "summary.json")
    w = d.rsplit("_", 1)[-1]
    if not os.path.isfile(f):
        print(f"IntPhys2 {w}: summary 없음"); continue
    s = json.load(open(f))
    row = {"Overall": (s.get("overall") or {}).get("pairwise_accuracy")}
    # breakdown 키는 "condition=..|difficulty=..|camera=.." 결합 칸 → 난이도별로 n_correct / n_pairs 를 합친다
    acc = {}
    for k, v in (s.get("breakdown") or {}).items():
        p = dict(x.split("=", 1) for x in k.split("|") if "=" in x)
        if "difficulty" in p and isinstance(v, dict) and v.get("n_pairs"):
            a = acc.setdefault(p["difficulty"], [0, 0]); a[0] += v["n_correct"]; a[1] += v["n_pairs"]
    for lvl in ("Easy", "Medium", "Hard"):
        if lvl in acc:
            row[lvl] = acc[lvl][0] / acc[lvl][1]
    print(f"IntPhys2 {w} (best C={s.get('best_context_length')}): "
          + "  ".join(f"{k} {100 * v:.2f}" for k, v in row.items() if v is not None))
    for k, v in row.items():
        if v is not None and (k not in cols or v > cols[k][0]):
            cols[k] = (v, w)
if cols:
    order = sorted(cols, key=lambda k: (k == "Overall", k))
    print("IntPhys2 열별 최고: " + "  ".join(f"{k} {100 * cols[k][0]:.2f} ({cols[k][1]})" for k in order))
