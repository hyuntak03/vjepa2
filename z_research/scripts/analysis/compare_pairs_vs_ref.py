#!/usr/bin/env python3
"""두 모델의 matched-pair 채점을 **같은 쌍끼리** 비교 (2026-10-31, release_aux_perm_open 판정용).

정확도 두 개를 따로 비교하면 표준오차 ±2.4pt (432 쌍) 라 둔하다. 같은 쌍에서 맞음 → 틀림 / 틀림 → 맞음 을 세면 (McNemar) 훨씬 민감하다.
방향을 나눈다 (CLAUDE.md §1-7, 외형 편향을 숨기지 않기 위해):
  A = pos_a vs imp_ab (문맥에 물체가 있음 — 물체가 다시 나와야 / 그대로 나와야 맞음)
  B = pos_b vs imp_ba (VOBench: 문맥 b 쪽 / real_bench: 처음부터 없음 — 안 나와야 맞음)

  /venv/vjepa/bin/python z_research/scripts/analysis/compare_pairs_vs_ref.py --ds vobench_v1_late --ref vith --new vast_aux_open_e3
입력: cvpr/results/eval/testbed/<ds>_<모델>/per_block.json · index = 데이터셋 레지스트리의 root/index.csv
"""
import argparse, csv, json, math, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--ref", default="vith")
ap.add_argument("--new", required=True)
ap.add_argument("--index", default=None, help="기본: data_csv/<ds>/index.csv")
ap.add_argument("--group", default="condition")
ap.add_argument("--task", default="testbed", help="cvpr/results/eval/<task> (intphys1_late 등)")
a = ap.parse_args()
R = os.path.join(REPO, "cvpr/results/eval", a.task)
S = {m: json.load(open(os.path.join(R, f"{a.ds}_{m}", "per_block.json")))["per_video_surprise"] for m in (a.ref, a.new)}
rows = list(csv.DictReader(open(a.index or os.path.join(REPO, "data_csv", a.ds, "index.csv"))))
by = {}
for r in rows:                                               # 짝 = (block, pair_id) 안의 가능 1 · 불가능 1 (VOBench · real_bench: pair_id A/B, IntPhys1 1/2, IntPhys2 1)
    by.setdefault((r["block_id"], r.get("pair_id", "")), {}).setdefault(str(r.get("plausible", "")), r)
pairs = []                                                   # (방향 = pair_id, 그룹, ref 맞음, new 맞음, ref 차, new 차)
for (b, pid), d in by.items():
    if "1" not in d or "0" not in d: continue
    vp, vi = d["1"]["video_id"], d["0"]["video_id"]
    if vp not in S[a.ref] or vp not in S[a.new] or vi not in S[a.ref] or vi not in S[a.new]: continue
    m0, m1 = S[a.ref][vi] - S[a.ref][vp], S[a.new][vi] - S[a.new][vp]
    pairs.append((pid, d["1"].get(a.group, ""), m0 > 0, m1 > 0, m0, m1))

def line(name, pp):
    n = len(pp)
    if not n: return
    r0 = sum(p[2] for p in pp); r1 = sum(p[3] for p in pp)
    b = sum(p[2] and not p[3] for p in pp); c = sum(p[3] and not p[2] for p in pp)   # 맞음→틀림, 틀림→맞음
    pv = math.erfc(abs(c - b) / math.sqrt(2 * (b + c))) if b + c else 1.0            # McNemar (정규 근사, 양측)
    print(f"  {name:34s} n {n:4d} | {a.ref} {100 * r0 / n:5.1f} → {a.new} {100 * r1 / n:5.1f} ({100 * (r1 - r0) / n:+5.1f}) | 틀림→맞음 {c:3d} · 맞음→틀림 {b:3d} · p {pv:.3f}")


print(f"{a.ds}: {a.ref} vs {a.new} (같은 쌍 비교)")
line("전체", pairs)
dirs = sorted({p[0] for p in pairs})
if len(dirs) > 1:
    for dirn in dirs:
        line(f"방향 {dirn}", [p for p in pairs if p[0] == dirn])
for g in sorted({p[1] for p in pairs}):
    for dirn in dirs:
        line(f"{g}" + (f" · {dirn}" if len(dirs) > 1 else ""), [p for p in pairs if p[1] == g and p[0] == dirn])
