#!/usr/bin/env python3
"""VOBench v1 late (216 블록 · 432 쌍) 를 위반 종류 (permanence · color · shape) × 물체 움직임 / 정지 × A / B 로 나눠 센다 (2026-11-02). GPU 불필요.

A = 문맥의 물체 A 가 미래에 그대로인 짝 (pos_a vs imp_ab), B = 반대 방향 (pos_b vs imp_ba).
  permanence: A = 사라짐 · B = 나타남 / color · shape: A · B 모두 "다시 나온 물체의 색 · 모양이 바뀜" (방향만 다름).
물체 정지 = motion flat_static. 판정: s(불가능) > s(가능) 정답, 같으면 0.5. 결과 = cvpr/results/eval/testbed/vobench_v1_late_<모델>/per_block.json
⚠️ VOBench 로 이어 학습한 판 (vobpft) 은 split v2 학습 절반이 이 432 쌍에 섞여 있다 — 그 판은 이 표로 읽지 말 것.
  /venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob1late.py <모델> ...
"""
import csv, json, os, sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
by = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1_late/index.csv")):
    by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
P = [(v["pos"], v["imp"]) for v in by.values()]
COLS = [(vt, m, ab) for vt in ("permanence", "color", "shape") for m in ("moving", "static") for ab in ("A", "B")]
mv = lambda r: "static" if r["motion"] == "flat_static" else "moving"
print(" | ".join(["model"] + [f"{vt[:5]}:{m}:{ab}" for vt, m, ab in COLS] + ["moving 전체", "전체"]))
for m in sys.argv[1:]:
    pv = json.load(open(f"{REPO}/cvpr/results/eval/testbed/vobench_v1_late_{m}/per_block.json"))["per_video_surprise"]
    j = lambda p, i: 1.0 if pv[i["video_id"]] > pv[p["video_id"]] else (0.5 if pv[i["video_id"]] == pv[p["video_id"]] else 0.0)
    row = []
    for vt, mm, ab in COLS:
        s = [j(p, i) for p, i in P if p["violation_type"] == vt and mv(p) == mm and p["pair_id"] == ab]
        row.append(f"{100 * sum(s) / len(s):.1f}")
    mov = [j(p, i) for p, i in P if mv(p) == "moving"]; al = [j(p, i) for p, i in P]
    print(" | ".join([m] + row + [f"{100 * sum(mov) / len(mov):.1f}", f"{100 * sum(al) / len(al):.1f}"]))
print(" | ".join(["n pairs"] + [str(sum(1 for p, i in P if p["violation_type"] == vt and mv(p) == mm and p["pair_id"] == ab)) for vt, mm, ab in COLS]
                 + [str(sum(1 for p, i in P if mv(p) == "moving")), str(len(P))]))
