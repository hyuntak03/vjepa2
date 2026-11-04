#!/usr/bin/env python3
"""시트 Object_permanence 의 세 벤치 칸을 vast 표준 하네스 결과 (per_block.json 의 per_video_surprise) 에서 센다 (GPU 불필요, 2026-11-02).

칸 정의는 main z_research/scripts/analysis/object_permanence_cells.py (origin/main fc59160) 와 같다:
  IntPhys 1 late occ  index data_csv/intphys1_late_uni — block_type O1 · group {moving,static}/occluded (30 쌍).
                      A = 가능 쪽 event_kind != not_seen_before_d (문맥에서 물체를 봤다 → 불가능 = 사라짐), B = 나머지 (나타남)
  IntPhys 2 late occ  index data_csv/intphys2_late_final — condition permanence 120 쌍, Camera Moving 68 / Fixed 52 (A/B 라벨 없음)
  real world perm.    index data_csv/real_world_permanence_uniform — pair_id A (사라짐 95) / B (나타남 95)
  real world 보류 절반 index_test.csv (48 영상 · 96 쌍) — real bench 로 이어 학습한 판의 held-out 칸 (rwpT)
판정: s(불가능) > s(가능) 정답, 같으면 0.5. overall = 칸 평균이 아니라 짝 가중.
결과 폴더: cvpr/results/eval/intphys1_late/intphys1_late_uni_<모델> · cvpr/results/eval/testbed/{intphys2_late_final,real_world_permanence_uniform}_<모델>

  /venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py vith vast_ar_vw_state [...] [--tsv]
"""
import csv
import json
import os
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = f"{REPO}/cvpr/results/eval"
IDX = {"ip1": f"{REPO}/data_csv/intphys1_late_uni/index.csv",
       "ip2": f"{REPO}/data_csv/intphys2_late_final/index.csv",
       "rwp": f"{REPO}/data_csv/real_world_permanence_uniform/index.csv",
       "rwpT": f"{REPO}/data_csv/real_world_permanence_uniform/index_test.csv"}   # 보류 절반 48 영상 (z_training/data/build_realbench_split.py)
DIR = {"ip1": f"{RES}/intphys1_late/intphys1_late_uni_{{m}}",
       "ip2": f"{RES}/testbed/intphys2_late_final_{{m}}",
       "rwp": f"{RES}/testbed/real_world_permanence_uniform_{{m}}",
       "rwpT": f"{RES}/testbed/real_world_permanence_uniform_{{m}}"}   # 같은 채점 결과에서 시험 절반 영상만
# 시트 열 순서 (Object_permanence 탭 Y–AG 열과 같다)
COLS = [("ip1", "moving/A"), ("ip1", "moving/B"), ("ip1", "static/A"), ("ip1", "static/B"), ("ip1", "overall"),
        ("ip2", "Moving"), ("ip2", "Fixed"), ("ip2", "overall"),
        ("rwp", "A"), ("rwp", "B"), ("rwp", "overall"),
        ("rwpT", "A"), ("rwpT", "B"), ("rwpT", "overall")]   # real bench 로 학습한 판은 이 열 (보류 절반) 만 읽는다


def pairs(bench):
    """[(pos video_id, imp video_id, 가능 쪽 행)]"""
    by = defaultdict(dict)
    for r in csv.DictReader(open(IDX[bench])):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    out = [(v["pos"]["video_id"], v["imp"]["video_id"], v["pos"]) for v in by.values()]
    assert all(len(v) == 2 for v in by.values()), bench
    return out


def cell_filter(bench, cname):
    if bench == "ip1":
        base = lambda r: r["block_type"] == "O1" and r["group"].endswith("/occluded")
        if cname == "overall":
            return base
        m, ab = cname.split("/")
        isA = lambda r: r["event_kind"] != "not_seen_before_d"
        return lambda r: base(r) and r["group"] == f"{m}/occluded" and (isA(r) if ab == "A" else not isA(r))
    if bench == "ip2":
        perm = lambda r: r["condition"] == "permanence"
        return perm if cname == "overall" else (lambda r: perm(r) and r["Camera"] == cname)
    return (lambda r: True) if cname == "overall" else (lambda r: r["pair_id"] == cname)


def score(model):
    row, n = {}, {}
    for bench in ("ip1", "ip2", "rwp", "rwpT"):
        f = f"{DIR[bench].format(m=model)}/per_block.json"
        if not os.path.exists(f):
            continue
        pv = json.load(open(f))["per_video_surprise"]
        P = pairs(bench)
        for b, cname in COLS:
            if b != bench:
                continue
            sel = [(p, i) for p, i, r in P if cell_filter(bench, cname)(r)]
            miss = [k for p, i in sel for k in (p, i) if k not in pv]
            if miss:
                row[(b, cname)] = f"missing {len(miss)}"; continue
            acc = [1.0 if pv[i] > pv[p] else (0.5 if pv[i] == pv[p] else 0.0) for p, i in sel]
            row[(b, cname)] = round(100 * sum(acc) / len(acc), 1); n[(b, cname)] = len(acc)
    return row, n


def main():
    models = [a for a in sys.argv[1:] if not a.startswith("--")]
    tsv = "--tsv" in sys.argv
    hdr = [f"{b}:{c}" for b, c in COLS]
    print(("\t" if tsv else " | ").join(["model"] + hdr))
    nrow = None
    for m in models:
        row, n = score(m)
        nrow = nrow or n
        print(("\t" if tsv else " | ").join([m] + [str(row.get(k, "-")) for k in COLS]))
    if nrow:
        print(("\t" if tsv else " | ").join(["n pairs"] + [str(nrow.get(k, "-")) for k in COLS]))


if __name__ == "__main__":
    main()
