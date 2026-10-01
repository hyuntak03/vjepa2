#!/usr/bin/env python3
"""IntPhys 1 dev 인덱스 (data_csv/intphys1_dev/index.csv) 를 원본 PNG 폴더에서 만든다.

  python z_research/scripts/data/build_intphys1_index.py            # 검사만
  python z_research/scripts/data/build_intphys1_index.py --write    # 쓰기
  python z_research/scripts/data/build_intphys1_pairs.py --write    # 다음 단계: pair_id (픽셀 분기점)

배치: ${BENCH_ROOT}/IntPhys1/dev/{O1,O2,O3}/<4중항 01..30>/<run 1..4>/{scene,depth,masks}/ + status.json
공식 로더와 같은 규칙 (jepa-intuitive-physics/.../intphys_dataset.py:89-123):
  * 4중항은 sorted(listdir), run 도 sorted
  * 라벨 = status.json 의 header.is_possible
  ⚠️ starting_kit/dev/reference.txt 를 라벨로 쓰지 말 것 (2026-09-29). 그건 옛 dev 배포본
     (폴더 `01_test_visible_static_nobj1`, run 1·2 가능 고정) 기준이라 이 tar(`01`, run 섞임)와
     영상 172/360 만 일치한다. status.json 라벨로 기준값 88.89 가 정확히 재현된다.
pair_id 는 여기서 비워 둔다 — build_intphys1_pairs.py 가 공식 get_breaking_points/get_matches 로 채운다.

컬럼: video_id, file(비움 — PNG 직독), block, quadruplet, run, block_id, block_type, variant, plausible, pair_id
  * block_id    4중항마다 정수 0..89 (LIMIT 스모크가 int 로 정렬한다)
  * block_type  O1 / O2 / O3 (property, 채점 breakdown)
  * variant     pos_1 / imp_3 처럼 라벨_run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from paths import BENCH_ROOT, DATA_CSV  # noqa: E402  경로 정본: harness/paths.env

COLS = ["video_id", "file", "block", "quadruplet", "run", "block_id", "block_type",
        "variant", "plausible", "pair_id"]
N_FRAMES = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(BENCH_ROOT, "IntPhys1", "dev"))
    ap.add_argument("--out", default=os.path.join(DATA_CSV, "intphys1_dev"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows, bid = [], 0
    for blk in ("O1", "O2", "O3"):
        for quad in sorted(os.listdir(os.path.join(a.root, blk))):
            qd = os.path.join(a.root, blk, quad)
            runs = sorted(os.listdir(qd))
            if runs != ["1", "2", "3", "4"]:
                sys.exit(f"4중항이 아니다: {qd} -> {runs}")
            labs = []
            for run in runs:
                sd = os.path.join(qd, run, "scene")
                n = len([f for f in os.listdir(sd) if f.endswith(".png")])
                if n != N_FRAMES:
                    sys.exit(f"프레임 {n}장 (기대 {N_FRAMES}): {sd}")
                poss = bool(json.load(open(os.path.join(qd, run, "status.json")))["header"]["is_possible"])
                labs.append(poss)
                rows.append(dict(video_id=f"intphys1_{blk}_{quad}_{run}", file="", block=blk, quadruplet=quad, run=run,
                                 block_id=bid, block_type=blk,
                                 variant=f"{'pos' if poss else 'imp'}_{run}", plausible=int(poss), pair_id=""))
            if sum(labs) != 2:
                sys.exit(f"가능 {sum(labs)}/4 (기대 2): {qd}")
            bid += 1

    print(f"영상 {len(rows)} / 4중항 {bid} / 가능 {sum(r['plausible'] for r in rows)}")
    if a.write:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, "index.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader(); w.writerows(rows)
        print("썼다:", p)
    else:
        print("(--write 를 줘야 실제로 쓴다)")


if __name__ == "__main__":
    main()
