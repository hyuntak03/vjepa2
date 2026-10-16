#!/usr/bin/env python3
"""VOBench (본판 + gravity) 를 (물체 모양, 배경) 조합으로 학습 / held-out 으로 나눈다 (2026-10-15, 사용자 결정).

  python z_research/scripts/data/build_vobench_heldout.py

held-out 조합 (objset 마다 12 조합 중 4 개 — 배경마다 2 개, 같은 모양이 두 배경에서 모두 빠지지 않게):
  primitive : v11 {cone, torus}            · assets {cube, capsule}
  realistic : v11 {car, tennis_ball}       · assets {bus, food_apple}
block 규칙: 그 block 의 물체 A 또는 물체 B (모양 위반의 바뀐 물체) 가 held-out 조합이면 test block. 학습은 나머지 block 의 **가능 arm 만**
  (본판 pos_a · pos_b, gravity pos). 불가능 arm 은 학습에 절대 넣지 않는다.
입력: data_csv/vobench_v1/index.csv · data_csv/vobench_v1_gravity/index.csv (build_vobench_index.py · build_vobench_gravity_index.py)
출력: data_csv/vobench_v1_heldout/{index.csv (test block 의 4 arm, 채점용), train_pos.csv (학습 가능 arm), split.json}
      data_csv/vobench_v1_gravity_heldout/{index.csv, train_pos.csv, split.json}
      (context_integrity.json 은 원본 폴더에서 복사)
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
HELD = {
    ("primitive", "v11"): {"cone", "torus"},
    ("primitive", "assets"): {"cube", "capsule"},
    ("realistic", "v11"): {"car_641e", "tennis_ball_edc3"},
    ("realistic", "assets"): {"bus_4e19", "food_apple_01"},
}


def shape(o):
    return (o or "").split("/")[0]


def held(objset, bg, *objs):
    return any(shape(o) in HELD[(objset, bg)] for o in objs if o and o not in ("empty", "-"))


def split(src, dst, obj_cols, possible):
    rows = list(csv.DictReader(open(os.path.join(REPO, src, "index.csv"))))
    cols = list(rows[0].keys())
    by_block = {}
    for r in rows:
        by_block.setdefault(r["source_block"], []).append(r)
    test_blocks = {b for b, rs in by_block.items()
                   if any(held(r["objset"], r["background"], *[r.get(c) for c in obj_cols]) for r in rs)}
    test = [r for r in rows if r["source_block"] in test_blocks]
    train = [r for r in rows if r["source_block"] not in test_blocks and r["variant"] in possible]
    os.makedirs(os.path.join(REPO, dst), exist_ok=True)
    for name, rs in (("index.csv", test), ("train_pos.csv", train)):
        with open(os.path.join(REPO, dst, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
    ci = os.path.join(REPO, src, "context_integrity.json")
    if os.path.exists(ci):
        shutil.copy(ci, os.path.join(REPO, dst, "context_integrity.json"))
    # 누수 검사: 학습 arm 의 (objset, 배경, 모양) 중 held-out 조합이 없어야 한다
    leak = [r["video_id"] for r in train if held(r["objset"], r["background"], *[r.get(c) for c in obj_cols])]
    info = {"held_out": {f"{k[0]}/{k[1]}": sorted(v) for k, v in HELD.items()},
            "blocks_total": len(by_block), "blocks_test": len(test_blocks), "blocks_train": len(by_block) - len(test_blocks),
            "test_clips": len(test), "train_pos_clips": len(train), "leak": leak,
            "train_by_condition": Counter(r["condition"] for r in train),
            "test_by_condition": Counter(r["condition"] for r in test if r["variant"] in possible)}
    json.dump(info, open(os.path.join(REPO, dst, "split.json"), "w"), indent=1, default=dict)
    print(f"{src} → {dst}: block {info['blocks_total']} (test {info['blocks_test']} · train {info['blocks_train']}) | "
          f"test clip {len(test)} · 학습 가능 arm {len(train)} | 누수 {len(leak)}")
    return info


a = split("data_csv/vobench_v1", "data_csv/vobench_v1_heldout", ["obj_a", "obj_b"], {"pos_a", "pos_b"})
g = split("data_csv/vobench_v1_gravity", "data_csv/vobench_v1_gravity_heldout", ["obj"], {"pos"})
if a["leak"] or g["leak"]:
    sys.exit("누수!")
