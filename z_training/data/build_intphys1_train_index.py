#!/usr/bin/env python3
"""IntPhys1 train 분할(가능 영상만) 의 학습 인덱스를 만든다 — world_model_analysis 14컬럼 스키마.

  python z_training/data/build_intphys1_train_index.py            # -> data_csv/intphys1_train/index.csv
  python z_training/data/build_intphys1_train_index.py --root /local_datasets/world/IntPhys1 --n-frames 100

scene 폴더 `<id>/scene/scene_001.png .. scene_100.png` 이 전부 있는 id 만 넣는다.
status.json 의 header.is_possible 이 false 면 죽는다 (train 은 전부 가능이어야 한다).
csv 는 .gitignore 대상이라 untracked 가 정상이다.
"""
import argparse
import csv
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/local_datasets/world/IntPhys1")
ap.add_argument("--out", default="/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphys1_train/index.csv")
ap.add_argument("--n-frames", type=int, default=100)
a = ap.parse_args()

COLS = ["video_id", "file", "file_name", "block_id", "source_block", "variant", "plausible", "pair_id",
        "condition", "motion", "has_occlusion", "violation_type", "game_name", "role"]
ids = sorted(d for d in os.listdir(a.root) if re.fullmatch(r"\d{5}", d))
rows, bad = [], []
for sid in ids:
    sd = os.path.join(a.root, sid, "scene")
    if not os.path.isdir(sd):
        bad.append((sid, "scene 없음")); continue
    missing = [f for f in range(1, a.n_frames + 1) if not os.path.isfile(os.path.join(sd, f"scene_{f:03d}.png"))]
    if missing:
        bad.append((sid, f"프레임 {len(missing)}장 없음 (첫 {missing[:3]})")); continue
    st = os.path.join(a.root, sid, "status.json")
    if os.path.isfile(st):
        with open(st) as f:
            hdr = json.load(f).get("header", {})
        if hdr.get("is_possible") is False:
            raise SystemExit(f"{sid}: is_possible=false — train 에 불가능 영상이 있다")
    rows.append({"video_id": f"intphys1_train_{sid}", "file": "", "file_name": sid, "block_id": sid,
                 "source_block": sid, "variant": "pos", "plausible": "1", "pair_id": "", "condition": "train",
                 "motion": "", "has_occlusion": "", "violation_type": "none", "game_name": "", "role": "pos_obj"})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} clips -> {a.out}" + (f"  (제외 {len(bad)}: {bad[:5]})" if bad else ""))
