#!/usr/bin/env python3
"""vll7 노드 로컬 학습 영상 -> `"<mp4> <label>"` csv (2026-10-16).

  python z_vll7_training/data/build_vll7_index.py --dataset k400_320
  python z_vll7_training/data/build_vll7_index.py --dataset ssv2

이 노드의 배치 (다른 노드와 다르다 — data_csv/ssv2/train_min48.csv 등은 /data2 경로라 여기서 안 맞는다):
  K400 320p train : /local_datasets/Kinetics400/kinetics400_320p/train/*.mp4   (240,436 편, 평평한 폴더)
  SSv2            : /local_datasets/something-something-v2-mp4/*.mp4           (146,773 편)
라벨 칸은 VideoDataset 이 안 쓰므로 0 을 적는다 (형식만 맞춘다).
길이 · fps 는 여기서 재지 않는다 — `build_fps4_table.py` 가 한 번에 재고, 그 표에서
고정 창용 min-frames csv 도 파생한다 (`derive_fixed_csv.py`).
"""
import argparse
import os
import pathlib

ROOTS = {
    "k400_320": "/local_datasets/Kinetics400/kinetics400_320p/train",
    "k400_256": "/local_datasets/Kinetics400/kinetics400_256p/train",
    "ssv2": "/local_datasets/something-something-v2-mp4",
}
OUT = {
    "k400_320": "data_csv/vll7_k400_320/all.csv",
    "k400_256": "data_csv/vll7_k400_256/all.csv",
    "ssv2": "data_csv/vll7_ssv2/all.csv",
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(ROOTS))
ap.add_argument("-o", "--out")
a = ap.parse_args()

root = pathlib.Path(ROOTS[a.dataset])
if not root.is_dir():
    raise SystemExit(f"없다: {root}")
out = pathlib.Path(a.out or OUT[a.dataset])
out.parent.mkdir(parents=True, exist_ok=True)
paths = sorted(str(p) for p in root.glob("*.mp4"))
if not paths:                                   # 하위 폴더 배치 (클래스별) 도 받아 준다
    paths = sorted(str(p) for p in root.rglob("*.mp4"))
with open(out, "w", encoding="utf-8") as g:
    for p in paths:
        g.write(f"{p} 0\n")
print(f"{a.dataset}: {len(paths):,} 편 -> {out}")
