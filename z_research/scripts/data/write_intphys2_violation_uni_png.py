#!/usr/bin/env python3
"""IntPhys2 violation-uni — index_mp4.csv 의 frames (원본 프레임 번호 32 개) 대로 PNG (256², bilinear) 를 병렬로 다시 쓰고 index.csv (PNG 판) 를 맞춘다 (2026-10-22).

  /venv/vjepa/bin/python z_research/scripts/data/write_intphys2_violation_uni_png.py [--procs 32]

라벨이 바뀌면: build_intphys2_violation_uni.py <라벨 JSON> --csv-only --shift=-10 (overrides.json 반영) → 이 스크립트 → build_intphys2_violation_uni_split.py.
PNG 는 학습 로더 (frames_index) 용. 채점 · 렌더는 mp4 를 바로 읽는다."""
import csv, os, sys
from multiprocessing import Pool
import numpy as np, decord
from PIL import Image
ROOT = "/workspace/dataset/benchmarks/IntPhys2/Main"
OUT = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys2_violation_uni"
D = "data_csv/intphys2_violation_uni"
P = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 32


def one(r):
    ts = [int(t) for t in r["frames"].split()]
    fr = decord.VideoReader(os.path.join(ROOT, r["file"]), num_threads=1).get_batch(ts).asnumpy()
    vd = os.path.join(OUT, r["video_id"]); os.makedirs(vd, exist_ok=True)
    for k, f in enumerate(fr):
        Image.fromarray(f).resize((256, 256), Image.BILINEAR).save(f"{vd}/{k:06d}.png")
    return r["video_id"]


rows = list(csv.DictReader(open(f"{D}/index_mp4.csv")))
with Pool(P) as pool:
    for i, _ in enumerate(pool.imap_unordered(one, rows, chunksize=4)):
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
with open(f"{D}/index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows([dict(r, file=r["video_id"]) for r in rows])
print(f"PNG {len(rows)} 영상 · index.csv 갱신")
