#!/usr/bin/env python3
"""video csv 를 **원본 길이**로 한 번 더 거른다 (재인코딩 사본 csv 용, probe 다시 안 함).

  python z_training/data/filter_min_frames.py --csv data_csv/k710_320/train_min112.csv \
      --lengths data_csv/k710/lengths.tsv --map K710_320/videos=K710/videos --min-frames 224 \
      -o data_csv/k710_320/train_min224.csv

왜 (2026-09-30): fps 4 x 32 장 창은 30 fps 원본에서 raw 224 장이 필요하다 (fstp 7). min112 로 거른 사본 csv 를
그대로 쓰면 짧은 영상에서 `filter_short_videos` 가 런타임 재추첨을 해 DDP 가 멈출 수 있다 (datasets.md ## ssv2_min48).
재인코딩은 프레임 수를 안 바꾸므로 원본 lengths.tsv 로 거르면 된다. --map 은 사본 경로 -> 원본 경로 치환 (lengths 의 키).
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--lengths", required=True)
ap.add_argument("--map", default=None, help="A=B : csv 경로의 A 를 B 로 바꿔 lengths 를 찾는다")
ap.add_argument("--min-frames", type=int, required=True)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

L = {}
for ln in open(a.lengths, encoding="utf-8"):
    k, v = ln.rstrip("\n").rsplit("\t", 1)
    L[k] = int(v)
src, dst = (a.map.split("=", 1) if a.map else (None, None))
n = m = miss = 0
with open(a.csv, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
    for ln in f:
        if not ln.strip():
            continue
        path = ln.rstrip("\n").rsplit(" ", 1)[0]
        n += 1
        key = path.replace(src, dst, 1) if src else path
        if key not in L:
            miss += 1
            continue
        if L[key] >= a.min_frames:
            g.write(ln if ln.endswith("\n") else ln + "\n")
            m += 1
print(f"{a.out}: {m:,} / {n:,} ({100 * m / max(n, 1):.1f} %) >= {a.min_frames} 장, lengths 에 없음 {miss}")
if miss:
    print("  ⚠️ lengths 에 없는 행은 뺐다 (--map 이 맞는지 확인)")
