#!/usr/bin/env python3
"""fps4 표 -> **고정 창** 샘플러용 min-frames csv (2026-10-16).

  python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_k400_320/fps4_table.tsv \\
         --mode fps4 --min 32 -o data_csv/vll7_k400_320/fps4_min32.csv
  python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_ssv2/fps4_table.tsv \\
         --mode raw  --min 32 -o data_csv/vll7_ssv2/fs1_min32.csv

왜 미리 거르나: `VideoDataset` 은 창을 못 채우는 영상을 **런타임에 재추첨**한다 — DDP 에서 rank 마다
재추첨 횟수가 달라 멈출 수 있다 (`configs/training/datasets.md` 의 `*_min<N>` 관례).

  --mode fps4 : 표의 n4 (= fps 4 로 뽑을 수 있는 최대 장 수) >= min   -> 데이터셋 spec 에 `fps: 4`
  --mode raw  : 표의 n_frames >= min                                 -> 데이터셋 spec 에 `frame_step: 1`
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--mode", required=True, choices=("fps4", "raw"))
ap.add_argument("--min", type=int, required=True)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

col = 4 if a.mode == "fps4" else 2          # path, fps, n_frames, fstp, n4
keep = tot = 0
with open(a.out, "w", encoding="utf-8") as g:
    for ln in open(a.table, encoding="utf-8"):
        f = ln.rstrip("\n").split("\t")
        tot += 1
        if int(f[4]) > 0 and int(f[col]) >= a.min:    # n4 == 0 = 안 열리는 파일
            g.write(f"{f[0]} 0\n")
            keep += 1
print(f"{a.out}: {keep:,} / {tot:,} 편 ({a.mode} >= {a.min})")
