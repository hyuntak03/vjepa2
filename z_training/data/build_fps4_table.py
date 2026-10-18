#!/usr/bin/env python3
"""video csv → **fps 4 로 뽑을 수 있는 프레임 수** 표 (2026-10-02, 가변 창 로더 `var_window` 용).

  python z_training/data/build_fps4_table.py --csv data_csv/k710_320/train_min112.csv -o data_csv/k710_320/fps4_table.tsv
  python z_training/data/build_fps4_table.py --csv data_csv/ssv2/train_min32.csv      -o data_csv/ssv2/fps4_table.tsv
  python z_training/data/build_fps4_table.py --csv data_csv/ht100m_1pct/train.csv     -o data_csv/ht100m_1pct/fps4_table.tsv

출력 tsv: `path \\t fps(ceil) \\t n_frames \\t fstp \\t n4`
  fstp = ceil(fps) // 4 (최소 1) — VideoDataset 의 fps 4 규칙과 같다 (src/datasets/video_dataset.py)
  n4   = (n_frames - 1) // fstp + 1 — 그 간격으로 뽑을 수 있는 최대 장 수. 창 W 장이 들어가려면 n4 >= W
왜: 창 길이 (C+K 장) 를 step 마다 바꾸면서 **패딩 없이** 채울 수 있는 영상만 뽑아야 한다 (AR 에서 패딩 = 정지 화면 = 복사 정답).
    lengths.tsv 는 프레임 수만 있고 fps 가 없어서 (K710 은 30/25/24/15 fps 가 섞여 있다) 다시 잰다.
"""
import argparse
import math
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--workers", type=int, default=200)
ap.add_argument("--target-fps", type=int, default=4)
a = ap.parse_args()

paths = [ln.rstrip("\n").rsplit(" ", 1)[0] for ln in open(a.csv, encoding="utf-8") if ln.strip()]


def probe(p):
    import decord
    try:
        vr = decord.VideoReader(p, num_threads=1)
        n = len(vr)
        f = int(math.ceil(vr.get_avg_fps()))
    except Exception:
        return p, 0, 0, 0, 0
    fstp = max(1, f // a.target_fps) if f > 0 else 0
    n4 = (n - 1) // fstp + 1 if (fstp and n > 0) else 0
    return p, f, n, fstp, n4


print(f"probe {len(paths):,} 편 (workers {a.workers}) <- {a.csv}", flush=True)
rows = []
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for i, r in enumerate(ex.map(probe, paths, chunksize=64)):
        rows.append(r)
        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:,}/{len(paths):,}", flush=True)
with open(a.out, "w", encoding="utf-8") as g:
    for r in rows:
        g.write("\t".join(map(str, r)) + "\n")
ok = [r for r in rows if r[4] > 0]
print(f"썼다 {a.out}: {len(ok):,} / {len(rows):,} 열림")
for W in (4, 8, 16, 24, 32, 40, 48, 64):
    print(f"  창 {W:2d} 장 (fps {a.target_fps}): {sum(r[4] >= W for r in ok):>8,} 편 ({100 * sum(r[4] >= W for r in ok) / max(1, len(rows)):5.1f} %)")
