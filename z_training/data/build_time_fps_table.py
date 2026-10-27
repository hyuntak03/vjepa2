#!/usr/bin/env python3
"""fps4 표 → **시간 기준 fps T** 표 (2026-10-27, 사용자 결정: 학습을 채점과 같은 fps 10 으로).
  python z_training/data/build_time_fps_table.py --src data_csv/k710_320/fps4_table.tsv --fps 10 -o data_csv/k710_320/fps10t_table.tsv
입력 = build_fps4_table.py 의 tsv (path, fps(ceil), n_frames, fstp, n4) — 원본 fps · 프레임 수는 다시 재지 않고 그대로 쓴다.
출력 = 같은 5 열, 단 fstp = -T (음수 = 시간 기준 표시), n4 자리 = 시간 기준으로 뽑을 수 있는 최대 장 수
  n = floor((n_frames - 1) / fps · T) + 1.  로더 (var_window._read) 는 fstp < 0 이면 장 k 의 원본 index = round((t0 + k / T) · fps) 로 읽는다
  (정수 간격이 아니라 가장 가까운 프레임 — 25 · 24 · 12 fps 영상도 평균 간격이 정확히 1 / T 초)."""
import argparse, math
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
n_in = n_out = 0
with open(a.out, "w") as w:
    for ln in open(a.src, encoding="utf-8"):
        p, f, n, _, _ = ln.rstrip("\n").split("\t")
        f, n = int(f), int(n); n_in += 1
        m = int(math.floor((n - 1) / f * a.fps)) + 1 if (f > 0 and n > 0) else 0
        w.write(f"{p}\t{f}\t{n}\t{-a.fps}\t{m}\n"); n_out += m > 0
print(f"{a.src} → {a.out}: {n_in:,} 줄 · 쓸 수 있는 영상 {n_out:,}")
