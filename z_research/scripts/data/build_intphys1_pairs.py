#!/usr/bin/env python
"""IntPhys 1 dev 의 **matched pair_id** 를 픽셀 분기점으로 구해 index.csv 에 쓴다.

    python z_research/scripts/data/build_intphys1_pairs.py            # 검사만
    python z_research/scripts/data/build_intphys1_pairs.py --write    # index.csv 에 pair_id 쓰기

⚠️ **2026-09-21 재작성.** 원래 `z_scripts/world_model_analysis/build_intphys1_pairs.py` 였는데
그 폴더를 지우면서 같이 사라졌다 (git 추적이 없었다). 로직은 공식 코드에 그대로 있어 복원했다.

왜 필요한가 — 4중항 4개 중 **문맥이 픽셀 단위로 같은 2쌍만** 채점한다 (`scoring.pairing: matched`).
설정이 다른 쌍은 위반 전부터 문맥이 갈라져 있어 surprise 차이가 위반 때문인지 설정 때문인지
분리되지 않는다. 그래서 논문도 그 쌍을 안 본다.

공식 구현 (`jepa-intuitive-physics/evaluation_code/evals/intuitive_physics/utils.py:157-174`):

    def get_breaking_points(clip):            # clip: (4, C, T, H, W)
        bps = []
        for diff in [clip[0]-clip[1], clip[0]-clip[2], clip[0]-clip[3]]:
            i = argwhere(diff.sum(2).sum(2).sum(0) != 0)[0,0]   # 처음 달라지는 프레임
            bps.append(i)                                       # 없으면 T
    def get_matches(bps):
        argmax(bps) == 0 -> [[0,1],[2,3]]
        argmax(bps) == 1 -> [[0,2],[1,3]]
        else             -> [[0,3],[1,2]]

즉 **0번과 가장 늦게 갈라지는 것**이 0번의 짝이다 (그 둘이 문맥을 가장 길게 공유한다).
여기서는 4중항의 run 1~4 를 그 순서로 읽어 같은 계산을 하고, 두 쌍에 `pair_id` 1·2 를 준다.
"""
from __future__ import annotations
import argparse, csv, os, pathlib, sys
from collections import defaultdict

import numpy as np

ROOT = "/local_datasets/world/world_analysis/IntPhys1_dev_videos"
FRAMES = "/local_datasets/world/world_analysis/IntPhys1_dev_frame_png"
PATTERN = "{block}/{quadruplet}/{run}/scene/scene_{frame:03d}.png"
N_PROBE = 100          # dev 영상은 100 프레임


def breaking_points(frames: list[np.ndarray]) -> list[int]:
    """frames[0] 대비 1·2·3 이 처음 달라지는 프레임. 안 달라지면 길이."""
    T = frames[0].shape[0]
    bps = []
    for k in (1, 2, 3):
        d = np.abs(frames[0].astype(np.int32) - frames[k].astype(np.int32)).sum(axis=(1, 2, 3))
        nz = np.nonzero(d)[0]
        bps.append(int(nz[0]) if len(nz) else T)
    return bps


def matches(bps: list[int]) -> list[list[int]]:
    a = int(np.argmax(bps))
    return [[0, 1], [2, 3]] if a == 0 else ([[0, 2], [1, 3]] if a == 1 else [[0, 3], [1, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--frames-root", default=FRAMES)
    a = ap.parse_args()
    from PIL import Image

    idx = os.path.join(a.root, "index.csv")
    rows = list(csv.DictReader(open(idx)))
    quads = defaultdict(dict)
    for r in rows:
        quads[(r["block"], r["quadruplet"])][int(r["run"])] = r

    agree = changed = 0
    for (blk, quad), runs in sorted(quads.items()):
        if sorted(runs) != [1, 2, 3, 4]:
            sys.exit(f"4중항이 아니다: {blk}/{quad} -> {sorted(runs)}")
        clips = []
        for run in (1, 2, 3, 4):
            fr = [np.asarray(Image.open(os.path.join(
                     a.frames_root, PATTERN.format(block=blk, quadruplet=quad, run=run, frame=f))).convert("RGB"))
                  for f in range(1, N_PROBE + 1)]
            clips.append(np.stack(fr))
        m = matches(breaking_points(clips))
        for pid, pair in enumerate(m, start=1):
            for j in pair:
                r = runs[j + 1]
                if r.get("pair_id", "") == str(pid): agree += 1
                else: changed += 1
                r["pair_id"] = str(pid)
        print(f"  {blk}/{quad}  bps={breaking_points(clips)}  pairs={m}")

    print(f"\n기존 pair_id 와 일치 {agree} / 바뀜 {changed}")
    if a.write:
        with open(idx, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("썼다:", idx)
    else:
        print("(--write 를 줘야 실제로 쓴다)")


if __name__ == "__main__":
    main()
