"""InfLevel-lab 을 property 별로 레포 인덱스 스키마로 만든다.

    python z_research/scripts/data/build_inflevel_index.py --write

## 왜 property 별로 쪼개나
영상 길이가 property 마다 다르다 (전수 실측 2026-09-20):
    continuity 630~634 (n=2268) / gravity 375~440 (n=2592) / solidity 465~483 (n=912)
우리 하네스는 창 그리드를 **데이터셋당 한 번** `ds.n_frames` 로 계산하므로 셋을 한 데이터셋에
넣으면 최솟값(375)에 맞춰 잘라야 하고 continuity 가 40% 날아간다.
property 안에서는 편차가 작아 (손실 0.1% / 7.9% / 0.6%) 쪼개면 거의 무손실이다.
공식 `evaluator.py` 도 property 별로 따로 채점하므로 구조도 맞다.

## 채점 (공식 evaluator.py 그대로)
가능(real) / 불가능(magic) 배정:
    continuity : real vv, ii   / magic iv, vi
    solidity   : real ui, cv   / magic uv, ci
    gravity    : real ui, cv   / magic uv, ci
`c` = cut(바닥 없는 컵·뒤가 잘린 원통), `u` = uncut, `i`/`v` = 물체가 끝에 안 보임/보임.

`(camera, cover, obj, dir)` 로 묶어 **4 종이 다 있는 그룹만** 쓴다 (공식도 불완전 그룹을 버린다).
그 그룹 안에서 가능 2 x 불가능 2 = **4 비교** -> `scoring.pairing: cross` 가 그 계산이다.

⚠️ gravity·solidity 는 컵이 잘렸는지가 본 영상 앞의 contextualization 영상에만 나와서
   원리적으로 못 푼다 (Garrido §E). **믿을 수 있는 건 continuity 다.**
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

ROOT = "/data/dataset/world/Benchmarks/InfLevel/InfLevel"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv"
REAL = {"continuity": {"vv", "ii"}, "solidity": {"ui", "cv"}, "gravity": {"ui", "cv"}}
# 전수 실측 최솟값 (2026-09-20). raw_frames 로 쓴다.
MIN_FRAMES = {"continuity": 630, "gravity": 375, "solidity": 465}

COLS = ["video_id", "file", "block_id", "source_block", "variant", "plausible",
        "pair_id", "condition", "block_type", "trial_type", "camera", "cover", "obj", "dir", "role"]


def build(prop):
    groups = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{ROOT}/{prop}/*.mp4")):
        b = os.path.basename(f)
        if b.startswith("._"):                       # macOS 더미
            continue
        parts = b[:-4].split("__")
        if len(parts) < 5:
            continue
        cam, p, cover, obj, ttype = parts[:5]
        assert p == prop, (p, prop)
        dirn = parts[5] if len(parts) > 5 else ""
        groups[(cam, cover, obj, dirn)][ttype] = f

    rows, skipped = [], 0
    for (cam, cover, obj, dirn), d in sorted(groups.items()):
        if len(d) != 4:                              # 공식도 불완전 그룹은 버린다
            skipped += 1
            continue
        gid = f"{cam}|{cover}|{obj}|{dirn}"
        for ttype, f in sorted(d.items()):
            plaus = "1" if ttype in REAL[prop] else "0"
            rows.append(dict(
                video_id=os.path.basename(f)[:-4], file=os.path.abspath(f),
                block_id=gid, source_block=gid,
                variant=("pos_" + ttype if plaus == "1" else "imp_" + ttype),
                plausible=plaus, pair_id=gid, condition=prop, block_type=prop,
                trial_type=ttype, camera=cam, cover=cover, obj=obj, dir=dirn,
                role=("pos_obj" if plaus == "1" else "imp_obj")))
    return rows, skipped, len(groups)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    for prop in ("continuity", "gravity", "solidity"):
        rows, skipped, n_grp = build(prop)
        n_block = len({r["block_id"] for r in rows})
        pos = sum(r["plausible"] == "1" for r in rows)
        print(f"{prop:11s} 영상 {len(rows):5d}  block {n_block:4d}  "
              f"가능 {pos} / 불가능 {len(rows)-pos}  버린 그룹 {skipped} (전체 {n_grp})")
        assert n_block * 4 == len(rows), "block 하나에 4 영상이 아니다"
        assert pos * 2 == len(rows), "가능/불가능이 반반이 아니다"
        assert os.path.isfile(rows[0]["file"])
        if a.write:
            d = f"{OUT}/inflevel_{prop}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/index.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
            print(f"            -> {d}/index.csv   (raw_frames: {MIN_FRAMES[prop]})")
    if not a.write:
        print("(--write 를 붙여야 저장한다)")
