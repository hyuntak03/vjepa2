#!/usr/bin/env python3
"""VOBench_v2 late permanence (216 block × 4 arm = 864 clip, matched 쌍 432) — 채점 index + 프레임 NFS 사본 (2026-11-02, 시트 Object_permanence 의 VOBench 열을 v2 로).

원본: /data2/local_datasets/world/Benchmark/VOBench_v2 (vll5 로컬 HDD, 다른 노드에 없다). VOBench_v2_permanence 는 같은 파일의 하드 링크 (inode 같음).
index: data_csv/vobench_v2/index.csv 에서 violation_type == permanence & occlusion == late 만 + blocks.csv 의 replica 열.
  replica 0 = VOBench_v1 의 block 그대로 (같은 렌더) · 1 / 2 = 같은 설계 칸 · 같은 경로, A 물체만 새것.
프레임: 채점이 읽는 보폭 3 의 32 장 (0, 3, …, 93) 만 원본 해상도 PNG 그대로 복사 → 채점기 리사이즈 경로가 v1 표준 채점과 같다.
출력: data_csv/vobench_v2_late_perm/index.csv · /data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_perm/<file_name>/<frame:06d>.png
채점 SET: data.root=<index 폴더> data.frames_root=<사본> (데이터셋 preset vobench_v1 = 같은 frames_pattern · stride 3)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/data/build_vobench_v2_late_perm.py [--copy] [--workers 32]
"""
import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data2/local_datasets/world/Benchmark/VOBench_v2"
DST = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_perm"
FRAMES = list(range(0, 96, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    rep = {r["block_id"]: r["replica"] for r in csv.DictReader(open(f"{SRC}/blocks.csv"))}
    rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v2/index.csv"))
            if r["violation_type"] == "permanence" and r["occlusion"] == "late"]
    for r in rows:
        r["replica"] = rep[r["source_block"]]
    assert len(rows) == 864, len(rows)
    od = f"{REPO}/data_csv/vobench_v2_late_perm"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} 행 -> {od}/index.csv")
    if not a.copy:
        return
    jobs = [(f"{SRC}/{r['file_name']}/{fr:06d}.png", f"{DST}/{r['file_name']}/{fr:06d}.png") for r in rows for fr in FRAMES]
    for d in {os.path.dirname(t) for _, t in jobs}:
        os.makedirs(d, exist_ok=True)

    def cp(st):
        s, t = st
        if os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s):
            return 0
        shutil.copyfile(s, t)
        return 1

    with ThreadPoolExecutor(a.workers) as ex:
        n = sum(ex.map(cp, jobs))
    bad = [t for s, t in jobs if os.path.getsize(t) != os.path.getsize(s)]
    print(f"복사 {n} / {len(jobs)} (이미 있던 것 {len(jobs) - n}) · 크기 불일치 {len(bad)} -> {DST}")


if __name__ == "__main__":
    main()
