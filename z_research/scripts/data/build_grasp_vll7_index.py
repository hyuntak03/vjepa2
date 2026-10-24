#!/usr/bin/env python
"""GRASP level2 index 를 vll7 에서 읽을 수 있는 경로로 다시 쓴다 (2026-10-21).

원본 `data_csv/grasp_level2/index.csv` 의 `file` 열은 **vll5 노드 로컬** 경로다:
    /data2/local_datasets/world/Benchmark/GRASP/videos/level2/...
vll7 에는 `/data2` 마운트 자체가 없다 (확인: `df -h /data2` -> No such file or directory).
NFS 에 같은 트리가 풀려 있으므로 (`/data/dataset/world/Benchmarks/GRASP/videos/level2`,
mp4 5,760 편 = level1 + level2) 그쪽을 가리키는 `index_vll7.csv` 를 만든다.

⚠️ 영상 **내용은 같은 파일**이다 (stage.sh 가 이 NFS 사본에서 /data2 로 풀었다) — 채점값은 안 변한다.
⚠️ 복사하지 않는다. level2 약 5.5 GB 를 pass 당 한 번 읽을 뿐이고 비용은 decode 가 지배한다.

    python z_research/scripts/data/build_grasp_vll7_index.py
"""
import csv, os, sys

SRC = "data_csv/grasp_level2/index.csv"
DST = "data_csv/grasp_level2/index_vll7.csv"
OLD = "/data2/local_datasets/world/Benchmark/GRASP"
NEW = "/data/dataset/world/Benchmarks/GRASP"

rows = list(csv.DictReader(open(SRC)))
miss = []
for r in rows:
    f = r["file"]
    if not f.startswith(OLD):
        sys.exit(f"예상 밖 경로: {f}")
    r["file"] = f.replace(OLD, NEW, 1)
    if not os.path.isfile(r["file"]):
        miss.append(r["file"])
if miss:
    sys.exit(f"없는 파일 {len(miss)} 개, 예: {miss[:3]}")

with open(DST, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"{DST}: {len(rows)} 행, 실물 전수 확인 완료")
