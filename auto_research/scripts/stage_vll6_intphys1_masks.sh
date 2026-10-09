#!/bin/bash
# IntPhys 1 dev 의 masks/ + status.json 을 vll6 에 더 푼다 (stage_vll6.sh 는 scene/ 만 풀었다).
# 물체 중심·가시성 라벨을 masks 에서 만든다 (auto_research H-F). _by_scene 분석물도 옮긴다.
set -euo pipefail
[ "$(hostname)" = "vll6" ] || { echo "vll6 에서만"; exit 1; }
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
DST=/data2/local_datasets/world/world_analysis
tar xzf /data/dataset/world/Benchmarks/IntPhys1_dev.tar.gz -C "$DST/IntPhys1_dev_frame_png" --strip-components=1 \
    --wildcards '*/masks/*' '*/status.json'
mkdir -p "$DST/IntPhys1_dev_by_scene"; cp -r "$REPO/auto_research/_stage/IntPhys1_dev_by_scene/." "$DST/IntPhys1_dev_by_scene/"
ln -sfn "$DST/IntPhys1_dev_by_scene" /local_datasets/world/world_analysis/IntPhys1_dev_by_scene 2>/dev/null || true
find "$DST/IntPhys1_dev_frame_png" -name '*.png' -path '*masks*' | wc -l
echo DONE
