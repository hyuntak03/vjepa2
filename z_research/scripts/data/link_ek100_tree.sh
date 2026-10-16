#!/bin/bash
# EK100 256p 미러 (평평한 videos/<vid>.MP4) → 엔진 file_format 0 이 읽는 <pid>/videos/<vid>.MP4 심볼릭 트리 (2026-10-13, vast)
#   bash z_research/scripts/data/link_ek100_tree.sh [src=/workspace/dataset/benchmarks/EK100/videos_256p/videos] [dst=$CVPR_EK100]
SRC=${1:-/workspace/dataset/benchmarks/EK100/videos_256p/videos}
DST=${2:-/workspace/world_model/local_datasets/EPIC-KITCHENS_256p}
n=0
for f in "$SRC"/*.MP4; do
  b=$(basename "$f"); pid=${b%%_*}
  mkdir -p "$DST/$pid/videos"; ln -sfn "$f" "$DST/$pid/videos/$b"; n=$((n+1))
done
echo "링크 $n 개 -> $DST"
