#!/bin/bash
# [T] 라벨 추가 (world 행렬 · 카메라 · 깊이) 비용 재기: 같은 3 장면을 깊이 있음 / 없음으로 (GPU 7, 워커 1)
cd /workspace/world_model/vjepa2
BL=/workspace/tools/blender-4.2.3-linux-x64/blender; G=auto_research/scripts/gen_phys3d_cf_blender.py; OUT=/workspace/dataset/training_data/phys3d_cf_v1
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4
$BL -t 4 -b -noaudio -P $G -- --scenes 0,1,2 --split pilot3_depth --seed-base 7300000 --out $OUT --keep-frames --depth > auto_research/logs/cf_pilot3_depth.log 2>&1
$BL -t 4 -b -noaudio -P $G -- --scenes 0,1,2 --split pilot3_nodepth --seed-base 7300000 --out $OUT --keep-frames > auto_research/logs/cf_pilot3_nodepth.log 2>&1
echo "cost pilot done $(date -u +%H:%M)"
