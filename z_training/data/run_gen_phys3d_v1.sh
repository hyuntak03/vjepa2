#!/bin/bash
# phys3d_v1 생성 (2026-10-19). 끊기면 같은 명령으로 이어 한다 (결과가 있는 작업은 건너뛴다).
#
#   BLENDER=<blender 4.2.3> OUT=<저장 위치> RENDER_GPUS=2,3,4,5,6,7 GPU_PER=2 CPU_WORKERS=200 bash z_training/data/run_gen_phys3d_v1.sh
#
# 재현: 라벨 (Bullet 물리 · 투영 · 사건 · 행) 은 (코드, seed, Blender 4.2.3) 으로 정해진다 — 워커 수 · 기계와 무관.
#   프레임 픽셀은 Cycles GPU (OptiX) 와 CPU 가 미세하게 다르다 (4 샘플 + 디노이즈). 어느 클립이 GPU 로 렌더되는지는 작업을 먼저 집는
#   워커가 정하므로, 픽셀까지 같아야 하면 RENDER_GPUS= (빈 값) 로 CPU 만 쓴다.
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
OUT=${OUT:-/workspace/dataset/training_data/phys3d_v1}
RENDER_GPUS=${RENDER_GPUS-2,3,4,5,6,7}   # ⚠️ GPUS 가 아니다 — Vast 기계에는 GPUS=8 (장 수) 이 환경에 이미 있다
GPU_PER=${GPU_PER:-2}
CPU_WORKERS=${CPU_WORKERS:-$(( $(nproc) - 16 ))}
mkdir -p $OUT
"$BLENDER" --version | head -1 | tee $OUT/versions.txt
python3 z_training/data/gen_phys3d.py --out $OUT/train --per-scenario 2200 --gpus "$RENDER_GPUS" --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS 2>&1 | tee -a $OUT/gen_train.log
python3 z_training/data/gen_phys3d.py --out $OUT/val --per-scenario 80 --seed-base 1000000 --gpus "$RENDER_GPUS" --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS 2>&1 | tee -a $OUT/gen_val.log
python3 z_training/data/fingerprint_phys_data.py --root $OUT/val   --write $OUT/val/fingerprint.tsv
python3 z_training/data/fingerprint_phys_data.py --root $OUT/train --write $OUT/train/fingerprint.tsv
