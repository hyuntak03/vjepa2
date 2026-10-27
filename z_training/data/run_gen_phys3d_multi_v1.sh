#!/bin/bash
# phys3d_multi_v1 (2026-10-25) — 4 fps 64 장 · 문맥 끝 고정 (C* = 2 · 4 · 8 · 16 · 32 층화) · 가림 4 종 · 배경 (방 / HDRI) · 시나리오 33. 끊기면 같은 명령으로 이어 한다.
#
#   TRAIN_PER=600 VAL_PER=30 CPU_WORKERS=224 RENDER_PER=2 bash z_training/data/run_gen_phys3d_multi_v1.sh
#
# 라벨 (CPU 1 코어, 편당 ~10~30 초) 이 병목, 렌더는 GPU 렌더 워커가 편당 1.3~2.5 초 (persistent data).
# 문서 z_training/docs/PHYS3D_MULTI_DATASET_2026-10-25.md · 원칙 z_training/docs/SYNTHETIC_TRAINSET_PRINCIPLES_2026-10-25.md
set -e
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
PY=${PY:-python3}
OUT=${OUT:-/workspace/dataset/training_data/phys3d_multi_v1}
RENDER_GPUS=${RENDER_GPUS-0,1,2,3,4,5,6,7}
GPU_PER=${GPU_PER:-30}
CPU_WORKERS=${CPU_WORKERS:-224}
TRAIN_PER=${TRAIN_PER:-600}
VAL_PER=${VAL_PER:-30}
RENDER_PER=${RENDER_PER:-2}
# 라벨 (CPU 워커 CPU_WORKERS 개) 과 렌더 (GPU 마다 RENDER_PER 개, 오래 사는 프로세스) 를 나눈다 — GPU 워커 수십 개가 렌더마다 CUDA 를 잡으면 드라이버 잠금에서 멈췄다 (2026-10-25)
ARGS="--multi --defer-render --render-per $RENDER_PER --gpus=$RENDER_GPUS --cpu-workers $CPU_WORKERS --nice 5"
mkdir -p $OUT
v=$("$BLENDER" --version | head -1)
echo "$v | python $($PY -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')" | tee $OUT/versions.txt
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/val --per-scenario $VAL_PER --seed-base 1000000 2>&1 | tee -a $OUT/gen_val.log
$PY z_training/data/audit_phys3d_multi.py --root $OUT/val > $OUT/val/audit.md
[ "$VAL_ONLY" = 1 ] && exit 0
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/train --per-scenario $TRAIN_PER 2>&1 | tee -a $OUT/gen_train.log
$PY z_training/data/audit_phys3d_multi.py --root $OUT/train > $OUT/train/audit.md
