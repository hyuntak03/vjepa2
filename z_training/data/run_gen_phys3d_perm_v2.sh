#!/bin/bash
# phys3d_perm_v2 (2026-10-31) — phys3d_perm_v1 + 영속 · 이식 원칙 (z_training/docs/DATA_DESIGN_PRINCIPLES_2026-10-26.md 1.5 부). 끊기면 같은 명령으로 이어 한다.
#   사용자 (2026-10-31): "원칙을 이거 기록하고 빠르게 blender 로 3D dataset … cpu, gpu 병렬 최대화" · "그릇 덮기 사건은 안해도 될 듯"
#
#   bash z_training/data/run_gen_phys3d_perm_v2.sh                       # 기본: 학습 500 · 검증 20 / 시나리오 (48)
#   TRAIN_PER=100 VAL_PER=5 OUT=/tmp/x bash z_training/data/run_gen_phys3d_perm_v2.sh
#
# v1 과 다른 것 (gen_phys3d_perm_blender.py 의 PHYS3D_PERM_V2 갈고리):
#   · 움직이는 물체 그림자 끔 (원칙 6 ①) · 미끼 가림막 1~2 개, 클립 70 % (원칙 6 ②) · 앵커 가림 중 끝까지 안 나타나는 것 (원칙 7, 시험 25 %)
#   · 카메라 렌즈 29 → 45 mm (물체 반지름 중앙 15 → 22 px, VOBench 26)
# 속도 (2026-10-31 실측): 렌더가 병목이었다 — GPU 를 학습과 나눠 쓰면 장당 1 초 (혼자면 0.02 초). → CPU 렌더 워커 (nice 15) 가 같은 대기열을 함께 비우고,
#   GPU 렌더 워커는 GPU 마다 RENDER_PER 개, Blender 가 바로 mp4 를 쓴다 (PNG → ffmpeg 없음). 시험 144 편 4.5 분 (라벨 128 · CPU 렌더 96 · GPU 렌더 4 × 4).
set -e
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
export PHYS3D_PERM_V2=1 PHYS3D_LENS=${PHYS3D_LENS:-45} PHYS3D_GPU_DENOISER=OPENIMAGEDENOISE PHYS3D_DIRECT_MP4=1
PY=${PY:-python3}
OUT=${OUT:-/workspace/dataset/training_data/phys3d_perm_v2}
RENDER_GPUS=${RENDER_GPUS-0,1,2,3,4,5,6,7}
RENDER_PER=${RENDER_PER:-4}
CPU_RENDER=${CPU_RENDER:-96}
CPU_WORKERS=${CPU_WORKERS:-160}
TRAIN_PER=${TRAIN_PER:-500}
VAL_PER=${VAL_PER:-20}
ARGS="--multi --perm --defer-render --render-per $RENDER_PER --gpus=$RENDER_GPUS --cpu-render $CPU_RENDER --cpu-workers $CPU_WORKERS --nice 5"
mkdir -p $OUT
v=$("$BLENDER" --version | head -1)
echo "$v | python $($PY -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)') | lens $PHYS3D_LENS" | tee $OUT/versions.txt
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/val --per-scenario $VAL_PER --seed-base 2000000 2>&1 | tee -a $OUT/gen_val.log
$PY z_training/data/audit_phys3d_multi.py --root $OUT/val > $OUT/val/audit.md || true
[ "$VAL_ONLY" = 1 ] && exit 0
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/train --per-scenario $TRAIN_PER --seed-base 3000000 2>&1 | tee -a $OUT/gen_train.log
$PY z_training/data/audit_phys3d_multi.py --root $OUT/train > $OUT/train/audit.md || true
