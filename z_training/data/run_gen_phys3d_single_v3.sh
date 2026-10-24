#!/bin/bash
# phys3d_single_v3 (2026-10-23) — v2 + 놓기 전 공중에 떠 있지 않음 (내려놓기 · 바닥에서 던지기 · 막대) · 렌더 · 저장 4 fps (학습 stride 1). 끊기면 같은 명령으로 이어 한다.
#
#   TRAIN_PER=<시나리오당 train 편 수> BLENDER=<blender 4.2.3> OUT=<저장 위치> CPU_WORKERS=<n> bash z_training/data/run_gen_phys3d_single_v3.sh
#
# 렌더는 CPU 만 (픽셀 재현). 시험 (169 편, 2026-10-23): 영상당 325 초 (v2 704 초) → 216 코어에서 TRAIN_PER 1200 × 13 ≈ 6.5 시간.
# 문서 z_training/docs/PHYS3D_SINGLE_V3_DATASET_2026-10-23.md
set -e
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
PY=${PY:-python3}
OUT=${OUT:-/workspace/dataset/training_data/phys3d_single_v3}
RENDER_GPUS=${RENDER_GPUS-}                     # ⚠️ GPUS 가 아니다 — Vast 기계에는 GPUS=8 (장 수) 이 이미 있다
GPU_PER=${GPU_PER:-2}
CPU_WORKERS=${CPU_WORKERS:-$(( $(nproc) - 40 ))}
TRAIN_PER=${TRAIN_PER:-1200}
VAL_PER=${VAL_PER:-80}
REF=z_training/data/fingerprints/phys3d_single_v3_val.tsv
ARGS="--single-v3 --stride 1 --frames 288 --gpus=$RENDER_GPUS --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS"
mkdir -p $OUT
v=$("$BLENDER" --version | head -1)
echo "$v | python $($PY -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')" | tee $OUT/versions.txt
case "$v" in *"Blender 4.2.3"*) ;; *) [ "$ALLOW_VERSION_MISMATCH" = 1 ] || { echo "Blender 4.2.3 이 아니다 — ALLOW_VERSION_MISMATCH=1 로 무시"; exit 1; } ;; esac
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/val   --per-scenario $VAL_PER --seed-base 1000000 2>&1 | tee -a $OUT/gen_val.log
$PY z_training/data/audit_phys2d_single.py --three-d --stride 1 --root $OUT/val > $OUT/val/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/val --write $OUT/val/fingerprint.tsv
if [ -f $REF ]; then $PY z_training/data/fingerprint_phys_data.py --root $OUT/val --check $REF | tee $OUT/val/fingerprint_check.txt; fi
[ "$VAL_ONLY" = 1 ] && exit 0
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/train --per-scenario $TRAIN_PER 2>&1 | tee -a $OUT/gen_train.log
$PY z_training/data/audit_phys2d_single.py --three-d --stride 1 --root $OUT/train > $OUT/train/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/train --write $OUT/train/fingerprint.tsv
