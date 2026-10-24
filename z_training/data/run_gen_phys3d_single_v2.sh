#!/bin/bash
# phys3d_single_v2 (2026-10-21) — 사건 창 C · K ∈ {2..32} (stride 3) · 이어지는 안 보임 ≤ 45 장 · 최대 288 장. 끊기면 같은 명령으로 이어 한다.
#
#   TRAIN_PER=<시나리오당 train 편 수> BLENDER=<blender 4.2.3> OUT=<저장 위치> CPU_WORKERS=<n> bash z_training/data/run_gen_phys3d_single_v2.sh
#
# 렌더는 CPU 만 (픽셀 재현). 클립 평균 228 장 · 장당 3.5 core-s → 216 코어에서 시간당 ~1,000 편 (TRAIN_PER 1200 × 13 ≈ 16 시간).
# 문서 z_training/docs/PHYS3D_SINGLE_V2_DATASET_2026-10-21.md
set -e
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
PY=${PY:-python3}
OUT=${OUT:-/workspace/dataset/training_data/phys3d_single_v2}
RENDER_GPUS=${RENDER_GPUS-}                     # ⚠️ GPUS 가 아니다 — Vast 기계에는 GPUS=8 (장 수) 이 이미 있다
GPU_PER=${GPU_PER:-2}
CPU_WORKERS=${CPU_WORKERS:-$(( $(nproc) - 40 ))}
TRAIN_PER=${TRAIN_PER:-1200}
VAL_PER=${VAL_PER:-80}
REF=z_training/data/fingerprints/phys3d_single_v2_val.tsv
ARGS="--single-v2 --stride 3 --frames 288 --gpus=$RENDER_GPUS --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS"
mkdir -p $OUT
v=$("$BLENDER" --version | head -1)
echo "$v | python $($PY -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')" | tee $OUT/versions.txt
case "$v" in *"Blender 4.2.3"*) ;; *) [ "$ALLOW_VERSION_MISMATCH" = 1 ] || { echo "Blender 4.2.3 이 아니다 — ALLOW_VERSION_MISMATCH=1 로 무시"; exit 1; } ;; esac
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/val   --per-scenario $VAL_PER --seed-base 1000000 2>&1 | tee -a $OUT/gen_val.log
$PY z_training/data/audit_phys2d_single.py --three-d --stride 3 --root $OUT/val > $OUT/val/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/val --write $OUT/val/fingerprint.tsv
if [ -f $REF ]; then $PY z_training/data/fingerprint_phys_data.py --root $OUT/val --check $REF | tee $OUT/val/fingerprint_check.txt; fi
[ "$VAL_ONLY" = 1 ] && exit 0
$PY z_training/data/gen_phys3d.py $ARGS --out $OUT/train --per-scenario $TRAIN_PER 2>&1 | tee -a $OUT/gen_train.log
$PY z_training/data/audit_phys2d_single.py --three-d --stride 3 --root $OUT/train > $OUT/train/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/train --write $OUT/train/fingerprint.tsv
