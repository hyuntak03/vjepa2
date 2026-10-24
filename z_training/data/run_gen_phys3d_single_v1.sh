#!/bin/bash
# phys3d_single_v1 (2026-10-19) — 물체 1 개 · 시간 고른 가림 (3D, Blender). train 13 시나리오 × 2200 · val × 80. 끊기면 같은 명령으로 이어 한다.
#
#   BLENDER=<blender 4.2.3> OUT=<저장 위치> CPU_WORKERS=<n> bash z_training/data/run_gen_phys3d_single_v1.sh
#
# 재현: 라벨은 (코드, seed, Blender 4.2.3, numpy) 로 정해진다 — 워커 수 · 기계 · 실행 순서 · PYTHONHASHSEED 와 무관 (시험 52 편 일치).
#   렌더는 CPU (Cycles 스레드 1) 만 쓴다 — GPU (OptiX) 와 CPU 는 픽셀이 미세하게 달라서. RENDER_GPUS=2,3 처럼 주면 GPU 워커도 쓴다 (픽셀 재현 포기).
#   끝나면 지문을 만들고, val 은 레포의 기준 지문과 대조한다.
set -e
cd "$(dirname "$0")/../.."
export BLENDER=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
export PYTHONHASHSEED=0
PY=${PY:-python3}
OUT=${OUT:-/workspace/dataset/training_data/phys3d_single_v1}
RENDER_GPUS=${RENDER_GPUS-}                     # ⚠️ GPUS 가 아니다 — Vast 기계에는 GPUS=8 (장 수) 이 이미 있어 GPU 8 워커가 생겼다
GPU_PER=${GPU_PER:-2}
CPU_WORKERS=${CPU_WORKERS:-$(( $(nproc) - 40 ))}
REF=z_training/data/fingerprints/phys3d_single_v1_val.tsv
mkdir -p $OUT
v=$("$BLENDER" --version | head -1)
echo "$v | python $($PY -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')" | tee $OUT/versions.txt
case "$v" in *"Blender 4.2.3"*) ;; *) [ "$ALLOW_VERSION_MISMATCH" = 1 ] || { echo "Blender 4.2.3 이 아니다 — ALLOW_VERSION_MISMATCH=1 로 무시"; exit 1; } ;; esac
$PY z_training/data/gen_phys3d.py --single --out $OUT/val   --per-scenario 80   --seed-base 1000000 --gpus "$RENDER_GPUS" --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS 2>&1 | tee -a $OUT/gen_val.log
$PY z_training/data/gen_phys3d.py --single --out $OUT/train --per-scenario 2200                     --gpus "$RENDER_GPUS" --gpu-per $GPU_PER --cpu-workers $CPU_WORKERS 2>&1 | tee -a $OUT/gen_train.log
$PY z_training/data/audit_phys2d_single.py --three-d --root $OUT/val   > $OUT/val/audit.md
$PY z_training/data/audit_phys2d_single.py --three-d --root $OUT/train > $OUT/train/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/val   --write $OUT/val/fingerprint.tsv
$PY z_training/data/fingerprint_phys_data.py --root $OUT/train --write $OUT/train/fingerprint.tsv
if [ -f $REF ]; then $PY z_training/data/fingerprint_phys_data.py --root $OUT/val --check $REF | tee $OUT/val/fingerprint_check.txt; fi
