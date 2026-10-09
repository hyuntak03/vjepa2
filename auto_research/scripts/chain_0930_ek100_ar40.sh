#!/bin/bash
# 2026-09-30 EK100 action anticipation probe 를 AR ep40 predictor 로 학습 (released 규약 · grid8 head), vll1 (EK100 66 GB 를 NFS tar 에서 /local_datasets 로).
set -uo pipefail
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT
LOC=/local_datasets; STG=$ROOT/auto_research/_stage
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
[[ -d $LOC/EPIC-KITCHENS_resized/P01 ]] || { log "untar EK100"; tar xf $STG/epic_kitchens_resized.tar -C $LOC || exit 1; }
df -h / | tail -1
export CK=${CK:-$ROOT/z_training/ariel_eval/ckpt/ar_ep40.pt}
COMMON="data.base_path=$LOC/EPIC-KITCHENS_resized model_kwargs.predictor_checkpoint=$CK"
log "smoke (GPU 1)"
LOCAL=1 GPUS=1 SMOKE=1 TAG=ek100_ar40_smoke SET="$COMMON" bash z_research/anticipation/EK100/train_vith.sh released > $STG/results_ar/ek100_ar40_smoke.log 2>&1 || { tail -30 $STG/results_ar/ek100_ar40_smoke.log; exit 1; }
log "smoke ok → 본 학습 (GPU 8, grid8)"
LOCAL=1 GPUS=8 HEADS=grid8 TAG=ek100_vith_official256_released_grid8_ar40 SET="$COMMON" bash z_research/anticipation/EK100/train_vith.sh released > $STG/results_ar/ek100_ar40_train.log 2>&1
log "exit $?  CHAIN DONE"
