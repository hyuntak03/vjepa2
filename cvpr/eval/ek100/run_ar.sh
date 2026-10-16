#!/bin/bash
# EK100 action anticipation — kind=ar predictor 판 (2026-10-14, vast). 엔진·프로토콜은 cvpr/eval/ek100 그대로,
# 모듈만 vit_encoder_ar_predictor_concat (encoder 출력 + AR 이 굴린 1 s 뒤 tubelet).
#   bash cvpr/eval/ek100/run_ar.sh <run 이름> [ar_feedback=state]
#   기본: 전체 batch 512 (rank 64 × 8 GPU), head 8 개 (HEADS=b512h8), 20 epoch, num_workers 12
# 결과: cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vith_vast_<run>_b512_h8/val_metrics.jsonl
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
RUN=${1:?run 이름 (z_training/runs/<run>)}; FB=${2:-state}
CK="$REPO/z_training/runs/$RUN/latest.pt"; [[ -f $CK ]] || { echo "없음: $CK"; exit 1; }
SET="model_kwargs.module_name=evals.action_anticipation_frozen.modelcustom.vit_encoder_ar_predictor_concat \
model_kwargs.pretrain_kwargs.predictor_ar.checkpoint=$CK model_kwargs.pretrain_kwargs.predictor_ar.ar_feedback=$FB \
experiment.optimization.batch_size=${BS:-64} experiment.data.num_workers=${NW:-12} ${SET:-}" \
SUFFIX=${SUFFIX:-_vast_${RUN}_b512_h8} HEADS=${HEADS:-b512h8} GPUS=${GPUS:-8} bash "$HERE/run.sh" ek100 vith
