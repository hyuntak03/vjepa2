#!/bin/bash
# 2026-11-06 사용자 "(1) 해봐 지금": 블록 AR 에 시간 순 인코딩 입력 (model.ctx_encoding=causal) — predictor · 블록별 정답 그대로, 학습 없음.
#   크게 오르면 병목 = predictor 가 이력을 스스로 못 쌓음. 그대로면 꺼내 쓰기 (최근 편향) · 학습량 쪽.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106; G="0 1 2 3 4 5 6 7"
for m in vast_ar_vw_state vast_ar_vw_state_v4bc4_negimp; do
  for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
    SET="model.ctx_encoding=causal" SUFFIX=_ctxcausal GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/arcausal_${d}_$m.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $m $d exit $?"
  done
  SET="model.ctx_encoding=causal" SUFFIX=_ctxcausal DS=intphys1_late_uni GPU_IDS="$G" GPUS=8 bash cvpr/eval/intphys1_late/run.sh $m > $L/arcausal_ip1late_$m.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $m intphys1_late exit $?"
done
echo AR_CAUSAL_INPUT_DONE
