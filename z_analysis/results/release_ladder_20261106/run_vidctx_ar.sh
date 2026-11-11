#!/bin/bash
# 2026-11-06 cvpr_auto (사용자 확인): 통째 입력 + 한 걸음씩 판 — vast_ar_vw_state_vidctx (natural_ar_vw_state_vidctx, e65) 를 같은 사다리 꼴로.
#   기본 _fb · T1 _kopast · R1 _ctxcausal_fb. T2 는 AR 이라 구조상 해당 없음 (블록 9 query = 블록 8 토큰 자신). R1 + T1 ≡ T1 이라 안 돌림.
#   run_predictor_ladder_v2.sh 가 끝난 뒤 돈다.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106
G="0 1 2 3 4 5 6 7"; M=vast_ar_vw_state_vidctx; FB="surprise.per_future_block=true"
until grep -q PREDICTOR_LADDER_V2_DONE $L/run_predictor_ladder_v2.log; do sleep 60; done
run() { FORCE=1 SET="$FB $3" SUFFIX=$2 GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $1 $M > $L/v2_${1}_${M}$2.log 2>&1; echo "$(TZ=Asia/Seoul date '+%T KST') $1 $M $2 exit $?"; }
for d in vobench_v3 vobench_v1_perm intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
  run $d _fb ""
  run $d _kopast "surprise.attn_knockout=past"
  run $d _ctxcausal_fb "model.ctx_encoding=causal"
done
echo VIDCTX_AR_DONE
