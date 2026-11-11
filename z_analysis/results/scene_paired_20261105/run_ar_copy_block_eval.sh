#!/bin/bash
# 2026-11-05 정답 공간 교차 채점의 대조: 블록별 정답 공간의 복사 기준선 (마지막 문맥 블록 LN(h) 를 미래 블록에 그대로, predictor 안 씀).
#   model.predictor.ar_copy=true · ar_target=block (기본). 체크포인트 무관 (predictor 미사용). 추론만, GPU 8 장.
cd /workspace/world_model/vjepa2
L=z_analysis/results/scene_paired_20261105
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
m=vast_ar_vw_state_v4vc
for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
  X="model.predictor.ar_copy=true"; [[ $d == vobench_v3 || $d == intphys1_surrogate ]] && X="$X $TS"
  SET="$X" SUFFIX=_arcopyblk GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/eval_arcopyblk_${d}.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $d exit $?"
done
SET="model.predictor.ar_copy=true" SUFFIX=_arcopyblk DS=intphys1_late_uni GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/intphys1_late/run.sh $m > $L/eval_arcopyblk_intphys1_late.log 2>&1
echo "$(TZ=Asia/Seoul date '+%T KST') intphys1_late exit $?"
echo AR_COPY_BLOCK_DONE
