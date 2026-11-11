#!/bin/bash
# 2026-11-06: 교란 가르기 — 릴리즈 predictor + 인과 문맥 (R1, model.ctx_encoding=causal) 에도 끊기 T1 · T2.
#   릴리즈 T1 은 문맥 encoder 가 16 장을 한 번에 봐서 블록 8 토큰에 과거가 이미 섞여 있다 → 인과 문맥판에서 T1 이 무너지는지로 가른다.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106
G="0 1 2 3 4 5 6 7"
until grep -q PREDICTOR_LADDER_KO_DONE $L/run_predictor_ladder_ko.log; do sleep 60; done
for d in vobench_v3 vobench_v1_perm intphys1_surrogate real_world_permanence_uniform; do
  SET="model.ctx_encoding=causal" SUFFIX=_ctxcausal GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d vith > $L/r1ko_${d}_base.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') R1 $d base exit $?"
  for ko in past last; do
    SET="model.ctx_encoding=causal surprise.attn_knockout=$ko" SUFFIX=_ctxcausal_ko$ko GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d vith > $L/r1ko_${d}_$ko.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') R1 $d $ko exit $?"
  done
done
echo R1_KO_DONE
