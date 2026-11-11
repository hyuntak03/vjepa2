#!/bin/bash
# 2026-11-06 release → AR 사다리 R1–R3 (학습 없음, release 가중치 고정, 채점 정답은 창 전체 그대로). trinity 세션과 분담 (vast = R1–R3).
#   R1 = 앞만 보는 문맥 인코딩 (model.ctx_encoding=causal), R2 = 블록마다 따로 (block),
#   R3a = 문맥 encoder 를 target encoder 로 (model.context_encoder_key=target_encoder), R3b = R2 + target encoder.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106
G="0 1 2 3 4 5 6 7"
for row in "R1:_ctxcausal:model.ctx_encoding=causal" "R2:_ctxblock:model.ctx_encoding=block" "R3a:_ctxtgt:model.context_encoder_key=target_encoder" "R3b:_ctxblocktgt:model.ctx_encoding=block model.context_encoder_key=target_encoder"; do
  IFS=: read tag suf set <<<"$row"
  for d in vobench_v3 intphys1_surrogate intphys2_late_final real_world_permanence_uniform; do
    SET="$set" SUFFIX=$suf GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d vith > $L/eval_${tag}_$d.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $tag $d exit $?"
  done
  SET="$set" SUFFIX=$suf DS=intphys1_late_uni GPU_IDS="$G" GPUS=8 bash cvpr/eval/intphys1_late/run.sh vith > $L/eval_${tag}_intphys1_late.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $tag intphys1_late exit $?"
done
echo LADDER_R1R3_DONE
