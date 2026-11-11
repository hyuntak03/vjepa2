#!/bin/bash
# 2026-11-06 사용자 "최신으로 ㄱ": v4b 다섯 판 최신 체크포인트 (복사본 /tmp/claude-0/-workspace-world-model-vjepa2/17011fb5-0381-4a91-8527-2cfdb89569ed/scratchpad/ckpt) 를 VOBench v3 로 중간 채점. 채점만.
cd /workspace/world_model/vjepa2
for pair in v4b_ar_vw_state_postft:vast_ar_vw_state_v4b:e11 v4b_ar_vw_state_negimp:vast_ar_vw_state_v4b_negimp:e10 v4b_ar_vw_state_negpair_w3:vast_ar_vw_state_v4b_negpair_w3:e9 v4b_release_postft:vith_release_v4b:e14 v4b_release_negpair_w3:vith_release_v4b_negpair_w3:e9; do
  IFS=: read R M E <<<"$pair"
  SET="model.predictor_checkpoint=/tmp/claude-0/-workspace-world-model-vjepa2/17011fb5-0381-4a91-8527-2cfdb89569ed/scratchpad/ckpt/${R}_latest.pt" SUFFIX=_$E GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v3 $M > z_training/logs/vob3_${E}_$M.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $M $E exit $?"
done
echo VOB3_LATEST_DONE
