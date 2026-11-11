#!/bin/bash
# 2026-11-05 사용자 "release 에 블록 했을 때 성능 별로 안 오르면 괜찮은 거 아닌가": 릴리즈 계열을 AR 의 정답 공간 (블록별) 으로 채점.
#   surprise.target_blockwise=true (eval.py, _ArBlockEncoder) + 같은 공간 복사 기준선. 추론만, GPU 8 장.
#   정합 검사: 복사 기준선은 predictor 를 안 쓰므로 AR 쪽 블록별 복사 (*_arcopyblk) 와 같아야 한다.
cd /workspace/world_model/vjepa2
L=z_analysis/results/scene_paired_20261105
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
for m in vith_release_v4vc vith; do
  for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
    X="surprise.target_blockwise=true surprise.copy_baseline=true"; [[ $d == vobench_v3 || $d == intphys1_surrogate ]] && X="$X $TS"
    SET="$X" SUFFIX=_tgtblk GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/eval_tgtblk_${d}_${m}.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $d $m exit $?"
  done
  SET="surprise.target_blockwise=true surprise.copy_baseline=true" SUFFIX=_tgtblk DS=intphys1_late_uni GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/intphys1_late/run.sh $m > $L/eval_tgtblk_intphys1_late_${m}.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') intphys1_late $m exit $?"
done
echo RELEASE_BLOCK_DONE
