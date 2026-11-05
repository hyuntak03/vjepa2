#!/bin/bash
# 2026-11-05 tak's_story 요청 ①: 물체 영역 (투영 중심 3×3) vs 먼 배경 (중심에서 3 칸 이상) 으로 나눈 surprise. 추론만, GPU 4 장.
cd /workspace/world_model/vjepa2
S='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
for m in vith_release_v4vc vith_release_v4vc_negimp vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negimp; do
  for d in intphys1_surrogate vobench_v3; do
    SET="$S" SUFFIX=_region GPU_IDS="0 1 2 3" GPUS=4 bash cvpr/eval/testbed/run.sh $d $m > z_analysis/results/scene_paired_20261105/eval_region_${d}_${m}.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $d $m exit $?"
  done
done
echo REGION_EVAL_DONE
