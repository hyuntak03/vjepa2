#!/bin/bash
# 2026-11-05 tak's_story 세션 (vll5 논문) 요청: WROP C (+ 같은 엔진 대조 D) 를 시트 보드로 채점. 추론만 (학습 없음), GPU 4 장 (학습 · EK100 과 겹침).
cd /workspace/world_model/vjepa2
for m in vast_permwrop_C vast_permwrop_D; do
  for d in vobench_v3 intphys1_surrogate intphys2_late_final real_world_permanence_uniform; do
    GPU_IDS="0 1 2 3" GPUS=4 bash cvpr/eval/testbed/run.sh $d $m > z_analysis/results/scene_paired_20261105/eval_${d}_${m}.log 2>&1; echo "$(TZ=Asia/Seoul date '+%T KST') $d $m exit $?"
  done
done
echo WROP_CD_EVAL_DONE
