#!/bin/bash
# 2026-11-05 tak's_story 요청 (splice 서명 대조): 릴리즈 계열을 '미래만 정답' (target encoder 가 미래 16 장만 본다) 으로 다시 채점 + 같은 공간의 복사 기준선. 추론만, GPU 4 장.
# AR 판은 채점 정답이 원래 블록별 (블록 = 자기 2 장만) 이라 다시 채점하지 않는다.
cd /workspace/world_model/vjepa2
L=z_analysis/results/scene_paired_20261105
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
# 0) 검사: 전체 창 정답 + 복사 기준선 켬 → 'all' 이 기존 채점과 같아야 한다
SET="surprise.copy_baseline=true $TS" SUFFIX=_copychk GPU_IDS="0 1 2 3" GPUS=4 bash cvpr/eval/testbed/run.sh intphys1_surrogate vith > $L/eval_copychk.log 2>&1; echo "$(TZ=Asia/Seoul date '+%T KST') copychk exit $?"
for m in vith vith_release_v4vc vith_release_v4vc_negimp; do
  for d in intphys1_surrogate vobench_v3 real_world_permanence_uniform; do
    X="surprise.target_future_only=true surprise.copy_baseline=true"; [[ $d != real_world_permanence_uniform ]] && X="$X $TS"
    SET="$X" SUFFIX=_futonly GPU_IDS="0 1 2 3" GPUS=4 bash cvpr/eval/testbed/run.sh $d $m > $L/eval_futonly_${d}_${m}.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $d $m exit $?"
  done
done
echo FUTONLY_EVAL_DONE
