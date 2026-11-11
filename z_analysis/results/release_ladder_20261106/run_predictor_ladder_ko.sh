#!/bin/bash
# 2026-11-06 cvpr_auto 요청 (사용자 지시 "vast 한테도 전달해서 실험 쭉") — predictor 사다리 기본 채점 + 끊기 T1 · T2. 학습 없음.
#   T1 (_kopast) = 블록 8 이후의 모든 query 가 문맥 블록 1–7 을 못 본다. T2 (_kolast) = 모든 query 가 블록 8 을 못 본다 (한 번에 예측하는 판만;
#   AR 은 블록 9 query 가 블록 8 자신이라 구조상 해당 없음). 끊기 = analysis/attention/knockout_sdpa.py (surprise.attn_knockout).
#   기본 채점은 이미 있으면 하네스가 건너뛴다. 벤치 순서 = 신호가 갈리는 칸부터.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106
G="0 1 2 3 4 5 6 7"
ONESHOT="vith vast_tube_cd64 vast_ar_oneshot_vidctx vast_ar_oneshot_vidctx_causal vast_ar_oneshot_causal"
AR="vast_ar_vw_state vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negpair"
for d in vobench_v3 vobench_v1_perm intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
  for m in $ONESHOT $AR; do
    for ko in none past last; do
      [[ $ko == last && " $AR " == *" $m "* ]] && continue
      if [[ $ko == none ]]; then
        GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/ko_${d}_${m}_base.log 2>&1
      else
        SET="surprise.attn_knockout=$ko" SUFFIX=_ko$ko GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/ko_${d}_${m}_$ko.log 2>&1
      fi
      echo "$(TZ=Asia/Seoul date '+%T KST') $d $m $ko exit $?"
    done
  done
done
echo PREDICTOR_LADDER_KO_DONE
