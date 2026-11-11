#!/bin/bash
# 2026-11-06 cvpr_auto 요청 2 차 (사용자 지시 "vast 한테도 전달해서 실험 쭉") — predictor 사다리 기본 채점 + 끊기 T1 · T2 + 인과 문맥 (R1). 학습 없음.
#   모든 실행에 surprise.per_future_block=true (미래 블록 9..16 별 surprise → 블록별 짝 정답률).
#   T1 (_kopast) = 블록 8 이후 모든 query 가 문맥 블록 1–7 을 못 본다. T2 (_kolast) = 모든 query 가 블록 8 을 못 본다 (한 번에 예측하는 판만).
#   R1 (_ctxcausal) = 문맥 블록 t 표현 = 블록 1..t 만 encoder 에 (영상 문맥 판만: 릴리즈 · vidctx · vidctx_causal). R1 + T1 은 판 기본 + T1 과
#   수학적으로 같아서 (T1 이 블록 1–7 을 예측 경로에서 빼고, 블록 8 토큰은 인과 · 한 번에 인코딩이 같다) 돌리지 않는다 — 시험 8 영상 일치 확인.
#   끊기 = analysis/attention/knockout_sdpa.py. 앞 묶음 (run_predictor_ladder_ko.sh) 은 미래 블록별 출력이 없어 20:19 KST 에 멈추고 이것으로 대체.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106
G="0 1 2 3 4 5 6 7"
ONESHOT="vith vast_tube_cd64 vast_ar_oneshot_vidctx vast_ar_oneshot_vidctx_causal vast_ar_oneshot_causal"
AR="vast_ar_vw_state vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negpair"
VID="vith vast_ar_oneshot_vidctx vast_ar_oneshot_vidctx_causal"
FB="surprise.per_future_block=true"
run() {  # run <bench> <model> <suffix> <SET>
  FORCE=1 SET="$FB $4" SUFFIX=$3 GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $1 $2 > $L/v2_${1}_${2}$3.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $1 $2 $3 exit $?"
}
for d in vobench_v3 vobench_v1_perm intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
  for m in $ONESHOT $AR; do
    run $d $m _fb ""
    run $d $m _kopast "surprise.attn_knockout=past"
    [[ " $ONESHOT " == *" $m "* ]] && run $d $m _kolast "surprise.attn_knockout=last"
  done
  for m in $VID; do
    run $d $m _ctxcausal_fb "model.ctx_encoding=causal"
    run $d $m _ctxcausal_kolast "model.ctx_encoding=causal surprise.attn_knockout=last"
  done
done
echo PREDICTOR_LADDER_V2_DONE
