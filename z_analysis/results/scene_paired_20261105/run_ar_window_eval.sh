#!/bin/bash
# 2026-11-05 22:30 KST 사용자 "해보던가": AR 판을 '창 전체 정답' (릴리즈 채점 정의, model.predictor.ar_target=window) 으로 다시 채점.
#   질문 — AR 이 합성 보드에서 릴리즈 + v4 보다 잘하는 것이 블록별 정답 공간 덕인가, AR 자체 (블록 문맥 입력 · 굴리기) 덕인가.
#   대조 = 같은 체크포인트의 기본 채점 (블록별 정답, 기존 결과 폴더). 추론만, 학습 없음. GPU 8 장.
cd /workspace/world_model/vjepa2
L=z_analysis/results/scene_paired_20261105
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
for m in vast_ar_vw_state_v4vc vast_ar_vw_state; do
  for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
    X="model.predictor.ar_target=window"; [[ $d == vobench_v3 || $d == intphys1_surrogate ]] && X="$X $TS"
    SET="$X" SUFFIX=_artgtwin GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/eval_artgtwin_${d}_${m}.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $d $m exit $?"
  done
  SET="model.predictor.ar_target=window" SUFFIX=_artgtwin DS=intphys1_late_uni GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/intphys1_late/run.sh $m > $L/eval_artgtwin_intphys1_late_${m}.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') intphys1_late $m exit $?"
done
echo AR_WINDOW_EVAL_DONE
