#!/bin/bash
# 2026-11-07 사용자 "딱 mash-vlm 처럼 … 일단 학습 없이 먼저 확인": MASH-VLM Harmonic-RoPE 를 시간 축에 — 시간 RoPE 회전 짝 일부만 회전 (고유 번호),
#   나머지 짝은 회전 안 함 (균형 번호 = 모든 블록 같은 시간). 학습 없음. model.predictor.time_pairs (src/models/utils/modules.py set_time_pairs).
#   짝 (원 코드 짝 버그 그대로): 0=(1, .158) 1=(.025, .004) 2=(.0006, 1) 3=(.158, .025) 4=(.004, .0006) [블록당 rad]
#   비교: 원본 (모든 짝) · 시간 전부 제거 (_tr_s0, 이미 있음).
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106; G="0 1 2 3 4 5 6 7"
for tp in 0-2-4 1-3; do
  sfx=_tp${tp//-/}
  for m in vast_ar_vw_state vast_ar_vw_state_v4bc4; do
    for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
      SET="model.predictor.time_pairs=$tp" SUFFIX=$sfx GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/harmonic_${d}_${m}${sfx}.log 2>&1
      echo "$(TZ=Asia/Seoul date '+%T KST') $m $d $tp exit $?"
    done
    SET="model.predictor.time_pairs=$tp" SUFFIX=$sfx DS=intphys1_late_uni GPU_IDS="$G" bash cvpr/eval/intphys1_late/run.sh $m > $L/harmonic_ip1late_${m}${sfx}.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $m intphys1_late $tp exit $?"
  done
done
echo HARMONIC_TIME_DONE
