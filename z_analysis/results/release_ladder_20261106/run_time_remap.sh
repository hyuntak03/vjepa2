#!/bin/bash
# 2026-11-06 사용자: "추론할 때 개입해서 real bench 나 다른 bench 점수 오르는지 봐" — 블록 AR 시간 RoPE 위치 바꾸기 (학습 없음).
#   analysis/attention/time_remap.py · surprise.time_remap. 기준판 (AR v4bc4 λ1) 먼저, 다음 AR 기본.
cd /workspace/world_model/vjepa2
L=z_analysis/results/release_ladder_20261106; G="0 1 2 3 4 5 6 7"
for m in vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state; do
  for row in "_tr_s05:scale:0.5" "_tr_s025:scale:0.25" "_tr_s0:scale:0" "_tr_nov:novelty" "_tr_oracle:oracle"; do
    suf=${row%%:*}; mode=${row#*:}
    for d in vobench_v3 intphys1_surrogate real_world_permanence_uniform intphys2_late_final; do
      [[ $mode == oracle && $d != vobench_v3 && $d != intphys1_surrogate ]] && continue
      SET="surprise.time_remap=$mode" SUFFIX=$suf GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $m > $L/tr_${d}_${m}$suf.log 2>&1
      echo "$(TZ=Asia/Seoul date '+%T KST') $m $d $mode exit $?"
    done
    [[ $mode == oracle ]] && continue
    SET="surprise.time_remap=$mode" SUFFIX=$suf DS=intphys1_late_uni GPU_IDS="$G" GPUS=8 bash cvpr/eval/intphys1_late/run.sh $m > $L/tr_ip1late_${m}$suf.log 2>&1
    echo "$(TZ=Asia/Seoul date '+%T KST') $m intphys1_late $mode exit $?"
  done
done
echo TIME_REMAP_DONE
