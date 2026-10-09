#!/bin/bash
# TrainingEffects — 새로 학습한 모델들의 표준 채점을 lane 별로 돈다 (2026-09-25).
#   v11     : surprise_c16t32 × v11_split_test (학습 안 한 block 절반, 10,752 쌍)
#   IntPhys1: intphys1_sliding × intphys1_dev, 창(C+M) 16 / 32 를 따로 (Garrido A.8, PROTOCOLS.md §2)
# 결과 폴더는 기존 관례 그대로 — v11 은 IntPhysGenV11/exp_results, IntPhys1 은 Benchmarks/exp_results.
# 이미 summary.json 이 있는 칸은 건너뛴다. 집계는 z_research/TrainingEffects/ 의 스크립트가 한다.
#
#   bash z_research/scripts/analysis/training_effects_run.sh            # vll5 8 GPU 할당 안에서
#   LANES="lane7" bash z_research/scripts/analysis/training_effects_run.sh   # 일부 lane 만
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
LOG=z_research/TrainingEffects/logs; mkdir -p "$LOG"
BX=z_research/Benchmarks/exp_results
VX=z_research/IntPhysGenV11/exp_results

v11() {  # model gpu_ids port
  local m=$1 out=$VX/surprise_c16t32__v11_split_test_$1
  if [[ -f $out/summary.json ]]; then echo "$(date +%T) skip v11 $m"; return; fi
  echo "$(date +%T) start v11 $m (gpu $2)"
  GPU_IDS="$2" EVAL_DDP_PORT=$3 WMA_BAR=off bash z_research/scripts/run.sh surprise_c16t32 v11_split_test "$m" \
    > "$LOG/v11_$m.log" 2>&1 && echo "$(date +%T) done v11 $m" || echo "$(date +%T) FAIL v11 $m"
}
ip1() {  # model window gpu_ids port
  local m=$1 w=$2 out=$BX/intphys1_sliding__intphys1_dev_$1_w$2
  if [[ -f $out/summary.json ]]; then echo "$(date +%T) skip ip1 $m w$w"; return; fi
  echo "$(date +%T) start ip1 $m w$w (gpu $3)"
  GPU_IDS="$3" EVAL_DDP_PORT=$4 WMA_BAR=off TAG=intphys1_dev_${m}_w$w OUTDIR=$out \
    SET="surprise.intphys1.window_sizes=[$w] model.window_size=$w" \
    bash z_research/scripts/run.sh intphys1_sliding intphys1_dev "$m" \
    > "$LOG/ip1_${m}_w$w.log" 2>&1 && echo "$(date +%T) done ip1 $m w$w" || echo "$(date +%T) FAIL ip1 $m w$w"
}
both() { v11 "$1" "$2" "$3"; ip1 "$1" 16 "$2" "$3"; ip1 "$1" 32 "$2" "$3"; }

lane0() { v11 vith_ariel_bc_ep43 "0 1 2 3" 29611
          both vittiny_synphys_30k_e225 "0 1 2 3" 29611
          both vittiny_k400ssv2_30k_e225 "0 1 2 3" 29611; }
lane4() { both vitb_k400_e240 4 29614; }
lane5() { both vitb_synphys_30k 5 29615; }
lane6() { both vitb_k400_e100 6 29616; }
lane7() { both vittiny_k400 7 29617; both vittiny_k400_5k 7 29617; both vittiny_synphys_5k 7 29617; }

LANES=${LANES:-"lane0 lane4 lane5 lane6 lane7"}
for l in $LANES; do $l > "$LOG/$l.out" 2>&1 & done
wait
echo "$(date +%T) ALL LANES DONE"; cat "$LOG"/lane*.out
