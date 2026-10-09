#!/usr/bin/env bash
# ctx_ar 판 하나를 IntPhys1 (w16 · w32) + v11_split_test 로 채점 (2026-09-29, 사용자 요청). GPU 8 장 job 안에서.
#   EP=18 bash z_training/ariel_eval/run_ctx_ar_ip1_v11.sh
# 모델 = configs/protocols/models.md 의 vith_ariel_ctx_ar_ep${EP} (고정 사본 z_training/ariel_eval/ckpt/ctx_ar_ep${EP}.pt)
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
EP=${EP:?EP 를 줄 것}; m=vith_ariel_ctx_ar_ep$EP; CK=$PWD/z_training/ariel_eval/ckpt/ctx_ar_ep$EP.pt
L=z_training/ariel_eval; BX=z_research/Benchmarks/exp_results
[[ -f $CK ]] || { echo "사본 없음 $CK"; exit 1; }
echo "$(date +%T) 시작 $m"
# 스모크 (GPU 0, 2 영상) — 경로 · kind · 되먹임 모양 검사. 실패하면 멈춘다.
GPU_IDS="0" EVAL_DDP_PORT=29780 WMA_BAR=off LIMIT=2 SET="surprise.intphys1.window_sizes=[16] model.window_size=16" \
  bash z_research/scripts/run.sh intphys1_sliding intphys1_dev $m > $L/ip1_${m}_smoke.log 2>&1 || { echo "smoke 실패"; tail -30 $L/ip1_${m}_smoke.log; exit 1; }
echo "$(date +%T) smoke ok"
ip1() { w=$1; g="$2"; p=$3
  GPU_IDS="$g" EVAL_DDP_PORT=$p WMA_BAR=off TAG=intphys1_dev_${m}_w$w OUTDIR=$BX/intphys1_sliding__intphys1_dev_${m}_w$w \
  SET="surprise.intphys1.window_sizes=[$w] model.window_size=$w" \
  bash z_research/scripts/run.sh intphys1_sliding intphys1_dev $m > $L/ip1_${m}_w$w.log 2>&1; echo "$(date +%T) ip1 w$w exit $?"; }
ip1 16 "0 1 2 3" 29791 & ip1 32 "4 5 6 7" 29792 &
wait
GPU_IDS="0 1 2 3 4 5 6 7" EVAL_DDP_PORT=29793 WMA_BAR=off bash z_research/scripts/run.sh surprise_c16t32 v11_split_test $m > $L/v11_$m.log 2>&1
echo "$(date +%T) v11 exit $?"
echo "$(date +%T) ALL DONE"
