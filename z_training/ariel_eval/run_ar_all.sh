#!/usr/bin/env bash
# AR 새 판 하나를 세 벤치마크로 채점한다 (2026-09-28). GPU 8 장 (job 안).
#   EP=38 bash z_training/ariel_eval/run_ar_all.sh
#   GPU 7        : IntPhys2 w16 (가장 긴 단일 GPU 작업이라 먼저)
#   GPU 0-6      : IntPhys1 (w16 0-3 · w32 4-6) -> v11_split_test (0-6) -> IntPhys2 w48 (0-3) · w32 (4-6)
# 모델 = configs/protocols/models.md 의 vith_ariel_ar_ep${EP} (고정 사본 z_training/ariel_eval/ckpt/ar_ep${EP}.pt)
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
EP=${EP:?EP 를 줄 것}; m=vith_ariel_ar_ep$EP; CK=$PWD/z_training/ariel_eval/ckpt/ar_ep$EP.pt
L=z_training/ariel_eval; BX=z_research/Benchmarks/exp_results
[[ -f $CK ]] || { echo "사본 없음 $CK"; exit 1; }
echo "$(date +%T) 시작 $m"
[[ ${SKIP_IP2_W16:-0} == 1 ]] || CK=$CK ARTAG=ar_ep$EP GPU_W16=7 bash $L/run_ip2_ar.sh &
ip1() { w=$1; g="$2"; p=$3
  GPU_IDS="$g" EVAL_DDP_PORT=$p WMA_BAR=off TAG=intphys1_dev_${m}_w$w OUTDIR=$BX/intphys1_sliding__intphys1_dev_${m}_w$w \
  SET="surprise.intphys1.window_sizes=[$w] model.window_size=$w" \
  bash z_research/scripts/run.sh intphys1_sliding intphys1_dev $m > $L/ip1_${m}_w$w.log 2>&1; echo "$(date +%T) ip1 w$w exit $?"; }
[[ ${SKIP_IP1:-0} == 1 ]] || { ip1 16 "0 1 2 3" 29791 & ip1 32 "4 5 6" 29792 & }
while pgrep -f "[r]un.sh intphys1_sliding intphys1_dev $m" >/dev/null; do sleep 5; done
GPU_IDS="0 1 2 3 4 5 6" EVAL_DDP_PORT=29793 WMA_BAR=off bash z_research/scripts/run.sh surprise_c16t32 v11_split_test $m > $L/v11_$m.log 2>&1
echo "$(date +%T) v11 exit $?"
CK=$CK ARTAG=ar_ep$EP GPU_W48=0,1,2,3 GPU_W32=4,5,6 bash $L/run_ip2_ar.sh
while pgrep -f "[b]ench_intphys2_main_ariel_ar_ep${EP}_" >/dev/null; do sleep 15; done
echo "$(date +%T) ALL DONE"
