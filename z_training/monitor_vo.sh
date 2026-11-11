#!/bin/bash
# =============================================================================
# 시간을 고르기에서 가져오기로 (VO) 세 판 모니터 (2026-11-07). 학습 진행 · 1 epoch 마다 real bench val · 채점 사슬 상태.
#
#   watch -c -n 10 bash z_training/monitor_vo.sh
#
# val = real world permanence 95 장면 × 4 = 380 클립 (190 쌍). "val acc" = 쌍 정확도, 괄호 = A 사라짐 / B 나타남.
# 기준 (학습 전 AR 기본판): real95 58.2 (A 37 / B 79). epoch 0 = 학습 전 (at_start).
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
R=z_training/runs
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
echo "=== VO 세 판 모니터 · $(kst) ==="
for row in "vo_nat_ours:ours (고르기 시간 없음 + 가져오기 시간)" "vo_nat_control:control (구조 그대로)" "vo_nat_notime:notime (고르기 시간 없음)"; do
  run=${row%%:*}; name=${row#*:}
  log=$R/$run/train.log
  echo
  echo "── $name  [$run]"
  if [[ ! -f $log ]]; then echo "   대기 중 (아직 시작 안 함)"; continue; fi
  alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$run/config.yaml")
  last=$(grep -o "\[ *[0-9]*, *[0-9]*\] loss [0-9.]* (avg [0-9.]*).*iter [0-9]* ms" "$log" | tail -1)
  ep=$(echo "$last" | sed -n 's/^\[ *\([0-9]*\),.*/\1/p'); it=$(echo "$last" | sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p')
  ms=$(echo "$last" | grep -o "iter [0-9]* ms" | grep -o "[0-9]*")
  avg=$(echo "$last" | grep -o "(avg [0-9.]*)" | grep -o "[0-9.]*")
  if [[ -n $ep ]]; then
    done_steps=$(( (ep - 1) * 294 + it )); left=$(( 5 * 294 - done_steps ))
    eta=$(( left * ${ms:-1700} / 1000 / 60 ))
    echo "   진행: epoch $ep/5 · step $done_steps/1470 · 손실 평균 $avg · ${ms} ms/step · 남은 약 ${eta} 분 · $([[ $alive -gt 0 ]] && echo 학습 중 || echo 학습 프로세스 없음)"
  else
    echo "   시작 중 (모델 · 데이터 준비) · $([[ $alive -gt 0 ]] && echo 프로세스 있음 || echo 프로세스 없음)"
  fi
  echo "   real bench val (epoch 마다):"
  grep -o "\[epoch [0-9]*\] val acc [0-9.]*%[^|]*" "$log" | sed 's/^/     /' | tail -6
  grep -q "Traceback" "$log" && echo "   ⚠️ Traceback 있음 — tail -50 $log"
  if [[ "$run" == "vo_nat_ours" ]]; then
    st=$(grep -o "vo 세기[^|]*" "$log" | tail -1); [[ -n $st ]] && echo "   $st"
  fi
done
echo
echo "── 채점 사슬 (z_training/logs/chain_vo_nat_20261107.log)"
tail -4 z_training/logs/chain_vo_nat_20261107.log 2>/dev/null | sed 's/^/   /'
echo
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
