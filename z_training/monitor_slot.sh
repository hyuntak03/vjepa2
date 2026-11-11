#!/bin/bash
# =============================================================================
# v4bc4 L1 post-train 구조 판 모니터 (2026-11-07): slot · slot+맞추기 · Δ 예측 · 다중 블록 예측. 학습 진행 · 학습 중 val · 채점 사슬 · GPU.
#
#   watch -c -n 10 bash z_training/monitor_slot.sh
#
# 판: slot (GPU 0–3) · slot+맞추기 (GPU 4–7) · Δ 예측 (8 장) · 다중 블록 예측 (8 장). 전부 20 epoch × 100 step = 2,000 step, 전체 배치 48.
# 학습 중 val = VOBench v1 late (2 epoch 마다, 쌍 정확도). 벤치 점수가 아니다 — 벤치는 채점 사슬 뒤 시트 표로 본다.
# 대조판 (v4bc4 예측 L1 만, slot 없음) 의 같은 epoch val 을 옆에 둔다. 두 판 모두 음의 손실 없음.
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
R=z_training/runs
BASE=v4bc4_ar_vw_state_postft      # 대조 = v4bc4 예측 L1 만 (slot 없음, 같은 레시피)
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
vals() { grep -o "\[epoch [0-9]*\] val acc [0-9.]*%" "$1" 2>/dev/null | sed 's/\[epoch \([0-9]*\)\] val acc \([0-9.]*\)%/\1 \2/'; }
echo "=== v4bc4 L1 구조 판 · $(kst) ==="
for row in "v4bc4_ar_vw_state_slot:slot (구조만)" "v4bc4_ar_vw_state_slotmatch:slot + 맞추기" "v4bc4_ar_vw_state_delta:Δ 예측 (x_k + P)" "v4bc4_ar_vw_state_mtp:다중 블록 예측 (h=2,3 head)" "v4bc4_ar_vw_state_pastgate:now/past gate" "v4bc4_ar_vw_state_negimp_pastgate:now/past gate + 음의 손실:v4bc4_ar_vw_state_negimp"; do
  run=${row%%:*}; rest=${row#*:}; name=${rest%%:*}; base=$BASE; [[ $rest == *:* ]] && base=${rest##*:}   # 행마다 대조판 (기본 L1, 음의 손실 판은 음의 손실 기준판)
  log=$R/$run/train.log
  echo
  echo "── $name  [$run]"
  if [[ ! -f $log ]]; then echo "   아직 시작 안 함"; continue; fi
  alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$run/config.yaml")
  last=$(grep -o "\[ *[0-9]*, *[0-9]*\] loss [0-9.]* (avg [0-9.]*).*iter [0-9]* ms" "$log" | tail -1)
  ep=$(sed -n 's/^\[ *\([0-9]*\),.*/\1/p' <<< "$last"); it=$(sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p' <<< "$last")
  ms=$(grep -o "iter [0-9]* ms" <<< "$last" | grep -o "[0-9]*")
  avg=$(grep -o "(avg [0-9.]*)" <<< "$last" | grep -o "[0-9.]*")
  if [[ -n $ep ]]; then
    done_steps=$(( (ep - 1) * 100 + it + 1 )); left=$(( 2000 - done_steps ))
    eta=$(( left * ${ms:-1500} / 1000 / 60 ))
    state=$([[ $alive -gt 0 ]] && echo "학습 중" || { [[ -f $R/$run/e20.pt ]] && echo "끝남" || echo "⚠️ 프로세스 없음"; })
    echo "   진행: epoch $ep/20 · step $done_steps/2000 · 손실 평균 $avg · ${ms} ms/step · 남은 약 ${eta} 분 (val 제외) · $state"
  else
    echo "   시작 중 (모델 · 데이터 준비, 출발 val) · $([[ $alive -gt 0 ]] && echo 프로세스 있음 || echo 프로세스 없음)"
  fi
  echo "   학습 중 val (VOBench v1 late 쌍 정확도 %, epoch: 이 판 / 대조 $base):"
  join -a1 <(vals "$log" | sort -k1,1) <(vals "$R/$base/train.log" | sort -k1,1) | sort -n | awk '{printf "     epoch %2d: %5s / %s\n", $1, $2, ($3 == "" ? "-" : $3)}' | tail -11
  grep -q "Traceback" "$log" && echo "   ⚠️ Traceback 있음 — tail -50 $log"
done
echo
for c in chain_v4bc4_struct_20261107 chain_v4bc4_l1_official_20261107 chain_v4bc4_pastgate_20261107 chain_v4bc4_negimp_pastgate_20261107; do
  echo "── 채점 사슬 $c ($(pgrep -f "$c.sh" > /dev/null && echo 살아 있음 || echo 프로세스 없음))"
  tail -3 z_training/logs/$c.log 2>/dev/null | sed 's/^/   /'
done
echo
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
