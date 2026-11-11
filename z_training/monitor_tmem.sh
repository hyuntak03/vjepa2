#!/bin/bash
# =============================================================================
# 시간 누적 하이브리드 AR scratch 모니터 (2026-11-08): 진행 · 같은 step 손실 대조 · real bench val · 채점 사슬 · GPU.
#
#   watch -c -n 30 bash z_training/monitor_tmem.sh
#
# 대조 = AR scratch 기준판 natural_ar_vw_state (같은 데이터 · 같은 seed → 같은 step 에 같은 batch · 같은 창).
#   그래서 같은 [epoch, step] 의 손실을 바로 비교할 수 있다 (rank 0 줄, 손실 = 누적 평균 avg).
# val = real world permanence (95 장면, 쌍 정확도 A 사라짐 / B 나타남). 기준판의 val 은 다른 세트라 옆에 두지 않는다.
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
R=z_training/runs; RUN=${RUN:-natural_ar_vw_state_tmem}; BASE=natural_ar_vw_state
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
row() { grep -o "\[ *[0-9]*, *[0-9]*\] loss [0-9.]* (avg [0-9.]*) \[tf [0-9.]* ar [0-9.]*.*iter [0-9]* ms" "$1" 2>/dev/null \
        | awk '!seen[substr($0,1,index($0,"]"))]++'; }   # [ep, step] 마다 첫 줄 (rank 0)
echo "=== 시간 누적 하이브리드 scratch · $(kst) ==="
log=$R/$RUN/train.log
[[ -f $log ]] || { echo "아직 시작 안 함"; exit 0; }
alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$RUN/config.yaml")
last=$(row "$log" | tail -1)
ep=$(sed -n 's/^\[ *\([0-9]*\),.*/\1/p' <<< "$last"); it=$(sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p' <<< "$last")
ms=$(grep -o "iter [0-9]* ms" <<< "$last" | grep -o "[0-9]*")
done_steps=$(( (ep - 1) * 294 + it + 1 )); left=$(( 20580 - done_steps ))
eta_min=$(( left * ${ms:-2000} / 1000 / 60 + (70 - ep) * 35 / 60 ))
echo "진행: epoch $ep/70 · step $done_steps/20580 · ${ms} ms/step (누적 평균) · 남은 약 $((eta_min / 60))시간 $((eta_min % 60))분 · $([[ $alive -gt 0 ]] && echo 학습 중 || echo '⚠️ 프로세스 없음')"
grep -q Traceback "$log" && echo "⚠️ Traceback 있음 — tail -50 $log"
echo
echo "── 같은 step 손실 (누적 평균, 이 판 / 기준판 · 차이 = 이 판 − 기준판, 음수면 이 판이 낮다)"
join -t'|' <(row "$log" | sed -E 's/^\[ *([0-9]+), *([0-9]+)\] loss [0-9.]* \(avg ([0-9.]+)\) \[tf ([0-9.]+) ar ([0-9.]+).*/\1-\2|\3|\4|\5/' | sort -t'|' -k1,1) \
            <(row "$R/$BASE/train.log" | sed -E 's/^\[ *([0-9]+), *([0-9]+)\] loss [0-9.]* \(avg ([0-9.]+)\) \[tf ([0-9.]+) ar ([0-9.]+).*/\1-\2|\3|\4|\5/' | sort -t'|' -k1,1) \
  | awk -F'|' '{split($1,a,"-"); if (a[2] % 100 == 0 || a[2] == 290) printf "%d %d %s %s %s %s\n", a[1], a[2], $2, $5, $3, $6}' | sort -n -k1,1 -k2,2 \
  | awk '{printf "   epoch %2d step %3d: avg %s / %s (%+.4f) · tf(그 step) %s / %s\n", $1, $2, $3, $4, $3-$4, $5, $6}' | tail -12
echo
echo "── real bench val (쌍 정확도 %, A 사라짐 / B 나타남)"
grep -o "\[epoch [0-9]*\] val acc [0-9.]*% (A [0-9.]* / B [0-9.]*)" "$log" | awk '!s[$0]++' | sed 's/^/   /' | tail -12
echo
echo "── 채점 사슬 chain_tmem_20261108 ($(pgrep -f chain_tmem_20261108.sh > /dev/null && echo 살아 있음 || echo 프로세스 없음))"
tail -2 z_training/logs/chain_tmem_20261108.log 2>/dev/null | sed 's/^/   /'
echo
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
