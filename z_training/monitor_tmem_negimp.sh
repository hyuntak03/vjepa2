#!/bin/bash
# =============================================================================
# 시간 누적 하이브리드 e43 → v4bc4 음의 손실 이어 학습 모니터 (2026-11-08): 진행 · real bench val · 같은 step 손실 대조 · 채점 사슬 · GPU.
#
#   watch -c -n 30 bash z_training/monitor_tmem_negimp.sh
#
# 대조 = v4bc4_ar_vw_state_negimp (AR scratch e70 → 같은 레시피, real 72.1 = 74.7 / 69.5). 같은 seed · 같은 데이터라 같은 step 에 같은 batch.
#   ⚠️ 대조판의 학습 중 val 은 VOBench v1 late 라 real val 은 옆에 둘 값이 없다 — 대조판 real 은 학습 끝 채점값 (72.1) 하나뿐.
# val = real world permanence 95 장면 (쌍 정확도 %, A 사라짐 / B 나타남), 매 epoch.
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
R=z_training/runs; RUN=v4bc4_tmem_negimp; BASE=v4bc4_ar_vw_state_negimp; EPOCHS=20; IPE=100
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
row() { grep -o "\[ *[0-9]*, *[0-9]*\] loss [0-9.]* (avg [0-9.]*) \[tf [0-9.]* ar [0-9.]*.*iter [0-9]* ms" "$1" 2>/dev/null \
        | awk '!seen[substr($0,1,index($0,"]"))]++'; }   # [ep, step] 마다 첫 줄 (rank 0)
echo "=== 하이브리드 e43 → v4bc4 음의 손실 · $(kst) ==="
log=$R/$RUN/train.log
[[ -f $log ]] || { echo "아직 시작 안 함"; exit 0; }
alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$RUN/config.yaml")
last=$(row "$log" | tail -1)
if [[ -n $last ]]; then
  ep=$(sed -n 's/^\[ *\([0-9]*\),.*/\1/p' <<< "$last"); it=$(sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p' <<< "$last")
  ms=$(grep -o "iter [0-9]* ms" <<< "$last" | grep -o "[0-9]*")
  done_steps=$(( (ep - 1) * IPE + it + 1 )); left=$(( EPOCHS * IPE - done_steps ))
  eta=$(( left * ${ms:-2500} / 1000 / 60 + (EPOCHS - ep) * 40 / 60 ))
  state=$([[ $alive -gt 0 ]] && echo 학습 중 || { [[ -f $R/$RUN/e$EPOCHS.pt ]] && echo 끝남 || echo '⚠️ 프로세스 없음'; })
  echo "진행: epoch $ep/$EPOCHS · step $done_steps/$((EPOCHS * IPE)) · ${ms} ms/step (누적 평균) · 남은 약 ${eta} 분 (val 포함) · $state"
else
  echo "시작 중 (모델 · 데이터 준비, 출발 val) · $([[ $alive -gt 0 ]] && echo 프로세스 있음 || echo 프로세스 없음)"
fi
grep -q Traceback "$log" && echo "⚠️ Traceback 있음 — tail -50 $log"
echo
echo "── real bench val (쌍 정확도 %, A 사라짐 / B 나타남) · 대조판 학습 끝 채점 = 72.1 (74.7 / 69.5), 출발 전 하이브리드 e43 = 58.2 (37.4 / 78.9)"
grep -o "\[epoch [0-9]*\] val acc [0-9.]*% (A [0-9.]* / B [0-9.]*)" "$log" | awk '!s[$0]++' | sed 's/^/   /' | tail -21
echo
echo "── 같은 step 손실 (누적 평균 avg · 이 판 / 대조판, 차이 = 이 판 − 대조판) · ar = rollout L1"
join -t'|' <(row "$log" | sed -E 's/^\[ *([0-9]+), *([0-9]+)\] loss [0-9.]* \(avg ([0-9.]+)\) \[tf ([0-9.]+) ar ([0-9.]+).*/\1-\2|\3|\5/' | sort -t'|' -k1,1) \
            <(row "$R/$BASE/train.log" | sed -E 's/^\[ *([0-9]+), *([0-9]+)\] loss [0-9.]* \(avg ([0-9.]+)\) \[tf ([0-9.]+) ar ([0-9.]+).*/\1-\2|\3|\5/' | sort -t'|' -k1,1) \
  | awk -F'|' '{split($1,a,"-"); if (a[2] == 0 || a[2] == 50 || a[2] == 90) printf "%d %d %s %s %s %s\n", a[1], a[2], $2, $4, $3, $5}' | sort -n -k1,1 -k2,2 \
  | awk '{printf "   epoch %2d step %2d: avg %s / %s (%+.4f) · ar(그 step) %s / %s\n", $1, $2, $3, $4, $3-$4, $5, $6}' | tail -9
echo
echo "── 채점 사슬 chain_tmem_negimp_20261108 ($(pgrep -f chain_tmem_negimp_20261108.sh > /dev/null && echo 살아 있음 || echo 프로세스 없음))"
tail -3 z_training/logs/chain_tmem_negimp_20261108.log 2>/dev/null | sed 's/^/   /'
echo
echo "── 같이 도는 학습 (GPU 공유)"
ps -eo args | grep "[l]aunch.py --fname" | sed -E 's#.*/runs/([^/]+)/config.yaml.*#   \1#' | sort -u
echo
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
