#!/bin/bash
# =============================================================================
# 물체 영역 음의 손실 두 판 모니터 (2026-11-08): A = AR e70 → objregion (GPU 0–3) · B = 하이브리드 e43 → objregion (GPU 4–7).
#
#   watch -c -n 30 bash z_training/monitor_objregion.sh
#
# 판마다: 진행 · real bench val (매 epoch, A 사라짐 / B 나타남) · epoch 평균 손실 항
#   (굴린 L1 가능 = 가능 미래 rollout L1, 전 토큰 · 불가능 L1 = target 토큰에서만 · 음의 손실 항 = max(0, 1 − 불가능 L1)).
# 대조 (학습 끝 채점): AR → 음의 손실 (전 토큰) real 72.1 (74.7 / 69.5) · 하이브리드 → 음의 손실 (전 토큰) 68.9 (77.9 / 60.0)
#   · AR → 물체 토큰 margin 1.0 64.2 (48.4 / 80.0) · AR → L1 만 66.3 (54 / 79).
# 2026-11-09: 기본 줄 = margin 1.5 두 판 + 하이브리드 L1 만 (C 판은 음의 손실 칸이 비어 있다).
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
R=z_training/runs
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
echo "=== 물체 영역 음의 손실 두 판 · $(kst) ==="
for row in ${ROWS:-"v4bc4_ar_vw_state_negimp_objregion_m15:A  AR e70 → 음의 손실 (target 토큰만, margin 1.5)" "v4bc4_tmem_negimp_objregion_m15:B  하이브리드 e43 → 음의 손실 (target 토큰만, margin 1.5)" "v4bc4_tmem_postft:C  하이브리드 e43 → L1 만"}; do
  run=${row%%:*}; name=${row#*:}; log=$R/$run/train.log
  echo; echo "── $name  [$run]"
  [[ -f $log ]] || { echo "   아직 시작 안 함"; continue; }
  alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$run/config.yaml")
  last=$(grep -oE "\[ *[0-9]+, *[0-9]+\] loss.*iter [0-9]+ ms" "$log" | tail -1)
  ep=$(sed -n 's/^\[ *\([0-9]*\),.*/\1/p' <<< "$last"); it=$(sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p' <<< "$last")
  ms=$(grep -oE "iter [0-9]+ ms" <<< "$last" | grep -oE "[0-9]+")
  if [[ -n $ep ]]; then
    done_steps=$(( (ep - 1) * 100 + it + 1 )); left=$(( 2000 - done_steps )); eta=$(( left * ${ms:-1500} / 1000 / 60 + (20 - ep) ))
    echo "   진행: epoch $ep/20 · step $done_steps/2000 · ${ms} ms/step · 남은 약 ${eta} 분 (val 포함) · $([[ $alive -gt 0 ]] && echo 학습 중 || { [[ -f $R/$run/e20.pt ]] && echo 끝남 || echo '⚠️ 프로세스 없음'; })"
  fi
  grep -q Traceback "$log" && echo "   ⚠️ Traceback 있음 — tail -50 $log"
  echo "   epoch | real val (A / B) | 굴린 L1 가능 | 불가능 L1 (target 토큰) | 음의 손실 항"
  join <(grep -oE "\[epoch [0-9]+\] val acc [0-9.]*% \(A [0-9.]* / B [0-9.]*\)" "$log" | awk '!s[$0]++' | sed -E 's/\[epoch ([0-9]+)\] val acc ([0-9.]+)% \(A ([0-9.]+) \/ B ([0-9.]+)\)/\1 \2(\3\/\4)/' | sort -k1,1) \
       <(grep -E "\[ *[0-9]+, *[0-9]+\] loss" "$log" | sed -E 's/^.*\[ *([0-9]+), *([0-9]+)\] loss.*neg pos ([0-9.]+) imp ([0-9.]+).*iloss ([0-9.]+).*/\1 \2 \3 \4 \5/' \
          | awk '!s[$1" "$2]++' | awk '{n[$1]++; p[$1]+=$3; i[$1]+=$4; l[$1]+=$5} END {for (e in n) printf "%d %.3f %.3f %.3f\n", e-1, p[e]/n[e], i[e]/n[e], l[e]/n[e]}' | sort -k1,1) -a1 \
    | sort -n | awk '{printf "   %5d | %-16s | %s | %s | %s\n", $1, $2, ($3==""?"-":$3), ($4==""?"-":$4), ($5==""?"-":$5)}' | tail -21
done
echo; echo "   (손실 항 줄 = 그 val 뒤 epoch 의 평균 — val epoch N 옆은 epoch N+1 학습 손실)"
for c in chain_objregion_m15_20261109 chain_tmem_postft_20261108; do
  echo; echo "── 채점 사슬 $c ($(pgrep -f "[${c:0:1}]${c:1}.sh" > /dev/null && echo 살아 있음 || echo 프로세스 없음))"
  tail -3 z_training/logs/$c.log 2>/dev/null | sed 's/^/   /'
done
echo; echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
