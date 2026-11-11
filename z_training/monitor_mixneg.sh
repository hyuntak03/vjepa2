#!/bin/bash
# =============================================================================
# AR 처음부터 학습 (자연 영상 + 매 step v4bc4 음의 손실) 모니터 (2026-11-07). 진행 · 손실 · real 벤치 val (매 epoch) · 채점 사슬 · GPU.
#
#   watch -c -n 10 bash z_training/monitor_mixneg.sh
#
# val = real world permanence 95 장면 × 4 = 380 클립 (190 쌍). "val acc" = 쌍 정확도, 괄호 = A 사라짐 / B 나타남.
# 참고 (시트 real95): AR 기본판 (자연만 처음부터) 58.2 (37/79) · AR → v4bc4 음의 손실 이어 학습 72.1 (75/69).
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
RUN=natural_ar_vw_state_v4bc4_negimp_mix
LOG=z_training/runs/$RUN/train.log
kst() { TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST'; }
echo "=== 자연 영상 + v4bc4 음의 손실 처음부터 학습 · $(kst) ==="
if [[ ! -f $LOG ]]; then echo "아직 시작 안 함"; exit 0; fi
alive=$(ps -eo args | grep -c "[l]aunch.py --fname \S*/runs/$RUN/config.yaml")
last=$(grep -o "\[ *[0-9]*, *[0-9]*\] loss [0-9.]* (avg [0-9.]*).*iter [0-9]* ms" "$LOG" | tail -1)
ep=$(sed -n 's/^\[ *\([0-9]*\),.*/\1/p' <<< "$last"); it=$(sed -n 's/^\[ *[0-9]*, *\([0-9]*\)\].*/\1/p' <<< "$last")
ms=$(grep -o "iter [0-9]* ms" <<< "$last" | grep -o "[0-9]*")
if [[ -n $ep ]]; then
  done_steps=$(( (ep - 1) * 294 + it + 1 )); left=$(( 70 * 294 - done_steps ))
  eta_min=$(( left * ${ms:-2000} / 1000 / 60 + (70 - ep) * 40 / 60 ))
  fin=$(TZ=Asia/Seoul date -d "+${eta_min} minutes" '+%m-%d %H:%M KST')
  echo "진행: epoch $ep/70 · step $done_steps/20580 · ${ms} ms/step · 남은 약 $((eta_min / 60))시간 $((eta_min % 60))분 (끝 예상 $fin) · $([[ $alive -gt 0 ]] && echo 학습 중 || { [[ -f z_training/runs/$RUN/e70.pt ]] && echo 끝남 || echo '⚠️ 프로세스 없음'; })"
  echo "마지막 step: $(grep -oE '\[tf [0-9.]+ ar [0-9.]+ R=[0-9]+\]|neg pos [0-9.na]+ imp [0-9.na]+|iloss [0-9.]+ act [0-9.]+|mask var \S+ \S+' <<< "$(grep -E '\] loss' "$LOG" | tail -1)" | tr '\n' ' ')"
else
  echo "시작 중 (모델 · 데이터 준비 / 출발 val) · $([[ $alive -gt 0 ]] && echo 프로세스 있음 || echo 프로세스 없음)"
fi
echo
echo "epoch 별 음의 손실 요약 (가능 L1 · 불가능 L1 · hinge 활성):"
grep -oE "epoch [0-9]+ negimp \{[^}]*\}" "$LOG" | tail -5 | sed -E 's/.*epoch ([0-9]+) negimp .*"pos_l1": ([0-9.]+).*"imp_l1": ([0-9.]+).*"hinge_act": ([0-9.]+).*/  epoch \1: 가능 \2 · 불가능 \3 · hinge \4/'
echo
echo "real 벤치 val (epoch: 쌍 정확도 % (A / B)):"
grep -oE "\[epoch [0-9]+\] val acc [0-9.]+% \(A [0-9.]+ / B [0-9.]+\)" "$LOG" | sed -E 's/\[epoch ([0-9]+)\] val acc ([0-9.]+)% \(A ([0-9.]+) \/ B ([0-9.]+)\)/  epoch \1: \2 (\3 \/ \4)/' | tail -12
grep -q "Traceback" "$LOG" && echo "⚠️ Traceback 있음 — tail -50 $LOG"
echo
echo "── 채점 사슬 (z_training/logs/chain_mixneg_20261107.log)"
tail -3 z_training/logs/chain_mixneg_20261107.log 2>/dev/null | sed 's/^/   /'
echo
echo "── GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
