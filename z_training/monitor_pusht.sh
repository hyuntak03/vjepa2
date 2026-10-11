#!/bin/bash
# Push-T 두 팔 (post-FT → scratch 순차, tmux 세션 train_pusht) 을 한 화면에서 본다.
#   watch -c -n 2 bash z_training/monitor_pusht.sh
# 위: 어느 팔이 돌고 있나 + tmux 세션 생존.  가운데: 돌고 있는 run 의 monitor.sh.  아래: 두 run 의 마지막 val 줄 (tf/ar vs 기준선).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RUNS=(pusht_ar_postft pusht_ar_scratch)
B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; D=$'\e[2m'; N=$'\e[0m'
[[ "${MONITOR_COLOR:-1}" == 0 ]] && B= G= Y= R= D= N=
sess=$(tmux has-session -t train_pusht 2>/dev/null && echo "${G}살아있음${N}" || echo "${R}없음${N}")
echo "${B} Push-T 팔 (순차: post-FT → scratch)${N}   tmux train_pusht: $sess   $(date '+%m-%d %H:%M:%S')"
active=""
for r in "${RUNS[@]}"; do
  log=z_training/runs/$r/train.log
  if [[ ! -f $log ]]; then st="${D}대기${N}"
  elif grep -q "^\[INFO.*\] done$" "$log" 2>/dev/null; then st="${G}완료${N}"
  elif grep -q "Traceback\|OutOfMemoryError" "$log" 2>/dev/null && ! pgrep -f "NAME=$r\|runs/$r" >/dev/null; then st="${R}에러로 중단${N}"
  else st="${Y}진행 중${N}"; [[ -z $active ]] && active=$r; fi
  ep=$(grep -oE "^\[INFO.*epoch ([0-9]+) avg loss" "$log" 2>/dev/null | tail -1 | grep -oE "epoch [0-9]+" )
  printf "   %-18s %s  %s\n" "$r" "$st" "${ep:+($ep 끝)}"
done
echo
if [[ -n $active ]]; then MONITOR_COLOR=${MONITOR_COLOR:-1} bash z_training/monitor.sh "$active" 2>/dev/null | sed -n '4,30p'; fi
echo "${D}──────────────────────────────── val (블록 LN(h) L1, 낮을수록 좋다; copy/zero 는 기준선) ────────────────────────────${N}"
for r in "${RUNS[@]}"; do
  log=z_training/runs/$r/train.log; [[ -f $log ]] || continue
  echo "${B} $r${N}"
  grep -oE "\[epoch [0-9]+\] val\[pusht[^]]*\] tf [0-9.]+ ar [0-9.]+ \| copy [0-9.]+ copy_last [0-9.]+ zero [0-9.]+" "$log" | tail -n 4 | sed 's/^/   /'
done
L=z_research/action_planning/exp_results/eval_after_train.log
sess_e=$(tmux has-session -t eval_pusht 2>/dev/null && echo "${G}대기/진행${N}" || echo "${D}없음${N}")
echo "${D}──────────────────────────────── 평가 (tmux eval_pusht: 학습 둘 다 끝나면 자동) ─────────────────────────────${N}"
echo "   eval_pusht: $sess_e"; [[ -f $L ]] && tail -n 4 "$L" | sed 's/^/   /'

D2=z_research/action_planning/exp_results
for r in pusht_ar_postft pusht_ar_scratch; do
  tot=0; ok=0; it=""
  for f in $D2/${r}_latest_gH5_dset_plan_s300o30_shard*of4/logs.json; do
    [[ -f $f ]] || continue
    n=$(grep -c '"mpc/success_rate"' "$f"); it="$it$n "
    sr=$(grep '"mpc/success_rate"' "$f" | tail -1 | grep -oE '"mpc/success_rate": [0-9.]+' | cut -d' ' -f2)
    sz=$(grep -oE '"shard": "[0-9]+/[0-9]+"' $(dirname $f)/../plan_${r}_shard0.log 2>/dev/null | head -1)
    i=$(basename $(dirname $f) | grep -oE 'shard[0-9]+' | tr -dc 0-9); k=$(( i < 2 ? 13 : 12 ))
    [[ -n $sr ]] && { tot=$((tot+k)); ok=$(python3 -c "print($ok+round($sr*$k))"); }
  done
  [[ -n $it ]] && echo "   planning $r: MPC 반복 (조각별) $it/ 5   누적 성공 $ok / $tot"
done
for m in $D2/*_merged.json; do [[ -f $m ]] && python3 -c "import json;d=json.load(open('$m'));print('   최종', d['prefix'].split('/')[-1], 'success_rate %.3f (%d/%d)'%(d['success_rate'],sum(d['successes']),d['n_evals']))"; done
