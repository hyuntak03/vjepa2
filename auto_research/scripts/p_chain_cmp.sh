#!/bin/bash
# [P] 1b 흐름 없는 비교 보조 과제: 흐름판 대기열이 끝난 뒤 학습 2 판 → (a) → (c) → (b)(d). GPU 하나 · 스레드 8.
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/venv/vjepa/bin/python; D=/workspace/auto_research_cache/ar_ft; S=auto_research/scripts
until grep -q "b,d done" auto_research/logs/p_chain_stab.log 2>/dev/null; do sleep 20; done
for spec in "ft_cmp_l05_lr1e4 0.5" "ft_cmp_l01_lr1e4 0.1"; do
  set -- $spec
  [ -f $D/$1.pt ] || $PY -W ignore $S/ar_cmp_ft.py --name $1 --lam $2 --lr 1e-4 > auto_research/logs/$1.log 2>&1
  echo "trained $1 $(date +%H:%M)"
done
for n in ft_cmp_l05_lr1e4 ft_cmp_l01_lr1e4; do
  $PY -W ignore $S/stab_mechanism.py --ft $n --only_ft 1 > auto_research/logs/stab_mechanism_$n.log 2>&1
  $PY -W ignore $S/sample_presence.py --run natural_ar_vw_state latest.pt --ft $D/$n.pt --K 1 > auto_research/logs/sp_$n.log 2>&1
  echo "a,c $n $(date +%H:%M)"
done
$PY -W ignore $S/ft_eval_reliability.py --ft ft_l1_lr1e4 ft_cmp_l05_lr1e4 ft_cmp_l01_lr1e4 --out auto_research/results/ft_eval_reliability_cmp.json > auto_research/logs/ft_eval_reliability_cmp.log 2>&1
echo "b,d done $(date +%H:%M)"
