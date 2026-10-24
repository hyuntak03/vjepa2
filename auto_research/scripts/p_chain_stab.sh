#!/bin/bash
# [P] 안정도 보조 과제: 학습 3 판 → (a) 은닉 판독 → (c) 3D 합성 운반 비 → (b)(d) [T] 평가기. GPU 하나 · 스레드 8, 차례로.
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/venv/vjepa/bin/python; D=/workspace/auto_research_cache/ar_ft; S=auto_research/scripts
for spec in "ft_stab_l05_lr1e4 0.5 1e-4" "ft_stab_l01_lr1e4 0.1 1e-4" "ft_stab_l05_lr1e5 0.5 1e-5"; do
  set -- $spec
  [ -f $D/$1.pt ] || $PY -W ignore $S/ar_stab_ft.py --name $1 --lam $2 --lr $3 > auto_research/logs/$1.log 2>&1
  echo "trained $1 $(date +%H:%M)"
done
for n in ft_stab_l05_lr1e4 ft_stab_l01_lr1e4 ft_stab_l05_lr1e5; do
  $PY -W ignore $S/stab_mechanism.py --ft $n --only_ft 1 > auto_research/logs/stab_mechanism_$n.log 2>&1
  $PY -W ignore $S/sample_presence.py --run natural_ar_vw_state latest.pt --ft $D/$n.pt --K 1 > auto_research/logs/sp_$n.log 2>&1
  echo "a,c $n $(date +%H:%M)"
done
$PY -W ignore $S/ft_eval_reliability.py --ft ft_l1 ft_l1_lr1e4 ft_stab_l05_lr1e4 ft_stab_l01_lr1e4 ft_stab_l05_lr1e5 --out auto_research/results/ft_eval_reliability_stab.json > auto_research/logs/ft_eval_reliability_stab.log 2>&1
echo "b,d done $(date +%H:%M)"
