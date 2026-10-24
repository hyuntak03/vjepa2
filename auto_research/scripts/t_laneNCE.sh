#!/bin/bash
# [T] 가까운 음성 대조 이어 학습 4 팔 + 판정 (GPU 7 하나, 하나씩, 스레드 8). ⚠️ 사용자 "올려" 뒤에만 올린다 (리더 KST 18:22).
# 사전 등록: 진행판 [T] KST 18:33. 준비물: 다시 인코딩한 학습 캐시 /workspace/auto_research_cache/natural_train_reenc (끝).
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
FD=/workspace/auto_research_cache/nce_ft
for arm in "l0:0" "l001:0.01" "l01:0.1" "l05:0.5"; do
  n=${arm%%:*}; lam=${arm#*:}
  run auto_research/scripts/nce_ft.py --name nce_$n --lam $lam --tau 0.025 --steps 2000 --lr 3e-5 --out $FD > auto_research/logs/nce_ft_$n.log 2>&1
  echo "train nce_$n $(date -u +%H:%M)"
done
M="base nce_l0 nce_l001 nce_l01 nce_l05"
run auto_research/scripts/nce_rank_eval.py --models $M --ftdir $FD > auto_research/logs/nce_rank_eval.log 2>&1; echo "rank $(date -u +%H:%M)"
run auto_research/scripts/cf_eval.py --models $M --ftdir $FD --parts ip2,vob,hold --out auto_research/results/nce_eval.json > auto_research/logs/nce_eval.log 2>&1; echo "eval $(date -u +%H:%M)"
/venv/vjepa/bin/python auto_research/scripts/cf_compare.py --res auto_research/results/nce_eval.json --base nce_l0 --arms nce_l001 nce_l01 nce_l05 --out auto_research/results/nce_compare.json > auto_research/logs/nce_compare.log 2>&1
run auto_research/scripts/transport_placebo_arms.py --arms nce_l0 nce_l001 nce_l01 nce_l05 --ftdir $FD --out auto_research/results/nce_transport.json > auto_research/logs/nce_transport.log 2>&1
echo "all done $(date -u +%H:%M)"
