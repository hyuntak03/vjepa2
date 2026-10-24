#!/bin/bash
# [T] 실제 latent 방법 개입 (GPU 1 · 프로세스 1 개 · CPU 스레드 8): 스모크 → 학습 캐시 → 세 팔 이어 학습 → 빠른 판정 → 장난감 표본 4 개 CRPS
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "t_laneL.sh" > /dev/null; do sleep 10; done
SP=/tmp/claude-0/-workspace-world-model-vjepa2/87b968ef-e9ae-4e6b-b526-a71edee7ec67/scratchpad
P=auto_research/scripts/ar_crps_ft.py
G="CUDA_VISIBLE_DEVICES=1"
# 스모크 (작은 캐시, 3 step) — 실패하면 멈춘다
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss wta --nh 3 --name smoke_wta --steps 3 --bs 2 --cache $SP/ftcache --out $SP/ftout > auto_research/logs/ar_ft_smoke.log 2>&1 || exit 1
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss crps --xi laplace --xi_fix 1 --s_file /workspace/auto_research_cache/nugget_sigma_natural.npy --name smoke_crps --steps 3 --bs 2 --cache $SP/ftcache --out $SP/ftout >> auto_research/logs/ar_ft_smoke.log 2>&1 || exit 1
for s in k710_320 ht100m_1pct ssv2; do
  env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_natural.py --src $s --n 300 --nb 8 --flow 0 --seed 7 --out /workspace/auto_research_cache/natural_train > auto_research/logs/cache_natural_train_$s.log 2>&1
done
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss l1 --name ft_l1 --steps 2000 > auto_research/logs/ar_ft_l1.log 2>&1
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss wta --nh 3 --wta_eps 0.05 --name ft_wta3 --steps 2000 > auto_research/logs/ar_ft_wta3.log 2>&1
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss crps --xi laplace --xi_fix 1 --s_file /workspace/auto_research_cache/nugget_sigma_natural.npy --name ft_crpsxi --steps 2000 > auto_research/logs/ar_ft_crpsxi.log 2>&1
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_eval.py --ft ft_l1 ft_wta3 ft_crpsxi --K 8 > auto_research/logs/ar_crps_eval.log 2>&1
env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name b_j0_crps_m4 --rope 1 --obj_fixed 1.0 --steps 9000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 --jit 0 --loss crps --ns 4 > auto_research/logs/toy_factor_b_j0_crps_m4.log 2>&1
