#!/bin/bash
# [T] (GPU 1 · 하나씩 · 스레드 8) CRPS 팔 끝 → 세 팔 판정 → 양성 대조 (인공 두 갈래) 다중 가설 → 판정 → 강한 학습 다중 가설 (lr 1e-4 · 6000) → 판정 → 장난감 (표본 4 CRPS · 블록 노름 energy score)
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while kill -0 2346564 2>/dev/null; do sleep 15; done
P=auto_research/scripts/ar_crps_ft.py
E=auto_research/scripts/ar_crps_eval.py
run() { env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore "$@"; }
run $E --ft ft_l1 ft_wta3 ft_crpsxi --K 8 > auto_research/logs/ar_crps_eval.log 2>&1
cp auto_research/results/ar_crps_eval.json auto_research/results/ar_crps_eval_3arms.json
run $P --loss wta --nh 3 --wta_eps 0.05 --pos_ctrl 1 --name ft_wta3_posctrl --steps 2000 > auto_research/logs/ar_ft_wta3_posctrl.log 2>&1
run $E --ft ft_wta3_posctrl --K 8 > auto_research/logs/ar_crps_eval_posctrl.log 2>&1
cp auto_research/results/ar_crps_eval.json auto_research/results/ar_crps_eval_posctrl.json
run $P --loss wta --nh 3 --wta_eps 0.05 --lr 1e-4 --name ft_wta3_lr1e4 --steps 6000 > auto_research/logs/ar_ft_wta3_lr1e4.log 2>&1
run $E --ft ft_wta3_lr1e4 --K 8 > auto_research/logs/ar_crps_eval_lr1e4.log 2>&1
cp auto_research/results/ar_crps_eval.json auto_research/results/ar_crps_eval_lr1e4.json
T=auto_research/scripts/toy_factor.py
toy() { n=$1; shift; run $T --name $n --rope 1 --obj_fixed 1.0 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
toy b_j0_es2 --jit 0 --loss es --ns 2 --steps 6000
toy b_j0_crps_m4 --jit 0 --loss crps --ns 4 --steps 9000
toy b_j0_es4 --jit 0 --loss es --ns 4 --steps 6000
