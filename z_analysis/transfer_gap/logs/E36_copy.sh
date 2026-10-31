#!/usr/bin/env bash
# [X2] R3 E36 보조: IntPhys 1 복사 기준선 (잡음 판 seed 별, 복사 열만). job 의 GPU 하나에 seed 하나씩 동시에.
#   사용: SEEDS="1 2" sbatch --gres=gpu:2 ... E36_copy.sh
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
i=0
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=$i $P z_research/scripts/analysis/transfer_gap_ip1copy.py extract --seed $s > z_analysis/transfer_gap/logs/E36_copy_src${s}_${SLURM_JOB_ID}.out 2>&1 &
  i=$((i + 1))
done
wait
echo E36_COPY_DONE $SEEDS
