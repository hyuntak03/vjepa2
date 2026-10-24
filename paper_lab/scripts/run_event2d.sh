#!/bin/bash
# PLAN §13 A 첫 실험 (진행자 2, 2026-10-22). GPU 한 장 (CUDA_VISIBLE_DEVICES). 학습 없음.
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
L=paper_lab/_scratch/event2d/run.log; mkdir -p paper_lab/_scratch/event2d
echo "[$(date '+%F %T')] start CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> $L; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $L
$P paper_lab/scripts/event2d.py prep --n 400 >> $L 2>&1 || { echo "PREP FAILED" >> $L; exit 1; }
for m in vast70:z_training/vast/ar_state/latest.pt sup_v1:z_training/runs/phys2d_sup_vast_ar_state_pft/latest.pt nosup:z_vll7_training/runs/phys2d_single_v1_pft_nosup/latest.pt mix:z_vll7_training/runs/phys2d_single_v1_pft_mix/latest.pt; do
  echo "[$(date '+%F %T')] model ${m%%:*}" >> $L
  $P paper_lab/scripts/event2d.py model --ckpt ${m#*:} --name ${m%%:*} >> $L 2>&1 || echo "MODEL ${m%%:*} FAILED" >> $L
done
$P paper_lab/scripts/event2d.py score > paper_lab/_scratch/event2d/score.txt 2>&1
echo "[$(date '+%F %T')] EVENT2D DONE" >> $L
