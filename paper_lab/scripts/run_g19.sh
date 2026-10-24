#!/bin/bash
# G19 · G20 · G21 GPU 패스 (진행자 1, 2026-10-22). 학습 없음. 정의 · 판정 규칙 = paper_lab/EVIDENCE.md §3-2a (결과 전에 적음).
# Usage: CUDA_VISIBLE_DEVICES=<g> bash paper_lab/scripts/run_g19.sh A|B
#   lane A = 세트 캐시 (V-JEPA 2 연속 블록 + 픽셀 판 pix) → 모델 패스 (옮긴 계수 c)
#   lane B = G20 블록 안 움직임 (velprobe) · G21 떨림 / 칸 미만 (encprops) · encoder 변형 prep (dup · dinov2 · dinov2avg)
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CC=paper_lab/scripts/carry_coef.py
CE=paper_lab/scripts/carry_ext.py
export CARRY_THREADS=4 OMP_NUM_THREADS=4
run() { echo "[$(date '+%F %T')] >>> $*"; "$@" || echo "[$(date '+%F %T')] !!! FAILED: $*"; }
if [ "$1" = A ]; then
  run $P $CE prep --set natural --enc vjepa2 --n 200
  run $P $CE prep --set vobench --enc vjepa2 --n 400
  run $P $CE prep --set phys2d --enc vjepa2 --n 200
  run $P $CE prep --set natural12 --enc vjepa2 --n 200
  S=natural,vobench,phys2d
  run $P $CC model --ckpt z_training/vast/ar_state/latest.pt --name vast70 --sets $S
  run $P $CC model --ckpt z_training/ariel/ar_future_1e_5/latest.pt --name ariel_ar40 --sets natural,natural12,vobench,phys2d
  for a in A0 AX A2c A4; do run $P $CC model --ckpt paper_lab/_scratch/runs/pl_pilot_${a}/e10.pt --name pilot${a}_e10 --sets $S; done
  run $P $CC model --ckpt paper_lab/_scratch/runs/ext_ar_vw_shortctx/e30.pt --name shortctx_c8 --sets $S --ctx-in 8
  run $P $CC model --ckpt paper_lab/_scratch/runs/ext_ar_vw_shortctx/e30.pt --name shortctx_c2 --sets $S --ctx-in 2
  for s in mix nosup w005; do run $P $CC model --ckpt z_vll7_training/runs/phys2d_single_v1_pft_${s}/latest.pt --name state_${s} --sets $S; done
  run $P $CC model --ckpt z_training/runs/phys2d_sup_vast_ar_state_pft/latest.pt --name state_sup_v1 --sets $S
  echo "[$(date '+%F %T')] LANE A DONE"
else
  run $P $CE velprobe --n 400
  run $P $CE encprops --set natural --n 100
  run $P $CE encprops --set vobench --n 100
  for e in dup dinov2; do for s in natural vobench phys2d; do run $P $CE prep --set $s --enc $e --n $([ $s = vobench ] && echo 400 || echo 200); done; done
  for s in natural vobench; do run $P $CE prep --set $s --enc dinov2avg --n $([ $s = vobench ] && echo 400 || echo 200); done
  echo "[$(date '+%F %T')] LANE B DONE"
fi
