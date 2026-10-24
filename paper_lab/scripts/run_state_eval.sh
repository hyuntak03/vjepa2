#!/bin/bash
# paper_lab 진행자 2 (2026-10-22) — 사용자 상태 감독 블록 AR 체크포인트를 합격선 하네스로 (학습 없음, run 폴더는 읽기만).
# 2026-10-22 04:5x 리더 지시로 측정만 먼저 띄운다 (R5 법칙 검정에도 같은 체크포인트가 필요) — 방향 ('관측 + 합성 정답' 으로 가나) 은 여전히 사용자 결정. GPU 한 장 (CUDA_VISIBLE_DEVICES) · 체크포인트당 약 35–40 분 (빠른 판).
#   CUDA_VISIBLE_DEVICES=<g> bash paper_lab/scripts/run_state_eval.sh w005 nosup        # 핵심 대조 (같은 합성 데이터, 상태 감독 유무)
#   CUDA_VISIBLE_DEVICES=<g> bash paper_lab/scripts/run_state_eval.sh mix sup_v1         # 자연 영상 + 상태 보조 · 2D 여러 물체 상태 감독
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; E=paper_lab/eval/passline.py
declare -A CK=(
  [mix]=z_vll7_training/runs/phys2d_single_v1_pft_mix/latest.pt
  [nosup]=z_vll7_training/runs/phys2d_single_v1_pft_nosup/latest.pt
  [w005]=z_vll7_training/runs/phys2d_single_v1_pft_w005/latest.pt
  [sup_v1]=z_training/runs/phys2d_sup_vast_ar_state_pft/latest.pt
)
for k in "$@"; do
  ck=${CK[$k]:-}; [ -n "$ck" ] && [ -f "$ck" ] || { echo "missing $k"; continue; }
  name=state_$k
  echo "[$(date '+%F %T')] extract $name <- $ck"
  $P -u $E extract --ckpt "$ck" --name "$name" --gpus 1 --bs 8 --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 --fast || { echo "EXTRACT $name FAILED"; continue; }
  $P -u $E analyze --name "$name" --split all
  echo "[$(date '+%F %T')] done $name"
done
