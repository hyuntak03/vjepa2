#!/bin/bash
# ⚠️ 2026-09-27: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 가 지워져 이 스크립트는 그대로 재실행되지 않는다 (기록용). 새 경로: z_training/ARIEL_CHECKPOINTS.md
# 2026-09-25 묶음 F: 구조적 시그니처가 학습된 predictor 에서도 남는가 — P3b (마지막 토큰) · M6 (역사 끌림) × {Ariel, pv1} + E-SSv2 (Ariel). 새 학습 없음.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
for pair in "ariel:$Z/ariel/block_causal_future_only_1e_5_ep_43/latest.pt" "pv1:$Z/runs/predictor_v1_postft/latest.pt"; do
  tag=${pair%%:*}; export ARLIB_PRED_CKPT=${pair#*:}
  run $P $S/p3b_closure_v2.py --gpus 8 --out $C/v3_p3b_$tag
  run $P $S/m6_history_shift_v3.py --gpus 8 --out $C/v3_m6_$tag
done
export ARLIB_PRED_CKPT=$Z/ariel/block_causal_future_only_1e_5_ep_43/latest.pt
run $P $S/e_ssv2_extract.py --gpus 8 --out $C/ssv2_e3_ariel
