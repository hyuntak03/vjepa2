#!/bin/bash
# ⚠️ 2026-09-27: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 가 지워져 이 스크립트는 그대로 재실행되지 않는다 (기록용). 새 경로: z_training/ARIEL_CHECKPOINTS.md
# 2026-09-25 묶음 D: z_training 에 이미 학습된 predictor 들에 진단 시그니처 (P2 지평 모형 · P3 닫힘 · M1/M3) — 새 학습 없음 (사용자 결정).
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
# 스모크: prefix kind 체크포인트 적재 경로
ARLIB_PRED_CKPT=$Z/ariel/block_causal_future_only_1e_5_ep_43/latest.pt run $P $S/p2_stride_sweep_v3.py --gpus 1 --limit 4 --out $C/_smoke/p2_ariel || exit 1
for pair in "pv1:$Z/runs/predictor_v1_postft/latest.pt" "ariel:$Z/ariel/block_causal_future_only_1e_5_ep_43/latest.pt" "v11ft:$Z/runs/v11_postft/latest.pt"; do
  tag=${pair%%:*}; ck=${pair#*:}
  export ARLIB_PRED_CKPT=$ck
  run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2_$tag
  run $P $S/p3_closure_v3.py --gpus 8 --out $C/v3_p3_$tag
  run $P $S/m13_memory_state_v3.py --gpus 8 --out $C/v3_m13_$tag
done
