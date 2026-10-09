#!/bin/bash
# ⚠️ 2026-09-27: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 가 지워져 이 스크립트는 그대로 재실행되지 않는다 (기록용). 새 경로: z_training/ARIEL_CHECKPOINTS.md
# 2026-09-25 묶음 H (oral 방향 sanity check): SC1 긴 미래 (분할 raw 16) 로 "학습 지평에서 멈추나" · SC2 라벨 없는 SSv2 실영상 지평 — predictor 넷.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training; A=$Z/ariel/block_causal_future_only_1e_5_ep
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
run $P $S/e_ssv2_track.py --gpus 1 --limit 4 --out $C/_smoke/ssv2_track || exit 1
for pair in "release:" "pv1:$Z/runs/predictor_v1_postft/latest.pt" "ariel:${A}_43/latest.pt" "v11ft:$Z/runs/v11_postft/latest.pt"; do
  tag=${pair%%:*}; ck=${pair#*:}
  if [ -n "$ck" ]; then export ARLIB_PRED_CKPT=$ck; else unset ARLIB_PRED_CKPT; fi
  P2_ARMS=s2_S16,s1_S16 run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2s16_$tag
  run $P $S/e_ssv2_track.py --gpus 8 --out $C/ssv2_track_$tag
done
