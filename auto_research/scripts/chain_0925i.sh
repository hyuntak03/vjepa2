#!/bin/bash
# ⚠️ 2026-09-27: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 가 지워져 이 스크립트는 그대로 재실행되지 않는다 (기록용). 새 경로: z_training/ARIEL_CHECKPOINTS.md
# 2026-09-25 묶음 I (SC2v2): flow 유사 정답 지평 곡선. v3 보정 (GT 대조; 릴리즈 · Ariel) → SSv2 predictor 넷.
# SC2 1 판 (encoder 템플릿 궤적 유사 정답) 은 SSv2 에서 무효 (슬롯 사이 점프 평균 5.5 칸) 였다. flow 추적은 CPU v3 보정에서 GT 1 칸 안 0.92–0.95.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training; A=$Z/ariel/block_causal_future_only_1e_5_ep
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
run $P $S/e_flowtrack.py --mode ssv2 --gpus 1 --limit 4 --out $C/_smoke/flowtrack || exit 1
run $P $S/e_flowtrack.py --mode v3 --gpus 1 --limit 4 --out $C/_smoke/flowtrack_v3 || exit 1
for pair in "release:" "ariel:${A}_43/latest.pt" "pv1:$Z/runs/predictor_v1_postft/latest.pt" "v11ft:$Z/runs/v11_postft/latest.pt"; do
  tag=${pair%%:*}; ck=${pair#*:}
  if [ -n "$ck" ]; then export ARLIB_PRED_CKPT=$ck; else unset ARLIB_PRED_CKPT; fi
  run $P $S/e_flowtrack.py --mode ssv2 --gpus 8 --out $C/flowtrack_ssv2_$tag
  case $tag in release|ariel) run $P $S/e_flowtrack.py --mode v3 --every 4 --gpus 8 --out $C/flowtrack_v3_$tag ;; esac
done
