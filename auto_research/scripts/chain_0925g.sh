#!/bin/bash
# ⚠️ 2026-09-27: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 가 지워져 이 스크립트는 그대로 재실행되지 않는다 (기록용). 새 경로: z_training/ARIEL_CHECKPOINTS.md
# 2026-09-25 묶음 G (리뷰 B1 · B5): 학습 run 안에서 진화 시그니처가 epoch 따라 어떻게 변하나 (IntPhys 점수 곡선은 기존: Ariel README · TrainingEffects)
# + 기존 predictor 넷을 인과 외형 템플릿 (loc_appc) 포함 P2 로 재추출. 새 학습 없음.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training; A=$Z/ariel/block_causal_future_only_1e_5_ep
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
run $P $S/p2_stride_sweep_v3.py --gpus 1 --limit 4 --out $C/_smoke/p2c || exit 1
run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2c_release
for pair in "pv1:$Z/runs/predictor_v1_postft/latest.pt" "ariel:${A}_43/latest.pt" "v11ft:$Z/runs/v11_postft/latest.pt"; do
  tag=${pair%%:*}; export ARLIB_PRED_CKPT=${pair#*:}
  run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2c_$tag
done
for pair in "ariel5:${A}_5/latest.pt" "ariel19:${A}_19/latest.pt" "ariel30:${A}_30/latest.pt" "pv1e1:$Z/runs/predictor_v1_postft/e1.pt" "pv1e5:$Z/runs/predictor_v1_postft/e5.pt" "v11e1:$Z/runs/v11_postft/e1.pt" "v11e5:$Z/runs/v11_postft/e5.pt"; do
  tag=${pair%%:*}; export ARLIB_PRED_CKPT=${pair#*:}
  run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2c_$tag
  run $P $S/p3_closure_v3.py --gpus 8 --out $C/v3_p3_$tag
done
# 리뷰 B7: 학습 stride 3 (raw 2..59, 문맥 5 / 미래 5 튜블릿) 판 — predictor 넷
export P2_ARMS=s3_C10
unset ARLIB_PRED_CKPT; run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2s3_release
for pair in "pv1:$Z/runs/predictor_v1_postft/latest.pt" "ariel:${A}_43/latest.pt" "v11ft:$Z/runs/v11_postft/latest.pt"; do
  tag=${pair%%:*}; export ARLIB_PRED_CKPT=${pair#*:}
  run $P $S/p2_stride_sweep_v3.py --gpus 8 --out $C/v3_p2s3_$tag
done
unset P2_ARMS
