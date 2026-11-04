#!/bin/bash
# 11-03 사용자 "계속 진행해": e40 채점 → natural_ar_vw_state_oneshot_vidctx 70 ep 까지 재개 → 끝나면 네 벤치 자동 채점
cd /workspace/world_model/vjepa2
R=natural_ar_vw_state_oneshot_vidctx
RUN=$R MODEL=vast_ar_oneshot_vidctx_e40 EP=40 bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_vast_ar_oneshot_vidctx_e40.log 2>&1
echo "[resume] $(TZ=Asia/Seoul date '+%F %T KST') e40 채점 끝 → 학습 재개"
GPUS=8 bash z_training/tmux_train.sh $R
sleep 60
RUN=$R MODEL=vast_ar_oneshot_vidctx EP=70 bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_vast_ar_oneshot_vidctx.log 2>&1
echo "RESUME_VIDCTX_DONE $(TZ=Asia/Seoul date '+%F %T KST')"
