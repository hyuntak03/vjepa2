#!/bin/bash
# 2026-11-04 사용자: "이거 학습하고. 그 다음으로 block 단위로 입력주는거 해봐"
#   ① natural_ar_vw_state_oneshot_vidctx_causal (영상 문맥 + 블록 인과, 19:21 KST 시작) 70 ep 끝 → 시트 다섯 벤치 채점
#   ② natural_ar_vw_state_oneshot_causal (블록 문맥 + 블록 인과, 19:20 KST 에 e15 latest 에서 멈춤) 을 같은 이름으로 다시 올려 latest.pt 에서 이어 학습
#   ③ ② 70 ep 끝 → 다섯 벤치 채점 → 표
#   setsid nohup bash z_training/chain_oneshot_causal_20261104.sh > z_training/logs/chain_oneshot_causal_20261104.log 2>&1 &
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
score() {   # $1 run  $2 model
  RUN=$1 MODEL=$2 EP=70 bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1 || { echo "[chain] $(ts) ⚠️ $2 채점 대기 실패 — 멈춤"; exit 1; }
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1
  echo "[chain] $(ts) $2 채점 끝"
}
echo "[chain] $(ts) 시작"
score natural_ar_vw_state_oneshot_vidctx_causal vast_ar_oneshot_vidctx_causal
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_oneshot_causal && echo "[chain] $(ts) 블록 문맥판 이어 학습 올림"
sleep 120
score natural_ar_vw_state_oneshot_causal vast_ar_oneshot_causal
M="vast_tube_cd64 vast_ar_oneshot_vidctx vast_ar_oneshot_vidctx_causal vast_ar_oneshot_causal vast_ar_vw_state"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py $M
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py $M
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py $M
echo "CHAIN_ONESHOT_CAUSAL_DONE $(ts)"
