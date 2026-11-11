#!/bin/bash
# 블록 입력 · 블록 인과 · one-shot 판 (natural_ar_vw_state_oneshot_causal) 을 e15 → e70 이어 학습하고 채점 (2026-11-11, 사용자 "이거 해봐").
#   왜: e14 에서 이미 블록 AR (e70) 과 VOB3 · Surrogate · 실제 영상 · IP2 late 가 같다 → 같은 예산 (70 ep) 에서 예측 방식 (one-shot vs rollout) 축을 확인.
#   순서: 앞 사슬 (scratch 섞기 판 채점 · 공식 채점, pid WAIT_PID) 이 끝날 때까지 기다림 → 이어 학습 (GPU 8) → 시트 네 벤치 → Surrogate → 표 → 공식 IntPhys1 · 2.
#   setsid nohup bash z_training/chain_oneshot_causal_resume_20261111.sh > z_training/logs/chain_oneshot_causal_resume_20261111.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
WAIT_PID=${WAIT_PID:-2427290}
RUN=natural_ar_vw_state_oneshot_causal; M=vast_ar_oneshot_causal
echo "[$(ts)] 앞 사슬 (pid $WAIT_PID) 끝나길 기다림"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(ts)] 앞 사슬 끝 → 이어 학습 올림 ($RUN, latest.pt = epoch 15)"
GPUS=8 bash z_training/tmux_train.sh $RUN || { echo "[$(ts)] 학습 올리기 실패"; exit 1; }
RUN=$RUN MODEL=$M EP=70 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[$(ts)] 학습 실패 또는 중단"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_EVAL_DONE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_oneshot_vidctx_causal $M --md
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
echo "OFFICIAL_DONE $(ts)"; tail -12 z_training/runs/$RUN/eval/BENCH_latest.txt 2>/dev/null
