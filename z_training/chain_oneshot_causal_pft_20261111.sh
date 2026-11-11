#!/bin/bash
# one-shot 판 (블록 입력 · 블록 인과 · one-shot) e70 → v4bc4 L1 · v4bc4 음의 손실 p 0.2 이어 학습 → 채점 (2026-11-11, 사용자 "이거 하고 pft 로 v4bc4 하고 negative loss p 0.2 까지 chain").
#   chain_oneshot_causal_resume_20261111.sh (출발판 채점만 하던 사슬) 를 대신한다.
#   1) natural_ar_vw_state_oneshot_causal e70 + 학습 끝 기다림
#   2) PFT 두 판을 같은 GPU 8 장에 같이 올림 (판마다 GPU 8 × 6 = 전역 48, 2,000 step — 블록 AR 판들과 같은 설정)
#   3) 그동안 출발판 시트 네 벤치 + Surrogate
#   4) PFT 두 판 e20 끝나는 대로 시트 네 벤치 + Surrogate
#   5) 표 (블록 AR 짝과 나란히) → 6) 공식 IntPhys1 · 2 (학습 없을 때만, 세 판 차례로)
#   setsid nohup bash z_training/chain_oneshot_causal_pft_20261111.sh > z_training/logs/chain_oneshot_causal_pft_20261111.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
R0=natural_ar_vw_state_oneshot_causal;            M0=vast_ar_oneshot_causal
R1=v4bc4_oneshot_causal_postft;                   M1=vast_ar_oneshot_causal_v4bc4
R2=v4bc4_oneshot_causal_negimp_vanish_app02;      M2=vast_ar_oneshot_causal_v4bc4_negimp_vanish_app02
GP="0 1 2 3 4 5 6 7"
running() { ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$1/config\.yaml"; }

echo "[$(ts)] $R0 e70 + 학습 끝 기다림"
until [[ -f z_training/runs/$R0/e70.pt ]] && ! running $R0; do
  if ! tmux has-session -t "=train_$R0" 2>/dev/null && ! running $R0; then echo "[$(ts)] ⚠️ $R0 학습이 e70 없이 사라졌다 — 멈춤"; exit 1; fi
  sleep 120
done
sleep 30
echo "[$(ts)] $R0 끝 → PFT 두 판 올림"
GPUS=8 bash z_training/tmux_train.sh $R1 || { echo "[$(ts)] $R1 올리기 실패"; exit 1; }
GPUS=8 bash z_training/tmux_train.sh $R2 || { echo "[$(ts)] $R2 올리기 실패"; exit 1; }

echo "[$(ts)] 출발판 $M0 시트 채점 (PFT 와 같이)"
RUN=$R0 MODEL=$M0 EP=70 GPU_IDS="$GP" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M0.log 2>&1
GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M0 > z_training/logs/surrogate_eval_$M0.log 2>&1
echo "[$(ts)] $M0 시트 채점 끝"

for pair in "$R1 $M1" "$R2 $M2"; do
  set -- $pair
  RUN=$1 MODEL=$2 EP=20 GPU_IDS="$GP" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1 || { echo "[$(ts)] ⚠️ $1 학습 실패 또는 중단 — 채점 건너뜀"; continue; }
  GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1
  echo "[$(ts)] $2 시트 채점 끝"
done
echo "SHEET_EVAL_DONE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02 vast_ar_oneshot_vidctx_causal $M0 $M1 $M2 --md

for r in $R0 $R1 $R2; do
  while running $R1 || running $R2; do sleep 120; done          # 공식 채점은 학습이 없을 때만 (11-09 사고)
  echo "[$(ts)] 공식 채점 $r"
  GPUS=8 bash z_training/eval_bench.sh $r > z_training/logs/eval_bench_$r.log 2>&1
  tail -12 z_training/runs/$r/eval/BENCH_latest.txt 2>/dev/null
done
echo "OFFICIAL_DONE $(ts)"
