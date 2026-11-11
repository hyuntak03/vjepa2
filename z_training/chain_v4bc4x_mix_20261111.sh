#!/bin/bash
# v4bc4x (0.2M 증축) 데이터가 완성되면: 시험 200 묶음 떼기 → 학습 올림 (scratch, 자연 + v4bc4x, 음의 손실 p 0.2, 합성 batch 4, val 5 ep) → 기준판들 시험셋 채점 → e70 뒤 채점 (2026-11-11, 사용자).
#   setsid nohup bash z_training/chain_v4bc4x_mix_20261111.sh > z_training/logs/chain_v4bc4x_mix_20261111.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
L=/workspace/dataset/training_data/blender_track/v4bc4x/logs/run_v4bc4x.out
RUN=natural_ar_vw_state_v4bc4x_negimp_vanish_app02_mix; M=vast_ar_vw_state_v4bc4x_negimp_vanish_app02_mix
GP="0 1 2 3 4 5 6 7"
echo "[$(ts)] 데이터 완성 (V4BC4X_DONE) 기다림"
until grep -q V4BC4X_DONE $L 2>/dev/null; do
  if ! pgrep -f "bash .*run_v4bc4[x].sh" >/dev/null; then echo "[$(ts)] ⚠️ 렌더 사슬이 끝 표시 없이 사라졌다 — 멈춤"; exit 1; fi
  sleep 60
done
echo "[$(ts)] 데이터 완성 → 시험셋 떼기"
source z_research/scripts/data/blender_track/env.sh
$BT_PY z_research/scripts/data/blender_track/holdout_v4bc4x.py --src-name blender_track_v4bc4x --train-name blender_track_v4bc4x_train \
  --test-root /workspace/dataset/benchmarks/v4bc4x_test --test-name v4bc4x_test --n-v4b 140 --n-v4c 60 || { echo "[$(ts)] ⚠️ 시험셋 떼기 실패"; exit 1; }
DRYRUN=1 GPUS=8 NAME=_dry_$RUN bash z_training/train.sh $RUN > z_training/logs/dryrun_$RUN.log 2>&1 || { echo "[$(ts)] ⚠️ DRYRUN 실패"; tail -5 z_training/logs/dryrun_$RUN.log; exit 1; }
GPUS=8 bash z_training/tmux_train.sh $RUN || { echo "[$(ts)] ⚠️ 학습 올리기 실패"; exit 1; }
echo "[$(ts)] 학습 올림"
for m in vith vast_ar_vw_state vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02; do   # 기준판들 시험셋 (학습과 같이, 작다)
  GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh v4bc4x_test $m > z_training/logs/v4bc4x_test_$m.log 2>&1 && echo "[$(ts)] 시험셋 채점 $m 끝" || echo "[$(ts)] ⚠️ 시험셋 채점 $m 실패"
done
RUN=$RUN MODEL=$M EP=70 GPU_IDS="$GP" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[$(ts)] ⚠️ 학습 실패 또는 중단"; exit 1; }
GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh v4bc4x_test $M > z_training/logs/v4bc4x_test_$M.log 2>&1
echo "SHEET_EVAL_DONE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02 $M --md 2>&1 | tail -20
