#!/bin/bash
# 2026-11-08 AR 처음부터: 시간 누적 하이브리드 (configs/training/natural_ar_vw_state_tmem.yaml) — (GPU 가 빌 때까지 대기) → 학습 → 채점.
#   WAIT_PID=<먼저 도는 학습의 launch.py PID> 를 주면 그 프로세스가 끝난 뒤 올린다 (없으면 바로).
#   채점: 시트 네 벤치 (chain_full_eval) → Surrogate → 시트 표 (AR scratch 기준판과 비교) → 공식 IP1 · IP2.
#   setsid nohup env WAIT_PID=<pid> bash z_training/chain_tmem_20261108.sh > z_training/logs/chain_tmem_20261108.log 2>&1 &
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=${RUN:-natural_ar_vw_state_tmem}; M=${M:-vast_ar_vw_state_tmem}
if [[ -n "${WAIT_PID:-}" ]]; then
  echo "대기: PID $WAIT_PID 가 끝날 때까지 $(ts)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "PID $WAIT_PID 끝남 $(ts) — GPU 8 장이 3 분 연속 빌 때까지 대기 (그 판의 채점 사슬과 겹치지 않게)"
fi
idle=0
while (( idle < 3 )); do   # 모든 GPU 메모리 < 2 GB 가 1 분 간격 3 번 연속
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  if (( busy == 0 )); then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
echo "GPU 빔 $(ts)"
GPUS=8 bash z_training/tmux_train.sh "$RUN" || { echo "⚠️ 학습을 못 올림 $(ts)"; exit 1; }
echo "TRAIN_LAUNCHED $RUN $(ts)"
RUN=$RUN MODEL=$M EP=70 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e70 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state $M
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$RUN.log
echo "CHAIN_TMEM_DONE $(ts)"
