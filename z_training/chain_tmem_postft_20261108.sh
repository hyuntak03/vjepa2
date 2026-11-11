#!/bin/bash
# 2026-11-08 하이브리드 e43 → v4bc4 L1 만 (v4bc4_tmem_postft) 채점: e20 + 학습 끝 → 시트 네 벤치 · Surrogate (8 장) → 시트 표 (AR → v4bc4 L1 과 나란히) → 공식 IP1 · IP2.
#   공식 채점은 물체 영역 사슬 (chain_objregion_20261108) 의 공식 채점이 끝난 뒤 (같은 8 장을 동시에 쓰지 않게).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_tmem_postft; M=vast_ar_vw_state_tmem_v4bc4
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_tmem $M
while pgrep -f '[c]hain_objregion_20261108.sh' > /dev/null; do sleep 120; done
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$RUN.log
echo "CHAIN_TMEM_POSTFT_DONE $(ts)"
