#!/bin/bash
# 2026-11-08 시간 누적 하이브리드 e43 → v4bc4 음의 손실 이어 학습 채점 (사용자 "real 이나 IntPhys2 에서 더 좋아지는지 봐야겠다"):
#   e20 + 학습 끝 → 시트 네 벤치 (VOB v3 · IP1 late · IP2 late · real) → Surrogate → 시트 표 (기준판 AR + 음의 손실 · 하이브리드 e43 과 나란히) → 공식 IP1 · IP2.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_tmem_negimp; M=vast_ar_vw_state_tmem_v4bc4_negimp
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_tmem $M
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$RUN.log
echo "CHAIN_TMEM_NEGIMP_DONE $(ts)"
