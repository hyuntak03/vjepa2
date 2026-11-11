#!/bin/bash
# 2026-11-08 물체 영역 음의 손실 두 판 (둘 다 GPU 8 장 × 6 으로 동시 학습): A = AR e70 → objregion, B = 하이브리드 e43 → objregion.
#   채점은 GPU 를 4 장씩 나눈다 (A 0–3 · B 4–7).
#   각자 e20 + 학습 끝 → 시트 네 벤치 · Surrogate (자기 GPU 4 장) → 둘 다 끝나면 시트 표 → 공식 IP1 · IP2 (8 장, 차례로).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
one() {   # RUN MODEL GPU_IDS
  RUN=$1 MODEL=$2 EP=20 GPU_IDS="$3" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1 || { echo "⚠️ $1 e20 없이 끝남 $(ts)"; return 1; }
  GPU_IDS="$3" GPUS=4 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1
  echo "SHEET_DONE $2 $(ts)"
}
one v4bc4_ar_vw_state_negimp_objregion vast_ar_vw_state_v4bc4_negimp_objregion "0 1 2 3" &
one v4bc4_tmem_negimp_objregion vast_ar_vw_state_tmem_v4bc4_negimp_objregion "4 5 6 7" &
wait
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4_negimp_objregion vast_ar_vw_state_tmem_v4bc4_negimp vast_ar_vw_state_tmem_v4bc4_negimp_objregion
for r in v4bc4_ar_vw_state_negimp_objregion v4bc4_tmem_negimp_objregion; do
  GPUS=8 bash z_training/eval_bench.sh $r > z_training/logs/eval_bench_$r.log 2>&1
  echo "$r:"; grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$r.log
done
echo "CHAIN_OBJREGION_DONE $(ts)"
