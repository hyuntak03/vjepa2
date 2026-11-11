#!/bin/bash
# 2026-11-09 물체 토큰 음의 손실 margin 1.5 두 판 (둘 다 GPU 8 × 6 동시 학습): A = AR e70, B = 하이브리드 e43.
#   각자 e20 + 학습 끝 → 시트 네 벤치 · Surrogate (채점 GPU A 0–3 · B 4–7) → 둘 다 끝나면 시트 표 → 공식 IP1 · IP2 (8 장, 차례로,
#   하이브리드 L1 만 사슬 chain_tmem_postft_20261108 이 끝난 뒤).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
one() {   # RUN MODEL GPU_IDS
  RUN=$1 MODEL=$2 EP=20 GPU_IDS="$3" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1 || { echo "⚠️ $1 e20 없이 끝남 $(ts)"; return 1; }
  GPU_IDS="$3" GPUS=4 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1
  echo "SHEET_DONE $2 $(ts)"
}
one v4bc4_ar_vw_state_negimp_objregion_m15 vast_ar_vw_state_v4bc4_negimp_objregion_m15 "0 1 2 3" &
one v4bc4_tmem_negimp_objregion_m15 vast_ar_vw_state_tmem_v4bc4_negimp_objregion_m15 "4 5 6 7" &
wait
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4_negimp_objregion vast_ar_vw_state_v4bc4_negimp_objregion_m15 vast_ar_vw_state_tmem_v4bc4_negimp vast_ar_vw_state_tmem_v4bc4_negimp_objregion_m15
while pgrep -f '[c]hain_tmem_postft_20261108.sh' > /dev/null; do sleep 120; done
for r in v4bc4_ar_vw_state_negimp_objregion_m15 v4bc4_tmem_negimp_objregion_m15; do
  GPUS=8 bash z_training/eval_bench.sh $r > z_training/logs/eval_bench_$r.log 2>&1
  echo "$r:"; grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$r.log
done
echo "CHAIN_OBJREGION_M15_DONE $(ts)"
