#!/bin/bash
# 2026-11-08 20:0x KST 사용자 "지금 여기서 스탑하고 benchmark 들 쭉 돌려봐 (VOBench v3, IntPhys1 surrogate, IntPhys2 late occ, real world bench)":
#   시간 누적 하이브리드 scratch 를 epoch 43 (latest.pt, step 12,642) 에서 멈추고 네 벤치를 GPU 8 장으로 채점 → 기준판 (AR scratch e70) 과 시트 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
M=vast_ar_vw_state_tmem; G="0 1 2 3 4 5 6 7"
for d in vobench_v3 intphys1_surrogate intphys2_late_final real_world_permanence_uniform; do
  echo "[$(ts)] $d 시작"
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $M > z_training/logs/tmem_e43_$d.log 2>&1 || echo "⚠️ $d 실패 (z_training/logs/tmem_e43_$d.log)"
done
echo "SHEET_TABLE $(ts)"
P=/venv/vjepa/bin/python; A=z_research/scripts/analysis
$P $A/op_cells_vob3.py vast_ar_vw_state $M --tsv
$P $A/op_cells_surrogate.py vast_ar_vw_state $M --tsv
$P $A/op_cells_vast.py vast_ar_vw_state $M --tsv
echo "CHAIN_TMEM_E43_DONE $(ts)"
