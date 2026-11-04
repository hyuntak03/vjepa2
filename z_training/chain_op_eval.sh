#!/bin/bash
# =============================================================================
# 학습이 끝나면 시트 Object_permanence 의 세 벤치를 채점한다 (2026-11-02, 사용자: "IntPhys1 late occ · IntPhys2 late occ · real bench permanence 측정").
#   끝남 = run 폴더에 e<EP>.pt 가 있고 그 run 의 launch.py 프로세스가 없다.
#
#   RUN=v4vc_ar_vw_state_postft MODEL=vast_ar_vw_state_v4vc EP=20 GPU=6 \
#     setsid nohup bash z_training/chain_op_eval.sh > cvpr/results/eval/_logs_op_20261102/chain_<MODEL>.log 2>&1 &
#
#   MODEL = cvpr/registry/models.yaml 의 이름 (predictor_checkpoint 가 그 run 의 latest.pt)
#   채점: IntPhys 1 late (cvpr/eval/intphys1_late, 90 쌍 — 표는 O1 가림 30 쌍) · IntPhys 2 late final (500 쌍 — 표는 permanence 120) ·
#         real world permanence (190 쌍). 표: z_research/scripts/analysis/op_cells_vast.py
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN:?}" "${MODEL:?}" "${EP:=20}" "${GPU:=0}"
R=z_training/runs/$RUN
echo "[chain_op_eval] $(date '+%F %T') 기다림: $R/e$EP.pt + launch.py 종료 → $MODEL 채점 (GPU $GPU)"
until [[ -f $R/e$EP.pt ]] && ! pgrep -f "launch.py --fname .*/runs/$RUN/config.yaml" > /dev/null; do sleep 60; done
sleep 30
echo "[chain_op_eval] $(date '+%F %T') 학습 끝 → 채점 시작"
DS=intphys1_late_uni GPU_IDS=$GPU bash cvpr/eval/intphys1_late/run.sh "$MODEL"   # ⚠️ run.sh 기본 DS 는 다른 판 (intphys1_late) — 시트는 _uni
for d in intphys2_late_final real_world_permanence_uniform; do
  GPU_IDS=$GPU GPUS=1 bash cvpr/eval/testbed/run.sh "$d" "$MODEL"
done
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py vith vast_ar_vw_state "$MODEL"
echo "CHAIN_OP_DONE $MODEL $(date '+%F %T')"
