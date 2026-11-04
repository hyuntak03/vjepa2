#!/bin/bash
# =============================================================================
# 학습이 끝나면 시트 Object_permanence 의 네 벤치를 GPU 8 장 한꺼번에 채점한다 (2026-11-03).
#   끝남 = run 폴더에 e<EP>.pt 가 있고 그 run 의 launch.py 프로세스가 없다.
#   채점: VOBench v3 (840 블록) · IntPhys 1 late (DS=intphys1_late_uni, 표는 O1 가림 30 쌍) · IntPhys 2 late final · real world permanence.
#   (사용자 11-02: "모델 별로 gpu 나눠쓰지 말고 8 장 한 번에")
#
#   RUN=natural_ar_vw_state_v4vc MODEL=vast_ar_vw_state_v4vc_scratch EP=70 \
#     setsid nohup bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_<MODEL>.log 2>&1 &
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN:?}" "${MODEL:?}" "${EP:=70}" "${GPU_IDS:=0 1 2 3 4 5 6 7}"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
R=z_training/runs/$RUN
echo "[chain_full_eval] $(ts) 기다림: $R/e$EP.pt + launch.py 종료 → $MODEL 채점 (GPU $GPU_IDS)"
until [[ -f $R/e$EP.pt ]] && ! ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$RUN/config\.yaml"; do
  if ! tmux has-session -t "=train_$RUN" 2>/dev/null && ! ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$RUN/config\.yaml"; then
    echo "[chain_full_eval] $(ts) ⚠️ 학습 세션이 e$EP.pt 없이 사라졌다 — 채점 안 함"; exit 1
  fi
  sleep 120
done
sleep 30
NG=$(wc -w <<< "$GPU_IDS")
echo "[chain_full_eval] $(ts) 학습 끝 → 채점 시작"
GPU_IDS="$GPU_IDS" GPUS=$NG bash cvpr/eval/testbed/run.sh vobench_v3 "$MODEL"
DS=intphys1_late_uni GPU_IDS="$GPU_IDS" bash cvpr/eval/intphys1_late/run.sh "$MODEL"   # ⚠️ run.sh 기본 DS 는 다른 판 (intphys1_late)
for d in intphys2_late_final real_world_permanence_uniform; do
  GPU_IDS="$GPU_IDS" GPUS=$NG bash cvpr/eval/testbed/run.sh "$d" "$MODEL"
done
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py vast_ar_vw_state vast_ar_vw_state_v4vc "$MODEL"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py vast_ar_vw_state vast_ar_vw_state_v4vc "$MODEL"
echo "CHAIN_FULL_EVAL_DONE $MODEL $(ts)"
