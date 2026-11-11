#!/bin/bash
# 2026-11-06 사용자 "L1 · 짝 없는 음의 손실 람다 1 · 짝 있는 음의 손실 람다 3 · 릴리즈 pft · 릴리즈 pft 짝 있는 음의 손실 람다 3 ㄱㄱ" — v4b (긴 가림) 다섯 판 채점.
#   판마다 e20 끝 → 시트 벤치 (chain_full_eval: VOBench v3 · IntPhys1 late · IntPhys2 late · real) + Surrogate + VOBench v1 late + 영역 분리 (Surrogate · VOB v3). 채점만.
#   setsid nohup bash z_training/chain_v4b_20261106.sh > z_training/logs/chain_v4b_20261106.log 2>&1 &
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
G="0 1 2 3 4 5 6 7"
for pair in v4b_ar_vw_state_postft:vast_ar_vw_state_v4b v4b_ar_vw_state_negimp:vast_ar_vw_state_v4b_negimp v4b_ar_vw_state_negpair_w3:vast_ar_vw_state_v4b_negpair_w3 \
            v4b_release_postft:vith_release_v4b v4b_release_negpair_w3:vith_release_v4b_negpair_w3; do
  R=${pair%%:*}; M=${pair##*:}
  echo "[chain] $(ts) $M 기다림 → 채점"
  RUN=$R MODEL=$M EP=20 GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain] $(ts) ⚠️ $M 학습이 e20 없이 끝남 — 건너뜀"; continue; }
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_late $M > z_training/logs/vob1late_eval_$M.log 2>&1
  for d in intphys1_surrogate vobench_v3; do
    SET="$TS" SUFFIX=_region GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $M > z_training/logs/region_${d}_$M.log 2>&1
  done
  echo "[chain] $(ts) $M 채점 끝"
done
MS="vith_release_v4vc vith_release_v4b vith_release_v4b_negpair_w3 vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negimp vast_ar_vw_state_v4b vast_ar_vw_state_v4b_negimp vast_ar_vw_state_v4b_negpair_w3"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py $MS
echo "CHAIN_V4B_DONE $(ts)"
