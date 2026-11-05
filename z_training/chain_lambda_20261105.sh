#!/bin/bash
# 2026-11-05 사용자 "0.5, 2 이렇게 좀 지금 돌려봐 ... 그리고 이거 benchmark 들 평가 좀": negimp λ ablation 네 판이 끝나면 차례로 채점.
#   시트 다섯 벤치 (VOBench v3 · IntPhys1 late · IntPhys2 late · real · Surrogate) + 영역 분리 (Surrogate · VOB v3) + VOBench v1 late
#   + 릴리즈 판은 미래만 정답 (Surrogate · VOB v3 · real). 채점만 (학습 없음).
#   setsid nohup bash z_training/chain_lambda_20261105.sh > z_training/logs/chain_lambda_20261105.log 2>&1 &
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
G="0 1 2 3 4 5 6 7"
for pair in v4vc_release_negimp_w05:vith_release_v4vc_negimp_w05 v4vc_release_negimp_w2:vith_release_v4vc_negimp_w2 \
            v4vc_ar_vw_state_negimp_w05:vast_ar_vw_state_v4vc_negimp_w05 v4vc_ar_vw_state_negimp_w2:vast_ar_vw_state_v4vc_negimp_w2; do
  R=${pair%%:*}; M=${pair##*:}
  echo "[chain] $(ts) $M 기다림 → 채점"
  RUN=$R MODEL=$M EP=20 GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain] $(ts) ⚠️ $M 학습이 e20 없이 끝남 — 건너뜀"; continue; }
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_late $M > z_training/logs/vob1late_eval_$M.log 2>&1
  for d in intphys1_surrogate vobench_v3; do
    SET="$TS" SUFFIX=_region GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $M > z_training/logs/region_${d}_$M.log 2>&1
  done
  if [[ $M == vith_* ]]; then
    for d in intphys1_surrogate vobench_v3 real_world_permanence_uniform; do
      X="surprise.target_future_only=true surprise.copy_baseline=true"; [[ $d != real_world_permanence_uniform ]] && X="$X $TS"
      SET="$X" SUFFIX=_futonly GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $M > z_training/logs/futonly_${d}_$M.log 2>&1
    done
  fi
  echo "[chain] $(ts) $M 채점 끝"
done
MS="vith_release_v4vc vith_release_v4vc_negimp_w05 vith_release_v4vc_negimp vith_release_v4vc_negimp_w2 vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negimp_w05 vast_ar_vw_state_v4vc_negimp vast_ar_vw_state_v4vc_negimp_w2"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py $MS
echo "CHAIN_LAMBDA_DONE $(ts)"
