#!/bin/bash
# 2026-11-05 사용자 "{AR v4 negative loss lambda 1} 이렇게 학습좀 해봐 ... 결과 다시 알려줘": 짝 손실 판이 끝나면 채점 (학습 없음).
#   시트 다섯 벤치 = VOBench v3 · IntPhys1 late · IntPhys2 late · real (chain_full_eval) + Surrogate. (사용자: "VOBench v3 로 해야지" — v1 late 는 뺐다) + 물체 영역 (v3 · Surrogate)
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
TS='scoring.token_subset.modes=[all,object,far_bg] scoring.token_subset.object_radius=1 scoring.token_subset.far_min=3'
G="0 1 2 3 4 5 6 7"; R=v4vc_ar_vw_state_negpair; M=vast_ar_vw_state_v4vc_negpair
RUN=$R MODEL=$M EP=20 GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain] $(ts) ⚠️ e20 없이 끝남"; exit 1; }
GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
for d in intphys1_surrogate vobench_v3; do SET="$TS" SUFFIX=_region GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh $d $M > z_training/logs/region_${d}_$M.log 2>&1; done
MS="vast_ar_vw_state vast_ar_vw_state_v4vc vast_ar_vw_state_v4vc_negimp $M"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py $MS
echo "CHAIN_NEGPAIR_DONE $(ts)"
