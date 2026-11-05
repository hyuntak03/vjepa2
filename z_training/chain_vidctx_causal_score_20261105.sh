#!/bin/bash
# 2026-11-05 사용자: 블록 문맥판 이어 학습 (chain_oneshot_causal_20261104.sh 의 ② · ③) 은 "일단 하지 말아봐" → 그 체인을 멈추고,
#   ① 영상 문맥 + 블록 인과 판 (natural_ar_vw_state_oneshot_vidctx_causal) 의 e70 채점만 한다 (시트 다섯 벤치 + 표). 학습은 올리지 않는다.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
M=vast_ar_oneshot_vidctx_causal
RUN=natural_ar_vw_state_oneshot_vidctx_causal MODEL=$M EP=70 bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain] $(ts) ⚠️ 학습이 e70 없이 끝남"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
MS="vast_tube_cd64 vast_ar_oneshot_vidctx $M vast_ar_vw_state"
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vob3.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_surrogate.py $MS
/venv/vjepa/bin/python z_research/scripts/analysis/op_cells_vast.py $MS
echo "VIDCTX_CAUSAL_SCORE_DONE $(ts)"
