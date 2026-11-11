#!/bin/bash
# 2026-11-09 전이 탐색 묶음 (사용자: "바꿔치기는 별로 · VRAM 남으니 최대한 다양하게 · 데이터를 바꾸면 L1 만 판도 · 다른 건 유지하고 IP2 와 real 이 올라야").
#   학습 5 판 동시 (GPU 8 × 6 = 48, 같은 GPU 공유) + 1 판 대기 (먼저 끝나는 판 뒤) → 판마다 끝나면 채점 (시트 네 벤치 · Surrogate, 채점은 잠금으로 한 번에 하나)
#   → 전부 끝나면 전이 판정 표 (z_research/scripts/analysis/transfer_table.py, 기준 = 같은 데이터의 L1 만 판).
#   다른 세션의 v4bc4f_ar_vw_state_rank_real (학습 끝 · 채점 비어 있음 — 레지스트리 항목이 없었다) 도 먼저 채점한다.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
LOCK=z_training/logs/.eval_lock_20261109
evalrun() {   # RUN MODEL — 학습 끝 기다림 → 잠금 → 네 벤치 · Surrogate
  ( until [[ -f z_training/runs/$1/e20.pt ]] && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml"; do
      if ! tmux has-session -t "=train_$1" 2>/dev/null && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml" && [[ ! -f z_training/runs/$1/e20.pt ]]; then echo "⚠️ $1 학습 세션이 e20 없이 사라짐 $(ts)"; exit 1; fi
      sleep 120; done
    flock -x $LOCK bash -c "RUN=$1 MODEL=$2 EP=20 GPU_IDS='0 1 2 3 4 5 6 7' bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1;
                            GPU_IDS='0 1 2 3 4 5 6 7' GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1"
    echo "EVAL_DONE $2 $(ts)" )
}
launch() { GPUS=8 bash z_training/tmux_train.sh $1 && echo "TRAIN_LAUNCHED $1 $(ts)"; sleep 25; }
# 0. 다른 세션 판 채점 (학습 끝)
evalrun v4bc4f_ar_vw_state_rank_real vast_ar_vw_state_v4bc4f_rank_real &
# 1. 학습 5 판
for r in v4bc4f_ar_vw_state_postft v4bc4_ar_vw_state_negimp_noswap v4bc4_ar_vw_state_negimp_vanish v4bc4_ar_vw_state_negimp_appear v4bc4_ar_vw_state_negimp_noswap_w075; do launch $r; done
evalrun v4bc4f_ar_vw_state_postft vast_ar_vw_state_v4bc4f &
evalrun v4bc4_ar_vw_state_negimp_noswap vast_ar_vw_state_v4bc4_negimp_noswap &
evalrun v4bc4_ar_vw_state_negimp_vanish vast_ar_vw_state_v4bc4_negimp_vanish &
evalrun v4bc4_ar_vw_state_negimp_appear vast_ar_vw_state_v4bc4_negimp_appear &
evalrun v4bc4_ar_vw_state_negimp_noswap_w075 vast_ar_vw_state_v4bc4_negimp_noswap_w075 &
# 2. 여섯째 판: 다섯 중 하나가 끝나면 (e20) 올린다
until ls z_training/runs/{v4bc4f_ar_vw_state_postft,v4bc4_ar_vw_state_negimp_noswap,v4bc4_ar_vw_state_negimp_vanish,v4bc4_ar_vw_state_negimp_appear,v4bc4_ar_vw_state_negimp_noswap_w075}/e20.pt >/dev/null 2>&1 \
      || [[ $(ls z_training/runs/{v4bc4f_ar_vw_state_postft,v4bc4_ar_vw_state_negimp_noswap,v4bc4_ar_vw_state_negimp_vanish,v4bc4_ar_vw_state_negimp_appear,v4bc4_ar_vw_state_negimp_noswap_w075}/e20.pt 2>/dev/null | wc -l) -ge 1 ]]; do sleep 120; done
launch v4bc4f_ar_vw_state_negimp_noswap
evalrun v4bc4f_ar_vw_state_negimp_noswap vast_ar_vw_state_v4bc4f_negimp_noswap &
wait
echo "TRANSFER_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py \
  vast_ar_vw_state_v4bc4_negimp=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_w075=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_noswap=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_noswap_w075=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_appear=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4f_negimp=vast_ar_vw_state_v4bc4f vast_ar_vw_state_v4bc4f_negimp_noswap=vast_ar_vw_state_v4bc4f \
  vast_ar_vw_state_v4bc4f_rank_real=vast_ar_vw_state_v4bc4f
echo "CHAIN_TRANSFER_DONE $(ts)"
