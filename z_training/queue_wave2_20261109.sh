#!/bin/bash
# 2026-11-09 전이 탐색 2 묶음 (손실 설계) + 1 묶음의 나타남만 판 (02:23 시작 때 OOM 으로 죽어 맨 앞에 다시) + v4bc4 · SSv2 20 % 섞기 두 판 (L1 만 · 음의 손실) — GPU 메모리가 비는 대로 한 판씩 올린다 (모든 GPU < 76 GB 가 1 분 간격 3 번 연속),
#   판마다 학습 끝나면 채점 (잠금으로 한 번에 하나) → 전부 끝나면 전이 판정 표 (기준 = v4bc4 L1 만).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
LOCK=z_training/logs/.eval_lock_20261109
RUNS="v4bc4_ar_vw_state_negimp_appear v4bc4_ar_vw_state_negpair_hinge_w075 v4bc4ssv2_ar_vw_state_postft v4bc4ssv2_ar_vw_state_negimp_noswap v4bc4_ar_vw_state_negimp_noswap_m065 v4bc4_ar_vw_state_negpair_lin_l2 v4bc4_ar_vw_state_negpair_hinge_w090 v4bc4_ar_vw_state_negpair_lin_l1"
model() { local n=${1/_ar_vw_state/}; echo "vast_ar_vw_state_${n%_postft}"; }
evalrun() {
  ( until [[ -f z_training/runs/$1/e20.pt ]] && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml"; do
      if ! tmux has-session -t "=train_$1" 2>/dev/null && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml" && [[ ! -f z_training/runs/$1/e20.pt ]]; then echo "⚠️ $1 학습이 e20 없이 끝남 $(ts)"; exit 1; fi
      sleep 120; done
    M=$(model $1)
    flock -x $LOCK bash -c "RUN=$1 MODEL=$M EP=20 GPU_IDS='0 1 2 3 4 5 6 7' bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1;
                            GPU_IDS='0 1 2 3 4 5 6 7' GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1"
    echo "EVAL_DONE $M $(ts)" )
}
for r in $RUNS; do
  ok=0
  while (( ok < 3 )); do
    full=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 76000' | wc -l)
    if (( full == 0 )); then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
  GPUS=8 bash z_training/tmux_train.sh $r && echo "TRAIN_LAUNCHED $r $(ts)"
  evalrun $r &
  sleep 300        # 메모리가 다 잡힐 때까지
done
wait
echo "TRANSFER_TABLE_W2 $(ts)"
args=""; for r in $RUNS; do
  case $r in
    v4bc4ssv2_ar_vw_state_negimp_noswap) args="$args $(model $r)=vast_ar_vw_state_v4bc4ssv2" ;;   # 같은 섞기의 L1 만 판이 기준
    *) args="$args $(model $r)=vast_ar_vw_state_v4bc4" ;;                                           # SSv2 L1 판은 v4bc4 L1 대비 (섞기 자체의 효과)
  esac; done
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py vast_ar_vw_state_v4bc4_negimp=vast_ar_vw_state_v4bc4 $args
echo "QUEUE_WAVE2_DONE $(ts)"
