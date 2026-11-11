#!/bin/bash
# 2026-11-09 전이 탐색 3 묶음 + 2 묶음 남은 판 (2 묶음 대기열을 대신한다 — 이미 올라간 2 묶음 4 판의 채점은 그 대기열의 채점 하위 프로세스가 계속 한다).
#   1 묶음 통과판 '사라짐 음성만' 을 바탕으로 한 판을 앞에, 2 묶음 손실 설계 남은 판을 뒤에. 메모리가 비는 대로 한 판씩 (모든 GPU < 76 GB, 1 분 × 3).
#   맨 처음: 통과판 (v4bc4 사라짐만) 의 공식 IntPhys1 · IntPhys2 (채점 잠금).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
LOCK=z_training/logs/.eval_lock_20261109
RUNS="v4bc4f_ar_vw_state_negimp_vanish v4bc4_ar_vw_state_negimp_vanish_app02 v4bc4ssv2_ar_vw_state_negimp_vanish v4bc4_ar_vw_state_negimp_vanish_w125 v4bc4_ar_vw_state_negimp_noswap_m065 v4bc4_ar_vw_state_negpair_lin_l2 v4bc4_ar_vw_state_negimp_vanish_p07 v4bc4_ar_vw_state_negpair_hinge_w090 v4bc4_ar_vw_state_negpair_lin_l1"
W2DONE="v4bc4_ar_vw_state_negimp_appear v4bc4_ar_vw_state_negpair_hinge_w075 v4bc4ssv2_ar_vw_state_postft v4bc4ssv2_ar_vw_state_negimp_noswap"
model() { local n=${1/_ar_vw_state/}; echo "vast_ar_vw_state_${n%_postft}"; }
base() { case $1 in v4bc4f_*) echo vast_ar_vw_state_v4bc4f ;; v4bc4ssv2_ar_vw_state_postft) echo vast_ar_vw_state_v4bc4 ;; v4bc4ssv2_*) echo vast_ar_vw_state_v4bc4ssv2 ;; *) echo vast_ar_vw_state_v4bc4 ;; esac; }
evalrun() {
  ( until [[ -f z_training/runs/$1/e20.pt ]] && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml"; do
      if ! tmux has-session -t "=train_$1" 2>/dev/null && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$1/config\.yaml" && [[ ! -f z_training/runs/$1/e20.pt ]]; then echo "⚠️ $1 학습이 e20 없이 끝남 $(ts)"; exit 1; fi
      if grep -qE "OutOfMemoryError|rank 실패" z_training/runs/$1/train.log 2>/dev/null; then echo "⚠️ $1 학습 실패 (OOM / rank) $(ts)"; exit 1; fi
      sleep 120; done
    M=$(model $1)
    flock -x $LOCK bash -c "RUN=$1 MODEL=$M EP=20 GPU_IDS='0 1 2 3 4 5 6 7' bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1;
                            GPU_IDS='0 1 2 3 4 5 6 7' GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1"
    echo "EVAL_DONE $M $(ts)" )
}
( flock -x $LOCK bash -c "GPUS=8 bash z_training/eval_bench.sh v4bc4_ar_vw_state_negimp_vanish > z_training/logs/eval_bench_v4bc4_ar_vw_state_negimp_vanish.log 2>&1"
  echo "OFFICIAL_DONE v4bc4_ar_vw_state_negimp_vanish $(ts)"; grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_v4bc4_ar_vw_state_negimp_vanish.log ) &
for r in $RUNS; do
  ok=0
  while (( ok < 3 )); do
    full=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 76000' | wc -l)
    if (( full == 0 )); then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
  GPUS=8 bash z_training/tmux_train.sh $r && echo "TRAIN_LAUNCHED $r $(ts)"
  evalrun $r &
  sleep 300
done
wait
for r in $W2DONE; do until [[ -f cvpr/results/eval/testbed/intphys1_surrogate_$(model $r)/per_block.json ]] || ! tmux has-session -t "=train_$r" 2>/dev/null && [[ ! -f z_training/runs/$r/e20.pt ]]; do sleep 120; done; done
echo "TRANSFER_TABLE_W23 $(ts)"
args="vast_ar_vw_state_v4bc4_negimp_vanish=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_noswap=vast_ar_vw_state_v4bc4"
for r in $W2DONE $RUNS; do args="$args $(model $r)=$(base $r)"; done
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py $args
echo "QUEUE_WAVE3_DONE $(ts)"
