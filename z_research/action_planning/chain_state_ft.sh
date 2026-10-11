#!/bin/bash
# 상태 되먹임 AR (nat_ar_state_scratch) 위 action post-FT 대 scratch — Push-T · Wall (2026-10-11, 사용자 결정: 저데이터·Wall 별도 chain 대신 이것만).
# v2 (2026-10-11 20:5x, 사용자 지시 "2 까지 하고 평가하고 다시 3 으로"): 환경마다 두 팔을 먼저 학습하고, 두 팔 평가를 **GPU 4 장씩 동시에** 돌린 뒤 다음 환경.
#   tmux new-session -d -s chain_state "bash z_research/action_planning/chain_state_ft.sh; exec bash"
# nat_ar_state_scratch 가 "done" 을 찍으면: 4 run 을 순서대로 학습 (8 GPU) → planning 평가 (8 조각 배치 CEM, n_evals 128) → 조각 합치기 → action 민감도.
# 네 run: kind ar_state, lr 3e-4, 1,600 step (25 epoch × 64), warmup 1, batch 36 × 8, 데이터 100 %. 초기값만 다르다 (무작위 / nat_ar_state_scratch).
# 평가: Push-T = DINO-WM plan_pusht (goal dset, CEM 300 × 30), Wall = plan_wall (goal random_state, CEM 300 × 10). goal_H 5, MPC 5 회, seed 99.
# 이미 끝난 run (train.log done / merged.json) 은 건너뛴다 → 다시 올리면 이어 간다. 로그 exp_results/chain_state_ft.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
export SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=z_research/action_planning/exp_results; mkdir -p $OUT z_training/logs; LOG=$OUT/chain_state_ft.log
WAIT_FOR=${WAIT_FOR:-nat_ar_state_scratch}
NE=${NE:-128}; NS=4
INIT=/nas2/data/takhyun03/project/2026/world_model/vjepa2/z_training/runs/nat_ar_state_scratch/latest.pt
COMMON="model.predictor.kind=ar_state optimization.epochs=25 optimization.warmup=1 optimization.lr=3.0e-4 meta.save_every_freq=5"
RUNS=(
  "pusht_ars_scratch|pusht_ar_scratch|$COMMON"
  "pusht_ars_postft|pusht_ar_postft|$COMMON model.init_predictor=$INIT"
  "wall_ars_scratch|wall_ar_scratch|$COMMON data.datasets=[{\"name\":\"wall_single\",\"virtual_len\":18685}]"
  "wall_ars_postft|wall_ar_postft|$COMMON model.init_predictor=$INIT data.datasets=[{\"name\":\"wall_single\",\"virtual_len\":18685}]"
)
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }
evaluate() {
  local r=$1 pre
  case $r in wall_*) pre=$OUT/${r}_latest_gH5_random_state_plan_s300o10_n${NE}_batched ;; *) pre=$OUT/${r}_latest_gH5_dset_plan_s300o30_n${NE}_batched ;; esac
  if [ -f ${pre}_merged.json ]; then say "$r 평가 이미 있음 — 건너뜀"; return; fi
  say "$r 평가 시작 (n_evals $NE, $NS 조각, GPU ${2:-0}..$(( ${2:-0}+NS-1 )))"
  local g0=${2:-0}
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$((g0+i)) python z_research/action_planning/plan_pusht.py --run $r --ckpt latest.pt --n_evals $NE --goal_H 5 \
      --batched --wm_chunk 600 --shard $i/$NS > $OUT/plan_${r}_n${NE}_shard$i.log 2>&1 &
  done; wait
  python z_research/action_planning/merge_shards.py $pre 2>&1 | tee -a $LOG
  CUDA_VISIBLE_DEVICES=${2:-0} python z_research/action_planning/action_sensitivity.py --run $r --ckpt latest.pt 2>&1 | grep -v -i warn > $OUT/action_sensitivity_$r.log
  say "$r 민감도: $(grep -E 'zero − gt' $OUT/action_sensitivity_$r.log)"
}
say "chain 시작 — $WAIT_FOR 종료 대기"
until grep -q "\] done$" z_training/runs/$WAIT_FOR/train.log 2>/dev/null; do sleep 120; done
say "$WAIT_FOR 끝 — post-FT 초기값 $INIT"
train_one() {   # $1 = "NAME|CFG|SETS"
  IFS='|' read -r NAME CFG SETS <<< "$1"
  if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then say "$NAME 학습 이미 끝 — 건너뜀"; return 0; fi
  if [ -n "${ATTACH:-}" ] && [ "$NAME" = "$ATTACH" ]; then          # chain v1 에서 이미 돌고 있는 학습: 끝나기만 기다린다
    say "$NAME 학습 진행 중 (chain v1 에서 시작) — 끝날 때까지 대기"
    until grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; do sleep 60; done
  else
    gpus_free; say "$NAME 학습 시작 (config $CFG)"
    GPUS=8 NAME=$NAME SET="$SETS" bash z_training/train.sh $CFG > z_training/logs/$NAME.chain.log 2>&1
  fi
  if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
    say "$NAME 학습 끝: $(grep -oE '\[epoch [0-9]+\] val\[pusht[^]]*\] tf [0-9.]+ ar [0-9.]+' z_training/runs/$NAME/train.log | tail -1)"
  else
    say "⚠️ $NAME 학습 실패 — 로그 z_training/logs/$NAME.chain.log"
  fi
}
for pair in "0 1" "2 3"; do                                       # (Push-T 두 팔) → 평가, (Wall 두 팔) → 평가
  set -- $pair
  train_one "${RUNS[$1]}"; train_one "${RUNS[$2]}"
  A=$(echo "${RUNS[$1]}" | cut -d'|' -f1); B=$(echo "${RUNS[$2]}" | cut -d'|' -f1)
  gpus_free
  evaluate $A 0 & evaluate $B 4 & wait                             # GPU 0-3 / 4-7 동시에
done
say "chain 전부 끝"
grep -hE "success_rate" $LOG | tee -a $LOG
