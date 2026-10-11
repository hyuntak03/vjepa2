#!/bin/bash
# Wall chain (2026-10-11): Push-T chain (chain_lowdata.sh) 이 "chain 전부 끝" 을 찍은 뒤 Wall 에서 post-FT 대 scratch.
#   tmux new-session -d -s chain_wall "bash z_research/action_planning/chain_wall.sh; exec bash"
# Push-T chain v2 와 같은 규약: lr 3e-4 두 팔, 1,600 step (25 epoch × 64, virtual_len 18,685), batch 36 × 8, 부분집합 seed 0 (포함 관계).
# 평가: DINO-WM plan_wall.yaml (goal random_state, CEM 300 × 10, goal_H 5, MPC 5 회), n_evals 128, seed 99, 8 조각 배치 CEM.
# Wall 은 목표를 env 가 무작위로 만들어 학습 데이터와 겹칠 걱정이 없다. 로그 exp_results/chain_wall.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
export SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=z_research/action_planning/exp_results; mkdir -p $OUT z_training/logs; LOG=$OUT/chain_wall.log
NE=${NE:-128}; NS=8
SHORT="optimization.epochs=25 optimization.warmup=1 optimization.lr=3.0e-4 meta.save_every_freq=5"
sub() { echo "data.datasets=[{\"name\":\"wall_single\",\"subset_frac\":$1,\"virtual_len\":18685}]"; }
RUNS=(
  "wall_ar_scratch_e25_p100|wall_ar_scratch|$(sub 1.0) $SHORT"
  "wall_ar_postft_e25_p100|wall_ar_postft|$(sub 1.0) $SHORT"
  "wall_ar_scratch_e25_p10|wall_ar_scratch|$(sub 0.1) $SHORT"
  "wall_ar_postft_e25_p10|wall_ar_postft|$(sub 0.1) $SHORT"
)
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }
evaluate() {
  local r=$1 pre=$OUT/${1}_latest_gH5_random_state_plan_s300o10_n${NE}_batched
  if [ -f ${pre}_merged.json ]; then say "$r 평가 이미 있음 — 건너뜀"; return; fi
  gpus_free; say "$r 평가 시작 (n_evals $NE, $NS 조각)"
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$i python z_research/action_planning/plan_pusht.py --run $r --ckpt latest.pt --n_evals $NE --goal_H 5 \
      --batched --wm_chunk 600 --shard $i/$NS > $OUT/plan_${r}_n${NE}_shard$i.log 2>&1 &
  done; wait
  python z_research/action_planning/merge_shards.py $pre 2>&1 | tee -a $LOG
  CUDA_VISIBLE_DEVICES=0 python z_research/action_planning/action_sensitivity.py --run $r --ckpt latest.pt 2>&1 | grep -v -i warn > $OUT/action_sensitivity_$r.log
  say "$r 민감도: $(grep -E 'zero − gt' $OUT/action_sensitivity_$r.log)"
}
say "Wall chain 시작 — Push-T chain 종료 대기"
until grep -q "chain 전부 끝" $OUT/chain_lowdata.log 2>/dev/null; do sleep 120; done
say "Push-T chain 끝"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r NAME CFG SETS <<< "$spec"
  if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
    say "$NAME 학습 이미 끝 — 건너뜀"
  else
    gpus_free; say "$NAME 학습 시작 (config $CFG, SET $SETS)"
    GPUS=8 NAME=$NAME SET="$SETS" bash z_training/train.sh $CFG > z_training/logs/$NAME.chain.log 2>&1
    if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
      say "$NAME 학습 끝: $(grep -oE '\[epoch [0-9]+\] val\[pusht[^]]*\] tf [0-9.]+ ar [0-9.]+' z_training/runs/$NAME/train.log | tail -1)"
    else
      say "⚠️ $NAME 학습 실패 — 로그 z_training/logs/$NAME.chain.log. 다음 run 으로 넘어간다"; continue
    fi
  fi
  evaluate $NAME
done
say "Wall chain 전부 끝"
grep -hE "success_rate" $LOG | tee -a $LOG
