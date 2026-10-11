#!/bin/bash
# Push-T 저데이터 chain v2 (2026-10-10, 사용자 결정 "16 시간 안으로"): action 데이터 1 % · 10 % · 100 % 에서 post-FT 대 scratch.
#   v1 (48 h: 6,400 step × 7 run, post-FT lr 5e-5 와 3e-4 둘 다) 을 줄였다 — **lr 은 두 팔 모두 3e-4** (초기값 하나만 다르다),
#   **1,600 step (25 epoch × 64)**, warmup 1 epoch, 평가 목표 128 개. 결론은 "제한된 학습 예산 (1,600 step) 에서" 로 한정한다.
#   tmux new-session -d -s chain_pusht "bash z_research/action_planning/chain_lowdata.sh; exec bash"
# 앞 run (nat_ar_state_scratch) 이 "done" 을 찍을 때까지 기다린 뒤, 아래 RUNS 를 순서대로: 학습 (8 GPU) → planning 평가 (8 조각, 배치 CEM)
# → 조각 합치기 → action 민감도. 이미 끝난 run (train.log 에 done) 은 학습을 건너뛰고, 평가가 있으면 평가도 건너뛴다 → 다시 실행해도 이어 간다.
# 모든 run: 1,600 step (25 epoch × 64), batch 36 × 8, 부분집합은 고정 seed 0 · 포함 관계 (1 % ⊂ 10 %), virtual_len 18,685 로 epoch 길이 동일.
# 평가: n_evals 128 (DINO-WM 프로토콜, goal_H 5, CEM 300 × 30, MPC 5 회), seed 99. 로그 z_research/action_planning/exp_results/chain_lowdata.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
export SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=z_research/action_planning/exp_results; mkdir -p $OUT z_training/logs; LOG=$OUT/chain_lowdata.log
WAIT_FOR=${WAIT_FOR:-nat_ar_state_scratch}
NE=${NE:-128}; NS=8
SHORT="optimization.epochs=25 optimization.warmup=1 optimization.lr=3.0e-4 meta.save_every_freq=5"
sub() { echo "data.datasets=[{\"name\":\"pusht_noise\",\"subset_frac\":$1,\"virtual_len\":18685}]"; }
# NAME | config | SET
RUNS=(
  "pusht_ar_scratch_e25_p10|pusht_ar_scratch|$(sub 0.1) $SHORT"
  "pusht_ar_postft_e25_p10|pusht_ar_postft|$(sub 0.1) $SHORT"
  "pusht_ar_scratch_e25_p01|pusht_ar_scratch|$(sub 0.01) $SHORT"
  "pusht_ar_postft_e25_p01|pusht_ar_postft|$(sub 0.01) $SHORT"
  "pusht_ar_scratch_e25_p100|pusht_ar_scratch|$SHORT"
  "pusht_ar_postft_e25_p100|pusht_ar_postft|$SHORT"
)
EVAL_ONLY_AFTER=(pusht_ar_scratch pusht_ar_postft)   # 시간이 남으면: 기존 6,400 step 100 % 두 팔을 같은 n_evals 로 재평가 (맨 끝)
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }
evaluate() {   # $1 = run
  local r=$1 pre=$OUT/${1}_latest_gH5_dset_plan_s300o30_n${NE}_batched
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
say "chain 시작 — $WAIT_FOR 종료 대기"
until grep -q "\] done$" z_training/runs/$WAIT_FOR/train.log 2>/dev/null; do sleep 120; done
say "$WAIT_FOR 끝"
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
for r in "${EVAL_ONLY_AFTER[@]}"; do evaluate $r; done
say "chain 전부 끝"
grep -hE "success_rate" $LOG | tee -a $LOG
