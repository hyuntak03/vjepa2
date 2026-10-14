#!/bin/bash
# pusht_ars_scratch 를 100 epoch 로 (2026-10-12, 사용자 지시 "ars_scratch 가지고 100 epoch 학습하는 것도 뒤에 chain 으로").
#   tmux new-session -d -s chain_e100 "bash z_research/action_planning/chain_ars_e100.sh; exec bash"
# 왜: 25 epoch 에서 scratch 0.05 대 post-FT 0.55 (MPC 1 회차). 이전 render 팔 (pusht_ar_scratch, 100 epoch) 은 0.60.
#   이 run 은 pusht_ar_scratch 와 **kind (ar_state) 하나만** 다르다 — epochs 100 · warmup 5 · lr 3e-4 · batch 36 × 8 · 데이터 100 %
#   (config pusht_ar_state_scratch = extends pusht_ar_scratch). 25 epoch 판과는 길이 (와 그에 딸린 warmup) 만 다르다.
# chain_state_ft.sh (Wall 까지) 가 끝나고 GPU 가 비면 시작 → 학습 (8 GPU, ~6 h) → planning 평가 (8 조각, n_evals 128) → 합치기 → action 민감도.
# 이미 끝난 단계는 건너뛴다. 로그 exp_results/chain_ars_e100.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
export SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=12 MKL_NUM_THREADS=12   # 2026-10-12: CPU 과다구독 방지
OUT=z_research/action_planning/exp_results; mkdir -p $OUT z_training/logs; LOG=$OUT/chain_ars_e100.log
NAME=pusht_ars_scratch_e100; CFG=pusht_ar_state_scratch
NE=${NE:-128}; NS=8
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }

say "chain 시작 — chain_state_ft.sh 종료 대기"
while pgrep -f "bash z_research/action_planning/chain_state_ft.sh" > /dev/null; do sleep 120; done
while pgrep -f "plan_pusht.py --run wall_ars" > /dev/null; do sleep 60; done   # 2026-10-12: 손으로 다시 돌린 Wall 평가도 기다린다 (경합 방지)
say "chain_state_ft.sh 끝"

if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
  say "$NAME 학습 이미 끝 — 건너뜀"
else
  gpus_free; say "$NAME 학습 시작 (config $CFG, 100 epoch = 6,400 step)"
  GPUS=8 NAME=$NAME bash z_training/train.sh $CFG > z_training/logs/$NAME.chain.log 2>&1
fi
if ! grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
  say "⚠️ $NAME 학습 실패 — 로그 z_training/logs/$NAME.chain.log"; exit 1
fi
say "$NAME 학습 끝: $(grep -oE '\[epoch [0-9]+\] val\[pusht[^]]*\] tf [0-9.]+ ar [0-9.]+' z_training/runs/$NAME/train.log | tail -1)"

pre=$OUT/${NAME}_latest_gH5_dset_plan_s300o30_n${NE}_batched_l1     # 2026-10-12: CEM 점수 L1 (plan_pusht.py 기본)
if [ -f ${pre}_merged.json ]; then
  say "$NAME 평가 이미 있음 — 건너뜀"
else
  gpus_free; say "$NAME 평가 시작 (n_evals $NE, $NS 조각, GPU 0..$((NS-1)))"
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$i python z_research/action_planning/plan_pusht.py --run $NAME --ckpt latest.pt --n_evals $NE --goal_H 5 \
      --batched --wm_chunk 600 --shard $i/$NS > $OUT/plan_${NAME}_n${NE}_shard$i.log 2>&1 &
  done; wait
  python z_research/action_planning/merge_shards.py $pre 2>&1 | tee -a $LOG
fi
CUDA_VISIBLE_DEVICES=0 python z_research/action_planning/action_sensitivity.py --run $NAME --ckpt latest.pt 2>&1 | grep -v -i warn > $OUT/action_sensitivity_$NAME.log
say "$NAME 민감도: $(grep -E 'zero − gt' $OUT/action_sensitivity_$NAME.log)"

# ── 2026-10-12 사용자 결정 "CEM 점수는 L1": 25 epoch 네 팔도 L1 로 다시 평가 (그 전 결과는 MSE). 팔마다 GPU 4 장, 두 팔씩 동시에
reeval() {   # $1 = run, $2 = 첫 GPU
  local r=$1 g0=$2 pre
  case $r in wall_*) pre=$OUT/${r}_latest_gH5_random_state_plan_s300o10_n${NE}_batched_l1 ;; *) pre=$OUT/${r}_latest_gH5_dset_plan_s300o30_n${NE}_batched_l1 ;; esac
  if [ -f ${pre}_merged.json ]; then say "$r L1 평가 이미 있음 — 건너뜀"; return; fi
  say "$r L1 평가 시작 (GPU $g0..$((g0+3)))"
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$((g0+i)) python z_research/action_planning/plan_pusht.py --run $r --ckpt latest.pt --n_evals $NE --goal_H 5 \
      --batched --wm_chunk 600 --shard $i/4 > $OUT/plan_${r}_n${NE}_l1_shard$i.log 2>&1 &
  done; wait
  python z_research/action_planning/merge_shards.py $pre 2>&1 | tee -a $LOG
}
gpus_free
reeval pusht_ars_scratch 0 & reeval pusht_ars_postft 4 & wait
reeval wall_ars_scratch 0 & reeval wall_ars_postft 4 & wait
say "chain 끝"
