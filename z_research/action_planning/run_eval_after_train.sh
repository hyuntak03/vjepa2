#!/bin/bash
# 두 Push-T 팔 학습이 끝나면 (train.log "done") 평가를 자동으로 돈다 (v2, 2026-10-10: 목표 조각을 GPU 8 장에 나눈다).
#   tmux new-session -d -s eval_pusht "bash z_research/action_planning/run_eval_after_train.sh"
# 1) action 민감도 (scratch; post-FT 는 이미 함)   2) planning: 팔마다 목표 50 개를 4 조각 → GPU 0-3 (post-FT), 4-7 (scratch)
# 3) 합치기 merge_shards.py.  결과: exp_results/<tag>_merged.json, eval_after_train.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
export SDL_VIDEODRIVER=dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=z_research/action_planning/exp_results; mkdir -p $OUT; LOG=$OUT/eval_after_train.log
NS=4
echo "[$(date '+%m-%d %H:%M')] v2: scratch 학습 종료 대기" | tee -a $LOG
until grep -q "\] done$" z_training/runs/pusht_ar_scratch/train.log 2>/dev/null; do sleep 60; done
pkill -f "plan_pusht.py --run pusht_ar_postft --ckpt latest.pt --n_evals 50 --goal_H 5 --wm_chunk" 2>/dev/null   # 학습과 나눠 쓰던 느린 단일 run
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 20; done
echo "[$(date '+%m-%d %H:%M')] 평가 시작 (GPU 8 장, 팔당 ${NS} 조각)" | tee -a $LOG
CUDA_VISIBLE_DEVICES=7 python z_research/action_planning/action_sensitivity.py --run pusht_ar_scratch --ckpt latest.pt 2>&1 | grep -v -i warn > $OUT/action_sensitivity_pusht_ar_scratch.log
g=0
for r in pusht_ar_postft pusht_ar_scratch; do
  for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=$g python z_research/action_planning/plan_pusht.py --run $r --ckpt latest.pt --n_evals 50 --goal_H 5 --wm_chunk 150 --shard $i/$NS > $OUT/plan_${r}_shard${i}.log 2>&1 &
    g=$((g+1))
  done
done
wait
for r in pusht_ar_postft pusht_ar_scratch; do
  python z_research/action_planning/merge_shards.py $OUT/${r}_latest_gH5_dset_plan_s300o30 2>&1 | tee -a $LOG
done
echo "== action 민감도" | tee -a $LOG; tail -n 6 $OUT/action_sensitivity_pusht_ar_postft.log $OUT/action_sensitivity_pusht_ar_scratch.log | tee -a $LOG
echo "[$(date '+%m-%d %H:%M')] 전부 끝" | tee -a $LOG
