#!/bin/bash
# 2026-10-12: Push-T 25 epoch 두 팔 L1 재평가 (손으로 띄운 scratch 5 조각 + post-FT 3 조각) 가 끝나면 합치고,
#   내려 두었던 chain_e100 (e100 학습 이어 가기 → 평가 → L1 재평가) · chain_ksh 를 다시 올린다.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
E=z_research/action_planning/exp_results; LOG=$E/chain_ars_e100.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
say "Push-T L1 재평가 (손으로 띄움) 대기"
while pgrep -f "plan_pusht.py --run pusht_ars_(scratch|postft) --ckpt" > /dev/null; do sleep 30; done
for r in scratch postft; do python z_research/action_planning/merge_shards.py $E/pusht_ars_${r}_latest_gH5_dset_plan_s300o30_n128_batched_l1 2>&1 | tee -a $LOG; done
tmux new-session -d -s chain_e100 "bash z_research/action_planning/chain_ars_e100.sh; exec bash"
sleep 5
tmux new-session -d -s chain_ksh "bash z_research/action_planning/chain_full_ksh.sh; exec bash"
say "chain_e100 · chain_ksh 다시 올림"
