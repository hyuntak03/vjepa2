#!/bin/bash
# 2026-11-09 13:1x KST 사용자 "GPU 별로 안 쓰는데 평가 한 번에 올리면 안 되나": 남은 10 판 채점을 4 개씩 병렬 (잠금 없이).
#   판마다 시트 네 벤치 (chain_full_eval) + Surrogate. 끝나면 (이미 돌던 짝 hinge λ0.75 채점까지) 전이 판정 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
model() { local n=${1/_ar_vw_state/}; echo "vast_ar_vw_state_${n%_postft}"; }
one() {
  M=$(model $1)
  RUN=$1 MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "EVAL_DONE $M $(ts)"
}
export -f one model ts
printf '%s\n' v4bc4f_ar_vw_state_negimp_vanish v4bc4_ar_vw_state_negimp_vanish_app02 v4bc4ssv2_ar_vw_state_negimp_vanish v4bc4_ar_vw_state_negimp_vanish_w125 \
  v4bc4_ar_vw_state_negimp_vanish_p07 v4bc4ssv2_ar_vw_state_negimp_noswap v4bc4_ar_vw_state_negimp_noswap_m065 v4bc4_ar_vw_state_negpair_lin_l2 \
  v4bc4_ar_vw_state_negpair_hinge_w090 v4bc4_ar_vw_state_negpair_lin_l1 | xargs -P 4 -I{} bash -c 'one {}'
until [[ -f cvpr/results/eval/testbed/intphys1_surrogate_vast_ar_vw_state_v4bc4_negpair_hinge_w075/per_block.json ]]; do sleep 60; done
echo "TRANSFER_TABLE_ALL $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py \
  vast_ar_vw_state_v4bc4_negimp=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_noswap=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_noswap_w075=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_appear=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish_app02=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_w125=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_p07=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_noswap_m065=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negpair_hinge_w075=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negpair_hinge_w090=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negpair_lin_l1=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negpair_lin_l2=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4ssv2=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4ssv2_negimp_noswap=vast_ar_vw_state_v4bc4ssv2 vast_ar_vw_state_v4bc4ssv2_negimp_vanish=vast_ar_vw_state_v4bc4ssv2 \
  vast_ar_vw_state_v4bc4f_negimp_noswap=vast_ar_vw_state_v4bc4f vast_ar_vw_state_v4bc4f_negimp_vanish=vast_ar_vw_state_v4bc4f
echo "EVAL_PARALLEL_DONE $(ts)"
