#!/bin/bash
# 2026-11-09 14:0x KST chain_wave4_20261109 를 대신한다 (사용자 "release + v4bc4 + negative loss p = 0.2 도 학습") — 메모리를 보며 올린다.
#   (1) 릴리즈 · p 0.2 바로 (릴리즈 판은 GPU 당 약 11 GB)  (2) 공식 IntPhys 채점 (사라짐만 · p 0.2, 앞 사슬의 하위 프로세스가 계속 돈다) 이 끝나면
#   (3) 4 묶음 남은 2 판을 메모리가 비는 대로 (모든 GPU < 76 GB, 1 분 × 3)  (4) 학습이 다 끝나면 4 개씩 병렬 채점 → 전이 판정 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
model() { case $1 in v4bc4_release_*) local n=${1#v4bc4_release_}; echo "vith_release_v4bc4_$n" ;; *) local n=${1/_ar_vw_state/}; echo "vast_ar_vw_state_${n%_postft}" ;; esac; }
REL=v4bc4_release_negimp_vanish_app02
FIRST="v4bc4_ar_vw_state_negimp_vanish_app03 v4bc4_ar_vw_state_negimp_vanish_app02_m065 v4bc4_ar_vw_state_negimp_vanish_m065"
LATER="v4bc4f_ar_vw_state_negimp_vanish_app02 v4bc4_ar_vw_state_negimp_vanish_app01"
GPUS=8 bash z_training/tmux_train.sh $REL && echo "TRAIN_LAUNCHED $REL $(ts)"
while pgrep -f "eval_bench.sh v4bc4_ar_vw_state_negimp_vanish" > /dev/null; do sleep 60; done
echo "공식 채점 끝 $(ts)"; grep -E "OFFICIAL_DONE|격자 최고|열별 최고" z_training/logs/chain_wave4_20261109.log
for r in $LATER; do
  ok=0; while (( ok < 3 )); do
    full=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 76000' | wc -l)
    if (( full == 0 )); then ok=$((ok + 1)); else ok=0; fi; sleep 60; done
  GPUS=8 bash z_training/tmux_train.sh $r && echo "TRAIN_LAUNCHED $r $(ts)"; sleep 300
done
for r in $FIRST $LATER $REL; do
  until [[ -f z_training/runs/$r/e20.pt ]] && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$r/config\.yaml"; do
    if grep -qE "OutOfMemoryError|rank 실패" z_training/runs/$r/train.log 2>/dev/null; then echo "⚠️ $r 학습 실패 $(ts)"; break; fi
    sleep 120; done
done
echo "학습 전부 끝 $(ts)"
one() { M=$(model $1); RUN=$1 MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
        GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1; echo "EVAL_DONE $M $(ts)"; }
export -f one model ts
printf '%s\n' $FIRST $LATER $REL | xargs -P 4 -I{} bash -c 'one {}'
echo "TRANSFER_TABLE_W4 $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py \
  vast_ar_vw_state_v4bc4_negimp_vanish=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish_app01=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app03=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish_m065=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02_m065=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4f_negimp_vanish_app02=vast_ar_vw_state_v4bc4f \
  vith_release_v4bc4_negimp=vith_release_v4bc4 vith_release_v4bc4_negimp_vanish_app02=vith_release_v4bc4
echo "CHAIN_WAVE4B_DONE $(ts)"
