#!/bin/bash
# 2026-11-09 13:3x KST 전이 탐색 4 묶음 + 통과판 공식 채점.
#   (1) 공식 IntPhys1 · IntPhys2: v4bc4 사라짐만 · 사라짐 + 나타남 p0.2 (둘 병렬, 학습 3 판과 같이 — 메모리 여유 확인)
#   (2) 학습 5 판: 처음 3 판 바로, 나머지 2 판은 공식 채점이 끝난 뒤 (OOM 으로 채점이 멈췄던 11-09 05 시 사고를 피한다)
#   (3) 학습이 끝나면 시트 네 벤치 · Surrogate 를 4 개씩 병렬 → 전이 판정 표 (기준 = 같은 데이터 L1 만)
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
model() { local n=${1/_ar_vw_state/}; echo "vast_ar_vw_state_${n%_postft}"; }
off() { GPUS=8 bash z_training/eval_bench.sh $1 > z_training/logs/eval_bench_$1.log 2>&1; echo "OFFICIAL_DONE $1 $(ts)"; grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$1.log; }
off v4bc4_ar_vw_state_negimp_vanish & P1=$!
sleep 60
off v4bc4_ar_vw_state_negimp_vanish_app02 & P2=$!
sleep 120
FIRST="v4bc4_ar_vw_state_negimp_vanish_app03 v4bc4_ar_vw_state_negimp_vanish_app02_m065 v4bc4_ar_vw_state_negimp_vanish_m065"
LATER="v4bc4f_ar_vw_state_negimp_vanish_app02 v4bc4_ar_vw_state_negimp_vanish_app01"
for r in $FIRST; do GPUS=8 bash z_training/tmux_train.sh $r && echo "TRAIN_LAUNCHED $r $(ts)"; sleep 30; done
wait $P1 $P2
for r in $LATER; do GPUS=8 bash z_training/tmux_train.sh $r && echo "TRAIN_LAUNCHED $r $(ts)"; sleep 30; done
# 학습이 다 끝날 때까지
for r in $FIRST $LATER; do
  until [[ -f z_training/runs/$r/e20.pt ]] && ! ps -eo args | grep -qE "launch\.py --fname \S*/runs/$r/config\.yaml"; do
    if grep -qE "OutOfMemoryError|rank 실패" z_training/runs/$r/train.log 2>/dev/null; then echo "⚠️ $r 학습 실패 $(ts)"; break; fi
    sleep 120; done
done
one() { M=$(model $1); RUN=$1 MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
        GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1; echo "EVAL_DONE $M $(ts)"; }
export -f one model ts
printf '%s\n' $FIRST $LATER | xargs -P 4 -I{} bash -c 'one {}'
echo "TRANSFER_TABLE_W4 $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py \
  vast_ar_vw_state_v4bc4_negimp_vanish=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish_app01=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app03=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4_negimp_vanish_m065=vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02_m065=vast_ar_vw_state_v4bc4 \
  vast_ar_vw_state_v4bc4f_negimp_vanish_app02=vast_ar_vw_state_v4bc4f
echo "CHAIN_WAVE4_DONE $(ts)"
