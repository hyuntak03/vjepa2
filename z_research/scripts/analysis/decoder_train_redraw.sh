#!/bin/bash
# 새 학습셋으로 정체 자 (위치 + 56 조합 + 없음) 를 학습하고 RollOutV3/figures 그림까지 한 번에 (2026-09-26).
#   1 프레임 페이지 캐시 예열 (HDD RAID 의 작은 PNG 랜덤 읽기는 17 MB/s — 먼저 올린다)
#   2 p·z·h 특징 추출 → /dev/shm (묶음 32 clip, GPU 8 장) — /data2 캐시에 이미 있으면 건너뛴다
#   3 정체 자 학습 (rollout2_identity_readout.py --launch --name, p/z/h 를 GPU 2 장씩 동시에) → 치우침까지
#   4 특징을 /data2 캐시로 옮긴다 (뒤에서, 다시 읽기와 겹쳐)
#   5 new_archive_redraw.py --decoder <이름> : v3 16 창 · v11 late · mid 다시 읽기 + 그림 네 폴더
#   6 v11 빈 장면 판 오탐 검사 (v11_panel_false_alarm.py)
#
# GPU 가 보이는 곳에서 돌린다. GPU 없는 job 의 세션이면 GPU job 에 step 으로 얹는다:
#   srun --jobid=<GPU job> --overlap --cpu-bind=none -n1 -c 96 --mem=0 --gres=gpu:8 \
#        env SET=training_r8 NAME=identity_r8 FRAMES=/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8 \
#        bash z_research/scripts/analysis/decoder_train_redraw.sh
# 로그: z_research/RollOutV3/exp_results/redraw_<NAME>/chain_*.log
set -euo pipefail
: "${SET:?SET=<data_csv/rollout_v2_ 뒤 이름> 을 줄 것}" "${NAME:?NAME=<exp_results 아래 자 이름> 을 줄 것}" "${FRAMES:?FRAMES=<세트 프레임 폴더> 를 줄 것}"
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
SHMF=/dev/shm/rollout2_${SET}_feats
DISKF=/data2/local_datasets/world/world_analysis/cache/rollout2_${SET}_feats
LOG=z_research/RollOutV3/exp_results/redraw_${NAME}; mkdir -p "$LOG"
t0=$(date +%s); log() { echo "[chain $(date +%H:%M:%S) +$(( ($(date +%s)-t0)/60 ))m] $*"; }
log "SET=$SET NAME=$NAME FRAMES=$FRAMES  GPU $(nvidia-smi -L | wc -l) 장 · CPU $(nproc)"

log "1/6 프레임 캐시 예열"
find "$FRAMES/Images" -name "*.png" -print0 | xargs -0 -P 64 -n 200 cat > /dev/null
log "    끝"

if [[ -f "$DISKF/h.npy" ]]; then
  FEAT=$DISKF; log "2/6 특징이 이미 있다 ($DISKF) — 건너뜀"
else
  FEAT=$SHMF; log "2/6 특징 추출 → $SHMF (로그 $LOG/chain_extract.log)"
  PRESENCE_SET=$SET $P -u z_research/scripts/analysis/rollout2_presence_readout.py --set "$SET" --feat-dir "$SHMF" \
      --extract-only --extract-bs 32 --gpus "$(nvidia-smi -L | wc -l)" > "$LOG/chain_extract.log" 2>&1
  log "    $(grep -E '^\[extract\] 끝' "$LOG/chain_extract.log" || true)"
fi

log "3/6 정체 자 학습 → exp_results/$NAME (로그 $LOG/chain_train.log)"
PRESENCE_SET=$SET IDENTITY_FEAT_DIR=$FEAT $P -u z_research/scripts/analysis/rollout2_identity_readout.py \
    --launch --name "$NAME" > "$LOG/chain_train.log" 2>&1
grep -E "정체 57" "$LOG/chain_train.log" | sed 's/^/    /' || true

MOVER=""
if [[ "$FEAT" == "$SHMF" ]]; then
  log "4/6 특징을 $DISKF 로 옮긴다 (뒤에서)"
  ( mkdir -p "$DISKF" && cp "$SHMF"/p.npy "$SHMF"/z.npy "$SHMF"/h.npy "$DISKF"/ && rm -rf "$SHMF" && echo "moved" ) > "$LOG/chain_move.log" 2>&1 &
  MOVER=$!
fi

log "5/6 v3 · v11 다시 읽기 + 그림 (로그 $LOG/chain_redraw.log)"
$P -u z_research/scripts/figures/new_archive_redraw.py --decoder "$NAME" > "$LOG/chain_redraw.log" 2>&1
tail -3 "$LOG/chain_redraw.log" | sed 's/^/    /'

log "6/6 v11 빈 장면 판 오탐 (로그 $LOG/chain_panel.log)"
R3_DECODER=$NAME R3_OUT=$NAME $P z_research/scripts/analysis/v11_panel_false_alarm.py > "$LOG/chain_panel.log" 2>&1
cat "$LOG/chain_panel.log" | grep -v -i warn | sed 's/^/    /'

[[ -n "$MOVER" ]] && wait "$MOVER" && log "    특징 이동: $(cat "$LOG/chain_move.log")"
log "끝"
