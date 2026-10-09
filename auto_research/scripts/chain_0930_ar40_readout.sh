#!/bin/bash
# 2026-09-30 AR ep40 의 p 로 정체 자 (identity_r8 구조: 위치 + 56 조합 + 없음) 를 **새로 학습**하고 RollOut_v3 16 창 · v11 (visible + late) 에 건다.
#   사용자: "AR ep40 을 v11 · RollOutV3 로, decoder 새로 학습해서 — 중력을 이해했는지". decoder_train_redraw.sh 의 AR · 다른 노드 판.
#   vll1 (8 GPU, 데이터 없음): NFS tar → /local_datasets (/ 96 GB) 로 푼다. 특징은 /dev/shm (p·h 각 45 GB) 에만 두고 학습 뒤 지운다.
#   AR 의 'p' = rollout(블록별 LN(target_encoder) 문맥 8 블록, 8), 'h' = 블록별 LN(target_encoder) — 세 스크립트 모두 같은 정의 (is_ar 분기). z 는 없다.
set -uo pipefail
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; AN=z_research/scripts/analysis
NAME=${NAME:-identity_ar40}; CK=${CK:-$ROOT/z_training/ariel_eval/ckpt/ar_ep40.pt}
LOC=/local_datasets/world/world_analysis; STG=$ROOT/auto_research/_stage
LOG=z_research/RollOutV3/exp_results/redraw_${NAME}; mkdir -p $LOG
t0=$(date +%s); log() { echo "[chain $(date +%H:%M:%S) +$(( ($(date +%s)-t0)/60 ))m] $*"; }
run() { log ">>> $*"; "$@"; r=$?; log "<<< exit $r"; return $r; }
log "NAME=$NAME CK=$CK host $(hostname) GPU $(nvidia-smi -L | wc -l)"
mkdir -p $LOC
[[ -f $LOC/RollOut_v2_training_v8/Images/panel/roll/rollout2_training_v8panel_panel_00216_pos_roll/000000.png ]] || { rm -rf $LOC/RollOut_v2_training_v8; run tar xf $STG/rollout_v2_training_v8.tar -C $LOC || exit 1; }   # 09-30: 첫 tar 가 심볼릭 링크만 담아 실패 → -h 판으로 다시
[[ -d $LOC/RollOut_v3/Images ]]             || run tar xf $STG/rollout_v3.tar -C $LOC || exit 1
[[ -f $LOC/IntPhysGen_v11/metadata.csv ]]   || run tar xf $STG/v11_readout_subset.tar -C $LOC || exit 1
df -h / /dev/shm | tail -2
export PRED_CKPT=$CK PRESENCE_SET=training_r8 PRESENCE_FRAMES=$LOC/RollOut_v2_training_v8 PRESENCE_REPS=p,h
SHMF=/dev/shm/rollout2_training_r8_ar40_feats
log "1/5 특징 추출 (AR p · 블록별 h) → $SHMF"
run $P -u $AN/rollout2_presence_readout.py --set training_r8 --feat-dir $SHMF --extract-only --extract-bs 32 --gpus 8 > $LOG/chain_extract.log 2>&1 || { tail -20 $LOG/chain_extract.log; exit 1; }
grep -E "extract" $LOG/chain_extract.log | tail -2
log "2/5 정체 자 학습 → z_research/RollOutV3/exp_results/$NAME (p=GPU0-3, h=GPU4-7)"
IDENTITY_FEAT_DIR=$SHMF run $P -u $AN/rollout2_identity_readout.py --launch --name $NAME --reps p h --plan "p=0,1,2,3 h=4,5,6,7" > $LOG/chain_train.log 2>&1 || { tail -20 $LOG/chain_train.log; exit 1; }
grep -E "정체 57|test" $LOG/chain_train.log | tail -6
log "3/5 치우침 (decoder_bias)"
IDENTITY_FEAT_DIR=$SHMF R3_DECODER=$NAME R3_OUT=$NAME run $P -u $AN/decoder_bias.py > $LOG/chain_bias.log 2>&1 || tail -20 $LOG/chain_bias.log
rm -rf $SHMF; log "    특징 삭제 (shm)"
log "4/5 RollOut_v3 16 창 (p, h; p 에 h 자 이식)"
R3_DECODER=$NAME R3_OUT=$NAME R3_SRC=$LOC/RollOut_v3 R3_TMP=/dev/shm/r3w_$NAME run $P -u $AN/rollout3_window_readout.py --reps p h --cross h --gpus 8 > $LOG/chain_v3.log 2>&1 || tail -20 $LOG/chain_v3.log
log "5/5 v11 visible + late (p, h)"
R3_DECODER=$NAME R3_OUT=$NAME V11_FRAMES=$LOC/IntPhysGen_v11 V11_TMP=/dev/shm/v11_$NAME run $P -u $AN/v11_readout_online.py --gpus 8 --reps p h > $LOG/chain_v11.log 2>&1 || tail -20 $LOG/chain_v11.log
log "CHAIN DONE"
