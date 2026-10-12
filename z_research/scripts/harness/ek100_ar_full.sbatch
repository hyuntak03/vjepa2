#!/bin/bash
#SBATCH -J ek100_ek100_vith_official256_released_grid8_ar40_full   # monitor.sh 가 ek100_<TAG> 로 찾는다
#SBATCH -w vll1
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH -c 28
#SBATCH --mem=215G
#SBATCH -t 4-00:00:00
#SBATCH -o /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/previous/anticipation/EK100/slurm_logs/ek100_ar40_full_%j.out
# =============================================================================
# EK100 action anticipation probe — **전체 train** · Ariel AR predictor ep40 (2026-10-10)
#
#   sbatch z_research/scripts/harness/ek100_ar_full.sbatch
#
# 릴리즈 전체 데이터 기준선 `ek100_vith_official256_released_grid8` (action 35.34) 와 **같은 레시피** — released 규약 · grid8 head ·
# 16 × 8 GPU = global 128 · 20 epoch. 바뀌는 것은 predictor (`model_kwargs.predictor_checkpoint` = AR ep40) 와
# **val 5 epoch 마다** (`evaluation.val_every=5` → 5 · 10 · 15 · 20; 사용자 지시 2026-10-10, 속도. 기준선은 매 epoch — 최종 20 은 같이 잰다).
# 비디오: vll2 노드 로컬 `/local_datasets/EPIC-KITCHENS_resized` — 2026-10-10 기준선 데이터 (vll5 `/data2/local_datasets/…`) 와
#   파일 이름 · 크기 1,267 개 전부 일치 확인. 주석 (train 495 + val 138 = 633 비디오) 중 하나라도 없으면 `/data/dataset/EPIC-KITCHENS_resized.tar` (사용자 지정 원본, 66 GB) 를
#   `/data2/local_datasets` 에 풀어 쓴다 (vll2 루트 디스크는 71 G 남음).
#
# ⚠️ 2026-10-10 job 220604 (vll2, --mem=230G, worker 6) 가 OOM 으로 죽었다 — rank 2 가 cgroup OOM kill (−9), 나머지 7 개는 allreduce 에서
#    NCCL timeout 2 h 를 기다린 뒤 abort. 기준선 213732 는 vll3 · --mem=320G · MaxRSS 196G 였다 (rank 8 × worker 6 = 디코드 48 프로세스).
#    2026-10-11 재제출: vll1 (CPU 32 · RAM 239 GiB 가 노드 전체) 에 맞춰 -c 28 · --mem=215G · data.num_workers=2 (NUM_WORKERS 로 덮음).
#    worker 수는 데이터 로딩 속도만 바꾸고 학습 결과 (샘플 · 순서 규칙) 는 같다. vll1 의 `/local_datasets/EPIC-KITCHENS_resized` 는 mp4 700 개 —
#    기준선 633 개가 이름 · 크기 전부 일치하고, 나머지 67 개는 train/val 주석에 없는 비디오라 로더가 쓰지 않는다 (epickitchens.py:289 가 주석 기준으로 고른다).
#    rank 하나가 죽으면 2 h 를 기다리지 않도록 rank 프로세스 수를 감시해 줄면 job 을 끝낸다 (아래 감시 루프).
#
# ⚠️ 2026-09-30 사고: predictor_checkpoint 키가 init_module 에 안 닿아 릴리즈 predictor 로 돌았다 (10-01 수정).
#    그래서 본 실행 로그에 `predictor <- … kind ar` 가 40 분 안에 안 나오면 job 을 스스로 죽인다 (smoke 따로 안 돈다 — 09-30 smoke 는 2 h 15 m).
# =============================================================================
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT
DATA=/local_datasets/EPIC-KITCHENS_resized
TAR=/data/dataset/EPIC-KITCHENS_resized.tar
CK=$ROOT/z_training/ariel/ar_future_1e_5/latest.pt          # AR ep40 (step 35,840) — ariel_eval/ckpt/ar_ep40.pt 와 같은 파일 (cmp)
TAG=${TAG:-ek100_vith_official256_released_grid8_ar40_full}
LOG=$ROOT/z_research/previous/anticipation/EK100/slurm_logs/${TAG}_${SLURM_JOB_ID:-local}.train.log
log() { echo "[ek100_ar $(date '+%m-%d %H:%M:%S')] $*"; }

log "$(hostname) job ${SLURM_JOB_ID:-} · $(nvidia-smi -L | wc -l) GPU · ckpt $CK"
ANN=/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations
missing() {  # train · val 주석에 있는 비디오 중 $1 에 없는 개수 (로더 규칙 그대로: <pid>/videos/<vid>.MP4)
  python -c "
import pandas as pd, os, sys
v = set(pd.read_csv('$ANN/EPIC_100_train.csv').video_id) | set(pd.read_csv('$ANN/EPIC_100_validation.csv').video_id)
print(len(v), sum(not os.path.exists(os.path.join(sys.argv[1], x.split('_')[0], 'videos', x + '.MP4')) for x in v))" "$1"
}
read nv nm < <(missing $DATA); log "주석 비디오 $nv 개 중 없는 것 $nm @ $DATA (mp4 $(find $DATA -iname '*.mp4' 2>/dev/null | wc -l) 개)"
if [[ $nv -ne 633 || $nm -ne 0 ]]; then
  DATA=/data2/local_datasets/EPIC-KITCHENS_resized
  read nv nm < <(missing $DATA)
  if [[ $nm -ne 0 ]]; then
    log "untar $TAR -> /data2/local_datasets"; mkdir -p /data2/local_datasets; rm -rf $DATA
    tar xf $TAR -C /data2/local_datasets || { log "untar 실패"; exit 1; }
    read nv nm < <(missing $DATA)
  fi
  log "주석 비디오 $nv 개 중 없는 것 $nm @ $DATA"; [[ $nv -eq 633 && $nm -eq 0 ]] || { log "데이터 불완전 — 중단"; exit 1; }
fi

LOCAL=1 GPUS=8 HEADS=grid8 TAG=$TAG \
  SET="data.base_path=$DATA model_kwargs.predictor_checkpoint=$CK evaluation.val_every=5 data.num_workers=${NUM_WORKERS:-2}" \
  bash z_research/previous/anticipation/EK100/train_vith.sh released > "$LOG" 2>&1 &
PID=$!
log "train pid $PID · log $LOG"

# AR predictor 가 실제로 로드됐는지 확인 (최대 40 분)
for i in $(seq 1 240); do
  if grep -q "predictor <- .*kind ar" "$LOG" 2>/dev/null; then log "OK: $(grep -m1 'predictor <- ' "$LOG" | sed 's/.*predictor <- /predictor <- /')"; break; fi
  if ! kill -0 $PID 2>/dev/null; then log "학습이 먼저 끝났다 (실패?) — 로그 끝:"; tail -30 "$LOG"; exit 1; fi
  sleep 10
done
if ! grep -q "predictor <- .*kind ar" "$LOG"; then
  log "FAIL: 40 분 안에 'predictor <- … kind ar' 가 없다 — 릴리즈 predictor 로 도는 것일 수 있어 중단"; pkill -P $PID; kill $PID; sleep 5; exit 1
fi

# rank 감시: evals.main 의 직계 자식 (rank 8 개 + 보조 프로세스) 수가 줄면 rank 가 죽은 것 — NCCL timeout 2 h 를 기다리지 않고 끝낸다
MAIN=$(pgrep -u $USER -f "python -m evals\.main" | head -1); n0=$(ps --ppid ${MAIN:-0} -o pid= 2>/dev/null | wc -l)
log "감시: evals.main pid ${MAIN:-?} · 자식 $n0 개"
while kill -0 $PID 2>/dev/null; do
  sleep 60
  n=$(ps --ppid ${MAIN:-0} -o pid= 2>/dev/null | wc -l)
  if [[ -n "$MAIN" && $n -lt $n0 ]] && kill -0 $MAIN 2>/dev/null; then
    log "FAIL: rank 프로세스 $n0 → $n 개 (OOM 등으로 죽음) — job 종료"; free -g | head -2
    pkill -9 -P $MAIN; kill -9 $MAIN; pkill -P $PID; kill $PID; sleep 5; exit 1
  fi
done
wait $PID; rc=$?
log "train exit $rc"
exit $rc
