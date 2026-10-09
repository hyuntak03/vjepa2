#!/bin/bash
# 2026-09-30 EK100 probe 를 train 1/3 (영상 단위 seed 0, 22.4k clip) · 10 epoch 로: AR ep40 → 릴리즈 → 릴리즈+copy_last, 같은 부분집합에서 셋 다 (비교 가능). vll1.
#   기준선 전체 데이터 20 epoch 은 40 시간 (2 h/epoch) 이라 사용자 결정으로 줄였다. val 은 전체 9,296 clip 그대로.
set -uo pipefail
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT
LOC=/local_datasets; STG=$ROOT/auto_research/_stage; R=$STG/results_ar
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
if [[ $(ls $LOC/EPIC-KITCHENS_resized 2>/dev/null | wc -l) -lt 42 ]]; then log "untar EK100"; rm -rf $LOC/EPIC-KITCHENS_resized; tar xf $STG/epic_kitchens_resized.tar -C $LOC || exit 1; fi
df -h / | tail -1
CK=$ROOT/z_training/ariel_eval/ckpt/ar_ep40.pt
# 2026-09-30 02:25 사용자: VRAM 채워 빠르게 → batch 16 → 32/GPU (global 256, 5.4 → ~11 GB), val 은 5 · 10 epoch 에만 (val 9,296 clip 이 epoch 당 ~40 %), 워커 4/GPU (vll1 CPU 32).
#   step 수: 22,264 / 256 ≈ 87 step/epoch × 10 = 870 (기준선 20 ep × 525 = 10.5k). 셋 다 같은 설정이라 서로는 비교 가능, 기준선 35.34 와는 불가.
# 2026-09-30 02:40 정정: 기준선 곡선이 epoch 16–17 (≈ 9k step) 에야 수렴 → 1/3 데이터 · 10 epoch (870 step) 은 미수렴이라 무의미. batch 32 는 grid8 head 와 함께 OOM.
#   → 기준선과 같은 레시피 (전체 train · batch 16×8 · 20 epoch), val 은 5 epoch 마다. AR 한 run 만 (기준선 35.34 · copy 32.80 은 이미 있다). vll1 NVMe 라 HDD 기준선 (40 h) 보다 빠를 것.
# 2026-09-30 03:50 사용자 재확인: 속도를 위해 train 1/3 (영상 단위 seed 0, 22,264 clip). 20 epoch · batch 128 유지 → 174 step/epoch ≈ 33 분, 20 epoch ≈ 12 h/run.
#   같은 부분집합에서 릴리즈도 이어 돌려 비교선을 만든다 (전체 데이터 기준선 35.34 와는 비교 불가 — step 수 3.5k vs 10.5k).
COMMON="data.base_path=$LOC/EPIC-KITCHENS_resized data.dataset_train=$ROOT/data_csv/ek100/EPIC_100_train_third_seed0.csv optimization.num_epochs=20 data.num_workers=4 evaluation.val_every=5"
log "smoke AR (GPU 1)"
[[ -n "${SKIP_SMOKE:-}" ]] || LOCAL=1 GPUS=1 SMOKE=1 TAG=ek100_ar40_smoke SET="$COMMON model_kwargs.predictor_checkpoint=$CK" bash z_research/anticipation/EK100/train_vith.sh released > $R/ek100_ar40_smoke.log 2>&1 || { tail -30 $R/ek100_ar40_smoke.log; exit 1; }
grep -q "predictor <- .*kind ar" $R/ek100_ar40_smoke.log || { log "SMOKE FAIL: AR predictor 가 로드되지 않았다 (predictor <- ... kind ar 로그 없음)"; grep -n "predictor" $R/ek100_ar40_smoke.log | head; exit 1; }
log "smoke ok (AR predictor 로드 확인)"
run() { tag=$1; extra=$2; log "train $tag"; LOCAL=1 GPUS=8 HEADS=grid8 TAG=$tag SET="$COMMON $extra" bash z_research/anticipation/EK100/train_vith.sh released > $R/${tag}.log 2>&1; log "$tag exit $?"; }
# 2026-10-01 정정: 09-30 실행 (ar40 · third10 · official256 _ar40) 은 predictor_checkpoint 가 model_kwargs 최상위에 있어 init_module 에 안 닿았다 → 릴리즈 predictor 로 돌았다 (release 와 21.74 vs 21.80).
#   eval.py 가 이제 최상위 키를 pretrain_kwargs 로 넘기고, smoke 가 "predictor <- ... kind ar" 로그를 확인한다. release 는 09-30 결과 (21.80) 가 유효하므로 AR 만 다시.
run ek100_rel_third20_grid8_ar40_v2 "model_kwargs.predictor_checkpoint=$CK"
log "CHAIN DONE"
