#!/usr/bin/env bash
# =============================================================================
# EK100 action anticipation (vll7) — frozen encoder + predictor + attentive probe **학습 후** val (2026-10-17)
#
#   GPUS=6 bash z_vll7_training/eval_ek100.sh "vith_release_vobft vith_release_vobftg"
#   DRYRUN=1 bash z_vll7_training/eval_ek100.sh vith_release_vobft
#
# ⚠️ **이건 채점이 아니라 probe 학습 (20 epoch) 이다.** batch 를 바꾸면 최적화가 바뀌어 기준선과 비교가 깨진다.
#
# 기준선 (z_research/previous/anticipation/EK100/README.md): `ek100_vith_official256_released_grid8`
#   = verb 56.36 / noun 56.16 / action 35.34 (8 GPU x batch 16 = global 128, HEADS=grid8, **released** 규약)
# 그래서 여기서도 맞춘다:
#   * `anticipation_point_mode=released`  — protocol 기본값은 `paper` 다. 규약을 바꾸면 이식이 안 된다
#     (같은 체크포인트를 paper 로 재면 action 35.34 -> 14.14, README §A-1)
#   * global batch 126 (6 x 21) ~ 128.  ⚠️ 8 GPU 기준선과 2 clip 차이가 난다 (6 GPU 라 정확히 못 맞춘다)
#   * HEADS=grid8 — head 중 max 를 쓰므로 head 수를 늘리면 (sweep 20) 지표가 올라가 비교가 깨진다
# vll7 전용: CVPR_DATA2 shim (영상 = /local_datasets/EPIC-KITCHENS_resized), CVPR_PY = vjepa2_bw (sm_120)
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh" >/dev/null
source "$VJEPA2_ROOT/cvpr/env.sh"
export CVPR_DATA2=/local_datasets/world/_cvpr_data2_vll7
export CVPR_PY="$VJEPA2_PY"
MODELS=${1:?"사용법: GPUS=6 bash z_vll7_training/eval_ek100.sh \"<모델> [모델...]\""}
export GPUS=${GPUS:-6}
export HEADS=${HEADS:-cap16b96}  # ⚠️ grid8 이 아니다. BS 와 짝이다: BS=8 (global 48) -> cap16g2 / BS=16 (global 96) -> cap16b96
VAL_EVERY=${VAL_EVERY:-5}       # val 한 번이 train 한 epoch 과 비슷하다. 5 로 두면 epoch 당 비용이 거의 반
CKPT_EVERY=${CKPT_EVERY:-5}     # head 8 개면 체크포인트가 7.5 GB — NFS 57 MB/s 라 매 epoch 쓰면 epoch 당 2.2 분 (전체의 18 %)
BS=${BS:-16}            # rank 당. 6 x 16 = 96 (2026-10-18 사용자 지시). 검증된 기준은 BS=8 (global 48) + HEADS=cap16g2 다.
                        #    ⚠️ global 을 키우면 ipe 가 줄어 step 이 준다 (48 -> ipe 238 / 96 -> 119 / 126 -> 91).
                        #       lr 로 메워지는지는 ep5 를 기준 실행의 10.25 와 대 보고 판정한다.
NW=${NW:-4}             # ⚠️ **올리지 말 것.** 2026-10-18: 6 -> 4 (predictor_group 5 팔에서 호스트 RAM 276/300 GB 까지 올라갔다). 16 으로 했더니 6 rank x 16 = 96 디코드 워커가 job (CPU 108 · mem 300G) 을
                        #    질식시켜 `Training...` 직후 멈췄다 (2026-10-17, rank 2 개가 GPU 메모리도 못 잡고 공회전).
                        #    6 은 과거 8 GPU run 이 쓰던 값이고 vll7 에서 실측 4.2 s/iter · data time 0.7 ms 로 병목이 아니다

# ── 데이터 CSV (2026-10-17 사용자 결정: cap16metric 부분집합) ────────────────
#   train 11,470 행 / 470 편 · val 9,296 행 / 138 편. 레지스트리 기본값은 full EPIC_100_{train,validation}.csv 다.
#   full 로 돌리려면 TRAIN_CSV=registry VAL_CSV=registry.
#
#   ⚠️⚠️ **2026-10-18 — 이 CSV 에 grid8 을 쓰면 안 된다.** train 이 full 의 1/5.9 (11,470 vs 67,217) 라
#   20 epoch = 1,820 step 뿐이고, grid8 의 lr 격자 (3e-4 · 1e-4, warmup 0) 는 그 안에서 거의 못 움직인다.
#       cap16metric x grid8 : ep3 0.47 · ep4 0.56 · ep5 0.90   (ek100_vith_release_vobftg, 10-17 22:31 ~ 10-18 00:11 폐기)
#       cap16metric x cap16g2: ep5 10.25 · ep10 23.62 · ep15 30.52 (ek100_rel_cap16metric_32f_e20_g2, 같은 train CSV)
#       full x grid8         : ep2 4.08 · ep4 9.20            (ek100_vith_release_vobftg_b126_grid8_e3)
#
#   ⚠️⚠️ 2026-10-18 정정 — 범인은 lr **격자**가 아니라 **global batch** 다. 기준 실행에서 실제로 이긴 head 는
#   grid8 에도 들어 있는 lr 3e-4 / warmup 0 이었다 (1e-3 + warmup 2 는 ep5 에서 10.25 vs 9.99 로 졌다). 다른 건 batch 하나다:
#       기준 : global  48 (4 x 12) -> ipe 238, 20 epoch = 4,760 step
#       우리 : global 126 (6 x 21) -> ipe  91, 20 epoch = 1,820 step   <- step 2.6x 부족 + sample 당 유효 lr 2.6x 낮음
#   실측 (둘 다 batch 126): grid8 ep5 0.90 / cap16g2 ep5 2.32 (여기선 lr 1e-3 head 가 이긴다 = 선형 스케일링과 일치)
#   vs 기준 ep5 10.25. README §7 의 "warmup 0 에서 lr 상한 3e-4" 도 batch 128 기준이라 batch 를 키워 lr 로 메우는 길은 막혀 있다.
#   -> **batch 를 기준 실행에 맞춘다 (48).** 그래야 ep5 10.25 · ep10 23.62 · ep15 30.52 를 재현한다.
TRAIN_CSV=${TRAIN_CSV:-$DATA_CSV/ek100/EPIC_100_train_cap16metric_seed0.csv}
VAL_CSV=${VAL_CSV:-$DATA_CSV/ek100/EPIC_100_validation_cap16metric_seed0.csv}

# ── ⚠️ 호스트 RAM 가드 (2026-10-17 사고 뒤 추가) ────────────────────────────
#   2026-10-17 18:08 `BS=85 NW=4` (global 510) 로 돌렸다가 **job 의 mem 캡 300 GB 를 넘겨 rank 1 이 SIGKILL** 됐고,
#   나머지 5 rank 가 ALLREDUCE 에서 2 시간 기다리다 watchdog abort 했다 (cgroup: max_usage == limit 300.0 GB, failcnt 248 만).
#   터진 건 **VRAM 이 아니라 호스트 RAM** 이다 — GPU 는 43 / 96 GB 였다. in-flight 클립 수가 비용을 정한다:
#       clips = BS x NW x prefetch(2) x GPUS
#   ⚠️ clip 당 비용은 **이론치가 아니라 사고의 실측으로** 잡았다 — 이론 (32 x 256^2 x 3 float32 = 25 MB) 은 크게 과소추정이다.
#   ⚠️⚠️ 2026-10-18 재보정 75 -> 150 MB/clip. 그날 두 번째 OOM 에서 BS 16 x NW 6 x 2 x 6 = 1,152 clip 이 캡 300 GB 를
#      거의 다 먹었다 (실측 276 GB, 기준선 34 GB 제외하면 clip 당 약 150 MB). 옛 75 는 이 경우를 84 GB 로 봐서 못 걸렀다.
#      같이 고친 것: `init_multi_module` 이 공유 체크포인트 (9.66 GB x rank 6 = 58 GB) 를 모델 조립 뒤 놓는다.
#      캡의 70 % 를 넘으면 멈춘다. 무시: SKIP_MEM_GUARD=1
CGROUP_LIM=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID:-0}/memory.limit_in_bytes 2>/dev/null || echo 0)
if [[ "${SKIP_MEM_GUARD:-0}" != 1 && "$CGROUP_LIM" -gt 0 ]]; then
  CLIPS=$(( BS * NW * 2 * GPUS ))
  EST_GB=$(( CLIPS * 150 / 1024 ))         # 150 MB/clip = 2026-10-18 OOM 실측 (옛 75 는 2 배 과소추정이었다)
  LIM_GB=$(( CGROUP_LIM / 1024 / 1024 / 1024 ))
  echo "  메모리 : in-flight clip ${CLIPS} -> 추정 ${EST_GB} GB / job 캡 ${LIM_GB} GB"
  if (( EST_GB * 100 > LIM_GB * 70 )); then
    echo "ERROR: 추정 호스트 RAM ${EST_GB} GB 가 job 캡 ${LIM_GB} GB 의 70 % 를 넘는다 (2026-10-17 과 같은 OOM 위험)."
    echo "       BS 또는 NW 를 줄이거나, --mem 을 키운 새 job 에서 돌릴 것. 무시하려면 SKIP_MEM_GUARD=1"
    exit 1
  fi
fi

[[ -d "$CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized" ]] || {
  echo "ERROR: EK100 영상이 없다 -> $CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized"
  echo "  tar -xf /data/dataset/EPIC-KITCHENS_resized.tar -C /local_datasets  후 shim 심볼릭"; exit 1; }

S="experiment.data.anticipation_point_mode=released experiment.optimization.batch_size=$BS experiment.data.num_workers=$NW experiment.evaluation.val_every=$VAL_EVERY experiment.evaluation.ckpt_every=$CKPT_EVERY"
[[ "$TRAIN_CSV" != registry ]] && S="$S experiment.data.dataset_train=$TRAIN_CSV"
[[ "$VAL_CSV"   != registry ]] && S="$S experiment.data.dataset_val=$VAL_CSV"
echo "=================================================="
echo "  모델   : $MODELS"
echo "  규약   : released (기준선과 같음)   HEADS=$HEADS   val 은 $VAL_EVERY epoch 마다"
echo "  batch  : rank $BS x $GPUS = $((BS*GPUS))  (cap16metric 기준 실행 global 48)   workers $NW"
echo "  CSV    : train $(basename "$TRAIN_CSV") / val $(basename "$VAL_CSV")"
echo "  영상   : $CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized ($(find -L "$CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized" -name '*.MP4' | wc -l) 편)"
echo "=================================================="
SET="${SET:-} $S" exec bash "$VJEPA2_ROOT/cvpr/eval/ek100/run.sh" ek100 "$MODELS"
