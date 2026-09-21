#!/bin/bash
# EK100 ViT-H probe 학습 — 공식 입력 기하로 다시 만든 비디오 (2026-09-15) 로. 규약을 인자로 반드시 고른다.
#
#   bash z_research/anticipation/EK100/train_vith.sh paper       # 논문 글: context 가 action 시작 1 s 전에 끝남 (엄격한 anticipation)
#   bash z_research/anticipation/EK100/train_vith.sh released    # 릴리즈 코드: context 가 action 끝 1 s 전에 끝남 (논문 Table 5 수치가 나온 규약)
#   HEADS=sweep bash z_research/anticipation/EK100/train_vith.sh released   # 논문처럼 head 20 개 (lr 5 x wd 4), TAG 에 _sweep — RTX 4090 24 GB 에는 안 들어간다 (head 상태만 30.6 GB)
#   HEADS=grid8 NODE=vll3 MEM_PER_GPU=40G bash z_research/anticipation/EK100/train_vith.sh released
#                        # head 8 개 (lr {3e-4,1e-4} x wd 4 종), vll3 (RealMemory 329 GB). 2026-09-15 제출한 설정
#                        # head 마다 fwd->bwd (eval.py train_one_epoch) 로 activation 1 head 분 → 추정 GPU 당 ~20 GB, 20 epoch ~20 h (추정)
#   DRYRUN=1 bash z_research/anticipation/EK100/train_vith.sh paper         # 병합·검사만 (GPU 0장)
#   LOCAL=1 GPUS=8 bash z_research/anticipation/EK100/train_vith.sh paper   # SLURM 없이 이 셸에서
#
# 규약별로 바뀌는 것
#   paper    : anticipation_point_mode=paper,    time_source=timestamp     (구간 경계 = timestamp x 실제 fps)
#   released : anticipation_point_mode=released, time_source=frame_fixfps  (csv 프레임 번호 그대로, fps 가 다른 비디오만 timestamp)
# 나머지 (config 그대로): 32 f @ 8 fps, 256x256 short_side, train rrc + RandAugment + reprob 0.25, 20 epoch, global batch 128 (8 GPU x 16),
#   head 1 개 lr 3e-4 wd 0.01 (HEADS=sweep 이면 20 개), mask_index 0, val 을 정확히 한 번 (exact_val).
#
# ⚠️ TAG 가 결과 폴더이자 resume 키다. 2026-09-15 이전 run (ek100_vith_lr3e-4 등) 은 같은 경로 이름의 **옛 데이터** (짧은변 256 crop) 로
#    학습했으므로 새 TAG (official256) 를 쓴다. 같은 TAG 에 설정이 다르면 resolve.py 가 막는다.
set -euo pipefail
HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100
MODE=${1:-}
case "$MODE" in
  paper)    EXTRA="data.anticipation_point_mode=paper data.time_source=timestamp" ;;
  released) EXTRA="data.anticipation_point_mode=released data.time_source=frame_fixfps" ;;
  *) echo "usage: bash $0 paper|released   (README §2: context 가 action 시작 1 s 전 / 끝 1 s 전)" >&2; exit 2 ;;
esac
export TAG=${TAG:-ek100_vith_official256_${MODE}${HEADS:+_$HEADS}}
export SET="$EXTRA${SET:+ $SET}"
export GPUS=${GPUS:-8}
if [[ -n "${DRYRUN:-}" || -n "${LOCAL:-}" ]]; then
  exec bash "$HERE/run.sh"
fi
exec bash "$HERE/sbatch.sh"
