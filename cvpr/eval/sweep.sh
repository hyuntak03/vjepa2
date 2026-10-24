#!/usr/bin/env bash
# =============================================================================
# eval/sweep — predictor 체크포인트 여럿을 물리 벤치마크 묶음에 한 번에 건다 (2026-10-20)
#
#   GPUS=4 bash cvpr/eval/sweep.sh "vith_vast_ar_state_ep70 vith_pft_single_w005 vith_pft_single_nosup"
#   STAGES="testbed" GPUS=4 bash cvpr/eval/sweep.sh "<모델들>"     # 일부만
#   COPY=1 ...                                                      # 복사 기준선도 (vith 로 한 번 — 파라미터 0 이라 모델 무관)
#                                                                   #   ⚠️ InfLevel 복사는 엔진 미구현이라 건너뛴다
#   DRYRUN=1 ...                                                    # 경로 · 인덱스 검사만 (GPU 0 장)
#
# STAGES (기본 "testbed intphys1 intphys2", 공백 구분. ek100 은 **기본에서 뺐다**):
#   testbed   vobench_v1_late (432 matched 쌍) · vobench_v1_gravity (cross 채점)
#   intphys1  intphys1_dev · inflevel_{continuity,gravity,solidity}   — Garrido A.8 sliding, 창마다 실행을 나눈다
#   intphys2  intphys2_main_nfs (창 16/32/48)
#   ek100     action anticipation — ⚠️ **채점이 아니라 probe 학습**이다. 모델마다 몇 시간. 명시적으로 넣을 때만 돈다
#
# ⚠️ kind=ar_state 는 채점기가 analysis/predictors/ar_scoring.py 로 분기한다 (타깃 = 블록별 LN(target_encoder),
#    미래는 rollout). **표준 타깃과 다른 공간이라 쌍 정확도만 비교**하고 surprise 크기는 비교하지 않는다 (CLAUDE.md §3-1).
#    그래서 출발점 (vith_vast_ar_state_ep70) 을 **반드시 같은 묶음에 넣는다** — 외부 기준값 (IntPhys1 88.89 등) 과
#    직접 비교하면 프로토콜이 달라진다 (CLAUDE.md §1-3).
# ⚠️ VoE 점수는 예측 없이도 풀릴 수 있다 — COPY=1 로 복사 기준선을 같이 낸다 (CLAUDE.md §12 Q5).
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/.."/..; source cvpr/env.sh

MODELS=${1:?'사용법: sweep.sh "<모델1 모델2 ...>"   (목록: python cvpr/harness/resolve.py --list)'}
STAGES=${STAGES:-"testbed intphys1 intphys2"}
export GPUS=${GPUS:-4}

# ── VRAM 상한 (2026-10-20 실측, vll7 RTX PRO 6000 96 GB) ────────────────────
# 레포 기본값은 24 GB 카드 기준이라 96 GB 에서는 1/4 도 안 쓴다.
#   실측: testbed vobench_v1_late, batch 64, 1 GPU -> 피크 **24.0 GB**. 선형 최악 가정으로도 192 면 72 GB.
# ⚠️ **수치에 영향 없는 축만 올린다.**
#   올림  testbed surprise.batch_size · decode_workers / intphys1 max_batch / intphys2 max_window_batch
#         (run.sh · protocol 주석이 "VRAM 상한, 수치 영향 없음" 이라고 적어 둔 것들. clip 끼리 독립이다)
#   안 올림 intphys1 video_batch (6) · intphys2 video_batch (1)
#         fp16 reduction 순서가 바뀌어 근접 tie 가 뒤집힌다 (CLAUDE.md §7-2 · intphys2/protocol.yaml:14 가 ±0.5 pt 실측).
#         기존 수치가 전부 이 값으로 나왔으므로 바꾸면 과거와 비교가 끊긴다.
# ⚠️ 세 모델에 **같은 배치**를 쓴다 — 팔 사이 비교가 배치 때문에 흔들리지 않게.
TB_BATCH=${TB_BATCH:-192}        # testbed surprise.batch_size (rank 당, 기본 16)
TB_WORKERS=${TB_WORKERS:-16}     # testbed decode_workers (기본 8). rank 당 CPU 18 (72 / 4) 안쪽
IP1_BATCH=${IP1_BATCH:-144}      # intphys1 surprise.intphys1.max_batch (기본 48)
IP2_MULT=${IP2_MULT:-3}          # intphys2 max_window_batch 를 run.sh 가 고른 값의 몇 배로
TS=$(date +%m%d_%H%M)
LOG=${SWEEP_LOG:-$CVPR_LOGS/sweep_$TS.log}
mkdir -p "$(dirname "$LOG")"
declare -a OK=() BAD=()

say() { echo -e "$*" | tee -a "$LOG"; }
step() {  # step <라벨> <명령...>   — 본 채점. COPY 가 환경에 있어도 끈다 (안 그러면 복사만 두 번 돈다)
  local lab=$1; shift
  say "\n>>> [$(date +%H:%M:%S)] $lab"
  if COPY=0 "$@" >>"$LOG" 2>&1; then say "    OK   $lab"; OK+=("$lab")
  else                             say "    실패 $lab  (로그: $LOG)"; BAD+=("$lab"); fi
}
cstep() {  # cstep <라벨> <명령...>  — 복사 기준선 (파라미터 0). 모델 무관이라 첫 모델로 한 번만
  local lab=$1; shift
  say "\n>>> [$(date +%H:%M:%S)] $lab"
  if COPY=1 "$@" >>"$LOG" 2>&1; then say "    OK   $lab"; OK+=("$lab")
  else                             say "    실패 $lab  (로그: $LOG)"; BAD+=("$lab"); fi
}

say "═══ sweep $TS ═══"
say "모델   : $MODELS"
say "단계   : $STAGES   GPUS=$GPUS${DRYRUN:+   [DRYRUN]}${COPY:+   [+복사 기준선]}"
say "배치   : testbed $TB_BATCH (worker $TB_WORKERS) · intphys1 $IP1_BATCH · intphys2 x$IP2_MULT   [96 GB 실측 기준]"
say "로그   : $LOG"

for st in $STAGES; do
  case $st in
    testbed)
      for d in vobench_v1_late vobench_v1_gravity; do
        SET="surprise.batch_size=$TB_BATCH surprise.decode_workers=$TB_WORKERS" \
          step "testbed $d" bash cvpr/eval/testbed/run.sh "$d" "$MODELS"
        # 복사 기준선은 predictor 를 안 쓴다 (파라미터 0) -> 모델 무관. vobench_v1_gravity 판은 10-19 에 이미 나와 있다
        #   (cvpr/results/eval/testbed/copy/vobench_v1_gravity_vith: 216 쌍, 복사 17.59 % · 모델 0.00 %) — 다시 돌리지 않는다.
        #   ⚠️ copy_baseline.sh 는 SET 을 안 받아서 gravity 의 cross 채점을 못 넘긴다. 위 기존 값을 그대로 인용할 것.
        [[ ${COPY:-0} == 1 && $d != vobench_v1_gravity ]] && cstep "testbed $d (복사)" bash cvpr/eval/testbed/run.sh "$d" vith
      done ;;
    intphys1)
      for b in intphys1_dev inflevel_continuity inflevel_gravity inflevel_solidity; do
        SET="surprise.intphys1.max_batch=$IP1_BATCH" \
          step "intphys1 $b" bash cvpr/eval/intphys1/run.sh "$b" "$MODELS"
        # ⚠️ IntPhys 1 계열 복사 기준선은 **intphys1_dev x vith 만** 구현돼 있다 (te_intphys1_score.py:169 D=1280 ·
        #    :248-249 프로토콜 · 데이터셋 · 모델 리터럴). InfLevel 복사 기준선은 아직 없다 — 정본 §10 항목 2 가 그 일이다.
        #    복사는 predictor 를 안 쓰므로 (파라미터 0) 모델은 vith 로 족하다: 어느 팔을 재든 같은 기준선이다.
        [[ ${COPY:-0} == 1 && $b == intphys1_dev ]] && cstep "intphys1 $b (복사)" bash cvpr/eval/intphys1/run.sh "$b" vith
      done ;;
    intphys2)
      # intphys2/run.sh 가 모델 x 창 별로 max_window_batch 를 고른다 -> 그 위에 배수를 곱해 덮는다
      IP2_BATCH_MULT=$IP2_MULT step "intphys2 intphys2_main_nfs" bash cvpr/eval/intphys2/run.sh intphys2_main_nfs "$MODELS" ;;
    ek100)
      for m in $MODELS; do step "ek100 $m" bash cvpr/eval/ek100/run.sh ek100 "$m"; done ;;
    *) say "!! 모르는 단계: $st" ;;
  esac
done

say "\n═══ 끝 $(date +%H:%M:%S) ═══   성공 ${#OK[@]} · 실패 ${#BAD[@]}"
for x in "${BAD[@]:-}"; do [[ -n $x ]] && say "   실패: $x"; done
say "결과   : $CVPR_RESULTS/eval/...   요약: python cvpr/harness/resolve.py --list"
