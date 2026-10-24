#!/bin/bash
# =============================================================================
# 학습 run 하나를 외부 벤치마크 두 개로 잰다 (2026-10-11).
#
#   bash z_training/eval_bench.sh <run> [ckpt=latest.pt]
#   GPUS=8 WINDOWS_I1="32 16" WINDOWS_I2="16 32 48" bash z_training/eval_bench.sh natural_ar_vw_hint
#
# 1. IntPhys1 dev — Garrido A.8 sliding (`intphys1_sliding`, frame_skips [2,5,10], 창별 실행, 보고 = 격자 최고)
#    z_training/docs/RESULTS_2026-10-01.md 의 기준선 채점과 같은 명령 (z_research/scripts/run.sh)
# 2. IntPhys 2 Main — Bordes D.3 격자 (analysis/intphys2/run_grid.sh 와 같은 --set, 보고 = 열별 최고)
# predictor 구조 (kind, n_registers) 는 run config 에서 읽는다. kind=ar 은 analysis/intphys2/model.py 의
#   _ArScorer (문맥·타깃 블록별 target_encoder, **힌트 없음**, KV rollout). 힌트 가중치 (hint_attn.*) 는 로드 때 버린다.
# 결과: <run>/eval/{intphys1_sliding__intphys1_dev_<ckpt>_w<W>, intphys2_main_<ckpt>_w<W>}/  · 요약 <run>/eval/BENCH_<ckpt>.txt
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
PY=$VJEPA2_PY
RUN=${1:?"사용법: eval_bench.sh <run> [ckpt]"}; CKPT=${2:-latest.pt}
[[ -d "$RUN" ]] || RUN="$VJEPA2_ROOT/z_training/runs/$RUN"
RUN=$(cd "$RUN" && pwd); CK="$RUN/$CKPT"; STEM=${CKPT%.pt}
[[ -f "$CK" ]] || { echo "ERROR: $CK 가 없다"; exit 1; }
GPUS=${GPUS:-8}
WINDOWS_I1=${WINDOWS_I1-"32 16"}          # 빈 문자열이면 IntPhys1 건너뜀
WINDOWS_I2=${WINDOWS_I2-"16 32 48"}
VIDEO_BATCH=${VIDEO_BATCH:-1}     # IntPhys 2 영상 묶음. >1 이면 빠르지만 fp16 순서가 바뀌어 정확도가 ±0.5pt 흔들린다 (보고 기준은 1)
WIN_BATCH_SCALE=${WIN_BATCH_SCALE:-1}
I2_SPLIT=${I2_SPLIT:-Main}        # 2026-10-22: MainVuniTest = IntPhys2 violation-uni 장면 split 의 시험 장면만 (그 학습을 한 모델용). 태그에 _<split> 이 붙는다
KIND_SET=$("$PY" - "$RUN/config.yaml" <<'PY'
import yaml, sys
pc = (yaml.safe_load(open(sys.argv[1])).get("model") or {}).get("predictor") or {}
print(" ".join(f"model.predictor.{k}={pc[k]}" for k in ("kind", "n_registers", "ar_feedback", "ctx_target", "ctx_input", "ctx_encoder", "gating_residual", "gating_anchor", "gating_out_ln", "ar_input", "ar_target") if k in pc))   # ctx_target: 2026-10-16 (빠져서 ctx_state 를 창 타깃으로 잘못 채점했다)
PY
)
LOGD="$RUN/eval/logs"; mkdir -p "$LOGD"
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*"; }
say "run $RUN | ckpt $CKPT | predictor: ${KIND_SET:-oneshot}"

# ---- 1. IntPhys1 A.8 --------------------------------------------------------
for w in $WINDOWS_I1; do
  OUT="$RUN/eval/intphys1_sliding__intphys1_dev_${STEM}_w$w"
  [[ -f "$OUT/summary.json" ]] && { say "IntPhys1 w$w 이미 있음"; continue; }
  say "IntPhys1 A.8 w$w 시작"
  SET="model.predictor_checkpoint=$CK $KIND_SET surprise.intphys1.frame_skips=[2,5,10] surprise.intphys1.window_sizes=[$w] model.window_size=$w" \
    OUTDIR="$OUT" TAG="intphys1_$(basename "$RUN")_${STEM}_w$w" GPUS=$GPUS \
    bash z_research/scripts/run.sh intphys1_sliding intphys1_dev vith > "$LOGD/intphys1_${STEM}_w$w.log" 2>&1 \
    && say "  완료" || say "  !! 실패 — $LOGD/intphys1_${STEM}_w$w.log"
done

# ---- 2. IntPhys 2 Main ------------------------------------------------------
I2SET=()
for kv in $KIND_SET; do I2SET+=(--set "$kv"); done
for w in $WINDOWS_I2; do
  tag="intphys2_main_${STEM}_w$w"; [[ $VIDEO_BATCH != 1 ]] && tag="${tag}_vb$VIDEO_BATCH"; [[ $I2_SPLIT != Main ]] && tag="${tag}_$I2_SPLIT"
  [[ -f "$RUN/eval/$tag/summary.json" ]] && { say "IntPhys2 w$w 이미 있음"; continue; }
  C=$("$PY" -c "print([int($w*f) for f in (0.25,0.375,0.5,0.625,0.75,0.875)])")
  case $w in 16) b=32 ;; 32) b=24 ;; *) b=16 ;; esac
  b=$(( b * WIN_BATCH_SCALE ))
  say "IntPhys2 w$w 시작 (C=$C)"
  PYTHONPATH=$VJEPA2_ROOT "$(dirname "$PY")/torchrun" --nproc-per-node=$GPUS --master_port=$((24000 + RANDOM % 10000)) \
    -m analysis.intphys2.eval --config analysis/intphys2/configs/bench_intphys2_main_vith.yaml \
      --set tag=$tag --set folder="$RUN/eval" --set data.split=$I2_SPLIT \
      --set model.predictor_checkpoint="$CK" "${I2SET[@]}" \
      --set surprise.window_size=$w --set "surprise.context_length_sweep=$C" \
      --set surprise.context_length=$((w/2)) --set surprise.max_window_batch=$b --set surprise.video_batch=$VIDEO_BATCH \
    > "$LOGD/intphys2_${STEM}_w$w.log" 2>&1 \
    && say "  완료" || say "  !! 실패 — $LOGD/intphys2_${STEM}_w$w.log"
done

# ---- 요약 --------------------------------------------------------------------
"$PY" z_training/eval_bench_summary.py "$RUN" "$STEM" | tee "$RUN/eval/BENCH_${STEM}.txt"
