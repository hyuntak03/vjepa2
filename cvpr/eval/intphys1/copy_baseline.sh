#!/usr/bin/env bash
# =============================================================================
# eval/intphys1/copy_baseline.sh — IntPhys 1 (Garrido A.8 sliding) 복사 기준선. 창마다 문맥 마지막 튜블릿 LN(z)[Tc−1] 을
# 미래 전부에 복사해 같은 Filtered · AvgSurprise · 쌍 비교 · property macro 로 채점한다 (정의 · 단서는 README.md).
#
#   GPUS=8 bash cvpr/eval/intphys1/copy_baseline.sh intphys1_dev vith          # 격자 세 칸 (skip2_w16 · skip2_w32 · skip5_w16) 을 한 번에
#   COPY=1 GPUS=8 bash cvpr/eval/intphys1/run.sh intphys1_dev vith              # run.sh 의 COPY=1 이 여기로 온다 (dinof_* 는 로더 플래그)
#   DRYRUN=1 bash cvpr/eval/intphys1/copy_baseline.sh                           # 두 resolver 대조 · 창 격자 · 명령만 (GPU 0 장)
#   SUBSET=valid6 GPUS=1 bash cvpr/eval/intphys1/copy_baseline.sh               # 스모크: O1/O2/O3 앞 2 block (24 영상 · 12 쌍), 폴더에 _valid6
#   LIMIT=4 GPUS=1 bash cvpr/eval/intphys1/copy_baseline.sh                     # 스모크: 영상 앞 4 개, 폴더에 _smoke4
#   VALIDATE=1 ...                                                              # 채점 때 te --validate 도 (공식 per_window · H6e 대조, CPU)
#
# 엔진 = 기존 z_research/scripts/analysis/te_intphys1_score.py (predictor `release` + copy, 창 × 튜블릿 L1, 표준 · 인과 표적) 을
# copy_baseline.py 가 import 해 추출 (GPU) → --score (CPU) 로 돌린다. ⚠️ intphys1_dev × vith 만 된다 (README '엔진 수정 제안').
# 360 영상 · 8 GPU ≈ 15 분 + 로드 (te docstring 09-25 값, 여기서는 미측정).
# 결과: $CVPR_RESULTS/eval/intphys1/copy/<ds>_<model>[_smokeN|_valid6]/{copy_summary.json, <tag>_scores.json, doc/, manifest.json, shards/, arrays_v*.npz, _meta.json}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
BENCHES=${1:-${BENCHES:-intphys1_dev}}
MODELS=${2:-${MODELS:-vith}}
GPUS=${GPUS:-1}
GPU_IDS=${GPU_IDS:-$(seq -s ' ' 0 $((GPUS - 1)))}
GPUS=$(wc -w <<<"$GPU_IDS")
export CUDA_VISIBLE_DEVICES=$(tr ' ' ',' <<<"$GPU_IDS")      # te_*_score.py 는 mp.spawn 으로 보이는 GPU 앞 N 장을 cuda:0..N-1 로 쓴다 (evals.main 과 다르다)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${THREADS:-6}}
[[ -n ${WINDOWS:-}${WINDOW:-} ]] && echo "!! WINDOWS/WINDOW 는 복사 기준선에 적용되지 않는다 — te_intphys1_score.py 가 격자 세 칸을 한 번에 뽑는다"
[[ -n ${SET:-} ]] && echo "!! SET=\"$SET\" 은 복사 기준선에 적용되지 않는다 (intphys1_sliding 병합 config 그대로)"
ARGS=(--gpus "$GPUS")
[[ -n ${LIMIT:-} ]]      && ARGS+=(--limit "$LIMIT")
[[ -n ${SUBSET:-} ]]     && ARGS+=(--subset "$SUBSET")
[[ -n ${DRYRUN:-} ]]     && ARGS+=(--dryrun)
[[ -n ${VALIDATE:-} ]]   && ARGS+=(--validate)
[[ -n ${OUTDIR:-} ]]     && ARGS+=(--out "$OUTDIR")           # 벤치 · 모델이 하나일 때만 뜻이 있다
[[ -n ${THREADS:-} ]]    && ARGS+=(--threads "$THREADS")
[[ -n ${MAX_STARTS:-} ]] && ARGS+=(--max-starts "$MAX_STARTS")
cd "$CVPR_REPO"
fail=0
for b in $BENCHES; do for m in $MODELS; do
  echo "### intphys1/copy  $b / $m  ($(date '+%H:%M:%S'))  gpus=$GPUS [$CUDA_VISIBLE_DEVICES]"
  "$CVPR_PY" "$HERE/copy_baseline.py" --dataset "$b" --model "$m" "${ARGS[@]}" || { echo "!! 실패 copy $b/$m"; fail=1; }
done; done
exit $fail
