#!/usr/bin/env bash
# =============================================================================
# eval/testbed/copy_baseline.sh — 복사 기준선 (V-JEPA 계열, fixed c16t32). 예측 없이 문맥 마지막 튜블릿 LN(z)[7] 을
# 미래 8 슬롯 전부에 복사해 같은 matched pair 를 채점한다 (정의 · 단서는 README.md).
#
#   GPUS=8 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith                 # 데이터셋 · 모델 (기본값)
#   GPUS=8 bash cvpr/eval/testbed/copy_baseline.sh "v11_split_test v11_realistic" "vith vith_pft_v11_e10"   # 행렬 (공백 구분)
#   COPY=1 GPUS=8 bash cvpr/eval/testbed/run.sh v11_split_test vith                     # run.sh 의 COPY=1 이 여기로 온다 (dinof_* 는 로더 플래그)
#   DRYRUN=1 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith               # 두 resolver 대조 · 이름 · 명령만 (GPU 0 장)
#   LIMIT=16 GPUS=1 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith        # 스모크 (폴더에 _smoke16)
#   VALIDATE=1 ...                                                                     # 끝난 뒤 te --validate (v11_split_test 만)
#
# 엔진 = 기존 z_research/scripts/analysis/te_v11_score.py (predictor `own` + copy, 슬롯별 L1) 을 copy_baseline.py 가 import 해 돌린다.
# 모델 한 개당 ViT-H 8 GPU 로 v11_split_test 10,752 쌍 ≈ 0.2 s/clip/rank (te docstring 값, 이 코드 스페이스에서는 미측정).
# 결과: $CVPR_RESULTS/eval/testbed/copy/<ds>_<model>[_smokeN]/{summary.json, copy_summary.json, l1.npy, copy.npy, hcopy.npy, meta.json, _resolved.yaml, _meta.json}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
DATASETS=${1:-${DATASETS:-v11_split_test}}
MODELS=${2:-${MODELS:-vith}}
GPUS=${GPUS:-1}
GPU_IDS=${GPU_IDS:-$(seq -s ' ' 0 $((GPUS - 1)))}
GPUS=$(wc -w <<<"$GPU_IDS")
export CUDA_VISIBLE_DEVICES=$(tr ' ' ',' <<<"$GPU_IDS")      # te_*_score.py 는 mp.spawn 으로 보이는 GPU 앞 N 장을 cuda:0..N-1 로 쓴다 (evals.main 과 다르다)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${THREADS:-4}}
[[ -n ${WINDOW:-} && $WINDOW != c16t32 ]] && { echo "ERROR: 복사 기준선은 창 c16t32 고정이다 (te_v11_score.py 가 문맥 16 / 32 장을 검사한다). WINDOW=$WINDOW 는 못 쓴다"; exit 1; }
[[ -n ${SET:-} ]] && echo "!! SET=\"$SET\" 은 복사 기준선에 적용되지 않는다 (te_v11_score.py 는 surprise_c16t32 병합 config 를 그대로 쓴다)"
ARGS=(--gpus "$GPUS")
[[ -n ${LIMIT:-} ]]    && ARGS+=(--limit "$LIMIT")
[[ -n ${DRYRUN:-} ]]   && ARGS+=(--dryrun)
[[ -n ${VALIDATE:-} ]] && ARGS+=(--validate)
[[ -n ${OUTDIR:-} ]]   && ARGS+=(--out "$OUTDIR")             # 데이터셋 · 모델이 하나일 때만 뜻이 있다
[[ -n ${BATCH:-} ]]    && ARGS+=(--batch "$BATCH")
[[ -n ${WORKERS:-} ]]  && ARGS+=(--workers "$WORKERS")
[[ -n ${THREADS:-} ]]  && ARGS+=(--threads "$THREADS")
cd "$CVPR_REPO"
fail=0
for d in $DATASETS; do for m in $MODELS; do
  echo "### testbed/copy  $d / $m  ($(date '+%H:%M:%S'))  gpus=$GPUS [$CUDA_VISIBLE_DEVICES]"
  "$CVPR_PY" "$HERE/copy_baseline.py" --dataset "$d" --model "$m" "${ARGS[@]}" || { echo "!! 실패 copy $d/$m"; fail=1; }
done; done
exit $fail
