#!/usr/bin/env bash
# =============================================================================
# cvpr/harness/submit.sh — sbatch 템플릿 **하나**. run.sh 와 파일이 달라 자기제출 폭주가 구조적으로 없다.
#
#   GPUS=8 bash cvpr/harness/submit.sh cvpr/eval/testbed/run.sh v11 vith
#   GPUS=8 WINDOW=c8t32 SET="scoring.pairing=cross" bash cvpr/harness/submit.sh cvpr/eval/testbed/run.sh gravity_realistic
#   DEP=123456 bash cvpr/harness/submit.sh ...            # afterok 의존
#
# SLURM 자원 (기본값은 cvpr/env.sh 의 CVPR_*)
#   JOBNAME   기본 <task 폴더 이름>        NODE      기본 $CVPR_NODE (vll5). NODE=any 면 노드를 고정하지 않는다
#   PARTITION GPUS  CPUS_PER_GPU  MEM_PER_GPU  TIME     DEP (afterok job id)
# run.sh 가 읽는 env (SET · WINDOW · LIMIT · RESULTS_ROOT ...) 는 **그대로 job 에 넘어간다** (--export=ALL).
#   옛 `--export=ALL,SET=...` 는 콤마에서 잘려 base64 가 필요했다 — 여기서는 env 를 통째로 넘기므로 필요 없다.
# 로그: $CVPR_LOGS/<jobname>_<jobid>.out
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
RUN=${1:?"사용법: submit.sh <run.sh> [인자...]"}; shift
RUN=$(cd "$(dirname "$RUN")" && pwd)/$(basename "$RUN")
[[ -f $RUN ]] || { echo "ERROR: $RUN 이 없다"; exit 1; }
GPUS=${GPUS:-8}
JOBNAME=${JOBNAME:-$(basename "$(dirname "$RUN")")}
NODE=${NODE:-$CVPR_NODE}
mkdir -p "$CVPR_LOGS"
export GPUS
JID=$(sbatch --parsable \
  --job-name="$JOBNAME" \
  --partition="${PARTITION:-$CVPR_PARTITION}" \
  $([[ $NODE != any ]] && echo "--nodelist=$NODE") \
  --gres="gpu:$GPUS" \
  --cpus-per-gpu="${CPUS_PER_GPU:-$CVPR_CPUS_PER_GPU}" \
  --mem-per-gpu="${MEM_PER_GPU:-$CVPR_MEM_PER_GPU}" \
  --time="${TIME:-$CVPR_TIME}" \
  --output="$CVPR_LOGS/%x_%j.out" --error="$CVPR_LOGS/%x_%j.err" \
  ${DEP:+--dependency=afterok:$DEP} \
  --export=ALL \
  <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "[submit.sh] \$(hostname) \$(date '+%F %T')  job \$SLURM_JOB_ID  gpus=$GPUS"
cd "$CVPR_REPO"
exec bash "$RUN" $(printf '%q ' "$@")
EOF
)
echo "submitted $JID  ($JOBNAME, gpu:$GPUS, node=$NODE)  log: $CVPR_LOGS/${JOBNAME}_${JID}.out"
echo "$JID"
