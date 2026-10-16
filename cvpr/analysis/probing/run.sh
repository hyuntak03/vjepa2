#!/usr/bin/env bash
# =============================================================================
# analysis/probing — z / p / h 세 지점 attentive probing (정체성 · 배경을 읽는가)
#
#   GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith                 # attn_probe (기본)
#   PROTO=attn_probe_xfer GPUS=8 bash cvpr/analysis/probing/run.sh v11_full vith   # h<->p 양방향 이식
#   PROTO=attn_probe_imp  GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith        # 불가능 변이에서 target encoder 가 바뀐 정체성을 읽는가
#   SET="probing.optim=attn_50 probing.targets.shape=null" ...        # 일회성 변형 (yaml 을 새로 만들지 않는다)
#   WINDOW=c8t32 ...                                                  # 문맥 8 (토큰 캐시가 _wc8t32 로 갈린다)
#   DRYRUN=1 bash cvpr/analysis/probing/run.sh v11
#
# ⚠️ probing 은 가능(possible) 변이만 (block_types: [obj]) · split 은 block 단위 · dtype bfloat16 — 프로토콜 yaml 상단 참고.
# ⚠️ DDP probing 은 실행마다 흔들린다 (p self 최대 20 pt). 결정론이 필요하면 GPUS=1.
# ⚠️ 토큰 캐시 (features.cache_dir = $CVPR_CACHE/<tag>) 는 노드 로컬. v11 한 벌 ≈ 420 GiB.
# 결과: $CVPR_RESULTS/analysis/probing/<proto>__<ds>_<model>/{summary.json, predictions.json}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
PROTO=${PROTO:-attn_probe}
DATASETS=${1:-${DATASETS:-v11}}
MODELS=${2:-${MODELS:-vith}}
[[ -f "$HERE/$PROTO.yaml" ]] || { echo "ERROR: $HERE/$PROTO.yaml 이 없다 (attn_probe | attn_probe_imp | attn_probe_xfer)"; exit 1; }
fail=0
for d in $DATASETS; do for m in $MODELS; do
  echo "### probing/$PROTO  $d / $m  ($(date '+%H:%M:%S'))"
  bash "$CVPR_ROOT/harness/launch.sh" "$HERE/$PROTO.yaml" "$d" "$m" || { echo "!! 실패 $d/$m"; fail=1; }
done; done
exit $fail
