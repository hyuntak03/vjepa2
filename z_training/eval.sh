#!/bin/bash
# -----------------------------------------------------------------------------
# 학습한 predictor 를 기존 채점 하네스로 잰다 (run.sh 를 그대로 부른다).
#
#   GPUS=8 bash z_training/eval.sh <run 이름|폴더> [데이터셋=v11] [프로토콜=surprise_c16t32]
#   GPUS=8 bash z_training/eval.sh postft1 intphys1_dev intphys1_sliding
#   CKPT=e5.pt GPUS=8 bash z_training/eval.sh scratch1 v11
#   LIMIT=8 DRYRUN=1 bash z_training/eval.sh scratch1            # 병합 확인만
#
# 동작: encoder 는 run 의 base checkpoint(models.md 의 모델), predictor 만 <run>/<CKPT> 에서 읽는다
#       (analysis/intphys2/model.py 의 model.predictor_checkpoint). 통짜 파일을 만들 필요가 없다.
# 결과: <run>/eval/<프로토콜>__<데이터셋>_<ckpt>/summary.json  (기본 results_root 를 안 건드린다)
# 캐시 TAG 도 run 이름으로 갈라 두어 vith 토큰 캐시를 덮지 않는다 (surprise 는 캐시를 안 쓴다).
# 릴리즈 predictor 기준선은 그냥 run.sh 로: GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11 vith
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"   # 경로 정본: harness/paths.env
PY=$VJEPA2_PY
PROJ=${PROJ:-$VJEPA2_ROOT}
RUN=${1:?"사용법: eval.sh <run> [데이터셋] [프로토콜]"}
DATASET=${2:-v11}
PROTOCOL=${3:-surprise_c16t32}
CKPT=${CKPT:-latest.pt}
[[ -d "$RUN" ]] || RUN="$PROJ/z_training/runs/$RUN"
RUN=$(cd "$RUN" && pwd)
CK="$RUN/$CKPT"
[[ -f "$CK" ]] || { echo "ERROR: $CK 가 없다"; exit 1; }
MODEL=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model'].get('base','vith'))" "$RUN/config.yaml" 2>/dev/null || echo vith)
# predictor 구조(kind / n_registers)를 run config 에서 읽어 채점기에 넘긴다 — 안 넘기면 릴리즈 클래스로
# 지어져 prefix-마스크 predictor 가 full attention 으로 **조용히 틀리게** 채점된다 (2026-09-22)
KIND_SET=$("$PY" - "$RUN/config.yaml" <<'PY'
import yaml, sys
pc = (yaml.safe_load(open(sys.argv[1])).get("model") or {}).get("predictor") or {}
print(" ".join(f"model.predictor.{k}={pc[k]}" for k in ("kind", "n_registers") if k in pc))
PY
)
echo "predictor: ${KIND_SET:-oneshot(릴리즈 구조)}"
NAME=$(basename "$RUN"); STEM=${CKPT%.pt}
OUTDIR="$RUN/eval/${PROTOCOL}__${DATASET}_${STEM}"
TAG="${DATASET}_${NAME}_${STEM}"
echo "run $RUN | ckpt $CKPT | base model $MODEL | -> $OUTDIR"
SET="model.predictor_checkpoint=$CK $KIND_SET ${SET:-}" OUTDIR="$OUTDIR" TAG="$TAG" GPUS=${GPUS:-1} \
  bash z_research/scripts/run.sh "$PROTOCOL" "$DATASET" "$MODEL"
[[ -n "${DRYRUN:-}" ]] && exit 0
[[ -n "${LIMIT:-}" ]] && OUTDIR="${OUTDIR}_smoke${LIMIT}"      # run.sh 가 LIMIT 이면 _smoke{N} 을 붙인다
"$PY" - "$OUTDIR/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
sp = s.get("surprise") or {}
print("\n=== summary.surprise (요약) ===")
print(json.dumps({k: v for k, v in sp.items() if k in ("overall", "chance", "pairing", "by_block_type", "block_pairwise")}, indent=1, ensure_ascii=False)[:3000])
print(f"\n전체: {sys.argv[1]}")
PYEOF
