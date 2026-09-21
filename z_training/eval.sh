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
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
RUN=${1:?"사용법: eval.sh <run> [데이터셋] [프로토콜]"}
DATASET=${2:-v11}
PROTOCOL=${3:-surprise_c16t32}
CKPT=${CKPT:-latest.pt}
[[ -d "$RUN" ]] || RUN="$PROJ/z_training/runs/$RUN"
RUN=$(cd "$RUN" && pwd)
CK="$RUN/$CKPT"
[[ -f "$CK" ]] || { echo "ERROR: $CK 가 없다"; exit 1; }
MODEL=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model'].get('base','vith'))" "$RUN/config.yaml" 2>/dev/null || echo vith)
NAME=$(basename "$RUN"); STEM=${CKPT%.pt}
OUTDIR="$RUN/eval/${PROTOCOL}__${DATASET}_${STEM}"
TAG="${DATASET}_${NAME}_${STEM}"
echo "run $RUN | ckpt $CKPT | base model $MODEL | -> $OUTDIR"
SET="model.predictor_checkpoint=$CK ${SET:-}" OUTDIR="$OUTDIR" TAG="$TAG" GPUS=${GPUS:-1} \
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
