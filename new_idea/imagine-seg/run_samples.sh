#!/bin/bash
# 샘플 생성 (파일럿 확인용): S0 클립 → S1–S3 SAM2 물체 (GPU) → S5 가림 합성 → S7 보기
# 계획: DATA_PIPELINE_2026-10-31.md
#   DEV=cuda:0 bash run_samples.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venv/vjepa/bin/python
OUT=${OUT:-/workspace/dataset/training_data/imagine_seg_v0_samples}
BANK=${BANK:-/workspace/dataset/assets/occluder_bank_coco_v0}
DEV=${DEV:-cuda:0}
N=${N:-40}            # 소스별로 뽑을 클립 수 (거르기에서 빠지는 것 감안)
SHOW=${SHOW:-10}      # 소스별로 보여 줄 수

[ -f "$BANK/index.json" ] || $PY "$HERE/datagen/s4_occluder_bank.py" --out "$BANK" --max 3000
for S in ssv2 k400; do
  [ -d "$OUT/$S" ] || $PY "$HERE/datagen/s0_clips.py" --source $S --n $N --out "$OUT/$S"
  $PY "$HERE/datagen/s1_objects_sam2.py" --clips "$OUT/$S" --device "$DEV"
  $PY "$HERE/datagen/s5_synth.py" --clips "$OUT/$S" --bank "$BANK" --tag v0
done
$PY "$HERE/datagen/s7_viz.py" --clips "$OUT/ssv2" "$OUT/k400" --tag v0 --out "$HERE/samples/v0" --max_per_source $SHOW
