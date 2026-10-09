#!/bin/bash
# 2026-09-25 묶음 B: P3 닫힘 검정 (스모크 → 본 실행).
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; T=/data2/local_datasets/world/world_analysis/cache/auto_research/_smoke
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
( run $P $S/p3_closure_v3.py --gpus 1 --limit 4 --fit-clips 16 --out $T/p3 && run $P $S/p3_closure_v3.py --gpus 8 )
# M1/M3 (묶음 A 에서 스모크 실패 → donor 풀 수정 뒤 여기서)
( run $P $S/m13_memory_state_v3.py --gpus 1 --limit 4 --out $T/m13 && run $P $S/m13_memory_state_v3.py --gpus 8 )
# M4 조회 반경 기전 (stride 별 attention · knockout)
( run $P $S/m4_locality_stride_v3.py --gpus 1 --limit 4 --out $T/m4 && run $P $S/m4_locality_stride_v3.py --gpus 8 )
# E-SSv2 재추출 (흐림 기준선 · 마지막 운동 지도 추가)
run $P $S/e_ssv2_extract.py --gpus 8
