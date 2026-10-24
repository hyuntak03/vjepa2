#!/bin/bash
# 공정한 시험 판정 (진행자 2, 2026-10-22) — 네 run 의 e8 평가가 끝나면, 사슬이 e2–e6 을 다시 덮어쓰는 것과 겹치지 않게
# 결과 폴더를 스냅샷으로 복사한 뒤 그 복사본에서 사전 등록 판정 스크립트 (fair_verdict.py, 고치지 않음) 를 돌린다.
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
R=paper_lab/eval/results; SNAP=/local_datasets/world/world_analysis/cache/paper_lab_fair_snapshot
rm -rf $SNAP; mkdir -p $SNAP
for a in AXs0 AXs1 B0s0 B0s1; do for e in 2 4 6 8; do [ -d $R/fair${a}_e$e ] && cp -r $R/fair${a}_e$e $SNAP/; done; done
cp -r $R/vast70 $SNAP/
echo "[$(date '+%F %T')] snapshot done: $(ls $SNAP | tr '\n' ' ')"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $P - <<PYEOF
import sys
from pathlib import Path
sys.argv = ["fair_verdict"]
sys.path.insert(0, "paper_lab/scripts"); sys.path.insert(0, "paper_lab/eval")
import passline as PL
import fair_verdict as FV
PL.OUTROOT = Path("$SNAP"); FV.RES = Path("$SNAP")
FV.main()
PYEOF
