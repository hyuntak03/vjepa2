#!/bin/bash
# 학습된 predictor 배터리 분석: 태그별 P2 · P2b · P3 · M13 분석 → cmp_predictors.py.  사용: run_battery_analysis.sh <tag> (pv1 | ariel | v11ft)
t=$1; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; C=/data2/local_datasets/world/world_analysis/cache/auto_research
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/scripts
export OUT_TAG=_$t OMP_NUM_THREADS=16
[ -f $C/v3_p2_$t/loc_tru.npy ] && $P p2_analyze.py $C/v3_p2_$t > /dev/null && $P p2b_units.py $C/v3_p2_$t > /dev/null && echo "p2 ok"
[ -f $C/v3_p3_$t/p3.npy ] && $P p3_analyze.py $C/v3_p3_$t > /dev/null && echo "p3 ok"
$P -c "import numpy as np,sys; a=np.load('$C/v3_m13_$t/loc.npy', mmap_mode='r'); sys.exit(0 if np.isfinite(a[:,0,0,0]).all() else 1)" 2>/dev/null && $P m13_analyze.py $C/v3_m13_$t > /dev/null && echo "m13 ok" || echo "m13 미완"
