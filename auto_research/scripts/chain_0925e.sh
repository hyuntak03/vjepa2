#!/bin/bash
# 2026-09-25 묶음 E (적대 검증 후속): P3b 형식·누적·내용 · M6 역사 시간 이동 · E-SSv2 토큰 수준 오라클/β.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
( run $P $S/p3b_closure_v2.py --gpus 1 --limit 4 --out $C/_smoke/p3b && run $P $S/p3b_closure_v2.py --gpus 8 )
( run $P $S/m6_history_shift_v3.py --gpus 1 --limit 4 --out $C/_smoke/m6 && run $P $S/m6_history_shift_v3.py --gpus 8 )
( run $P $S/e_ssv2_extract.py --gpus 1 --limit 4 --out $C/_smoke/ssv2_e3 && run $P $S/e_ssv2_extract.py --gpus 8 --out $C/ssv2_e3 )
# M2b 경계 패칭 2 판 (적대 검증 반영: 천장 · 절단 경계 · block 짝 · visible V · 화면 밖 NaN)
( run $P $S/m2b_boundary_v2.py --gpus 1 --limit 8 --out $C/_smoke/m2b && run $P $S/m2b_boundary_v2.py --gpus 8 )
