#!/bin/bash
# 시트 AR_full_tabl 빈 칸 채점 (2026-11-09 21:xx KST, 사용자 "V-JEPA2 · COSMOS · VideoMAEv2 · V-JEPA2.1 · AR · AR L1 · AR negative loss (p=0.2) 를 Intuitive Physics 탭처럼").
#   v5id 렌더가 다 끝난 뒤 (run_v5_mps.out 의 V5_ALL_DONE) 에만 돈다 — GPU 를 렌더와 나눠 쓰지 않게.
#   (11-09 22:4x 사용자 "VOBench 는 gravity 제외" → gravity 뺌; 22:5x "V-JEPA2 IntPhys2 · InfLevel 은 다른 시트 값" → 릴리즈 채점 뺌) 2 GRASP · InfLevel (L1 · p 0.2 학습 run) 3 InfLevel (릴리즈) 4 IntPhys2 Main (릴리즈, 창 16 · 32 · 48) 5 표 다시 계산.
#   setsid nohup bash z_training/chain_ar_full_fill_20261109.sh > z_training/logs/chain_ar_full_fill_20261109.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
R5=/workspace/dataset/training_data/blender_track/v5id/logs/run_v5_mps.out
echo "대기: v5id 렌더 끝 $(ts)"
until grep -q V5_ALL_DONE $R5 2>/dev/null; do sleep 120; done
sleep 60
echo "## 2 GRASP · InfLevel (L1 · p0.2) $(ts)"
NOWAIT=1 SKIP_GPU_WAIT=1 RUNS="v4bc4_ar_vw_state_negimp_vanish_app02 v4bc4_ar_vw_state_postft" BENCHES="grasp_level2 inflevel_continuity inflevel_solidity inflevel_gravity" bash z_training/chain_bench.sh > /dev/null 2>&1
echo "## 5 표 $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/ar_full_table.py --json z_analysis/results/ar_full_table_20261109.json | tail -7
echo "AR_FULL_FILL_DONE $(ts)"
