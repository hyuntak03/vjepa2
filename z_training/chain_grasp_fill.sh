#!/bin/bash
# GRASP 빈 칸 채우기 (2026-10-30, 사용자 "GRASP 성능 비어져 있는 거 지금 다 좀 돌려봐"). chain_baselines_bench.sh 가 끝나면 (로그 "끝") 끝난 run 만 차례로.
#   nohup bash z_training/chain_grasp_fill.sh > /dev/null 2>&1 &      로그 z_training/logs/chain_bench.log
# 순서 = 표 (VARIANTS.md §4) 에 있는 것 · 최근 것 먼저. 학습이 70 epoch 를 다 못 채운 run 은 뺐다 (README 표에 '중단').
cd "$(dirname "$0")/.."
RUNS=${RUNS:-"release_postft_pretrain natural_ar_vw_state_delta natural_ar_vw_state_hint natural_ar_vw_hint natural_ar_vw_state_ctxt_video k710_ssv2_how_ep70_cooldown64 vobench_release_postft_g_e20 vobench_ar_vw_state_postft_g intphys2_vuni_release_postft intphys2_vuni_ar_vw_state_postft_lora natural_ar_vw_state_laplace natural_ar_vw_lam natural_ar_vw_state_phys2d natural_ar_vw_state_phys3d_event_gthead_frozen_gm natural_ar_vw_state_phys3d_only natural_ar_vw_state_fps10_c16k8 natural_ctx_state natural_tube_k710 k710_ssv2_how_ep70 k710_ssv2_how_ep70_cooldown32"}
until grep -q "끝$" z_training/logs/chain_baselines_bench.log 2>/dev/null; do sleep 60; done
for r in $RUNS; do mkdir -p z_training/runs/$r/eval/logs; done
NOWAIT=1 SKIP_GPU_WAIT=1 BENCHES=grasp_level2 RUNS="$RUNS" bash z_training/chain_bench.sh
