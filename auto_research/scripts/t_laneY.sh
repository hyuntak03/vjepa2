#!/bin/bash
# [T] B′ 학습 없는 준비 (GPU 7 · 하나씩 · 스레드 8): 장난감 레인 끝 → 순방향 흐름 4 개 → u 판정 (무학습 기준) → 끊기 두 꼴 스모크 → 스모크 체크포인트로 u 판정 배관 확인
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "t_laneX.sh" > /dev/null; do sleep 15; done
# 대조 팔 (사전 등록 밖, 원인 귀속용): 변위 머리 (옮기기) + 토큰 L1 — 머리 구조 vs 손실 공간을 가른다
TB="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000 --head disp"
for f in 25 10; do env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/toy_inertia.py --name in_disptok_$f --inert_frac 0.$f $TB > auto_research/logs/toy_inertia_in_disptok_$f.log 2>&1; echo "done in_disptok_$f $(date -u +%H:%M)"; done
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
F=auto_research/scripts/flow_for_cache.py
run $F --cache /workspace/auto_research_cache/natural_holdout --src k710_320_holdout --fwd 1 > auto_research/logs/flowfwd_k710_holdout.log 2>&1
for s in k710_320 ht100m_1pct ssv2; do run $F --src $s --fwd 1 > auto_research/logs/flowfwd_$s.log 2>&1; done
echo "flows done $(date -u +%H:%M)"
run auto_research/scripts/u_judge.py --out auto_research/results/u_judge_baselines.json > auto_research/logs/u_judge_baselines.log 2>&1
SM=/workspace/auto_research_cache/ar_ft_smoke
run auto_research/scripts/disp_ft.py --name sm_comp0 --comp 0 --steps 100 --out $SM > auto_research/logs/sm_comp0.log 2>&1
run auto_research/scripts/disp_ft.py --name sm_cut_um --comp 1 --u_sg 1 --m_sg 1 --steps 100 --out $SM > auto_research/logs/sm_cut_um.log 2>&1
run auto_research/scripts/u_judge.py --ftdir $SM --ft sm_comp0 sm_cut_um --out auto_research/results/u_judge_smoke.json > auto_research/logs/u_judge_smoke.log 2>&1
echo "prep done $(date -u +%H:%M)"
