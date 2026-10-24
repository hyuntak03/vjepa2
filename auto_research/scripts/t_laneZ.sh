#!/bin/bash
# [T] (GPU 7 · 하나씩 · 스레드 8): 장난감 대조 마무리 → 상태 head 확인 (1) 세 모델 → VOBench 상태 surprise (2) 세 모델 → B′ 학습 없는 준비
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "toy_inertia.py --name in_disptok_25" > /dev/null; do sleep 10; done
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
TB="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000 --head disp"
grep -q "done" auto_research/logs/toy_inertia_in_disptok_10.log 2>/dev/null || { run auto_research/scripts/toy_inertia.py --name in_disptok_10 --inert_frac 0.10 $TB > auto_research/logs/toy_inertia_in_disptok_10.log 2>&1; }
echo "toys done $(date -u +%H:%M)"
V3=/workspace/dataset/training_data/phys3d_single_v1/val/boundary.tsv; V2=/workspace/dataset/training_data/phys2d_single_v1/val/boundary.tsv
run auto_research/scripts/state_obs_check.py --run natural_ar_vw_state_phys3d_single --val $V3 > auto_research/logs/state_obs_phys3d_single.log 2>&1; echo "obs3s $(date -u +%H:%M)"
run auto_research/scripts/state_obs_check.py --run natural_ar_vw_state_phys3d_event --val $V3 > auto_research/logs/state_obs_phys3d_event.log 2>&1; echo "obs3e $(date -u +%H:%M)"
run auto_research/scripts/state_obs_check.py --run natural_ar_vw_state_phys2d_single --val $V2 > auto_research/logs/state_obs_phys2d_single.log 2>&1; echo "obs2s $(date -u +%H:%M)"
for r in phys3d_single phys3d_event phys2d_single; do run auto_research/scripts/state_vob.py --run natural_ar_vw_state_$r > auto_research/logs/state_vob_$r.log 2>&1; echo "vob $r $(date -u +%H:%M)"; done
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
