#!/bin/bash
# [T] (GPU 7 · 하나씩): 상태 head 관측 확인 마무리 → 관측용 판독 head (3D · 2D) → VOBench 상태 surprise (세 모델) → 관성 2 · 5 % 장난감 → B′ 준비
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "state_obs_check.py --run natural_ar_vw_state_phys3d_event" > /dev/null; do sleep 10; done
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
V2=/workspace/dataset/training_data/phys2d_single_v1/val/boundary.tsv
run auto_research/scripts/state_obs_check.py --run natural_ar_vw_state_phys2d_single --val $V2 > auto_research/logs/state_obs_phys2d_single.log 2>&1; echo "obs2s $(date -u +%H:%M)"
run auto_research/scripts/state_obs_head.py --dom phys3d_single_v1 --run natural_ar_vw_state_phys3d_single > auto_research/logs/state_obs_head_phys3d.log 2>&1; echo "ohead3 $(date -u +%H:%M)"
OH3=/workspace/auto_research_cache/state_obs/obs_head_phys3d_single_v1.pt
for r in phys3d_single phys3d_event; do run auto_research/scripts/state_vob.py --run natural_ar_vw_state_$r --obs_head $OH3 > auto_research/logs/state_vob_$r.log 2>&1; echo "vob $r $(date -u +%H:%M)"; done
run auto_research/scripts/state_obs_head.py --dom phys2d_single_v1 --run natural_ar_vw_state_phys2d_single > auto_research/logs/state_obs_head_phys2d.log 2>&1; echo "ohead2 $(date -u +%H:%M)"
run auto_research/scripts/state_vob.py --run natural_ar_vw_state_phys2d_single --obs_head /workspace/auto_research_cache/state_obs/obs_head_phys2d_single_v1.pt > auto_research/logs/state_vob_phys2d_single.log 2>&1; echo "vob phys2d_single $(date -u +%H:%M)"
TB="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000 --head disp --disp_loss 3"
for f in 05 02; do run auto_research/scripts/toy_inertia.py --name in_disp_$f --inert_frac 0.$f $TB > auto_research/logs/toy_inertia_in_disp_$f.log 2>&1; echo "toy in_disp_$f $(date -u +%H:%M)"; done
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
