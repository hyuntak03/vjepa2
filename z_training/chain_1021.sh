#!/bin/bash
# 2026-10-21 chain (사용자 지시): 3D only 학습 끝 → 3D 사건 감독 run 남은 2 epoch 이어 학습 → 시트 빈칸 채점 (EK100 probe 와 VRAM 을 나눠 쓴다)
#   tmux new-session -d -s chain_1021 "bash z_training/chain_1021.sh"     로그: z_training/logs/chain_1021.log
#   플래그: $FLAGS/phys3d_event_done (EK100 실행이 이걸 기다린다)
cd "$(dirname "$0")/.."
FLAGS=${FLAGS:?"FLAGS=<폴더>"}
L=z_training/logs/chain_1021.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a $L; }
# 2026-10-21 고침: tmux_train.sh 는 끝나도 창을 남겨 둔다 ("=== train.sh 종료") → 세션이 아니라 그 문구를 기다린다
wait_tmux() { while tmux has-session -t "$1" 2>/dev/null && ! tmux capture-pane -p -S -50 -t "$1" | grep -q "=== train.sh 종료"; do sleep 60; done; tmux kill-session -t "$1" 2>/dev/null; true; }

if [[ -z ${SKIP_TRAIN:-} ]]; then   # 2026-10-22: SKIP_TRAIN=1 → 1 · 2 를 건너뛴다 (이어 학습을 단독으로 끝낸 뒤 채점만)
say "1. 대기: train_natural_ar_vw_state_phys3d_only"
wait_tmux train_natural_ar_vw_state_phys3d_only
say "2. 이어 학습: natural_ar_vw_state_phys3d_event (e68 → e70, auto_resume)"
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_phys3d_event >> $L 2>&1
sleep 30; wait_tmux train_natural_ar_vw_state_phys3d_event
EP=$(/venv/vjepa/bin/python -c "import torch;print(torch.load('z_training/runs/natural_ar_vw_state_phys3d_event/latest.pt',map_location='cpu',weights_only=False).get('epoch'))")
say "   끝: latest epoch $EP"
touch $FLAGS/phys3d_event_done
fi
EP=$(/venv/vjepa/bin/python -c "import torch;print(torch.load('z_training/runs/natural_ar_vw_state_phys3d_event/latest.pt',map_location='cpu',weights_only=False).get('epoch'))")

CK=$PWD/z_training/runs/natural_ar_vw_state_phys3d_event/latest.pt
say "3. phys3d_event latest (e$EP): VOBench late · gravity"
for d in vobench_v1_late vobench_v1_gravity; do
  RESULTS_ROOT=legacy SUFFIX=_e$EP SET="model.predictor_checkpoint=$CK surprise.batch_size=4" GPUS=8 bash cvpr/eval/testbed/run.sh $d vast_ar_vw_state_phys3d_event >> $L 2>&1; say "   $d $?"
done
say "4. phys3d_event latest: IntPhys1 + IntPhys2"
GPUS=8 bash z_training/eval_bench.sh z_training/runs/natural_ar_vw_state_phys3d_event latest.pt >> $L 2>&1; say "   $?"
say "5. phys3d_event latest: IntPhys1 late (uni)"
DS=intphys1_late_uni SUFFIX=_e$EP SET="model.predictor_checkpoint=$CK" GPU_IDS="0 1 2 3 4 5 6 7" bash cvpr/eval/intphys1_late/run.sh vast_ar_vw_state_phys3d_event >> $L 2>&1; say "   $?"
say "6. ctxt_video: IntPhys2"
WINDOWS_I1="" WINDOWS_I2="16 32 48" GPUS=8 bash z_training/eval_bench.sh z_training/runs/natural_ar_vw_state_ctxt_video latest.pt >> $L 2>&1; say "   $?"
say "7. InfLevel + GRASP — phys3d_event · GRASP — gres · phys2d_single · ctxt_video (EK100 과 VRAM 공유: batch 절반)"
SKIP_GPU_WAIT=1 NOWAIT=1 MB=72 VB=8 RUNS="natural_ar_vw_state_phys3d_event" BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity grasp_level2" bash z_training/chain_bench.sh >> $L 2>&1
SKIP_GPU_WAIT=1 NOWAIT=1 MB=72 VB=8 RUNS="natural_ar_vw_state_gres natural_ar_vw_state_phys2d_single natural_ar_vw_state_ctxt_video" BENCHES="grasp_level2" bash z_training/chain_bench.sh >> $L 2>&1
say "끝"
