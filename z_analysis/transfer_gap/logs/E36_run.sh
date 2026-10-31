#!/usr/bin/env bash
# [X2] R3 E36 (2026-10-30): 잡음 판 (R2 E31c 정정판 프레임 frames_src{s}_ip1, 원본 내용 seed 확률적 반올림) 에서 IntPhys 1 Garrido 창 16 칸
#   (skip2_w16 · skip5_w16, WINDOW=garrido_w16) 을 릴리즈 · VOBench PFT 로 하네스 채점. skip2_w32 는 R2 E31c2 판을 재사용한다.
#   사용: SEEDS="1 2" PORT_BASE=31700 GPUS=2 JOBNAME=tgX2_E36a bash cvpr/harness/submit.sh z_analysis/transfer_gap/logs/E36_run.sh
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
C=/data2/local_datasets/world/world_analysis/cache/transfer_gap
RUNS=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_analysis/transfer_gap/results/q3/runs
mkdir -p "$RUNS"
for s in $SEEDS; do
  i=0
  for m in vith vith_release_vobftg; do
    i=$((i + 1))
    O=$RUNS/E36_ip1_src${s}_${m}_w16
    if [[ -f $O/summary.json ]]; then echo "skip $O"; continue; fi
    # 같은 노드에서 하네스 둘을 동시에 띄우면 자동 포트가 겹친다 (R2 E31c) → PORT_BASE 를 job 마다 따로, 판마다 포트를 정한다
    EVAL_DDP_PORT=$(( ${PORT_BASE:?PORT_BASE 필요} + 10 * s + i )) WINDOW=garrido_w16 SET="data.frames_root=$C/frames_src${s}_ip1" \
      TAG=tg_E36_ip1_src${s}_$m OUTDIR=$O GPUS=${GPUS:-2} \
      bash cvpr/harness/launch.sh cvpr/eval/intphys1/protocol.yaml intphys1_dev $m
  done
done
echo E36_RUN_DONE $SEEDS
