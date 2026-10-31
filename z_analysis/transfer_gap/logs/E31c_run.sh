#!/usr/bin/env bash
# [X2] R2 E31c (2026-10-30 정정판: 원본 내용으로 seed 한 frames_src — 첫 판 frames_sr 은 matched 문맥이 깨져 기록으로만): 확률적 반올림 seed 별 — VOBench test (한 번 통과 te, 릴리즈 · PFT · 복사) + IntPhys 1 Garrido skip2_w32 (하네스, 릴리즈 · PFT)
#   사용: SEEDS="1 2" sbatch --gres=gpu:2 ... E31c_run.sh
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
C=/data2/local_datasets/world/world_analysis/cache/transfer_gap
RUNS=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_analysis/transfer_gap/results/q3_r2/runs
R=z_training/runs
for s in $SEEDS; do
  [[ -f $C/te_E31c2_src$s/summary.json ]] || $P z_research/scripts/analysis/transfer_gap_te.py --name E31c2_src$s --dataset vobench_v1_split2_test \
     --set data.frames_root=$C/frames_src${s}_vob --preds release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt --gpus 2
  i=0
  for m in vith vith_release_vobftg; do
    i=$((i + 1))
    # 2026-10-30: job 둘이 동시에 하네스를 띄워 자동 포트가 겹쳤다 → PORT_BASE 를 job 마다 따로 주고 판마다 포트를 정한다
    EVAL_DDP_PORT=$(( ${PORT_BASE:?PORT_BASE 필요} + 10 * s + i )) WINDOW=garrido_skip2_w32 SET="data.frames_root=$C/frames_src${s}_ip1" TAG=tg_E31c2_ip1_src${s}_$m OUTDIR=$RUNS/E31c2_ip1_src${s}_${m}_skip2w32 GPUS=2 \
      bash cvpr/harness/launch.sh cvpr/eval/intphys1/protocol.yaml intphys1_dev $m
  done
done
echo E31C_RUN_DONE $SEEDS
