#!/usr/bin/env bash
# [X2] R2 E28 GPU 몫: IntPhys 2 문맥 끝 판 (uni_m10 · final) 빈 복사 + 처음 보는 물체 PFT 를 외부 문맥 끝 판 8 개에 채점
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
OBJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs/vobench_release_postft_objholdout/latest.pt
RUNS=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_analysis/transfer_gap/results/q3_r2/runs
for DS in intphys2_late_uni_m10 intphys2_late_final; do
  [[ -f cvpr/results/eval/intphys2_late/${DS}_vith/copy_baseline.json ]] || DS=$DS COPY=1 bash cvpr/eval/intphys2_late/run.sh vith
done
for DS in intphys2_late_uni intphys2_late_uni_m10 intphys2_late_final; do
  SET="model.predictor_checkpoint=$OBJ" TAG=tg_E28_${DS}_objho OUTDIR=$RUNS/E28_${DS}_objho GPUS=1 \
    bash cvpr/harness/launch.sh cvpr/eval/intphys2_late/protocol.yaml $DS vith
done
for P in uni hid rev rev_dn1 auto2-6r12fit; do
  DS=intphys1_late_$P
  SET="model.predictor_checkpoint=$OBJ" TAG=tg_E28_${DS}_objho OUTDIR=$RUNS/E28_${DS}_objho GPUS=1 \
    bash cvpr/harness/launch.sh cvpr/eval/intphys1_late/protocol.yaml $DS vith
done
echo E28_GPU_DONE
