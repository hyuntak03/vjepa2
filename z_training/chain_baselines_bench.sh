#!/bin/bash
# 다른 계열 모델 (V-JEPA 2.1 ViT-g · VideoMAEv2-g · DINO-Foresight …) 을 VOBench late · IntPhys 2 Main · GRASP 로 채점 (2026-10-30, 사용자:
#   "cosmos-4b, V-JEPA2.1, VideoMAEv2-g, DINO-Foresight 이거 IntPhys2 late occ 랑 GRASP 좀 돌려봐").
#   채점 정의는 z_training/VARIANTS.md §6 과 같다 — VOBench late = 쌍 정확도 (primitive / realistic), IntPhys 2 = Main 창 16 · 32 · 48 열별 최고,
#   GRASP = Garrido A.8 창 16 · 32 × skip 2 · 5 · 10 → garrido_rescore (property 별 최고 → macro).
#
#   MODELS="vjepa21g videomae2g" nohup bash z_training/chain_baselines_bench.sh > /dev/null 2>&1 &
#   로그: z_training/logs/chain_baselines_bench.log (각 칸 로그는 그 하네스 자리)
# 결과: VOBench  z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_late_<m>[_w…]/
#       IntPhys 2 cvpr/results/eval/intphys2/intphys2_main_<m>_w<창>/
#       GRASP    z_research/Benchmarks/exp_results/intphys1_sliding__grasp_level2_<m>_w{16,32}/
# dinof_*: CVPR_NFS=/workspace/world_model (레지스트리의 ${CVPR_NFS}/DINO-Foresight/checkpoints → checkpoint/dino_foresight 심볼릭), LOG= 로 따로.
# 이미 끝난 칸 (summary.json) 은 각 하네스가 건너뛴다. GPU 를 비울 때까지 기다리지 않는다 (렌더와 VRAM 을 나눠 쓴다).
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"vjepa21g videomae2g"}
STEPS=${STEPS:-"vobench intphys2 grasp"}
GPUS=${GPUS:-8}
I2_ROOT=${I2_ROOT:-/workspace/dataset/benchmarks/IntPhys2}
LOG=${LOG:-z_training/logs/chain_baselines_bench.log}; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
for m in $MODELS; do
  for st in $STEPS; do
    say "$m / $st 시작"
    case $st in
      vobench)
        W=""; [[ $m == videomae2g ]] && W=c8t16_s6
        WINDOW=$W RESULTS_ROOT=legacy GPUS=$GPUS bash cvpr/eval/testbed/run.sh vobench_v1_late "$m" >> "$LOG.$m.vobench" 2>&1 ;;
      intphys2)
        SET="data.root=$I2_ROOT" GPUS=$GPUS bash cvpr/eval/intphys2/run.sh intphys2_main "$m" >> "$LOG.$m.intphys2" 2>&1 ;;
      grasp)
        if [[ $m == dinof_* ]]; then   # dinof 는 옛 하네스 (configs/protocols/models.md) 에 없다 → cvpr intphys1 run.sh (창 dinof_w5 · w8 · w16, 문맥 4 고정)
          GPUS=$GPUS bash cvpr/eval/intphys1/run.sh grasp_level2 "$m" >> "$LOG.$m.grasp" 2>&1
        else
          BENCHES=grasp_level2 MODELS=$m GPUS=$GPUS bash z_research/Benchmarks/run_all.sh >> "$LOG.$m.grasp" 2>&1
        fi ;;
    esac
    say "$m / $st 끝 (rc=$?)"
  done
done
say "끝"
