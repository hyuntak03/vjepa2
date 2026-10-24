#!/usr/bin/env bash
# EK100 group 실행 대기 런처 (vast 2026-10-21). 플래그 파일이 생기면 → VRAM 여유 GPU 6 장을 고르고 → 1 GPU 스모크 → 전체 실행.
#   tmux new-session -d -s ek100_g5 "FLAG=<플래그> bash cvpr/eval/ek100/launch_group_after_flag.sh vast_ek100_g5"
#   로그: z_training/logs/ek100_g5_launch.log (런처) · 결과 폴더의 stdout.log (실행)
#   NEED_GB: 장당 필요 여유 (기본 55 GB — EK100 group 실측 ~35–45 GB + 여유). 6 장이 안 되면 2 분마다 다시 본다
cd "$(dirname "$0")/../../.."
MODEL=${1:-vast_ek100_g5}; FLAG=${FLAG:?"FLAG=<플래그 파일>"}; NEED_GB=${NEED_GB:-55}
L=z_training/logs/ek100_g5_launch.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a $L; }
say "대기: 플래그 $FLAG"
until [[ -f $FLAG ]]; do sleep 60; done
pick() {   # 여유 VRAM 큰 순 6 장 (NEED_GB 이상만)
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v need=$NEED_GB '{f=($2-$3)/1024; if (f>=need) print f, $1}' | sort -rn | head -6 | awk '{print $2}' | sort -n | tr '\n' ' '
}
while :; do
  IDS=$(pick); n=$(wc -w <<<"$IDS")
  say "여유 ${NEED_GB} GB 이상 GPU: [$IDS] ($n 장) · $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
  (( n >= 6 )) && break
  sleep 120
done
FIRST=$(awk '{print $1}' <<<"$IDS")
say "스모크 (GPU $FIRST, 영상 4 · 1 epoch · batch 2)"
GPU_IDS="$FIRST" GPUS=1 BS=2 NW=2 VAL_EVERY=1 CKPT_EVERY=100 LIMIT=4 TAG=ek100_${MODEL}_smoke4 \
  SET="experiment.optimization.num_epochs=1" timeout 3600 bash cvpr/eval/ek100/run_group.sh $MODEL >> $L 2>&1
SM=$(ls -d cvpr/results/eval/ek100/action_anticipation_frozen/ek100_${MODEL}_smoke4* 2>/dev/null | tail -1)   # 2026-10-22: run_group.sh 가 TAG 에 _smoke4 를 한 번 더 붙인다
nf=$(ls $SM/val_metrics__*.jsonl 2>/dev/null | wc -l)
if (( nf == 0 )); then say "!! 스모크 실패 — 팔별 val_metrics 없음 ($SM/stdout.log 확인). 전체 실행 안 함"; exit 1; fi
say "스모크 통과: 팔 $nf 개 val_metrics"
IDS=$(pick); (( $(wc -w <<<"$IDS") >= 6 )) || { say "!! 스모크 뒤 GPU 여유 부족 [$IDS] — 2 분 뒤 다시"; sleep 120; IDS=$(pick); }
say "전체 실행: GPU [$IDS] · rank 16 x 6 = 96 · HEADS cap16b96 · 20 epoch"
GPU_IDS="$IDS" GPUS=6 bash cvpr/eval/ek100/run_group.sh $MODEL >> $L 2>&1
say "끝 (exit $?) — /venv/vjepa/bin/python cvpr/eval/ek100/collect_group.py"
