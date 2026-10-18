#!/bin/bash
# 2026-10-13 AR 이 late permanence 를 푸는 이유 가르기: Ariel full (release 위상) · prefix 팔 = AR 과 같은 데이터 · 미래 전용 목표,
# 그러나 문맥은 context encoder 창 단위 + mask token (release 와 같은 입력 방식). VOBench 표준 채점 (release 와 같은 표적 공간).
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
for m in vith_ariel_full_ep40 vith_ariel_prefix_ep45; do
  echo "[$(date +%H:%M:%S)] $m"; GPUS=4 bash z_research/scripts/run.sh surprise_c16t32 vobench_v1 $m > auto_research/logs/vob_$m.log 2>&1; echo "[$m exit $?]"
done
echo WINDOW ARMS DONE
