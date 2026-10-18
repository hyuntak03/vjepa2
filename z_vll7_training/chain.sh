#!/usr/bin/env bash
# =============================================================================
# GPU 가 빌 때까지 기다렸다가 다음 학습을 올린다 (2026-10-17)
#
#   tmux new-session -d -s chain_vll7 "GPUS=6 bash z_vll7_training/chain.sh ctxt_video"
#   tmux kill-session -t chain_vll7      # 취소 (기다리는 중이면 학습은 안 올라간다)
#
# ⚠️ **판정은 GPU 메모리 하나로 한다.** 처음엔 "앞 tmux 세션이 살아 있나" 로 했다가 두 번 틀렸다 —
#    (1) 세션은 작업이 끝나도 `exec bash` 로 남는다 → 영원히 대기
#    (2) 프로세스 트리가 깊어 (pane → bash → tee → python → rank) 자식 검사가 도는 작업을 놓친다
#    GPU 가 비었나는 애매할 일이 없고, 어떤 방식으로 띄운 작업이든 잡는다.
# ⚠️ **실행 중인 이 파일을 고치지 말 것.** bash 는 스크립트를 조금씩 읽어서, 고치면 돌던 chain 이
#    엉뚱한 줄에서 syntax error 로 죽는다 (2026-10-17 실제로 그랬다).
#
# FREE_MIB  이 값 미만이면 "비었다" (기본 2000)   STABLE  연속 몇 번 비어야 하나 (기본 3, 1 분 간격)
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh" >/dev/null
CONFIG=${1:?"사용법: GPUS=6 bash z_vll7_training/chain.sh <config> [NAME]"}
NAME=${2:-${NAME:-$CONFIG}}
FREE_MIB=${FREE_MIB:-2000}
STABLE=${STABLE:-3}

echo "[$(date '+%H:%M:%S')] GPU 가 비길 기다린다 (피크 < ${FREE_MIB} MiB 가 ${STABLE} 번 연속) -> 그 뒤 $CONFIG"
ok=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
  used=${used:-0}
  if [[ "$used" -lt "$FREE_MIB" ]]; then
    ok=$((ok + 1))
    echo "[$(date '+%H:%M:%S')] 비어 있음 (피크 ${used} MiB) ${ok}/${STABLE}"
    [[ "$ok" -ge "$STABLE" ]] && break
  else
    [[ "$ok" -gt 0 ]] && echo "[$(date '+%H:%M:%S')] 다시 사용 중 (피크 ${used} MiB) — 카운터 초기화"
    ok=0
  fi
  sleep 60
done
echo "[$(date '+%H:%M:%S')] 학습 시작: $CONFIG -> $NAME"
GPUS=${GPUS:-6} NAME="$NAME" exec bash "$HERE/train.sh" "$CONFIG"
