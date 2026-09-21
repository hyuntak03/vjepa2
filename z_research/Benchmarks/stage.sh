#!/usr/bin/env bash
# =============================================================================
# 벤치마크 데이터를 **그 노드의 로컬 디스크**로 푼다.
#
#   bash   z_research/Benchmarks/stage.sh          # 지금 있는 노드에서
#   sbatch z_research/Benchmarks/stage.sbatch      # vll3 로 제출
#
# 규칙 (사용자 지시 2026-09-20):
#   - tar 로 있는 것   -> /data2/local_datasets/world/Benchmark 에 푼다
#   - 폴더로만 있는 것 -> /data/dataset/world/Benchmarks 에서 그냥 읽는다 (InfLevel)
#
# 여러 번 돌려도 안전하다 (`.state/<key>.done` 으로 건너뛴다).
# 진행률은 **공유 디스크**(`exp_results/_stage/<host>.tsv`)에 5 초마다 적는다.
# 노드 로컬이라 다른 노드에서는 안 보이기 때문이다 — monitor.sh 가 그 파일을 읽는다.
# =============================================================================
set -uo pipefail

REPO=${REPO:-/data/hyuntak/project/2026/2027_cvpr/vjepa2}
SRC=${BENCH_SRC:-/data/dataset/world/Benchmarks}
LOCAL=${BENCH_LOCAL:-/data2/local_datasets/world/Benchmark}
STATE="$LOCAL/.state"
SHARED="$REPO/z_research/Benchmarks/exp_results/_stage"
HOST=$(hostname)
PROG="$SHARED/$HOST.tsv"

mkdir -p "$LOCAL" "$STATE" "$SHARED" || { echo "FATAL: $LOCAL 를 못 만든다"; exit 1; }

# key  목적지  예상바이트   (2026-09-20 에 /data 의 풀린 사본에서 실측)
KEYS=(grasp intphys1 intphys2)
declare -A DEST=( [grasp]="GRASP" [intphys1]="IntPhys1_dev" [intphys2]="IntPhys2" )
declare -A EXP=(  [grasp]=7745911872 [intphys1]=3253738518 [intphys2]=1238462417 )

DECOMP=$(command -v pigz >/dev/null && echo "pigz -dc" || echo "gzip -dc")
log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 진행률 샘플러: 공유 디스크에 5 초마다 -------------------------------------
sampler() {
  while :; do
    {
      echo -e "#host\t$HOST"
      echo -e "#local\t$LOCAL"
      echo -e "#ts\t$(date +%s)"
      for k in "${KEYS[@]}"; do
        d="$LOCAL/${DEST[$k]}"
        b=$( [[ -d "$d" ]] && du -sb "$d" 2>/dev/null | cut -f1 || echo 0 )
        if   [[ -f "$STATE/$k.done"    ]]; then s=done
        elif [[ -f "$STATE/$k.failed"  ]]; then s=failed
        elif [[ -f "$STATE/$k.running" ]]; then s=running
        else s=pending; fi
        echo -e "$k\t$s\t${b:-0}\t${EXP[$k]}"
      done
    } > "$PROG.tmp" && mv "$PROG.tmp" "$PROG"
    sleep 5
  done
}

stage_one() {
  local key=$1; shift
  local log_f="$STATE/$key.log"
  if [[ -f "$STATE/$key.done" ]]; then log "$key: 이미 완료 — 건너뜀"; return 0; fi
  rm -f "$STATE/$key.failed"; : > "$STATE/$key.running"
  log "$key: 시작"
  if "$@" >>"$log_f" 2>&1; then
    rm -f "$STATE/$key.running"; : > "$STATE/$key.done"; log "$key: 완료"
  else
    rm -f "$STATE/$key.running"; : > "$STATE/$key.failed"
    log "$key: 실패 — $log_f"; return 1
  fi
}

do_grasp()    { tar -xf "$SRC/GRASP.tar" -C "$LOCAL"; }                    # -> GRASP/videos/{level1,level2}
do_intphys2() { tar -xf "$SRC/IntPhys2.tar" -C "$LOCAL" IntPhys2/Main; }   # Main 만 (사용자 지시)
do_intphys1() {                                                            # scene 프레임만 (depth/masks 제외)
  mkdir -p "$LOCAL/IntPhys1_dev"
  $DECOMP "$SRC/IntPhys1_dev.tar.gz" | tar -x -C "$LOCAL/IntPhys1_dev" --wildcards 'dev/*/*/*/scene/*'
}

log "호스트 $HOST | SRC=$SRC | LOCAL=$LOCAL"
df -h "$LOCAL" | tail -1
sampler & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

stage_one grasp    do_grasp    & P1=$!
stage_one intphys1 do_intphys1 & P2=$!
stage_one intphys2 do_intphys2 & P3=$!
wait $P1; r1=$?
wait $P2; r2=$?
wait $P3; r3=$?

sleep 6                      # 샘플러가 마지막 상태를 한 번 더 쓰게
kill $SAMPLER 2>/dev/null
log "----- 결과 -----"
for k in "${KEYS[@]}"; do
  s=$( [[ -f "$STATE/$k.done" ]] && echo done || ([[ -f "$STATE/$k.failed" ]] && echo FAILED || echo "?") )
  printf "  %-9s %-7s %s\n" "$k" "$s" "$(du -sh "$LOCAL/${DEST[$k]}" 2>/dev/null | cut -f1)"
done
log "InfLevel 은 tar 가 없다 -> $SRC/InfLevel 에서 바로 읽는다 (1.1 GB)"
exit $(( r1 | r2 | r3 ))
