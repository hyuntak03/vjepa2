#!/usr/bin/env bash
# flow 캐시 추출 진행 (z_vll7_training/extract_flow.sh) — shard 별 속도 · ETA · 용량.
OUT=${OUT:-/data/hyuntak/flow_cache}
printf '\033[1m flow 캐시 추출\033[0m  %s\n' "$OUT"
printf '%s\n' "──────────────────────────────────────────────────────────────────────"
tot_done=0; tot_all=0; rate=0
for f in "$OUT"/logs/*.log; do
  [ -e "$f" ] || continue
  nm=$(basename "$f" .log)
  ln=$(grep -a '편 ·' "$f" | tail -1)
  [ -z "$ln" ] && { printf '  %-26s (시작 전)\n' "$nm"; continue; }
  printf '  %-26s %s\n' "$nm" "$(echo "$ln" | sed 's/^ *//')"
done
printf '%s\n' "──────────────────────────────────────────────────────────────────────"
printf '  용량 %s   ·   GPU %s\n' "$(du -sh "$OUT" 2>/dev/null | cut -f1)" \
  "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' ')"
printf '  끝난 shard: %s / %s\n' "$(grep -lc ALLDONE "$OUT"/logs/*.log 2>/dev/null | wc -l)" "$(ls "$OUT"/logs/*.log 2>/dev/null | wc -l)"
printf '  남은 디스크 (/data) %s\n' "$(df -h /data | tail -1 | awk '{print $4}')"
