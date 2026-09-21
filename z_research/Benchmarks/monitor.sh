#!/usr/bin/env bash
# =============================================================================
#   watch -c -n 1 bash z_research/Benchmarks/monitor.sh
#
# 노드 로컬에서 벌어지는 일(스테이징·채점)을 **공유 디스크의 진행률 파일**로 본다.
#   exp_results/_stage/<host>.tsv   : stage.sh 의 샘플러가 5 초마다 갱신
#   exp_results/_runs/<tag>.tsv     : run 하나가 갱신 (done/total/elapsed)
# 속도·ETA 는 이전 샘플과의 차이로 낸다 (상태는 /tmp 에 둔다).
# =============================================================================
REPO=${REPO:-/data/hyuntak/project/2026/2027_cvpr/vjepa2}
BASE="$REPO/z_research/Benchmarks/exp_results"
PREV="/tmp/.bench_monitor_$(id -u).prev"
NOW=$(date +%s)

R='\033[0m'; B='\033[1m'; DIM='\033[2m'
GRN='\033[32m'; YEL='\033[33m'; RED='\033[31m'; CYN='\033[36m'; BLU='\033[34m'

hum() {  # bytes -> 사람이 읽는 단위
  awk -v b="${1:-0}" 'BEGIN{ s="B KB MB GB TB"; split(s,u," "); i=1;
    while (b>=1024 && i<5){ b/=1024; i++ } printf "%.1f%s", b, u[i] }'
}
hms() {  # 초 -> 1h02m / 3m20s / 45s
  awk -v t="${1:-0}" 'BEGIN{ t=int(t); if(t<0)t=0;
    if(t>=3600) printf "%dh%02dm", t/3600, (t%3600)/60;
    else if(t>=60) printf "%dm%02ds", t/60, t%60;
    else printf "%ds", t }'
}
bar() {  # pct width
  awk -v p="${1:-0}" -v w="${2:-24}" 'BEGIN{ n=int(p*w/100); if(n>w)n=w; if(n<0)n=0;
    for(i=0;i<n;i++) printf "█"; for(i=n;i<w;i++) printf "·" }'
}
prev_get() { grep -m1 "^$1 " "$PREV" 2>/dev/null | awk '{print $2" "$3}'; }

NEWPREV=$(mktemp)
printf "${B}${CYN}  Benchmarks  ${R}${DIM}%s${R}\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo

# ---------------------------------------------------------------- 스테이징 ----
shopt -s nullglob
stage_files=("$BASE"/_stage/*.tsv)
if (( ${#stage_files[@]} )); then
  printf "${B}데이터 스테이징${R} ${DIM}(tar -> 노드 로컬 /data2)${R}\n"
  for f in "${stage_files[@]}"; do
    host=$(awk -F'\t' '$1=="#host"{print $2}' "$f")
    ts=$(awk -F'\t' '$1=="#ts"{print $2}' "$f")
    age=$(( NOW - ${ts:-0} ))
    stale=""; (( age > 30 )) && stale="${DIM} (갱신 ${age}s 전)${R}"
    printf "  ${B}%s${R}%b\n" "${host:-?}" "$stale"
    while IFS=$'\t' read -r key st cur exp; do
      [[ "$key" == \#* || -z "$key" ]] && continue
      pct=$(awk -v c="${cur:-0}" -v e="${exp:-1}" 'BEGIN{ if(e<=0){print 0}else{p=100*c/e; if(p>100)p=100; printf "%.1f", p} }')
      read -r pb pt <<<"$(prev_get "stage:$host:$key")"
      rate=0; eta="-"
      if [[ -n "$pb" && -n "$pt" && $(( NOW - pt )) -ge 3 && "$cur" -gt "$pb" ]]; then
        rate=$(( (cur - pb) / (NOW - pt) ))
        (( rate > 0 )) && eta=$(hms $(( (exp - cur) / rate )))
      else
        read -r pb pt <<<"$(prev_get "stage:$host:$key")"
      fi
      # 샘플 보존: 값이 바뀌었을 때만 갱신해 rate 가 0 으로 죽지 않게
      if [[ -z "$pb" || "$cur" != "$pb" ]]; then echo "stage:$host:$key $cur $NOW" >> "$NEWPREV"
      else echo "stage:$host:$key ${pb} ${pt}" >> "$NEWPREV"; fi
      case "$st" in
        done)    c=$GRN; mark="✔";;
        running) c=$YEL; mark="▶";;
        failed)  c=$RED; mark="✘";;
        *)       c=$DIM; mark="·";;
      esac
      printf "    ${c}%s %-9s${R} [%b] ${B}%5s%%${R}  %8s / %-8s  %7s/s  ETA %s\n" \
        "$mark" "$key" "$(bar "$pct" 22)" "$pct" "$(hum "$cur")" "$(hum "$exp")" "$(hum "$rate")" "$eta"
    done < "$f"
  done
  echo
fi

# -------------------------------------------------------------------- 실행 ----
run_files=("$BASE"/_runs/*.tsv)
if (( ${#run_files[@]} )); then
  printf "${B}채점 진행${R} ${DIM}(샤드는 합쳐서 보여준다)${R}\n"
  printf "  ${DIM}%-26s %-10s %-22s %8s %9s %9s${R}\n" "run" "상태" "진행" "경과" "ETA" "속도"
  # <tag>.sN.tsv 를 <tag> 로 묶는다
  mapfile -t tags < <(for f in "${run_files[@]}"; do basename "$f" .tsv | sed 's/\.s[0-9]\+$//'; done | sort -u)
  for tag in "${tags[@]}"; do
    done_n=0; tot_n=0; t0=$NOW; n_done=0; n_file=0; st=running
    for f in "$BASE"/_runs/"$tag".tsv "$BASE"/_runs/"$tag".s*.tsv; do
      [[ -f "$f" ]] || continue
      n_file=$(( n_file + 1 ))
      d=$(awk -F'\t' '$1=="done"{print $2}'  "$f"); done_n=$(( done_n + ${d:-0} ))
      t=$(awk -F'\t' '$1=="total"{print $2}' "$f"); tot_n=$(( tot_n + ${t:-0} ))
      s=$(awk -F'\t' '$1=="start"{print $2}' "$f"); [[ -n "$s" && "$s" -lt "$t0" ]] && t0=$s
      [[ "$(awk -F'\t' '$1=="status"{print $2}' "$f")" == done ]] && n_done=$(( n_done + 1 ))
    done
    (( n_file > 0 && n_done == n_file )) && st=done
    el=$(( NOW - t0 ))
    pct=$(awk -v c="$done_n" -v e="$tot_n" 'BEGIN{ if(e<=0){print 0}else{printf "%.1f", 100*c/e} }')
    eta="-"; rps="-"
    if (( done_n > 0 && el > 0 )); then
      rps=$(awk -v d="$done_n" -v e="$el" 'BEGIN{printf "%.2f/s", d/e}')
      eta=$(hms $(awk -v d="$done_n" -v e="$el" -v t="$tot_n" 'BEGIN{ if(d<=0){print 0}else{printf "%d", (t-d)*e/d} }'))
    fi
    case "$st" in
      done)    c=$GRN;; running) c=$YEL;; failed) c=$RED;; *) c=$DIM;;
    esac
    printf "  %-26s ${c}%-10s${R} [%b] %5s%% %8s %9s %9s\n" \
      "$tag" "${st:-?}" "$(bar "$pct" 14)" "$pct" "$(hms $el)" "$eta" "$rps"
  done
  echo
fi

# -------------------------------------------------------------------- SLURM ---
printf "${B}SLURM${R}\n"
squeue -u "$(whoami)" -o "%.8i %.10P %.14j %.2t %.10M %.6D %R" 2>/dev/null | \
  awk -v g="$GRN" -v y="$YEL" -v r="$R" 'NR==1{print "  "$0; next}
    { c=($4=="R")? g : y; printf "  %s%s%s\n", c, $0, r }'

mv -f "$NEWPREV" "$PREV" 2>/dev/null || rm -f "$NEWPREV"
