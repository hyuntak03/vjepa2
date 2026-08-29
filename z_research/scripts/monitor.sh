#!/bin/bash
# -----------------------------------------------------------------------------
# WORLD MODEL ANALYSIS — 실행 모니터
#
#   watch -n 1 bash z_research/scripts/monitor.sh              # 최신 slurm job
#   watch -n 1 bash z_research/scripts/monitor.sh 189500       # job id 지정
#   watch -n 1 bash z_research/scripts/monitor.sh /tmp/run.log # bash 실행 로그
#
#   watch -c -n 1 'MONITOR_COLOR=1 bash z_research/scripts/monitor.sh'   # 컬러
#
# 인자가 없으면 z_research/scripts/slurm_logs 에서 가장 최근 .err 를 잡고,
# 없으면 구 경로 z_scripts/slurm_logs 로 폴백한다.
# wma 로거는 stderr 로 나가므로 slurm 은 .err 를 본다.
#
# z_scripts/world_model_analysis/monitor.sh 에서 옮겨 왔다 (z_scripts 는 .gitignore).
# 바뀐 것: 로그 경로 2곳 탐색 + probing 을 "순차 head" 모드로 읽는다
#          (현행 eval.py 는 head 를 rank 로 쪼개지 않고 45개를 8 rank DP 로 순차 학습한다)
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
LOGDIRS=("$PROJ/z_research/scripts/slurm_logs" "$PROJ/z_scripts/slurm_logs")
ARG="${1:-}"

JOBID=""; LOG=""
if [[ -z "$ARG" ]]; then
  LOG=$(ls -t "${LOGDIRS[0]}"/*.err "${LOGDIRS[1]}"/*.err 2>/dev/null | head -1)
  [[ -n "$LOG" ]] && JOBID=$(basename "$LOG" .err | sed 's/.*_//')
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
  JOBID="$ARG"
  LOG=$(ls -t "${LOGDIRS[0]}"/*_"$ARG".err "${LOGDIRS[1]}"/*_"$ARG".err 2>/dev/null | head -1)
else
  LOG="$ARG"
fi

# ── slurm 상태 ────────────────────────────────────────────────────────────────
STATE=""; NODE=""; ELAPSED=""; LIMIT=""; JOBNAME=""
if [[ -n "$JOBID" ]] && command -v squeue >/dev/null 2>&1; then
  Q=$(squeue -j "$JOBID" -h -o "%T|%N|%M|%l|%j" 2>/dev/null)
  if [[ -n "$Q" ]]; then
    IFS='|' read -r STATE NODE ELAPSED LIMIT JOBNAME <<<"$Q"
  else
    STATE=$(sacct -j "$JOBID" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    [[ -z "$STATE" ]] && STATE="큐에 없음"
  fi
fi

# ── GPU ───────────────────────────────────────────────────────────────────────
# job 이 도는 노드에 내가 있을 때만 의미가 있다. master 에서 보면 남의 GPU 다.
GPU=""
HERE=$(hostname -s 2>/dev/null)
if [[ -n "$NODE" && "$NODE" != "$HERE" ]]; then
  GPU="__다른노드__"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
fi

exec /usr/bin/env python3 - "$LOG" "$JOBID" "$STATE" "$NODE" "$ELAPSED" "$LIMIT" "$JOBNAME" "$GPU" <<'PYEOF'
import os, re, sys, time

log, jobid, state, node, elapsed, limit, jobname, gpu = (sys.argv[1:9] + [""] * 8)[:8]
W = int(os.environ.get("MONITOR_WIDTH", 100))
BAR = 20

# 기본은 무채색 — `watch -n 1 bash monitor.sh` 로 그냥 봐도 깨지지 않게.
# 컬러가 필요하면 `watch -c -n 1 'MONITOR_COLOR=1 bash monitor.sh'`.
_C = os.environ.get("MONITOR_COLOR", "0") != "0"
def c(s, code):
    return f"\033[{code}m{s}\033[0m" if _C else s
BOLD = lambda s: c(s, "1")
DIM  = lambda s: c(s, "2")
RED  = lambda s: c(s, "31")
GRN  = lambda s: c(s, "32")
YEL  = lambda s: c(s, "33")

def bar(cur, tot, w=BAR):
    if not tot:
        return DIM("." * w)
    f = max(0, min(w, round(w * cur / tot)))
    return GRN("#" * f) + DIM("." * (w - f))

def hms(sec):
    sec = int(float(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def cut(s, w=None):
    w = w or (W - 2)
    return s if len(s) <= w else s[: w - 1] + "…"

def line(ch="-"):
    print(ch * W)

if not log or not os.path.exists(log):
    line("="); print(" 로그를 찾을 수 없다")
    print(f" 찾은 경로: {log or '(없음)'}")
    print(" 사용법: monitor.sh [<job id> | <로그 경로>]"); line("="); sys.exit(0)

# 진행바가 \r 로 겹쳐 찍히므로 줄로 펴서 읽는다
txt = open(log, "r", errors="replace").read().replace("\r", "\n")
age = time.time() - os.path.getmtime(log)

def last(pat, s=txt):
    m = re.findall(pat, s)
    return m[-1] if m else None

# ── 헤더 ──────────────────────────────────────────────────────────────────────
ws   = last(r"world_size=(\d+)")
data = last(r"\[data\] (.+)")
tag  = last(r"\[wma\] (\S+) \| world_size")
done = "[wma] done ->" in txt
err  = re.findall(r"^(?:Traceback|ERROR:|RuntimeError:|\S*Error:.*)$", txt, re.M)

line("=")
head = f" {jobname or tag or '?'}"
if jobid:   head += f" / {jobid}"
if state:   head += f"   {state}"
if node:    head += f" @ {node}"
if elapsed: head += f"   경과 {elapsed}" + (f" / {limit}" if limit else "")
print(BOLD(cut(head)))
print(cut(f" log  {log}  ({int(age)}s 전 갱신)"))
if ws:   print(f" world_size {ws}")
if data: print(cut(f" data {data.strip()}"))
if gpu == "__다른노드__":
    print(DIM(f" GPU  (이 셸은 {os.uname().nodename.split('.')[0]} — job 은 {node} 에서 돈다)"))
elif gpu:
    # nvidia-smi 출력 형식은 노드마다 다를 수 있다. 못 읽으면 조용히 넘어간다.
    used, mem = [], []
    for cell in gpu.strip(";").split(";"):
        f = [x.strip() for x in cell.split(",")]
        if len(f) < 3 or not f[1].isdigit() or not f[2].isdigit():
            continue
        mem.append(f"{int(f[1])//1024}G")
        used.append(f"{f[2]}%")
    if used:
        print(cut(f" GPU({node or 'local'}) util {' '.join(used)}   mem {' '.join(mem)}"))
line("=")

nranks = int(ws) if ws and ws.isdigit() else 8

# ── surprise ─────────────────────────────────────────────────────────────────
sur_start = dict(re.findall(r"\[surprise\] rank(\d+) 시작 (\d+)개", txt))
combos = re.findall(r"intphys1 (\S+): ctx=(\[[^\]]*\]) stride=(\d+) -> window (\d+)개/video", txt)
if sur_start:
    h = "\n " + BOLD("[surprise]")
    if combos:
        nf = last(r"총 (\d+) forward/video")
        h += DIM(f"  sliding — 조합 {len(combos)}개" + (f", {nf} forward/video" if nf else ""))
    print(h)
    for r in range(nranks):
        tot = int(sur_start.get(str(r), 0))
        fin = last(rf"\[surprise\] rank{r} 끝 (\d+)개 ([\d.]+)s")
        if fin:
            print(f"   rank{r} {bar(1,1)} {fin[0]}/{fin[0]}  " + GRN(f"완료 {fin[1]}s"))
        else:
            p = last(rf"\[surprise\] rank{r} (\d+)/(\d+) \(([\d.]+)s")
            cur = int(p[0]) if p else 0
            eta = ""
            if p and cur:
                el = float(p[2]); eta = f"  {hms(el)} 경과, 남은 ~{hms(el/cur*(tot-cur))}"
            print(f"   rank{r} {bar(cur,tot)} {cur}/{tot}{eta}")
    rows = re.findall(r"\[surprise/\s*(\S+?)\s*\] block_pairwise=([\d.]+) "
                      r"\(block (\d+), pair (\d+)\) \| 전부정답=([\d.]+)", txt)
    if not rows:
        bp = last(r"block_pairwise=([\d.]+) \(chance [\d.]+, block (\d+), pair (\d+)\)")
        if bp:
            print("   -> " + BOLD(f"block_pairwise={bp[0]}") + f" (block {bp[1]}, pair {bp[2]})")
    else:
        print("\n   " + BOLD(f"{'조합/집계':22s} {'pairwise':>9s} {'전부정답':>9s} {'block':>6s}"))
        best = max(float(x[1]) for x in rows)
        for k, v, nb, npair, pf in rows:
            row = f"   {k:22s} {float(v):9.4f} {float(pf):9.4f} {nb:>6s}"
            print(GRN(row) if float(v) == best else row)

# ── feature 추출 ─────────────────────────────────────────────────────────────
bases = last(r"\[feat\] bases (\{.+?\}) -> ([\d.]+) GiB")
if last(r"(\[feat\] cache hit)"):
    print("\n " + BOLD("[추출]") + "  캐시 히트 — 건너뜀"
          + (DIM(f"  ({bases[1]} GiB)") if bases else ""))
elif re.search(r"\[feat\] rank\d+", txt):
    print("\n " + BOLD("[추출]") + (f"  캐시 {bases[1]} GiB" if bases else ""))
    for r in range(nranks):
        fin = last(rf"\[feat\] rank{r} 끝 (\d+)개 ([\d.]+)s = decode ([\d.]+)s \+ fwd ([\d.]+)s \+ write ([\d.]+)s")
        if fin:
            print(f"   rank{r} {bar(1,1)} {fin[0]}/{fin[0]}  완료 {fin[1]}s "
                  f"(decode {fin[2]} / fwd {fin[3]} / write {fin[4]})")
        else:
            p = last(rf"\[feat\] rank{r} (\d+)/(\d+) \(decode ([\d.]+)s / fwd ([\d.]+)s / write ([\d.]+)s\)")
            if p:
                cur, tot = int(p[0]), int(p[1])
                eta = f"  남은 ~{hms((tot-cur)/cur*float(p[2] or 0))}" if cur else ""
                print(f"   rank{r} {bar(cur,tot)} {cur}/{tot}  "
                      f"(decode {p[2]} / fwd {p[3]} / write {p[4]})")
            else:
                print(f"   rank{r} {bar(0,1)} 대기")

# ── probing ──────────────────────────────────────────────────────────────────
plan = last(r"\[probe\] head (\d+)개를 (\d+) rank data-parallel[^\n]*")
if re.search(r"\[probe\]", txt):
    print("\n " + BOLD("[probing]") + (DIM("  " + plan[0] + " head / " + plan[1] + " rank DP 순차") if plan else ""))

    fin = re.findall(r"\[probe\] rank0 (.+?) train_acc=([\d.]+) \| fit ([\d.]+)s eval ([\d.]+)s", txt)
    st  = last(r"\[probe\] rank0 (.+?) 학습 시작 n_train=(\d+) epochs=(\d+) \[(\d+)/(\d+)\]")

    ndone = len(fin)
    ktot  = int(st[4]) if st else (int(plan[0]) if plan else 0)

    # 현재 head
    if st:
        name, ntr, nep, i, k = st
        ep = last(rf"\[probe\] rank0 {re.escape(name)} ep (\d+)/(\d+) loss=([\d.]+) acc=([\d.]+) \| ([\d.]+)s 경과, 남은 예상 ([\d.]+)s")
        short = name.replace(", groups=", " ").replace("None", "전체").replace("'", "")
        cur_left = 0.0
        if ep:
            cur_left = float(ep[5])
            print(f"   [{i:>2}/{k}] {cut(short, 34):34s} {bar(int(ep[0]),int(ep[1]))} "
                  f"ep {ep[0]:>2}/{ep[1]}  loss {float(ep[2]):.4f} acc {float(ep[3]):.4f}"
                  f"  {hms(ep[4])} 경과, 남은 ~{hms(ep[5])}")
        else:
            print(f"   [{i:>2}/{k}] {cut(short, 34):34s} {bar(0,int(nep))} "
                  f"ep  0/{nep}  n_train={ntr} 시작")

        # 전체 ETA — 완료 head 평균 x 남은 개수 + 현재 head 잔여
        if fin and ktot:
            times = [float(f[2]) + float(f[3]) for f in fin]
            mean = sum(times) / len(times)
            rest = max(0, ktot - int(i))
            print(DIM(f"   head {ndone}/{ktot} 완료 | head당 평균 {hms(mean)} "
                      f"(최소 {hms(min(times))} 최대 {hms(max(times))})"))
            print("   " + BOLD(f"전체 남은 예상 ~{hms(cur_left + rest * mean)}")
                  + DIM(f"  (남은 head {rest}개 + 현재 head)"))

    # 최근 완료 head
    if fin:
        print(DIM(f"\n   최근 완료 (총 {ndone}개)"))
        for name, acc, ft, ev in fin[-5:]:
            short = name.replace(", groups=", " ").replace("None", "전체").replace("'", "")
            print(f"     {cut(short, 46):46s} train_acc {float(acc):.4f}  {hms(float(ft)+float(ev))}")

# ── 마무리 ───────────────────────────────────────────────────────────────────
print()
line()
tail = [l for l in txt.splitlines() if l.startswith("[wma]")][-3:]
for l in tail:
    print(cut(" " + l.replace("[wma] ", "", 1)))
if err:
    print()
    print(RED(" [!] 에러 감지 — 로그를 직접 확인할 것"))
    for l in err[-3:]:
        print(cut("   " + l.strip()))
if done:
    out = last(r"\[wma\] done -> (\S+)")
    print()
    print(GRN(f" [완료] -> {out}"))
line("=")
PYEOF
