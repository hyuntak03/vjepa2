#!/usr/bin/env bash
# =============================================================================
# EK100 action anticipation (4 모델 chain) 모니터 — 2026-10-17
#
#   watch -c -n 1 bash z_vll7_training/monitor_ek100.sh          # 기본 세션 ek100_4models
#   SESSION=ek100_vll7 watch -c -n 1 bash z_vll7_training/monitor_ek100.sh
#   MONITOR_COLOR=0 bash z_vll7_training/monitor_ek100.sh        # 색 끄기
#
# ⚠️ **host RAM 을 맨 위에 둔다.** 2026-10-17 에 BS=85 로 올렸다가 job 캡 300 GB 를 넘겨 rank 1 이 SIGKILL 되고
#    나머지 5 rank 가 ALLREDUCE 에서 2 시간 멈췄다 (watchdog abort). 터지는 건 VRAM 이 아니라 호스트 RAM 이다.
#
# 읽는 것: tmux 세션 (지금 모델 · iter) · cgroup 메모리 · nvidia-smi ·
#          cvpr/results/eval/ek100/action_anticipation_frozen/<tag>/{log_r0.csv, stdout.log, val_metrics.jsonl}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh" >/dev/null
SESSION=${SESSION:-$(tmux ls -F "#{session_name}" 2>/dev/null | grep "^ek100" | head -1)}
SESSION=${SESSION:-ek100_4models}
RES="$VJEPA2_ROOT/cvpr/results/eval/ek100/action_anticipation_frozen"

PANE=$(tmux capture-pane -p -J -t "$SESSION" -S -4000 2>/dev/null)   # -J: 줄바꿈된 로그를 이어 붙인다
ALIVE=$(tmux has-session -t "=$SESSION" 2>/dev/null && echo 1 || echo 0)
B=/sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID:-0}
# ⚠️ memory.usage_in_bytes 는 **page cache 를 포함**해 실제 압박을 과장한다 (영상 디코드로 cache 가 100 GB 쌓인다).
#    OOM 을 부르는 건 total_rss 다 — cache 는 커널이 먼저 회수한다.
MEM_CUR=$(awk '/^total_rss /{print $2}' "$B/memory.stat" 2>/dev/null || echo 0)
MEM_CACHE=$(awk '/^total_cache /{print $2}' "$B/memory.stat" 2>/dev/null || echo 0)
MEM_MAX=$(cat "$B/memory.max_usage_in_bytes" 2>/dev/null || echo 0)
MEM_LIM=$(cat "$B/memory.limit_in_bytes" 2>/dev/null || echo 0)
MEM_FAIL=$(cat "$B/memory.failcnt" 2>/dev/null || echo 0)
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)

printf '%s' "$PANE" > /tmp/_ek100_pane.$$
trap 'rm -f /tmp/_ek100_pane.$$' EXIT

"$VJEPA2_PY" - "/tmp/_ek100_pane.$$" "$RES" "$ALIVE" "$MEM_CUR" "$MEM_MAX" "$MEM_LIM" "$MEM_FAIL" "$GPU" "$NPROC" "$MEM_CACHE" <<'PY'
import json, os, re, sys, glob
from datetime import datetime

pane_f, RES, ALIVE, CUR, MAXU, LIM, FAIL, GPU, NPROC, CACHE = sys.argv[1:11]
ALIVE, NPROC = int(ALIVE), int(NPROC)
CUR, MAXU, LIM, FAIL, CACHE = (int(x) for x in (CUR, MAXU, LIM, FAIL, CACHE))
pane = open(pane_f, encoding="utf-8", errors="ignore").read()
COL = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, k): return f"\033[{k}m{s}\033[0m" if COL else str(s)
B, DIM, GRN, YEL, RED, CYA, MAG = "1", "2", "32", "33", "31", "36", "35"
W = 88
def line(ch="─"): print(c(ch * W, DIM))
def bar(fr, w=34, col=GRN):
    fr = max(0.0, min(1.0, fr)); n = int(round(fr * w))
    return c("█" * n, col) + c("░" * (w - n), DIM)
GB = 1024 ** 3

MODELS = ["vith", "vith_release_vobftg", "vith_vast_ar_state_ep70", "vith_vast_ar_state_ep70_vobftg"]
LABEL = {"vith": "release", "vith_release_vobftg": "release + pft(vob+grav)",
         "vith_vast_ar_state_ep70": "ar (vast ep70)", "vith_vast_ar_state_ep70_vobftg": "ar(vast) + pft(vob+grav)"}

# ── 헤더 ────────────────────────────────────────────────────────────────────
print(c("━" * W, DIM))
started = [m.group(1) for m in re.finditer(r"### ek100\s+ek100 / (\S+)", pane)]
cur = started[-1] if started else None
done_ok = set(re.findall(r"### ek100\s+ek100 / (\S+)", pane)[:-1]) if len(started) > 1 else set()
failed = set(re.findall(r"!! 실패 \S+/(\S+)", pane))
all_done = "ALL_DONE" in pane or "EK100_DONE" in pane
st = c("● 돌는 중", GRN) + c(f" · rank {NPROC}", DIM) if (ALIVE and NPROC) else (
     c("✓ 전부 끝", CYA) if all_done else (c("◐ 세션은 있으나 rank 없음", YEL) if ALIVE else c("○ 세션 없음", RED)))
print(f" {c('EK100 action anticipation', B+';'+CYA)}  ·  4 모델 chain   {st}")

# ── ⚠️ host RAM (이것 때문에 죽었다) ────────────────────────────────────────
line()
if LIM > 0:
    fr = CUR / LIM
    col = GRN if fr < 0.5 else (YEL if fr < 0.75 else RED)
    print(f" {c('host RSS', B):>9}  {bar(fr, 34, col)} {c(f'{CUR/GB:5.0f}', B)} / {LIM/GB:.0f} GB"
          f"  ({100*fr:.0f} %)   {c(f'+ page cache {CACHE/GB:.0f} GB (회수 가능)', DIM)}")
    if MAXU >= LIM * 0.98:
        print(c(f"           ⚠️ 이 job 은 과거에 캡에 닿은 적이 있다 (누적 최대 {MAXU/GB:.0f} GB · failcnt {FAIL:,}) — "
                "2026-10-17 BS=85 사고. 지금 값이 아니라 job 시작 이후 누적이다", DIM))
    elif fr > 0.75:
        print(c("           ⚠️ RSS 가 캡의 75 % 를 넘었다 — BS·NW 를 줄일 것", RED))
else:
    print(f" {c('host RAM', B):>9}  (cgroup 을 못 읽는다)")

# ── 모델 진행 ───────────────────────────────────────────────────────────────
line()
def run_dir(m):
    """tag 는 SUFFIX 로 바뀔 수 있다 (예: ek100_vith_b252). 가장 최근에 손댄 것을 고른다."""
    cands = sorted(glob.glob(os.path.join(RES, f"ek100_{m}")) + glob.glob(os.path.join(RES, f"ek100_{m}_*")),
                   key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return cands[0] if cands else os.path.join(RES, f"ek100_{m}")

for m in MODELS:
    d = run_dir(m)
    tag = os.path.basename(d)
    vm = os.path.join(d, "val_metrics.jsonl")
    rows = []
    if os.path.isfile(vm):
        for ln in open(vm, encoding="utf-8"):
            try: rows.append(json.loads(ln))
            except Exception: pass
    if m in failed:            mark, note = c("✗", RED), c("실패", RED)
    elif rows and rows[-1].get("epoch", 0) >= 20: mark, note = c("✓", GRN), ""
    elif m == cur and ALIVE:   mark, note = c("▶", YEL), c("진행 중", YEL)
    elif m in done_ok:         mark, note = c("·", DIM), c("끝(지표 없음)", DIM)
    else:                      mark, note = c("·", DIM), c("대기", DIM)
    best = ""
    if rows:
        bi = max(rows, key=lambda r: r.get("action", {}).get("recall", -1))
        last = rows[-1]
        best = (f"  e{last['epoch']:>2}/20  "
                f"action {c(f'{last['action']['recall']:5.2f}', B)}  verb {last['verb']['recall']:5.2f}  noun {last['noun']['recall']:5.2f}"
                f"   {c(f'최고 e{bi['epoch']} {bi['action']['recall']:.2f}', GRN)}")
    print(f"  {mark} {c(LABEL[m], B):<28} {c(tag, DIM):<40}{best}  {note}")

# ── 지금 모델의 iteration ───────────────────────────────────────────────────
line()
src = pane
if cur:
    so = os.path.join(run_dir(cur), "stdout.log")
    if os.path.isfile(so):
        src = open(so, encoding="utf-8", errors="ignore").read()[-400000:]   # 줄바꿈 없는 원본
its = re.findall(r"\[\s*(\d+)\] acc \(v/n\): ([\d.]+)% \(([\d.]+)% ([\d.]+)%\) recall \(v/n\): ([\d.]+)% .*?\[mem: ([\d.e+]+)\] \[data: ([\d.]+) ms\]", src)
ep = re.findall(r"Epoch (\d+)", src)
if its:
    i, a, av, an, r, mem, dt = its[-1]
    print(f" {c('지금', B):>9}  {c(LABEL.get(cur, cur or '?'), B+';'+MAG)}"
          + (f"   epoch {ep[-1]}/20" if ep else "")
          + f"   iter {c(i, B)}   acc {a}% (v {av} n {an})   recall {r}%"
          f"   {c(f'GPU mem {float(mem)/1024:.1f} GB', DIM)}  {c(f'data {float(dt):.0f} ms', DIM)}")
    if len(its) >= 2:
        print(f" {c('iter 추이', B):>9}  " + "  ".join(f"[{x[0]}] {x[1]}%" for x in its[-6:]))
else:
    tail = [l for l in pane.strip().splitlines() if l.strip()][-1:] if pane.strip() else []
    print(f" {c('지금', B):>9}  (아직 iteration 로그 없음)  {c(tail[0][:70] if tail else '', DIM)}")

# ── ETA ─────────────────────────────────────────────────────────────────────
# iters/epoch = train 행 수 // global batch. s/iter 는 stdout.log 의 연속 iteration 타임스탬프 중앙값.
# epoch 당 시간은 **실측 epoch 경계 간격**을 쓰고 (val 포함), 아직 한 epoch 도 안 끝났으면 train x 1.4 로 추정한다.
if cur:
    import statistics as stv
    d_cur = run_dir(cur)
    ipe = gb = spi = None
    nep = 20
    try:
        import yaml
        e = yaml.safe_load(open(os.path.join(d_cur, "_resolved.yaml"), encoding="utf-8"))["experiment"]
        gb = int(e["optimization"]["batch_size"]) * max(1, len([x for x in GPU.split(";") if x.strip()]))
        nep = int(e["optimization"]["num_epochs"])
        n = sum(1 for _ in open(e["data"]["dataset_train"], encoding="utf-8")) - 1
        ipe = n // gb
    except Exception:
        pass
    ts = re.findall(r"(\d\d:\d\d:\d\d)\]\[[^\]]*\]\[[^\]]*\]\s*\[\s*(\d+)\] acc", src)
    sec = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    rates = [(sec(ts[k][0]) - sec(ts[k-1][0])) / (int(ts[k][1]) - int(ts[k-1][1]))
             for k in range(1, len(ts)) if int(ts[k][1]) > int(ts[k-1][1])]
    if rates:
        spi = stv.median(rates[-12:])
    # 완료 epoch 수 · 실측 epoch 주기
    vm = os.path.join(d_cur, "val_metrics.jsonl")
    vrows = []
    if os.path.isfile(vm):
        for ln in open(vm, encoding="utf-8"):
            try: vrows.append(json.loads(ln))
            except Exception: pass
    done_ep = len(vrows)
    ep_starts = [sec(t) for t in re.findall(r"(\d\d:\d\d:\d\d)\][^\n]*\[\s*0\] acc", src)]
    per_ep = None
    if len(ep_starts) >= 2:
        gaps = [ep_starts[k] - ep_starts[k-1] for k in range(1, len(ep_starts))]
        per_ep = stv.median(gaps)            # 실측 (train + val)
    elif ipe and spi:
        per_ep = ipe * spi * 1.4             # val 을 train 의 40 % 로 가정
    if per_ep and ipe and spi:
        cur_it = int(its[-1][0]) if its else 0
        left_this = max(0.0, (nep - done_ep) * per_ep - cur_it * spi)
        idx = MODELS.index(cur) if cur in MODELS else 0
        left_all = left_this + (len(MODELS) - idx - 1) * (nep * per_ep + 240)   # 모델당 기동 ~4 분
        def hm(x):
            h, m = int(x // 3600), int(x % 3600 // 60)
            return f"{h}h {m:02d}m" if h else f"{m}m"
        fin = datetime.now().timestamp() + left_all
        src_tag = "실측" if len(ep_starts) >= 2 else "추정"
        line()
        print(f" {c('ETA', B):>9}  이 모델 {c(hm(left_this), B + ';' + YEL)} 남음  ·  전체 4 모델 {c(hm(left_all), B + ';' + YEL)} 남음"
              f"   끝 ≈ {c(datetime.fromtimestamp(fin).strftime('%m-%d(%a) %H:%M'), B + ';' + YEL)}")
        print(f"           {c(f'{spi:.2f} s/iter · {ipe} iter/epoch (train {ipe * gb:,} clip / global {gb}) · '
                             f'epoch 당 {per_ep / 60:.1f} 분 ({src_tag}, val 포함) · {done_ep}/{nep} epoch 완료', DIM)}")

# ── GPU ─────────────────────────────────────────────────────────────────────
gs = []
for g in [x for x in GPU.split(";") if x.strip()]:
    try:
        i, u, used, tot = [t.strip() for t in g.split(",")]
        gs.append((int(i), int(u), int(used), int(tot)))
    except Exception: pass
if gs:
    line()
    avg = sum(u for _, u, _, _ in gs) / len(gs)
    pk = max(mm for _, _, mm, _ in gs) / 1024
    print(f" {c('GPU', B):>9}  {len(gs)} 장 · 평균 util {c(f'{avg:.0f}%', B)} · 피크 {c(f'{pk:.1f} GB', B)} / {gs[0][3]/1024:.0f} GB")
    for k in range(0, len(gs), 3):
        print("           " + "   ".join(
            f"{c(f'g{i}', B)} {bar(u/100, 8, GRN if u >= 70 else (YEL if u >= 25 else RED))}{u:3d}% {used/1024:4.1f}G"
            for i, u, used, _ in gs[k:k+3]))

# ── 오류 ────────────────────────────────────────────────────────────────────
errs = [l.strip() for l in pane.splitlines()
        if re.search(r"Watchdog caught|rank 실패|!! 실패|CUDA out of memory|Killed|Traceback", l)]
if errs:
    line()
    print(f" {c('⚠ 오류', B+';'+RED):>9}")
    for e in errs[-3:]:
        print("           " + c(e[:W-12], RED))
print(c("━" * W, DIM))
PY
