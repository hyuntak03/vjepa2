#!/bin/bash
# -----------------------------------------------------------------------------
# z_training 학습 모니터 (2026-09-19)
#
#   watch -c -n 1 bash z_training/monitor.sh                   # 가장 최근 학습 slurm job
#   watch -c -n 1 bash z_training/monitor.sh 215017            # job id
#   watch -c -n 1 bash z_training/monitor.sh intphys2_postft   # run 이름 (z_training/runs/<이름>)
#   MONITOR_COLOR=0 ...                                          # 색 끄기
#
# 읽는 것: runs/<run>/log_r0.csv (epoch,itr,loss,ctx_frames,lr,iter-time…), params-train.yaml (epochs · batch),
#          metrics.jsonl (epoch 끝 요약·val), train.log / slurm .err (오류), e{N}.pt (저장된 체크포인트).
# GPU: job 이 다른 노드에 있으면 `srun --jobid --overlap nvidia-smi` 로 읽고 15 초 캐시한다 (/tmp/zt_monitor_gpu_<job>).
# ETA = 남은 step × 최근 20 step 평균 iter 시간. step/epoch 는 log 의 itr 최댓값 + 1 (첫 epoch 이 끝나기 전엔 추정).
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
LOGDIR=$PROJ/z_training/slurm_logs
ARG="${1:-}"

JOBID=""; RUN=""
if [[ -z "$ARG" ]]; then
  f=$(ls -t "$LOGDIR"/*.out 2>/dev/null | head -1)
  [[ -n "$f" ]] && JOBID=$(basename "$f" .out | sed 's/.*_//')
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
  JOBID="$ARG"
else
  RUN="$ARG"
fi
OUT=""; ERR=""
if [[ -n "$JOBID" ]]; then
  OUT=$(ls "$LOGDIR"/*_"$JOBID".out 2>/dev/null | head -1); ERR="${OUT%.out}.err"
  [[ -z "$RUN" && -n "$OUT" ]] && RUN=$(grep -m1 -o "NAME=[^ |]*" "$OUT" 2>/dev/null | cut -d= -f2)
fi
if [[ -z "$JOBID" && -n "$RUN" ]]; then          # run 이름만 줬으면 그 run 의 최신 job 을 찾는다
  f=$(grep -l "NAME=$RUN " "$LOGDIR"/*.out 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  [[ -n "$f" ]] && { JOBID=$(basename "$f" .out | sed 's/.*_//'); OUT="$f"; ERR="${f%.out}.err"; }
fi

STATE=""; NODE=""; ELAPSED=""; LIMIT=""
if [[ -n "$JOBID" ]]; then
  Q=$(squeue -j "$JOBID" -h -o "%T|%N|%M|%l" 2>/dev/null)
  if [[ -n "$Q" ]]; then IFS='|' read -r STATE NODE ELAPSED LIMIT <<<"$Q"
  else STATE=$(sacct -j "$JOBID" -X -n -o State 2>/dev/null | head -1 | tr -d ' '); STATE=${STATE:-큐에없음}; fi
fi

GPU=""
if [[ "$STATE" == "RUNNING" ]]; then
  C=/tmp/zt_monitor_gpu_$JOBID
  if [[ ! -f $C || $(( $(date +%s) - $(stat -c %Y $C) )) -gt 15 ]]; then
    if [[ "$NODE" == "$(hostname -s)" ]]; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > $C.tmp 2>/dev/null
    else
      env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID timeout 8 srun --jobid "$JOBID" --overlap \
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > $C.tmp 2>/dev/null
    fi
    [[ -s $C.tmp ]] && mv $C.tmp $C || touch $C
  fi
  GPU=$(tr '\n' ';' < $C 2>/dev/null)
fi

exec /data/hyuntak/anaconda3/envs/vjepa2/bin/python - "$PROJ" "$RUN" "$JOBID" "$STATE" "$NODE" "$ELAPSED" "$LIMIT" "$GPU" "$ERR" <<'PYEOF'
import csv, glob, json, os, re, sys, time
PROJ, RUN, JOB, STATE, NODE, ELA, LIM, GPU, ERR = sys.argv[1:10]
COL = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, code): return f"\033[{code}m{s}\033[0m" if COL else str(s)
B, DIM, G, Y, R, CY, MG = "1", "2", "32", "33", "31", "36", "35"
W = 78
def bar(fr, w=40, col=G):
    fr = max(0.0, min(1.0, fr)); n = int(round(fr * w)); return c("█" * n, col) + c("░" * (w - n), DIM)
def hms(s):
    if s is None or s != s: return "  -  "
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60); return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
SPARK = "▁▂▃▄▅▆▇█"
def spark(v):
    if not v: return ""
    lo, hi = min(v), max(v); r = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((x - lo) / r * 7.999))] for x in v)

print(c("━" * W, CY)); print(c(" z_training 학습 모니터".ljust(W - 20) + time.strftime("%Y-%m-%d %H:%M:%S"), B)); print(c("━" * W, CY))
if not RUN:
    print(c(" run 을 못 찾았다 — 인자로 job id 또는 run 이름을 준다", R)); sys.exit()
rd = os.path.join(PROJ, "z_training/runs", RUN)
scol = {"RUNNING": G, "PENDING": Y, "COMPLETED": CY}.get(STATE, R)
print(f" run   {c(RUN, B)}    job {JOB or '-'}  {c(STATE or '-', scol)}  node {NODE or '-'}  경과 {ELA or '-'} / {LIM or '-'}")

# ---- config
ep_total = bs = ngpu = None; lr_cfg = None; ctx = None; ds = None
p = os.path.join(rd, "params-train.yaml")
if os.path.isfile(p):
    import yaml
    try:
        y = yaml.safe_load(open(p)); ep_total = y["optimization"]["epochs"]; bs = y["data"]["batch_size"]; lr_cfg = y["optimization"]["lr"]
        ctx = ",".join(str(v) for m in y.get("mask", []) for v in (m.get("context_frames") or [m.get("type")])); ds = [d.get("csv") or d.get("root") or d.get("name") for d in y["data"]["datasets"]]
        nf = y["data"].get("n_frames")
    except Exception:
        nf = None
ngpu = len(glob.glob(os.path.join(rd, "log_r*.csv"))) or None
if ep_total:
    print(f" 설정  epochs {ep_total} · batch {bs}/rank × {ngpu} GPU = {bs * ngpu if bs and ngpu else '-'} · 창 {nf} 장 · C {ctx} · lr {lr_cfg}")
    for d in ds or []: print(c(f"       data {os.path.relpath(d, PROJ) if d and d.startswith(PROJ) else d}", DIM))

# ---- log
rows = []
lp = os.path.join(rd, "log_r0.csv")
if os.path.isfile(lp):
    for r in csv.DictReader(open(lp)):
        try: rows.append((int(r["epoch"]), int(r["itr"]), float(r["loss"]), int(r["ctx_frames"]), float(r["lr"]), float(r["iter-time(ms)"])))
        except Exception: pass
print(c("─" * W, DIM))
if not rows:
    print(c(" 아직 step 로그가 없다 (모델 로딩 중일 수 있다 — ViT-H 로드 ~2 분)", Y))
else:
    ep, it, loss, C, lr, itms = rows[-1]
    by_ep = {}
    for e, i, l, *_ in rows: by_ep.setdefault(e, []).append(l)
    done_eps = [e for e in by_ep if e < ep]
    ipe = max(i for e, i, *_ in rows if e in done_eps) + 1 if done_eps else max(it + 1, 1)
    step = (ep - 1) * ipe + it + 1; total = (ep_total or ep) * ipe
    recent = [x[5] for x in rows[-20:]]; t_it = sum(recent) / len(recent) / 1000
    eta = (total - step) * t_it
    print(f" epoch {c(f'{ep:>3}/{ep_total or "?"}', B)}  {bar(ep / (ep_total or ep))}  {100 * ep / (ep_total or ep):5.1f}%")
    print(f" step  {c(f'{step:>4}/{total}', B)}  {bar(step / total, col=CY)}  ({ipe} step/epoch)")
    print(f" ETA   {c(hms(eta), B + ';' + MG)}   ({t_it:.2f} s/step, 최근 20 step)   끝나는 시각 ≈ {time.strftime('%H:%M', time.localtime(time.time() + eta))}")
    print(c("─" * W, DIM))
    l10 = [x[2] for x in rows[-10:]]; first = by_ep[min(by_ep)]
    print(f" loss  지금 {c(f'{loss:.4f}', B)} (C={C})   최근 10 평균 {c(f'{sum(l10) / len(l10):.4f}', B)}   첫 epoch 평균 {sum(first) / len(first):.4f}   lr {lr:.2e}")
    em = [sum(v) / len(v) for e, v in sorted(by_ep.items()) if e < ep]
    if em:
        d = em[-1] - em[0]
        print(f" epoch 평균 loss  {c(spark(em[-60:]), G)}  {em[0]:.4f} → {em[-1]:.4f}  ({c(f'{d:+.4f}', G if d < 0 else R)})")
    # C 별 최근 loss (C 가 섞여 뽑히므로 C 끼리 비교)
    byC = {}
    for e, i, l, cc, *_ in rows[-10 * ipe:]: byC.setdefault(cc, []).append(l)
    print(" C 별 (최근 10 epoch)  " + "  ".join(f"C{k}:{sum(v) / len(v):.4f}" for k, v in sorted(byC.items())))

# ---- checkpoints / metrics
ck = sorted(glob.glob(os.path.join(rd, "e*.pt")), key=lambda f: int(re.findall(r"e(\d+)\.pt", f)[0]))
if ck:
    last = ck[-1]; age = time.time() - os.path.getmtime(last)
    print(f" 저장  {len(ck)} 개 (마지막 {os.path.basename(last)}, {hms(age)} 전)  latest.pt {'있음' if os.path.isfile(os.path.join(rd, 'latest.pt')) else '없음'}")
mp = os.path.join(rd, "metrics.jsonl")
if os.path.isfile(mp):
    L = [json.loads(l) for l in open(mp) if l.strip()]
    vals = [(m["epoch"], m["val"]["acc"]) for m in L if isinstance(m.get("val"), dict) and "acc" in m["val"]]
    if vals: print(f" val   " + "  ".join(f"e{e}:{100 * a:.1f}" for e, a in vals[-8:]))

# ---- GPU
print(c("─" * W, DIM))
if GPU:
    for g in [x for x in GPU.split(";") if x.strip()]:
        try:
            i, used, tot, ut = [v.strip() for v in g.split(",")]; fr = float(used) / float(tot)
            print(f" GPU{i} {bar(fr, 24, R if fr > .95 else (Y if fr > .85 else G))} {float(used) / 1024:5.1f}/{float(tot) / 1024:4.1f} G  util {ut:>3}%")
        except Exception: pass
elif STATE == "RUNNING":
    print(c(" GPU  (읽는 중 — 15 초마다 갱신)", DIM))

# ---- 오류
bad = []
for f in [ERR, os.path.join(rd, "train.log")]:
    if f and os.path.isfile(f):
        try:
            tail = open(f, errors="ignore").read()[-200000:].splitlines()
            bad += [l for l in tail if re.search(r"Traceback|out of memory|OutOfMemory|Error:|Killed|NCCL.*(timeout|error)", l)]
        except Exception: pass
if bad:
    print(c("─" * W, DIM)); print(c(f" ⚠ 오류 흔적 {len(bad)} 줄 — 마지막:", R))
    for l in bad[-3:]: print(c("   " + l[:W - 3], R))
print(c("━" * W, CY))
PYEOF
