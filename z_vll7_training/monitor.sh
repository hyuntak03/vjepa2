#!/bin/bash
# =============================================================================
# vll7 학습 모니터 (2026-10-16). z_training/monitor.sh 의 진행 막대 · 색 + monitor_ar.sh 의 정보 구성을 합쳤다.
#
#   watch -c -n 5 bash z_vll7_training/monitor.sh            # 도는 run 자동 선택
#   bash z_vll7_training/monitor.sh ctxt_video_tgtvideo      # 지정
#   MONITOR_COLOR=0 bash ...                                 # 색 끄기
#
# 읽는 것
#   config.yaml      팔 · 창 · 데이터 · lr
#   log_r0.csv       **rank 0 만** — step 통계는 여기서 (train.log 는 rank 6 개가 같이 써서 6 중복이다)
#   train.log        (C,K) · 데이터셋 · tf/ar 분해 (step 별 중복 제거)
#   metrics.jsonl    epoch 끝 요약 · in-loop val
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" >/dev/null
RUN=${1:-}
if [[ -z "$RUN" ]]; then
  RUN=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' | head -1 | sed 's/^train_//')
  [[ -z "$RUN" ]] && RUN=$(ls -t "$VLL7_RUNS" 2>/dev/null | head -1)
fi
D="$VLL7_RUNS/$RUN"
[[ -d "$D" ]] || { echo "run 없음: $D   (있는 것: $(ls "$VLL7_RUNS" 2>/dev/null | tr '\n' ' '))"; exit 1; }

tmux has-session -t "=train_$RUN" 2>/dev/null && SESS=1 || SESS=0

# ── GPU 읽기 ────────────────────────────────────────────────────────────────
# ⚠️ 이 셸이 GPU 할당 **밖**이면 (로그인 셸, --gres 없는 job) nvidia-smi 가 빈 결과를 낸다.
#    그때는 학습이 도는 SLURM job 을 찾아 `srun --jobid --overlap` 으로 읽는다. 둘 다 안 되면 이유를 찍는다.
Q="--query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
GPU=$(nvidia-smi $Q 2>/dev/null | tr '\n' ';')
GPU_SRC="local"
if [[ -z "${GPU//;/}" ]]; then
  # 이 노드에서 GPU 를 쥐고 도는 내 job (gres:gpu:N, N>=1)
  JID=$(squeue -u "$USER" -h -o "%i|%b|%N|%T" 2>/dev/null         | awk -F'|' -v n="$(hostname -s)" '$2 ~ /gpu:[1-9]/ && $3==n && $4=="RUNNING"{print $1; exit}')
  if [[ -n "${JID:-}" ]]; then
    GPU=$(env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID           timeout 8 srun --jobid "$JID" --overlap nvidia-smi $Q 2>/dev/null | tr '\n' ';')
    GPU_SRC="srun job $JID"
  fi
fi
[[ -z "${GPU//;/}" ]] && GPU_SRC="none"

# ── 이 run 의 GPU 만 남긴다 (2026-10-20) ────────────────────────────────────
# 노드 하나에 run 을 둘 이상 올리면 (GPU_IDS 로 나눠 쓰는 경우) nvidia-smi 는 **전부**를 보여 준다.
# <run>/pids.txt 의 rank pid 와 compute-apps 의 pid 를 맞춰 자기 카드만 고른다.
# pids.txt 가 없거나 (옛 run) 겹치는 pid 가 없으면 (이미 죽은 run) 예전처럼 전부 보여 준다.
MYGPU=""
if [[ -f "$D/pids.txt" ]]; then
  MYGPU=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F', *' -v pl="$(tr '\n' ' ' < "$D/pids.txt")" 'BEGIN{split(pl,a," "); for(i in a) P[a[i]]=1} $2 in P {print $1}' \
    | while read -r u; do nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', *' -v u="$u" '$2==u{print $1}'; done | sort -n | tr '\n' ' ')
fi
if [[ -n "${MYGPU// /}" ]]; then
  GPU=$(echo "$GPU" | tr ';' '\n' | awk -F', *' -v ids="$MYGPU" 'BEGIN{split(ids,a," "); for(i in a) K[a[i]]=1} $1 in K' | tr '\n' ';')
  NR=$(echo $MYGPU | wc -w | tr -dc '0-9')
else
  NR=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -dc '0-9')
  if [[ -z "$NR" || "$NR" == "0" ]]; then NR=$(pgrep -cf "python.*launch.py" 2>/dev/null | tr -dc '0-9'); fi
fi
[[ -z "$NR" ]] && NR=0

exec "$VJEPA2_PY" - "$D" "$RUN" "$SESS" "$NR" "$GPU" "$GPU_SRC" <<'PY'
import csv, json, os, re, sys, time
from collections import Counter, OrderedDict
from datetime import datetime, timedelta

D, RUN, SESS, NR, GPU, GPU_SRC = sys.argv[1:7]
SESS, NR = int(SESS), int(NR)
COL = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, code):  return f"\033[{code}m{s}\033[0m" if COL else str(s)
B, DIM, GRN, YEL, RED, CYA, MAG = "1", "2", "32", "33", "31", "36", "35"
W = 86
def line(ch="─"): print(c(ch * W, DIM))
def bar(fr, w=38, col=GRN):
    fr = max(0.0, min(1.0, fr)); n = int(round(fr * w))
    return c("█" * n, col) + c("░" * (w - n), DIM)
def hms(s):
    s = int(max(0, s)); return f"{s//3600}:{s%3600//60:02d}:{s%60:02d}"
def spark(vals, lo=None, hi=None):
    if not vals: return ""
    ch = "▁▂▃▄▅▆▇█"
    lo = min(vals) if lo is None else lo; hi = max(vals) if hi is None else hi
    if hi - lo < 1e-9: return ch[0] * len(vals)
    return "".join(ch[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)

cfg = {}
try:
    import yaml; cfg = yaml.safe_load(open(os.path.join(D, "config.yaml"), encoding="utf-8")) or {}
except Exception: pass
O = cfg.get("optimization", {}) or {}
M = cfg.get("model", {}) or {}
DA = cfg.get("data", {}) or {}
L = cfg.get("loss", {}) or {}
P = M.get("predictor") or {}
ipe, eps, ga = int(O.get("ipe") or 0), int(O.get("epochs") or 0), int(O.get("grad_accum", 1))
total = ipe * eps

# ── 헤더 ────────────────────────────────────────────────────────────────────
print(c("━" * W, DIM))
if SESS and NR:   status = c("● 학습 중", GRN) + c(f" · rank {NR}", DIM)
elif SESS:        status = c("◐ 세션은 있으나 rank 없음", YEL)
else:             status = c("○ 멈춤 (tmux 세션 없음)", RED)
started = None
tl = os.path.join(D, "train.log")
if os.path.isfile(tl):
    try:
        first = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", open(tl, errors="ignore").readline())
        started = datetime.strptime(first.group(1), "%Y-%m-%d %H:%M:%S") if first else None
    except Exception: pass
el = f"  {c('경과', DIM)} {hms((datetime.now() - started).total_seconds())}" if started else ""
print(f" {c(RUN, B+';'+CYA)}   {status}{el}")
line()

# ── 설정 ────────────────────────────────────────────────────────────────────
kind = P.get("kind", "?")
ctxsrc = "영상(창 전체)" if kind in ("ctx_state", "ctx_ar") else "블록별"
tgt = {"window": "영상(창 전체)", "block": "블록별"}.get(P.get("ctx_target"), "블록별")
print(f" {c('팔', B):>6}   kind={c(kind, B+';'+MAG)}  문맥={ctxsrc}  타깃={c(tgt, B)}"
      f"  loss = {L.get('tf_weight', 1.0)}×tf + rollout(R≤{L.get('rollout_steps')})")
vw = DA.get("var_window")
if vw:
    print(f" {c('창', B):>6}   가변 (fps 4 · 패딩 없음)  C{vw['context_frames']} × K{vw['predict_frames']} 장"
          f"  ·  rank 당 B = {vw['token_budget']} // (C+K)")
    ds_w = {d["name"]: d.get("weight", 1.0) for d in vw["datasets"]}
else:
    print(f" {c('창', B):>6}   고정 {DA.get('n_frames')} 장 × batch {DA.get('batch_size')}/rank"
          + (f" × accum {ga}" if ga > 1 else "")
          + f"  ·  mask {cfg.get('mask', [{}])[0].get('context_blocks')} → {cfg.get('mask', [{}])[0].get('predict_blocks')} 블록")
    ds_w = {s.get("name", "?"): s.get("weight", 1.0) for s in DA.get("datasets", [])}
sched = f"WSD (마지막 {O.get('anneal_epochs')} ep 감소)" if O.get("anneal_epochs") else ("선형 감소" if O.get("is_anneal") else "cosine")
print(f" {c('데이터', B):>6}   " + ", ".join(f"{k}×{v}" for k, v in ds_w.items())
      + f"   ·   lr {O.get('lr')} {sched}   ·   {eps} ep × ipe {ipe}" + (f" × {ga}" if ga > 1 else "") + f" = {total:,} step")
line()

# ── 진행 · ETA (log_r0.csv = rank 0 만) ─────────────────────────────────────
rows = []
f0 = os.path.join(D, "log_r0.csv")
if os.path.isfile(f0):
    rows = [r for r in csv.DictReader(open(f0)) if (r.get("iter-time(ms)") or "").replace(".", "", 1).isdigit()]
done, med = 0, None
if rows:
    try:
        last = re.findall(r"\] \[(\d+), +(\d+)\] loss", open(tl, errors="ignore").read())[-1]
        done = (int(last[0]) - 1) * ipe * ga + int(last[1]) + 1
    except Exception:
        done = len(rows)
    it = sorted(float(r["iter-time(ms)"]) / 1000 for r in rows[-200:])
    med = it[len(it) // 2] if it else None
if total and done:
    ep_now = min(eps, done // max(1, ipe * ga) + 1)
    print(f" {c('진행', B):>6}   {bar(done / total)}  {c(f'{100*done/total:5.1f}%', B)}"
          f"   {c(f'{ep_now}/{eps}', B)} ep   {done:,}/{total:,} step")
    if med:
        left = (total - done) * med
        end = (datetime.now() + timedelta(seconds=left)).strftime("%m-%d %H:%M")
        print(f" {c('ETA', B):>6}   {c(hms(left), B+';'+YEL)} 남음   끝 ≈ {c(end, B+';'+YEL)}"
              f"   ({med:.2f} s/step 중앙값, {med*ipe*ga/60:.1f} 분/epoch)")
else:
    print(f" {c('진행', B):>6}   {bar(0)}   (첫 step 대기 — ViT-H 로드 · torch.compile)")
line()

# ── loss · (C,K) · 데이터 비율 (train.log, step 중복 제거) ──────────────────
# MONITOR_NO_LOSS=1 이면 이 구역을 통째로 건너뛴다 — monitor_state_head.sh 가 자기 형식으로 다시 찍는다
#   (예전에는 출력에서 grep -v 로 지웠는데, 색이 켜지면 `^ *loss` 앞에 escape 가 붙어 안 지워졌다)
pat = re.compile(r"\[(\d+), +(\d+)\] loss ([\d.]+) \(avg ([\d.]+)\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\].*?"
                 r"\| mask var C(\d+)K(\d+) (\w+) B(\d+)")
# 2026-10-20: 고정 창 팔은 `| mask prefix C8K8 C=16` 이라 위 regex 에 안 걸려 loss 줄이 통째로 빠져 있었다.
#   (C,K) 별 · 데이터 비율은 가변 창에만 있는 축이라 그대로 가변 전용으로 둔다.
pat_fix = re.compile(r"\[(\d+), +(\d+)\] loss ([\d.]+) \(avg ([\d.]+)\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\].*?"
                     r"\| mask (\S+) C(\d+)K(\d+)")
seen, seen_fix = OrderedDict(), OrderedDict()
if os.path.isfile(tl) and not os.environ.get("MONITOR_NO_LOSS"):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen[(m.group(1), m.group(2))] = m.groups(); continue
        m = pat_fix.search(ln)
        if m: seen_fix[(m.group(1), m.group(2))] = m.groups()
S = list(seen.values())
if not S and seen_fix:
    e, i, l, avg, tf, ar, R, ds, C, K = list(seen_fix.values())[-1]
    print(f" {c('loss', B):>6}   {c(f'{float(l):.4f}', B)}  =  tf {float(tf):.4f}  +  ar {float(ar):.4f}"
          f"   {c(f'(epoch 평균 {float(avg):.4f})', DIM)}   {c(f'지금 {ds} C{C}K{K} R={R}', DIM)}")
if S:
    e, i, l, avg, tf, ar, R, C, K, ds, Bb = S[-1]
    print(f" {c('loss', B):>6}   {c(f'{float(l):.4f}', B)}  =  tf {float(tf):.4f}  +  ar {float(ar):.4f}"
          f"   {c(f'(epoch 평균 {float(avg):.4f})', DIM)}"
          f"   {c(f'지금 C{C}K{K} {ds} B{Bb} R={R}', DIM)}")
    ck = {}
    for g in S[-400:]: ck.setdefault((int(g[7]), int(g[8])), []).append(float(g[5]))
    cells = sorted(ck.items())
    print(f" {c('(C,K)별', B):>6} " + "  ".join(f"C{a}K{b}:{sum(v[-5:])/len(v[-5:]):.3f}" for (a, b), v in cells[:7]))
    if len(cells) > 7:
        print(" " * 8 + "  ".join(f"C{a}K{b}:{sum(v[-5:])/len(v[-5:]):.3f}" for (a, b), v in cells[7:]))
    cnt = Counter(g[9] for g in S); tot = sum(cnt.values())
    share = "  ".join(f"{k} {100*v/tot:.1f}%" for k, v in cnt.most_common())
    print(f" {c('데이터', B):>6}   {share}   {c('(실측 기대 k400 76.7 : ssv2 23.3)', DIM)}")

# ── epoch · val ─────────────────────────────────────────────────────────────
mrows = []
mp = os.path.join(D, "metrics.jsonl")
if os.path.isfile(mp):
    for ln in open(mp, encoding="utf-8"):
        try: mrows.append(json.loads(ln))
        except Exception: pass
if mrows:
    ls = [r["train_loss"] for r in mrows]
    print(f" {c('epoch', B):>6}   loss {spark(ls)}  " + "  ".join(f"e{r['epoch']}:{r['train_loss']:.3f}" for r in mrows[-6:]))
    vs = [(r["epoch"], 100 * r["val"]["acc"]) for r in mrows if (r.get("val") or {}).get("acc") is not None]
    if vs:
        best = max(vs, key=lambda x: x[1])
        print(f" {c('val', B):>6}   IntPhys1 dev 60 쌍 (rollout)  {spark([a for _, a in vs])}  "
              + "  ".join(f"e{e}:{a:.1f}" for e, a in vs[-6:])
              + f"   {c(f'최고 e{best[0]} {best[1]:.1f}%', B+';'+GRN)}")
line()

# ── GPU (2 개씩 한 줄) ──────────────────────────────────────────────────────
gpus = []
for g in [x for x in GPU.split(";") if x.strip()]:
    try:
        idx, util, used, tot_m = [t.strip() for t in g.split(",")]
        gpus.append((int(idx), int(util), int(used), int(tot_m)))
    except Exception:
        pass
if gpus:
    avg = sum(u for _, u, _, _ in gpus) / len(gpus)
    pk = max(m for _, _, m, _ in gpus) / 1024
    print(f" {c('GPU', B):>6}   {len(gpus)} 장 · 평균 util {c(f'{avg:.0f}%', B)} · 피크 {c(f'{pk:.1f} GB', B)}"
          f" / {gpus[0][3]/1024:.0f} GB" + ("" if GPU_SRC == "local" else c(f"   [{GPU_SRC}]", DIM)))
    for k in range(0, len(gpus), 2):
        cells = []
        for idx, u, used, tot_m in gpus[k:k+2]:
            col = GRN if u >= 80 else (YEL if u >= 30 else RED)
            cells.append(f"{c(f'gpu{idx}', B)} {bar(u/100, 14, col)} {u:3d}%  {used/1024:5.1f}G")
        print("        " + "    ".join(cells))
else:
    why = ("이 셸에 GPU 가 안 보인다 — 로그인 셸이거나 --gres 없는 job 이다.\n"
           "          학습 job 안에서 돌리거나 (srun --jobid <id> --overlap --pty bash), 그 job 이 없는지 확인할 것")
    print(f" {c('GPU', B):>6}   {c('읽을 수 없음', YEL)} — {why}")
    # 폴백: train.log 가 찍은 rank 0 의 max_memory_allocated
    if S:
        mm = re.findall(r"mem ([\d.]+)G", open(tl, errors="ignore").read()[-20000:])
        if mm:
            print(f"        {c('(train.log 의 rank 0 메모리 피크)', DIM)} {float(mm[-1]):.1f} GB")

# ── 오류 · 체크포인트 ───────────────────────────────────────────────────────
errs = []
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        if re.search(r"Traceback|rank 실패|OutOfMemory|loss 가 nan", ln): errs.append(ln.rstrip()[:W-6])
if errs:
    line(); print(f" {c('⚠ 오류', B+';'+RED):>6}")
    for e in errs[-3:]: print("   " + c(e, RED))
cks = sorted(x for x in os.listdir(D) if x.endswith(".pt"))
line()
print(f" {c('ckpt', B):>6}   " + (("  ".join(cks)) if cks else c("(아직 없음)", DIM)) + c(f"   {D}", DIM))
print(c("━" * W, DIM))
PY
