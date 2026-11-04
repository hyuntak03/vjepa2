#!/bin/bash
# -----------------------------------------------------------------------------
# 학습 여러 개를 한 화면에서 (2026-11-02). run 마다 몇 줄로 줄이고, GPU 표 · 채점 대기열을 한 번만 보여 준다.
#
#   watch -c -n 2 bash z_training/monitor_multi.sh                                    # 기본: 오늘 v4 세 판 (scaffold 는 있으면)
#   watch -c -n 2 bash z_training/monitor_multi.sh v4vc_ar_vw_state_postft v4vc_release_postft
#   watch -c -n 2 bash z_training/monitor_multi.sh <run 이름 | run 폴더 절대경로> ...   # 다른 worktree 의 run 은 폴더 경로로
#   MONITOR_COLOR=0 ...                                                                 # 색 끄기
#
# run 마다: 살아 있나 (tmux train_<run> · launch.py) · step / 전체 · 진행 막대 · ETA (최근 20 step 평균 iter) ·
#          최근 loss · lr · iter / data ms · 학습 중 검증 (VOBench late 432 쌍: 전체 · 영속 late P / R) 이력 · 마지막 체크포인트 · 오류.
# 아래: GPU 표 (메모리 · 사용률) · 채점 대기열 (z_training/chain_op_eval.sh 로그) · 시트 칸 (op_cells_vast.py, 결과가 있을 때).
# 한 run 의 자세한 화면은 z_training/monitor.sh <run>.
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -eq 0 ]]; then
  set -- v4vc_ar_vw_state_postft v4vc_release_postft
  for d in /workspace/world_model/vjepa2_scaffold/z_training/runs/v4vc_release_scaffold; do [[ -d $d ]] && set -- "$@" "$d"; done
fi
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
ALIVE=""
for a in "$@"; do
  n=$(basename "$a")
  t=0; tmux has-session -t "=train_$n" 2>/dev/null && t=1
  p=$(pgrep -fc "launch.py --fname .*/runs/$n/config.yaml" 2>/dev/null || echo 0)
  ALIVE="$ALIVE$n=$t,$p;"
done
exec /venv/vjepa/bin/python - "$PROJ" "$GPU" "$ALIVE" "$@" <<'PYEOF'
import csv, glob, json, os, re, subprocess, sys, time
PROJ, GPU, ALIVE, RUNS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
COL = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, code): return f"\033[{code}m{s}\033[0m" if COL else str(s)
B, DIM, G, Y, R, CY = "1", "2", "32", "33", "31", "36"
def bar(fr, w=34):
    fr = max(0.0, min(1.0, fr)); n = int(round(fr * w)); return c("█" * n, G) + c("░" * (w - n), DIM)
def hms(s):
    s = int(max(0, s)); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
def kst(ts): return time.strftime("%H:%M", time.gmtime(ts + 9 * 3600))
alive = {}
for kv in ALIVE.strip(";").split(";"):
    if kv:
        k, v = kv.split("="); t, p = v.split(","); alive[k] = (t == "1", int(p) > 0)
now = time.time()
print(c(f"학습 모니터  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now + 9 * 3600))} KST", B))
for a in RUNS:
    d = a if os.path.isdir(a) else f"{PROJ}/z_training/runs/{a}"
    n = os.path.basename(d.rstrip("/"))
    print(c("─" * 78, DIM))
    if not os.path.isdir(d):
        print(c(f"{n}", B), c("폴더 없음 (아직 안 올라감)", Y)); continue
    tm, lp = alive.get(n, (False, False))
    E, ipe = 0, 0
    try:
        y = open(f"{d}/params-train.yaml").read()
        E = int(re.search(r"\n\s+epochs:\s*(\d+)", y).group(1)); m = re.search(r"\n\s+ipe:\s*(\d+)", y); ipe = int(m.group(1)) if m else 0
    except Exception:
        pass
    rows = []
    if os.path.exists(f"{d}/log_r0.csv"):
        rows = list(csv.DictReader(open(f"{d}/log_r0.csv")))
    if rows and not ipe:
        ipe = max(int(r["itr"]) for r in rows) + 1
    total = E * ipe if E and ipe else 0
    done_ck = os.path.exists(f"{d}/e{E}.pt") if E else False
    state = (c("완료", G) if done_ck and not lp else c("도는 중", CY) if lp else c("멈춤?", R) if rows else c("준비 중", Y))
    print(c(n, B), " ", state, c(f"(tmux {'O' if tm else 'X'} · 프로세스 {'O' if lp else 'X'})", DIM))
    if rows:
        last = rows[-1]; ep, it = int(last["epoch"]), int(last["itr"])
        step = (ep - 1) * ipe + it + 1
        it_ms = [float(r["iter-time(ms)"]) for r in rows[-20:]]
        avg = sum(it_ms) / len(it_ms) / 1000
        eta = (total - step) * avg if total else 0
        print(f"  step {step:>5}/{total}  {bar(step / total if total else 0)} {100 * step / max(total, 1):5.1f}%"
              + (f"  ETA {hms(eta)} (≈{kst(now + eta)} KST, 검증 제외)" if not done_ck else ""))
        print(f"  epoch {ep}/{E} · loss {float(last['loss']):.4f} · lr {float(last['lr']):.2e} · iter {avg * 1000:.0f} ms · data {float(last['data-time(ms)']):.0f} ms"
              + c(f" · 로그 {int(now - os.path.getmtime(f'{d}/log_r0.csv'))} 초 전", DIM))
    vals = []
    tl = f"{d}/train.log"
    if os.path.exists(tl):
        m = re.search(r"\[epoch 0\] val acc ([0-9.]+)%", open(tl, errors="ignore").read())
        if m: vals.append((0, float(m.group(1)), None, None))
    if os.path.exists(f"{d}/metrics.jsonl"):
        for ln in open(f"{d}/metrics.jsonl"):
            try: j = json.loads(ln)
            except Exception: continue
            v = j.get("val") or {}
            if "acc" in v:
                bt = v.get("by_type", {})
                pp = bt.get("primitive/permanence/late", {}).get("acc"); rp = bt.get("realistic/permanence/late", {}).get("acc")
                vals.append((j["epoch"], 100 * v["acc"], None if pp is None else 100 * pp, None if rp is None else 100 * rp))
    if vals:
        s = " → ".join(f"e{e} {a:.1f}" for e, a, _, _ in vals[-6:])
        e, a, pp, rp = vals[-1]
        print(f"  검증 (VOBench late 432) {s}" + (f"  │ 영속 late P/R {pp:.0f}/{rp:.0f}" if pp is not None else ""))
    cks = sorted(glob.glob(f"{d}/e*.pt"), key=os.path.getmtime)
    if cks:
        print(c(f"  체크포인트 {', '.join(os.path.basename(x)[:-3] for x in cks)} (마지막 {kst(os.path.getmtime(cks[-1]))} KST)", DIM))
    if os.path.exists(tl):
        tail = open(tl, errors="ignore").read()[-20000:]
        bad = [ln for ln in tail.splitlines() if re.search(r"Traceback|Error|out of memory|NaN|nan loss", ln)]
        if bad: print(c(f"  ⚠ {bad[-1][:150]}", R))
print(c("─" * 78, DIM))
g = [x.split(", ") for x in GPU.strip(";").split(";") if x]
if g:
    print("GPU  " + "  ".join(f"{i}:{int(u):>3}% {int(m) / 1024:4.1f}G" for i, m, t, u in g[:4]))
    if len(g) > 4: print("     " + "  ".join(f"{i}:{int(u):>3}% {int(m) / 1024:4.1f}G" for i, m, t, u in g[4:]))
L = f"{PROJ}/cvpr/results/eval/_logs_op_20261102"
chains = sorted(glob.glob(f"{L}/chain_*.log"))
if chains:
    print(c("채점 대기열 (chain_op_eval)", B))
    for f in chains:
        t = [x for x in open(f, errors="ignore").read().splitlines() if x.strip()]
        st = next((x for x in reversed(t) if x.startswith(("CHAIN_OP_DONE", "[chain_op_eval]", "###", "!!"))), t[-1] if t else "")
        print(f"  {os.path.basename(f)[6:-4]:28s} {st[:90]}")
models = ["vith", "vast_ar_vw_state", "vast_ar_vw_state_vobpft"] + [os.path.basename(f)[6:-4] for f in chains]   # vobpft = 시트 "AR └ VOBench Post Training"
try:
    out = subprocess.run(["/venv/vjepa/bin/python", f"{PROJ}/z_research/scripts/analysis/op_cells_vast.py", *models],
                         capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    if out:
        print(c("시트 Object_permanence 칸 (IntPhys1 late occ · IntPhys2 late occ · real world permanence)", B))
        hdr = [h.split(":")[-1] for h in out[0].split(" | ")]
        print("  " + " ".join(f"{h[:9]:>9}" if i else f"{h:24s}" for i, h in enumerate(hdr)))
        for ln in out[1:]:
            cs = ln.split(" | ")
            print("  " + " ".join(f"{x[:9]:>9}" if i else f"{x[:24]:24s}" for i, x in enumerate(cs)))
except Exception as e:
    print(c(f"  (표 실패: {e})", DIM))
PYEOF
