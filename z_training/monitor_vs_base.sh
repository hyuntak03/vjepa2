#!/bin/bash
# -----------------------------------------------------------------------------
# 섞기 학습 하나를 기준 run 과 같은 step 끼리 나란히 보는 모니터 (2026-11-03).
#   기본: natural_ar_vw_state_v4vc (처음부터 v4 10 % 섞기) vs natural_ar_vw_state (섞기 없음, 10-12 학습).
#
#   watch -c -n 5 bash z_training/monitor_vs_base.sh
#   watch -c -n 5 bash z_training/monitor_vs_base.sh <run> <기준 run>
#   EVAL_LOG=z_training/logs/<채점 chain 로그> ...   # 아래 채점 상태 줄 (기본: v4 섞기 판 chain_full_eval 로그)
#   MONITOR_COLOR=0 ...                              # 색 끄기
#
# 보여 주는 것
#   진행: step / 전체 · 막대 · epoch · lr · GPU 메모리 · 경과 시간
#   ETA : 기준 run 의 시각표 × (지금 run 경과 / 기준 run 같은 step 경과).
#         두 run 은 섞기 step 말고는 계획 (C, K, 데이터셋, B) 이 같아서 남은 시간 모양이 같다. 체크포인트 저장 시간 포함.
#   loss: 섞기 step 은 (seed, step) 으로 다시 계산한다 (var_window._is_fixed 와 같은 식).
#         자연 step = 기준 run 과 같은 step · 같은 영상 (rank 0) → 차이 = 섞기 비용.  v4 step = 따로 평균.
#   체크포인트 · train.log 오류 · GPU 표 · 채점 chain 상태 (끝나면 표).
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN=${1:-natural_ar_vw_state_v4vc}; BASE=${2:-natural_ar_vw_state}
EVAL_LOG=${EVAL_LOG:-$PROJ/z_training/logs/chain_full_eval_vast_ar_vw_state_v4vc_scratch.log}
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
T=0; tmux has-session -t "=train_$RUN" 2>/dev/null && T=1
P=$(ps -eo args | grep -cE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$RUN/config\.yaml")
exec /venv/vjepa/bin/python - "$PROJ" "$RUN" "$BASE" "$EVAL_LOG" "$GPU" "$T" "$P" <<'PYEOF'
import calendar, csv, glob, os, random, re, sys, time
import yaml
PROJ, RUN, BASE, EVAL_LOG, GPU, T, P = sys.argv[1:8]
COL = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, code): return f"\033[{code}m{s}\033[0m" if COL else str(s)
B, DIM, G, Y, R, CY = "1", "2", "32", "33", "31", "36"
def bar(fr, w=40):
    fr = max(0.0, min(1.0, fr)); n = int(round(fr * w)); return c("█" * n, G) + c("░" * (w - n), DIM)
def hms(s):
    s = int(max(0, s)); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
def kst(ts, fmt="%m-%d %H:%M"): return time.strftime(fmt, time.gmtime(ts + 9 * 3600))
now = time.time()
rd, bd = f"{PROJ}/z_training/runs/{RUN}", f"{PROJ}/z_training/runs/{BASE}"
print(c(f"학습 모니터  {kst(now, '%Y-%m-%d %H:%M:%S')} KST", B), c(f"  {RUN}  vs 기준 {BASE}", DIM))
print(c("─" * 84, DIM))
if not os.path.isdir(rd):
    print(c(f"{rd} 없음 (아직 안 올라감)", Y)); sys.exit()

cfg = yaml.safe_load(open(f"{rd}/config.yaml"))
seed = int(cfg["meta"]["seed"]); E = int(cfg["optimization"]["epochs"]); ipe = int(cfg["optimization"]["ipe"])
fx = ((cfg.get("data") or {}).get("var_window") or {}).get("fixed") or {}
p_fix = float(fx.get("p", 0) or 0)
mix = "+".join(s.get("name", "?") for s in fx.get("sources", [])) or "-"
total = E * ipe
def is_fixed(step): return p_fix > 0 and random.Random(f"{seed}-fixed-{step}").random() < p_fix

def read_csv(d):
    f = f"{d}/log_r0.csv"
    if not os.path.exists(f): return {}
    out = {}
    for r in csv.DictReader(open(f)):
        try: out[(int(r["epoch"]) - 1) * ipe + int(r["itr"])] = (float(r["loss"]), int(r["ctx_frames"]), float(r["lr"]), float(r["iter-time(ms)"]))
        except (ValueError, KeyError): pass
    return out

LINE = re.compile(r"\]\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\[app\.vjepa_frozen\.train\]\S*\s*\]\s*\[\s*(\d+),\s*(\d+)\]")
def timeline(d):
    """train.log 의 step 로그 줄 → {step: UTC 초} (첫 줄만). 첫 step 0 의 시각을 시작으로 쓴다."""
    f = f"{d}/train.log"; tl = {}
    if not os.path.exists(f): return tl
    for m in LINE.finditer(open(f, errors="ignore").read()):
        s = (int(m.group(2)) - 1) * ipe + int(m.group(3))
        if s not in tl: tl[s] = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    return tl

run, base = read_csv(rd), read_csv(bd)
tl_r, tl_b = timeline(rd), timeline(bd)
lp = int(P) > 0
done = os.path.exists(f"{rd}/e{E}.pt")
state = c("완료", G) if done and not lp else c("도는 중", CY) if lp else c("멈춤?", R) if run else c("준비 중", Y)
print(c(RUN, B), " ", state, c(f"(tmux {'O' if T == '1' else 'X'} · 프로세스 {'O' if lp else 'X'}) · 섞기 {mix} p {p_fix} · seed {seed}", DIM))
if not run:
    sys.exit()
step = max(run); ep, it = step // ipe + 1, step % ipe
L, ctx, lr, _ = run[step]
mem = ""
tail = open(f"{rd}/train.log", errors="ignore").read()[-30000:] if os.path.exists(f"{rd}/train.log") else ""
m = re.findall(r"mem ([0-9.]+G)", tail)
if m: mem = f" · GPU 메모리 {m[-1]}"
t0 = tl_r.get(0)
el = (now - t0) if t0 else None
print(f"  step {step + 1:>6}/{total}  {bar((step + 1) / total)} {100 * (step + 1) / total:5.1f}%")
print(f"  epoch {ep}/{E} (itr {it}/{ipe}) · lr {lr:.2e}{mem}" + (f" · 경과 {hms(el)} (시작 {kst(t0)} KST)" if el else "")
      + c(f" · 로그 {int(now - os.path.getmtime(f'{rd}/log_r0.csv'))} 초 전", DIM))

# ETA — 기준 run 시각표를 지금 속도로 늘인다
if not done and tl_r and tl_b and len(tl_b) > 1:
    s_r = max(tl_r); s_b = max(k for k in tl_b if k <= s_r) if any(k <= s_r for k in tl_b) else None
    b0, bend = tl_b[min(tl_b)], tl_b[max(tl_b)]
    if s_b is not None and tl_b[s_b] > b0:
        ratio = (tl_r[s_r] - t0) / (tl_b[s_b] - b0)
        rem = (bend - tl_b[s_b]) * ratio - (now - tl_r[s_r])
        tot_b = bend - b0
        print(f"  ETA  {c(hms(rem), B)} 남음 → 학습 끝 ≈ {c(kst(now + rem) + ' KST', B)}"
              + c(f"  (기준 run 같은 step 대비 속도 ×{ratio:.2f}, 기준 전체 {hms(tot_b)} → 예상 전체 {hms(tot_b * ratio)})", DIM))
        print(c(f"       채점 (VOBench v3 · IntPhys1 late · IntPhys2 late · real world, 8 GPU) 약 1 시간 더 → ≈ {kst(now + rem + 3600)} KST", DIM))

# loss — 자연 step 은 기준 run 과 같은 step, v4 step 은 따로
def window(lo, hi):
    nat = [(run[s][0], base[s][0]) for s in range(lo, hi) if s in run and s in base and not is_fixed(s)]
    fix = [run[s][0] for s in range(lo, hi) if s in run and is_fixed(s)]
    return nat, fix
print(c("  loss (rank 0)            자연 step: 지금 run │ 기준 run │ 차이        v4 step: 평균 (개수)", DIM))
spans = [("최근 1 epoch", max(0, step + 1 - ipe), step + 1), ("최근 10 epoch", max(0, step + 1 - 10 * ipe), step + 1), ("처음부터", 0, step + 1)]
for name, lo, hi in spans:
    nat, fix = window(lo, hi)
    if not nat: continue
    a = sum(x for x, _ in nat) / len(nat); b = sum(y for _, y in nat) / len(nat); dlt = a - b
    dc = R if dlt > 0.005 else G if dlt < -0.005 else DIM
    fs = f"{sum(fix) / len(fix):.4f} ({len(fix)})" if fix else "-"
    print(f"  {name:14s}   {len(nat):>6} step   {a:.4f} │ {b:.4f} │ {c(f'{dlt:+.4f}', dc)}      {fs}")
n_fix = sum(1 for s in run if is_fixed(s))
print(c(f"  v4 step 지금까지 {n_fix} / {len(run)} ({100 * n_fix / max(len(run), 1):.1f} %) · 전체 예정 ≈ {int(round(total * p_fix))}"
        f" · 기준 run 의 같은 step 은 자연 영상 (그 step 은 비교에서 뺐다)", DIM))

cks = sorted(glob.glob(f"{rd}/e*.pt"), key=os.path.getmtime)
if cks:
    print(c(f"  체크포인트 {', '.join(os.path.basename(x)[:-3] for x in cks[-6:])} (마지막 {kst(os.path.getmtime(cks[-1]))} KST)", DIM))
bad = [ln for ln in tail.splitlines() if re.search(r"Traceback|Error|out of memory|nan loss|loss nan", ln)]
if bad: print(c(f"  ⚠ {bad[-1][:160]}", R))

print(c("─" * 84, DIM))
g = [x.split(", ") for x in GPU.strip(";").split(";") if x]
if g:
    print("GPU  " + "  ".join(f"{i}:{int(u):>3}% {int(m) / 1024:4.1f}G" for i, m, t, u in g[:4]))
    if len(g) > 4: print("     " + "  ".join(f"{i}:{int(u):>3}% {int(m) / 1024:4.1f}G" for i, m, t, u in g[4:]))

print(c("─" * 84, DIM))
if os.path.exists(EVAL_LOG):
    lines = [x for x in open(EVAL_LOG, errors="ignore").read().splitlines() if x.strip()]
    dn = next((i for i, x in enumerate(lines) if x.startswith("CHAIN_FULL_EVAL_DONE")), None)
    stage = [x for x in lines if x.startswith("[chain_full_eval]")]
    print(c("채점 chain", B), c(os.path.relpath(EVAL_LOG, PROJ), DIM))
    if dn is not None:
        print(c(f"  {lines[dn]}", G))
        # op_cells_vob3 · op_cells_vast 표 두 개에서 전체 칸만 골라 한 표로 (칸 전부는 로그에)
        want = ["panel:prim:overall", "panel:real:overall", "mover:prim:overall", "mover:real:overall",
                "ip1:overall", "ip2:overall", "rwp:A", "rwp:B", "rwp:overall"]
        short = ["VOB 판P", "VOB 판R", "VOB 움P", "VOB 움R", "IP1", "IP2", "rw A", "rw B", "rw 전체"]
        rows, hdr = {}, None
        for x in lines[:dn]:
            cs = [s.strip() for s in x.split(" | ")]
            if cs[0] == "model": hdr = cs; continue
            if hdr and len(cs) == len(hdr) and cs[0] != "n pairs":
                rows.setdefault(cs[0], {}).update(dict(zip(hdr[1:], cs[1:])))
        if rows:
            print("  " + f"{'':30s}" + "".join(f"{h:>8s}" for h in short))
            for k, v in rows.items():
                print("  " + f"{k[:30]:30s}" + "".join(f"{v.get(w, '-'):>8s}" for w in want))
            print(c("  VOB = VOBench v3 (판 = 가림판 · 움 = 움직이는 가림막, P = primitive · R = realistic) · IP1/IP2 late · rw = real world (A 사라짐 · B 나타남)", DIM))
    else:
        if stage: print(f"  {stage[-1][18:][:140]}")
        if lines and lines[-1] not in stage: print(c(f"  … {lines[-1][:140]}", DIM))
else:
    print(c(f"채점 chain 로그 없음: {EVAL_LOG}", DIM))
PYEOF
