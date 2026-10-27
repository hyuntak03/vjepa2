#!/bin/bash
# =============================================================================
# 블록 AR + register 학습 모니터 (2026-10-27).
#
#   watch -c -n 10 bash z_training/monitor_reg.sh [run] [기준 run]
#     기본 run = natural_ar_vw_state_reg4 · 기준 = natural_ar_vw_state (register 없는 같은 설정)
#
#   ① 진행 · ETA (KST) · 메모리 최고 (OOM 위험)
#   ② latent 손실 (TF / 굴린) — epoch 별 중앙값을 기준 run 과 같은 epoch 끼리 나란히 (자연 영상 · 같은 데이터라 비교 가능)
#   ③ VOBench late val (epoch 별) — 같은 val 을 쓰는 판독기 run 들과 나란히 (기준 AR 의 학습 중 val 은 다른 세트라 뺀다)
#   ④ GPU
# =============================================================================
RUN=${1:-natural_ar_vw_state_reg4}
BASE=${2:-natural_ar_vw_state}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
/venv/vjepa/bin/python - "$ROOT/z_training/runs" "$RUN" "$BASE" <<'EOF'
import os, re, sys, json, subprocess, datetime, statistics as st, yaml, unicodedata
RR, RUN, BASE = sys.argv[1:4]
R = os.path.join(RR, RUN)
B, DIM, X = "\033[1m", "\033[2m", "\033[0m"
RED, GRN, YEL, BLU, MAG, CYN, GRY = "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[90m"
W = 92
now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9)
def wlen(s):
    out, esc = 0, False
    for ch in s:
        if ch == "\033": esc = True
        elif esc:
            if ch == "m": esc = False
        else: out += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return out
def pad(s, n, right=False): s = str(s); f = " " * max(0, n - wlen(s)); return f + s if right else s + f
def bar(f, n=30, c=CYN): f = max(0.0, min(1.0, f)); k = int(round(f * n)); return c + "█" * k + GRY + "·" * (n - k) + X
def med(xs): xs = [x for x in xs if x == x]; return st.median(xs) if xs else float("nan")
def head(t): print(f"\n {B}{BLU}▌{X}{B} {t}{X}")
def f(x, d=3): return "—" if not (isinstance(x, (int, float)) and x == x) else f"{x:.{d}f}"

cfg = yaml.safe_load(open(f"{R}/config.yaml"))
E, ipe = int(cfg["optimization"]["epochs"]), int(cfg["optimization"]["ipe"])
pc = cfg["model"]["predictor"]
alive = subprocess.run(["tmux", "has-session", "-t", f"train_{RUN}"], capture_output=True).returncode == 0
print(f"{BLU}╔{'═' * (W - 2)}╗{X}")
print(f"{BLU}║{X}{pad(f' {B}블록 AR + register 학습{X}   {CYN}{RUN}{X}', W - 2)}{BLU}║{X}")
t2 = (f" {(GRN + '● RUNNING') if alive else (YEL + '○ 멈춤 / 끝')}{X}   {DIM}register {pc.get('n_registers', 0)} 개 · 넘김 {pc.get('reg_carry', 'state')}"
      f" · 되먹임 {pc.get('ar_feedback')} · 기준 {BASE}{X}")
print(f"{BLU}║{X}{pad(t2, W - 2)}{BLU}║{X}")
print(f"{BLU}║{X}{pad(f' {DIM}{now:%Y-%m-%d %H:%M:%S} KST{X}', W - 2)}{BLU}║{X}")
print(f"{BLU}╚{'═' * (W - 2)}╝{X}")

def latent(run, cut=None):        # train.log → {epoch: (tf 중앙, ar 중앙)} · 메모리 최고 · 마지막 줄. cut = (epoch, itr): 그 epoch 은 itr 까지만
    d, mem, last = {}, 0.0, None
    p = os.path.join(RR, run, "train.log")
    if not os.path.exists(p): return d, mem, last
    for l in open(p, errors="ignore"):
        m = re.search(r"\] \[(\d+),\s*(\d+)\] loss \S+ \(avg \S+\) \[tf ([\d.]+) ar ([\d.]+)", l)
        if m and not (cut and int(m.group(1)) == cut[0] and int(m.group(2)) > cut[1]):
            d.setdefault(int(m.group(1)), []).append((float(m.group(3)), float(m.group(4)))); last = l
        mm = re.search(r"mem ([\d.]+)G", l)
        if mm: mem = max(mem, float(mm.group(1)))
    return {e: (med([a for a, _ in v]), med([b for _, b in v])) for e, v in d.items()}, mem, last

rows = [l.split(",") for l in open(f"{R}/log_r0.csv").read().splitlines()[1:] if l and l[0].isdigit()] if os.path.exists(f"{R}/log_r0.csv") else []
LR, mem, last = latent(RUN)
if not rows:
    print("\n   아직 step 없음 (데이터 · 모델 준비 중)"); sys.exit()
ep, it = int(rows[-1][0]), int(rows[-1][1]); done = (ep - 1) * ipe + it + 1; tot = E * ipe
ms = med([float(r[5]) for r in rows[-200:]]); eta = (tot - done) * ms / 1000
head("진행")
print(f"   epoch  {bar(ep / E, 40)}  {B}{ep}{X}/{E}")
print(f"   step   {bar(done / tot, 40)}  {B}{done:,}{X}/{tot:,}  {DIM}({100 * done / tot:.1f}%){X}")
print(f"   남은 시간 {B}{int(eta // 3600)}:{int(eta % 3600 // 60):02d}{X}  →  끝 {B}{now + datetime.timedelta(seconds=eta):%m-%d %H:%M}{X} KST"
      f"   {DIM}step {ms / 1000:.2f} s (최근 200 중앙) · lr {float(rows[-1][4]):.2e}{X}")
mc = RED if mem > 92 else YEL if mem > 88 else GRN
print(f"   메모리 최고 (rank 0) {mc}{B}{mem:.1f} GB{X} / 95.6  {DIM}(기준 run 최고 {latent(BASE)[1]:.1f} GB · 92 넘으면 위험){X}")
if last:
    m = re.search(r"\[tf ([\d.]+) ar ([\d.]+).*?mask var (\S+) (\S+) B(\d+)", last)
    if m: print(f"   {DIM}지금 step: TF {m.group(1)} · 굴린 {m.group(2)} · 창 {m.group(3)} · {m.group(4)} · B {m.group(5)}{X}")

head(f"latent 손실 (epoch 중앙, 자연 영상)  {DIM}TF / 굴린 · 기준 = register 없는 같은 설정{X}")
LB = latent(BASE, cut=(ep, it))[0]       # 진행 중 epoch 은 같은 step 까지만 비교
print(f"   {pad('epoch', 7)} {pad('이 run TF', 10, True)} {pad('기준 TF', 10, True)} {pad('차이', 8, True)}   {pad('이 run 굴린', 12, True)} {pad('기준 굴린', 10, True)} {pad('차이', 8, True)}")
print(f"   {GRY}{'─' * 76}{X}")
show = sorted(LR)
show = show[:3] + [e for e in show[3:] if e % 5 == 0 or e >= show[-1] - 2] if len(show) > 12 else show
if not LR: print("   아직 로그 줄 없음")
for e in show:
    a, b = LR[e]; ba, bb = LB.get(e, (float("nan"), float("nan")))
    da, db = a - ba, b - bb
    ca = GRN if da < -0.003 else RED if da > 0.003 else GRY
    cb = GRN if db < -0.003 else RED if db > 0.003 else GRY
    print(f"   {pad('e' + str(e), 7)} {B}{f(a):>10s}{X} {f(ba):>10s} {ca}{f(da):>8s}{X}   {B}{f(b):>12s}{X} {f(bb):>10s} {cb}{f(db):>8s}{X}")
print(f"   {DIM}초록 = 이 run 이 더 낮음 (좋음) · 빨강 = 더 높음 · 진행 중 epoch 은 기준도 같은 step ({it}) 까지만{X}")

def val(run):
    p = os.path.join(RR, run, "metrics.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); v = d.get("val")
            if v and v.get("n_pair") == 432:
                bt = v.get("by_type", {}); g = lambda k: next((100 * x["acc"] for t, x in bt.items() if k in t), float("nan"))
                out[d["epoch"]] = (100 * v["acc"], g("permanence"), g("shape"), g("color"), v["margin"])
    return out
V = val(RUN); REF = [("natural_ar_vw_state_phys3d_event_gthead_frozen_gm", "판독기 크기맞춤"), ("natural_ar_vw_state_phys3d_event", "3D 감독")]
VR = {n: val(r) for r, n in REF}
head(f"VOBench late val  {DIM}(432 쌍 · 학습 중 고른 val 이라 held-out 아님 · 기준 AR 의 학습 중 val 은 다른 세트라 없음){X}")
if V:
    print(f"   {pad('epoch', 7)} {pad('이 run', 26)} {pad('영속', 6, True)} {pad('모양', 6, True)} {pad('색', 6, True)} {pad('margin', 8, True)}   " + "  ".join(pad(n, 12, True) for _, n in REF))
    print(f"   {GRY}{'─' * 88}{X}")
    for e in sorted(V)[-12:]:
        a, p_, s_, c_, m_ = V[e]
        print(f"   {pad('e' + str(e), 7)} {bar(a / 100, 18)} {B}{a:5.1f}{X}  {p_:6.1f} {s_:6.1f} {c_:6.1f} {m_:8.4f}   " +
              "  ".join(pad(f(VR[n].get(e, (float('nan'),))[0], 1), 12, True) for _, n in REF))
else:
    print("   아직 없음 (epoch 1 끝에 첫 val)")
head("GPU")
gl = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True).stdout.strip().splitlines()
cells = []
for o in gl:
    i, u, t, g = [s.strip() for s in o.split(",")]
    cells.append(f"{i} {bar(float(u) / float(t), 8, GRN if int(g) > 50 else YEL)} {float(u) / 1024:4.0f}G {int(g):3d}%")
for k in range(0, len(cells), 4):
    print("   " + "   ".join(cells[k:k + 4]))
print(f"{BLU}{'─' * W}{X}")
EOF
