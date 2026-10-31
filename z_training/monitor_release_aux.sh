#!/bin/bash
# =============================================================================
# 고정 릴리즈 predictor + 보조 기억 모듈 학습 모니터 (2026-10-30, monitor_release_postft.sh 에서 갈라짐).
#
#   watch -c -n 10 bash z_training/monitor_release_aux.sh [run]
#     기본 run = release_aux_perm (configs/training/release_aux_perm.yaml)
#
#   ① 진행 · ETA (KST) · lr · 메모리 최고
#   ② 미래 토큰 latent L1 (epoch 중앙) · 첫 step (≈ 학습 전 릴리즈, gate 0) 대비
#   ③ gate (층별, epoch 끝) — 0 근처면 모듈을 안 쓰는 것
#   ④ VOBench late val (432 쌍, epoch 별) — e0 = 학습 전 (= 릴리즈)
#   ⑤ GPU
# =============================================================================
RUN=${1:-release_aux_perm}
BASE=${2:-none}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
/venv/vjepa/bin/python - "$ROOT/z_training/runs" "$RUN" "$BASE" <<'EOF'
import os, re, sys, json, subprocess, datetime, statistics as st, yaml, unicodedata
RR, RUN, BASE = sys.argv[1:4]
R = os.path.join(RR, RUN)
B, DIM, X = "\033[1m", "\033[2m", "\033[0m"
RED, GRN, YEL, BLU, CYN, GRY = "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[36m", "\033[90m"
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
def f(x, d=4): return "—" if not (isinstance(x, (int, float)) and x == x) else f"{x:.{d}f}"

if not os.path.exists(f"{R}/config.yaml"):
    print(f"아직 run 폴더가 없다: {R}"); sys.exit()
cfg = yaml.safe_load(open(f"{R}/config.yaml"))
E = int(cfg["optimization"]["epochs"]); ipe = cfg["optimization"].get("ipe")
if ipe is None:                                  # ipe: null = 로더 길이 → train.log 의 "ipe N" 에서 (2026-10-30)
    _m = re.search(r"\| ipe (\d+)", open(f"{R}/train.log", errors="ignore").read()) if os.path.exists(f"{R}/train.log") else None
    ipe = int(_m.group(1)) if _m else 1
ipe = int(ipe)
alive = subprocess.run(["tmux", "has-session", "-t", f"train_{RUN}"], capture_output=True).returncode == 0
ds = " · ".join(d["name"] for d in cfg["data"]["datasets"])
print(f"{BLU}╔{'═' * (W - 2)}╗{X}")
print(f"{BLU}║{X}{pad(f' {B}고정 릴리즈 + 보조 기억 모듈{X}   {CYN}{RUN}{X}', W - 2)}{BLU}║{X}")
print(f"{BLU}║{X}{pad(f' {(GRN + chr(9679) + ' RUNNING') if alive else (YEL + chr(9675) + ' 멈춤 / 끝')}{X}   {DIM}{ds} · 기준 {BASE}{X}', W - 2)}{BLU}║{X}")
print(f"{BLU}║{X}{pad(f' {DIM}{now:%Y-%m-%d %H:%M:%S} KST{X}', W - 2)}{BLU}║{X}")
print(f"{BLU}╚{'═' * (W - 2)}╝{X}")

def rows_of(run):
    p = os.path.join(RR, run, "log_r0.csv")
    return [l.split(",") for l in open(p).read().splitlines()[1:] if l and l[0].isdigit()] if os.path.exists(p) else []
def mem_of(run):
    p, m = os.path.join(RR, run, "train.log"), 0.0
    if os.path.exists(p):
        for l in open(p, errors="ignore"):
            mm = re.search(r"mem ([\d.]+)G", l)
            if mm: m = max(m, float(mm.group(1)))
    return m
rows = rows_of(RUN)
if not rows:
    print("\n   아직 step 없음 (데이터 · 모델 준비 중, 학습 전 val 이 먼저 돈다)")
else:
    ep, it = int(rows[-1][0]), int(rows[-1][1]); done = (ep - 1) * ipe + it + 1; tot = E * ipe
    ms = med([float(r[5]) for r in rows[-200:]]); eta = (tot - done) * ms / 1000
    head("진행")
    print(f"   epoch  {bar(ep / E, 40)}  {B}{ep}{X}/{E}")
    print(f"   step   {bar(done / tot, 40)}  {B}{done:,}{X}/{tot:,}  {DIM}({100 * done / tot:.1f}%){X}")
    print(f"   남은 시간 {B}{int(eta // 3600)}:{int(eta % 3600 // 60):02d}{X}  →  끝 {B}{now + datetime.timedelta(seconds=eta):%m-%d %H:%M}{X} KST"
          f"   {DIM}step {ms / 1000:.2f} s (최근 200 중앙) · lr {float(rows[-1][4]):.2e}{X}")
    mem = mem_of(RUN); mc = RED if mem > 92 else YEL if mem > 88 else GRN
    print(f"   메모리 최고 (rank 0) {mc}{B}{mem:.1f} GB{X} / 95.6")

    _sup = os.path.exists(f"{R}/train.log") and " occ " in open(f"{R}/train.log", errors="ignore").read()
    head(("전체 손실 = latent L1 + 가중 × (칸 CE · 이동량 L1) (epoch 중앙) · 성분은 아래 칸 감독 표" if _sup else "미래 토큰 latent L1 (epoch 중앙)")
         + f"  {DIM}문맥 C ∈ {{10,12,14,16}} 무작위라 step 마다 출렁인다{X}")
    def by_ep(rs, cut=None):
        d = {}
        for r in rs:
            e, i = int(r[0]), int(r[1])
            if cut and e == cut[0] and i > cut[1]: continue
            d.setdefault(e, []).append(float(r[2]))
        return {e: med(v) for e, v in d.items()}
    LR, LB = by_ep(rows), by_ep(rows_of(BASE), cut=(ep, it))
    first = float(rows[0][2])
    first20 = med([float(r[2]) for r in rows[:20]])
    print(f"   첫 20 step 중앙 (≈ 학습 전 릴리즈) {B}{first20:.4f}{X}")
    print(f"   {pad('epoch', 7)} {pad('이 run', 10, True)} {pad('첫 20 대비', 11, True)}")
    print(f"   {GRY}{'─' * 32}{X}")
    show = sorted(LR)
    show = show[:3] + [e for e in show[3:] if e % 5 == 0 or e >= show[-1] - 2] if len(show) > 12 else show
    for e in show:
        a = LR[e]; d_ = a - first20
        c = GRN if d_ < -0.003 else RED if d_ > 0.003 else GRY
        print(f"   {pad('e' + str(e), 7)} {B}{f(a):>10s}{X} {c}{f(d_):>11s}{X}")

SUP = {}                                           # 2026-10-31 aux.sup: train.log 의 "lat · occ · vis R/P · hid R/P" (rank 0, log_freq 마다)
if os.path.exists(f"{R}/train.log"):
    for l in open(f"{R}/train.log", errors="ignore"):
        m = re.search(r"\[(\d+),\s+\d+\] loss .*? lat ([\d.]+) occ ([\d.]+) vis R ([\d.na]+) P ([\d.na]+) hid R ([\d.na]+) P ([\d.na]+)", l)
        if m: SUP.setdefault(int(m.group(1)), []).append([float(x) for x in m.groups()[1:]])
DSP = {}                                           # 2026-10-31 aux.sup.disp: "disp a (0이동 b)" — 이동량 L1 (칸) · 0 이동 예측의 L1
if os.path.exists(f"{R}/train.log"):
    for l in open(f"{R}/train.log", errors="ignore"):
        m = re.search(r"\[(\d+),\s+\d+\] loss .*? disp ([\d.]+) \(0이동 ([\d.]+)\)", l)
        if m: DSP.setdefault(int(m.group(1)), []).append([float(m.group(2)), float(m.group(3))])
if SUP:
    head(f"보조 모듈 칸 감독 (epoch 중앙)  {DIM}latent L1 · 칸 CE · 보이는 물체 / 가려진 물체 칸 재현율 R · 정밀도 P (rank 0 한 배치씩){X}")
    print(f"   {pad('epoch', 7)} {pad('latent', 8, True)} {pad('CE', 8, True)} {pad('보임 R', 8, True)} {pad('보임 P', 8, True)} {pad('가려짐 R', 9, True)} {pad('가려짐 P', 9, True)}")
    print(f"   {GRY}{'─' * 64}{X}")
    ks = sorted(SUP); ks = ks[:2] + ks[-8:] if len(ks) > 10 else ks
    for e in ks:
        v = [med([r[j] for r in SUP[e]]) for j in range(6)]
        dd = f"   이동 L1 {med([r[0] for r in DSP[e]]):.3f} (0이동 {med([r[1] for r in DSP[e]]):.3f})" if e in DSP else ""
        print(f"   {pad('e' + str(e), 7)} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.2f} {v[3]:8.2f} {v[4]:9.2f} {v[5]:9.2f}{dd}")

def val(run):
    p = os.path.join(RR, run, "metrics.jsonl"); out = {}
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); v = d.get("val")
            if v and v.get("n_pair") == 432:
                bt = v.get("by_type", {}); g = lambda k: next((100 * x["acc"] for t, x in bt.items() if k in t), float("nan"))
                out[d["epoch"]] = (100 * v["acc"], g("permanence"), g("shape"), g("color"), v["margin"])
    p = os.path.join(RR, run, "train.log")           # 학습 전 val (e0) 은 metrics.jsonl 에 안 남는다 → train.log 에서 (종류별 값은 없다)
    if os.path.exists(p):
        for l in open(p, errors="ignore"):
            m = re.search(r"\[epoch (\d+)\] val acc ([\d.]+)% \(n_pair 432.*?margin ([+-][\d.]+)", l)
            if m and int(m.group(1)) not in out:
                out[int(m.group(1))] = (float(m.group(2)), float("nan"), float("nan"), float("nan"), float(m.group(3)))
    return out
G = []
if os.path.exists(os.path.join(R, "metrics.jsonl")):
    for l in open(os.path.join(R, "metrics.jsonl")):
        d = json.loads(l)
        if d.get("aux_gates"): G.append((d["epoch"], d["aux_gates"]))
head(f"gate (층별, epoch 끝)  {DIM}0 = 릴리즈 그대로 · 크기가 커질수록 모듈을 더 쓴다{X}")
if G:
    ks = sorted(G[-1][1], key=int)
    print(f"   {pad('epoch', 7)} " + " ".join(pad('층 ' + str(int(k) + 1), 10, True) for k in ks))
    for e, g in G[-8:]:
        print(f"   {pad('e' + str(e), 7)} " + " ".join(f"{g[k]:10.4f}" for k in ks))
else:
    print("   아직 없음 (epoch 1 끝에 기록)")
V = val(RUN)
head(f"VOBench late val  {DIM}(432 쌍 · in-loop 감시용, 정식 수치는 z_training/eval.sh · e0 = 학습 전 릴리즈){X}")
if V:
    print(f"   {pad('epoch', 7)} {pad('전체', 26)} {pad('영속', 6, True)} {pad('모양', 6, True)} {pad('색', 6, True)} {pad('margin', 9, True)}")
    print(f"   {GRY}{'─' * 64}{X}")
    ks = sorted(V); ks = ks[:1] + ks[-11:] if len(ks) > 12 else ks
    for e in ks:
        a, p_, s_, c_, m_ = V[e]
        print(f"   {pad('e' + str(e), 7)} {bar(a / 100, 18)} {B}{a:5.1f}{X}  {p_:6.1f} {s_:6.1f} {c_:6.1f} {m_:9.5f}")
else:
    print("   아직 없음 (학습 전 val 이 끝나면 e0 이 나온다)")

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
