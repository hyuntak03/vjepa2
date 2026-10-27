#!/bin/bash
# =============================================================================
# 판독기 고정 + predictor 학습 (state_head.freeze) 모니터 (2026-10-26).
#
#   watch -c -n 10 bash z_training/monitor_frozen.sh [run]     # 기본 run = natural_ar_vw_state_phys3d_event_gthead_frozen
#
# 읽는 곳: <run>/monitor.jsonl (state_head.monitor_json — 로그 step 마다 모니터 값 전부) · log_r0.csv (진행) · metrics.jsonl (epoch 별 VOBench val)
#   ① 진행 · ETA (KST)
#   ② 손실 — 합성 latent · 판독기 (고정) 손실, 추이
#   ③ 고정 판독기로 읽은 상태 — 정답 토큰 (바닥) / TF 예측 / 굴린 예측
#   ④ predictor 로 가는 gradient 크기 (autograd, 손실 항별) — 크기 막대 (log) · latent 대비 배수 · cos · 판독기 안 성분 비율 · 추이
#   ⑤ epoch 별 VOBench late val · GPU
# =============================================================================
RUN=${1:-natural_ar_vw_state_phys3d_event_gthead_frozen_gm}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
/venv/vjepa/bin/python - "$ROOT/z_training/runs/$RUN" "$RUN" <<'EOF'
import os, sys, json, subprocess, datetime, statistics as st, yaml, math, unicodedata
R, RUN = sys.argv[1:3]
B, DIM, X = "\033[1m", "\033[2m", "\033[0m"
RED, GRN, YEL, BLU, MAG, CYN, GRY = "\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[90m"
W = 92
now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9)


def wlen(s):          # 터미널 칸 수 (한글 2 칸, 색 코드 0 칸)
    out, esc = 0, False
    for ch in s:
        if ch == "\033": esc = True
        elif esc:
            if ch == "m": esc = False
        else: out += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return out
def pad(s, n, right=False): s = str(s); f = " " * max(0, n - wlen(s)); return f + s if right else s + f
def bar(f, n=30, c=CYN): f = max(0.0, min(1.0, f)); k = int(round(f * n)); return c + "█" * k + GRY + "·" * (n - k) + X
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]; return st.median(xs) if xs else float("nan")
def spark(v, c=CYN):
    v = [x for x in v if x == x and x > 0]
    if len(v) < 2: return GRY + "(기록 부족)" + X
    lo, hi = math.log10(min(v)), math.log10(max(v)); S = "▁▂▃▄▅▆▇█"
    return c + "".join(S[0 if hi == lo else int((math.log10(x) - lo) / (hi - lo) * 7)] for x in v) + X
def head(t): print(f"\n {B}{BLU}▌{X}{B} {t}{X}")
def f(x, d=1): return "—" if not (isinstance(x, (int, float)) and x == x) else f"{x:.{d}f}"

cfg = yaml.safe_load(open(f"{R}/config.yaml"))
E, ipe = int(cfg["optimization"]["epochs"]), int(cfg["optimization"]["ipe"])
SH = cfg.get("state_head", {})
alive = subprocess.run(["tmux", "has-session", "-t", f"train_{RUN}"], capture_output=True).returncode == 0

print(f"{BLU}╔{'═' * (W - 2)}╗{X}")
t1 = f" {B}판독기 고정 predictor 학습{X}   {CYN}{RUN}{X}"
print(f"{BLU}║{X}{pad(t1, W - 2)}{BLU}║{X}")
t2 = (f" {(GRN + '● RUNNING') if alive else (YEL + '○ 멈춤 / 끝')}{X}   {DIM}판독기 {SH.get('init_checkpoint', '').split('runs/')[-1].replace('/', ':')}"
      f" (고정) · 가중치 {SH.get('weight')} · grad 측정 {SH.get('grad_probe_every')} step{X}")
print(f"{BLU}║{X}{pad(t2, W - 2)}{BLU}║{X}")
print(f"{BLU}║{X}{pad(f' {DIM}{now:%Y-%m-%d %H:%M:%S} KST{X}', W - 2)}{BLU}║{X}")
print(f"{BLU}╚{'═' * (W - 2)}╝{X}")

# ---- ① 진행 ------------------------------------------------------------------------------------
rows = [l.split(",") for l in open(f"{R}/log_r0.csv").read().splitlines()[1:] if l and l[0].isdigit()] if os.path.exists(f"{R}/log_r0.csv") else []
if rows:
    ep, it = int(rows[-1][0]), int(rows[-1][1]); done = (ep - 1) * ipe + it + 1; tot = E * ipe
    ms = med([float(r[5]) for r in rows[-200:]]); eta = (tot - done) * ms / 1000
    head("진행")
    print(f"   epoch  {bar(ep / E, 40)}  {B}{ep}{X}/{E}")
    print(f"   step   {bar(done / tot, 40)}  {B}{done:,}{X}/{tot:,}  {DIM}({100 * done / tot:.1f}%){X}")
    print(f"   남은 시간 {B}{int(eta // 3600)}:{int(eta % 3600 // 60):02d}{X}  →  끝 {B}{now + datetime.timedelta(seconds=eta):%m-%d %H:%M}{X} KST"
          f"   {DIM}step {ms / 1000:.2f} s · lr {float(rows[-1][4]):.2e}{X}")
recs = [json.loads(l) for l in open(f"{R}/monitor.jsonl")] if os.path.exists(f"{R}/monitor.jsonl") else []
if not recs:
    print("\n   monitor.jsonl 아직 없음"); sys.exit()

# ---- ② 손실 ------------------------------------------------------------------------------------
last = recs[-50:]
head(f"손실  {DIM}(최근 {len(last)} 기록 중앙 · 추이는 기록마다){X}")
for k, n, c in (("syn_latent", "합성 latent", CYN), ("head", "판독기", MAG), ("loss", "전체", GRY)):
    print(f"   {pad(n, 12)} {B}{f(med([r.get(k) for r in last]), 4):>9s}{X}   {spark([r.get(k, float('nan')) for r in recs[-70:]], c)}")

# ---- ③ 판독기로 읽은 상태 ----------------------------------------------------------------------
syn = [r for r in recs if r.get("hlog")]
if syn:
    s50 = syn[-50:]; h = lambda k: med([r["hlog"].get(k) for r in s50])
    head(f"고정 판독기로 읽은 상태  {DIM}(합성 batch · 보이는 블록 · 최근 {len(s50)} 기록 중앙){X}")
    cols = (("gt", "정답 토큰 (바닥)"), ("tf", "TF 예측"), ("ar", "굴린 예측"))
    print(f"   {pad('', 14)}" + "".join(pad(n, 18, True) for _, n in cols))
    print(f"   {GRY}{'─' * 68}{X}")
    for key, n, scale, d, lower in (("pos_px_vis", "위치 오차 px", 1, 1, True), ("ex_acc", "존재 %", 100, 1, False), ("id_acc", "정체 %", 100, 1, False),
                                    ("dir", "방향 L1", 1, 3, True), ("spd", "속력 L1", 1, 3, True)):
        g0 = h("gt_" + key) * scale
        cells = []
        for a, _ in cols:
            v = h(a + "_" + key) * scale
            if a == "gt": c = DIM
            else:
                gap = (v / g0 if lower else g0 / max(v, 1e-9)) if g0 == g0 and v == v and g0 > 0 else float("nan")
                c = GRN if gap < 1.5 else (YEL if gap < 4 else RED)
            cells.append(pad(f"{c}{f(v, d)}{X}", 18, True))
        print(f"   {pad(n, 14)}" + "".join(cells))
    print(f"   {DIM}색: 정답 토큰 대비  {GRN}■{X}{DIM} 1.5 배 안  {YEL}■{X}{DIM} 4 배 안  {RED}■{X}{DIM} 그 밖{X}")

# ---- ④ gradient ------------------------------------------------------------------------------
gp = [r for r in recs if r.get("gprobe")]
head(f"predictor 로 가는 gradient ‖∇‖  {DIM}(autograd · 손실 항별 · rank 0 · 측정 {len(gp)} 회, 최근 10 회 중앙){X}")
if gp:
    g10 = gp[-10:]; q = lambda k: med([r["gprobe"].get(k) for r in g10])
    lat = q("lat")
    terms = [("nat", "자연 latent", CYN), ("lat", "합성 latent", CYN), ("head", "판독기 (가중 후)", MAG)] + \
            [(c, n, MAG) for c, n in (("pos", "  └ 위치 (전)"), ("dir", "  └ 방향 (전)"), ("spd", "  └ 속력 (전)"), ("ex", "  └ 존재정체(전)"))]
    val = {}
    for k, *_ in terms:
        val[k] = q(k) if k in ("nat", "lat", "head") else q("ar_" + k) + q("tf_" + k)
    vmax = max(v for v in val.values() if v == v)
    lo = math.log10(min(v for v in val.values() if v == v and v > 0) / 2)
    print(f"   {pad('', 14)} {pad('크기 (log 막대)', 32)} {pad('‖∇‖', 9, True)} {pad('÷ latent', 10, True)} {pad('cos(·, latent)', 16, True)}")
    print(f"   {GRY}{'─' * 84}{X}")
    for k, n, c in terms:
        v = val[k]
        fr = (math.log10(v) - lo) / (math.log10(vmax) - lo) if v == v and v > 0 else 0
        if k in ("nat", "lat"): cs = "—"
        elif k == "head": cs = f(q("cos_head"), 3)
        else: cs = f"{f(q('cos_ar_' + k), 2)} / {f(q('cos_tf_' + k), 2)}"
        rc = (RED if v / lat > 100 else YEL if v / lat > 10 else GRN) if k not in ("nat", "lat") else ""
        print(f"   {pad(n, 14)} {bar(fr, 32, c)} {pad(f(v, 3), 9, True)} {pad(rc + f(v / lat, 1) + 'x' + X, 10, True)} {pad(cs, 16, True)}")
    print(f"   {DIM}성분 = 굴린 + TF 크기 합 · 성분 cos = 굴린 / TF · 크기는 벡터 노름이라 성분 합 ≠ 전체{X}")
    hs = sum(val[c] for c in ("pos", "dir", "spd", "ex"))
    if hs > 0:
        segs = [("pos", "위치", GRN), ("dir", "방향", YEL), ("spd", "속력", CYN), ("ex", "존재·정체", MAG)]
        line = "".join(c + "█" * max(1 if val[k] > 0 else 0, int(round(60 * val[k] / hs))) + X for k, _, c in segs)
        print(f"\n   판독기 성분 크기 비율  {line}")
        print("   " + "   ".join(f"{c}■{X} {n} {100 * val[k] / hs:.0f}%" for k, n, c in segs))
    if "w_pos" in gp[-1]["gprobe"]:      # grad_match: 성분 가중치 (목표 배수 / EMA 비율) — 가중 후 ‖∇‖ ≈ 목표 × latent
        GMc = SH.get("grad_match") or {}
        print(f"\n   {B}gradient 크기 맞춤{X}  {DIM}성분 가중치 w = 목표 ÷ (‖∇ 성분‖ ÷ ‖∇ latent‖) 의 EMA · 위 표의 성분 줄은 가중 전 크기{X}")
        print("   " + "   ".join(f"{n} 목표 {GMc.get(c, 0):g}x  w {gp[-1]['gprobe'].get('w_' + c, 0):.2e}" for c, n in (("pos", "위치"), ("dir", "방향"), ("ex", "존재·정체"), ("spd", "속력"))))
        print(f"   가중 후 판독기 전체 ÷ latent = {B}{q('head') / lat:.2f}x{X}  {DIM}(목표 합 ≈ {sum(float(GMc.get(c, 0)) for c in ('pos', 'dir', 'ex', 'spd')):g}x 안팎 · 성분끼리 방향이 다르면 더 작다){X}")
    ratio = [r["gprobe"]["head"] / r["gprobe"]["lat"] for r in gp if r["gprobe"].get("lat")]
    posr = [(r["gprobe"].get("ar_pos", 0) + r["gprobe"].get("tf_pos", 0)) / r["gprobe"]["lat"] for r in gp if r["gprobe"].get("lat")]
    print(f"\n   추이 (log)  판독기 ÷ latent  {spark(ratio[-60:], MAG)}  {ratio[0]:.0f}x → {B}{ratio[-1]:.0f}x{X}")
    print(f"               위치 ÷ latent    {spark(posr[-60:], GRN)}  {posr[0]:.1f}x → {B}{posr[-1]:.1f}x{X}")
    print(f"   {DIM}predictor grad 는 합이 clip_grad {cfg['optimization'].get('clip_grad')} 을 넘으면 잘린다 — 배수가 크면 업데이트 방향은 판독기 손실이 정한다{X}")

# ---- ⑤ val · GPU ------------------------------------------------------------------------------
if os.path.exists(f"{R}/metrics.jsonl"):
    ms_ = [m for m in (json.loads(l) for l in open(f"{R}/metrics.jsonl")) if m.get("val")]
    if ms_:
        head("VOBench late val  (epoch 별)")
        print(f"   {pad('epoch', 7)} {pad('전체', 32)} {pad('영속', 7, True)} {pad('모양', 7, True)} {pad('색', 7, True)}")
        for m in ms_[-10:]:
            bt = m["val"].get("by_type", {}); g = lambda k: next((100 * x["acc"] for t, x in bt.items() if k in t), float("nan"))
            print(f"   {pad('e' + str(m['epoch']), 7)} {bar(m['val']['acc'], 22)} {B}{100 * m['val']['acc']:5.1f}{X}  {g('permanence'):7.1f} {g('shape'):7.1f} {g('color'):7.1f}")
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
