#!/bin/bash
# =============================================================================
# 상태 판독기 사전학습 (state_head.head_only) 모니터 (2026-10-26).
#
#   watch -c -n 5 bash z_training/monitor_head.sh [run]        # 기본 run = phys3d_state_head_pretrain
#
# 보이는 것: 진행 · ETA (KST) · 학습 batch 의 위치 오차 (정답 토큰, 보이는 블록, px) 최근 / epoch 별 · 손실 · GPU ·
#            끝난 뒤 검증 영상 평가 (z_training/logs/head_pretrain_eval.log, state_head_transfer.py) 가 있으면 그 표.
# 기준: 이전 gthead 판독기 (predictor 와 같이 돌며 정답 토큰에서 학습, e20) 검증 위치 오차 16 px.
# =============================================================================
RUN=${1:-phys3d_state_head_pretrain}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/z_training/runs/$RUN
EVAL=$ROOT/z_training/logs/head_pretrain_eval.log
/venv/vjepa/bin/python - "$R" "$RUN" "$EVAL" <<'EOF'
import os, re, sys, subprocess, datetime, yaml, statistics as st
R, RUN, EVAL = sys.argv[1:4]
B, G, Y, C, D, X = "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"
def bar(f, n=40): f = max(0.0, min(1.0, f)); k = int(round(f * n)); return "█" * k + "░" * (n - k)
line = "━" * 78
print(line); print(f" {B}상태 판독기 사전학습 모니터{X}   run {C}{RUN}{X}   {datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9):%Y-%m-%d %H:%M:%S} KST"); print(line)
if not os.path.exists(f"{R}/train.log"):
    print(" train.log 없음"); sys.exit()
cfg = yaml.safe_load(open(f"{R}/config.yaml"))
E, ipe = int(cfg["optimization"]["epochs"]), int(cfg["optimization"]["ipe"])
alive = subprocess.run(["tmux", "has-session", "-t", f"train_{RUN}"], capture_output=True).returncode == 0
pat = re.compile(r"\]\[(\d+-\d+-\d+ \d+:\d+:\d+)\]\[.*?\] \[(\d+),\s*(\d+)\] loss ([\d.]+).*? posx ([\d.na]+)/.*?\| iter (\d+) ms")
rows = []
for ln in open(f"{R}/train.log", errors="ignore"):
    m = pat.search(ln)
    if m and m.group(5)[0].isdigit():
        rows.append((m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))))
if not rows:
    print(" 아직 step 로그 없음"); sys.exit()
_, ep, it, loss, px, ms = rows[-1]
done = (ep - 1) * ipe + it + 1; tot = E * ipe
print(f" 상태  {G + 'RUNNING' + X if alive else Y + '멈춤 / 끝' + X}")
print(f" epoch {ep:3d}/{E}   {bar(ep / E)} {100 * ep / E:5.1f}%")
print(f" step  {done:5d}/{tot}  {bar(done / tot)} {100 * done / tot:5.1f}%")
med_ms = st.median([r[5] for r in rows[-100:]])
eta = (tot - done) * med_ms / 1000              # iter ms = step 당 시간
end = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9, seconds=eta)
print(f" ETA   {int(eta // 3600)}:{int(eta % 3600 // 60):02d} 남음  (step {med_ms / 1000:.2f} s)   끝 ≈ {end:%H:%M} KST")
print("─" * 78)
rec = rows[-100:]
print(f" {B}위치 오차 (학습 batch, 정답 토큰, 보이는 블록){X}   최근 100 줄 중앙값 {B}{st.median(r[4] for r in rec):.1f} px{X}   지금 {px:.1f} px")
print(f" 손실   최근 중앙값 {st.median(r[3] for r in rec):.4f}   지금 {loss:.4f}")
by = {}
for r in rows:
    by.setdefault(r[1], []).append(r[4])
print(f" {B}epoch 별 위치 오차 중앙값 (px){X}   {D}기준: 이전 gthead 판독기 검증 16 px{X}")
mx = max(st.median(v) for v in by.values())
for e in sorted(by):
    v = st.median(by[e]); print(f"   e{e:<3d} {bar(v / max(mx, 1e-9), 30)} {v:6.1f}")
print("─" * 78)
TSV = f"{R}/head_eval.tsv"
print(f" {B}검증 영상 정답 토큰, 출력별{X}  {D}(학습에 안 쓴 clip, 보이는 블록 · 체크포인트마다 head_ckpt_eval.py 가 채움){X}")
if os.path.exists(TSV) and len(open(TSV).read().splitlines()) > 1:
    print(f"   {'epoch':>5s} │ {'물체: 존재 %':>11s} {'정체 %':>7s} │ {'위치: 오차 px (중앙/평균)':>22s} │ {'속도: 방향 cos':>13s} {'속력 L1':>8s}")
    for l in open(TSV).read().splitlines()[1:]:
        c = l.split("\t")
        print(f"   {c[1]:>5s} │ {c[2]:>11s} {c[3]:>7s} │ {c[4] + ' / ' + c[5]:>22s} │ {c[6]:>13s} {c[7]:>8s}")
    print(f"   {D}기준 (이전 gthead 판독기 e20, 검증 정답 토큰): 위치 16.3~16.7 px · 방향 cos 0.99{X}")
else:
    print("   아직 없음 — 첫 체크포인트 (e2) 저장 뒤 몇 분")
print("─" * 78)
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    for o in out:
        i, u, t, g = [s.strip() for s in o.split(",")]
        print(f" GPU{i} {bar(float(u) / float(t), 24)} {float(u) / 1024:5.1f}/{float(t) / 1024:.1f} G  util {g}%")
except Exception:
    pass
if os.path.exists(EVAL) and os.path.getmtime(EVAL) > os.path.getmtime(f"{R}/config.yaml"):
    print("─" * 78); print(f" {B}검증 영상 평가 (학습에 안 쓴 clip){X}")
    txt = open(EVAL, errors="ignore").read()
    if "같은 state head" in txt:
        blk = txt[txt.index("같은 state head"):].splitlines()
        for l in blk[:4]:
            print("  " + l.strip())
    else:
        last = [l for l in txt.splitlines() if l.strip()][-1:] or ["시작 전"]
        print(f"  진행 중: {last[0][:70]}")
print(line)
EOF
