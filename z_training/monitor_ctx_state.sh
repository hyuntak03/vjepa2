#!/bin/bash
# =============================================================================
# ctx_state (rollout) 학습 전용 모니터 (2026-09-30). 기존 monitor.sh 화면 + 이 팔에 필요한 것:
#   · ETA 를 **중앙값** s/step 으로 (첫 step 들의 compile 스파이크가 평균을 부풀린다)
#   · (C, K) 조합별 최근 손실 (rollout 손실), tf/ar 분리
#   · in-loop val (IntPhys1 dev 60 쌍, rollout 주지표 / tf 보조) 추이
#   · 이어 실행될 chain (natural_tube_k710) 상태
#
#   watch -c -n 1 bash z_training/monitor_ctx_state.sh [run 이름, 기본 natural_ctx_state]
# =============================================================================
RUN=${1:-natural_ctx_state}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$ROOT/z_training/runs/$RUN"
PY=/venv/vjepa/bin/python
[[ -x "$PY" ]] || PY=python3

bash "$ROOT/z_training/monitor.sh" "$RUN" 2>/dev/null | grep -v '^ ETA '

"$PY" - "$D" "$ROOT" <<'EOF'
import csv, json, os, re, statistics as st, sys, time, subprocess
D, ROOT = sys.argv[1], sys.argv[2]
B, R, G, Y, N = "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
line = "─" * 78
print(line)
f = os.path.join(D, "log_r0.csv")
if not os.path.isfile(f):
    print(" (아직 step 로그 없음)"); sys.exit(0)
rows = list(csv.DictReader(open(f)))
cfg_total = None
try:
    import yaml
    c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
    ipe, ep = int(c["optimization"]["ipe"]), int(c["optimization"]["epochs"])
    cfg_total = ipe * ep
except Exception:
    pass
done = len(rows)
if rows:
    it = [float(r["iter-time(ms)"]) / 1000 for r in rows[-100:]]
    med = st.median(it)
    left = (cfg_total - done) * med if cfg_total else None
    from datetime import datetime, timedelta, timezone
    eta = (datetime.now(timezone(timedelta(hours=9))) + timedelta(seconds=left)).strftime("%m-%d %H:%M KST") if left else "?"
    h = int(left // 3600) if left else 0; m = int((left % 3600) // 60) if left else 0
    print(f" {B}ETA{N}   {h}:{m:02d} 남음  (중앙값 {med:.2f} s/step, 최근 {len(it)} step)   끝나는 시각 ≈ {B}{eta}{N}"
          + (f"   [{done:,}/{cfg_total:,} step]" if cfg_total else ""))

# (C, K) 별 최근 rollout 손실 — train.log 의 10 step 마다 찍힌 줄에서
tl = os.path.join(D, "train.log")
ck, tfar = {}, []
if os.path.isfile(tl):
    pat = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\] \| mask prefix C(\d+)K(\d+)")
    seen = []
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen.append(m.groups())
    for g in seen[-200:]:
        ck.setdefault((int(g[4]), int(g[5])), []).append(float(g[2]))
    if seen:
        l, tf, ar, R, C, K = seen[-1]
        print(f" {B}rollout{N} 지금 ar {float(ar):.4f}  tf {float(tf):.4f}  (C{C}K{K}, R={R})")
if ck:
    s = "  ".join(f"C{c}K{k}:{st.mean(v[-5:]):.3f}" for (c, k), v in sorted(ck.items()))
    print(f" (C,K) 별 최근 ar 손실  {s}")

# val 추이 (metrics.jsonl)
mp = os.path.join(D, "metrics.jsonl")
vals = []
if os.path.isfile(mp):
    for ln in open(mp):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        v = r.get("val") or {}
        if "acc" in v:
            vals.append((r["epoch"], 100 * v["acc"], v.get("margin", 0.0)))
at0 = None
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = re.search(r"\[epoch 0\] .*?acc ([\d.]+)%", ln)
        if m:
            at0 = float(m.group(1)); break
if vals or at0 is not None:
    s = (f"e0:{at0:.1f}  " if at0 is not None else "") + "  ".join(f"e{e}:{a:.1f}" for e, a, _ in vals[-8:])
    best = max(vals, key=lambda x: x[1]) if vals else None
    print(f" {B}val{N} IntPhys1 dev (60 쌍, rollout)  {s}" + (f"   최고 e{best[0]} {best[1]:.1f}%" if best else ""))
else:
    print(" val  (첫 epoch 끝나면 찍힌다)")

# chain 상태
cl = os.path.join(ROOT, "z_training/logs/chain_natural_tube_k710.log")
alive = subprocess.run(["tmux", "has-session", "-t", "chain_tube"], capture_output=True).returncode == 0
last = open(cl).read().strip().splitlines()[-1] if os.path.isfile(cl) else "(로그 없음)"
print(f" {B}chain{N} (2) natural_tube_k710: {'대기 중' if alive else '세션 없음'}  | {last[:60]}")
print("━" * 78)
EOF
