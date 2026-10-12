#!/bin/bash
# =============================================================================
# AR (kind=ar, 가변 창) 학습 + chain_ar 모니터 (2026-10-11).
#   watch -c -n 2 bash /workspace/world_model/vjepa2/z_training/monitor_ar.sh            # 지금 도는 run 자동 선택
#   watch -c -n 2 bash /workspace/world_model/vjepa2/z_training/monitor_ar.sh natural_ar_vw
# 화면: monitor.sh (GPU·진행) + ETA (중앙값 s/step) · tf/ar 손실 · 힌트 수 · (C,K) 별 ar 손실 · in-loop val · chain · 채점 요약
# =============================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=${1:-}
if [[ -z "$RUN" ]]; then      # 지금 도는 train_ 세션 (없으면 가장 최근 run)
  RUN=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' | head -1 | sed 's/^train_//')
  [[ -z "$RUN" ]] && RUN=$(ls -t "$ROOT/z_training/runs" | grep -v '^_' | head -1)
fi
D="$ROOT/z_training/runs/$RUN"
PY=/venv/vjepa/bin/python
[[ -x "$PY" ]] || PY=python3

# 기본 monitor.sh 의 '설정' 줄은 고정 batch·창을 가정해 가변 창 run 에서 틀린 값을 낸다 → 빼고 아래에서 다시 쓴다
bash "$ROOT/z_training/monitor.sh" "$RUN" 2>/dev/null | grep -v '^ ETA ' | grep -v '^ 설정 ' | grep -v '^ val ' | grep -v '^ C 별 '

"$PY" - "$D" "$ROOT" "$RUN" <<'EOF'
import csv, json, os, re, statistics as st, sys, subprocess
from datetime import datetime, timedelta, timezone
D, ROOT, RUN = sys.argv[1:4]
B, Y, N = "\033[1m", "\033[33m", "\033[0m"
print("─" * 78)
f = os.path.join(D, "log_r0.csv")
total = None
try:
    import yaml
    c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
    total = int(c["optimization"]["ipe"]) * int(c["optimization"]["epochs"])
    hint_cfg = c.get("hint")
except Exception:
    hint_cfg = None
try:
    vw = c["data"]["var_window"]
    print(f" {B}설정{N}  가변 창 (fps 4, 패딩 없음) · 문맥 C {vw['context_frames']} 장 × 예측 K {vw['predict_frames']} 장 · "
          f"rank 당 B = {vw['token_budget']} // (C+K) [{vw.get('b_min')}~{vw.get('b_max')}] · "
          f"데이터 {'/'.join(d['name'] for d in vw['datasets'])} 1:1:1 · lr {c['optimization']['lr']} (WSD, 마지막 {c['optimization'].get('anneal_epochs', 0)} ep 감소)")
except Exception:
    pass
if os.path.isfile(f):
    # 재개하면 csv 에 헤더가 다시 붙는다 → 숫자 줄만. 진행 step 은 train.log 의 마지막 [epoch, itr] 로 센다
    rows = [r for r in csv.DictReader(open(f)) if r.get("iter-time(ms)", "").replace(".", "", 1).isdigit()]
    done = len(rows)
    try:
        ipe = int(c["optimization"]["ipe"])
        last = re.findall(r"\] \[(\d+), +(\d+)\] loss", open(os.path.join(D, "train.log"), errors="ignore").read())[-1]
        done = (int(last[0]) - 1) * ipe + int(last[1]) + 1
    except Exception:
        pass
    it = [float(r["iter-time(ms)"]) / 1000 for r in rows[-200:]]
    if it and total:
        med = st.median(it); left = (total - done) * med
        eta = (datetime.now(timezone(timedelta(hours=9))) + timedelta(seconds=left)).strftime("%m-%d %H:%M KST")
        print(f" {B}ETA{N}   {int(left // 3600)}:{int(left % 3600 // 60):02d} 남음  (중앙값 {med:.2f} s/step)"
              f"   끝 ≈ {B}{eta}{N}   [{done:,}/{total:,} step, {100 * done / total:.1f}%]")
else:
    print(" (아직 step 로그 없음)")

tl = os.path.join(D, "train.log")
pat = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\](?: hint (\d+))? \| mask var C(\d+)K(\d+) (\w+) B(\d+)")
seen = []
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen.append(m.groups())
if seen:
    l, tf, ar, R, hn, C, K, ds, Bb = seen[-1]
    h = f"   힌트 {B}{hn}{N}/256 개" if hn is not None else ""
    print(f" {B}loss{N}  {float(l):.4f}  = tf {float(tf):.4f} + ar {float(ar):.4f}   (지금 C{C}K{K} {ds} B{Bb}, R={R}){h}")
    if hint_cfg and total:
        a, b = float(hint_cfg.get("start", 0.05)), float(hint_cfg.get("end", 0.9))
        print(f"       힌트 일정: t {a:.2f}~{b:.2f} sin² 종 모양, 정점 {round(float(hint_cfg.get('ratio_max', 0.1)) * 256)} 개 (t {(a + b) / 2:.3f})")
    ck = {}
    for g in seen[-400:]:
        ck.setdefault((int(g[5]), int(g[6])), []).append(float(g[2]))
    s = "  ".join(f"C{c}K{k}:{st.mean(v[-5:]):.3f}" for (c, k), v in sorted(ck.items()))
    print(f" (C,K) 별 최근 ar 손실\n   {s}")

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
            vals.append((r["epoch"], 100 * v["acc"]))
if vals:
    print(f" {B}val{N} IntPhys1 dev 60 쌍 (rollout, 힌트 없음, 참고용)  " + "  ".join(f"e{e}:{a:.1f}" for e, a in vals[-8:]))

cl = os.path.join(ROOT, "z_training/logs/chain_ar.log")
alive = subprocess.run(["tmux", "has-session", "-t", "=chain_ar"], capture_output=True).returncode == 0
last = open(cl).read().strip().splitlines()[-1] if os.path.isfile(cl) and os.path.getsize(cl) else "(로그 없음)"
print(f" {B}chain_ar{N} {'살아 있음' if alive else '세션 없음'} | {last[:90]}")
import glob as _g
for r in sorted(os.path.basename(os.path.dirname(os.path.dirname(x))) for x in _g.glob(os.path.join(ROOT, "z_training/runs/natural_ar_vw*/eval/BENCH_latest.txt"))):
    bf = os.path.join(ROOT, "z_training/runs", r, "eval", "BENCH_latest.txt")
    if os.path.isfile(bf):
        print(f" {B}채점{N} {r}:")
        for ln in open(bf).read().strip().splitlines()[1:]:
            print("   " + ln)
print("━" * 78)
EOF
