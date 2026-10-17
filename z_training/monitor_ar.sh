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
    wts = " : ".join(f"{d['name']} {d.get('weight', 1.0)}" for d in vw["datasets"])
    pc = c["model"].get("predictor") or {}
    arm = (f"kind={pc.get('kind')}" + (f" 되먹임={pc['ar_feedback']}" if pc.get("ar_feedback") else "")
           + (f" 타깃={pc['ctx_target']}" if pc.get("ctx_target") else "")
           + (f" tf_span={c['loss']['tf_span']}" if c["loss"].get("tf_span") else "")
           + (f" tf_weight={c['loss']['tf_weight']}" if "tf_weight" in c["loss"] and pc.get("kind") == "ctx_state" else ""))
    print(f" {B}설정{N}  {arm} · 가변 창 (fps 4, 패딩 없음) · 문맥 C {vw['context_frames']} 장 × 예측 K {vw['predict_frames']} 장 · "
          f"rank 당 B = {vw['token_budget']} // (C+K) [{vw.get('b_min')}~{vw.get('b_max')}] · "
          f"데이터 {wts} · lr {c['optimization']['lr']} (WSD, 마지막 {c['optimization'].get('anneal_epochs', 0)} ep 감소)")
    for key in ("fixed", "boundary"):
        if vw.get(key):
            bx = vw[key]
            print(f"       + {key} {bx.get('name', '')}: step 의 {100 * float(bx['p']):.0f} %"
                  + (f", 경계 정렬 · 문맥 {bx['context_frames']} 장 · 예측 {bx['predict_frames']} 장 · stride {bx['stride']}" if key == "boundary" else ""))
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
pat = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\](?: hint (\d+))?(?: dl [\d.]+)?(?: dyn [\d.]+)?(?: hole [\d.]+ rec [\d.]+)? \| mask var C(\d+)K(\d+) (\w+) B(\d+)")
pat_ms = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\] \| mask ms W(\d+) (\w+) B(\d+) s([\d-]+)")   # multistart (10-14)
seen = []
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen.append(m.groups())
ms = [m.groups() for m in pat_ms.finditer(open(tl, errors="ignore").read())] if os.path.isfile(tl) else []
if ms and not seen:
    l, tf, ar, R, W, ds, Bb, st = ms[-1]
    print(f" {B}loss{N}  {float(l):.4f}  = tf {float(tf):.4f} + ar {float(ar):.4f}   (multistart · 지금 W{W} {ds} B{Bb}, 시작점 {st}, 최대 R={R})")
    byw = {}
    for g in ms[-400:]:
        byw.setdefault((g[5], int(g[4])), []).append(float(g[2]))
    print(" (데이터, 창) 별 최근 ar 손실\n   " + "  ".join(f"{d}W{w}:{st.mean(v[-5:]):.3f}" for (d, w), v in sorted(byw.items())))
if seen:
    l, tf, ar, R, hn, C, K, ds, Bb = seen[-1]
    h = f"   힌트 {B}{hn}{N}/256 개" if hn is not None else ""
    tail = open(tl, errors="ignore").read()[-4000:]
    mdl = re.findall(r" dl ([\d.]+)(?: dyn [\d.]+)? \| mask var", tail)
    mdy = re.findall(r" dyn ([\d.]+) \| mask var", tail)
    dl_s = (f" + dl {float(mdl[-1]):.4f} (차이 L1)" if mdl else "") + (f" + dyn {float(mdy[-1]):.4f} (움직임 크기, ×λ 전)" if mdy else "")
    mrc = re.findall(r" hole ([\d.]+) rec ([\d.]+) \| mask var", tail)       # 2026-10-17: 문맥 구멍 — 복원 손실은 걸 때만 (가중치 > 0) 손실 식에 넣는다
    if mrc:
        rw = 0.0
        try:
            rw = float((c["loss"].get("ctx_hole") or {}).get("recon_weight", 1.0))
        except Exception:
            pass
        if rw > 0:
            dl_s += f" + 복원 {rw:g}×{float(mrc[-1][1]):.4f}"
        dl_s += f"   [문맥 구멍 비율 {float(mrc[-1][0]):.2f}]"
    print(f" {B}loss{N}  {float(l):.4f}  = tf {float(tf):.4f} + ar {float(ar):.4f}{dl_s}   (지금 C{C}K{K} {ds} B{Bb}, R={R}){h}")
    if hint_cfg and total:
        a, b = float(hint_cfg.get("start", 0.05)), float(hint_cfg.get("end", 0.9))
        n_max = round(float(hint_cfg.get('ratio_max', 0.1)) * 256)
        if str(hint_cfg.get("shape", "sin2")) == "const":
            print(f"       힌트 일정: step {int(a * total):,}~{int(b * total) - 1:,} 동안 {n_max} 개 고정, 그 뒤 0 (cross-attn 차단)")
        else:
            print(f"       힌트 일정: t {a:.2f}~{b:.2f} sin² 종 모양, 정점 {n_max} 개 (t {(a + b) / 2:.3f})")
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

if os.path.isfile(tl):      # 데이터별 step 수 (op320 · vob 같은 고정/경계 소스가 계획대로 섞이는지)
    from collections import Counter
    cnt = Counter(g[7] for g in seen)
    if cnt:
        tot = sum(cnt.values())
        print(" 데이터별 step  " + "  ".join(f"{k} {v:,} ({100 * v / tot:.1f}%)" for k, v in cnt.most_common()))
for ch in ("chain_hole", "chain_ctxt", "chain_ar"):        # 지금 도는 chain 이 있으면 그것만, 없으면 가장 최근 로그
    cl = os.path.join(ROOT, f"z_training/logs/{ch}.log")
    alive = subprocess.run(["tmux", "has-session", "-t", f"={ch}"], capture_output=True).returncode == 0
    if not alive and ch in ("chain_hole", "chain_ctxt") and not os.path.isfile(cl):
        continue
    lines = open(cl).read().strip().splitlines() if os.path.isfile(cl) and os.path.getsize(cl) else ["(로그 없음)"]
    print(f" {B}{ch}{N} {'살아 있음' if alive else '세션 없음'}")
    for ln in [x for x in lines if x.startswith("[")][-3:]:
        print("   " + ln[:110])
    break
import glob as _g
for r in sorted(os.path.basename(os.path.dirname(os.path.dirname(x))) for x in _g.glob(os.path.join(ROOT, "z_training/runs/natural_ar_vw*/eval/BENCH_latest.txt"))):
    bf = os.path.join(ROOT, "z_training/runs", r, "eval", "BENCH_latest.txt")
    if os.path.isfile(bf):
        print(f" {B}채점{N} {r}:")
        for ln in open(bf).read().strip().splitlines()[1:]:
            print("   " + ln)
print("━" * 78)
EOF
