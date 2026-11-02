#!/bin/bash
# v3 ablation 학습 한눈에 — 팔마다 상태 · 노드 · GPU 구성 · step · 손실 · step 시간 · 남은 시간 (2026-11-01).
#   bash predictor_lab/v3_ablation/monitor.sh              # 한 번
#   watch -n 30 bash predictor_lab/v3_ablation/monitor.sh  # 30 초마다
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
python3 - "$R" <<'PY'
import os, re, sys, glob, time
R = sys.argv[1]; Q = f"{R}/predictor_lab/v3_ablation/queue"; TOTAL = 2000
rows = [l.rstrip("\n").split("\t") for l in open(f"{R}/predictor_lab/v3_ablation/runs.tsv") if l.strip() and not l.startswith("#")]
pat = re.compile(r"\[\s*(\d+),\s*(\d+)\] loss ([0-9.]+) \(avg ([0-9.]+)\).*?iter (\d+) ms")
print(time.strftime("== %m-%d %H:%M:%S"), "  (2,000 step = 20 epoch × 100, 전역 batch 24)")
print(f"{'팔':22s} {'상태':6s} {'노드':5s} {'GPU×b':6s} {'step':>11s} {'%':>4s} {'손실(평균)':>10s} {'s/step':>6s} {'남은':>7s}")
cnt = {}
for name, arm, seed in rows:
    lock, done = os.path.isdir(f"{Q}/{name}.lock"), os.path.exists(f"{Q}/{name}.done")
    logs = glob.glob(f"{Q}/{name}.*.log"); node = os.path.basename(sorted(logs, key=os.path.getmtime)[-1]).split(".")[1] if logs else "-"
    nf = open(f"{Q}/{name}.nfail").read().strip() if os.path.exists(f"{Q}/{name}.nfail") else ""
    st = "끝" if done else ("도는중" if lock else "대기")
    cfgp, tl = f"{R}/z_training/runs/{name}/config.yaml", f"{R}/z_training/runs/{name}/train.log"
    gb = "-"
    if logs:
        m = re.search(r"devices ((?:cuda:\d+ ?)+)", open(sorted(logs, key=os.path.getmtime)[-1], errors="ignore").read())
        ng = len(m.group(1).split()) if m else 0
        b = re.search(r"^\s*batch_size:\s*(\d+)", open(cfgp).read(), re.M) if os.path.exists(cfgp) else None
        gb = f"{ng}×{b.group(1) if b else '?'}"
    step = loss = avg = it = None
    if os.path.exists(tl):
        for line in reversed(open(tl, errors="ignore").read().splitlines()[-400:]):
            mm = pat.search(line)
            if mm:
                ep, itr = int(mm.group(1)), int(mm.group(2)); step = (ep - 1) * 100 + itr + 1
                loss, avg, it = float(mm.group(3)), float(mm.group(4)), int(mm.group(5)); break
        if lock and time.time() - os.path.getmtime(tl) > 600 and not done:
            st = "멈춤?"
    cnt[st] = cnt.get(st, 0) + 1
    if step:
        rem = (TOTAL - step) * it / 1000
        print(f"{name:22s} {st:6s} {node:5s} {gb:6s} {step:5d}/{TOTAL} {100*step/TOTAL:3.0f}% {avg:10.4f} {it/1000:6.2f} {rem/60:6.0f}분" + (f"  실패{nf}" if nf else ""))
    else:
        print(f"{name:22s} {st:6s} {node:5s} {gb:6s} {'-':>11s} {'':>4s} {'':>10s} {'':>6s} {'':>7s}" + (f"  실패{nf}" if nf else ""))
print("요약:", " · ".join(f"{k} {v}" for k, v in cnt.items()))
PY
