#!/bin/bash
# pixel decoder 학습 모니터.   watch -c -n 2 bash z_pixel_decoder/monitor.sh [name]   (기본 vobench_vits)
# 진행 · ETA · 최근 손실 (L1 · LPIPS) · val PSNR 추이 · 최신 복원 그림 · GPU
cd "$(dirname "$0")/.."
NAME=${1:-vobench_vits}; D=z_pixel_decoder/runs/$NAME
/venv/vjepa/bin/python - "$D" "$NAME" <<'EOF'
import os, re, sys, glob, csv, statistics as st, subprocess, yaml
D, NAME = sys.argv[1:3]
B, G, Y, N, DIM = "\033[1m", "\033[32m", "\033[33m", "\033[0m", "\033[2m"
alive = subprocess.run(["tmux", "has-session", "-t", f"=pixdec_{NAME}"], capture_output=True).returncode == 0
print("━" * 78)
print(f" {B}pixel decoder{N}  {NAME}   " + (f"{G}● 학습 중{N}" if alive else f"{Y}○ 세션 없음{N}"))
L = os.path.join(D, "train.log")
if not os.path.isfile(L):
    print(" (아직 로그 없음)"); sys.exit()
txt = open(L, errors="ignore").read()
m = re.search(r"pixel decoder ([\d.]+)M 파라미터 · train clip (\d+) · val clip (\d+) · (\d+) step × (\d+)/GPU × (\d+) GPU · lr ([\d.e-]+) · 손실 (\w+)(?: \+ LPIPS ×([\d.]+))?", txt)
if m:
    tot = int(m[4])
    print(f" 설정   {m[1]}M · train {m[2]} / val {m[3]} clip · batch {m[5]} × {m[6]} GPU · lr {m[7]} · {m[8]}" + (f" + LPIPS ×{m[9]}" if m[9] else ""))
steps = re.findall(r"^step (\d+) loss ([\d.]+)(?: l1 ([\d.]+))?(?: lpips ([\d.]+))? lr ([\d.e-]+) \((\d+)s\)", txt, re.M)
print("─" * 78)
if steps and m:
    s, sec = int(steps[-1][0]), int(steps[-1][5])
    rate = sec / max(1, s)
    left = (tot - s) * rate
    bar = int(40 * s / tot)
    print(f" 진행   {'█' * bar}{'░' * (40 - bar)}  {100 * s / tot:5.1f}%   {s:,}/{tot:,} step   {rate:.2f} s/step   남은 {int(left // 3600)}:{int(left % 3600 // 60):02d}")
    rec = steps[-5:]
    f = lambda i: st.mean(float(r[i]) for r in rec if r[i])
    print(f" 손실   {B}{f(1):.4f}{N}" + (f"   L1 {f(2):.4f}" if rec[-1][2] else "") + (f"   LPIPS {f(3):.4f}" if rec[-1][3] else "") + f"   lr {rec[-1][4]}   {DIM}(최근 {len(rec)} 줄 평균){N}")
vals = re.findall(r"\[val step (\d+)\] mse ([\d.]+) · PSNR ([\d.]+) dB", txt)
if vals:
    best = max(vals, key=lambda v: float(v[2]))
    print(f" val    지금 step {vals[-1][0]}  PSNR {B}{float(vals[-1][2]):.2f} dB{N}   최고 {float(best[2]):.2f} (step {best[0]})   mse {float(vals[-1][1]):.4f}")
    print("        추이 " + "  ".join(f"{int(v[0]) // 1000 if int(v[0]) >= 1000 else int(v[0])}{'k' if int(v[0]) >= 1000 else ''}:{float(v[2]):.1f}" for v in vals[-10:]))
sm = sorted(glob.glob(os.path.join(D, "samples", "step_*.png")))
if sm:
    print(f" 그림   {os.path.abspath(sm[-1])}   {DIM}(위 원본 · 아래 복원, val 8 블록){N}")
if "Traceback" in txt:
    print(f" {Y}⚠ Traceback 이 로그에 있다 — tail -50 {L}{N}")
print("─" * 78)
g = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
for i in range(0, len(g), 4):
    print(" GPU  " + "   ".join(f"{a.strip()}: {int(b) / 1024:4.1f}/{int(c) / 1024:.0f}G {int(u):3d}%" for a, b, c, u in (x.split(",") for x in g[i:i + 4])))
print("━" * 78)
EOF
