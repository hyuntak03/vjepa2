#!/bin/bash
# RollOutV3 presence·위치 자 학습 모니터 (ANSI 색).
#
#   watch -c -n 1 bash /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV3/monitor.sh
#   watch -c -n 1 bash .../monitor.sh /경로/특정로그.out      # 로그 지정 (기본 = 가장 최근 presence_*.out)
#   NO_COLOR=1 bash .../monitor.sh                             # 색 없이 (파일로 남길 때)
#
# 보여 주는 것: 실행 상태·경과 · 라벨 집계 (양성/음성/제외) 와 split · 단계별 진행률과 남은 시간 (특징 추출 / probe 학습) ·
#   GPU 사용률 · 최신 손실 (probe 3 종의 좌표·presence) · 끝난 표현의 결과표 (중심 오차, ROC/PR, 출처별 오탐) · 에러 줄.
set -uo pipefail
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2
LOGDIR=$ROOT/z_research/scripts/slurm_logs
LOG=${1:-$(ls -t "$LOGDIR"/presence_*.out 2>/dev/null | head -1)}
exec /data/hyuntak/anaconda3/envs/vjepa2/bin/python - "$LOG" "$ROOT" <<'PYEOF'
import os, re, subprocess, sys, time
from pathlib import Path

log, root = Path(sys.argv[1]) if sys.argv[1] else None, Path(sys.argv[2])
C = not os.environ.get("NO_COLOR")
c = lambda s, *k: (f"\033[{';'.join(map(str, k))}m{s}\033[0m" if C else str(s))
B, DIM, RED, GRN, YEL, BLU, CYN, GRY = 1, 2, 31, 32, 33, 34, 36, 90
W = 100
sec = lambda t: f"{int(t)//3600}:{int(t)%3600//60:02d}:{int(t)%60:02d}" if t >= 3600 else f"{int(t)//60}:{int(t)%60:02d}"


def bar(x, n=34, col=GRN):
    k = max(0, min(n, round(n * x)))
    return c("█" * k, col) + c("░" * (n - k), GRY)


def head(t):
    print(c(f"── {t} ", CYN, B) + c("─" * max(W - len(t) - 4, 0), GRY))


print(c("━" * W, CYN))
if not log or not log.exists():
    print(c(" presence 로그가 없다", B, RED) + c(f"  ({root}/z_research/scripts/slurm_logs/presence_*.out)", GRY))
    raise SystemExit
txt = log.read_text(errors="ignore")
lines = txt.splitlines()
age = time.time() - log.stat().st_mtime
# ⚠️ 살아 있는지는 **로그 갱신 시각**으로 본다 — 이 창과 학습이 다른 노드일 수 있어 pgrep 은 못 믿는다 (2026-09-22)
local = subprocess.run(["pgrep", "-f", "rollout2_presence_readout.py"], capture_output=True).returncode == 0
done = "[done]" in txt
fresh = age < 180
state = (c(" DONE ", B, 30, 42) if done else
         (c(" RUNNING ", B, 30, 42) if fresh else c(" STALLED ", B, 30, 41)))
where = "이 노드" if local else "다른 노드 (로그로만 확인)"
print(c(" RollOutV3 · presence + 위치 자", B, CYN) + f"   {state}  " +
      c(f"로그 {log.name} · {age:.0f}s 전 갱신 · 실행 {where}", GRY))

g = lambda p: (re.findall(p, txt) or [None])[-1]
d = g(r"\[data\].*양성 (\d+) / 음성 (\d+) / 제외 (\d+)")
sp = g(r"\[split\] train (\d+) / val (\d+) / test (\d+)")
if d and sp:
    print(c(f" 라벨  양성 {int(d[0]):,} · 음성 {int(d[1]):,} · 제외 {int(d[2]):,}", 0) +
          c(f"   split  train {sp[0]} / val {sp[1]} / test {sp[2]} clip (block 단위)", GRY))
print(c("━" * W, CYN))

# ── 단계 ──────────────────────────────────────────────────────────────────
EP = int((g(r"epoch\s+(\d+)/") or [0])[0]) if g(r"epoch\s+(\d+)/") else 300
ex = g(r"\[extract\] ~(\d+)/(\d+)\s+(\d+)s")        # 로그 형식: "[extract] ~408/8352  79s  (예상 27분)"
ex_done = "[extract] 끝" in txt
tr = re.findall(r"\[(p|z|h)\] epoch\s+(\d+)\s+(.*?)\s+\((\d+)s\)", txt)
reps = re.findall(r"\[train\] (p|z|h)", txt)
head("progress")
if not ex_done and ex:
    n, tot, t = int(ex[0]), int(ex[1]), int(ex[2])
    rate = n / max(t, 1)
    print(f" 1/2 특징 추출  {bar(n / tot)}  {n:,}/{tot:,} clip   {rate:.1f} clip/s   "
          f"남은 {sec((tot - n) / max(rate, 1e-9))}")
    print(c(f"     clip 마다 p · z · h 세 표현 (GPU 8 장). 끝나면 probe 학습으로 넘어간다", GRY))
elif tr:
    rep, ep, loss, el = tr[-1][0], int(tr[-1][1]), tr[-1][2], int(tr[-1][3])
    ri = (["p", "z", "h"].index(rep) if rep in "pzh" else 0)
    nrep = max(len(set(reps)), 3 if "--reps" not in txt else len(set(reps)))   # 기본 실행은 p·z·h 셋
    spe = el / max(ep + 1, 1)
    left = (EP - ep - 1) * spe + (nrep - 1 - ri) * EP * spe
    print(f" 2/2 probe 학습  {bar((ri * EP + ep + 1) / (nrep * EP))}  표현 {rep} ({ri+1}/{nrep})  "
          f"epoch {ep+1}/{EP}   {spe:.1f} s/epoch   남은 {sec(left)}")
    print(c(f"     최신 손실  {loss}", GRY))
    print(c(f"     xy = 좌표 (양성 튜블릿만) · pres = 있음/없음 BCE (양성+음성)", GRY))
elif ex_done:
    print(" 특징 추출 끝, probe 학습 시작 대기")
else:
    print(" 시작 준비 중 (모델 로딩)")

try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
    u = [int(x.split(",")[0]) for x in q]; m = [int(x.split(",")[1]) for x in q]
    print(f" GPU  {bar(sum(u) / (100 * len(u)), 20, YEL)}  평균 {sum(u)//len(u)}%   메모리 {sum(m)//1024} GB / {24*len(m)} GB   ({len(u)} 장)")
except Exception:
    print(c(" GPU  이 노드에는 GPU 가 없다 (학습은 다른 노드에서 돈다)", GRY))

# ── 끝난 결과 ─────────────────────────────────────────────────────────────
# ⚠️ top1 (attention 질량) 은 2026-09-23 에 추가됐다 — 옛 로그에는 없으니 선택 그룹으로 둔다
res = re.findall(r"\[(p|z|h)/(\w+)\s*\]\s+([\d,]+)p\s+중심 ([\d.]+)/([\d.]+)/([\d.]+) 칸\s+"
                 r"ROC ([\d.]+) PR ([\d.]+)\s+(?:top1 ([\d.]+)/([\d.]+)\s+)?오탐 (.*?)\s+\(양성 유지 ([\d.]+)\)", txt)
head("결과 (test, 끝난 것부터)")
if not res:
    print(c(" 아직 없음 — 표현 하나가 끝나야 한 줄씩 생긴다", GRY))
else:
    print(c(f" {'표현':4s} {'probe':7s} {'파라미터':>10s} {'중심 평균':>9s} {'중앙':>6s} {'상위90%':>7s} "
            f"{'ROC':>6s} {'PR':>6s} {'양성유지':>7s} {'top1 양/음':>12s}  오탐 (val 문턱)", B))
    for rep, kind, par, mu, med, p90, roc, pr, a1p, a1n, fpr, rec in res:
        col = GRN if float(roc) > 0.95 else (YEL if float(roc) > 0.85 else RED)
        t1 = f"{float(a1p):.3f}/{float(a1n):.3f}" if a1p else "  -  "
        print(f" {rep:4s} {kind:7s} {par:>10s} " + c(f"{float(mu):>9.2f}", col) +
              f" {float(med):>6.2f} {float(p90):>7.2f} " + c(f"{float(roc):>6.3f}", col) +
              f" {float(pr):>6.3f} {float(rec):>6.1f}% {t1:>12s}  " + c(fpr, GRY))
    print(c(" 중심 오차 = 모든 GT 양성에서 (presence 판정과 무관), 1 칸 = 18 px", GRY))
    print(c(" top1 = 상위 1 토큰 attention 질량 (양성/음성). 균등 = 0.0039 — 클수록 map 이 날카롭다", GRY))

err = [l for l in lines if re.search(r"Traceback|Error|CUDA out of memory|Killed", l)]
if err:
    head("에러")
    [print(c(" " + l[:W - 2], RED)) for l in err[-3:]]
if done:
    head("산출물")
    print(c(f" {root}/z_research/RollOutV3/exp_results/presence/RESULTS.md", GRN))
print(c("━" * W, CYN))
print(c(f" {time.strftime('%m-%d %H:%M:%S')} · {os.uname().nodename}", GRY))
PYEOF
