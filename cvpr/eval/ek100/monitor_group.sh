#!/usr/bin/env bash
# =============================================================================
# EK100 predictor_group 실행 모니터 (2026-10-22). 진행 막대 · 속도 · 남은 시간 · 팔별 검증 표 · GPU 막대.
#
#   watch -c -n 10 bash cvpr/eval/ek100/monitor_group.sh vast_ek100_i2pft
#   bash cvpr/eval/ek100/monitor_group.sh vast_ek100_i2pft          # 한 번만
#
# 인자 = 레지스트리 모델 이름 (결과 폴더 ek100_<이름>). 기본 vast_ek100_i2pft.
# epoch 은 stdout.log 의 step 번호가 0 으로 돌아오는 횟수로 센다 (로그에 epoch 줄이 없다).
# epoch 길이 = 학습 CSV 행 수 ÷ (batch × GPU 수) — GPU 수는 nvidia-smi 의 이 실행 프로세스 수 (못 찾으면 WS 환경변수, 기본 8).
# 검증 값 = val_metrics__<팔>.jsonl (5 epoch 마다) — verb / noun / action mean class recall@5, head 중 best.
# =============================================================================
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M=${1:-vast_ek100_i2pft}
D="$HERE/../../results/eval/ek100/action_anticipation_frozen/ek100_$M"
[[ -d "$D" ]] || { echo "결과 폴더 없음: $D"; exit 1; }
GPUQ=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 ps -o args= -p 2>/dev/null | grep -c "evals.main\|action_anticipation" )
GPUQ="$GPUQ" WS=${WS:-8} /venv/vjepa/bin/python - "$D" <<'PY'
import glob, json, math, os, re, sys, datetime as dt
D = sys.argv[1]
B, DIM, R, G, Y, C, RED = "\033[1m", "\033[2m", "\033[0m", "\033[32m", "\033[33m", "\033[36m", "\033[31m"
W = 66
def line(ch="─"): print(DIM + ch * W + R)
def bar(frac, n=30, col=G):
    k = int(round(frac * n)); return col + "█" * k + DIM + "░" * (n - k) + R
cfg = open(os.path.join(D, "_resolved.yaml")).read()
g = lambda pat, d: (re.search(pat, cfg).group(1) if re.search(pat, cfg) else d)
EP, BS = int(g(r"num_epochs:\s*(\d+)", 20)), int(g(r"batch_size:\s*(\d+)", 16))
ntrain = sum(1 for _ in open(g(r"dataset_train:\s*(\S+)", ""))) - 1
heads = re.findall(r"(?<![_a-z])lr:\s*([\d.e-]+)", cfg.split("heads:")[1]) if "heads:" in cfg else []
WS = int(os.environ.get("WS", 8))
gpus = [l.split(", ") for l in os.environ["GPUQ"].strip().splitlines()]
busy = sum(1 for x in gpus if float(x[1]) > 1000)
WS = busy or WS
pat = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*train_one_epoch.*\[\s*(\d+)\] acc \(v/n\): ([\d.]+)% \(([\d.]+)% ([\d.]+)%\).*\[mem: ([\d.e+]+)\]")
recs = [(dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), int(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[6]))
        for m in (pat.search(l) for l in open(os.path.join(D, "stdout.log"), errors="ignore")) if m]
kst = lambda t: (t + dt.timedelta(hours=9)).strftime("%H:%M")
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

line("━")
print(f"{B}  EK100 action anticipation{R}  {C}{os.path.basename(D)[6:]}{R}" + f"{DIM}{'KST ' + kst(now):>{W - 30 - len(os.path.basename(D)[6:])}}{R}")
print(f"{DIM}  batch {BS} × GPU {WS} = {BS * WS} · epoch {EP} · head lr {' / '.join(heads)} · 학습 clip {ntrain:,}{R}")
line("━")
if not recs:
    print(f"  {Y}로딩 중 — 아직 학습 step 로그 없음{R}"); line(); sys.exit()
ep, prev, maxit = 1, -1, 0
for r in recs:
    if r[1] < prev: ep += 1
    maxit = max(maxit, r[1]); prev = r[1]
ipe = math.ceil(ntrain / (BS * WS))
done = (ep - 1) * ipe + recs[-1][1]
t0, tl, last = recs[0][0], recs[-1][0], recs[-1]
rate = (tl - t0).total_seconds() / max(done, 1)
left = max(EP * ipe - done, 0)
eta = tl + dt.timedelta(seconds=left * rate)
stale = (now - tl).total_seconds()
tot = done / (EP * ipe)
print(f"  {B}전체{R}   {bar(tot)} {B}{100 * tot:5.1f}%{R}  {done}/{EP * ipe} step")
print(f"  {B}epoch{R}  {bar(min(last[1] / ipe, 1), col=C)} {ep:>2d}/{EP}   step {last[1]}/{ipe}")
print()
if done < 10:
    print(f"  속도      {Y}측정 중 (step 10 이후 표시){R}")
else:
    print(f"  속도      {B}{rate:.1f}{R} s/step {DIM}(검증 포함 평균){R}")
    print(f"  남은 시간 {B}{left * rate / 3600:.1f} h{R}   →   종료 예상 {B}{G}{kst(eta)} KST{R}")
st = f"{RED}⚠ {stale / 60:.0f}분째 새 로그 없음 (검증 중이거나 멈춤){R}" if stale > 300 else f"{G}● 진행 중{R} {DIM}({stale:.0f}s 전 로그){R}"
print(f"  상태      {st}")
print(f"  학습 acc  action {last[2]:.1f}%  {DIM}verb {last[3]:.1f} · noun {last[4]:.1f} · mem {last[5] / 1e3:.1f} GB{R}")
line()
print(f"  {B}검증{R} {DIM}mean class recall@5 · head 중 best · 5 epoch 마다{R}")
fs = sorted(glob.glob(os.path.join(D, "val_metrics__*.jsonl")))
print(f"  {DIM}{'팔':<20s} {'epoch':>5s} {'verb':>7s} {'noun':>7s} {'action':>8s}{R}")
print(f"  {DIM}{'시트 AR (main 6×16)':<20s} {'20':>5s} {'48.9':>7s} {'49.0':>7s} {'33.1':>8s}{R}")
if not fs:
    print(f"  {Y}아직 없음{R}")
for f in fs:
    arm = os.path.basename(f)[len("val_metrics__"):-6]
    rows = [r for r in (json.loads(l) for l in open(f) if l.strip()) if not r.get("val_only")]
    for i, r in enumerate(rows):
        print(f"  {(arm if i == 0 else ''):<20s} {r['epoch']:>5d} {r['verb']['recall']:7.1f} {r['noun']['recall']:7.1f} {B}{r['action']['recall']:8.1f}{R}")
line()
print(f"  {B}GPU{R}")
for i, used, total, util in gpus:
    u = float(used) / float(total)
    print(f"  {i}  mem {bar(u, 20, Y)} {float(used) / 1024:5.1f}/{float(total) / 1024:.0f} GB   util {bar(float(util) / 100, 10, C)} {int(util):3d}%")
line("━")
PY
