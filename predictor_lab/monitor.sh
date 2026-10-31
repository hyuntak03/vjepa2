#!/bin/bash
# predictor_lab 학습 진행 보기 — 잡 상태 · 진행 step · 손실 · 남은 시간 (2026-11-01).
#
#   bash predictor_lab/monitor.sh btrack_v1ns_l1                 # 한 번
#   watch -n 30 bash predictor_lab/monitor.sh btrack_v1ns_l1     # 30 초마다 (Ctrl-C 로 끝)
#   bash predictor_lab/monitor.sh                                # 내 lab 잡 전부
#
# 읽는 곳: squeue (잡 이름 lab_tr_<run>) · z_training/runs/<run>/train.log · slurm 로그 z_training/slurm_logs/lab_tr_<run>_<job>.out
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  mapfile -t RUNS < <(squeue -h -u "$USER" -o "%j" | sed -n 's/^lab_tr_//p' | sort -u)
fi
echo "== $(date '+%m-%d %H:%M:%S')"
for R in "${RUNS[@]}"; do
  echo "-- $R"
  squeue -h -u "$USER" -n "lab_tr_$R" -o "   잡 %i  %T  %M  %R" | head -3
  L="$REPO/z_training/runs/$R/train.log"
  if [[ ! -f $L ]]; then
    S=$(ls -t "$REPO"/z_training/slurm_logs/lab_tr_"${R}"_*.out 2>/dev/null | head -1)
    echo "   train.log 아직 없음${S:+ — slurm 로그 끝:}"; [[ -n $S ]] && tail -3 "$S" | sed 's/^/   /'
    continue
  fi
  python3 - "$L" <<'EOF'
import re, sys
L = open(sys.argv[1], errors="ignore").read().splitlines()
ipe = ep = None; last = None; done = False
for s in L:
    m = re.search(r"\| ipe (\d+)", s);  ipe = int(m.group(1)) if m else ipe
    m = re.search(r"epochs (\d+)\s*$", s); ep = int(m.group(1)) if m else ep
    m = re.search(r"\[\s*(\d+),\s*(\d+)\] loss ([0-9.]+) \(avg ([0-9.]+)\).*?iter (\d+) ms.*?data (\d+) ms", s)
    if m: last = tuple(float(x) for x in m.groups())
    if re.search(r"\] done\s*$", s): done = True
err = [s for s in L if "Traceback" in s or "Error" in s][-1:]
if last and ipe and ep:
    e, i, loss, avg, it, dt = last
    step, tot = int((e - 1) * ipe + i + 1), ep * ipe
    eta = (tot - step) * it / 1000 / 60
    print(f"   epoch {int(e)}/{ep} · step {step}/{tot} ({100 * step / tot:.0f} %) · loss {loss:.4f} (epoch 평균 {avg:.4f}) · step {it / 1000:.1f} s (데이터 {dt:.0f} ms) · 남은 시간 ~{eta:.0f} 분")
else:
    print("   아직 step 로그 없음 (모델 · 데이터 준비 중)")
ep_lines = [s for s in L if re.search(r"epoch \d+ avg loss", s)]
if ep_lines:
    print("   epoch 평균 손실:", " → ".join(re.search(r"avg loss ([0-9.]+)", s).group(1) for s in ep_lines[-6:]))
if done: print("   ✔ 끝남 — 평가: bash predictor_lab/eval.sh", sys.argv[1].split("/")[-2])
if err: print("   ⚠", err[0][-160:])
EOF
done
