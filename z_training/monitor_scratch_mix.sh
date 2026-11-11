#!/bin/bash
# =============================================================================
# 자연 영상 + 합성 음의 손실 처음부터 학습 판 모니터 (2026-11-10). 70 epoch × 294 step.
#
#   watch -c -n 30 bash z_training/monitor_scratch_mix.sh                       # 기본: v4bc4 p 0.2 scratch (+ 취소한 v5id 판 대조)
#   watch -c -n 30 bash z_training/monitor_scratch_mix.sh <run> [<run> ...]
#
# 판마다 epoch 별: real bench val (사라짐 A / 나타남 B, 190 쌍) · 자연 L1 (TF / 굴림) · 합성 가능 L1 · 불가능 L1 · 차이 · 음의 손실 항 (= max(0, 1 − 불가능 L1)).
# 음의 손실이 일하면 불가능 L1 이 올라가고 차이가 벌어진다 (이어 학습 p 0.2 판: 0.52 → 0.71, 차이 0.12). 원래 AR 자연 L1 은 epoch 10 0.537 / 0.553 · epoch 54 0.531 / 0.548.
# =============================================================================
cd "$(dirname "$0")/.." || exit 1
RUNS=${*:-"natural_ar_vw_state_v4bc4_negimp_vanish_app02_mix natural_ar_vw_state_v5id_negimp_mix"}
echo "=== 자연 + 합성 음의 손실 scratch 모니터 · $(TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST') ==="
for run in $RUNS; do
  /venv/vjepa/bin/python - "$run" <<'PY'
import re, sys, os, collections, time
run = sys.argv[1]; R = f"z_training/runs/{run}"; L = f"{R}/train.log"
print(f"\n── {run}")
if not os.path.exists(L):
    print("   아직 시작 안 함"); sys.exit()
alive = os.popen(f"ps -eo args | grep -c '[l]aunch.py --fname \\S*/runs/{run}/config.yaml'").read().strip()
st = collections.defaultdict(lambda: collections.defaultdict(list)); val = {}; last = None
for line in open(L, errors="ignore"):
    m = re.search(r"\]\s+\[\s*(\d+),\s*(\d+)\] loss .*?\[tf ([0-9.]+) ar ([0-9.]+) R=\d+\]", line)
    if m:
        e = int(m.group(1)); d = st[e]; d["tf"].append(float(m.group(3))); d["ar"].append(float(m.group(4))); last = (e, int(m.group(2)))
        n = re.search(r"neg pos ([0-9.]+) imp ([0-9.]+).*?iloss ([0-9.]+)", line)
        if n:
            d["pos"].append(float(n.group(1))); d["imp"].append(float(n.group(2))); d["il"].append(float(n.group(3)))
        t = re.search(r"iter (\d+) ms", line)
        if t: d["ms"].append(int(t.group(1)))
    v = re.search(r"\[epoch (\d+)\] val acc ([0-9.]+)% \(A ([0-9.]+) / B ([0-9.]+)\)", line)
    if v: val[int(v.group(1))] = (float(v.group(2)), float(v.group(3)), float(v.group(4)))
mean = lambda x: sum(x) / len(x) if x else float("nan")
if last:
    step = (last[0] - 1) * 294 + last[1] + 1
    ms = mean(st[last[0]]["ms"][-50:]) or 2000
    left_h = (20580 - step) * ms / 1000 / 3600 + (70 - last[0]) * 40 / 3600
    end = time.strftime("%m-%d %H:%M", time.gmtime(time.time() + left_h * 3600 + 9 * 3600))
    state = "학습 중" if alive != "0" else ("끝남" if os.path.exists(f"{R}/e70.pt") else "⚠️ 멈춤 (프로세스 없음)")
    print(f"   진행: epoch {last[0]}/70 · step {step}/20580 · {ms:.0f} ms/step · 남은 약 {left_h:.1f} 시간 (끝 ≈ {end} KST) · {state}")
if os.popen(f"grep -c Traceback {L}").read().strip() != "0":
    print(f"   ⚠️ Traceback 있음 — tail -50 {L}")
print("   epoch | real val (A / B)     | 자연 L1 TF / 굴림 | 합성 가능 L1 | 불가능 L1 | 차이   | 음의 손실 항")
eps = sorted(set(st) | set(val))
show = eps if len(eps) <= 16 else eps[:3] + eps[-13:]
for e in show:
    d = st.get(e, {}); v = val.get(e)
    vs = f"{v[0]:5.1f} ({v[1]:4.1f}/{v[2]:4.1f})" if v else "        -         "
    gap = mean(d.get("imp", [])) - mean(d.get("pos", []))
    print(f"   {e:5d} | {vs:20s} | {mean(d.get('tf', [])):.3f} / {mean(d.get('ar', [])):.3f}    | {mean(d.get('pos', [])):.3f}        | {mean(d.get('imp', [])):.3f}     | {gap:+.3f} | {mean(d.get('il', [])):.3f}")
if val:
    b = max(val.items(), key=lambda x: x[1][0]); print(f"   real val 최고: epoch {b[0]} {b[1][0]:.1f} ({b[1][1]:.1f}/{b[1][2]:.1f})")
PY
done
echo; echo "── 채점 사슬"; tail -2 z_training/logs/chain_v4bc4_mix_eval_20261110.log 2>/dev/null | sed 's/^/   /'
echo; echo "── GPU"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU%s %s %s\n",$1,$2,$3}'
