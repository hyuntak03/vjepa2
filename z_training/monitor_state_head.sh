#!/bin/bash
# -----------------------------------------------------------------------------
# state_head PFT 모니터 (2026-10-19) — phys2d_sup 상태 supervision run 전용
#
#   watch -c -n 5 bash z_training/monitor_state_head.sh                                  # 기본 run
#   watch -c -n 5 bash z_training/monitor_state_head.sh phys2d_sup_vast_ar_state_pft     # run 이름
#
# 보여 주는 것
#   진행   epoch · step · ETA (log_r0.csv 최근 20 step 평균)
#   최근   마지막 로그 줄들 (4 rank) 평균 — 잠재 L1 (tf / ar) · head 손실 · 존재 BCE · 위치 L1 · rollout 위치 오차 px (보임 / 가림)
#   epoch  metrics.jsonl 의 epoch 평균 + in-loop IntPhys1 val (출발 81.7 %)
#   GPU    nvidia-smi
#   채점   학습 뒤 자동 채점 (<run>.after.log 의 === 줄)
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
RUN="${1:-phys2d_sup_vast_ar_state_pft}"
D=$PROJ/z_training/runs/$RUN
[[ -d "$D" ]] || { echo "run 폴더가 없다: $D"; exit 1; }

/data/hyuntak/anaconda3/envs/vjepa2/bin/python - "$D" <<'EOF'
import json, os, re, sys, time, csv
D = sys.argv[1]
B, N, G, Y, R = "\033[1m", "\033[0m", "\033[32m", "\033[33m", "\033[31m"
run = os.path.basename(D)
print(f"{B}{run}{N}   {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 설정
epochs = ipe = None
try:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
    epochs, ipe = int(cfg["optimization"]["epochs"]), int(cfg["optimization"]["ipe"])
    sh = cfg.get("state_head") or {}
    print(f"  data {[d.get('name') for d in cfg['data']['datasets']]} · batch {cfg['data']['batch_size']}/rank · lr {cfg['optimization']['lr']}"
          f" · head weight {sh.get('weight')} lr×{sh.get('lr_mult')}" + (f" · {Y}freeze_predictor{N}" if sh.get("freeze_predictor") else ""))
except Exception as e:
    print(f"  (config 못 읽음: {e})")

# 진행 · ETA
rows = []
p = os.path.join(D, "log_r0.csv")
if os.path.exists(p):
    with open(p) as f:
        rows = [r for r in csv.reader(f) if r and r[0].isdigit()]
if rows:
    ep, it = int(rows[-1][0]), int(rows[-1][1])
    done = (ep - 1) * ipe + it + 1 if ipe else None
    total = epochs * ipe if (epochs and ipe) else None
    t = [float(r[5]) for r in rows[-20:]]
    ms = sum(t) / len(t)
    eta = (total - done) * ms / 1000 if total else 0
    pct = 100 * done / total if total else 0
    bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
    print(f"\n{B}진행{N}  epoch {ep}/{epochs} · step {it + 1}/{ipe} · 전체 {done:,}/{total:,}  {bar} {pct:.1f}%")
    print(f"      {ms / 1000:.2f} s/step · 남은 {int(eta // 3600)}:{int(eta % 3600 // 60):02d} · 끝 ≈ {time.strftime('%H:%M', time.localtime(time.time() + eta))}")
else:
    print(f"\n{Y}아직 step 로그가 없다 (모델 로딩 · at_start val 중일 수 있다){N}")

# 최근 로그 (4 rank 마지막 줄 평균)
pat = re.compile(r"\[(\d+),\s*(\d+)\] loss ([\d.]+) .*?\[tf ([\d.]+) ar ([\d.]+) R=\d+\] \[head ([\d.]+) \| ex ([\d.]+)/([\d.]+) pos ([\d.]+)/([\d.]+) "
                 r"dir ([\d.]+)/([\d.]+) spd ([\d.]+)/([\d.]+) \| ar px vis ([\d.na]+) hid ([\d.na]+)\]")
lines = []
lp = os.path.join(D, "train.log")
if os.path.exists(lp):
    with open(lp, errors="ignore") as f:
        f.seek(max(0, os.path.getsize(lp) - 400_000))
        lines = f.read().splitlines()
hits = [m.groups() for m in map(pat.search, lines) if m][-4:]
if hits:
    def avg(i):
        v = [float(h[i]) for h in hits if h[i] != "nan"]
        return sum(v) / len(v) if v else float("nan")
    print(f"\n{B}최근{N} (epoch {hits[-1][0]} itr {hits[-1][1]}, 4 rank 평균)")
    print(f"  손실 전체 {avg(2):.3f} = 잠재 L1 [tf {avg(3):.4f} · ar {avg(4):.4f}] + head {avg(5):.3f}")
    print(f"  head    존재 BCE tf {avg(6):.3f} / ar {avg(7):.3f} · 위치 L1 {avg(8):.4f} / {avg(9):.4f}"
          f" · 방향 {avg(10):.3f} / {avg(11):.3f} · 속력 {avg(12):.3f} / {avg(13):.3f}")
    print(f"  rollout 위치 오차  보임 {G}{avg(14):.1f} px{N} · 가림 {G}{avg(15):.1f} px{N}")
err = [l for l in lines[-300:] if "Traceback" in l or "Error" in l or "nan/inf" in l]
if err:
    print(f"\n{R}오류: {err[-1][:200]}{N}")

# epoch 표
mp = os.path.join(D, "metrics.jsonl")
if os.path.exists(mp):
    recs = [json.loads(l) for l in open(mp) if l.strip()]
    if recs:
        print(f"\n{B}epoch{N}  (train 평균 · in-loop IntPhys1 val, 출발 81.7%)")
        print(f"  {'ep':>3} {'latent tf':>9} {'ar':>7} {'head':>6} {'ex_acc':>7} {'pos_ar':>7} {'px vis':>7} {'px hid':>7} {'IP1 val':>8}")
        for r in recs:
            h = r.get("train_head") or {}
            v = (r.get("val") or {}).get("acc")
            vs = f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"
            print(f"  {r['epoch']:>3} {r.get('train_jloss', float('nan')):>9.4f} {r.get('train_sloss', float('nan')):>7.4f} {h.get('head', float('nan')):>6.3f} "
                  f"{h.get('ar_ex_acc', float('nan')) * 100:>6.1f}% {h.get('ar_pos', float('nan')):>7.4f} {h.get('ar_pos_px_vis', float('nan')):>7.1f} "
                  f"{h.get('ar_pos_px_hid', float('nan')):>7.1f} {vs:>8}")
ck = sorted([f for f in os.listdir(D) if re.match(r"e\d+\.pt$", f)], key=lambda s: int(s[1:-3]))
if ck:
    print(f"  체크포인트: {' '.join(ck)}" + (" · latest.pt" if os.path.exists(os.path.join(D, 'latest.pt')) else ""))

# 학습 뒤 자동 채점
ap = D + ".after.log"
if os.path.exists(ap):
    st = [l for l in open(ap, errors="ignore").read().splitlines() if l.startswith("===")]
    print(f"\n{B}자동 채점{N}  " + (" → ".join(s.strip("= ").split(" 2026")[0].replace(" done", " ✓") for s in st) if st else "학습 끝나기를 기다리는 중"))
EOF

echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
  awk -F', ' '{printf "  GPU %s  %4s  %s / %s\n", $1, $2, $3, $4}'
