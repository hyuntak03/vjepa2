#!/bin/bash
# -----------------------------------------------------------------------------
# 학습 하나를 비교 run 들과 **같은 epoch 끼리** 학습 중 검증으로 나란히 보는 모니터 (2026-11-04). 시각은 전부 KST.
#   기본: natural_ar_vw_state_oneshot_vidctx_causal (영상 문맥 · 한 번에 예측 + 전부 블록 인과) vs
#         natural_ar_vw_state_oneshot_vidctx (영상 문맥 · 한 번에 예측 + 양방향, 시트 "future prediction") ·
#         natural_ar_vw_state_oneshot_causal (블록 문맥 · 한 번에 예측 + 블록 인과, e15 에서 멈춤 → 체인 ② 에서 이어 학습) ·
#         natural_ar_vw_state_v4vc (블록 AR + v4 10 % 섞기 scratch, 시트 "AR - scratch VMX0.8 + blender_v4")
#
#   watch -c -n 10 bash z_training/monitor_val_compare.sh
#   watch -c -n 10 bash z_training/monitor_val_compare.sh <run> <비교 run> ...
#   EVAL_LOG=z_training/logs/<자동 채점 로그> ...      # 아래 채점 상태 줄 (기본: chain_oneshot_causal_20261104.log)
#
# 학습 중 검증 = VOBench v1 late 432 쌍 (학습에 안 쓴 영상, 문맥 16 장 → 미래 16 장). 감시용이고 체크포인트 선택에 안 쓴다.
#   정식 수치는 e70 으로 시트 다섯 벤치 (z_training/chain_full_eval.sh). ⚠️ 기본 블록 AR (natural_ar_vw_state) 은
#   학습 중 검증이 IntPhys1 dev 라 이 표에 못 넣는다 (비교 run 의 val 데이터셋이 다르면 열 머리에 표시).
# ETA = 남은 step × 최근 epoch 들의 실측 속도 (epoch 끝 시각 차 / step 수 — 검증 · 저장 시간 포함).
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN=${1:-natural_ar_vw_state_oneshot_vidctx_causal}; shift || true
REFS=${*:-natural_ar_vw_state_oneshot_vidctx natural_ar_vw_state_oneshot_causal natural_ar_vw_state_v4vc}
EVAL_LOG=${EVAL_LOG:-$PROJ/z_training/logs/chain_oneshot_causal_20261104.log}
T=0; tmux has-session -t "=train_$RUN" 2>/dev/null && T=1
P=$(ps -eo args | grep -cE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$RUN/config\.yaml")
GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
/venv/vjepa/bin/python - "$PROJ" "$RUN" "$T" "$P" "$GPU" "$EVAL_LOG" $REFS <<'EOF'
import json, os, re, sys, glob, datetime as dt
proj, run, tm, npr, gpu, evlog, refs = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7:]
KST = dt.timezone(dt.timedelta(hours=9)); UTC = dt.timezone.utc
now = dt.datetime.now(KST)
R = lambda r: f"{proj}/z_training/runs/{r}"
def kst(s):   # 로그 · metrics 시각은 UTC 문자열
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).astimezone(KST)
def metrics(r):
    f = f"{R(r)}/metrics.jsonl"; out = {}
    if os.path.exists(f):
        for l in open(f):
            try: d = json.loads(l)
            except Exception: continue
            out[int(d["epoch"])] = d
    return out
def cfg_val(r):
    f = f"{R(r)}/config.yaml"
    if not os.path.exists(f): return "?"
    s = open(f).read(); m = re.search(r"^val:\s*\n\s+dataset:\s*(\S+)", s, re.M)
    return m.group(1) if m else "?"
def cfg_opt(r):
    s = open(f"{R(r)}/config.yaml").read()
    ep = int(re.search(r"^optimization:[\s\S]*?\n\s+epochs:\s*(\d+)", s, re.M).group(1))
    ipe = int(re.search(r"^optimization:[\s\S]*?\n\s+ipe:\s*(\d+)", s, re.M).group(1))
    return ep, ipe
G, Y, Rd, B, D, X = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[2m", "\033[0m"
bar = lambda f, w=36: "█" * int(round(f * w)) + "░" * (w - int(round(f * w)))

M = metrics(run); EP, IPE = cfg_opt(run); TOT = EP * IPE
# 지금 step: train.log 끝 부분의 마지막 "[ep, itr] loss" 줄
cur_ep, cur_it, cur_loss, last_ts = 0, -1, None, None
tl = f"{R(run)}/train.log"
if os.path.exists(tl):
    with open(tl, "rb") as fh:
        fh.seek(max(0, os.path.getsize(tl) - 400_000)); tail = fh.read().decode("utf-8", "ignore").replace("\r", "\n")
    for m in re.finditer(r"\]\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\[app\.vjepa_frozen\.train\]\[main\s*\] \[(\d+),\s+(\d+)\] loss ([\d.]+)", tail):
        last_ts, cur_ep, cur_it, cur_loss = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    err = [l for l in tail.splitlines()[-400:] if re.search(r"Traceback|Error|NaN|nan loss", l)]
else:
    err = []
step = (cur_ep - 1) * IPE + cur_it + 1 if cur_ep else 0
alive = f"{G}도는 중{X}" if (tm and npr) else (f"{Y}tmux 만 있음 (프로세스 없음){X}" if tm else f"{Rd}멈춤{X}")
print(f"{B}학습 감시 · 같은 epoch 비교{X}  {now:%F %T} KST")
print("─" * 86)
print(f"{B}{run}{X}   {alive}  (tmux {'O' if tm else 'X'} · 프로세스 {npr})")
# 속도: 최근 epoch 끝 시각 차 (검증 · 저장 포함)
eps = sorted(M)
eta_s = None
if len(eps) >= 2:
    k = eps[-min(6, len(eps)):]
    t0, t1 = kst(M[k[0]]["time"]), kst(M[k[-1]]["time"])
    sps = (t1 - t0).total_seconds() / max(1, (k[-1] - k[0]) * IPE)
    eta_s = (TOT - step) * sps
    started = kst(M[eps[0]]["time"]) - dt.timedelta(seconds=sps * IPE)
    print(f"  step {step:>6}/{TOT}  {bar(step / TOT)}  {100 * step / TOT:5.1f}%   epoch {cur_ep}/{EP}")
    print(f"  속도 {sps:.2f} s/step (최근 {k[-1] - k[0]} epoch, 검증 포함)   경과 {str(now - started).split('.')[0]}   "
          f"{B}남은 시간 {str(dt.timedelta(seconds=int(eta_s)))}  → 끝 ≈ {(now + dt.timedelta(seconds=eta_s)):%m-%d %H:%M} KST{X}")
else:
    print(f"  step {step}/{TOT}  epoch {cur_ep}/{EP}  (속도는 epoch 2 개가 끝난 뒤)")
if cur_loss is not None:
    el = [M[e]["train_loss"] for e in eps]
    print(f"  loss 지금 {cur_loss:.4f} · epoch 평균 {el[0]:.4f} (e{eps[0]}) → {el[-1]:.4f} (e{eps[-1]})" if el else f"  loss {cur_loss:.4f}",
          f"  · 마지막 로그 {int((now - kst(last_ts)).total_seconds())} 초 전")
ck = sorted(glob.glob(f"{R(run)}/e*.pt"), key=os.path.getmtime)
if ck:
    print(f"  체크포인트 {len(ck)} 개, 마지막 {os.path.basename(ck[-1])} ({dt.datetime.fromtimestamp(os.path.getmtime(ck[-1]), KST):%H:%M} KST)")
if err:
    print(f"  {Rd}train.log 오류 줄:{X} {err[-1][:150]}")

# 같은 epoch 검증 표
print("─" * 86)
cols = [run] + refs
Ms = {r: (M if r == run else metrics(r)) for r in cols}
vds = {r: cfg_val(r) for r in cols}
short = {"natural_ar_vw_state_oneshot_vidctx_causal": "영상문맥·블록인과", "natural_ar_vw_state_oneshot_causal": "블록문맥·블록인과", "natural_ar_vw_state_oneshot_vidctx": "영상문맥·양방향",
         "natural_ar_vw_state_v4vc": "AR + v4 scratch", "natural_ar_vw_state": "기본 AR"}
print(f"{B}학습 중 검증{X} (VOBench late 432 쌍, %): 전체 · 영속 late primitive / realistic (각 72 쌍)  {D}— 감시용, 정식은 e70 시트 채점{X}")
hdr = f"  {'epoch':>5} " + "".join(f"│ {short.get(r, r)[:20]:<20}{'' if vds[r] == 'vobench_v1_late' else '⚠' + vds[r][:6]:>1} " for r in cols)
print(hdr)
show = [e for e in sorted(M) if e in (1, 2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 70) or e > max(M) - 5] if M else []
def cell(r, e):
    d = Ms[r].get(e)
    if not d or "val" not in d: return f"{'—':<21}"
    v = d["val"]; bt = v.get("by_type", {})
    pp = bt.get("primitive/permanence/late", {}).get("acc"); rp = bt.get("realistic/permanence/late", {}).get("acc")
    s = f"{100 * v['acc']:5.1f}" + (f"  {100 * pp:3.0f}/{100 * rp:3.0f}" if pp is not None else "")
    return f"{s:<21}"
for e in show:
    print(f"  {e:>5} " + "".join(f"│ {cell(r, e)} " for r in cols))
fin = [r for r in refs if Ms[r]]
if fin:
    print(f"  {'최종':>5} " + "".join(f"│ {cell(r, max(Ms[r])) if r != run else '':<21} " for r in cols) + f" {D}(비교 run 마지막 epoch){X}")

# 자동 채점 상태
print("─" * 86)
if os.path.exists(evlog):
    ls = [l.rstrip() for l in open(evlog, errors="ignore") if l.strip()]
    done = [l for l in ls if "DONE" in l]
    st = done[-1] if done else (ls[-1][:120] if ls else "(빈 로그)")
    print(f"{B}자동 채점{X} ({os.path.basename(evlog)}): {st}")
    if not done:
        for cl in sorted(glob.glob(f"{proj}/z_training/logs/chain_full_eval_vast_ar_oneshot*causal.log"), key=os.path.getmtime)[-1:]:
            last = [l.rstrip() for l in open(cl, errors="ignore") if "chain_full_eval" in l]
            if last: print(f"  {D}{os.path.basename(cl)}: {last[-1][:110]}{X}")
else:
    print(f"자동 채점 로그 없음 ({evlog})")
g = [x.split(",") for x in gpu.strip(";").split(";") if x]
print("GPU  " + "  ".join(f"{i.strip()}:{u.strip():>3}% {float(m) / 1024:4.1f}G" for i, m, u in g))
EOF
