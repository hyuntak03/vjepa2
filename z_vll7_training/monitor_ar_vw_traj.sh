#!/usr/bin/env bash
# ar_vw_shortctx_traj 전용 모니터 — monitor.sh 위에 **궤적 보조 손실** 판을 얹는다 (2026-10-22).
#
# 읽는 법 (이게 전부다):
#   모든 값은 **"0 을 예측했을 때" 로 나눈 상대값**이다. 1.00 = 0 예측과 같음.
#   head  = flow_head 가 낸 변위의 L1 (상대)
#   등속  = 파라미터 0 기준선. 진짜 첫 지평선 변위를 비례로 늘린 것 (CLAUDE.md §12 Q5 복사 기준선)
#   ✅ 는 head < min(1.00, 등속) 일 때만 — 그때만 "0 예측도 등속도 아닌 무언가를 배웠다" 고 말할 수 있다.
#   감독 지평선은 1 · 2 · 4 블록 (0.5 · 1 · 2 s) 이고 셋 다 제대로 된 등속 기준선을 갖는다
#   (반 블록짜리 h 1 은 기준선 재료로만 쓰고 감독하지 않는다 — flow_head.supervise).
NAME=${1:-ar_vw_shortctx_traj}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MONITOR_NO_LOSS=1 bash "$ROOT/z_vll7_training/monitor.sh" "$NAME" | awk '!/^ *━/'
LOG="$ROOT/z_vll7_training/runs/$NAME/train.log"
[ -f "$LOG" ] || exit 0
/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python - "$LOG" <<'PY'
import re, sys, collections
pat = re.compile(r"\[(\d+), +(\d+)\] loss ([\d.]+) .*?\[tf ([\d.]+) ar ([\d.]+) R=(\d+)\] "
                 r"\[traj ([\d.]+) tf ([\d.]+) ar ([\d.]+) cv ([\d.]+) val ([\d.]+) h (.*?)\]")
ep = collections.defaultdict(lambda: collections.defaultdict(list))
last = None
for ln in open(sys.argv[1], errors="ignore"):
    m = pat.search(ln)
    if not m:
        continue
    e = int(m.group(1))
    ep[e]["lat"].append(float(m.group(3))); ep[e]["ar_l"].append(float(m.group(9)))
    ep[e]["ar_c"].append(float(m.group(10))); ep[e]["val"].append(float(m.group(11)))
    for tok in m.group(12).split():
        h, v = tok.split(":"); a, b = v.split("/")
        ep[e][f"h{h}"].append(float(a)); ep[e][f"c{h}"].append(float(b))
    last = m
if not ep:
    print("  (궤적 기록 없음)"); sys.exit()
HS = sorted({int(k[1:]) for k in ep[max(ep)] if k.startswith("h")})
S = {1: "0.5블록", 2: "1블록", 4: "2블록", 8: "4블록", 16: "8블록"}
B = "\033[1m"; D = "\033[2m"; G = "\033[32m"; Y = "\033[33m"; R = "\033[0m"
avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{D}{'─'*86}{R}")
print(f" {B}궤적{R}  보조 손실 (0 예측 = 1.00 기준의 상대 L1) · 유효 칸 {avg(ep[max(ep)]['val']):.0%}"
      f"   {D}지금 traj {float(last.group(7)):.3f} · latent {float(last.group(3)):.3f}{R}")
print(f" {D}      앵커 = 그 블록 자신. 'h 장 뒤 이 내용이 어디 가 있나' (블록 간 누적 변위){R}")
hdr = "".join(f"{S.get(h, str(h)):>11}" for h in HS)
print(f" {B}{'지평선':<10}{R}{hdr}")
eps = sorted(ep)
show = eps[-6:] if len(eps) > 6 else eps
for e in show:
    row = "".join(f"{avg(ep[e][f'h{h}']):>11.3f}" for h in HS)
    print(f" {D}ep {e:<7}{R}{row}")
print(f" {B}{'등속':<10}{R}" + "".join(f"{avg(ep[max(ep)][f'c{h}']):>11.3f}" for h in HS))
mark = []
for h in HS:
    hv, cv = avg(ep[max(ep)][f"h{h}"]), avg(ep[max(ep)][f"c{h}"])
    tgt = min(1.0, cv) if cv > 0 else 1.0
    mark.append(f"{G}✅{R}" if hv < tgt else f"{Y}··{R}")
print(f" {B}{'판정':<10}{R}" + "".join(f"{m:>20}" for m in mark))
PY
