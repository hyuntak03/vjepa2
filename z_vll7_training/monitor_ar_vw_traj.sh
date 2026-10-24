#!/usr/bin/env bash
# 궤적(optical flow) 보조 손실 run 전용 모니터 — monitor.sh 위에 두 판을 얹는다 (2026-10-22).
#
#   ① 궤적 판 : 지평선별 head vs **등속 외삽** (파라미터 0 기준선, CLAUDE.md §12 Q5)
#   ② 대조 판 : 같은 epoch 에서 보조 손실 없는 run 과 latent 손실 · IntPhys1 을 나란히
#               — 보조 손실이 본 목표를 깎고 있는지가 이 실험의 성패를 가른다
#
# 읽는 법 (이게 전부다):
#   모든 궤적 값은 **"0 을 예측했을 때" 로 나눈 상대 L1**. 1.00 = 0 예측과 같음.
#   head  = flow head 가 낸 변위의 L1 (상대) · 등속 = 첫 지평선의 진짜 변위를 비례로 늘린 것 (파라미터 0)
#   ✅ 는 head < 0.95 x min(1.00, 등속) 일 때만.
#   ⚠️ 정정 (2026-10-22) — 처음엔 문턱을 min(1.00, 등속) 으로 뒀다. 2·4 블록에서는 등속이 1.00 보다
#      **나쁘므로** 문턱이 그냥 1.00 이 되어 **0 만 예측해도 ✅ 가 떴다** (실제로 그랬다: 세 칸 모두 0.97).
#      여유 5 % 를 두고, 지평선끼리 값이 거의 같으면 (퍼짐 < 0.02) "0 수렴" 경고를 따로 찍는다 —
#      진짜로 배웠다면 쉬운 칸 (1 블록) 과 어려운 칸 (4 블록) 의 값이 **달라야** 한다.
#
#   사용법:  bash z_vll7_training/monitor_ar_vw_traj.sh [NAME] [대조군NAME]
NAME=${1:-ar_vw_shortctx_trajdiff}
BASE=${2:-ar_vw_shortctx}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MONITOR_NO_LOSS=1 bash "$ROOT/z_vll7_training/monitor.sh" "$NAME" | awk '!/^ *━/'
/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python - "$ROOT/z_vll7_training/runs/$NAME" "$ROOT/z_vll7_training/runs/$BASE" <<'PY'
import json, os, sys, collections

RUN, BASE = sys.argv[1], sys.argv[2]
B = "\033[1m"; D = "\033[2m"; G = "\033[32m"; Y = "\033[33m"; Rr = "\033[31m"; R = "\033[0m"


def rows(run):
    p = os.path.join(run, "metrics.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for ln in open(p, errors="ignore"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        e = r.get("epoch")
        if e is None:
            continue
        out[e] = r
    return out


cur, base = rows(RUN), rows(BASE)
if not cur:
    print("  (기록 없음)"); sys.exit()
last = cur[max(cur)]
hd = last.get("train_head") or {}
HS = sorted(int(k[3:]) for k in hd if k.startswith("f_l"))          # 지평선 (장)
if not HS:
    print("  (궤적 기록 없음)"); sys.exit()

# 어떤 입력을 쓰는 팔인지 — _resolved.yaml 에서 읽는다 (로그 해석이 달라진다)
inp = "?"
for nm in ("config.yaml", "params-train.yaml", "_resolved.yaml"):
    ry = os.path.join(RUN, nm)
    if os.path.exists(ry):
        for ln in open(ry, errors="ignore"):
            if ln.strip().startswith("input:"):
                inp = ln.split(":", 1)[1].split("#")[0].strip(); break
    if inp != "?":
        break
WHAT = {"diff": "인접(간격 g 블록) 출력의 **차** p_{b+g} − p_b 를 읽어 그 구간 이동량",
        "token": "블록 토큰 하나 p_b 를 읽어 h 장 뒤 누적 변위"}.get(inp, inp)

print(f"{D}{'─'*86}{R}")
print(f" {B}궤적{R}  입력 {B}{inp}{R} — {WHAT}")
print(f" {D}      상대 L1 (1.00 = 0 예측) · 유효 칸 {hd.get('f_valid', float('nan')):.0%}"
      f" · 지금 traj {hd.get('flow', float('nan')):.3f} · latent {hd.get('latent', float('nan')):.3f}{R}")
# ⚠️ epoch 평균 (metrics.jsonl) 은 쌍이 안 나오는 step 의 nan 에 오염될 수 있다 —
#    궤적 값은 train.log 의 step 단위에서 직접 모으고 nan 을 뺀다.
import re as _re
pat = _re.compile(r"\[(\d+), +\d+\].*?h ((?:\d+:[\d.naN]+/[\d.naN]+ ?)+)\]")
per = collections.defaultdict(lambda: collections.defaultdict(list))
lg = os.path.join(RUN, "train.log")
if os.path.exists(lg):
    for ln in open(lg, errors="ignore"):
        m = pat.search(ln)
        if not m:
            continue
        e = int(m.group(1))
        for tok in m.group(2).split():
            hh, vv = tok.split(":"); a, b = vv.split("/")
            for key, val in ((f"l{hh}", a), (f"c{hh}", b)):
                try:
                    f = float(val)
                except ValueError:
                    continue
                if f == f:
                    per[e][key].append(f)
avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
eps = sorted(per) or sorted(cur)
print(f" {B}{'지평선':<10}{R}" + "".join(f"{str(h//2)+' 블록':>11}" for h in HS))
for e in eps[-6:]:
    print(f" {D}ep {e:<7}{R}" + "".join(f"{avg(per[e][f'l{h}']):>11.3f}" for h in HS))
LASTC = {h: avg(per[eps[-1]][f"c{h}"]) for h in HS} if eps else {}
LASTL = {h: avg(per[eps[-1]][f"l{h}"]) for h in HS} if eps else {}
print(f" {B}{'등속':<10}{R}" + "".join(f"{LASTC.get(h, float('nan')):>11.3f}" for h in HS))
mark = []
for h in HS:
    hv, cv = LASTL.get(h, float("nan")), LASTC.get(h, float("nan"))
    tgt = 0.95 * (min(1.0, cv) if cv == cv and cv > 0 else 1.0)
    mark.append(f"{G}✅{R}" if hv == hv and hv < tgt else f"{Y}··{R}")
print(f" {B}{'판정':<10}{R}" + "".join(f"{m:>20}" for m in mark) + f"   {D}문턱 = 0.95 x min(1, 등속){R}")
curv = [LASTL.get(h, float("nan")) for h in HS]
if all(v == v for v in curv) and max(curv) - min(curv) < 0.02:
    print(f" {Y}⚠ 지평선끼리 차이가 {max(curv)-min(curv):.3f} 뿐 — 지평선과 무관한 한 해 (= 거의 0) 에 수렴한 모습이다.{R}")
    print(f" {D}  진짜로 운동을 배웠다면 1 블록 (쉬움) 과 {HS[-1]//2} 블록 (어려움) 의 값이 달라야 한다.{R}")

# ── 대조 판 ───────────────────────────────────────────────────────────────────
if base:
    print(f"{D}{'─'*86}{R}")
    print(f" {B}대조{R}  보조 손실 없는 run {B}{os.path.basename(BASE)}{R} 과 **같은 epoch** 비교")
    print(f" {D}      latent = 보조 손실을 뺀 본 손실. 이게 대조군보다 꾸준히 높으면 보조 손실이 본 목표를 깎는 것이다{R}")
    print(f" {Y}      ⚠ 한 변수 대조가 아니다 — 대조군은 snap_start 가 없어 시간 위상 표본이 다르다.{R}")
    print(f" {D}        깨끗한 대조군은 같은 config 에서 손실만 끈 것이다 (config 주석의 SET 명령).{R}")
    print(f" {B}{'ep':>4}{R}  {'latent (이 run)':>15} {'대조군':>9} {'차':>8}   |  {'IntPhys1':>9} {'대조군':>8} {'차':>7}")
    both = [e for e in sorted(cur) if e in base][-8:]      # eps 는 train.log 기준이라 진행 중 epoch 이 섞인다
    dl = dv = 0.0; n = 0
    for e in both:
        h2 = cur[e].get("train_head") or {}
        la = h2.get("latent"); lb = base[e].get("train_loss")
        va = ((cur[e].get("val") or {}).get("acc") or 0) * 100
        vb = ((base[e].get("val") or {}).get("acc") or 0) * 100
        if la is None or lb is None:
            continue
        d1, d2 = la - lb, va - vb
        dl += d1; dv += d2; n += 1
        c1 = Rr if d1 > 0.002 else (G if d1 < -0.002 else D)
        c2 = Rr if d2 < -1 else (G if d2 > 1 else D)
        print(f" {e:>4}  {la:>15.4f} {lb:>9.4f} {c1}{d1:>+8.4f}{R}   |  {va:>9.1f} {vb:>8.1f} {c2}{d2:>+7.1f}{R}")
    if n:
        print(f" {B}{'평균':>4}{R}  {'':>15} {'':>9} {Rr if dl/n>0.002 else D}{dl/n:>+8.4f}{R}"
              f"   |  {'':>9} {'':>8} {Rr if dv/n<-1 else D}{dv/n:>+7.1f}{R}")
PY
