#!/bin/bash
# =============================================================================
# 상태 supervision (state_head) PFT 모니터 — vll7, 고정 창 + VOBench late val (2026-10-19)
#
#   watch -c -n 5 bash z_vll7_training/monitor_state_head.sh                        # 도는 run 자동 선택
#   watch -c -n 5 bash z_vll7_training/monitor_state_head.sh phys2d_sup_v2_light_pft
#
# z_vll7_training/monitor.sh (헤더 · 진행 · ETA · epoch · GPU · ckpt) 를 그대로 쓰고,
# 가변 창 전용 줄 ((C,K) 별 · 데이터 비율) 과 IntPhys1 로 고정된 val 줄만 빼고 아래를 덧붙인다:
#   loss        tf / ar 분해 (고정 창 `mask prefix` 형식)
#   상태 head   손실 (×weight) vs 잠재 L1 · rollout 위치 오차 (보임 / 가림 px) · 존재 정확도 · 성분별
#   grad norm   clip 전 값 (train.py 가 head 가 있으면 늘 잰다). clip_grad 와 같이 본다
#   val         VOBench late — 전체 + 위반별 (영속 · 모양 · 색) × (primitive · realistic)
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" >/dev/null
RUN=${1:-}
if [[ -z "$RUN" ]]; then
  RUN=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' | head -1 | sed 's/^train_//')
  [[ -z "$RUN" ]] && RUN=$(ls -t "$VLL7_RUNS" 2>/dev/null | head -1)
fi
D="$VLL7_RUNS/$RUN"
[[ -d "$D" ]] || { echo "run 없음: $D   (있는 것: $(ls "$VLL7_RUNS" 2>/dev/null | tr '\n' ' '))"; exit 1; }

# 가변 창 전용 줄과 IntPhys1 고정 val 줄을 뺀다 (아래에서 다시 쓴다). 마지막 ━ 줄도 빼고 끝에 다시 긋는다
MONITOR_NO_LOSS=1 bash "$(dirname "${BASH_SOURCE[0]}")/monitor.sh" "$RUN" 2>/dev/null \
  | grep -v 'IntPhys1 dev 60' \
  | awk '!(/^.*─────/ && prev ~ /─────/) {print} {prev=$0}'

"$VJEPA2_PY" - "$D" <<'EOF'
import json, os, re, sys
from collections import OrderedDict
D = sys.argv[1]
col = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, a): return f"\033[{a}m{s}\033[0m" if col else str(s)
B, DIM, GRN, YEL, RED, MAG = "1", "2", "32", "33", "31", "35"
W = 78
def line(ch="─"): print(c(ch * W, DIM))

cfg = {}
try:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(D, "config.yaml"))) or {}
except Exception:
    pass
SH = cfg.get("state_head") or {}
clipg = (cfg.get("optimization") or {}).get("clip_grad")
tl = os.path.join(D, "train.log")

# ── loss (고정 창) + 상태 head ───────────────────────────────────────────────
# [e, i] loss L (avg A) [tf T ar R R=n] [head H | ex a/b pos c/d dir e/f spd g/h | ar px vis V hid Hd] gn G | mask prefix CxKy
pat = re.compile(
    r"\[(\d+), +(\d+)\] loss ([\d.]+) \(avg ([\d.]+)\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\]"
    r"(?: \[head ([\d.]+) \| ex ([\d.]+)/([\d.]+) pos ([\d.]+)/([\d.]+) dir ([\d.]+)/([\d.]+) spd ([\d.]+)/([\d.]+)"
    r" \| ar px vis ([\d.na]+) hid ([\d.na]+)\])?(?: id ([\d.na]+))?(?: conc ([\d.na]+))?(?: gn ([\d.naif]+))? \| mask (\S+) C(\d+)K(\d+)")
seen = OrderedDict()
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen[(m.group(1), m.group(2))] = m.groups()
S = list(seen.values())

def f(x):
    try: return float(x)
    except Exception: return float("nan")

if S:
    g = S[-1]
    print(f" {c('loss', B):>6}   {c(f'{f(g[2]):.4f}', B)}  =  tf {f(g[4]):.4f}  +  ar {f(g[5]):.4f}"
          f"   {c(f'(epoch 평균 {f(g[3]):.4f})', DIM)}   {c(f'지금 {g[21]} C{g[22]}K{g[23]} R={g[6]}', DIM)}")
    H = [x for x in S if x[7] is not None]
    if H:
        rec = H[-8:]
        def a(i):
            v = [f(x[i]) for x in rec if x[i] not in (None, "nan")]
            return sum(v) / len(v) if v else float("nan")
        wgt = SH.get("weight", 1.0)
        lat = a(4) + a(5)
        # groups(): 7 head · 8/9 ex · 10/11 pos · 12/13 dir · 14/15 spd · 16 px_vis · 17 px_hid
        #           · 18 id · 19 conc (single head 만) · 20 gn · 21 mask · 22 C · 23 K
        print(f" {c('head', B):>6}   {c(SH.get('type', 'attn'), B + ';' + MAG)}  손실 {c(f'{a(7):.3f}', B)} (×{wgt:g})  vs  잠재 L1 {lat:.3f}"
              f"   {c(f'(최근 {len(rec)} step 평균)', DIM)}")
        print(f"         rollout 위치 오차  보임 {c(f'{a(16):.1f} px', B + ';' + GRN)} · 가림 {c(f'{a(17):.1f} px', B + ';' + GRN)}"
              f"   {c('(1 칸 = 18 px = 물체 반폭)', DIM)}")
        ex_name = "물체CE" if SH.get("type") == "single" else "존재BCE"
        print(f"         성분 (tf/ar)  {ex_name} {a(8):.3f}/{a(9):.3f}  위치 {a(10):.4f}/{a(11):.4f}"
              f"  방향 {a(12):.3f}/{a(13):.3f}  속력 {a(14):.3f}/{a(15):.3f}")
        if any(x[18] is not None for x in rec):
            # single head: 49 지선다 정체 정확도 · attention 집중도 (w_max x 256, 균등 = 1)
            print(f"         정체 (49 지선다) {c(f'{100 * a(18):.0f} %', B + ';' + GRN)}"
                  f"   attention 집중도 {c(f'{a(19):.1f}x', B)}   {c('(균등 = 1. 1 에 붙으면 토큰 평균 = 기본값 읽기)', DIM)}")
        gns = [f(x[20]) for x in S if x[20] not in (None, "nan")]
        if gns:
            hot = sum(1 for v in gns if clipg and v > float(clipg))
            tag = c(f"{hot}/{len(gns)} step 이 clip {clipg} 초과", YEL) if clipg and hot else c("clip 없음", YEL) if not clipg else "clip 안 걸림"
            print(f"         grad norm (clip 전)  최근 {c(f'{gns[-1]:.2f}', B)} · 최근 50 최대 {max(gns[-50:]):.2f} · 전체 최대 {max(gns):.2f}   {tag}")

# ── epoch 별 head 추이 + VOBench late val ───────────────────────────────────
mrows = []
mp = os.path.join(D, "metrics.jsonl")
if os.path.isfile(mp):
    for ln in open(mp, encoding="utf-8"):
        try: mrows.append(json.loads(ln))
        except Exception: pass
hh = [(r["epoch"], r.get("train_head") or {}) for r in mrows if r.get("train_head")]
if hh:
    print(f" {c('head추이', B):>6} " + "  ".join(
        f"e{e}:{h.get('head', float('nan')):.2f}/{h.get('ar_pos_px_vis', float('nan')):.0f}·{h.get('ar_pos_px_hid', float('nan')):.0f}px"
        for e, h in hh[-7:]) + c("   (손실/위치 보임·가림)", DIM))

vs = [(r["epoch"], r["val"]) for r in mrows if (r.get("val") or {}).get("acc") is not None]
if not vs and os.path.isfile(tl):
    # metrics.jsonl 은 epoch 끝에 쓰인다 — at_start val (epoch 0) 은 아직 없으므로 train.log 에서 읽는다
    vpat = re.compile(r"\[epoch (\d+)\] val acc ([\d.]+)% \(n_pair (\d+)")
    bpat = re.compile(r"([\w/]+): ([\d.]+)")
    lg = open(tl, errors="ignore").read().splitlines()
    tmp = OrderedDict()
    for i, ln in enumerate(lg):
        m = vpat.search(ln)
        if not m:
            continue
        d = {"acc": float(m.group(2)) / 100, "n_pair": int(m.group(3)), "by_type": {}}
        nxt = lg[i + 1] if i + 1 < len(lg) else ""
        if "/late" in nxt or ("/" in nxt and "[tf]" not in nxt):
            d["by_type"] = {k: {"acc": float(v) / 100} for k, v in bpat.findall(nxt)}
        tmp[int(m.group(1))] = d
    vs = sorted(tmp.items())
vname = (cfg.get("val") or {}).get("dataset", "?")
if vs:
    accs = [(e, 100 * v["acc"]) for e, v in vs]
    best = max(accs, key=lambda x: x[1])
    npair = vs[-1][1].get("n_pair", "?")
    print(f" {c('val', B):>6}   {c(vname, B + ';' + MAG)} {npair} 쌍  "
          + "  ".join(f"e{e}:{a:.1f}" for e, a in accs[-7:])
          + f"   {c(f'최고 e{best[0]} {best[1]:.1f}%', B + ';' + GRN)}"
          + (f"   {c(f'출발 e{accs[0][0]} {accs[0][1]:.1f}%', DIM)}" if len(accs) > 1 else ""))
    bt = vs[-1][1].get("by_type") or {}
    if bt and any("/" in k for k in bt):
        # VOBench late: <objset>/<violation>/late
        viol = ["permanence", "shape", "color"]
        for objset in ("primitive", "realistic"):
            cells = []
            for vi in viol:
                e = bt.get(f"{objset}/{vi}/late")
                cells.append(f"{vi[:4]} {100*e['acc']:5.1f}" if e else f"{vi[:4]}   —  ")
            if any("—" not in x for x in cells):
                print(f"         {objset:<10} " + "  ".join(cells))
    elif bt:
        print("         " + "  ".join(f"{k}: {100*v['acc']:.1f}" for k, v in bt.items()))
else:
    print(f" {c('val', B):>6}   {c(vname, DIM)}  {c('아직 없음 (at_start 는 epoch 0 에 찍힌다)', DIM)}")
print(c("━" * W, DIM))
EOF
