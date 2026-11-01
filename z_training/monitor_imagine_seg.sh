#!/bin/bash
# =============================================================================
# imagine-seg 공정 비교 세 판 (대조군 · 보조 head · in-path) 을 한 화면에 (2026-10-31).
#
#   watch -c -n 30 bash z_training/monitor_imagine_seg.sh
#   bash z_training/monitor_imagine_seg.sh imagine_seg_v1_ctrl imagine_seg_v1_head     # 판 고르기
#
#   ① 진행 · ETA (KST) · step 시간          ② 학습 손실 (epoch 중앙): latent L1 · 점유 손실 · 보이는 / 가려진 물체 칸 재현율 (rank 0)
#   ③ 학습 중 val (VOBench split v2 test late 216 쌍, 2 epoch 마다): 전체 · 영속 (단순 도형 / 실제 물체) · margin
#   ④ GPU
# 설계 · 계획: new_idea/imagine-seg/FAIR_COMPARISON_V1_2026-10-31.md
# =============================================================================
RUNS=${*:-"imagine_seg_v1_ctrl imagine_seg_v1_head imagine_seg_v1_inpath"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
/venv/vjepa/bin/python - "$ROOT/z_training/runs" $RUNS <<'EOF'
import os, re, sys, json, datetime, statistics as st, subprocess, yaml, unicodedata
RR, runs = sys.argv[1], sys.argv[2:]
B, DIM, X = "\033[1m", "\033[2m", "\033[0m"
GRN, YEL, RED, BLU, CYN, GRY = "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[36m", "\033[90m"
now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=9)
def wl(s):
    n, esc = 0, False
    for ch in s:
        if ch == "\033": esc = True
        elif esc: esc = ch != "m"
        else: n += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return n
def pad(s, n, r=False): s = str(s); f = " " * max(0, n - wl(s)); return f + s if r else s + f
def med(xs): xs = [x for x in xs if x == x]; return st.median(xs) if xs else float("nan")
def f(x, d=3): return "—" if not (isinstance(x, (int, float)) and x == x) else f"{x:.{d}f}"
def head(t): print(f"\n {B}{BLU}▌{X}{B} {t}{X}")
short = lambda r: r.replace("imagine_seg_v1_", "")
print(f"{BLU}{'═' * 96}{X}\n {B}imagine-seg 공정 비교{X}   {DIM}{now:%Y-%m-%d %H:%M:%S} KST{X}\n{BLU}{'═' * 96}{X}")

info = {}
for r in runs:
    R = os.path.join(RR, r)
    if not os.path.exists(f"{R}/config.yaml"):
        info[r] = None; continue
    cfg = yaml.safe_load(open(f"{R}/config.yaml")); E = int(cfg["optimization"]["epochs"])
    log = open(f"{R}/train.log", errors="ignore").read() if os.path.exists(f"{R}/train.log") else ""
    m = re.search(r"\| ipe (\d+)", log); ipe = int(m.group(1)) if m else 354
    rows = [l.split(",") for l in open(f"{R}/log_r0.csv").read().splitlines()[1:] if l and l[0].isdigit()] if os.path.exists(f"{R}/log_r0.csv") else []
    sup = {}
    for mm in re.finditer(r"\[(\d+),\s+\d+\] loss .*? lat ([\d.]+) occ ([\d.]+) vis R ([\d.na]+) P ([\d.na]+) hid R ([\d.na]+) P ([\d.na]+)", log):
        sup.setdefault(int(mm.group(1)), []).append([float(x) for x in mm.groups()[1:]])
    val = {}
    for mm in re.finditer(r"\[epoch (\d+)\] val acc ([\d.]+)% \(n_pair (\d+).*?margin ([+-][\d.]+)", log):
        val[int(mm.group(1))] = dict(acc=float(mm.group(2)), margin=float(mm.group(4)))
    if os.path.exists(f"{R}/metrics.jsonl"):
        for l in open(f"{R}/metrics.jsonl"):
            d = json.loads(l); v = d.get("val") or {}
            if v.get("by_type"):
                g = lambda k: [round(100 * x["acc"], 1) for t, x in sorted(v["by_type"].items()) if k in t]
                val.setdefault(d["epoch"], {}).update(perm=g("permanence"), shape=g("shape"), color=g("color"))
    alive = subprocess.run(["pgrep", "-f", f"runs/{r}/config.yaml"], capture_output=True).returncode == 0
    done_l = "] done" in log
    info[r] = dict(E=E, ipe=ipe, rows=rows, sup=sup, val=val, alive=alive, done=done_l)

head("진행")
print(f"   {pad('판', 9)} {pad('상태', 8)} {pad('epoch', 8)} {pad('step', 14)} {pad('step 시간', 10)} {pad('끝 (KST)', 14)}")
print(f"   {GRY}{'─' * 70}{X}")
for r in runs:
    I = info[r]
    if I is None:
        print(f"   {pad(short(r), 9)} {GRY}아직 없음{X}"); continue
    rows = I["rows"]
    if not rows:
        print(f"   {pad(short(r), 9)} {YEL}준비 중{X}"); continue
    ep, it = int(rows[-1][0]), int(rows[-1][1]); done = (ep - 1) * I["ipe"] + it + 1; tot = I["E"] * I["ipe"]
    ms = med([float(x[5]) for x in rows[-100:]]); eta = (tot - done) * ms / 1000
    stt = f"{GRN}● 도는 중{X}" if I["alive"] else (f"{CYN}✔ 끝{X}" if I["done"] else f"{RED}✖ 멈춤{X}")
    end = "—" if not I["alive"] else f"{now + datetime.timedelta(seconds=eta):%m-%d %H:%M}"
    print(f"   {pad(short(r), 9)} {pad(stt, 8)} {pad(f'{ep}/{I[chr(69)]}', 8)} {pad(f'{done:,}/{tot:,}', 14)} {pad(f'{ms / 1000:.2f} s', 10)} {end}")

head(f"학습 손실 (epoch 중앙, rank 0)  {DIM}점유 손실 = BCE + dice (대조군은 가중 0 이라 head 가 안 배운다){X}")
print(f"   {pad('판', 9)} {pad('epoch', 6)} {pad('latent', 8, True)} {pad('점유', 8, True)} {pad('보임 R', 8, True)} {pad('가려짐 R', 9, True)}")
print(f"   {GRY}{'─' * 54}{X}")
for r in runs:
    I = info[r]
    if not I or not I["sup"]: continue
    eps = sorted(I["sup"]); show = eps[:1] + eps[-3:] if len(eps) > 4 else eps
    for e in show:
        v = [med([x[j] for x in I["sup"][e]]) for j in range(6)]
        print(f"   {pad(short(r), 9)} {pad('e' + str(e), 6)} {f(v[0], 4):>8s} {f(v[1]):>8s} {f(v[2], 2):>8s} {f(v[4], 2):>9s}")
    print()

head(f"학습 중 val — VOBench held-out late 216 쌍  {DIM}(e0 = 학습 전 릴리즈 60.6 · 표준오차 약 ±3.3pt · 영속 열은 36 쌍씩 ±8pt){X}")
eps = sorted({e for r in runs if info[r] for e in info[r]["val"]})
print(f"   {pad('판', 9)} " + " ".join(pad(f"e{e}", 7, True) for e in eps[-8:]) + f"   {pad('영속 (도형/실물, 최근)', 22)}")
print(f"   {GRY}{'─' * (12 + 8 * len(eps[-8:]) + 24)}{X}")
for r in runs:
    I = info[r]
    if not I: continue
    cells = []
    for e in eps[-8:]:
        a = I["val"].get(e, {}).get("acc")
        c = (GRN if a >= 66 else RED if a < 57 else "") if a is not None else ""
        cells.append(pad(f"{c}{a:.1f}{X}" if a is not None else "—", 7, True))
    last = [I["val"][e] for e in sorted(I["val"]) if "perm" in I["val"][e]]
    pm = "/".join(f"{x:.0f}" for x in last[-1]["perm"]) if last else "—"
    print(f"   {pad(short(r), 9)} " + " ".join(cells) + f"   {pm}")

head("GPU")
gl = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
print("   " + "   ".join(f"{i.split(',')[0].strip()}: {float(i.split(',')[1]) / 1024:4.0f}G {int(i.split(',')[2]):3d}%" for i in gl))
print(f"{BLU}{'─' * 96}{X}")
EOF
