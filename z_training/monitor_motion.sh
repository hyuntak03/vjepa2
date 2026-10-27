#!/bin/bash
# =============================================================================
# motion (MotionJEPA DISReg 이식) 학습 모니터 — 2026-10-24
#
#   watch -c -n 10 bash z_training/monitor_motion.sh                                   # 기본 natural_ar_vw_motion_v3
#   watch -c -n 10 bash z_training/monitor_motion.sh natural_ar_vw_motion_v3 natural_ar_vw_motion natural_ar_vw_motion_v2
#                                                    ^ 볼 run                   ^ 같은 epoch 비교 run (생략하면 v1 · v2)
#
# 위: monitor_ar.sh 의 진행 · 시간 · ETA · VOBench late 검증 줄
# 아래 (학습 로그 train.log 의 "mo" 항목에서 다시 계산):
#   비율은 모두 d 분산 대비 오차 (낮을수록 좋음). d = 차분 encoder 의 움직임 코드
#     정답 쌍   = 차분 예측기 (h_b, h_{b+1})   — d 가 encoder 와 맞는 정도 (진단 D: v1 설정 0.91 · v2/v3 설정 0.53)
#     TF · 굴린 = 고정 차분 예측기 (앞 블록, 예측) — 예측에 움직임이 담긴 정도
#     멈춤      = 고정 차분 예측기 (h_b, h_b)     — "바뀐 것 없음" 기준
#   정상: 정답 쌍 < TF ≤ 굴린 < 멈춤.  TF 나 굴린이 정답 쌍보다 낮으면 판독기 속임 의심 (⚠)
#   잠재 L1 은 같은 epoch 기준 AR (natural_ar_vw_state) 과 나란히
#   독립 측정 (광류 판독, motion_where_probe.py) 결과 json 이 run 폴더에 있으면 3D 합성 다음 움직임 줄을 붙인다
# =============================================================================
RUN=${1:-natural_ar_vw_motion_v3}; shift
CMP=${*:-natural_ar_vw_motion natural_ar_vw_motion_v2}
cd "$(dirname "$0")/.." || exit 1
bash z_training/monitor_ar.sh "$RUN" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^ (run|epoch|step|시간|ETA|val) |^   (지금|추이|primitive|realistic) " | grep -v "평균 loss"
python3 - "$RUN" $CMP <<'EOF'
import collections, glob, json, os, re, sys
B, Y, R, G, D, N = "\033[1m", "\033[33m", "\033[31m", "\033[32m", "\033[2m", "\033[0m"
run, cmps = sys.argv[1], sys.argv[2:]
PAT = re.compile(r"\[(\d+), +(\d+)\].*lat ([\d.]+) mo sig ([\d.]+) fit ([\d.]+) tf ([\d.]+) ar ([\d.]+) copy ([\d.]+) dvar ([\d.]+) gn ([\d.]+) mgn ([\d.]+)")
K = "lat sig fit tf ar copy dvar gn mgn".split()

def parse(r):
    p = f"z_training/runs/{r}/train.log"
    if not os.path.exists(p):
        return {}, [], {}
    ep, last, val = collections.defaultdict(list), [], {}
    for l in open(p, errors="ignore"):
        m = PAT.search(l)
        if m:
            v = [float(x) for x in m.groups()[2:]]
            ep[int(m.group(1))].append(v); last.append(v)
        mv = re.search(r"\[epoch (\d+)\] val acc ([\d.]+)%.*margin ([+-][\d.]+)", l)
        if mv and int(mv.group(1)) not in val:
            val[int(mv.group(1))] = (float(mv.group(2)), float(mv.group(3)))
    avg = lambda rows: dict(zip(K, [sum(x[i] for x in rows) / len(rows) for i in range(len(K))]))
    return {e: avg(v) for e, v in ep.items()}, (avg(last[-50 * 8:]) if last else None), val

def base_lat():
    p = "z_training/runs/natural_ar_vw_state/train.log"; out = {}
    for l in open(p, errors="ignore") if os.path.exists(p) else []:
        m = re.search(r"epoch (\d+) avg loss [\d.]+ \(tf ([\d.]+) ar ([\d.]+)\)", l)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = float(m.group(2)) + float(m.group(3))
    return out

def rat(d):
    v = d["dvar"] or float("nan")
    return d["fit"] / v, d["tf"] / v, d["ar"] / v, d["copy"] / v

ep, last, val = parse(run)
bl = base_lat()
print("──────────────────────────────────────────────────────────────────────────────")
if not ep:
    print(f" motion  {run}: 학습 로그에 motion 줄이 아직 없다"); sys.exit()
f, t, a, c = rat(last)
warn = f" {R}⚠ 예측 쌍이 정답 쌍보다 낮음 — 판독기 속임 의심{N}" if min(t, a) < f else ""
print(f" {B}motion{N}  최근 50 step (전 rank)   d 분산 대비 오차 (낮을수록 좋음)")
print(f"   정답 쌍 {B}{f:.3f}{N}   TF {B}{t:.3f}{N}   굴린 {B}{a:.3f}{N}   멈춤 {c:.3f}   "
      f"| d 분산 {last['dvar']:.3f}  SIGReg {last['sig']:.2f}  grad norm predictor {last['gn']:.2f} · motion {last['mgn']:.1f}{warn}")
gap = c - f
if gap > 1e-6:
    print(f"   멈춤→정답 간격 중 메운 몫   TF {G}{100 * (c - t) / gap:5.1f} %{N}   굴린 {G}{100 * (c - a) / gap:5.1f} %{N}   {D}(0 % = 멈춘 세계, 100 % = 정답 쌍만큼){N}")
print(f" {B}epoch 평균{N}   잠재 L1 (기준 AR) · 정답 쌍 / TF / 굴린 / 멈춤 · predictor grad · VOBench late 검증 (margin)")
for e in sorted(ep):
    d = ep[e]; f, t, a, c = rat(d)
    v = val.get(e)
    vs = f"{v[0]:5.1f} ({v[1]:+.4f})" if v else "    —"
    b = bl.get(e)
    print(f"   e{e:<3d} L1 {d['lat']:.3f} ({b:.3f})" if b else f"   e{e:<3d} L1 {d['lat']:.3f} (  —  )", end="")
    print(f"   {f:.3f} / {t:.3f} / {a:.3f} / {c:.3f}   gn {d['gn']:4.2f}   val {vs}")
if cmps:
    print(f" {B}같은 epoch 비교{N}   정답 쌍 / TF · 잠재 L1 · 검증")
    es = sorted(ep)[-3:]
    for r in [run] + cmps:
        e2, _, v2 = parse(r)
        cells = []
        for e in es:
            if e in e2:
                f, t, a, c = rat(e2[e]); vv = v2.get(e)
                cells.append(f"e{e} {f:.2f}/{t:.2f} · {e2[e]['lat']:.3f} · {vv[0]:.1f}" if vv else f"e{e} {f:.2f}/{t:.2f} · {e2[e]['lat']:.3f} · —")
            else:
                cells.append(f"e{e} —")
        print(f"   {r[-24:]:24s}  " + "   ".join(cells))
js = sorted(glob.glob(f"z_training/runs/{run}/motion_where_probe*.json"), key=os.path.getmtime)
if js:
    print(f" {B}독립 측정{N} (광류 판독, 3D 합성 다음 움직임: predictor 층 1~12 평균 R² · 정답 자를 예측 출력에 씌움 쌍)   {D}{os.path.basename(js[-1])}{N}")
    r = json.load(open(js[-1])).get("phys3d", {}).get("dnext", {})
    for nm in json.load(open(js[-1])).get("models", {}):
        ls = [r.get(f"{nm}|{i}|pred", {}).get("r2") for i in range(1, 13)]
        ls = [x for x in ls if x is not None]
        tr = r.get(f"transfer|{nm}|pair", {}).get("transfer")
        if ls:
            print(f"   {nm:14s} 층 평균 {sum(ls) / len(ls):.3f}   이식 {tr:.3f}" if tr is not None else f"   {nm:14s} 층 평균 {sum(ls) / len(ls):.3f}")
EOF
