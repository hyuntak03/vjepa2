#!/bin/bash
# =============================================================================
# AR (kind=ar, 가변 창) 학습 + chain_ar 모니터 (2026-10-11).
#   watch -c -n 2 bash /workspace/world_model/vjepa2/z_training/monitor_ar.sh            # 지금 도는 run 자동 선택
#   watch -c -n 2 bash /workspace/world_model/vjepa2/z_training/monitor_ar.sh natural_ar_vw
# 화면: monitor.sh (GPU·진행) + ETA (중앙값 s/step) · tf/ar 손실 · 힌트 수 · (C,K) 별 ar 손실 · in-loop val · chain · 채점 요약
# =============================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=${1:-}
if [[ -z "$RUN" ]]; then      # 지금 도는 train_ 세션 (없으면 가장 최근 run)
  RUN=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' | head -1 | sed 's/^train_//')
  [[ -z "$RUN" ]] && RUN=$(ls -t "$ROOT/z_training/runs" | grep -v '^_' | head -1)
fi
D="$ROOT/z_training/runs/$RUN"
PY=/venv/vjepa/bin/python
[[ -x "$PY" ]] || PY=python3

# 기본 monitor.sh 의 '설정' 줄은 고정 batch·창을 가정해 가변 창 run 에서 틀린 값을 낸다 → 빼고 아래에서 다시 쓴다
TZ=Asia/Seoul bash "$ROOT/z_training/monitor.sh" "$RUN" 2>/dev/null | grep -v '^ ETA ' | grep -v '^ 설정 ' | grep -v '^ val ' | grep -v '^ C 별 '

"$PY" - "$D" "$ROOT" "$RUN" <<'EOF'
import csv, json, os, re, statistics as st, sys, subprocess
from datetime import datetime, timedelta, timezone
D, ROOT, RUN = sys.argv[1:4]
B, Y, N = "\033[1m", "\033[33m", "\033[0m"
print("─" * 78)
f = os.path.join(D, "log_r0.csv")
total = None
try:
    import yaml
    c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
    total = int(c["optimization"]["ipe"]) * int(c["optimization"]["epochs"])
    hint_cfg = c.get("hint")
except Exception:
    hint_cfg = None
try:
    vw = c["data"]["var_window"]
    wts = " : ".join(f"{d['name']} {d.get('weight', 1.0)}" for d in vw["datasets"])
    pc = c["model"].get("predictor") or {}
    arm = (f"kind={pc.get('kind')}" + (f" 되먹임={pc['ar_feedback']}" if pc.get("ar_feedback") else "")
           + (" 입력=영상(target_encoder 문맥 pass) 정답=창(전체 pass) TF=문맥만" if pc.get("ar_input") == "video" else "")
           + (f" 문맥={pc.get('ctx_encoder') or c['model'].get('context_encoder_key', 'encoder')}×{pc.get('ctx_input', 'video')}" if pc.get("kind") == "ctx_state" else "")
           + (f" 타깃={pc['ctx_target']}" if pc.get("ctx_target") else "")
           + (" gating=on" if pc.get("gating_residual") else "")
           + (f" tf_span={c['loss']['tf_span']}" if c["loss"].get("tf_span") else "")
           + (f" tf_weight={c['loss']['tf_weight']}" if "tf_weight" in c["loss"] and pc.get("kind") == "ctx_state" else ""))
    print(f" {B}설정{N}  {arm} · 가변 창 (fps 4, 패딩 없음) · 문맥 C {vw['context_frames']} 장 × 예측 K {vw['predict_frames']} 장 · "
          f"rank 당 B = {vw['token_budget']} // (C+K) [{vw.get('b_min')}~{vw.get('b_max')}] · "
          f"데이터 {wts} · lr {c['optimization']['lr']} (WSD, 마지막 {c['optimization'].get('anneal_epochs', 0)} ep 감소)")
    for key in ("fixed", "boundary"):
        if vw.get(key):
            bx = vw[key]
            print(f"       + {key} {bx.get('name', '')}: step 의 {100 * float(bx['p']):.0f} %"
                  + (f", 경계 정렬 · 문맥 {bx['context_frames']} 장 · 예측 {bx['predict_frames']} 장 · stride {bx['stride']}" if key == "boundary" else ""))
except Exception:
    pass
if os.path.isfile(f):
    # 재개하면 csv 에 헤더가 다시 붙는다 → 숫자 줄만. 진행 step 은 train.log 의 마지막 [epoch, itr] 로 센다
    rows = [r for r in csv.DictReader(open(f)) if r.get("iter-time(ms)", "").replace(".", "", 1).isdigit()]
    done = len(rows)
    try:
        ipe = int(c["optimization"]["ipe"])
        last = re.findall(r"\] \[(\d+), +(\d+)\] loss", open(os.path.join(D, "train.log"), errors="ignore").read())[-1]
        done = (int(last[0]) - 1) * ipe + int(last[1]) + 1
    except Exception:
        pass
    it = [float(r["iter-time(ms)"]) / 1000 for r in rows[-200:]]
    if it and total:
        med = st.median(it); left = (total - done) * med
        eta = (datetime.now(timezone(timedelta(hours=9))) + timedelta(seconds=left)).strftime("%m-%d %H:%M KST")
        print(f" {B}ETA{N}   {int(left // 3600)}:{int(left % 3600 // 60):02d} 남음  (중앙값 {med:.2f} s/step)"
              f"   끝 ≈ {B}{eta}{N}   [{done:,}/{total:,} step, {100 * done / total:.1f}%]")
else:
    print(" (아직 step 로그 없음)")

tl = os.path.join(D, "train.log")
pat = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\](?: hint (\d+))?(?: dl [\d.]+)?(?: dyn [\d.]+)?(?: gate [\d.]+ mov [\d.]+ sta [\d.]+ k1 [\d.]+ kR [\d.]+ >\.95 [\d.]+)?(?: hole [\d.]+ rec [\d.]+)?(?: st [^|]*)?(?: gn [\d.naif]+)? \| mask var C(\d+)K(\d+) (\w+) B(\d+)")
pat_ms = re.compile(r"\] loss ([\d.]+) \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\] \| mask ms W(\d+) (\w+) B(\d+) s([\d-]+)")   # multistart (10-14)
seen = []
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen.append(m.groups())
ms = [m.groups() for m in pat_ms.finditer(open(tl, errors="ignore").read())] if os.path.isfile(tl) else []
if ms and not seen:
    l, tf, ar, R, W, ds, Bb, st = ms[-1]
    print(f" {B}loss{N}  {float(l):.4f}  = tf {float(tf):.4f} + ar {float(ar):.4f}   (multistart · 지금 W{W} {ds} B{Bb}, 시작점 {st}, 최대 R={R})")
    byw = {}
    for g in ms[-400:]:
        byw.setdefault((g[5], int(g[4])), []).append(float(g[2]))
    print(" (데이터, 창) 별 최근 ar 손실\n   " + "  ".join(f"{d}W{w}:{st.mean(v[-5:]):.3f}" for (d, w), v in sorted(byw.items())))
if seen:
    l, tf, ar, R, hn, C, K, ds, Bb = seen[-1]
    h = f"   힌트 {B}{hn}{N}/256 개" if hn is not None else ""
    tail = open(tl, errors="ignore").read()[-4000:]
    mdl = re.findall(r" dl ([\d.]+)(?: dyn [\d.]+)? \| mask var", tail)
    mdy = re.findall(r" dyn ([\d.]+) \| mask var", tail)
    dl_s = (f" + dl {float(mdl[-1]):.4f} (차이 L1)" if mdl else "") + (f" + dyn {float(mdy[-1]):.4f} (움직임 크기, ×λ 전)" if mdy else "")
    mrc = re.findall(r" hole ([\d.]+) rec ([\d.]+) \| mask var", tail)       # 2026-10-17: 문맥 구멍 — 복원 손실은 걸 때만 (가중치 > 0) 손실 식에 넣는다
    if mrc:
        rw = 0.0
        try:
            rw = float((c["loss"].get("ctx_hole") or {}).get("recon_weight", 1.0))
        except Exception:
            pass
        if rw > 0:
            dl_s += f" + 복원 {rw:g}×{float(mrc[-1][1]):.4f}"
        dl_s += f"   [문맥 구멍 비율 {float(mrc[-1][0]):.2f}]"
    print(f" {B}loss{N}  {float(l):.4f}  = tf {float(tf):.4f} + ar {float(ar):.4f}{dl_s}   (지금 C{C}K{K} {ds} B{Bb}, R={R}){h}")
    if hint_cfg and total:
        a, b = float(hint_cfg.get("start", 0.05)), float(hint_cfg.get("end", 0.9))
        n_max = round(float(hint_cfg.get('ratio_max', 0.1)) * 256)
        if str(hint_cfg.get("shape", "sin2")) == "const":
            print(f"       힌트 일정: step {int(a * total):,}~{int(b * total) - 1:,} 동안 {n_max} 개 고정, 그 뒤 0 (cross-attn 차단)")
        else:
            print(f"       힌트 일정: t {a:.2f}~{b:.2f} sin² 종 모양, 정점 {n_max} 개 (t {(a + b) / 2:.3f})")
    # gating_residual (2026-10-17): rollout gate. 최근 50 줄 평균 (로그는 rank 마다 같은 줄이 8 번 — 중복 제거)
    gl = re.findall(r"\[(\d+), +(\d+)\] loss .*? gate ([\d.]+) mov ([\d.]+) sta ([\d.]+) k1 ([\d.]+) kR ([\d.]+) >\.95 ([\d.]+)", open(tl, errors="ignore").read())
    if gl:
        uniq = list({(a, b): g for a, b, *g in gl}.values())
        def _m(rows, i):
            return st.mean(float(r[i]) for r in rows)
        rec = uniq[-50:]
        g_, mv, sa, k1, kr, hi = (_m(rec, i) for i in range(6))
        ok1 = "✓" if mv < sa else "✗"
        ok2 = "✓" if kr < k1 else "✗"
        warn = f"  {Y}⚠ 1 근처 — 복사로 무너지는 중{N}" if hi > 0.5 and len(uniq) > 100 else ""
        print(f" {B}gate{N}  평균 {g_:.3f}  움직이는 칸 {mv:.3f} vs 정지 칸 {sa:.3f} {ok1}   첫 걸음 {k1:.3f} vs 마지막 {kr:.3f} {ok2}   >0.95 비율 {hi:.2f}{warn}")
        if len(uniq) >= 20:      # epoch 별 평균 gate 추이 (복사 → predictor 로 넘어가는지)
            byep = {}
            for (e, _), gg in zip({(a, b): 0 for a, b, *_ in gl}.keys(), uniq):
                byep.setdefault(int(e), []).append(float(gg[0]))
            print("       epoch 별 평균 gate  " + "  ".join(f"e{e}:{st.mean(v):.2f}" for e, v in sorted(byep.items())[-10:]))
    # 2026-10-19: 상태 감독 head (합성 step 에만). 줄마다 항목을 따로 읽는다 (옛 run 은 없는 항목이 빠진다)
    tl_txt = open(tl, errors="ignore").read()
    H = {}
    for ln in tl_txt.splitlines():
        m0 = re.search(r"\[(\d+), +(\d+)\] loss [\d.]+ \(avg [\d.]+\) \[tf ([\d.]+) ar ([\d.]+)", ln)
        if not m0:
            continue
        key = (int(m0[1]), int(m0[2]))
        mg = re.search(r" gn ([\d.]+)", ln)
        synth = " st " in ln
        r_ = {"ep": key[0], "lat": float(m0[3]) + float(m0[4]), "gn": float(mg[1]) if mg else None, "synth": synth}
        if synth:
            ms = re.search(r" st ([\d.]+) pos ([\d.na]+)/([\d.na]+)px ex ([\d.]+)", ln)
            if ms:
                r_.update(st=float(ms[1]), pv=float(ms[2]) if ms[2][0].isdigit() else float("nan"),
                          ph=float(ms[3]) if ms[3][0].isdigit() else float("nan"))
            mi = re.search(r" id ([\d.]+)", ln)
            mc = re.search(r" cmp ([\d./na]+) ([\d./na]+) ([\d./na]+) ([\d./na]+)", ln)
            mz = re.search(r" conc ([\d.]+)", ln)
            if mi:
                r_["id"] = float(mi[1])
            if mc:
                r_["cmp"] = [tuple(float(x) if x[0].isdigit() else float("nan") for x in mc[j].split("/")) for j in range(1, 5)]
            if mz:
                r_["conc"] = float(mz[1])
        H[key] = r_
    rows = list(H.values())
    syn = [r_ for r_ in rows if r_["synth"] and "st" in r_]
    if syn:
        G, R_, DIM = "\033[32m", "\033[31m", "\033[2m"
        def mean(xs):
            xs = [x for x in xs if x is not None and x == x]
            return st.mean(xs) if xs else float("nan")
        SHc = (c.get("state_head") or {}) if isinstance(c, dict) else {}
        clipv = float(((c.get("optimization") or {}).get("clip_grad")) or 0) if isinstance(c, dict) else 0
        rec = syn[-8:]
        import unicodedata
        def lab(t, w=11):     # 한글은 두 칸 — 화면 폭으로 맞춘다
            n = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in t)
            return "   " + t + " " * max(1, w - n)
        print("─" * 78)
        print(f" {B}head{N}   {SHc.get('type', 'attn')}  ·  손실 ×{float(SHc.get('weight', 1.0)):g}  ·  head lr ×{SHc.get('lr_mult', 1)}"
              f"  ·  clip {'predictor 만' if SHc.get('clip_separate') else '함께'}   {DIM}(최근 {len(rec)} 합성 step 평균){N}")
        pv, ph = mean([r_.get("pv") for r_ in rec]), mean([r_.get("ph") for r_ in rec])
        col = lambda v: (G if v < 28 else Y if v < 60 else R_)        # 복사 기준선 28 px
        print(lab("손실") + f"head {B}{mean([r_['st'] for r_ in rec]):.3f}{N}   vs   잠재 L1 {mean([r_['lat'] for r_ in rec]):.3f}")
        print(lab("위치 오차") + f"보임 {col(pv)}{B}{pv:6.1f}{N} px   가림 {col(ph)}{B}{ph:6.1f}{N} px   {DIM}(1 칸 = 16 px · 복사 기준선 28 px){N}")
        if any("cmp" in r_ for r_ in rec):
            cm = [r_["cmp"] for r_ in rec if "cmp" in r_]
            f2 = lambda j, k, nd: f"{mean([x[j][k] for x in cm]):.{nd}f}"
            print(lab("성분 tf/ar") + f"물체CE {f2(0, 0, 3)}/{f2(0, 1, 3)}   위치 {f2(1, 0, 4)}/{f2(1, 1, 4)}"
                  f"   방향 {f2(2, 0, 3)}/{f2(2, 1, 3)}   속력 {f2(3, 0, 3)}/{f2(3, 1, 3)}")
        idv = mean([r_.get("id") for r_ in rec]) if any("id" in r_ for r_ in rec) else None
        cz = mean([r_.get("conc") for r_ in rec]) if any("conc" in r_ for r_ in rec) else None
        s_id = f"정체 {B}{100 * idv:4.0f} %{N} {DIM}(49 지선다, 우연 2 %){N}" if idv is not None else ""
        s_cz = (f"집중도 {B}{cz:5.1f}x{N} " + (f"{Y}⚠ 균등에 붙음 = 기본값 읽기{N}" if cz < 2 else f"{DIM}(균등 = 1){N}")) if cz is not None \
            else f"집중도 {DIM}— (이 run 로그에 없음){N}"
        print(lab("읽기") + s_id + "     " + s_cz)
        sg = [r_["gn"] for r_ in syn if r_["gn"] is not None]
        ng = [r_["gn"] for r_ in rows if not r_["synth"] and r_["gn"] is not None]
        if sg:
            nclip = sum(g > clipv for g in sg) if clipv else 0
            rs, rn = mean(sg[-8:]), mean(ng[-40:])
            ratio = rs / rn if rn == rn and rn > 0 else float("nan")
            tag = ((f"{R_}⚠ 합성이 주도{N}" if ratio > 3 else f"{G}✓ 제약 범위{N}") if ratio == ratio else "")
            print(lab("grad norm") + f"합성 {B}{rs:.2f}{N} / 자연 {B}{rn:.3f}{N}  →  {B}{ratio:.1f} 배{N} {tag} {DIM}(제약 = 3 배 이하){N}")
            print(lab("") + f"{DIM}합성 최근 50 최대 {max(sg[-50:]):.2f} · 전체 최대 {max(sg):.2f}"
                  + (f" · clip {clipv:g} 초과 {nclip}/{len(sg)}" if clipv else "") + f"{N}")
        byep = {}
        for r_ in syn:
            byep.setdefault(r_["ep"], []).append(r_)
        has_id = any("id" in r_ for r_ in syn)
        cells = [f"e{e} {mean([x['st'] for x in v]):5.2f} · {mean([x.get('pv') for x in v]):3.0f}/{mean([x.get('ph') for x in v]):3.0f}px"
                 + (f" · {100 * mean([x.get('id') for x in v]):2.0f}%" if has_id else "") for e, v in sorted(byep.items())[-6:]]
        print(lab("epoch 추이") + f"{DIM}(손실 · 위치 보임/가림" + (" · 정체" if has_id else "") + f"){N}")
        for i in range(0, len(cells), 3):
            print(lab("") + "   ".join(cells[i:i + 3]))
    ck = {}
    for g in seen[-400:]:
        ck.setdefault((int(g[5]), int(g[6])), []).append(float(g[2]))
    s = "  ".join(f"C{c}K{k}:{st.mean(v[-5:]):.3f}" for (c, k), v in sorted(ck.items()))
    print(f" (C,K) 별 최근 ar 손실\n   {s}")

mp = os.path.join(D, "metrics.jsonl")
vals = []
if os.path.isfile(mp):
    for ln in open(mp):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        v = r.get("val") or {}
        if "acc" in v:
            vals.append((r["epoch"], 100 * v["acc"], v.get("margin"), v.get("n_pair"), v.get("by_type") or {}))
if vals:   # 2026-10-18: margin 도 · val 데이터셋 이름은 config 에서 (기본 VOBench late 432 쌍 — 정식 채점과 같은 정의). 10-19 모양 정리
    vname = ((c.get("val") or {}).get("dataset") if isinstance(c, dict) else None) or "val"
    G_, DIM_ = "\033[32m", "\033[2m"
    print("─" * 78)
    _best = max(vals[1:] or vals, key=lambda t: t[1])
    print(f" {B}val{N}    {vname} · {vals[-1][3]} 쌍 · rollout   {DIM_}(epoch 0 = 학습 전, 우연 50){N}")
    import unicodedata
    def vlab(t, w=11):
        n = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in t)
        return "   " + t + " " * max(1, w - n)
    print(f"{vlab('지금')}{B}e{vals[-1][0]} {vals[-1][1]:.1f} %{N}   최고 e{_best[0]} {_best[1]:.1f} %   출발 e{vals[0][0]} {vals[0][1]:.1f} %"
          + (f"   margin {vals[-1][2]:+.4f}" if vals[-1][2] is not None else ""))
    print(vlab("추이") + "  ".join(f"e{e} {a_:.1f}" for e, a_, m_, _, _ in vals[-8:]))
    bt = vals[-1][4]
    if any("/late" in k for k in bt):     # VOBench late: 위반별 primitive / realistic
        g = lambda o, vi: bt.get(f"{o}/{vi}/late", {}).get("acc")
        for o, olab in (("primitive", "primitive"), ("realistic", "realistic")):
            if all(g(o, vi) is not None for vi in ("permanence", "shape", "color")):
                print(vlab(olab) + "   ".join(f"{lab_} {B}{100 * g(o, vi):5.1f}{N}" for vi, lab_ in (("permanence", "영속"), ("shape", "모양"), ("color", "색"))))
    print("─" * 78)

if os.path.isfile(tl):      # 데이터별 step 수 (op320 · vob 같은 고정/경계 소스가 계획대로 섞이는지)
    from collections import Counter
    cnt = Counter(g[7] for g in seen)
    if cnt:
        tot = sum(cnt.values())
        print(" 데이터별 step  " + "  ".join(f"{k} {v:,} ({100 * v / tot:.1f}%)" for k, v in cnt.most_common()))
# 2026-10-18: 측정판 (z_training/tools/panel.py → <run>/panel.jsonl). 점수가 가리는 것 — 운반 · 위치 · 물체 자리
pj = os.path.join(D, "panel.jsonl")
if pc.get("ar_input") == "video" if isinstance(pc, dict) else False:
    print(f" {B}측정판{N} 지원 안 함 — 정답이 영상 (창) 공간이라 블록 공간 측정판 (운반 · 위치 · 물체 자리) 을 댈 수 없다")
elif not os.path.isfile(pj):
    print(f" {B}측정판{N} 아직 없음 — tmux new-session -d -s panel_{RUN} \"bash z_training/tools/panel_watch.sh {RUN} 10 7\"")
if os.path.isfile(pj):
    recs = []
    for ln in open(pj):
        try:
            recs.append(json.loads(ln))
        except Exception:
            pass
    if recs:
        t = recs[-1]; tt = t["transport_true"]; tr = t["transport"]
        print(f" {B}측정판{N} ({t['ckpt']}, epoch {t['epoch']}) — 운반 걸음0 원래칸 {tr['s0']['src']:+.2f} (진짜 {tt['s0']['src']:+.2f}) "
              f"같은칸 {tr['s0']['same']:+.2f} ({tt['s0']['same']:+.2f}) 집중 {tr['s0']['conc']:.2f} ({tt['s0']['conc']:.2f}) · 걸음3 원래칸 {tr['s3']['src']:+.2f}")
        pe, pc_ = t["poserr"], t["poserr_copy"]
        print(f"       위치 오차 걸음 1·3·7 " + " ".join(f"{v:.2f}" for v in pe.values()) + "  (복사 " + " ".join(f"{v:.2f}" for v in pc_.values()) + ")"
              f"   late 물체 자리 {t['late']['obj']:.1f} (복사 {t['late_copy']['obj']:.1f}) · 화면 전체 {t['late']['full']:.1f} (복사 {t['late_copy']['full']:.1f})")
        if len(recs) > 1:
            print("       추이 (epoch: 원래칸/집중/위치7/late물체) " + "  ".join(
                f"e{r_['epoch']}:{r_['transport']['s0']['src']:+.2f}/{r_['transport']['s0']['conc']:.2f}/{list(r_['poserr'].values())[-1]:.1f}/{r_['late']['obj']:.0f}" for r_ in recs[-6:]))
# 2026-10-17: 살아 있는 chain 세션만 (죽은 옛 chain 은 안 보인다). 로그는 이 run 이름이 붙은 것 또는 세션 이름과 같은 것
sess = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True).stdout.split()
for ch in [x for x in sess if x.startswith("chain")]:
    cands = [os.path.join(ROOT, f"z_training/logs/chain_eval_{RUN}.log"), os.path.join(ROOT, f"z_training/logs/{ch}.log")]
    cl = next((x for x in cands if os.path.isfile(x) and os.path.getsize(x)), None)
    print(f" {B}{ch}{N} 살아 있음" + ("" if cl else " (로그 없음)"))
    if cl:
        for ln in [x for x in open(cl).read().strip().splitlines() if x.startswith("[")][-3:]:
            print("   " + ln[:110])
import glob as _g
for bf in sorted(_g.glob(os.path.join(D, "eval", "BENCH_*.txt"))):     # 2026-10-17: 이 run 의 채점만 (다른 run 결과는 안 보인다)
    print(f" {B}채점{N} {os.path.basename(bf)[6:-4]}:")
    for ln in open(bf).read().strip().splitlines()[1:]:
        print("   " + ln)
print("━" * 78)
EOF
