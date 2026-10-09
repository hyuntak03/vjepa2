#!/usr/bin/env python3
"""H8b — H8 knockout 재분석: 공통 고정 문턱 · paired bootstrap · 귀무 적중 · 적중 자리 분해 (CPU, 파라미터 없음).

왜: H8 원 표 (§3) 는 물체다움 문턱을 **명세마다 자기 슬롯 0–2 의 10 백분위** 로 잡았다. 기준 슬롯 (1–2) 을 손상시키는 명세
   (mm_L0-6) 는 문턱이 0.357 → 0.306 으로 내려가고, 그것만으로 먼 슬롯 물체다움이 크게 오른 것처럼 보였다 (적대 검증 2026-09-24).
   여기서는 문턱을 clean_null 에서 **하나로 고정**하고 명세 사이 차이를 clip 짝 bootstrap 으로 다시 잰다.
   진실 적중은 위치 사전확률 (귀무) 과 같이, 그리고 "마지막으로 본 자리" 적중과 나눠 읽는다.

입력: v3_h8/h8.npy (n, spec, 16 슬롯, 9) = loc_tru (y, x, maxcos, medcos) · loc_app (y, x, maxcos, medcos) · L1 |p−h|
      (h8_knockout_persistence_v3.py 산출물, vll6 로컬 디스크).

기호 (clip i, 미래 슬롯 k = 튜블릿 32+2k, 명세 s; 템플릿 = 슬롯 k 진실 칸의 LN(h) 토큰, cos 는 p 의 256 토큰 각각과):
  con     = maxcos − medcos
  ref_A   = P10( con[clean_null, 슬롯 0–2] )          H8 공통 문턱 (0.357)
  ref_B   = P10( con[clean_null, 슬롯 0] )            슬롯 0 만 — 기준 슬롯 1–2 손상과 무관
  ref_own = P10( con[s, 슬롯 0–2] )                    H8 원 표의 명세별 문턱 (재현·분해용)
  obj     = 1[ con ≥ ref ]
  hit     = 1[ |argmax − T_k|_∞ ≤ 1 ]                  T_k = 슬롯 k 의 진실 물체 칸 (18 px 칸, 16×16)
  null    = 같은 시나리오의 다른 clip j (슬롯 k 에서 물체가 보이는 clip) 전부에 대한 1[ |argmax_i − T_k^j|_∞ ≤ 1 ] 평균
            (위치 사전확률의 정확한 기대값 — h23_null_hit.py 의 무작위 순열 대신)
  last    = 1[ |argmax − L|_∞ ≤ 1 ]                    L = 문맥 마지막 튜블릿 (프레임 30–31) 의 물체 칸
  else    = 1 − 1[ hit ∨ last ]
  sep     = |T_k − L|_∞ ≥ 3 인 clip (두 3×3 창이 겹치지 않아 진실 적중과 마지막 자리 적중을 가를 수 있다)
  길위    = argmax 가 선분 L→T_k 에서 유클리드 1.5 칸 안 (물체다움 clip 의 봉우리 자리 분해용). hit·last 와 겹칠 수 있어
            분해표는 hit / last / 길위(hit·last 아님) / 그밖 으로 적고, "길위전체" 는 귀무 둘과 비교한다:
            시나리오 귀무 = 같은 시나리오 다른 clip 의 봉우리를 이 clip 의 선분에 대 본 비율, 면적 귀무 = 256 칸 중 선분 1.5 칸 안 비율
  진행률  = (argmax − L)·(T_k − L) / |T_k − L|²  (물체다움 · sep clip 의 중앙값)
  L1      = mean_토큰 |p − LN(h)| (슬롯 k)
  paired bootstrap: clip 을 복원추출 (B = 2000, seed 0) 해 Δ = 명세 − clean_null 의 95 % 백분위 CI.
  분모 = 슬롯 k 에서 물체가 화면 안 · 보임 · ignore 아님 (두 프레임 모두) 인 clip (L1 은 전 clip).

출력: auto_research/exp_results/h8/h8b_fixed_threshold.{json,txt}
  auto_research/scripts/srun6.sh 4 16G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h8b_fixed_threshold.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, G  # noqa: E402

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h8")
OUT = Path(__file__).resolve().parents[2] / "auto_research/exp_results/h8"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; SP = meta["specs"]; K = meta["K"]
H = np.load(I / "h8.npy").astype(np.float64)                 # (n, S, K, 9)
n, S = H.shape[:2]
R, X, Y, OK = load_index(ids)
scen = np.array([r["scenario"] for r in R])
base = SP.index("clean_null")
L = cell(tub_xy(X, Y, 30))                                     # (n, 2) x, y
con = H[..., 2] - H[..., 3]
refA = float(np.nanpercentile(con[:, base, :3], 10)); refB = float(np.nanpercentile(con[:, base, 0], 10))
ref_own = {sp: float(np.nanpercentile(con[:, si, :3], 10)) for si, sp in enumerate(SP)}
ref_own0 = {sp: float(np.nanpercentile(con[:, si, 0], 10)) for si, sp in enumerate(SP)}
cheb = lambda a, b: np.abs(a - b).max(-1)

B = 2000
Wb = np.random.default_rng(0).multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)   # (B, n)


def boot(vals, mask):
    """vals (n, Q), mask (n,) or (n, Q) → (B, Q) bootstrap 평균."""
    m = np.broadcast_to(mask if mask.ndim == 2 else mask[:, None], vals.shape).astype(np.float64)
    v = np.where(m > 0, vals, 0.0)
    return (Wb @ v) / np.maximum(Wb @ m, 1e-9)


def ci(bs):
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return lo, hi


QN = ["objA", "objB", "objown", "hit", "null", "hmn", "last", "else", "hit_sep", "last_sep", "l1"]
res = {"n_clips": n, "specs": SP, "refA": refA, "refB": refB, "ref_own": ref_own, "ref_own_slot0": ref_own0,
       "baseline": "clean_null", "B": B, "slots": {}}
per_slot_q = {}
for k in range(K):
    f0 = 32 + 2 * k
    T = cell(tub_xy(X, Y, f0)); ok = OK[:, f0] & OK[:, f0 + 1]
    sep = cheb(T, L) >= 3
    A = H[:, :, k, :2][..., ::-1].round().astype(int)          # (n, S, 2) x, y
    hit = (cheb(A, T[:, None]) <= 1).astype(float)
    last = (cheb(A, L[:, None]) <= 1).astype(float)
    els = 1.0 - np.maximum(hit, last)
    null = np.full((n, S), np.nan)
    for s in np.unique(scen):
        ix = np.where((scen == s) & ok)[0]
        if len(ix) < 2:
            continue
        d = np.abs(A[ix][:, None, :, :] - T[ix][None, :, None, :]).max(-1) <= 1   # (m_i, m_j, S)
        d = d.astype(float); m = len(ix)
        d[np.arange(m), np.arange(m)] = 0.0
        null[ix] = d.sum(1) / (m - 1)
    c = con[:, :, k]
    objA = (c >= refA).astype(float); objB = (c >= refB).astype(float)
    objown = np.stack([(c[:, si] >= ref_own[sp]).astype(float) for si, sp in enumerate(SP)], 1)
    l1 = H[:, :, k, 8]
    q = dict(objA=objA, objB=objB, objown=objown, hit=hit, null=np.nan_to_num(null), hmn=hit - np.nan_to_num(null),
             last=last, els=els, hit_sep=hit, last_sep=last, l1=l1)
    masks = dict(objA=ok, objB=ok, objown=ok, hit=ok, null=ok, hmn=ok, last=ok, els=ok, hit_sep=ok & sep, last_sep=ok & sep,
                 l1=np.ones(n, bool))
    slot = {"n_ok": int(ok.sum()), "n_sep": int((ok & sep).sum()), "spec": {}}
    for sp in SP:
        slot["spec"][sp] = {}
    for qn, v in q.items():
        mk = masks[qn]
        mean = (v[mk]).mean(0)
        dv = v - v[:, [base]]
        bs = boot(dv, mk); lo, hi = ci(bs)
        for si, sp in enumerate(SP):
            slot["spec"][sp][qn] = round(float(mean[si]), 4)
            if si != base:
                slot["spec"][sp][f"d_{qn}"] = [round(float(mean[si] - mean[base]), 4), round(float(lo[si]), 4), round(float(hi[si]), 4)]
    # 물체다움 (ref_A) 인 clip 의 봉우리 자리: hit / last / 길 위 (L→T 선분 1.5 칸 안, hit·last 아님) / 그 밖
    Lf, Tf = L.astype(float), T.astype(float); dLT = Tf - Lf; nn = np.maximum((dLT ** 2).sum(1), 1e-9)
    for si, sp in enumerate(SP):
        a = A[:, si].astype(float)
        u = np.clip(((a - Lf) * dLT).sum(1) / nn, 0, 1)
        segd = np.linalg.norm(a - (Lf + u[:, None] * dLT), axis=1)
        onseg = (segd <= 1.5) & (hit[:, si] == 0) & (last[:, si] == 0)
        prog = ((a - Lf) * dLT).sum(1) / nn
        mo = ok & (objA[:, si] > 0)
        slot["spec"][sp]["objA_n"] = int(mo.sum())
        if mo.sum() >= 20:
            # 귀무: (1) 같은 시나리오 다른 clip j 의 봉우리를 clip i 의 선분에 대 본 길위 비율 (정확한 기대값)
            #       (2) 16×16 칸 중 clip i 의 선분 1.5 칸 안에 드는 칸 비율 (균등 봉우리)
            segany = segd <= 1.5; sn = []
            for s_ in np.unique(scen):
                ii = np.where(mo & (scen == s_))[0]; jj = np.where(ok & (scen == s_))[0]
                if len(ii) == 0 or len(jj) < 2:
                    continue
                aj = A[jj, si].astype(float)[None]                                   # (1, mj, 2)
                Li, di, ni = Lf[ii][:, None], dLT[ii][:, None], nn[ii][:, None]
                uu = np.clip(((aj - Li) * di).sum(-1) / ni, 0, 1)
                on = np.linalg.norm(aj - (Li + uu[..., None] * di), axis=-1) <= 1.5   # (mi, mj)
                on = on & (ii[:, None] != jj[None])
                sn.append(on.sum(1) / (len(jj) - 1))
            gy, gx = np.mgrid[0:G, 0:G]; cells = np.stack([gx.ravel(), gy.ravel()], -1).astype(float)[None]   # (1,256,2)
            Li, di, ni = Lf[mo][:, None], dLT[mo][:, None], nn[mo][:, None]
            uu = np.clip(((cells - Li) * di).sum(-1) / ni, 0, 1)
            area = (np.linalg.norm(cells - (Li + uu[..., None] * di), axis=-1) <= 1.5).mean(1)
            slot["spec"][sp]["objA_at"] = dict(hit=round(float(hit[mo, si].mean()), 3), last=round(float(last[mo, si].mean()), 3),
                                               seg=round(float(onseg[mo].mean()), 3),
                                               els=round(float((els[mo, si] * (~onseg[mo])).mean()), 3),
                                               segany=round(float(segany[mo].mean()), 3),
                                               segany_null_scen=round(float(np.concatenate(sn).mean()), 3),
                                               segany_null_area=round(float(area.mean()), 3))
            ms = mo & sep
            slot["spec"][sp]["objA_prog_med_sep"] = round(float(np.median(prog[ms])), 3) if ms.sum() >= 20 else None
        slot["spec"][sp]["con_med"] = round(float(np.median(c[ok, si])), 4)
        slot["spec"][sp]["maxcos_med"] = round(float(np.median(H[ok, si, k, 2])), 4)
        slot["spec"][sp]["medcos_med"] = round(float(np.median(H[ok, si, k, 3])), 4)
    # 분해: clean@자기문턱 → clean@mm문턱 → mm@mm문턱
    dec = {}
    for sp in ("mm_L0-6", "mm_L7-11", "mm_all", "mmx_all", "mc_L7-11"):
        si = SP.index(sp)
        dec[sp] = [round(float((c[ok, base] >= ref_own["clean_null"]).mean()), 3),
                   round(float((c[ok, base] >= ref_own[sp]).mean()), 3),
                   round(float((c[ok, si] >= ref_own[sp]).mean()), 3)]
    slot["decomp_clean_own__clean_at_spec_ref__spec_own"] = dec
    res["slots"][k] = slot

# 전 슬롯 L1 (clip 평균 → 명세 차 bootstrap)
l1c = H[:, :, :, 8].mean(2)
bs = boot(l1c - l1c[:, [base]], np.ones(n, bool)); lo, hi = ci(bs)
res["l1_allslots"] = {sp: dict(mean=round(float(l1c[:, si].mean()), 4),
                               d=[round(float(l1c[:, si].mean() - l1c[:, base].mean()), 4), round(float(lo[si]), 4), round(float(hi[si]), 4)])
                      for si, sp in enumerate(SP)}
json.dump(res, open(OUT / "h8b_fixed_threshold.json", "w"), indent=1, ensure_ascii=False)

# ---- 사람이 읽는 표 ----
ln = [f"n = {n} clip (자유 운동 6 법칙 가능 clip 전부). 문턱 ref_A = {refA:.3f} (clean_null 슬롯 0–2 P10), ref_B = {refB:.3f} (슬롯 0 P10)",
      "명세별 문턱 (H8 원 표): " + ", ".join(f"{sp} {v:.3f}" for sp, v in ref_own.items()),
      "명세별 슬롯 0 문턱: " + ", ".join(f"{sp} {v:.3f}" for sp, v in ref_own0.items()), ""]
SL = [0, 2, 4, 6, 8, 10, 12, 14]


def row(qn, sp, fmt="{:.2f}"):
    return " ".join(fmt.format(res["slots"][k]["spec"][sp][qn]) for k in SL)


def drow(qn, sp):
    out = []
    for k in SL:
        d, lo, hi = res["slots"][k]["spec"][sp][f"d_{qn}"]
        out.append(f"{d:+.3f}[{lo:+.3f},{hi:+.3f}]")
    return "  ".join(out)


ln.append("슬롯        " + " ".join(f"{k:>4d}" for k in SL))
ln.append("n_ok        " + " ".join(f"{res['slots'][k]['n_ok']:>4d}" for k in SL))
ln.append("n_sep       " + " ".join(f"{res['slots'][k]['n_sep']:>4d}" for k in SL))
for qn, nm in (("objA", "물체다움@ref_A"), ("objB", "물체다움@ref_B"), ("objown", "물체다움@명세별(원표)"), ("hit", "진실적중"),
               ("null", "귀무적중"), ("hmn", "적중−귀무"), ("last", "마지막자리적중"), ("els", "그밖"),
               ("hit_sep", "진실적중|sep"), ("last_sep", "마지막적중|sep"), ("l1", "L1")):
    ln.append(f"\n== {nm}")
    for sp in SP:
        if sp == "clean":
            continue
        ln.append(f"{sp:11s} {row(qn, sp)}")
    if qn in ("objA", "objB", "hmn", "last", "hit_sep", "last_sep", "l1"):
        ln.append(f"  Δ vs clean_null [95% CI] (슬롯 {SL})")
        for sp in SP:
            if sp in ("clean", "clean_null"):
                continue
            ln.append(f"  {sp:11s} {drow(qn, sp)}")
ln.append("\n== 물체다움@ref_A 인 clip 의 봉우리 자리 (hit / last / 길위 / 그밖 · 진행률중앙값|sep), 슬롯 6 · 10 · 14")
for sp in SP:
    s = []
    for k in (6, 10, 14):
        e = res["slots"][k]["spec"][sp]
        if "objA_at" in e:
            o = e["objA_at"]; pm = e.get("objA_prog_med_sep")
            s.append(f"k{k} n{e['objA_n']:4d} {o['hit']:.2f}/{o['last']:.2f}/{o['seg']:.2f}/{o['els']:.2f} p{'–' if pm is None else f'{pm:.2f}'}"
                     f" 길위전체 {o['segany']:.2f} (귀무 시나리오 {o['segany_null_scen']:.2f} · 면적 {o['segany_null_area']:.2f})")
        else:
            s.append(f"k{k} n{e['objA_n']:4d}   –")
    ln.append(f"{sp:11s} " + "   ".join(s))
ln.append("\n== 대비 · cos 중앙값 (슬롯 0 / 8): con, maxcos, medcos")
for sp in SP:
    e0, e8 = res["slots"][0]["spec"][sp], res["slots"][8]["spec"][sp]
    ln.append(f"{sp:11s} con {e0['con_med']:.3f}/{e8['con_med']:.3f}  max {e0['maxcos_med']:.3f}/{e8['maxcos_med']:.3f}  med {e0['medcos_med']:.3f}/{e8['medcos_med']:.3f}")
ln.append("\n== 문턱 분해 (clean@clean문턱 → clean@명세문턱 → 명세@명세문턱), 슬롯 8 / 10 / 14")
for sp in ("mm_L0-6", "mm_L7-11", "mm_all", "mmx_all", "mc_L7-11"):
    ln.append(f"{sp:11s} " + "   ".join("→".join(f"{v:.2f}" for v in res['slots'][k]['decomp_clean_own__clean_at_spec_ref__spec_own'][sp]) for k in (8, 10, 14)))
ln.append("\n== 전 슬롯 L1 |p−h| 평균, Δ vs clean_null [95% CI]")
for sp in SP:
    e = res["l1_allslots"][sp]
    ln.append(f"{sp:11s} {e['mean']:.4f}  {e['d'][0]:+.4f} [{e['d'][1]:+.4f},{e['d'][2]:+.4f}]")
txt = "\n".join(ln)
print(txt); (OUT / "h8b_fixed_threshold.txt").write_text(txt)
