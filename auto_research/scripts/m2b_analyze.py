#!/usr/bin/env python3
"""M2b 분석 (적대 검증 반영). 입력 v11_m2b/.
물체 신호 = pos 에 물체가 있는 쌍 (vanish A · shape · color), 템플릿 = E pos 의 h 진실 칸 (같은 궤적).
  obj   : contrast (maxcos − median) — 연속값 · 문턱 3 종 (물체 쌍 E 팔 t0–2 P10 · 공통 0.357 · 빈 장면 E 템플릿 P95)
  hitnul: 적중 (1 칸) − 귀무 (같은 운동 **다른 궤적** 쌍의 진실 칸)
  천장  : hL · hE · hV (진짜 미래 h 에 같은 템플릿)
  슬롯  : t0–1 (L 가림 중), t2–4, t5–6 (t7 은 화면 밖이 많아 제외; 화면 밖 칸은 추출에서 NaN)
대비 (운동별): L+Eb − L (회복), E+Lb − E (파괴), L+Ebt − Lbt (역사 정보 없는 경계 관측의 회복), Eb_only − E, (L+Eb) − E (같은 경계 · 역사만 다름),
  L+Vb − L, V+Lb − V.  CI = late block 군집 bootstrap.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_m2b")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/m2b"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); P = meta["pairs"]; A = meta["arms"] + meta["ceil"]; n = len(P)
LOC = np.load(I / "loc.npy"); S = np.load(I / "scores.npy")
mot = np.array([p["motion"] for p in P]); viol = np.array([p["violation"] for p in P]); pid = np.array([p["pair_id"] for p in P])
blk = np.array([p["block"] for p in P]); direc = np.array([p["px"][:30] for p in P])
has_obj = (viol != "vanish") | (pid == "A")
rng = np.random.default_rng(0); ub = np.unique(blk); bix = {b: np.where(blk == b)[0] for b in ub}
ai = {a: i for i, a in enumerate(A)}
con = LOC[..., 2] - LOC[..., 3]                                                    # (n, arms, 8)
thr_obj = np.nanpercentile(con[has_obj, ai["E"], :3], 10)
thr_empty = np.nanpercentile(con[~has_obj, ai["E"], 2:5], 95) if (~has_obj).sum() > 20 else np.nan
THR = {"objP10": float(thr_obj), "common0.357": 0.357, "emptyP95": float(thr_empty)}
# 진실 칸 (추출과 같은 규칙) — 적중용
def cells(px, py):
    x = np.array([float(v) for v in px.split()]); y = np.array([float(v) for v in py.split()]); c = []
    for t in range(8):
        s0 = 16 + 2 * t; cx, cy = np.nanmean(x[s0:s0 + 2]), np.nanmean(y[s0:s0 + 2])
        c.append((np.nan, np.nan) if (np.isnan(cx) or cx < 0 or cx >= 288 or cy < 0 or cy >= 288) else (cx // 18, cy // 18))
    return c
CT = np.array([cells(p["px"], p["py"]) for p in P], float)                       # (n, 8, 2)
other = np.arange(n)
for mo in np.unique(mot):
    ix = np.where(mot == mo)[0]
    for i in ix:
        c = ix[direc[ix] != direc[i]]; other[i] = rng.choice(c) if len(c) else i
am = LOC[..., [1, 0]]
hit = (np.abs(am - CT[:, None]).max(-1) <= 1).astype(float); nul = (np.abs(am - CT[other][:, None]).max(-1) <= 1).astype(float)
valid = ~np.isnan(CT[..., 0]); hit[~np.broadcast_to(valid[:, None], hit.shape)] = np.nan; nul[~np.broadcast_to(valid[:, None], nul.shape)] = np.nan
con = np.where(np.broadcast_to(valid[:, None], con.shape), con, np.nan)


def boot(v, m, B=500):
    v = v[m]; bl = blk[m]; u = np.unique(bl); ix = {b: np.where(bl == b)[0] for b in u}
    mm = np.nanmean(v, 0); bs = [np.nanmean(v[np.concatenate([ix[b] for b in rng.choice(u, len(u))])], 0) for _ in range(B)]
    return np.round(mm, 3), np.round(np.nanpercentile(bs, 2.5, 0), 3), np.round(np.nanpercentile(bs, 97.5, 0), 3)


WIN = {"t0-1": slice(0, 2), "t2-4": slice(2, 5), "t5-6": slice(5, 7)}
CONTR = [("recover L+Eb−L", "L+Eb", "L"), ("destroy E+Lb−E", "E+Lb", "E"), ("trunc-bnd L+Ebt−Lbt", "L+Ebt", "Lbt"), ("trunc cost Lbt−L", "Lbt", "L"),
         ("hist-only (L+Eb)−E", "L+Eb", "E"), ("hist delete Eb_only−E", "Eb_only", "E"), ("vis bnd L+Vb−L", "L+Vb", "L"), ("vis destroy V+Lb−V", "V+Lb", "V")]
out = {"thresholds": THR, "n": int(n), "n_obj": int(has_obj.sum())}
for mo in ("ALL", "static", "moving_flat", "moving"):
    m = has_obj & ((mot == mo) if mo != "ALL" else True)
    rec = {"n": int(m.sum())}
    for wn, sl in WIN.items():
        c_ = np.nanmean(con[:, :, sl], -1); hn = np.nanmean(hit[:, :, sl] - nul[:, :, sl], -1)
        r = {"contrast": {a: boot(c_[:, ai[a]], m)[0].item() for a in A}, "hit_null": {a: boot(hn[:, ai[a]], m)[0].item() for a in A}}
        for tn, th in THR.items():
            if np.isnan(th): continue
            o = np.nanmean((con[:, :, sl] >= th).astype(float) + np.where(np.isnan(con[:, :, sl]), np.nan, 0), -1)
            r[f"obj@{tn}"] = {a: boot(o[:, ai[a]], m)[0].item() for a in A}
        for cn, a1, a0 in CONTR:
            for metric, arr in (("contrast", c_), ("hit_null", hn)):
                mm, lo, hi = boot(arr[:, ai[a1]] - arr[:, ai[a0]], m); r[f"{cn}|{metric}"] = [mm.item(), lo.item(), hi.item()]
        rec[wn] = r
    out[mo] = rec
json.dump(out, open(OUT / f"m2b_v11{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
print("문턱", THR, "n", n, "물체 쌍", int(has_obj.sum()))
for mo in ("ALL", "static", "moving_flat", "moving"):
    rec = out[mo]; print(f"\n== {mo} (n {rec['n']})")
    for wn in WIN:
        r = rec[wn]
        print(f"  {wn} contrast " + " ".join(f"{a}:{r['contrast'][a]:.3f}" for a in A))
        print(f"        hit−null " + " ".join(f"{a}:{r['hit_null'][a]:+.2f}" for a in A))
        print("        " + " | ".join(f"{cn} c{r[cn+'|contrast'][0]:+.3f}[{r[cn+'|contrast'][1]:+.3f},{r[cn+'|contrast'][2]:+.3f}] h{r[cn+'|hit_null'][0]:+.2f}" for cn, _, _ in CONTR))
