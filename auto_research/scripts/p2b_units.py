#!/usr/bin/env python3
"""P2b — 지평의 단위: 슬롯 (튜블릿 수) · raw 시간 · 이동 거리 중 어느 단위로 재면 팔 · 속도가 달라도 절벽이 한 자리에 모이나.

입력 v3_p2 (p2_stride_sweep_v3.py). 정의는 p2_analyze.py 와 같다 (hit − null, 공통 문턱).
절벽 위치 = (hit − null) 평균 곡선이 그 곡선 슬롯 0 값의 절반을 **처음 아래로 지나는 점** (인접 두 슬롯 선형 보간), 단위별로 환산.
(1) 팔 5 개 × 속도 3 분위 (법칙 안에서 문맥 끝 속도로 3 분위를 나눈 뒤 법칙을 합침) = 15 곡선.
(2) 단위 u ∈ {slot, raw_dt, dist} 마다 15 절벽의 변동계수 CV = sd/mean.  가장 작은 CV 의 단위가 지평을 정한다.
(3) 같은 문맥 튜블릿 수끼리 (8: s1_C16 · s2_C16, 4: s2_C8 · s4_C8) 와 같은 raw 구간끼리 (s1_C32 · s2_C16 · s4_C8) 따로.
CI: 궤적 군집 bootstrap (B=300) 으로 각 절벽의 구간.

위 (1)–(3) 은 기존 키 (cliffs · dispersion) 이고 값은 그대로 둔다 (무작위 짝 귀무, 교차 표본만의 CI).  cliff 에 t_since 필드,
ci_censored (비교차 = 창 끝으로 중도절단) 를 더했다.  적대 검증 (VERIFY_0925, 2026-09-25) 반영분은 파일 끝 "v2" 절의 새 키다.
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p2")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("P2_OUT_DIR", ROOT / "auto_research/exp_results/p2")); OUT.mkdir(parents=True, exist_ok=True)
CELL = 18.0
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
LT = np.load(I / "loc_tru.npy")
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
scen = np.array(meta["scenario"]); traj = np.array(["|".join(map(str, t)) for t in meta["traj"]])
rng = np.random.default_rng(0); ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
other = np.array([rng.choice(np.where((scen == scen[i]) & (traj != traj[i]))[0]) for i in range(n)])
spd = np.hypot(X[:, 31] - X[:, 27], Y[:, 31] - Y[:, 27]) / 4 / CELL                 # 문맥 끝 속도 (칸/raw 프레임)
terc = np.zeros(n, int)
for s in np.unique(scen):
    ix = np.where(scen == s)[0]; q = np.quantile(spd[ix], [1 / 3, 2 / 3]); terc[ix] = np.digitize(spd[ix], q)

curves = {}                                                                          # (arm, terc) → per-clip arrays
for ai, A in enumerate(arms):
    fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
    cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
    HN = np.full((n, nf), np.nan); DT = np.zeros(nf); DI = np.full((n, nf), np.nan); TS = np.zeros(nf)
    for t in range(nf):
        g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
        ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
        ok = INF[:, fa] & INF[:, fb] & INF[:, g0] & INF[:, g1] & ~np.isnan(LT[:, ai, t, 0])
        cti = np.clip(np.floor(ct), 0, 15); am = LT[:, ai, t, [1, 0]]
        hn = (np.abs(am - cti).max(-1) <= 1).astype(float) - (np.abs(am - cti[other]).max(-1) <= 1).astype(float)
        HN[ok, t] = hn[ok]; DI[ok, t] = np.linalg.norm(ct - cl, axis=-1)[ok]; DT[t] = (g0 + g1) / 2 - 31.5
        TS[t] = (g0 + g1) / 2 - (fa + fb) / 2                                        # 마지막 문맥 튜블릿 중심부터 (dist 와 같은 원점)
    curves[A["name"]] = (HN, DT, DI, TS)


def cliff(HN, DT, DI, sel, TS=None, censor=False):
    m = np.nanmean(HN[sel], 0); th = 0.5 * m[0]
    for t in range(1, len(m)):
        if m[t] < th:
            w = (m[t - 1] - th) / max(m[t - 1] - m[t], 1e-9)
            d0, d1 = np.nanmean(DI[sel, t - 1]), np.nanmean(DI[sel, t])
            out = dict(slot=t - 1 + w, raw_dt=DT[t - 1] + w * (DT[t] - DT[t - 1]), dist=d0 + w * (d1 - d0))
            if TS is not None: out["t_since"] = TS[t - 1] + w * (TS[t] - TS[t - 1])
            return out
    if censor:                                                                       # 중도절단: 창 끝 (하한) 으로 둔다
        t = len(m) - 1; out = dict(slot=float(t), raw_dt=DT[t], dist=np.nanmean(DI[sel, t]))
        if TS is not None: out["t_since"] = TS[t]
        return out
    return None                                                                      # 창 안에서 안 무너짐 (하한만)


res = {}
for A in arms:
    HN, DT, DI, TS = curves[A["name"]]
    for q in range(3):
        sel = terc == q
        c = cliff(HN, DT, DI, sel, TS)
        bs = []; bsc = []
        for _ in range(300):
            s = np.concatenate([tix[t] for t in rng.choice(ut, len(ut))]); s = s[terc[s] == q]
            cb = cliff(HN, DT, DI, s, TS)
            if cb: bs.append([cb["slot"], cb["raw_dt"], cb["dist"], cb["t_since"]])
            cc = cliff(HN, DT, DI, s, TS, censor=True); bsc.append([cc["slot"], cc["raw_dt"], cc["dist"], cc["t_since"]])
        bs = np.array(bs); bsc = np.array(bsc)
        res[f"{A['name']}|q{q}"] = dict(arm=A["name"], stride=A["stride"], n_ctx=A["n_ctx"], terc=q, speed=round(float(spd[sel].mean()), 4),
                                        cliff=None if c is None else {k: round(float(v), 2) for k, v in c.items()},
                                        ci=None if len(bs) < 50 else {k: [round(float(np.percentile(bs[:, j], 2.5)), 2), round(float(np.percentile(bs[:, j], 97.5)), 2)]
                                                                      for j, k in enumerate(("slot", "raw_dt", "dist", "t_since"))},
                                        frac_boot_with_cliff=round(len(bs) / 300, 2),
                                        ci_censored={k: [round(float(np.percentile(bsc[:, j], 2.5)), 2), round(float(np.percentile(bsc[:, j], 97.5)), 2)]
                                                     for j, k in enumerate(("slot", "raw_dt", "dist", "t_since"))})
def cv(keys):
    out = {}
    for u in ("slot", "raw_dt", "dist", "t_since"):
        v = np.array([res[k]["cliff"][u] for k in keys if res[k]["cliff"]])
        out[u] = dict(n=len(v), mean=round(float(v.mean()), 2), cv=round(float(v.std() / v.mean()), 3)) if len(v) >= 2 else None
    return out
groups = {"all15": list(res), "ctx8 (s1_C16, s2_C16)": [k for k in res if res[k]["arm"] in ("s1_C16", "s2_C16")],
          "ctx4 (s2_C8, s4_C8)": [k for k in res if res[k]["arm"] in ("s2_C8", "s4_C8")],
          "same raw span (s1_C32, s2_C16, s4_C8)": [k for k in res if res[k]["arm"] in ("s1_C32", "s2_C16", "s4_C8")]}
for a in ("s1_C32", "s1_C16", "s2_C16", "s2_C8", "s4_C8"): groups[f"speed only ({a})"] = [k for k in res if res[k]["arm"] == a]
disp = {g: cv(k) for g, k in groups.items()}
OUTD = dict(cliffs=res, dispersion=disp)
print("절벽 (슬롯 / raw 프레임 / 칸)  [95% CI]")
for k, r in res.items():
    c = r["cliff"]
    print(f"{k:12s} 속도 {r['speed']:.3f} 칸/프레임 | " + ("창 안에서 안 무너짐" if c is None else
          f"slot {c['slot']:5.2f}  raw {c['raw_dt']:5.1f}  dist {c['dist']:4.2f}") + (f"   CI {r['ci']}" if r["ci"] else "") + f"  (boot {r['frac_boot_with_cliff']})")
print("\n단위별 변동계수 (작을수록 그 단위가 절벽을 정한다)")
for g, d in disp.items():
    print(f"{g:34s} " + "  ".join(f"{u} cv {d[u]['cv']:.3f} (평균 {d[u]['mean']}, n {d[u]['n']})" if d[u] else f"{u} –" for u in ("slot", "raw_dt", "dist")))


# =====================================================================================================================
# v2 — 적대 검증 (VERIFY_0925_2026-09-25, B9–B13) 반영.  아래는 모두 **새 키** 다.  난수열은 rng2 로 분리해 위 기존 키 값은 바뀌지 않는다.
#   귀무      = 같은 법칙 · 다른 궤적 clip **전부** 의 진실 칸에 대한 적중률 평균 (B12)
#   절벽      = 보간 첫 교차 (절반 규칙).  창 안에서 안 지나면 창 끝으로 **중도절단** (하한, crossed=False) (B10)
#   시간 단위 = t_since: 마지막 문맥 튜블릿 중심부터의 raw 프레임 (dist 와 같은 원점) (B11)
#   δ         = 문맥 끝 속도 (칸/raw 프레임, 3 분위 평균) × 2s  = 튜블릿당 이동 (칸)
#   CI        = 궤적 군집 bootstrap (84 궤적, B=1000; 문맥 전부 가시 부분집합은 18 궤적)
#   표적      = 템플릿 · h_t 모두 표준 양방향 target encoder (인과 표적 판 없음)
# =====================================================================================================================
from scipy.optimize import least_squares

rng2 = np.random.default_rng(20260925)
THR = 0.357; B2 = int(os.environ.get("P2B_BOOT", 1000))
LAWS = np.unique(scen); SAME = {L: np.where(scen == L)[0] for L in LAWS}
vq = np.array([spd[terc == q].mean() for q in range(3)])
tr_of = {t: i for i, t in enumerate(ut)}; tix_arr = np.array([tr_of[t] for t in traj])
pos = lambda a, b: np.stack([(X[:, a] + X[:, b]) / 2, (Y[:, a] + Y[:, b]) / 2], -1) / CELL
GEO = []
for A in arms:
    fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
    cl = pos(fa, fb); ct = np.stack([pos(fr[2 * (nc + t)], fr[2 * (nc + t) + 1]) for t in range(nf)], 1)
    okt = np.stack([INF[:, fr[2 * (nc + t)]] & INF[:, fr[2 * (nc + t) + 1]] for t in range(nf)], 1)
    GEO.append(dict(name=A["name"], s=A["stride"], nc=nc, nf=nf, cl=cl, ct=ct, okl=INF[:, fa] & INF[:, fb], okt=okt,
                    cti=np.clip(np.floor(ct), 0, 15), DI=np.linalg.norm(ct - cl[:, None], axis=-1),
                    ts=np.array([(fr[2 * (nc + t)] + fr[2 * (nc + t) + 1]) / 2 - (fa + fb) / 2 for t in range(nf)]),
                    vis=np.stack([INF[:, fr[2 * k]] & INF[:, fr[2 * k + 1]] for k in range(nc)], 1).sum(1)))
NAMES = [g["name"] for g in GEO]; AI = {nm: i for i, nm in enumerate(NAMES)}


def score(AM, tol=1):
    """AM[ai]: (n, nf, 2) argmax 칸 (x, y).  → 팔마다 dict(HIT, NUL, HN) (n, nf), 무효 = nan.  귀무 = 전수 평균."""
    out = []
    for ai, g in enumerate(GEO):
        HIT = np.full((n, g["nf"]), np.nan); NUL = HIT.copy()
        for t in range(g["nf"]):
            am = AM[ai][:, t]; ok = g["okl"] & g["okt"][:, t] & ~np.isnan(am[:, 0])
            h = (np.abs(am - g["cti"][:, t]).max(-1) <= tol).astype(float); nu = np.full(n, np.nan)
            for L, ix in SAME.items():
                M = np.abs(am[ix][:, None] - g["cti"][ix, t][None]).max(-1) <= tol
                V = (traj[ix][:, None] != traj[ix][None]) & g["okt"][ix, t][None]
                nu[ix] = (M & V).sum(1) / np.maximum(V.sum(1), 1)
            HIT[ok, t] = h[ok]; NUL[ok, t] = nu[ok]
        out.append(dict(HIT=HIT, NUL=NUL, HN=HIT - NUL))
    return out


def wmean(M, w):
    v = ~np.isnan(M); return (np.where(v, M, 0).T @ w) / np.maximum(v.T @ w, 1e-12)


def at(arr, x):
    i = int(np.floor(x)); f = x - i
    return float(arr[i]) if i + 1 >= len(arr) else float(arr[i] + f * (arr[i + 1] - arr[i]))


def cliff_w(M, g, w, frac=0.5, abs_th=None):
    """가중 평균 곡선의 첫 교차 (보간).  안 지나면 창 끝으로 중도절단 (crossed=False)."""
    m = wmean(M, w); dm = wmean(np.where(np.isnan(M), np.nan, g["DI"]), w)
    th = frac * m[0] if abs_th is None else abs_th; x, crossed = float(len(m) - 1), False
    for t in range(1, len(m)):
        if m[t] < th: x, crossed = t - 1 + (m[t - 1] - th) / max(m[t - 1] - m[t], 1e-9), True; break
    return dict(slot=x, dist=at(dm, x), t_since=at(g["ts"], x), crossed=crossed, m0=float(m[0]))


def cliffs15(S, key="HN", w=None, sub=None, **rule):
    """팔 5 × 속도 3 분위.  S: score() 출력 또는 팔마다 dict(key → (n, nf))."""
    base = np.ones(n) if w is None else w; base = base * (1.0 if sub is None else sub)
    out = []
    for ai, g in enumerate(GEO):
        for q in range(3):
            c = cliff_w(S[ai][key], g, base * (terc == q), **rule)
            c.update(arm=g["name"], q=q, s=g["s"], nc=g["nc"], v=float(vq[q]), delta=float(vq[q] * 2 * g["s"])); out.append(c)
    return out


def pooled(S, key="HN", w=None, sub=None, **rule):
    base = np.ones(n) if w is None else w; base = base * (1.0 if sub is None else sub)
    return {g["name"]: cliff_w(S[ai][key], g, base, **rule) for ai, g in enumerate(GEO)}


# ---- 모형 (절벽 거리, 칸).  1 모수 셋은 비교 기준 (2 모수 모형과 같은 저울이 아님)
def _lin(cols):
    return (lambda d, s, v: np.stack(cols(d, s, v), 1).astype(float))


LINEAR = {"D0 (1p)": _lin(lambda d, s, v: [np.ones_like(d)]), "K*delta (1p)": _lin(lambda d, s, v: [d]), "T*speed (1p)": _lin(lambda d, s, v: [v]),
          "D0+T*speed": _lin(lambda d, s, v: [np.ones_like(d), v]), "D0+k*delta": _lin(lambda d, s, v: [np.ones_like(d), d]),
          "time+steps (T*speed+k*delta)": _lin(lambda d, s, v: [v, d]), "a+b*stride": _lin(lambda d, s, v: [np.ones_like(d), s]),
          "a+b*log2(stride)": _lin(lambda d, s, v: [np.ones_like(d), np.log2(s)]),
          "D0+k*delta+T*speed (3p)": _lin(lambda d, s, v: [np.ones_like(d), d, v]),
          "stride factor (3p)": _lin(lambda d, s, v: [s == 1, s == 2, s == 4])}


def fit_pred(name, d, s, v, y, d2, s2, v2):
    if name == "sqrt(R^2+(K*delta)^2)":
        r = least_squares(lambda p: np.sqrt(p[0] ** 2 + (p[1] * d) ** 2) - y, x0=[2.5, 2.0])
        return r.x, np.sqrt(r.x[0] ** 2 + (r.x[1] * d2) ** 2)
    Xd = LINEAR[name](d, s, v)
    if np.linalg.matrix_rank(Xd) < Xd.shape[1]: return None, None
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]; return b, LINEAR[name](d2, s2, v2) @ b


def arr15(C, drop=True):
    k = [c for c in C if c["crossed"] or not drop]
    return (np.array([c["delta"] for c in k]), np.array([c["s"] for c in k], float), np.array([c["v"] for c in k]),
            np.array([c["dist"] for c in k]), np.array([c["arm"] for c in k]))


def eval_models(C, names=None, drop=True, cv=True):
    d, s, v, y, arm = arr15(C, drop); sst = ((y - y.mean()) ** 2).sum(); out = {}
    for nm in (names or list(LINEAR) + ["sqrt(R^2+(K*delta)^2)"]):
        b, yh = fit_pred(nm, d, s, v, y, d, s, v)
        if b is None: continue
        r = dict(coef=[round(float(x), 3) for x in b], R2=round(float(1 - ((y - yh) ** 2).sum() / sst), 3), RMSE=round(float(np.sqrt(((y - yh) ** 2).mean())), 3), n=len(y))
        if cv:
            lo = []
            for st in (1, 2, 4):
                tr = s != st; b2, yh2 = fit_pred(nm, d[tr], s[tr], v[tr], y[tr], d[~tr], s[~tr], v[~tr])
                lo.append(None if b2 is None or (~tr).sum() == 0 else round(float(np.abs(y[~tr] - yh2).mean()), 3))
            la = []
            for a in np.unique(arm):
                tr = arm != a; b2, yh2 = fit_pred(nm, d[tr], s[tr], v[tr], y[tr], d[~tr], s[~tr], v[~tr])
                if b2 is not None: la.append(np.abs(y[~tr] - yh2).mean())
            pr = []
            for k in range(len(y)):
                tr = np.arange(len(y)) != k; b2, yh2 = fit_pred(nm, d[tr], s[tr], v[tr], y[tr], d[~tr], s[~tr], v[~tr])
                pr.append(np.nan if b2 is None else (y[k] - yh2[0]) ** 2)
            r.update(LOSO_mae_s1_s2_s4=lo, LOAO_mae=round(float(np.mean(la)), 3) if la else None, Q2_LOO=round(float(1 - np.nansum(pr) / sst), 3))
        out[nm] = r
    return out


def r2_quick(C, nm, drop=True):
    d, s, v, y, _ = arr15(C, drop); b, yh = fit_pred(nm, d, s, v, y, d, s, v)
    return (np.full(2, np.nan), np.nan) if b is None else (b, 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cvs(vals):
    v = np.asarray(vals, float); return float(v.std() / v.mean())


def pct(a, q=(2.5, 97.5)):
    return [round(float(x), 3) for x in np.nanpercentile(np.asarray(a, float), q)]


# ---- 판독 셋: p 위치 (진실 템플릿) · 물체다움 (공통 문턱) · 외형 템플릿 (p 에 건 것)
AMp = [LT[:, ai, :g["nf"], :][:, :, [1, 0]] for ai, g in enumerate(GEO)]
SP = score(AMp)
OBJ = []
for ai, g in enumerate(GEO):
    o = ((LT[:, ai, :g["nf"], 2] - LT[:, ai, :g["nf"], 3]) >= THR).astype(float)
    o[~(g["okl"][:, None] & g["okt"] & ~np.isnan(LT[:, ai, :g["nf"], 0]))] = np.nan; OBJ.append(dict(OBJ=o))
C15 = cliffs15(SP); O15 = cliffs15(OBJ, "OBJ")
fullvis = np.ones(n, bool)
for g in GEO: fullvis &= g["vis"] == g["nc"]

v2 = dict(meta=dict(null="같은 법칙 · 다른 궤적 clip 전부 평균", cliff="보간 첫 교차 · 절반 규칙 · 비교차는 창 끝 중도절단", t_since="마지막 문맥 튜블릿 중심부터 raw 프레임",
                    delta="문맥 끝 속도 × 2s (칸/튜블릿)", ci=f"궤적 군집 bootstrap B={B2}", target="표준 양방향 h (템플릿 · h_t)", tercile_speed=[round(float(x), 4) for x in vq]))
v2["context_visibility"] = {g["name"]: dict(n_ctx=g["nc"], mean=round(float(g["vis"].mean()), 2), median=float(np.median(g["vis"])), min=int(g["vis"].min())) for g in GEO}
v2["fullvis_subset"] = dict(n_clips=int(fullvis.sum()), n_traj=int(len(np.unique(traj[fullvis]))),
                            laws={L: int((fullvis & (scen == L)).sum()) for L in LAWS}, speed=round(float(spd[fullvis].mean()), 3), speed_all=round(float(spd.mean()), 3))
v2["curves_pooled"] = {g["name"]: dict(hit=[round(float(x), 3) for x in wmean(SP[ai]["HIT"], np.ones(n))], null=[round(float(x), 3) for x in wmean(SP[ai]["NUL"], np.ones(n))],
                                       hit_null=[round(float(x), 3) for x in wmean(SP[ai]["HN"], np.ones(n))], obj=[round(float(x), 3) for x in wmean(OBJ[ai]["OBJ"], np.ones(n))],
                                       t_since=g["ts"].tolist()) for ai, g in enumerate(GEO)}

# ---- 점추정: 15 절벽 · CV · 모형
key15 = lambda c: f"{c['arm']}|q{c['q']}"
rnd = lambda c: {k: (round(float(c[k]), 3) if isinstance(c[k], float) else c[k]) for k in ("slot", "dist", "t_since", "crossed", "delta", "m0")}
v2["cliffs_hit_null"] = {key15(c): rnd(c) for c in C15}
v2["cliffs_obj"] = {key15(c): rnd(c) for c in O15}
GROUPS = {"all15": NAMES, "ctx8 (s1_C16, s2_C16)": ["s1_C16", "s2_C16"], "ctx4 (s2_C8, s4_C8)": ["s2_C8", "s4_C8"],
          "same raw span (s1_C32, s2_C16, s4_C8)": ["s1_C32", "s2_C16", "s4_C8"], **{f"speed only ({a})": [a] for a in NAMES}}
disp_pt = lambda C: {gname: {u: round(cvs([c[u] for c in C if c["arm"] in arms_]), 3) for u in ("slot", "t_since", "dist")} for gname, arms_ in GROUPS.items()}
v2["dispersion_hit_null"] = disp_pt(C15); v2["dispersion_obj"] = disp_pt(O15)
v2["models_hit_null"] = eval_models(C15)
v2["models_obj"] = eval_models(O15, names=["D0+k*delta", "time+steps (T*speed+k*delta)", "a+b*stride", "T*speed (1p)", "D0 (1p)"], cv=False)
b_, _ = fit_pred("D0+k*delta", *arr15(C15)[:4], *arr15(C15)[:3])
slots = np.array([c["slot"] for c in C15]); dl = np.array([c["delta"] for c in C15]); di = np.array([c["dist"] for c in C15])
v2["slot_formula"] = dict(geometry_dist_minus_delta_slot_plus_1=dict(max_abs=round(float(np.abs(di - dl * (slots + 1)).max()), 3)),
                          doc_formula_2p64_over_delta_plus_1p89_mean_err=round(float((2.644 / dl + 1.892 - slots).mean()), 3),
                          corrected_D0_over_delta_plus_k_minus_1=dict(D0=round(float(b_[0]), 3), k=round(float(b_[1]), 3),
                                                                     mean_err=round(float((b_[0] / dl + b_[1] - 1 - slots).mean()), 3)))

# ---- 문턱 · 허용폭 민감도 (점추정, 교차한 점만 적합; n 을 같이 적는다)
SENS = ["D0+k*delta", "time+steps (T*speed+k*delta)", "a+b*stride", "D0+T*speed"]
v2["threshold_sensitivity"] = {}
for nm_, rule in [("rel0.3", dict(frac=0.3)), ("rel0.4", dict(frac=0.4)), ("rel0.5", dict(frac=0.5)), ("rel0.6", dict(frac=0.6)), ("rel0.7", dict(frac=0.7)),
                  ("abs0.0", dict(abs_th=0.0)), ("abs0.1", dict(abs_th=0.1)), ("abs0.2", dict(abs_th=0.2))]:
    C = cliffs15(SP, **rule)
    v2["threshold_sensitivity"][nm_] = dict(censored=[key15(c) for c in C if not c["crossed"]], models=eval_models(C, SENS, cv=True))


def freeze(K):
    return [np.clip(np.floor(g["cl"][:, None].repeat(g["nf"], 1) if K == 0 else g["ct"][:, np.minimum(np.arange(g["nf"]), K - 1)]), 0, 15) for g in GEO]


def radius(R, lost="freeze"):
    out = []
    for g in GEO:
        run = np.cumprod(g["DI"] <= R, axis=1).astype(bool); last = run.sum(1) - 1; P = g["ct"].copy()
        for i in range(n):
            k = last[i]; P[i, k + 1:] = (g["ct"][i, k] if (k >= 0 and lost == "freeze") else g["cl"][i])
        out.append(np.clip(np.floor(P), 0, 15))
    return out


ORC = {"copy (K=0)": freeze(0), "freeze K=1": freeze(1), "freeze K=2": freeze(2), "freeze K=3": freeze(3),
       "radius R=2 then freeze": radius(2), "radius R=3 then freeze": radius(3), "radius R=4 then freeze": radius(4), "radius R=3 then copy": radius(3, "copy")}
v2["oracles"] = {}
for nm_, AM in ORC.items():
    C = cliffs15(score(AM))
    v2["oracles"][nm_] = dict(dists=[round(c["dist"], 2) if c["crossed"] else None for c in C], slots=[round(c["slot"], 2) if c["crossed"] else None for c in C],
                              models=eval_models(C, ["D0+k*delta", "a+b*stride", "time+steps (T*speed+k*delta)"], cv=False))
v2["tolerance_sensitivity"] = {}
for tol in (0, 1, 2):
    row = {}
    for nm_, AM in (("p", AMp), ("freeze K=2", ORC["freeze K=2"]), ("freeze K=3", ORC["freeze K=3"])):
        C = cliffs15(score(AM, tol))
        row[nm_] = dict(censored=[key15(c) for c in C if not c["crossed"]], models=eval_models(C, SENS, cv=False))
    row["p_minus_freezeK2_D0"] = round(row["p"]["models"]["D0+k*delta"]["coef"][0] - row["freeze K=2"]["models"]["D0+k*delta"]["coef"][0], 3)
    row["p_minus_freezeK3_D0"] = round(row["p"]["models"]["D0+k*delta"]["coef"][0] - row["freeze K=3"]["models"]["D0+k*delta"]["coef"][0], 3)
    v2["tolerance_sensitivity"][f"tol{tol}"] = row

# ---- 법칙별 (팔 × 법칙, 분위 합침) + 법칙 고정효과 적합 (y = a_law + k·δ)
def law_cliffs(S, w=None):
    base = np.ones(n) if w is None else w; out = []
    for ai, g in enumerate(GEO):
        for L in LAWS:
            c = cliff_w(S[ai]["HN"], g, base * (scen == L)); c.update(arm=g["name"], law=L, s=g["s"], delta=float(spd[scen == L].mean() * 2 * g["s"])); out.append(c)
    return out


def law_fe(LC):
    k = [c for c in LC if c["crossed"]]; y = np.array([c["dist"] for c in k])
    Xd = np.column_stack([np.array([c["law"] == L for c in k], float) for L in LAWS] + [np.array([c["delta"] for c in k])])
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]; e = y - Xd @ b
    return b, float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


LC = law_cliffs(SP); bfe, r2fe = law_fe(LC)
v2["law"] = dict(cliffs={f"{c['arm']}|{c['law']}": dict(dist=round(c["dist"], 3), slot=round(c["slot"], 3), crossed=c["crossed"], delta=round(c["delta"], 3), m0=round(c["m0"], 3)) for c in LC},
                 per_law_fit={L: eval_models([dict(c, v=c["delta"] / 2 / c["s"]) for c in LC if c["law"] == L], ["D0+k*delta", "a+b*stride"], cv=False) for L in LAWS},
                 law_fixed_effect=dict(intercept_by_law={L: round(float(bfe[j]), 3) for j, L in enumerate(LAWS)}, k=round(float(bfe[-1]), 3), R2=round(r2fe, 3)))

# ---- δ 를 실제 미래 변위로 (첫 4 미래 튜블릿의 튜블릿당 평균 이동, 진실 궤적)
Cf = []
for c in C15:
    g = GEO[AI[c["arm"]]]; sel = terc == c["q"]
    Cf.append(dict(c, delta=float(np.mean([g["DI"][sel, t].mean() / (t + 1) for t in range(4)]))))
v2["delta_future"] = dict(delta=[round(c["delta"], 3) for c in Cf], models=eval_models(Cf, ["D0+k*delta", "a+b*stride", "time+steps (T*speed+k*delta)"], cv=True))

# ---- 진행: p argmax (칸 중심) 의 진실 방향 투영 거리 (중앙값) 대 진실 이동 (마지막 관측 기준)
v2["progress"] = {}
for ai, g in enumerate(GEO):
    for q in range(3):
        sel = (terc == q) & g["okl"]; rt, rp, ro, rh, rn = [], [], [], [], []
        for t in range(g["nf"]):
            ok = sel & g["okt"][:, t]; d = g["ct"][ok, t] - g["cl"][ok]; D = np.linalg.norm(d, axis=-1); u = d / np.maximum(D, 1e-6)[:, None]
            rt.append(float(np.median(D))); rp.append(float(np.median(((AMp[ai][ok, t] + 0.5 - g["cl"][ok]) * u).sum(-1))))
            ro.append(float(np.nanmean(OBJ[ai]["OBJ"][ok, t]))); rh.append(float(np.nanmean(SP[ai]["HIT"][ok, t]))); rn.append(float(np.nanmean(SP[ai]["NUL"][ok, t])))
        lag = np.array(rt) - np.array(rp); fs = int(np.argmax(lag > 0.5)) if (lag > 0.5).any() else g["nf"]     # 진실보다 0.5 칸 넘게 뒤처지기 전 슬롯 수
        v2["progress"][f"{g['name']}|q{q}"] = dict(delta=round(float(vq[q] * 2 * g["s"]), 3), truth=[round(x, 2) for x in rt], p_proj=[round(x, 2) for x in rp],
                                                   obj=[round(x, 3) for x in ro], hit=[round(x, 3) for x in rh], null=[round(x, 3) for x in rn],
                                                   follow_slots=fs, follow_dist=(round(rp[fs - 1], 2) if fs > 0 else 0.0))

# ---- 문맥 전부 가시 부분집합 (252 clip · 18 궤적) : 팔별 (분위 합침) 절벽
PF_hn = pooled(SP, sub=fullvis.astype(float)); PF_obj = pooled(OBJ, "OBJ", sub=fullvis.astype(float))
PA_hn = pooled(SP); PA_obj = pooled(OBJ, "OBJ")
PAIRS = [("s1_C16", "s2_C16"), ("s2_C8", "s4_C8"), ("s2_C16", "s2_C8"), ("s1_C32", "s1_C16"), ("s1_C32", "s2_C16"), ("s2_C16", "s4_C8")]

# ---- 궤적 군집 bootstrap (전체 84 궤적) : 적합 · CV · 팔별 절벽 · 차이 · 법칙 FE
bt = {k: [] for k in ("D0", "k", "R2_D0kd", "R2_abs", "R2_timesteps", "R2_sqrt", "dR2_D0kd_minus_abs", "n_censored", "D0_drop", "k_drop", "R2_D0kd_drop",
                      "dR2_drop", "k_lawFE")}
cvb = {f"{rd}|{grp}": [] for rd in ("hit_null", "obj") for grp in ("all15", "s1_C16")}
cl_b = {key15(c): [] for c in C15}; pa_b = {f"{rd}|{a}": [] for rd in ("hit_null", "obj") for a in NAMES}
for b in range(B2):
    w = np.bincount(rng2.integers(0, len(ut), len(ut)), minlength=len(ut))[tix_arr].astype(float)
    Cb = cliffs15(SP, w=w); Ob = cliffs15(OBJ, "OBJ", w=w)
    (bd, r2d), (_, r2a), (_, r2t), (_, r2s) = (r2_quick(Cb, m, drop=False) for m in ("D0+k*delta", "a+b*stride", "time+steps (T*speed+k*delta)", "sqrt(R^2+(K*delta)^2)"))
    (bd2, r2d2), (_, r2a2) = (r2_quick(Cb, m, drop=True) for m in ("D0+k*delta", "a+b*stride"))
    bt["D0"].append(bd[0]); bt["k"].append(bd[1]); bt["R2_D0kd"].append(r2d); bt["R2_abs"].append(r2a); bt["R2_timesteps"].append(r2t); bt["R2_sqrt"].append(r2s)
    bt["dR2_D0kd_minus_abs"].append(r2d - r2a); bt["n_censored"].append(sum(not c["crossed"] for c in Cb))
    bt["D0_drop"].append(bd2[0]); bt["k_drop"].append(bd2[1]); bt["R2_D0kd_drop"].append(r2d2); bt["dR2_drop"].append(r2d2 - r2a2)
    bt["k_lawFE"].append(law_fe(law_cliffs(SP, w))[0][-1])
    for rd, CC in (("hit_null", Cb), ("obj", Ob)):
        for grp, arms_ in (("all15", NAMES), ("s1_C16", ["s1_C16"])):
            k_ = [c for c in CC if c["arm"] in arms_]; cvb[f"{rd}|{grp}"].append([cvs([c[u] for c in k_]) for u in ("slot", "t_since", "dist")])
    for c in Cb: cl_b[key15(c)].append([c["slot"], c["dist"], c["t_since"]])
    for rd, P in (("hit_null", pooled(SP, w=w)), ("obj", pooled(OBJ, "OBJ", w=w))):
        for a in NAMES: pa_b[f"{rd}|{a}"].append(P[a]["slot"])
bt = {k: np.array(v, float) for k, v in bt.items()}
v2["bootstrap_fit"] = dict(B=B2, censor="창 끝", D0=pct(bt["D0"]), k=pct(bt["k"]), R2_D0kd=pct(bt["R2_D0kd"]), R2_a_b_stride=pct(bt["R2_abs"]),
                           R2_time_steps=pct(bt["R2_timesteps"]), R2_sqrt=pct(bt["R2_sqrt"]), dR2_D0kd_minus_a_b_stride=pct(bt["dR2_D0kd_minus_abs"]),
                           P_dR2_le_0=round(float((bt["dR2_D0kd_minus_abs"] <= 0).mean()), 3), frac_boot_any_censored=round(float((bt["n_censored"] > 0).mean()), 3),
                           drop_censored=dict(D0=pct(bt["D0_drop"]), k=pct(bt["k_drop"]), R2_D0kd=pct(bt["R2_D0kd_drop"]), dR2=pct(bt["dR2_drop"]),
                                              P_dR2_le_0=round(float((bt["dR2_drop"] <= 0).mean()), 3)),
                           k_law_fixed_effect=pct(bt["k_lawFE"]))
v2["bootstrap_cv"] = {}
for kk, arr in cvb.items():
    a = np.array(arr); rd, grp = kk.split("|"); C = C15 if rd == "hit_null" else O15; k_ = [c for c in C if c["arm"] in (NAMES if grp == "all15" else ["s1_C16"])]
    v2["bootstrap_cv"][kk] = dict(point={u: round(cvs([c[u] for c in k_]), 3) for u in ("slot", "t_since", "dist")},
                                  ci={u: pct(a[:, j]) for j, u in enumerate(("slot", "t_since", "dist"))},
                                  P_tsince_lt_dist=round(float((a[:, 1] < a[:, 2]).mean()), 3), P_slot_lt_dist=round(float((a[:, 0] < a[:, 2]).mean()), 3),
                                  P_slot_lt_tsince=round(float((a[:, 0] < a[:, 1]).mean()), 3))
v2["cliff_ci"] = {k: {u: pct(np.array(a)[:, j]) for j, u in enumerate(("slot", "dist", "t_since"))} for k, a in cl_b.items()}
v2["pooled_arm"] = {}
for rd, P in (("hit_null", PA_hn), ("obj", PA_obj)):
    v2["pooled_arm"][rd] = {a: dict(slot=round(P[a]["slot"], 3), dist=round(P[a]["dist"], 3), t_since=round(P[a]["t_since"], 3), crossed=P[a]["crossed"],
                                    slot_ci=pct(pa_b[f"{rd}|{a}"])) for a in NAMES}
    v2["pooled_arm"][rd + "_diff"] = {f"{a}-{b_}": dict(slot=round(P[a]["slot"] - P[b_]["slot"], 3), ci=pct(np.array(pa_b[f"{rd}|{a}"]) - np.array(pa_b[f"{rd}|{b_}"]))) for a, b_ in PAIRS}

# ---- 문맥 전부 가시 부분집합 bootstrap (18 궤적)
ufv = np.unique(traj[fullvis]); fv_b = {f"{rd}|{a}": [] for rd in ("hit_null", "obj") for a in NAMES}
for b in range(B2):
    cnt = np.bincount(rng2.integers(0, len(ufv), len(ufv)), minlength=len(ufv)); wt = dict(zip(ufv, cnt))
    w = np.array([wt.get(t, 0) for t in traj], float) * fullvis
    for rd, P in (("hit_null", pooled(SP, w=w)), ("obj", pooled(OBJ, "OBJ", w=w))):
        for a in NAMES: fv_b[f"{rd}|{a}"].append(P[a]["slot"])
v2["fullvis"] = {}
for rd, P in (("hit_null", PF_hn), ("obj", PF_obj)):
    v2["fullvis"][rd] = {a: dict(slot=round(P[a]["slot"], 3), dist=round(P[a]["dist"], 3), t_since=round(P[a]["t_since"], 3), crossed=P[a]["crossed"], m0=round(P[a]["m0"], 3),
                                 slot_ci=pct(fv_b[f"{rd}|{a}"])) for a in NAMES}
    v2["fullvis"][rd + "_diff"] = {f"{a}-{b_}": dict(slot=round(P[a]["slot"] - P[b_]["slot"], 3), ci=pct(np.array(fv_b[f"{rd}|{a}"]) - np.array(fv_b[f"{rd}|{b_}"])))
                                   for a, b_ in PAIRS[:2]}

OUTD["v2"] = v2
json.dump(OUTD, open(OUT / f"p2b_units{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)

# ---- 요약 출력
print("\n===== v2 (전수 귀무 · 중도절단 · t_since 원점) =====")
for c in C15: print(f"{key15(c):11s} δ {c['delta']:.3f} | hit−null 절벽 slot {c['slot']:5.2f} t_since {c['t_since']:5.1f} dist {c['dist']:4.2f}{'' if c['crossed'] else ' (중도절단)'}"
                    f"  CI slot {v2['cliff_ci'][key15(c)]['slot']}")
for nm_, r in v2["models_hit_null"].items(): print(f"  {nm_:32s} {r}")
print("bootstrap 적합:", json.dumps(v2["bootstrap_fit"], ensure_ascii=False))
for kk, r in v2["bootstrap_cv"].items(): print("CV", kk, json.dumps(r, ensure_ascii=False))
print("pooled:", json.dumps(v2["pooled_arm"], ensure_ascii=False))
print("fullvis:", json.dumps(v2["fullvis"], ensure_ascii=False), json.dumps(v2["fullvis_subset"], ensure_ascii=False))
print("oracles:", json.dumps({k: {m: (r['models'][m]['coef'], r['models'][m]['R2']) for m in r['models']} for k, r in v2["oracles"].items()}, ensure_ascii=False))
print("tol:", json.dumps({k: {m: (r[m]['models']['D0+k*delta']['coef'], r[m]['models']['D0+k*delta']['R2'], r[m]['censored']) for m in ('p', 'freeze K=2', 'freeze K=3')} for k, r in v2["tolerance_sensitivity"].items()}, ensure_ascii=False))
print("thr:", json.dumps({k: {m: (r['models'][m]['coef'], r['models'][m]['R2']) for m in r['models']} for k, r in v2["threshold_sensitivity"].items()}, ensure_ascii=False))
print("law:", json.dumps(v2["law"], ensure_ascii=False))
print("slot formula:", v2["slot_formula"])
print("progress (follow_slots, follow_dist, p_proj 마지막):", {k: (r["follow_slots"], r["follow_dist"], r["p_proj"][-1]) for k, r in v2["progress"].items()})
