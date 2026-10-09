#!/usr/bin/env python3
"""SC1 — 지평의 단위를 predictor 마다: 슬롯 (튜블릿 수) · 거리 (칸) · 경과 raw 프레임 중 무엇이 절벽을 한 자리에 모으나 (2026-09-25).

입력: predictor 태그마다 두 캐시 (p2_stride_sweep_v3.py 산출물)
  v3_p2{_tag}     분할 raw 32: s1_C32 · s2_C16 · s4_C8 · s1_C16 · s2_C8
  v3_p2s16_{tag}  분할 raw 16: s2_S16 (문맥 4 / 미래 12) · s1_S16 (문맥 4 / 미래 24)
정의 (p2_analyze.py 와 같음): hit = 진실 칸 템플릿 argmax 가 진실 칸 1 칸 안,  null_ex = 같은 법칙 · 다른 궤적 clip 전부의 진실 칸 적중률 평균.
곡선 = (팔 × 문맥 끝 속도 3 분위 [법칙 안에서 나눔]) 마다 hit − null_ex 의 슬롯 평균.
절벽 = 곡선이 슬롯 0 값의 절반을 처음 아래로 지나는 점 (선형 보간). 창 안에서 안 지나면 중도절단 (≥ 창 끝) — CV 에서 빼고 수를 적는다.
단위: slot · dist (문맥 끝 GT 자리에서의 평균 거리, 칸 = 18 px) · t_since (문맥 마지막 프레임에서 경과 raw 프레임).
CV = sd/mean (절벽들 사이). 궤적 군집 bootstrap B 회로 CV 의 CI 와 P(CV_slot < CV_dist).

  python sc1_horizon_units.py release= pv1=_pv1 ariel=_ariel v11ft=_v11ft      (= 뒤는 v3_p2 접미사; s16 은 v3_p2s16_<태그>)
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
OUT = ROOT / "auto_research/exp_results/p2"; OUT.mkdir(parents=True, exist_ok=True)
CELL = 18.0; B = int(os.environ.get("SC1_BOOT", 300))
IDX = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])


def load(I):
    meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; n = len(ids)
    LT = np.load(I / "loc_tru.npy")
    X = np.stack([f(IDX[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(IDX[v]["px_y_by_sample"]) for v in ids])
    INF = np.stack([f(IDX[v]["in_frame_by_sample"]) for v in ids]) > 0.5
    scen = np.array(meta["scenario"]); traj = np.array(["|".join(map(str, t)) for t in meta["traj"]])
    out = []
    for ai, A in enumerate(meta["arms"]):
        fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
        fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
        cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
        okl = INF[:, fa] & INF[:, fb]
        spd = np.hypot(X[:, fb] - X[:, fr[2 * nc - 4]], Y[:, fb] - Y[:, fr[2 * nc - 4]]) / (fb - fr[2 * nc - 4]) / CELL   # 칸/raw 프레임 (문맥 끝 한 튜블릿 반)
        HN = np.full((n, nf), np.nan); DI = np.full((n, nf), np.nan); TS = np.zeros(nf)
        for t in range(nf):
            g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
            ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
            ok = okl & INF[:, g0] & INF[:, g1] & ~np.isnan(LT[:, ai, t, 0])
            cti = np.clip(np.floor(ct), 0, 15); am = LT[:, ai, t, [1, 0]]
            hit = (np.abs(am - cti).max(-1) <= 1).astype(float)
            nex = np.full(n, np.nan)
            for L in np.unique(scen):
                ix = np.where(scen == L)[0]
                M = np.abs(am[ix][:, None] - cti[ix][None]).max(-1) <= 1
                V = (traj[ix][:, None] != traj[ix][None]) & ok[ix][None]
                nex[ix] = (M & V).sum(1) / np.maximum(V.sum(1), 1)
            HN[:, t] = np.where(ok, hit - nex, np.nan); DI[:, t] = np.where(ok, np.linalg.norm(ct - cl, axis=-1), np.nan)
            TS[t] = (g0 + g1) / 2 - fb
        terc = np.zeros(n, int)
        for L in np.unique(scen):
            ix = np.where(scen == L)[0]; q = np.nanquantile(spd[ix], [1 / 3, 2 / 3]); terc[ix] = np.digitize(spd[ix], q)
        out.append(dict(name=A["name"], stride=A["stride"], nc=nc, nf=nf, HN=HN, DI=DI, TS=TS, terc=terc, spd=spd))
    return out, traj


def cliff(curve, dist, ts):
    c0 = curve[0]
    if not np.isfinite(c0) or c0 <= 0: return None
    for t in range(1, len(curve)):
        if curve[t] < 0.5 * c0:
            a = (curve[t - 1] - 0.5 * c0) / max(curve[t - 1] - curve[t], 1e-9); a = min(max(a, 0.0), 1.0)
            lerp = lambda v: v[t - 1] + a * (v[t] - v[t - 1])
            return dict(slot=t - 1 + a, dist=float(lerp(dist)), t_since=float(lerp(ts)), censored=False)
    return dict(slot=float(len(curve) - 1), dist=float(dist[-1]), t_since=float(ts[-1]), censored=True)


def cliffs(arms, w):
    res = []
    for A in arms:
        for q in range(3):
            m = (A["terc"] == q)[:, None] * w[:, None]
            with np.errstate(invalid="ignore"):
                cur = np.nansum(A["HN"] * m, 0) / np.maximum(np.sum(np.isfinite(A["HN"]) * m, 0), 1e-9)
                dis = np.nansum(A["DI"] * m, 0) / np.maximum(np.sum(np.isfinite(A["DI"]) * m, 0), 1e-9)
            c = cliff(cur, dis, A["TS"])
            if c: res.append(dict(arm=A["name"], stride=A["stride"], nc=A["nc"], q=q, spd=float(np.nanmean(A["spd"][A["terc"] == q])), **c))
    return res


def cv(v):
    v = np.array(v, float); return float(v.std() / v.mean()) if len(v) > 1 and v.mean() > 0 else np.nan


def fitKD(cl):
    """절벽 슬롯 = K + D / δ  (δ = 문맥 끝 속도 × 2 × stride = 슬롯당 칸).  K = 거리와 무관한 슬롯 몫, D = 슬롯과 무관한 거리 몫.
    순수 슬롯 법칙이면 D ≈ 0, 순수 거리 법칙이면 K ≈ 0.  R2_slot_only = 상수 슬롯 모형, R2_dist_only = 상수 거리 모형 (절편 없는 D/δ) 의 R² (슬롯 공간)."""
    k_ = [c for c in cl if not c["censored"]]
    y = np.array([c["slot"] for c in k_]); x = np.array([1.0 / (2 * c["stride"] * c["spd"]) for c in k_])
    A = np.stack([np.ones_like(x), x], 1); (K, D), *_ = np.linalg.lstsq(A, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum(); r2 = 1 - ((y - A @ [K, D]) ** 2).sum() / sst
    Dd = (x * y).sum() / (x * x).sum(); r2d = 1 - ((y - Dd * x) ** 2).sum() / sst
    return dict(K=round(float(K), 3), D=round(float(D), 3), R2=round(float(r2), 3), R2_slot_only=0.0, R2_dist_only=round(float(r2d), 3), D_dist_only=round(float(Dd), 3), n=len(k_))


GROUPS = {"all": None, "ctx4": lambda c: c["nc"] == 4, "ctx8": lambda c: c["nc"] == 8, "split16": lambda c: c["arm"].endswith("S16"),
          "split32": lambda c: not c["arm"].endswith("S16")}
res = {}
for arg in sys.argv[1:]:
    tag, suf = arg.split("=", 1)
    a1, tr1 = load(C / f"v3_p2{suf}"); a2, tr2 = load(C / f"v3_p2s16_{tag}")
    assert np.array_equal(tr1, tr2), "두 캐시의 clip 순서가 다르다"
    arms = a1 + a2; traj = tr1; ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
    C0 = cliffs(arms, np.ones(len(traj)))
    rec = dict(cliffs=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in C0], cv={})
    rng = np.random.default_rng(0); BS = {g: [] for g in GROUPS}; FIT = []
    rec["fit_K_D"] = fitKD(C0)
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(ut), len(ut)), minlength=len(ut)); w = np.zeros(len(traj))
        for t, k in zip(ut, cnt): w[tix[t]] = k
        Cb = cliffs(arms, w); FIT.append(fitKD(Cb))
        for g, fn in GROUPS.items():
            k_ = [c for c in Cb if not c["censored"] and (fn is None or fn(c))]
            BS[g].append([cv([c[u] for c in k_]) for u in ("slot", "dist", "t_since")])
    for g, fn in GROUPS.items():
        k_ = [c for c in C0 if (fn is None or fn(c))]; kk = [c for c in k_ if not c["censored"]]
        a = np.array(BS[g], float)
        rec["cv"][g] = dict(n=len(k_), n_censored=len(k_) - len(kk),
                            point={u: round(cv([c[u] for c in kk]), 3) for u in ("slot", "dist", "t_since")},
                            mean={u: round(float(np.mean([c[u] for c in kk])), 2) if kk else np.nan for u in ("slot", "dist", "t_since")},
                            ci={u: [round(float(np.nanpercentile(a[:, j], 2.5)), 3), round(float(np.nanpercentile(a[:, j], 97.5)), 3)] for j, u in enumerate(("slot", "dist", "t_since"))},
                            P_slot_lt_dist=round(float(np.nanmean(a[:, 0] < a[:, 1])), 3), P_tsince_lt_dist=round(float(np.nanmean(a[:, 2] < a[:, 1])), 3))
    FA = np.array([[x["K"], x["D"], x["R2"], x["R2_slot_only"], x["R2_dist_only"]] for x in FIT], float)
    rec["fit_K_D"]["ci"] = {k: [round(float(np.nanpercentile(FA[:, j], 2.5)), 3), round(float(np.nanpercentile(FA[:, j], 97.5)), 3)]
                            for j, k in enumerate(("K", "D", "R2", "R2_slot_only", "R2_dist_only"))}
    res[tag] = rec

json.dump(res, open(OUT / f"sc1_horizon_units{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
for tag, rec in res.items():
    print(f"\n=== {tag}   절벽 (slot / dist 칸 / t_since raw)  * = 중도절단")
    for c in rec["cliffs"]:
        print(f"  {c['arm']:7s} q{c['q']} 속도 {c['spd']:.3f} | slot {c['slot']:5.2f}  dist {c['dist']:5.2f}  t_since {c['t_since']:5.1f} {'*' if c['censored'] else ''}")
    fk = rec["fit_K_D"]; print(f"  적합 slot = K + D/δ : K {fk['K']} {fk['ci']['K']}  D {fk['D']} {fk['ci']['D']}  R² {fk['R2']} {fk['ci']['R2']}  | 거리만 R² {fk['R2_dist_only']} (D {fk['D_dist_only']})  슬롯만 R² 0")
    for g, v in rec["cv"].items():
        print(f"  CV {g:8s} n {v['n']:2d} (절단 {v['n_censored']})  slot {v['point']['slot']:.3f} {v['ci']['slot']}  dist {v['point']['dist']:.3f} {v['ci']['dist']}  "
              f"t_since {v['point']['t_since']:.3f} {v['ci']['t_since']}  | 평균 {v['mean']}  P(slot<dist) {v['P_slot_lt_dist']}")
