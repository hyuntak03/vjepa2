#!/usr/bin/env python3
"""P2 분석 — stride sweep: 물체 신호의 지평이 튜블릿 수 · raw 시간 · 이동 거리 중 무엇에 묶이나.

입력 v3_p2/{meta.json, loc_tru.npy, loc_app.npy, loc_apph.npy, l1.npy}.  CI = 궤적 군집 bootstrap (B=1000).
정의 (PLAN_ENCODER_PREDICTOR_2026-09-25 §4):
  c*_t   미래 슬롯 t 의 진실 칸 (그 튜블릿 두 raw 프레임 평균 위치, 18 px 칸)   유효 = 두 프레임 모두 화면 안
  hit    ‖argmax − c*_t‖_∞ ≤ 1 (템플릿 = h_t 진실 칸)       null = 같은 법칙 · 다른 궤적 clip 의 c*_t 로 같은 판정
  obj    contrast (max − median cos) ≥ 0.357 (공통 문턱)
  app    외형 템플릿 (문맥 마지막 물체 토큰) argmax 가 c*_t 1 칸 안;  apph = 같은 것을 h_t 에 (encoder 천장)
  prog   외형 argmax 의 진행률 = (a − c_last)·(c*_t − c_last) / |c*_t − c_last|²   (|c*_t − c_last| ≥ 1 칸일 때만)
  EI     1 − |p_t − h_t|_win / |h_last − h_t|_win (진실 3×3 창)
  Δf     슬롯 중심 raw 프레임 − 31.5 (문맥 끝 두 프레임 중심에서 잰 경과)
  dist   |c*_t − c_last| (칸, 연속 위치)
지평 t* = (hit − null) 이 슬롯 0 값의 절반 아래로 처음 떨어지는 슬롯.
회귀 (선형 확률, 관측 단위 = clip×slot, 종속 = hit − null_i): ~ 1 + slot + Δf/8 + dist, 팔을 모두 합쳐서 (stride 가 slot 과 Δf 를 떼고,
  궤적 속도가 Δf 와 dist 를 뗀다). 계수 CI 는 궤적 군집 bootstrap.

추가 키 (적대 검증 2026-09-25, VERIFY_0925 B11 · B12 · §6).  기존 키 값은 그대로 (난수열 분리, rng2):
  t_since        슬롯 중심 raw 프레임 − 마지막 문맥 튜블릿 중심 (dist 와 같은 원점. Δf 의 원점 31.5 는 s4 에서 5.5 프레임 어긋난다)
  null_ex        같은 법칙 · 다른 궤적 clip **전부** 의 진실 칸에 대한 적중률 평균 (무작위 짝 1 개 대신 전수 평균)
  hit_null_ex    hit − null_ex;  t_star_ex = 그 곡선의 정수 t* (계획서 규칙)
  app_null_ex · apph_null_ex · *_minus_null_ex   외형 템플릿 판독 (§6) 의 위치 귀무와 귀무 보정값
  regression_ex_tsince   hit − null_ex ~ 1 + slot + t_since/8 + dist (궤적 군집 bootstrap)
표적: 템플릿 · h_t 는 모두 **표준 양방향 target encoder** (32 장 창 전체를 봄). 인과 표적 판은 없다.
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p2")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("P2_OUT_DIR", ROOT / "auto_research/exp_results/p2")); OUT.mkdir(parents=True, exist_ok=True)
CELL, THR = 18.0, 0.357
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
LT, LA, LH, L1 = (np.load(I / f"{k}.npy") for k in ("loc_tru", "loc_app", "loc_apph", "l1"))
LC = np.load(I / "loc_appc.npy") if (I / "loc_appc.npy").exists() else None          # 인과 외형 템플릿 (리뷰 B5; 없는 옛 추출은 건너뜀)
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
scen = np.array(meta["scenario"]); traj = np.array(["|".join(map(str, t)) for t in meta["traj"]])
rng = np.random.default_rng(0)
# 귀무 짝: 같은 법칙 · 다른 궤적
other = np.zeros(n, int)
for i in range(n):
    c = np.where((scen == scen[i]) & (traj != traj[i]))[0]; other[i] = rng.choice(c if len(c) else np.delete(np.arange(n), i))
ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
rng2 = np.random.default_rng(20260925)                                                        # 추가 키 전용 난수열
LAWS = np.unique(scen); SAME = {L: np.where(scen == L)[0] for L in LAWS}


def null_ex(am, cti, okt_t, tol=1):
    """귀무 전수 평균: 같은 법칙 · 다른 궤적 clip 전부의 진실 칸 (그 clip 에서 유효한 것만) 에 대한 적중률. am, cti: (n,2)."""
    out = np.full(n, np.nan)
    for L, ix in SAME.items():
        M = np.abs(am[ix][:, None] - cti[ix][None]).max(-1) <= tol
        V = (traj[ix][:, None] != traj[ix][None]) & okt_t[ix][None]
        out[ix] = (M & V).sum(1) / np.maximum(V.sum(1), 1)
    return out


def boot_mean2(v, B=300):
    m = np.nanmean(v); bs = []
    for _ in range(B):
        s = np.concatenate([tix[t] for t in rng2.choice(ut, len(ut))]); bs.append(np.nanmean(v[s]))
    return m, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def boot_mean(v, B=1000):
    """v: (n,) clip 값 (nan 허용) → 궤적 군집 bootstrap 평균 CI."""
    m = np.nanmean(v); bs = []
    for _ in range(B):
        s = np.concatenate([tix[t] for t in rng.choice(ut, len(ut))]); bs.append(np.nanmean(v[s]))
    return m, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


rows = []; per_arm = {}
for ai, A in enumerate(arms):
    fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]
    cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL          # (n,2) 연속 칸
    okl = INF[:, fa] & INF[:, fb]
    tab = []
    for t in range(nf):
        g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
        ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
        ok = okl & INF[:, g0] & INF[:, g1] & ~np.isnan(LT[:, ai, t, 0])
        cti = np.clip(np.floor(ct), 0, 15)
        am = LT[:, ai, t, [1, 0]]                                                              # (x, y)
        hit = (np.abs(am - cti).max(-1) <= 1).astype(float)
        nul = (np.abs(am - cti[other]).max(-1) <= 1).astype(float)
        obj = ((LT[:, ai, t, 2] - LT[:, ai, t, 3]) >= THR).astype(float)
        ap = LA[:, ai, t, [1, 0]]; aph = LH[:, ai, t, [1, 0]]
        app = (np.abs(ap - cti).max(-1) <= 1).astype(float); apph = (np.abs(aph - cti).max(-1) <= 1).astype(float)
        d = ct - cl; dist = np.linalg.norm(d, axis=-1)
        prog = np.where(dist >= 1, ((ap + 0.5 - cl) * d).sum(-1) / np.maximum(dist, 1e-6) ** 2, np.nan)
        ei = 1 - L1[:, ai, t, 2] / L1[:, ai, t, 3]
        df = (g0 + g1) / 2 - 31.5
        for v in (hit, nul, obj, app, apph, prog, ei): v[~ok] = np.nan
        r = dict(arm=A["name"], stride=A["stride"], slot=t, raw_dt=df, n=int(ok.sum()), dist=float(np.nanmean(np.where(ok, dist, np.nan))))
        for k, v in (("hit", hit), ("null", nul), ("hit_null", hit - nul), ("obj", obj), ("app", app), ("apph", apph), ("prog", prog), ("EI", ei)):
            m, lo, hi = boot_mean(v, B=300); r[k] = [round(m, 3), round(lo, 3), round(hi, 3)]
        # ---- 추가 키 (rng2): 전수 귀무 · 같은 원점 시간 · 외형 판독 귀무
        okt_t = INF[:, g0] & INF[:, g1]
        nex, nap, naph = null_ex(am, cti, okt_t), null_ex(ap, cti, okt_t), null_ex(aph, cti, okt_t)
        for v in (nex, nap, naph): v[~ok] = np.nan
        ts = (g0 + g1) / 2 - (fa + fb) / 2; r["t_since"] = ts
        for k, v in (("null_ex", nex), ("hit_null_ex", hit - nex), ("app_null_ex", nap), ("app_minus_null_ex", app - nap),
                     ("apph_null_ex", naph), ("apph_minus_null_ex", apph - naph)):
            m, lo, hi = boot_mean2(v); r[k] = [round(m, 3), round(lo, 3), round(hi, 3)]
        if LC is not None:                                                             # 인과 외형 템플릿: 적중 · 전수 귀무 · 진행
            apc = LC[:, ai, t, [1, 0]]; appc = (np.abs(apc - cti).max(-1) <= 1).astype(float); napc = null_ex(apc, cti, okt_t)
            progc = np.where(dist >= 1, ((apc + 0.5 - cl) * d).sum(-1) / np.maximum(dist, 1e-6) ** 2, np.nan)
            for v in (appc, napc, progc): v[~ok] = np.nan
            for k, v in (("appc", appc), ("appc_null_ex", napc), ("appc_minus_null_ex", appc - napc), ("progc", progc)):
                m, lo, hi = boot_mean2(v); r[k] = [round(m, 3), round(lo, 3), round(hi, 3)]
        tab.append(r); rows.append(dict(ai=ai, t=t, df=df, ok=ok, dist=dist, hn=hit - nul, stride=A["stride"], ts=ts, hn_ex=hit - nex))
    hn0 = tab[0]["hit_null"][0]
    tstar = next((r["slot"] for r in tab if r["hit_null"][0] < 0.5 * hn0), None)
    per_arm[A["name"]] = dict(table=tab, t_star=tstar, t_star_raw=(None if tstar is None else tab[tstar]["raw_dt"]),
                              t_star_dist=(None if tstar is None else tab[tstar]["dist"]))
    hx0 = tab[0]["hit_null_ex"][0]; tsx = next((r["slot"] for r in tab if r["hit_null_ex"][0] < 0.5 * hx0), None)
    per_arm[A["name"]].update(t_star_tsince=(None if tstar is None else tab[tstar]["t_since"]), t_star_ex=tsx,
                              t_star_ex_raw=(None if tsx is None else tab[tsx]["raw_dt"]), t_star_ex_tsince=(None if tsx is None else tab[tsx]["t_since"]),
                              t_star_ex_dist=(None if tsx is None else tab[tsx]["dist"]))
# 회귀: hit − null ~ 1 + slot + Δf/8 + dist  (모든 팔 · 슬롯)
Xs, ys, cs = [], [], []
for r in rows:
    ok = r["ok"]
    Xs.append(np.stack([np.ones(ok.sum()), np.full(ok.sum(), r["t"]), np.full(ok.sum(), r["df"] / 8), r["dist"][ok]], -1))
    ys.append(r["hn"][ok]); cs.append(traj[ok])
Xs, ys, cs = np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs)
beta = np.linalg.lstsq(Xs, ys, rcond=None)[0]
cix = {t: np.where(cs == t)[0] for t in ut}; bb = []
for _ in range(500):
    s = np.concatenate([cix[t] for t in rng.choice(ut, len(ut))]); bb.append(np.linalg.lstsq(Xs[s], ys[s], rcond=None)[0])
bb = np.array(bb)
reg = {k: [round(float(beta[j]), 4), round(float(np.percentile(bb[:, j], 2.5)), 4), round(float(np.percentile(bb[:, j], 97.5)), 4)]
       for j, k in enumerate(["intercept", "per_slot", "per_8raw_frames", "per_cell"])}
# 추가: 전수 귀무 · 같은 원점 시간으로 같은 회귀 (rng2)
Xs2, ys2, cs2 = [], [], []
for r in rows:
    ok = r["ok"]
    Xs2.append(np.stack([np.ones(ok.sum()), np.full(ok.sum(), r["t"]), np.full(ok.sum(), r["ts"] / 8), r["dist"][ok]], -1))
    ys2.append(r["hn_ex"][ok]); cs2.append(traj[ok])
Xs2, ys2, cs2 = np.concatenate(Xs2), np.concatenate(ys2), np.concatenate(cs2)
beta2 = np.linalg.lstsq(Xs2, ys2, rcond=None)[0]; cix2 = {t: np.where(cs2 == t)[0] for t in ut}; bb2 = []
for _ in range(500):
    s = np.concatenate([cix2[t] for t in rng2.choice(ut, len(ut))]); bb2.append(np.linalg.lstsq(Xs2[s], ys2[s], rcond=None)[0])
bb2 = np.array(bb2)
reg2 = {k: [round(float(beta2[j]), 4), round(float(np.percentile(bb2[:, j], 2.5)), 4), round(float(np.percentile(bb2[:, j], 97.5)), 4)]
        for j, k in enumerate(["intercept", "per_slot", "per_8raw_frames_since_last_obs", "per_cell"])}
out = dict(arms=per_arm, regression=reg, n_clips=n, n_traj=len(ut), regression_ex_tsince=reg2,
           notes="추가 키 (*_ex, t_since, t_star_ex*, regression_ex_tsince) 는 전수 귀무 · 마지막 문맥 튜블릿 중심 원점. 표적 = 표준 양방향 h.")
json.dump(out, default=float, fp=open(OUT / f"p2_v3{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1)
for A in arms:
    pa = per_arm[A["name"]]
    print(f"\n== {A['name']} (stride {A['stride']}, 문맥 {A['n_ctx']} 튜블릿) t*={pa['t_star']} raw_dt*={pa['t_star_raw']} dist*={pa['t_star_dist'] and round(pa['t_star_dist'],2)}")
    print(" t  raw_dt  dist  | hit   null  hit−null [CI]       obj   app   apph  prog   EI")
    for r in pa["table"]:
        print(f"{r['slot']:2d} {r['raw_dt']:6.1f} {r['dist']:5.2f} | {r['hit'][0]:.2f}  {r['null'][0]:.2f}  {r['hit_null'][0]:+.2f} [{r['hit_null'][1]:+.2f},{r['hit_null'][2]:+.2f}]  "
              f"{r['obj'][0]:.2f}  {r['app'][0]:.2f}  {r['apph'][0]:.2f}  {r['prog'][0]:+.2f}  {r['EI'][0]:+.2f}")
print("\n회귀 (hit−null):", json.dumps(reg))
print("회귀 (hit−null_ex, t_since):", json.dumps(reg2))
for A in arms:
    pa = per_arm[A["name"]]
    print(f"{A['name']:7s} hit−null_ex " + " ".join(f"{r['hit_null_ex'][0]:+.2f}" for r in pa["table"]) + f"  t*_ex={pa['t_star_ex']}")
    print(f"{'':7s} apph−null_ex " + " ".join(f"{r['apph_minus_null_ex'][0]:+.2f}" for r in pa["table"]))
    print(f"{'':7s} app−null_ex  " + " ".join(f"{r['app_minus_null_ex'][0]:+.2f}" for r in pa["table"]))
