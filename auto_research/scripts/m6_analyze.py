#!/usr/bin/env python3
"""M6 분석 — 역사 시간 이동이 예측 자리를 끄는가. 입력 v3_m6/.
진행 along_t = ((a_t + 0.5) − L)·u (칸), a_t = argmax 칸 (x, y), L = 문맥 마지막 튜블릿 (프레임 30, 31) 연속 위치, u = 진실 경로 현 방향 (L → 슬롯 15 진실).
대비: arm − base_iso (같은 12 프레임 창 인코딩 방식), 슬롯별, 궤적 군집 bootstrap. 진실 along 도 같이.
템플릿 두 개 (표준 h 진실 칸 / 문맥 마지막 외형) 를 따로 보고한다. 적중 − 귀무 (다른 궤적) 도.
기대 (역사에서 과거 자리를 가져와 뒤처짐): along(hist-8) < along(hist-4) < along(base_iso) < along(hist+4).

2026-09-25 적대 검증 2 차 반영 (키 추가만; 기존 키 truth/appearance 의 값은 같은 rng 순서라 그대로 재현된다):
  hit_null_full   귀무 = 같은 법칙 다른 궤적 **전수 평균** (README 읽기 규칙 2). 기존 hit_null (무작위 1 개) 은 호환용으로 둔다
  objectness      물체다움 = contrast (max − med) ≥ 0.357 (README 읽기 규칙 1) 비율 · 연속 contrast, 팔 × 슬롯
  pairs           팔 짝 차이 (along · hit_null_full · obj357 · contrast) — base 대비 · hist_none 대비 포함 (2×2 를 base_iso 로만 읽지 않게)
  obj_conditioned 두 팔 모두 물체다움 통과 칸만의 along 차 (+ n) — 판독 기본값 배제
  progress_ratio  진행 비율 = Σ along / Σ 진실 along (t4–15, t4–7), base vs hist_none 짝 차 포함
  inframe         이동 역사 창의 화면 안 비율 · 창 전체 in-frame 부분집합 (raw 8–31 전부 in-frame) · 창 in-frame 층 (all_in / partial / mostly_out)
  seam_jump · near_history · fe_regression · monotone   이음매 점프, 과거 물체 칸 착지, 법칙 고정효과 회귀, 슬롯별 단조
  invalid_arms    Ariel (prefix) predictor 에서는 문맥이 블록 0 에서 시작하지 않는 팔 (hist_none · bndM_only) 이 무효다
                  (src/models/rollout_predictor.py: n_ctx_blocks 를 개수로 세고 prefix 마스크가 절대 블록 번호와 비교 → 경계 토큰을 미래로 취급).
                  입력 경로에 'ariel' 이 들어 있거나 M6_INVALID_ARMS="a,b" 를 주면 그 팔을 NaN 으로 둔다.
CLI: python m6_analyze.py [입력 dir]   (OUT_TAG 환경변수 → exp_results/m6/m6_v3{OUT_TAG}.json)
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m6")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/m6"; OUT.mkdir(parents=True, exist_ok=True)
CELL = 18.0
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids); K = meta["K"]
M = np.load(I / "m6.npy")                                                      # (n, arms, K, 8)
INVALID = set(filter(None, os.environ.get("M6_INVALID_ARMS", "").split(",")))
if "ariel" in str(I).lower():
    INVALID |= {"hist_none", "bndM_only"}                                      # prefix 마스크: 문맥이 블록 0 에서 시작해야 한다
for _a in INVALID & set(arms):
    M[:, arms.index(_a)] = np.nan
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); scen = np.array(meta["scenario"])
rng = np.random.default_rng(0); ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
other = np.array([rng.choice(np.where((scen == scen[i]) & (traj != traj[i]))[0]) for i in range(n)])
xy = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
L = xy(30); Tk = np.stack([xy(32 + 2 * t) for t in range(K)], 1)             # (n, K, 2) 연속
u = Tk[:, -1] - L; un = np.linalg.norm(u, axis=-1); u = u / np.maximum(un, 1e-6)[:, None]
OK = np.stack([INF[:, 32 + 2 * t] & INF[:, 33 + 2 * t] for t in range(K)], 1) & (un >= 1)[:, None]


def boot(v, B=500):
    m = np.nanmean(v); bs = [np.nanmean(v[np.concatenate([tix[t] for t in rng.choice(ut, len(ut))])]) for _ in range(B)]
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


out = {}
true_al = np.where(OK, ((np.floor(Tk) + 0.5 - L[:, None]) * u[:, None]).sum(-1), np.nan)
b = arms.index("base_iso")
for tpl, cols in (("truth", [1, 0]), ("appearance", [5, 4])):
    a = M[..., cols]                                                           # (n, arms, K, 2) x, y
    al = np.where(OK[:, None], ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1), np.nan)
    rec = {"true_along": [boot(true_al[:, t]) for t in range(K)]}
    for ai, arm in enumerate(arms):
        rec[arm] = dict(along=[boot(al[:, ai, t]) for t in range(K)],
                        d_vs_base_iso=[boot(al[:, ai, t] - al[:, b, t]) for t in range(K)])
        if tpl == "truth":
            ct = np.floor(Tk)
            hit = (np.abs(a[:, ai] - ct).max(-1) <= 1).astype(float); nul = (np.abs(a[:, ai] - ct[other]).max(-1) <= 1).astype(float)
            okh = OK & ~np.isnan(a[:, ai, :, 0])                                # 무효 팔 (NaN) 은 0 이 아니라 NaN 으로
            rec[arm]["hit_null"] = [boot(np.where(okh[:, t], hit[:, t] - nul[:, t], np.nan)) for t in range(K)]
    out[tpl] = rec

# ======================================================================================================
# 2 차 적대 검증 반영 (2026-09-25) — 키 추가만. 별도 rng 로 뽑아 기존 키의 값은 바뀌지 않는다. B = 2000, 궤적 군집.
R2 = np.random.default_rng(1); B2 = 2000; A = {k: i for i, k in enumerate(arms)}; THR = 0.357; ALL = np.arange(n)
_W = {}


def wmat(key, sub):
    """궤적 군집 bootstrap 가중 행렬 (B2, n). sub 안에 있는 궤적만 다시 뽑는다 (가중 = 뽑힌 횟수)."""
    if key not in _W:
        tt = np.unique(traj[sub]); mem = [sub[traj[sub] == t] for t in tt]
        dr = R2.integers(0, len(tt), (B2, len(tt))); W = np.zeros((B2, n))
        for j, m in enumerate(mem):
            W[:, m] += (dr == j).sum(1)[:, None]
        _W[key] = W
    return _W[key]


def bt(v, key="all", sub=ALL):
    """[평균, 2.5 %, 97.5 %] — nanmean 을 궤적 재표집 가중으로 (자리마다 NaN 무시)."""
    v = np.asarray(v, float); W = wmat(key, sub); ok = ~np.isnan(v)
    if not ok[sub].any():
        return [None, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ np.where(ok, v, 0.0)) / (W @ ok)
    return [round(float(np.nanmean(v[sub])), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


TPL = {"truth": ([1, 0], 2, 3), "appearance": ([5, 4], 6, 7)}            # argmax (x, y) 열, max 열, med 열
ct = np.floor(Tk); same = [np.where((scen == scen[i]) & (traj != traj[i]))[0] for i in range(n)]
AL, HN, OBJ, CON = {}, {}, {}, {}
for tpl, (cxy, cmx, cmd) in TPL.items():
    a = M[..., cxy].astype(np.float64)                                          # (n, arms, K, 2)
    val = OK[:, None] & ~np.isnan(a[..., 0])
    AL[tpl] = np.where(val, ((a + 0.5 - L[:, None, None]) * u[:, None, None]).sum(-1), np.nan)
    hit = (np.abs(a - ct[:, None]).max(-1) <= 1).astype(float)
    nul = np.stack([(np.abs(a[i][None] - ct[same[i]][:, None]).max(-1) <= 1).mean(0) for i in range(n)])   # 전수 평균 귀무
    HN[tpl] = np.where(val, hit - nul, np.nan)
    con = (M[..., cmx] - M[..., cmd]).astype(np.float64)
    CON[tpl] = np.where(val, con, np.nan); OBJ[tpl] = np.where(val, (con >= THR).astype(float), np.nan)

PAIRS = [("base", "hist_none"), ("base", "base_iso"), ("base_iso", "hist_none"), ("hist-8", "base_iso"), ("hist-4", "base_iso"),
         ("hist+4", "base_iso"), ("hist+4", "hist_none"), ("bndM_only", "base"), ("histZ+bndM", "base"), ("bndM_only", "hist_none"),
         ("histZ+bndM", "bndM_only"), ("histZ+bndM", "hist_none")]
out["invalid_arms"] = sorted(INVALID & set(arms))
out["r2_meta"] = dict(B=B2, cluster="trajectory", null="same law, other trajectories, full mean", obj_thr=THR,
                      n=n, n_traj=int(len(ut)), ok_frac_per_slot=[round(float(x), 3) for x in OK.mean(0)])
out["hit_null_full"] = {tpl: {arm: [bt(HN[tpl][:, A[arm], t]) for t in range(K)] for arm in arms} for tpl in TPL}
out["objectness"] = {tpl: {arm: {"obj357": [bt(OBJ[tpl][:, A[arm], t]) for t in range(K)],
                                 "contrast": [bt(CON[tpl][:, A[arm], t]) for t in range(K)]} for arm in arms} for tpl in TPL}
out["pairs"] = {tpl: {f"{p}-{q}": {"along": [bt(AL[tpl][:, A[p], t] - AL[tpl][:, A[q], t]) for t in range(K)],
                                   "hit_null_full": [bt(HN[tpl][:, A[p], t] - HN[tpl][:, A[q], t]) for t in range(K)],
                                   "obj357": [bt(OBJ[tpl][:, A[p], t] - OBJ[tpl][:, A[q], t]) for t in range(K)],
                                   "contrast": [bt(CON[tpl][:, A[p], t] - CON[tpl][:, A[q], t]) for t in range(K)]}
                      for p, q in PAIRS} for tpl in TPL}
oc = {}
for tpl in TPL:
    oc[tpl] = {}
    for p, q in PAIRS:
        both = (OBJ[tpl][:, A[p]] == 1) & (OBJ[tpl][:, A[q]] == 1)
        d = np.where(both, AL[tpl][:, A[p]] - AL[tpl][:, A[q]], np.nan)
        oc[tpl][f"{p}-{q}"] = {"along": [bt(d[:, t]) for t in range(K)], "n": [int(both[:, t].sum()) for t in range(K)]}
out["obj_conditioned"] = oc


def _sums(tpl, arm_list, sl):
    """진행 비율용 clip 별 Σ along (팔마다) 과 Σ 진실 along — 모든 팔·진실이 유효한 칸만 (짝 비교에서 같은 칸)."""
    al = [AL[tpl][:, A[k], sl] for k in arm_list]; tr = true_al[:, sl]
    ok = ~np.isnan(tr)
    for x in al:
        ok &= ~np.isnan(x)
    return [np.where(ok, x, 0.0).sum(1) for x in al], np.where(ok, tr, 0.0).sum(1), ok


def _rb(num, den):
    W = wmat("all", ALL)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ num) / (W @ den)
    if den.sum() == 0:
        return [None, None, None]
    return [round(float(num.sum() / den.sum()), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


pr = {}
for tpl in TPL:
    pr[tpl] = {}
    for nm, sl in (("t4_15", slice(4, K)), ("t4_7", slice(4, 8))):
        rec = {}
        for arm in arms:
            (sa,), st, _ = _sums(tpl, [arm], sl); rec[arm] = _rb(sa, st)
        for p, q in (("base", "hist_none"), ("base", "base_iso"), ("hist-8", "base_iso"), ("hist+4", "base_iso")):
            (sp, sq), st, _ = _sums(tpl, [p, q], sl); rec[f"d_{p}-{q}"] = _rb(sp - sq, st)
        # 두 팔 모두 물체다움 통과 칸만 (base vs hist_none)
        both = (OBJ[tpl][:, A["base"], sl] == 1) & (OBJ[tpl][:, A["hist_none"], sl] == 1)
        sp = np.where(both, AL[tpl][:, A["base"], sl], 0.0).sum(1); sq = np.where(both, AL[tpl][:, A["hist_none"], sl], 0.0).sum(1)
        st = np.where(both, true_al[:, sl], 0.0).sum(1)
        rec["both_obj_base"] = _rb(sp, st); rec["both_obj_hist_none"] = _rb(sq, st); rec["both_obj_d_base-hist_none"] = _rb(sp - sq, st)
        rec["both_obj_n_cells"] = int(both.sum())
        pr[tpl][nm] = rec
out["progress_ratio"] = pr

# 이동 역사 창: 화면 밖 비율 · 부분집합
WIN = {"hist-8": 8, "hist-4": 12, "base_iso": 16, "hist+4": 20}
ifr = {"window": {k: {"raw": [s, s + 12], "all_inframe_frac": round(float(INF[:, s:s + 12].all(1).mean()), 3),
                      "mean_inframe_frac": round(float(INF[:, s:s + 12].mean()), 3), "gap_frames_to_boundary": 28 - (s + 12)}
                  for k, s in WIN.items()}}
FULL = np.where(INF[:, 8:32].all(1))[0]
sc = lambda sub: {str(k): int(v) for k, v in zip(*np.unique(scen[sub], return_counts=True))}
ifr["all_in_8_32"] = {"n": int(len(FULL)), "n_traj": int(len(np.unique(traj[FULL]))), "scen": sc(FULL)}
mF = np.isin(ALL, FULL)
for tpl in TPL:
    ifr["all_in_8_32"][tpl] = {}
    for arm in ("hist-8", "hist-4", "hist+4"):
        d = AL[tpl][:, A[arm]] - AL[tpl][:, A["base_iso"]]; dob = OBJ[tpl][:, A[arm]] - OBJ[tpl][:, A["base_iso"]]
        both = (OBJ[tpl][:, A[arm]] == 1) & (OBJ[tpl][:, A["base_iso"]] == 1)
        ifr["all_in_8_32"][tpl][arm] = {"d_vs_base_iso": [bt(np.where(mF, d[:, t], np.nan), "full", FULL) for t in range(K)],
                                        "d_vs_base_iso_both_obj": [bt(np.where(mF & both[:, t], d[:, t], np.nan), "full", FULL) for t in range(K)],
                                        "both_obj_n": [int((mF & both[:, t]).sum()) for t in range(K)],
                                        "obj357_d_vs_base_iso": [bt(np.where(mF, dob[:, t], np.nan), "full", FULL) for t in range(K)]}
ifr["strata"] = {}
for arm in ("hist-8", "hist-4"):
    s = WIN[arm]; fr = INF[:, s:s + 12].mean(1); ifr["strata"][arm] = {}
    for g, sub in (("all_in", np.where(fr == 1)[0]), ("partial_0.5_1", np.where((fr >= 0.5) & (fr < 1))[0]), ("mostly_out_lt0.5", np.where(fr < 0.5)[0])):
        if len(sub) == 0:
            continue
        key = f"strat:{arm}:{g}"; msk = np.isin(ALL, sub); rec = {"n": int(len(sub)), "n_traj": int(len(np.unique(traj[sub]))), "scen": sc(sub)}
        for tpl in TPL:
            d = AL[tpl][:, A[arm]] - AL[tpl][:, A["base_iso"]]
            both = (OBJ[tpl][:, A[arm]] == 1) & (OBJ[tpl][:, A["base_iso"]] == 1)
            rec[tpl] = {"d_vs_base_iso": [bt(np.where(msk, d[:, t], np.nan), key, sub) for t in range(K)],
                        "d_vs_base_iso_both_obj": [bt(np.where(msk & both[:, t], d[:, t], np.nan), key, sub) for t in range(K)],
                        "both_obj_n": [int((msk & both[:, t]).sum()) for t in range(K)],
                        "obj357_d_vs_base_iso": [bt(np.where(msk, OBJ[tpl][:, A[arm], t] - OBJ[tpl][:, A["base_iso"], t], np.nan), key, sub) for t in range(K)]}
        ifr["strata"][arm][g] = rec
out["inframe"] = ifr

# 이음매 점프: 역사 창 마지막 튜블릿 (s+10, s+11) → 경계 첫 튜블릿 (28, 29), u 방향 (칸)
out["seam_jump"] = {k: bt(((xy(28) - xy(s + 10)) * u).sum(-1)) for k, s in WIN.items()}
# 과거 물체 칸 착지: 팔 argmax (진실 칸 템플릿) 가 자기 역사 창의 물체 칸 (±1) 에 떨어지는 비율 − base_iso 가 같은 칸에 떨어지는 비율
at = M[..., [1, 0]].astype(np.float64); nh = {}
for arm, s in WIN.items():
    hc = np.stack([np.floor(xy(s + 2 * j)) for j in range(6)], 1)              # (n, 6, 2) — 화면 밖 자리는 격자 밖이라 적중 불가
    land = lambda ai, t: (np.abs(at[:, ai, t][:, None] - hc).max(-1) <= 1).any(1).astype(float)
    nh[arm] = {"arm": [bt(np.where(OK[:, t], land(A[arm], t), np.nan)) for t in range(K)],
               "base_iso_same_cells": [bt(np.where(OK[:, t], land(A["base_iso"], t), np.nan)) for t in range(K)],
               "diff": [bt(np.where(OK[:, t], land(A[arm], t) - land(A["base_iso"], t), np.nan)) for t in range(K)]}
out["near_history"] = nh
# 법칙 고정효과 회귀: d(arm − base_iso) ~ 역사 물체 변위 (칸, u 방향, 6 튜블릿 평균) + 창 화면 밖 비율 + 법칙 FE
SS = sorted(set(scen)); FE = np.stack([(scen == s_).astype(float) for s_ in SS], 1); Wall = wmat("all", ALL); fe = {}
for arm in ("hist-8", "hist-4"):
    s = WIN[arm]
    disp = np.stack([((xy(s + 2 * j) - xy(16 + 2 * j)) * u).sum(-1) for j in range(6)], 1).mean(1)
    outf = 1 - INF[:, s:s + 12].mean(1); Xd = np.column_stack([disp, outf, FE])
    fe[arm] = {"corr_disp_outframe": round(float(np.corrcoef(disp, outf)[0, 1]), 3)}
    for t in (4, 6, 8, 10, 12):
        y = AL["truth"][:, A[arm], t] - AL["truth"][:, A["base_iso"], t]; okr = ~np.isnan(y)

        def wls(w):
            sw = np.sqrt(w[okr])[:, None]
            return np.linalg.lstsq(Xd[okr] * sw, y[okr] * sw[:, 0], rcond=None)[0][:2]
        b0 = wls(np.ones(n)); bs = np.array([wls(Wall[j]) for j in range(B2)])
        fe[arm][f"t{t}"] = {"b_disp_per_cell": [round(float(b0[0]), 3)] + np.percentile(bs[:, 0], [2.5, 97.5]).round(3).tolist(),
                            "b_outframe_frac": [round(float(b0[1]), 3)] + np.percentile(bs[:, 1], [2.5, 97.5]).round(3).tolist()}
out["fe_regression"] = fe
out["monotone"] = {}                                                        # 슬롯별 hist-8 < hist-4 < base_iso < hist+4 (평균) 이면 M
for tpl in TPL:
    mm_ = np.array([[np.nanmean(AL[tpl][:, A[k], t]) for k in ("hist-8", "hist-4", "base_iso", "hist+4")] for t in range(K)])
    out["monotone"][tpl] = "".join("M" if (np.diff(r_) > 0).all() else "." for r_ in mm_)
json.dump(out, open(OUT / f"m6_v3{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
for tpl in ("truth", "appearance"):
    r = out[tpl]; print(f"\n== 템플릿 {tpl}: along (칸), 슬롯 0 2 4 6 8 10 12 14")
    print(f"  {'진실':10s} " + " ".join(f"{r['true_along'][t][0]:5.2f}" for t in range(0, K, 2)))
    for arm in arms:
        print(f"  {arm:10s} " + " ".join(f"{r[arm]['along'][t][0]:5.2f}" for t in range(0, K, 2))
              + "  | − base_iso " + " ".join(f"{r[arm]['d_vs_base_iso'][t][0]:+.2f}" for t in range(0, K, 2)))
print("\n적중−귀무 (truth 템플릿) 슬롯 0 2 4 6 8 10")
for arm in arms:
    print(f"  {arm:10s} " + " ".join(f"{out['truth'][arm]['hit_null'][t][0]:+.2f}" for t in range(0, 12, 2)))

# ---- 2 차 반영 요약 출력
fm = lambda v: "   —  " if v[0] is None else f"{v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]"
TS = (2, 4, 6, 8, 10, 12)
print(f"\n무효 팔: {out['invalid_arms']} · 단조 (truth / appearance): {out['monotone']['truth']} / {out['monotone']['appearance']}")
for tpl in TPL:
    print(f"\n== [{tpl}] 물체다움 ≥ {THR} 비율, 슬롯 {TS}")
    for arm in arms:
        print(f"  {arm:10s} " + " ".join("  —  " if out['objectness'][tpl][arm]['obj357'][t][0] is None else f"{out['objectness'][tpl][arm]['obj357'][t][0]:.3f}" for t in TS)
              + "  | 적중−귀무(전수) " + " ".join("  —  " if out['hit_null_full'][tpl][arm][t][0] is None else f"{out['hit_null_full'][tpl][arm][t][0]:+.3f}" for t in TS))
    print("  짝 차이 (along · obj357 · 두 팔 물체다움 통과 along)")
    for pq in out["pairs"][tpl]:
        pp = out["pairs"][tpl][pq]; oo = out["obj_conditioned"][tpl][pq]
        print(f"   {pq:22s} along " + " ".join(fm(pp['along'][t]) for t in (4, 6, 8)) + " | obj " + " ".join(fm(pp['obj357'][t]) for t in (4, 6, 8)))
        print(f"   {'':22s} cond  " + " ".join(fm(oo['along'][t]) + f"(n{oo['n'][t]})" for t in (4, 6, 8)))
    for nm in ("t4_15", "t4_7"):
        r = out["progress_ratio"][tpl][nm]
        print(f"  진행 비율 {nm}: base {fm(r['base'])} hist_none {fm(r['hist_none'])} base_iso {fm(r['base_iso'])} · d(base−hist_none) {fm(r['d_base-hist_none'])}"
              f" · 물체다움 통과만 base {fm(r['both_obj_base'])} hist_none {fm(r['both_obj_hist_none'])} d {fm(r['both_obj_d_base-hist_none'])} (칸 {r['both_obj_n_cells']})")
print("\n역사 창 화면 안:", json.dumps(out["inframe"]["window"], ensure_ascii=False))
fi = out["inframe"]["all_in_8_32"]; print(f"창 전체 in-frame (raw 8–31) n={fi['n']} 궤적 {fi['n_traj']} {fi['scen']}")
for tpl in TPL:
    for arm in ("hist-8", "hist-4", "hist+4"):
        print(f"  [{tpl}] {arm:7s} d_vs_base_iso " + " ".join(fm(fi[tpl][arm]['d_vs_base_iso'][t]) for t in (4, 6, 8, 10)))
for arm, gg in out["inframe"]["strata"].items():
    for g, r in gg.items():
        print(f"  층 {arm} {g:17s} n={r['n']} 궤적 {r['n_traj']}: truth " + " ".join(fm(r['truth']['d_vs_base_iso'][t]) for t in (4, 6, 8))
              + " | obj-cond " + " ".join(fm(r['truth']['d_vs_base_iso_both_obj'][t]) + f"(n{r['truth']['both_obj_n'][t]})" for t in (6, 8)))
print("이음매 점프:", {k: v[0] for k, v in out["seam_jump"].items()})
print("과거 칸 착지 diff t4/6/8/10:", {k: [v['diff'][t][0] for t in (4, 6, 8, 10)] for k, v in out["near_history"].items()})
print("FE 회귀:", json.dumps(out["fe_regression"]))
