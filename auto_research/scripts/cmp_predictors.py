#!/usr/bin/env python3
"""학습된 predictor 비교 (z_training · Ariel, 새 학습 없음) — 진단 시그니처를 한 표로.
입력: exp_results/{p2,p3,m13}/*{태그}.json  (태그 '' = 릴리즈, _pv1, _ariel, _v11ft).  먼저 각 태그로 분석 스크립트를 돌린다 (run_battery_analysis.sh).
시그니처 (초판 키, 값은 그대로 둔다):
  P2  hit−null 슬롯 곡선 (s1_C16, s2_C16, s4_C8) · 절벽 슬롯 (p2b, 분위 합침 평균)
  P3  j3–7 전진 기울기 (ar_z · ar_hcA · ar_pA) · 새 칸 착지 j4–7 평균
  M1  full 의 hit−null t0..t14 · hist_swap − full · pred_last2 − full (hit−null, t4–t8 평균)
  M3  p_obj 속도 R² (t0 · t8 · t14) · 충분성 p_full vs z_cv (t10)
  ⚠️ 초판 키의 결함 (적대 검증 2 차 VERIFY_0925_R2, 2026-09-25) — 값은 그대로 두고 아래 새 키로 대체한다:
    - "P2 * hit-null" 은 무작위 짝 귀무 1 개 (B12 위반). 대체: "P2 * hit-null_ex ci" (같은 법칙 · 다른 궤적 전수 귀무).
    - "P2 * cliff slot" 은 창 안에서 안 떨어진 분위를 빼고 평균한다 (중도절단 무시 → 학습 predictor 과소평가).
      대체: "P2 * cliff terciles (p2b, censored)" · "P2 * cliff slot lower bound (p2b, censored=window end)".
    - "M1 pred_last2-full t4-8" 의 Ariel 값은 무효다. 문맥이 블록 0 부터가 아니면 src/models/rollout_predictor.py 의 dense 폴백
      build_prefix_mask 가 절대 블록 번호로 문맥/미래를 갈라, 문맥 토큰을 미래로 분류한다. → "M1 pred_last2-full valid" 키.

추가 키 (적대 검증 2 차 반영, 2026-09-25) — 초판 키는 지우거나 바꾸지 않는다:
  A. JSON 에서 옮기는 키 (predictor 별 rec, 네 창 s1_C16 · s2_C16 · s2_C8 · s4_C8):
     P2 {창} slots · hit-null_ex ci · app-null_ex ci · app null_ex rate · app ci · apph-null_ex ci (ceiling) · prog ci · obj ci
     P2 {창} cliff terciles (p2b, censored) · cliff slot lower bound (p2b, censored=window end)
     P3 {팔} slope ci · slope_app ci · hit-null_exh j7 ci
     M1 pred_last2-full valid
     CI 는 각 분석 스크립트의 궤적 군집 bootstrap 이다 (p2_analyze 새 키 B=300, p3_analyze B=500/2000).
  B. 원자료 절 "_v2_raw" (vll6 캐시 v3_p2{태그} · v3_p3{태그}; 검증자 _verify_scratch/TRAINED 의 v_p2 · v_stall · v_lead · v_p3 를 옮김):
     판독 두 가지. (i) 진실 칸 템플릿 argmax (loc_tru; 템플릿 = 진실 칸의 표준 양방향 h_t 토큰 → 진실 위치를 쓴다)
                   (ii) 외형 템플릿 argmax (loc_app; 문맥 마지막 튜블릿의 물체 칸 h 토큰. 진실 미래 **위치** 는 쓰지 않지만
                        토큰 자체는 창 전체 (미래 포함) 를 본 **양방향** target encoder 에서 나온다).
     표적 (target) 은 전부 표준 양방향 h 다. 인과 표적 판은 없다.
     hit = ‖argmax − 진실 칸‖∞ ≤ 1.  귀무 null_ex = 같은 법칙 · 다른 궤적 clip 전부의 진실 칸에 대한 적중률 (전수 평균).
     slots[태그][t] : hit_null · app_null (각각 − null_ex) · app · app_nullrate · apph_null (진짜 미래 h 에 같은 외형 템플릿 = encoder 천장)
                      obj (contrast ≥ 0.357, 릴리즈 기준 문턱) · obj_given_farmiss (진실 칸에서 ≥ 3 칸 빗나간 clip 의 obj, n) · obj_given_hit
                      maxcos · medcos · paired_hn_minus_release · paired_an_minus_release (같은 clip 짝 차이)
     oracles[copy|cv|stop3][t] : 위치만 내는 가짜 예측기의 hit − null_ex. cv = 문맥 끝 **진실** 속도로 등속 외삽 (진실 위치를 쓴다),
                      copy = 마지막 문맥 칸, stop3 = 진실을 슬롯 2 까지 따라간 뒤 멈춤.
     cliff_terciles[태그] : 정수 규칙 절벽 (hit − null_ex 곡선이 슬롯 0 값의 절반 아래로 처음 떨어지는 슬롯), 문맥 끝 속도 3 분위별.
                      창 안에서 안 떨어지면 cliff=None, censored=True, lower_bound=창 끝 슬롯.
     path_slope[태그|app|tru][band] : 경로 방향 (문맥 끝 → 마지막 미래 진실) 으로 투영한 argmax 변위의 슬롯 기울기 (칸/튜블릿), arc 제외.
                      truth = 같은 투영의 진실 기울기.  along_mean_noarc = 슬롯별 평균 투영 변위 (칸).
     lead[태그][t] : 같은 슬롯 안에서 clip 사이 속도 차이를 이용한 회귀  예측 투영 변위 = a + b · 진실 변위  (arc 제외, 진실 변위 > 0.3 칸).
                      b ≈ 1, a ≈ 0 이면 진실을 따라간다. b ≈ 3 이면 "튜블릿당 3 배" 가설.
     per_law_far[태그][법칙] = [app − null_ex, hit − null_ex] (먼 슬롯: s1_C16 t10 · s2_* t6 · s4_C8 t3)
     p3[태그|팔|tru|app] : j3–7 기울기 · 새 칸 착지 j4–7 · j7 적중 − 전수 귀무 · 릴리즈와의 짝 차이 (궤적 군집 bootstrap B=2000)
     CI = 궤적 군집 bootstrap (P2 84 궤적 B=1000, seed 20260925; P3 56 궤적 B=2000, seed 20260926).

사용:  python cmp_predictors.py [캐시 루트]     (기본 /data2/local_datasets/world/world_analysis/cache/auto_research — vll6 에만 있다)
       env OUT_TAG (출력 접미사, 기본 ''),  CMP_SKIP_RAW=1 (원자료 절을 건너뛴다; 그때 "_v2_raw" 키는 빠진다)
출력:  exp_results/cmp/cmp_predictors{OUT_TAG}.json
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2] / "auto_research/exp_results"
ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research")
OUT_TAG = os.environ.get("OUT_TAG", "")
TAGS = [("release", ""), ("predictor_v1_postft", "_pv1"), ("ariel_bc_ep43", "_ariel"), ("v11_postft", "_v11ft")]
WIN4 = ("s1_C16", "s2_C16", "s2_C8", "s4_C8")
out = {}
for name, tg in TAGS:
    rec = {}
    f = R / f"p2/p2_v3{tg}.json"
    if f.exists():
        d = json.load(open(f))
        for arm in ("s1_C16", "s2_C16", "s4_C8"):
            if arm in d["arms"]:
                rec[f"P2 {arm} hit-null"] = [r["hit_null"][0] for r in d["arms"][arm]["table"]]
                rec[f"P2 {arm} obj"] = [r["obj"][0] for r in d["arms"][arm]["table"]]
    f = R / f"p2/p2b_units{tg}.json"
    if f.exists():
        d = json.load(open(f))["cliffs"]
        for arm in ("s1_C16", "s2_C16", "s4_C8"):
            v = [d[f"{arm}|q{q}"]["cliff"]["slot"] for q in range(3) if d.get(f"{arm}|q{q}", {}).get("cliff")]
            rec[f"P2 {arm} cliff slot"] = round(float(np.mean(v)), 2) if v else None
    f = R / f"p3/p3_v3{tg}.json"
    if f.exists():
        d = json.load(open(f))
        for arm in ("ar_z", "ar_hcA", "ar_pA"):
            if arm in d:
                rec[f"P3 {arm} slope"] = d[arm]["slope"][0]
                rec[f"P3 {arm} land_new j4-7"] = round(float(np.mean([x[0] for x in d[arm]["land_new"][4:]])), 3)
        if "ar_z" in d: rec["P3 true slope"] = d["ar_z"]["true_slope"][0]
    f = R / f"m13/m13_v3{tg}.json"
    if f.exists():
        d = json.load(open(f)); M1 = d["M1"]
        rec["M1 full hit-null"] = [M1["full"][t]["hit_null"][0] for t in range(0, 16, 2)]
        rec["M1 full obj"] = [M1["full"][t]["obj"][0] for t in range(0, 16, 2)]
        for arm in ("hist_swap", "pred_last2"):
            rec[f"M1 {arm}-full t4-8"] = round(float(np.mean([M1[arm][t]["hit_null"][0] - M1["full"][t]["hit_null"][0] for t in range(4, 9)])), 3)
        s = d["M3"]["slots"]
        rec["M3 p_obj R2 vx t0/t8/t14"] = [s[str(t)]["p_obj"][0] if str(t) in s else s[t]["p_obj"][0] for t in (0, 8, 14)]
        if "sufficiency" in d:
            rec["suff t10 p / z_cv / null"] = [d["sufficiency"]["p_full"][10][0], d["sufficiency"]["z_cv"][10][0], d["sufficiency"]["null_p"][10][0]]
    out[name] = rec

# ---------------------------------------------------------------- A. JSON 에서 옮기는 새 키 (적대 검증 2 차)
for name, tg in TAGS:
    rec = out[name]
    f = R / f"p2/p2_v3{tg}.json"
    if f.exists():
        d = json.load(open(f))
        for arm in WIN4:
            if arm not in d["arms"]: continue
            tab = d["arms"][arm]["table"]
            rec[f"P2 {arm} slots"] = [r["slot"] for r in tab]
            for key, src in (("hit-null_ex ci", "hit_null_ex"), ("app-null_ex ci", "app_minus_null_ex"), ("app null_ex rate", "app_null_ex"),
                             ("app ci", "app"), ("apph-null_ex ci (ceiling)", "apph_minus_null_ex"), ("prog ci", "prog"), ("obj ci", "obj")):
                if src in tab[0]: rec[f"P2 {arm} {key}"] = [r[src] for r in tab]
    f = R / f"p2/p2b_units{tg}.json"
    if f.exists():
        d = json.load(open(f))["cliffs"]
        for arm in WIN4:
            qs = [d.get(f"{arm}|q{q}") for q in range(3)]
            if not all(qs): continue
            n_fut = len(out[name].get(f"P2 {arm} slots", [])) or None
            last = (n_fut - 1) if n_fut else None
            terc = []
            for r in qs:
                c = r.get("cliff")
                terc.append(dict(speed=r.get("speed"), slot=None if c is None else c["slot"], ci_slot=(r.get("ci") or {}).get("slot"),
                                 frac_boot_with_cliff=r.get("frac_boot_with_cliff"), censored=c is None,
                                 lower_bound=None if c is not None else last))
            rec[f"P2 {arm} cliff terciles (p2b, censored)"] = terc
            v = [t["slot"] if not t["censored"] else t["lower_bound"] for t in terc]
            rec[f"P2 {arm} cliff slot lower bound (p2b, censored=window end)"] = (
                dict(mean=round(float(np.mean(v)), 2), n_censored=sum(t["censored"] for t in terc)) if None not in v else None)
    f = R / f"p3/p3_v3{tg}.json"
    if f.exists():
        d = json.load(open(f))
        for arm in ("ar_z", "ar_hcA", "ar_pA"):
            if arm not in d: continue
            rec[f"P3 {arm} slope ci"] = d[arm]["slope"]
            if "slope_app" in d[arm]: rec[f"P3 {arm} slope_app ci"] = d[arm]["slope_app"]
            if "hit_null_exh" in d[arm]: rec[f"P3 {arm} hit-null_exh j7 ci"] = d[arm]["hit_null_exh"][7]
    if f"M1 pred_last2-full t4-8" in rec:
        rec["M1 pred_last2-full valid"] = (tg != "_ariel")
        if tg == "_ariel":
            rec["M1 pred_last2-full note"] = ("무효: 문맥이 블록 0 부터가 아니라 dense 폴백 build_prefix_mask 가 문맥 토큰을 미래로 분류 "
                                              "(src/models/rollout_predictor.py). pred_last*/enc_last* 전부 해당, 마스크 수정 뒤 재추출 필요")

# ---------------------------------------------------------------- B. 원자료 절 (vll6)
RAW_OK = os.environ.get("CMP_SKIP_RAW", "") != "1" and (CACHE / "v3_p2" / "meta.json").exists()


def raw_section():
    CELL, THR = 18.0, 0.357
    tg4 = [tg for _, tg in TAGS]
    nm = {tg: n for n, tg in TAGS}
    metas = {t: json.load(open(CACHE / f"v3_p2{t}" / "meta.json")) for t in tg4}
    ids = metas[""]["video_ids"]
    for t in tg4:
        assert metas[t]["video_ids"] == ids, f"video_ids differ for {t}"
    arms = metas[""]["arms"]; n = len(ids)
    idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    fl = lambda s: np.array([float(v) for v in s.split()])
    X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids])
    INF = np.stack([fl(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
    scen = np.array(metas[""]["scenario"]); traj = np.array(["|".join(map(str, t)) for t in metas[""]["traj"]])
    ut = np.unique(traj); tix = {u: np.where(traj == u)[0] for u in ut}
    rng = np.random.default_rng(20260925)
    BI = [np.concatenate([tix[u] for u in rng.choice(ut, len(ut))]) for _ in range(1000)]
    BI_lead = BI[:500]
    SAME = {L: np.where(scen == L)[0] for L in np.unique(scen)}
    keep = scen != "arc"
    D = {t: {k: np.load(CACHE / f"v3_p2{t}" / f"{k}.npy") for k in ("loc_tru", "loc_app", "loc_apph")} for t in tg4}
    res = dict(n_clips=n, n_traj=int(len(ut)), laws={L: int(len(v)) for L, v in SAME.items()},
               sanity=dict(apph_identical_across_tags=bool(all(np.array_equal(np.nan_to_num(D[t]["loc_apph"][..., :2]), np.nan_to_num(D[""]["loc_apph"][..., :2])) for t in tg4))),
               defs="docstring 절 B 참고. 표적 = 표준 양방향 h. app 템플릿은 진실 미래 위치를 쓰지 않지만 양방향 target 토큰이다.")

    def ci(v, bi=BI):
        v = np.asarray(v, float); bs = np.array([np.nanmean(v[b]) for b in bi])
        return [round(float(np.nanmean(v)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]

    def null_ex(am, cti, okt):
        o = np.full(n, np.nan)
        for L, ix in SAME.items():
            M = np.abs(am[ix][:, None] - cti[ix][None]).max(-1) <= 1
            V = (traj[ix][:, None] != traj[ix][None]) & okt[ix][None]
            o[ix] = (M & V).sum(1) / np.maximum(V.sum(1), 1)
        return o

    SLOTS = {"s1_C16": list(range(16)), "s2_C16": list(range(8)), "s2_C8": list(range(8)), "s4_C8": list(range(4))}
    BANDS = {"s1_C16": [(0, 3), (3, 8), (8, 12), (12, 16)], "s2_C16": [(0, 3), (3, 8)], "s2_C8": [(0, 3), (3, 8)], "s4_C8": [(0, 4)]}
    LEAD = {"s1_C16": [2, 4, 6, 8, 10], "s2_C16": [2, 4, 6], "s2_C8": [2, 4, 6], "s4_C8": [1, 2, 3]}
    FAR = {"s1_C16": 10, "s2_C16": 6, "s2_C8": 6, "s4_C8": 3}
    wins = {}
    for ai, A in enumerate(arms):
        if A["name"] not in SLOTS: continue
        fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
        fa, fb = fr[2 * nc - 2], fr[2 * nc - 1]; fa2, fb2 = fr[2 * nc - 4], fr[2 * nc - 3]
        cl = np.stack([(X[:, fa] + X[:, fb]) / 2, (Y[:, fa] + Y[:, fb]) / 2], -1) / CELL
        cl2 = np.stack([(X[:, fa2] + X[:, fb2]) / 2, (Y[:, fa2] + Y[:, fb2]) / 2], -1) / CELL
        okl = INF[:, fa] & INF[:, fb]
        vel = (cl - cl2) / ((fb + fa) / 2 - (fb2 + fa2) / 2)                       # 칸 / raw 프레임 (문맥 끝 진실 속도)
        S = []
        for t in range(nf):
            g0, g1 = fr[2 * (nc + t)], fr[2 * (nc + t) + 1]
            ct = np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL
            S.append(dict(ct=ct, cti=np.clip(np.floor(ct), 0, 15), okt=INF[:, g0] & INF[:, g1], ts=(g0 + g1) / 2 - (fa + fb) / 2))
        W = dict(n_ctx_tubelets=nc, n_fut=nf, stride=A.get("stride"))
        per = {t: {} for t in tg4}; orc = {}
        for s, Sl in enumerate(S):
            ok = okl & Sl["okt"] & ~np.isnan(D[""]["loc_tru"][:, ai, s, 0])
            for on, am in (("copy", np.clip(np.floor(cl), 0, 15)), ("cv", np.clip(np.floor(cl + vel * Sl["ts"]), 0, 15)), ("stop3", S[min(s, 2)]["cti"])):
                h = (np.abs(am - Sl["cti"]).max(-1) <= 1).astype(float); nu = null_ex(am, Sl["cti"], Sl["okt"])
                h[~ok] = np.nan; nu[~ok] = np.nan
                orc.setdefault(on, {})[s] = dict(hit_null=ci(h - nu), hit=round(float(np.nanmean(h)), 3), null=round(float(np.nanmean(nu)), 3))
            for t in tg4:
                Lt = D[t]
                am = Lt["loc_tru"][:, ai, s][:, [1, 0]]; ap = Lt["loc_app"][:, ai, s][:, [1, 0]]
                hit = (np.abs(am - Sl["cti"]).max(-1) <= 1).astype(float); nu = null_ex(am, Sl["cti"], Sl["okt"])
                app = (np.abs(ap - Sl["cti"]).max(-1) <= 1).astype(float); nua = null_ex(ap, Sl["cti"], Sl["okt"])
                obj = ((Lt["loc_tru"][:, ai, s, 2] - Lt["loc_tru"][:, ai, s, 3]) >= THR).astype(float)
                for v in (hit, nu, app, nua, obj): v[~ok] = np.nan
                per[t][s] = dict(hn=hit - nu, an=app - nua, app=app, nua=nua, obj=obj, dtr=np.where(ok, np.abs(am - Sl["cti"]).max(-1), np.nan),
                                 mx=np.where(ok, Lt["loc_tru"][:, ai, s, 2], np.nan), md=np.where(ok, Lt["loc_tru"][:, ai, s, 3], np.nan),
                                 am=am, ap=ap, ok=ok)
            # encoder 천장: 진짜 미래 h 에 같은 외형 템플릿 (predictor 와 무관 → 릴리즈 캐시 하나로)
            ah = D[""]["loc_apph"][:, ai, s][:, [1, 0]]
            hh = (np.abs(ah - Sl["cti"]).max(-1) <= 1).astype(float); nh = null_ex(ah, Sl["cti"], Sl["okt"]); hh[~ok] = np.nan; nh[~ok] = np.nan
            orc.setdefault("apph_ceiling", {})[s] = dict(hit_null=ci(hh - nh), hit=round(float(np.nanmean(hh)), 3), null=round(float(np.nanmean(nh)), 3))
        W["oracles"] = orc
        sl = {}
        for t in tg4:
            rt = {}
            for s in SLOTS[A["name"]]:
                P = per[t][s]
                r = dict(hit_null=ci(P["hn"]), app_null=ci(P["an"]), app=ci(P["app"]), app_nullrate=ci(P["nua"]), obj=ci(P["obj"]),
                         maxcos=round(float(np.nanmean(P["mx"])), 3), medcos=round(float(np.nanmean(P["md"])), 3))
                fm = P["dtr"] >= 3; hh = P["dtr"] <= 1
                r["obj_given_farmiss"] = [round(float(np.nanmean(np.where(fm, P["obj"], np.nan))), 3) if fm.any() else None, int(np.nansum(fm))]
                r["obj_given_hit"] = [round(float(np.nanmean(np.where(hh, P["obj"], np.nan))), 3) if hh.any() else None, int(np.nansum(hh))]
                if t:
                    r["paired_hn_minus_release"] = ci(P["hn"] - per[""][s]["hn"])
                    r["paired_an_minus_release"] = ci(P["an"] - per[""][s]["an"])
                rt[s] = r
            sl[nm[t]] = rt
        W["slots"] = sl
        # 절벽 — 정수 규칙, 문맥 끝 속도 3 분위, 중도절단 표시
        spd = np.linalg.norm(vel, axis=-1); q = np.nanpercentile(spd[okl], [33.33, 66.67]); terc = np.digitize(spd, q)
        cc = {}
        for t in tg4:
            lst = []
            for k in range(3):
                m = terc == k
                curve = [float(np.nanmean(np.where(m, per[t][s]["hn"], np.nan))) for s in range(nf)]
                c0 = curve[0]; cs = next((s for s, v in enumerate(curve) if v < 0.5 * c0), None)
                lst.append(dict(cliff=cs, censored=cs is None, lower_bound=(nf - 1) if cs is None else None,
                                speed_cells_per_frame=round(float(np.nanmean(spd[m & okl])), 4), curve=[round(v, 3) for v in curve]))
            cc[nm[t]] = lst
        W["cliff_terciles"] = cc
        # 경로 기울기 — 문맥 끝 → 마지막 미래 진실 방향 투영, arc 제외
        CT = np.stack([Sl["ct"] for Sl in S], 1)
        u = CT[:, -1] - cl; u = u / np.maximum(np.linalg.norm(u, axis=-1), 1e-6)[:, None]
        true_al = ((CT - cl[:, None]) * u[:, None]).sum(-1)

        def bslope(al, lo, hi):
            tt = np.arange(lo, hi) - np.arange(lo, hi).mean()
            v = np.where(keep, (al[:, lo:hi] * tt).sum(1) / (tt ** 2).sum(), np.nan)
            return ci(v)
        ps = {"truth": {f"t{lo}-{hi - 1}": bslope(true_al, lo, hi) for lo, hi in BANDS[A["name"]]},
              "truth_along_mean_noarc": [round(float(np.nanmean(true_al[keep, s])), 2) for s in range(nf)]}
        for t in tg4:
            for tpl, key in (("app", "ap"), ("tru", "am")):
                Lc = np.stack([per[t][s][key] for s in range(nf)], 1)
                al = ((Lc + 0.5 - cl[:, None]) * u[:, None]).sum(-1)
                ps[f"{nm[t]}|{tpl}"] = {f"t{lo}-{hi - 1}": bslope(al, lo, hi) for lo, hi in BANDS[A["name"]]}
                ps[f"{nm[t]}|{tpl}|along_mean_noarc"] = [round(float(np.nanmean(al[keep, s])), 2) for s in range(nf)]
        W["path_slope"] = ps
        # 앞섬 회귀 — 같은 슬롯 안: 예측 투영 변위 = a + b · 진실 변위 (외형 템플릿, arc 제외)
        ld = {}
        for t in tg4:
            rr = {}
            for s in LEAD[A["name"]]:
                d = S[s]["ct"] - cl; dist = np.linalg.norm(d, axis=-1); uu = d / np.maximum(dist, 1e-6)[:, None]
                al = ((per[t][s]["ap"] + 0.5 - cl) * uu).sum(-1)
                m = keep & (dist > 0.3) & np.isfinite(al)
                b, a = np.polyfit(dist[m], al[m], 1)
                bs = np.array([np.polyfit(dist[bi[m[bi]]], al[bi[m[bi]]], 1) for bi in BI_lead])
                rr[s] = dict(b=[round(float(b), 3), round(float(np.percentile(bs[:, 0], 2.5)), 3), round(float(np.percentile(bs[:, 0], 97.5)), 3)],
                             a_cells=[round(float(a), 3), round(float(np.percentile(bs[:, 1], 2.5)), 3), round(float(np.percentile(bs[:, 1], 97.5)), 3)],
                             mean_true_cells=round(float(dist[m].mean()), 2), n=int(m.sum()))
            ld[nm[t]] = rr
        W["lead"] = ld
        far = FAR[A["name"]]
        W["per_law_far_slot"] = far
        W["per_law_far"] = {nm[t]: {L: [round(float(np.nanmean(per[t][far]["an"][ix])), 3), round(float(np.nanmean(per[t][far]["hn"][ix])), 3)]
                                    for L, ix in SAME.items()} for t in tg4}
        wins[A["name"]] = W
    res["p2"] = wins

    # ---- P3: 자기 예측 되먹임 (ar_pA) · 관측 되먹임 (ar_z · ar_hcA), 두 템플릿, 릴리즈와의 짝 차이
    if all((CACHE / f"v3_p3{t}" / "p3.npy").exists() for t in tg4):
        J = 8
        m3 = {t: json.load(open(CACHE / f"v3_p3{t}" / "meta.json")) for t in tg4}
        ids3 = m3[""]["video_ids"]; arms3 = list(m3[""]["arms"]); n3 = len(ids3)
        for t in tg4: assert m3[t]["video_ids"] == ids3 and list(m3[t]["arms"]) == arms3
        X3 = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids3]); Y3 = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids3])
        INF3 = np.stack([fl(idx[v]["in_frame_by_sample"]) for v in ids3]) > 0.5
        traj3 = np.array(["|".join(map(str, t)) for t in m3[""]["traj"]]); scen3 = np.array(m3[""]["scenario"])
        ut3 = np.unique(traj3); tix3 = {u_: np.where(traj3 == u_)[0] for u_ in ut3}
        rng3 = np.random.default_rng(20260926); BI3 = [np.concatenate([tix3[u_] for u_ in rng3.choice(ut3, len(ut3))]) for _ in range(2000)]
        xy = lambda f0: np.stack([(X3[:, f0] + X3[:, f0 + 1]) / 2, (Y3[:, f0] + Y3[:, f0 + 1]) / 2], -1) / CELL
        Lp = xy(30); T = np.stack([np.clip(np.floor(xy(32 + 2 * j)), 0, 15) for j in range(J)], 1)
        OK = np.stack([INF3[:, 30] & INF3[:, 31] & INF3[:, 32 + 2 * j] & INF3[:, 33 + 2 * j] for j in range(J)], 1)
        u3 = xy(32 + 2 * (J - 1)) - Lp; un = np.linalg.norm(u3, axis=-1); u3 = u3 / np.maximum(un, 1e-6)[:, None]
        okc = OK.all(1) & (un >= 1)
        js = np.arange(3, J) - np.arange(3, J).mean()

        def feats(a):
            al = ((a + 0.5 - Lp[:, None]) * u3[:, None]).sum(-1)
            slv = np.where(okc, (np.nan_to_num(al)[:, 3:] * js).sum(1) / (js ** 2).sum(), np.nan)
            ln = []
            for j in range(4, J):
                mv = OK[:, j] & (np.abs(T[:, j] - T[:, j - 1]).max(-1) > 0)
                ln.append(np.where(mv, (a[:, j] == T[:, j]).all(-1), np.nan))
            ln = np.nanmean(np.stack(ln, 1).astype(float), 1)
            j = 7; h = np.where(OK[:, j], np.abs(a[:, j] - T[:, j]).max(-1) <= 1, np.nan).astype(float)
            nx = np.full(n3, np.nan)
            for i in range(n3):
                if not OK[i, j]: continue
                p = np.where((scen3 == scen3[i]) & (traj3 != traj3[i]) & OK[:, j])[0]
                nx[i] = (np.abs(a[i, j][None] - T[p, j]).max(-1) <= 1).mean()
            return dict(sl=slv, ln=ln, hn7=h - nx, along=al)
        ci3 = lambda v: ci(v, BI3)
        true_sl = np.where(okc, (((T + 0.5 - Lp[:, None]) * u3[:, None]).sum(-1)[:, 3:] * js).sum(1) / (js ** 2).sum(), np.nan)
        P3 = dict(n_clips=n3, n_traj=int(len(ut3)), true_slope=ci3(true_sl), j0_arms_identical={})
        F = {}
        for t in tg4:
            M = np.load(CACHE / f"v3_p3{t}" / "p3.npy")
            P3["j0_arms_identical"][nm[t]] = bool(np.allclose(M[:, 0, :, 8], M[:, 0, :1, 8], atol=1e-4))
            for ai, arm in enumerate(arms3):
                if arm not in ("ar_z", "ar_pA", "ar_hcA"): continue
                for tpl, cols in (("tru", [1, 0]), ("app", [5, 4])):
                    F[(t, arm, tpl)] = feats(M[:, :, ai, cols])
        rr = {}
        for (t, arm, tpl), f in F.items():
            r = dict(slope=ci3(f["sl"]), land_new_j4_7=ci3(f["ln"]), hit_null_j7=ci3(f["hn7"]),
                     along_mean=[round(float(np.nanmean(f["along"][:, j])), 2) for j in range(J)])
            if t: r["paired_slope_minus_release"] = ci3(f["sl"] - F[("", arm, tpl)]["sl"])
            rr[f"{nm[t]}|{arm}|{tpl}"] = r
        P3["arms"] = rr
        res["p3"] = P3
    return res


full = dict(out)
if RAW_OK:
    full["_v2_raw"] = raw_section()
else:
    print(f"[cmp] 원자료 절 건너뜀 ({CACHE} 없음 또는 CMP_SKIP_RAW=1) — '_v2_raw' 키 없음")
(R / "cmp").mkdir(exist_ok=True)
json.dump(full, open(R / f"cmp/cmp_predictors{OUT_TAG}.json", "w"), indent=1, default=float)
keys = sorted({k for r in out.values() for k in r})
for k in keys:
    print(f"\n{k}")
    for name, _ in TAGS:
        v = out[name].get(k)
        if v is None: continue
        if isinstance(v, list) and v and isinstance(v[0], list):
            s = " ".join(f"{x[0]:+.2f}" for x in v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            s = " ".join("cens" if x.get("censored") else f"{x.get('slot')}" for x in v)
        else:
            s = " ".join(f"{x:+.2f}" if isinstance(x, float) else str(x) for x in v) if isinstance(v, list) else str(v)
        print(f"   {name:22s} {s}")
