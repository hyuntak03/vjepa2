#!/usr/bin/env python3
"""P3 분석 — 닫힘 검정 (계획 PLAN_ENCODER_PREDICTOR_2026-09-25 §5). 입력 v3_p3/. 정의는 H5b 와 같다:
  a_j = 팔의 argmax 칸 (템플릿 = 표준 h_j 진실 칸),  T_j = 진실 칸,  L = 문맥 마지막 칸
  hit_j − null_j (null = 같은 법칙 · 다른 궤적의 T_j),  landing: T_j ≠ T_{j−1} 인 clip 에서 P(a_j = T_j) 대 P(a_j = T_{j−1})
  along slope: (a_j − L)·u (u = 진실 경로 현 방향), j = 3..7 최소제곱 기울기 (칸/튜블릿) — 진실 기울기와 비교
  obj: contrast ≥ 0.357.   mapq: 사상 rel L1 (W, A).   CI = 궤적 군집 bootstrap.

⚠️ 2026-09-25 적대 검증 반영 (기존 키·값은 그대로 두고 키만 더했다; 새 키는 따로 쓰는 rng2 · B=2000 궤적 bootstrap):
  - 기존 `land_fed` 는 **진실 T_{j−1}** 착지다 (실제로 되먹인 토큰의 자리가 아니다). 같은 값을 `land_prev_true` 로도 싣는다.
    실제로 되먹인 자리: 관측 팔 (ar_z · ar_hc*) 은 T_{j−1}, 자기 예측 팔 (ar_p*) 은 직전 자기 argmax a_{j−1} → `land_prev_self` = P(a_j = a_{j−1}).
  - `hit_null_exh` : 귀무 = 같은 법칙 · 다른 궤적 · ok clip **전부** 의 평균 적중 (기존 `hit_null` 은 무작위 1 clip 귀무).
  - `*_app` : 외형 템플릿 (문맥 마지막 튜블릿의 물체 칸 표준 h 토큰, 미래 정보 없음) 로 읽은 착지 · 기울기.
    ⚠️ 정정 (2026-09-25, 적대 검증 2 차): "미래 정보 없음" 은 틀렸다. 이 템플릿은 진실 미래 **위치** 를 쓰지 않을 뿐,
    토큰은 창 [16,48) 전체를 본 표준 **양방향** target encoder 에서 나온다 (미래 프레임을 봤다). 옛 defs 문자열은 두고 r2.defs 에 정정본.
  - `stall_at_T` : P(a_j = T_k) (j = 6, 7; k = L, 0..4) — 멈춘 자리.
  - `paired` : 같은 clip 짝 차이 (기울기, 새 칸 착지 j3–7).
  - `H5_matched` : 같은 784 clip 의 H5 팔 (os · ar_p · ar_z · ar_h, v3_h5/h5.npy C16) 을 같은 정의로 읽고 P3 팔과 짝짓는다.
    ar_h = 같은 채널 어댑터 A 로 **양방향 표적 참 토큰** h_j 를 되먹인 팔. 형식 효과 = ar_h − ar_hcA, 내용 효과 = ar_pA − ar_h.
    같은 predictor 인지 검사한다: j0 (되먹임 전) 의 L1 이 H5 ar_p 와 평균 2e-3 안이어야 짝짓는다 (다른 predictor 결과 폴더면 건너뜀).
    H5 폴더는 env H5_DIR (기본 <입력의 부모>/v3_h5).
  - `r2` (2026-09-25 적대 검증 2 차, P3B 정정 2): 복사 기준선 기울기 (마지막 되먹인 참 칸 T_{j−1} 복사), 2×2 (역사 × 마지막 토큰) 상호작용 ·
    단순 효과, ar_z−ar_pA 격차 회복률, 걸음별 along 차이, 기울기 창 민감도 (j3–6 · j2–7 · j4–7), Chebyshev 거리 (j3–7),
    tf_last 가 되먹인 p^z 칸을 복사하는가, 되먹인 p^z 가 진실 칸에 놓인 clip 만의 tf_last 착지 (지평 의존), 두 템플릿 argmax 일치.
    H5 짝에 올바른 짝 `argmax_agree_ar_hA_vs_H5ar_h` · `l1_absdiff_*` 를 더했다 (옛 `argmax_agree_ar_hcA_vs_H5ar_h` 는 잘못 짝지은 키).

  python p3_analyze.py [입력 폴더]      (출력 exp_results/p3/p3_v3${OUT_TAG}.json)
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/p3"; OUT.mkdir(parents=True, exist_ok=True)
CELL, THR, J = 18.0, 0.357, 8
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
M = np.load(I / ("p3b.npy" if (I / "p3b.npy").exists() else "p3.npy"))
Q = np.load(I / "mapq.npy") if (I / "mapq.npy").exists() else np.full((n, J, 2), np.nan)
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); scen = np.array(meta["scenario"])
rng = np.random.default_rng(0); ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
other = np.array([rng.choice(np.where((scen == scen[i]) & (traj != traj[i]))[0] if ((scen == scen[i]) & (traj != traj[i])).any() else np.delete(np.arange(n), i)) for i in range(n)])
xy = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / CELL
cl = lambda v: np.clip(np.floor(v), 0, 15)
Lxy = xy(30); T = np.stack([cl(xy(32 + 2 * j)) for j in range(J)], 1)             # (n, J, 2)
OK = np.stack([INF[:, 30] & INF[:, 31] & INF[:, 32 + 2 * j] & INF[:, 33 + 2 * j] for j in range(J)], 1)
u = xy(32 + 2 * (J - 1)) - Lxy; un = np.linalg.norm(u, axis=-1); u = u / np.maximum(un, 1e-6)[:, None]


def boot(v, B=500):
    m = np.nanmean(v); bs = [np.nanmean(v[np.concatenate([tix[t] for t in rng.choice(ut, len(ut))])]) for _ in range(B)]
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


out = {"mapq_rel_l1": {"W": [round(float(np.nanmean(Q[:, j, 0])), 4) for j in range(J)], "A": [round(float(np.nanmean(Q[:, j, 1])), 4) for j in range(J)]}}
true_al = np.stack([((cl(xy(32 + 2 * j)) + 0.5 - Lxy) * u).sum(-1) for j in range(J)], 1)
for ai, arm in enumerate(arms):
    a = M[:, :, ai, [1, 0]]                                                       # (n, J, 2) x, y
    hit = (np.abs(a - T).max(-1) <= 1).astype(float); nul = (np.abs(a - T[other]).max(-1) <= 1).astype(float)
    obj = ((M[:, :, ai, 2] - M[:, :, ai, 3]) >= THR).astype(float)
    for v in (hit, nul, obj): v[~OK] = np.nan
    land_new, land_fed = [], []
    for j in range(J):
        prev = T[:, j - 1] if j > 0 else cl(Lxy)
        mv = OK[:, j] & (np.abs(T[:, j] - prev).max(-1) > 0)
        ln = np.where(mv, (a[:, j] == T[:, j]).all(-1), np.nan).astype(float); lf = np.where(mv, (a[:, j] == prev).all(-1), np.nan).astype(float)
        land_new.append(boot(ln)); land_fed.append(boot(lf))
    al = ((a + 0.5 - Lxy[:, None]) * u[:, None]).sum(-1)                         # (n, J)
    okc = OK.all(1) & (un >= 1)
    js = np.arange(3, J) - np.arange(3, J).mean()
    slope = lambda Yv: (Yv[:, 3:] * js).sum(1) / (js ** 2).sum()
    sl = np.where(okc, slope(np.nan_to_num(al)), np.nan); tsl = np.where(okc, slope(np.nan_to_num(true_al)), np.nan)
    out[arm] = dict(hit_null=[boot(hit[:, j] - nul[:, j]) for j in range(J)], obj=[boot(obj[:, j]) for j in range(J)],
                    land_new=land_new, land_fed=land_fed, slope=boot(sl), true_slope=boot(tsl), l1=[round(float(np.nanmean(M[:, j, ai, 8])), 4) for j in range(J)])

# ───────────── 2026-09-25 적대 검증 반영: 추가 키 (위 키·값은 그대로; 새 키는 rng2 로만 뽑는다) ─────────────
rng2 = np.random.default_rng(12345); B2 = 2000
BI = [np.concatenate([tix[t] for t in rng2.choice(ut, len(ut))]) for _ in range(B2)]   # 모든 새 CI · 짝 차이가 같은 재표본을 쓴다


def boot2(v):
    v = np.asarray(v, float); bs = np.array([np.nanmean(v[b]) for b in BI])
    return [round(float(np.nanmean(v)), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


okc = OK.all(1) & (un >= 1)
js = np.arange(3, J) - np.arange(3, J).mean()
slope35 = lambda Yv: (Yv[:, 3:] * js).sum(1) / (js ** 2).sum()
Lcell = cl(Lxy)
POOL = [[np.where((scen == scen[i]) & (traj != traj[i]) & OK[:, j])[0] if OK[i, j] else None for i in range(n)] for j in range(J)]


def readout(a):
    """a: (n, J, 2) x,y argmax 칸 → clip 단위 배열 (bootstrap 은 호출자가)."""
    hit = np.where(OK, np.abs(a - T).max(-1) <= 1, np.nan).astype(float)
    nx = np.full((n, J), np.nan)
    for j in range(J):
        for i in range(n):
            p = POOL[j][i]
            if p is not None and len(p):
                nx[i, j] = (np.abs(a[i, j][None] - T[p, j]).max(-1) <= 1).mean()
    ln, lpt, lps = [], [], []
    for j in range(J):
        prev = T[:, j - 1] if j > 0 else Lcell
        mv = OK[:, j] & (np.abs(T[:, j] - prev).max(-1) > 0)
        ln.append(np.where(mv, (a[:, j] == T[:, j]).all(-1), np.nan))
        lpt.append(np.where(mv, (a[:, j] == prev).all(-1), np.nan))
        lps.append(np.where(mv, (a[:, j] == a[:, j - 1]).all(-1), np.nan) if j > 0 else np.full(n, np.nan))
    al = ((a + 0.5 - Lxy[:, None]) * u[:, None]).sum(-1)
    return dict(hit=hit, null=nx, ln=np.stack(ln, 1).astype(float), lpt=np.stack(lpt, 1).astype(float),
                lps=np.stack(lps, 1).astype(float), al=al, sl=np.where(okc, slope35(np.nan_to_num(al)), np.nan))


def stall(a):
    r = {}
    for j in (6, 7):
        d = {"L": round(float(((a[:, j] == Lcell).all(-1) & OK[:, j]).mean()), 3)}
        for k in range(5):
            d[f"T{k}"] = round(float(((a[:, j] == T[:, k]).all(-1) & OK[:, j]).mean()), 3)
        r[f"a{j}"] = d
    return r


def summarize(tru, app, con):
    """tru/app: readout dict (진실 칸 템플릿 / 외형 템플릿), con: (n, J) contrast (진실 칸 템플릿)."""
    obj = np.where(OK, con >= THR, np.nan).astype(float)
    return dict(
        land_new_b2k=[boot2(tru["ln"][:, j]) for j in range(J)],
        land_prev_true=[boot2(tru["lpt"][:, j]) for j in range(J)],          # = 기존 land_fed (진실 T_{j−1})
        land_prev_self=[boot2(tru["lps"][:, j]) for j in range(J)],          # P(a_j = a_{j−1}); j0 은 NaN
        hit=[boot2(tru["hit"][:, j]) for j in range(J)], null_exh=[boot2(tru["null"][:, j]) for j in range(J)],
        hit_null_exh=[boot2(tru["hit"][:, j] - tru["null"][:, j]) for j in range(J)],
        obj_b2k=[boot2(obj[:, j]) for j in range(J)],
        slope_b2k=boot2(tru["sl"]), slope_by_scen={s: round(float(np.nanmean(tru["sl"][scen == s])), 3) for s in np.unique(scen)},
        along_mean=[round(float(np.nanmean(np.where(okc, tru["al"][:, j], np.nan))), 3) for j in range(J)],
        land_new_app=[boot2(app["ln"][:, j]) for j in range(J)], land_prev_true_app=[boot2(app["lpt"][:, j]) for j in range(J)],
        slope_app=boot2(app["sl"]), stall_at_T=stall(tru["a"]))


RD = {}
for ai, arm in enumerate(arms):
    tru = readout(M[:, :, ai, [1, 0]]); tru["a"] = M[:, :, ai, [1, 0]]
    app = readout(M[:, :, ai, [5, 4]])
    RD[arm] = (tru, app)
    out[arm].update(summarize(tru, app, M[:, :, ai, 2] - M[:, :, ai, 3]))
out["true_slope_b2k"] = boot2(np.where(okc, slope35(true_al), np.nan))
out["true_slope_by_scen"] = {s: round(float(np.nanmean(np.where(okc, slope35(true_al), np.nan)[scen == s])), 3) for s in np.unique(scen)}
out["true_along_mean"] = [round(float(np.nanmean(np.where(okc, true_al[:, j], np.nan))), 3) for j in range(J)]
out["n_clips"] = int(n); out["n_traj"] = int(len(ut)); out["boot"] = f"trajectory cluster, B={B2}, rng2 seed 12345 (new keys); legacy keys B=500 seed 0"
out["defs"] = {"template_tru": "standard bidirectional h_j (target encoder on frames [16,48)) token at true cell T_j",
               "template_app": "standard bidirectional h token of the last context tubelet at the last-observed cell L (no future info)",
               "land_prev_true": "P(a_j = T_{j-1}) among clips whose true cell changes (== legacy land_fed)",
               "land_prev_self": "P(a_j = a_{j-1}); for self-prediction arms this is the cell of the token actually fed",
               "hit_null_exh": "Chebyshev<=1 hit minus mean hit against all same-scenario other-trajectory ok clips"}


def pdiff(Ax, Ay, key):
    return boot2(Ax[key] - Ay[key]) if key == "sl" else [boot2(Ax["ln"][:, j] - Ay["ln"][:, j]) for j in range(3, J)]


PAIRS = [(a_, "ar_z") for a_ in arms if a_ != "ar_z" and "ar_z" in arms] + \
        [(x_, y_) for x_, y_ in (("ar_hcA", "ar_pA"), ("ar_hcW", "ar_pW"), ("ar_pW", "ar_pA")) if x_ in arms and y_ in arms]
out["paired"] = {f"{x_}-{y_}": dict(slope=pdiff(RD[x_][0], RD[y_][0], "sl"), slope_app=pdiff(RD[x_][1], RD[y_][1], "sl"),
                                    land_new_j3_7=pdiff(RD[x_][0], RD[y_][0], "ln"),
                                    argmax_agree=[round(float((M[:, j, arms.index(x_), :2] == M[:, j, arms.index(y_), :2]).all(-1).mean()), 3) for j in range(J)])
                 for x_, y_ in PAIRS}
out["mapq_W_clip_pct_5_50_95"] = [round(float(np.nanpercentile(Q[:, :, 0], q)), 3) for q in (5, 50, 95)]
out["mapq_A_clip_pct_5_50_95"] = [round(float(np.nanpercentile(Q[:, :, 1], q)), 3) for q in (5, 50, 95)]
if (I / "W.json").exists():
    out["W_fit"] = json.load(open(I / "W.json"))                                  # 적합: 양방향 전창 (표준 h, 32 장 문맥 z) 쌍

# ───────────── 2026-09-25 적대 검증 2 차 반영 (P3B 정정 2): 새 키 `r2` 하나에만 싣는다. 위 키·값은 그대로 ─────────────
# 같은 BI (rng2 seed 12345, B=2000 궤적 bootstrap). 검증자 독립 재계산 (_verify_scratch/P3B, seed 777) 과 셋째 자리까지 일치해야 한다.
alg = lambda a: ((a + 0.5 - Lxy[:, None]) * u[:, None]).sum(-1)                  # along (칸), 문맥 마지막 L 기준, 진실 경로 현 방향


def slope_w(al, js):
    js = np.asarray(js); w = js - js.mean()
    return np.where(okc, (np.nan_to_num(al)[:, js] * w).sum(1) / (w ** 2).sum(), np.nan)


def ratio_boot(num, den):
    m = np.nanmean(num) / np.nanmean(den); bs = np.array([np.nanmean(num[b]) / np.nanmean(den[b]) for b in BI])
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


TE = ("tru", "app")                                                               # RD[arm][0] = 진실 칸 템플릿, [1] = 외형 템플릿
Tprev = np.concatenate([Lcell[:, None], T[:, :-1]], 1)                            # T_{j−1} (T_{−1} = L)
Tprev2 = np.concatenate([Lcell[:, None], Lcell[:, None], T[:, :-2]], 1)          # T_{j−2}
r2 = {"defs": {
    "template_app_corrected": "h token of the last context tubelet (TC-1) at the last-observed cell L, from the STANDARD BIDIRECTIONAL target encoder "
                              "run on frames [16,48) — it does not use the true future LOCATION, but the token itself has seen future frames "
                              "(corrects legacy defs.template_app 'no future info')",
    "copy_baseline": "fake predictor that outputs the cell of the last fed TRUE token (T_{j-1}); its j3-7 along slope is the copy baseline for the slope metric",
    "twobytwo": "factors: history (z*_0..z*_{j-2} true | A(p^pA_0..p^pA_{j-2}) self) x last fed token (z*_{j-1} true | A(p_{j-1}) self). "
                "cells: ar_z=(T,T), tf_first=(S,T), tf_last=(T,S), ar_pA=(S,S). NOT a clean 2x2: the self last token is p^z_{j-1} (ar_z chain) "
                "in tf_last but p^pA_{j-1} (own chain) in ar_pA, so (tf_last - ar_pA) mixes history swap with last-token content. "
                "The 16-frame context z is TRUE in every arm; only j-1 fed tokens are swapped.",
    "recovery": "(arm - ar_pA) / (ar_z - ar_pA) on j3-7 slope; ratio of means, trajectory-cluster bootstrap",
    "cheb": "mean over j3-7 of Chebyshev distance (cells) between argmax and true cell T_j",
    "tf_last_fed": "fed token cell = ar_z argmax a^z_{j-1} under the same template; subset T_j != a^z_{j-1}",
    "tf_last_given_fed_correct": "truth-cell template only; subset T_j != T_{j-1} and a^z_{j-1} == T_{j-1} (fed p^z placed on the true cell)"}}
r2["copy_baseline_slope"] = {"copy_T_prev": boot2(slope_w(alg(Tprev), range(3, J))), "copy_T_prev2": boot2(slope_w(alg(Tprev2), range(3, J))),
                             "true": boot2(slope_w(alg(T), range(3, J)))}
r2["j0_exact_hit_T0"] = {t: round(float(((M[:, 0, 0, c] == T[:, 0]).all(-1) & OK[:, 0]).sum() / OK[:, 0].sum()), 3)
                         for t, c in (("tru", [1, 0]), ("app", [5, 4]))}
r2["template_argmax_agree"] = {arm: [round(float((M[:, j, ai, :2] == M[:, j, ai, 4:6]).all(-1).mean()), 3) for j in range(J)] for ai, arm in enumerate(arms)}
cons = {}
for ai, arm in enumerate(arms):
    at, aa = M[:, 3:, ai, [1, 0]], M[:, 3:, ai, [5, 4]]; ag = (at == aa).all(-1); ht = np.abs(at - T[:, 3:]).max(-1) <= 1
    cons[arm] = {"agree_frac_j3_7": round(float(ag.mean()), 3), "hit_tru_given_agree_j3_7": round(float(ht[ag].mean()), 3),
                 "hit_tru_given_disagree_j3_7": round(float(ht[~ag].mean()), 3)}
r2["template_consensus"] = cons
for ti, t in enumerate(TE):
    A_ = {arm: RD[arm][ti] for arm in arms}
    ch = {arm: np.where(okc, np.abs(M[:, :, arms.index(arm), [1, 0] if t == "tru" else [5, 4]] - T).max(-1)[:, 3:].mean(1), np.nan) for arm in arms}
    r2[f"cheb_j3_7_{t}"] = {arm: boot2(ch[arm]) for arm in arms}
    if "ar_z" in arms:
        r2[f"cheb_j3_7_paired_{t}"] = {f"{x}-ar_z": boot2(ch[x] - ch["ar_z"]) for x in arms if x != "ar_z"}
        r2[f"along_diff_vs_ar_z_{t}"] = {f"{x}-ar_z": [boot2(np.where(okc, A_[x]["al"][:, j] - A_["ar_z"]["al"][:, j], np.nan)) for j in range(J)]
                                          for x in arms if x != "ar_z"}
        r2[f"slope_window_vs_ar_z_{t}"] = {f"{x}-ar_z": {nm: boot2(slope_w(A_[x]["al"], js) - slope_w(A_["ar_z"]["al"], js))
                                                         for nm, js in (("j3_7", range(3, 8)), ("j3_6", range(3, 7)), ("j2_7", range(2, 8)), ("j4_7", range(4, 8)))}
                                           for x in arms if x != "ar_z"}
        r2[f"slope_vs_ar_z_by_scen_{t}"] = {f"{x}-ar_z": {s: round(float(np.nanmean((A_[x]["sl"] - A_["ar_z"]["sl"])[scen == s])), 3) for s in np.unique(scen)}
                                            for x in arms if x != "ar_z"}
    if {"ar_z", "ar_pA"} <= set(arms):
        den = A_["ar_z"]["sl"] - A_["ar_pA"]["sl"]
        r2[f"recovery_{t}"] = {x: ratio_boot(A_[x]["sl"] - A_["ar_pA"]["sl"], den) for x in arms if x not in ("ar_z", "ar_pA")}
    if {"ar_z", "ar_pA", "tf_first", "tf_last"} <= set(arms):
        s = {x: A_[x]["sl"] for x in ("ar_z", "ar_pA", "tf_first", "tf_last")}
        r2[f"twobytwo_{t}"] = {
            "interaction(ar_z-tf_first-tf_last+ar_pA)": boot2(s["ar_z"] - s["tf_first"] - s["tf_last"] + s["ar_pA"]),
            "hist_effect|last_true(ar_z-tf_first)": boot2(s["ar_z"] - s["tf_first"]),
            "hist_effect|last_self(tf_last-ar_pA)": boot2(s["tf_last"] - s["ar_pA"]),
            "last_effect|hist_true(ar_z-tf_last)": boot2(s["ar_z"] - s["tf_last"]),
            "last_effect|hist_self(tf_first-ar_pA)": boot2(s["tf_first"] - s["ar_pA"]),
            "tf_first-tf_last": boot2(s["tf_first"] - s["tf_last"])}
        az = M[:, :, arms.index("ar_z"), [1, 0] if t == "tru" else [5, 4]]; tl = M[:, :, arms.index("tf_last"), [1, 0] if t == "tru" else [5, 4]]
        d = {"n": [], "tf_last_eq_fed_pz": [], "tf_last_eq_T_j": [], "ar_z_eq_own_prev": []}
        for j in range(3, J):
            mv = OK[:, j] & (np.abs(T[:, j] - az[:, j - 1]).max(-1) > 0)
            d["n"].append(int(mv.sum())); d["tf_last_eq_fed_pz"].append(round(float((tl[mv, j] == az[mv, j - 1]).all(-1).mean()), 3))
            d["tf_last_eq_T_j"].append(round(float((tl[mv, j] == T[mv, j]).all(-1).mean()), 3))
            d["ar_z_eq_own_prev"].append(round(float((az[mv, j] == az[mv, j - 1]).all(-1).mean()), 3))
        r2[f"tf_last_fed_copy_vs_step_{t}"] = d
        r2[f"copy_fed_pz_slope_{t}"] = boot2(slope_w(alg(np.concatenate([az[:, :1], az[:, :-1]], 1)), range(3, J)))
        if t == "tru":                                                            # 되먹인 p^z 자리가 맞은 clip 만: 붕괴가 j 에 남는가 (지평 의존)
            cz = M[:, :, arms.index("ar_z"), 2] - M[:, :, arms.index("ar_z"), 3]
            g = {"n": [], "tf_last_eq_T_j": [], "tf_last_eq_T_prev": [], "n_obj": [], "tf_last_eq_T_j_obj": []}
            for j in range(3, J):
                mv = OK[:, j] & (np.abs(T[:, j] - T[:, j - 1]).max(-1) > 0) & (az[:, j - 1] == T[:, j - 1]).all(-1)
                mo = mv & (cz[:, j - 1] >= THR)
                g["n"].append(int(mv.sum())); g["tf_last_eq_T_j"].append(round(float((tl[mv, j] == T[mv, j]).all(-1).mean()), 3))
                g["tf_last_eq_T_prev"].append(round(float((tl[mv, j] == T[mv, j - 1]).all(-1).mean()), 3))
                g["n_obj"].append(int(mo.sum())); g["tf_last_eq_T_j_obj"].append(round(float((tl[mo, j] == T[mo, j]).all(-1).mean()), 3))
            r2["tf_last_given_fed_correct"] = g
out["r2"] = r2

# ── H5 짝 (같은 clip, 같은 predictor 일 때만) ──
H5_DIR = Path(os.environ.get("H5_DIR", str(I.parent / "v3_h5")))
h5note = "not found"
if (H5_DIR / "h5.npy").exists() and (H5_DIR / "meta.json").exists():
    m5 = json.load(open(H5_DIR / "meta.json")); pos5 = {v: k for k, v in enumerate(m5["video_ids"])}; a5 = list(m5["arms"])
    C3 = int(meta.get("C", 16))
    if all(v in pos5 for v in ids) and C3 in m5["Cs"]:
        H = np.asarray(np.load(H5_DIR / "h5.npy", mmap_mode="r")[[pos5[v] for v in ids], m5["Cs"].index(C3)])   # (n, J, arms5, 9)
        ref3, ref5 = ("ar_pA", "ar_p") if "ar_pA" in arms else ("ar_z", "ar_z")
        d0 = float(np.nanmean(np.abs(M[:, 0, arms.index(ref3), 8] - H[:, 0, a5.index(ref5), 8])))
        if d0 < 2e-3:
            h5o = {"dir": str(H5_DIR), "C": C3, "j0_l1_absdiff": round(d0, 6)}
            agree = lambda x3, x5: [round(float(((M[:, j, arms.index(x3), :2] == H[:, j, a5.index(x5), :2]).all(-1)).mean()), 3) for j in range(J)]
            for x3, x5 in (("ar_pA", "ar_p"), ("ar_z", "ar_z"), ("ar_hcA", "ar_h"), ("ar_hA", "ar_h")):
                if x3 in arms:
                    h5o[f"argmax_agree_{x3}_vs_H5{x5}"] = agree(x3, x5)
                    h5o[f"l1_absdiff_{x3}_vs_H5{x5}"] = [round(float(np.abs(M[:, j, arms.index(x3), 8] - H[:, j, a5.index(x5), 8]).mean()), 5) for j in range(J)]
            # ⚠️ 2026-09-25 적대 검증 2 차: 위 ("ar_hcA", "ar_h") 는 잘못 짝지은 비교다 (j4 뒤 0.10–0.51 은 재현 실패가 아니다).
            #    P3b 에서 H5 ar_h (A(양방향 표적 참 h_j)) 를 재현하는 팔은 ar_hA → argmax_agree_ar_hA_vs_H5ar_h 를 읽는다. 옛 키는 지우지 않는다.
            h5o["argmax_agree_note"] = "argmax_agree_ar_hcA_vs_H5ar_h is a MISPAIRED comparison (kept for continuity); the arm reproducing H5 ar_h is ar_hA"
            for arm5 in ("os", "ar_p", "ar_z", "ar_h"):
                if arm5 not in a5: continue
                k5 = a5.index(arm5)
                tru = readout(H[:, :, k5, [1, 0]]); tru["a"] = H[:, :, k5, [1, 0]]; app = readout(H[:, :, k5, [5, 4]])
                RD["H5_" + arm5] = (tru, app)
                h5o[arm5] = summarize(tru, app, H[:, :, k5, 2] - H[:, :, k5, 3])
            pr = {}
            for x_, y_, name in (("H5_ar_h", "ar_hcA", "form_effect(ar_h-ar_hcA)"), ("ar_pA", "H5_ar_h", "content_effect(ar_pA-ar_h)"),
                                 ("ar_hcA", "H5_ar_h", "ar_hcA-ar_h"), ("H5_os", "ar_pA", "os-ar_pA"), ("H5_ar_h", "ar_pA", "ar_h-ar_pA")):
                if x_ in RD and y_ in RD:
                    pr[name] = dict(slope=pdiff(RD[x_][0], RD[y_][0], "sl"), slope_app=pdiff(RD[x_][1], RD[y_][1], "sl"),
                                    land_new_j3_7=pdiff(RD[x_][0], RD[y_][0], "ln"))
            h5o["paired"] = pr
            out["H5_matched"] = h5o; h5note = f"paired ({H5_DIR}, j0 L1 diff {d0:.2e})"
        else:
            h5note = f"skipped: different predictor (j0 L1 diff {d0:.3e} ≥ 2e-3)"
    else:
        h5note = "skipped: clip ids or C not in H5 meta"
out["H5_note"] = h5note

json.dump(out, default=float, fp=open(OUT / f"p3_v3{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1)
print("사상 rel L1  W:", out["mapq_rel_l1"]["W"], " A:", out["mapq_rel_l1"]["A"])
for arm in arms:
    r = out[arm]
    print(f"\n{arm:7s} 적중−귀무 " + " ".join(f"{x[0]:+.2f}" for x in r["hit_null"]) + " | 물체 " + " ".join(f"{x[0]:.2f}" for x in r["obj"]))
    print(f"        새칸착지 " + " ".join(f"{x[0]:.2f}" for x in r["land_new"]) + " | 진실이전칸 T_{j-1} " + " ".join(f"{x[0]:.2f}" for x in r["land_fed"])
          + f" | 기울기 {r['slope'][0]:+.3f} [{r['slope'][1]:+.3f},{r['slope'][2]:+.3f}] (진실 {r['true_slope'][0]:+.3f})")
    print(f"        직전자기칸 a_(j-1) " + " ".join("  - " if np.isnan(x[0]) else f"{x[0]:.2f}" for x in r["land_prev_self"])
          + f" | 적중−귀무(전수) j7 {r['hit_null_exh'][7][0]:+.3f} [{r['hit_null_exh'][7][1]:+.3f},{r['hit_null_exh'][7][2]:+.3f}]"
          + f" | 기울기 B2k {r['slope_b2k']} · 외형 템플릿 {r['slope_app']}")
    print(f"        멈춘 자리 a7: {r['stall_at_T']['a7']}")
print(f"\n진실 기울기 {out['true_slope_b2k']}")
for k, v in out["paired"].items():
    print(f"짝 {k:18s} 기울기 {v['slope']} 외형 {v['slope_app']}")
print("\nH5:", h5note)
if "H5_matched" in out:
    h5o = out["H5_matched"]
    for arm5 in ("os", "ar_p", "ar_z", "ar_h"):
        if arm5 in h5o:
            r = h5o[arm5]
            print(f"H5 {arm5:5s} 새칸 " + " ".join(f"{x[0]:.3f}" for x in r["land_new_b2k"]) + " | T_{j-1} " + " ".join(f"{x[0]:.3f}" for x in r["land_prev_true"])
                  + f" | 기울기 {r['slope_b2k']} 외형 {r['slope_app']} | a7 {r['stall_at_T']['a7']}")
    for k, v in h5o["paired"].items():
        print(f"짝 {k:28s} 기울기 {v['slope']} 외형 {v['slope_app']} | 새칸 j3–7 " + " ".join(f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for x in v["land_new_j3_7"]))
    print({k: v for k, v in h5o.items() if k.startswith(("argmax", "j0"))})
if "r2" in out:
    r2 = out["r2"]
    print("\n[r2] 복사 기준선 기울기", r2["copy_baseline_slope"], "| j0 정확 적중", r2["j0_exact_hit_T0"])
    for t in TE:
        if f"twobytwo_{t}" in r2:
            print(f"[r2 {t}] 2×2", r2[f"twobytwo_{t}"])
        if f"recovery_{t}" in r2:
            print(f"[r2 {t}] 회복률", r2[f"recovery_{t}"])
        if f"slope_window_vs_ar_z_{t}" in r2 and "tf_first-ar_z" in r2[f"slope_window_vs_ar_z_{t}"]:
            print(f"[r2 {t}] 창 민감도 tf_first−ar_z", r2[f"slope_window_vs_ar_z_{t}"]["tf_first-ar_z"])
    if "tf_last_given_fed_correct" in r2:
        print("[r2] tf_last | 되먹인 p^z 진실 칸", r2["tf_last_given_fed_correct"])
