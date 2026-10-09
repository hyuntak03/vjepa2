#!/usr/bin/env python3
"""M4 분석 — 미래 질의의 문맥 물체 조회 (stride 별). 입력 v3_m4/ (+ 있으면 v3_p2/ 의 loc_tru, v3_m4_ringnull/ 의 CPU ring 귀무).

  python m4_analyze.py [<v3_m4 dir>]        (OUT_TAG 환경변수 = 출력 이름 접미사, 기본 '' → exp_results/m4/m4_v3.json)
  환경변수 M4_P2_DIR (기본 <입력>.replace('v3_m4','v3_p2')), M4_RINGNULL_DIR (기본 <입력>_ringnull) 로 바꿀 수 있다. 없으면 그 절은 건너뛴다.

표적: 적중 템플릿 = **표준 양방향 h** = LN(target_encoder(창 전체)) 의 진실 칸 토큰 (인과 표적 판 없음).  CI = 궤적 군집 bootstrap.

(0) 원판 키 (1차 문서, 보존): arms.<팔>.<Euclid 구간> 의 n, obj_traj, obj_traj_ratio_uniform (**균등 분모 = 9·nc 근사** — 자기 기둥 겹침 제외·가장자리
    잘림 무시; d 0–1 을 약 2 배 과소 추정), obj_traj_maxlayer, last_obj, last_obj_ratio_uniform, self_col, ctx, wdist, knock.
(1) 2026-09-25 적대 검증 뒤 추가 (VERIFY_0925 §M4, 검증 코드 verify0925/M4/m4_verify*.py 를 옮김):
    - obj_traj_ratio_exact: 균등 = 층 평균 문맥 질량 × (실제 key 수 Σ_u |3×3(c_u) − 3×3(query)| / 문맥 key 수).  비의 평균이 아니라 평균의 비.
    - Chebyshev 구간 (0/1/2/3/4–5/6+) 표 `cheb` — knockout 반경 r 이 Chebyshev 라 d 도 Chebyshev 로 맞춘다.  r = d 적중, ∞ 대비 0.03 안 최소 r.
    - r=∞ 위치 귀무 (P2 loc_tru argmax, 같은 clip): 같은 법칙 · 다른 궤적 전부의 진실 칸에 같은 판정 (P2 는 무작위 1 개, 여기는 기대값).
      knockout 팔은 argmax 를 저장하지 않아 귀무를 못 낸다.
    - 적중/실패 조건부 ×균등, 팔 차 (같은 clip 위 s2−s1 · s4−s1 · s4−s2), 법칙별, 같은 Cheb d 안 이른/늦은 슬롯, 법칙·팔 고정효과 회귀.
    - ring 귀무 (CPU fp32 부분집합, m4_ring_null_cpu.py): 같은 튜블릿 · 같은 Chebyshev 거리의 비물체 key 밀도. P-query (p 가 둔 칸). head 별.
알려진 한계: obj_traj (GPU) 는 화면 밖 문맥 튜블릿을 가장자리 칸으로 잘라 넣는다 (s2·s4 clip 의 약 79 %) → s2·s4 ×균등 과소. ring 귀무의 obj 는 화면 안만.
"""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m4")
ROOT = Path(__file__).resolve().parents[2]
TAG = os.environ.get("OUT_TAG", "")
OUT = ROOT / "auto_research/exp_results/m4"; OUT.mkdir(parents=True, exist_ok=True)
P2DIR = Path(os.environ.get("M4_P2_DIR", str(I.parent / I.name.replace("v3_m4", "v3_p2"))))
RNDIR = Path(os.environ.get("M4_RINGNULL_DIR", str(I) + "_ringnull"))
IDX = ROOT / "data_csv/rollout_v3/index_probe.csv"
meta = json.load(open(I / "meta.json")); arms = [a["name"] for a in meta["arms"]]; radii = meta["radii"]
A = np.load(I / "attn.npy"); K = np.load(I / "knock.npy"); Dd = np.load(I / "disp.npy")
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); rng = np.random.default_rng(0)
ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 30)]
S_TOK, G, CELL, SPLIT = 256, 16, 18.0, 32


def boot(vals, rows, B=300):
    """vals: (n_obs,), rows: (n_obs,) clip 인덱스 → 궤적 군집 bootstrap."""
    m = np.nanmean(vals); obs_by_t = {t: np.where(np.isin(rows, tix[t]))[0] for t in ut}; bs = []
    for _ in range(B):
        s = np.concatenate([obs_by_t[t] for t in rng.choice(ut, len(ut))]); bs.append(np.nanmean(vals[s]))
    return [round(float(m), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


# ------------------------------------------------------------------ (0) 원판 (값 · 키 보존)
out = {"repro": json.load(open(I / "repro.json")) if (I / "repro.json").exists() else None, "arms": {}}
for ai, arm in enumerate(arms):
    nc = meta["arms"][ai]["n_ctx"]; Nc = nc * S_TOK
    d = Dd[:, ai]; ok = ~np.isnan(d)
    rows = np.broadcast_to(np.arange(len(d))[:, None], d.shape)
    att = A[:, ai]                                                  # (n, K, L, 5)
    rec = {}
    for lo, hi in BINS:
        m = ok & (d >= lo) & (d < hi)
        if m.sum() < 30: continue
        r_ = rows[m]
        a_mean = att[m].mean(1)                                     # 층 평균 (n_obs, 5)
        a_maxl = att[m][:, :, 0].max(1)                             # 물체 궤적 질량의 층별 최대
        # (원판) 균등 기준: 물체 궤적 key ≈ 9 칸 × nc 튜블릿 (자기 기둥 겹침 제외·가장자리 무시 — 근사), 마지막 물체 9 key
        uni_traj = a_mean[:, 3] * 9 * nc / Nc; uni_last = a_mean[:, 3] * 9 / Nc
        rec[f"{lo}-{hi}"] = dict(n=int(m.sum()),
                                 obj_traj=boot(a_mean[:, 0], r_), obj_traj_ratio_uniform=round(float(np.nanmean(a_mean[:, 0]) / np.nanmean(uni_traj)), 2),
                                 obj_traj_maxlayer=boot(a_maxl, r_),
                                 last_obj=boot(a_mean[:, 1], r_), last_obj_ratio_uniform=round(float(np.nanmean(a_mean[:, 1]) / np.nanmean(uni_last)), 2),
                                 self_col=boot(a_mean[:, 2], r_), ctx=round(float(np.nanmean(a_mean[:, 3])), 3), wdist=round(float(np.nanmean(a_mean[:, 4])), 2),
                                 knock={str(r): round(float(np.nanmean(K[:, ai][m][:, k])), 3) for k, r in enumerate(radii)})
    out["arms"][arm] = rec

# ------------------------------------------------------------------ (1) 기하: 칸 · 정확한 key 수 · 화면 밖
rng2 = np.random.default_rng(1)
n = len(meta["video_ids"]); vids = meta["video_ids"]; law = np.array(meta["scenario"]); laws = sorted(set(law))
idx = {r["video_id"]: r for r in csv.DictReader(open(IDX))}
fl = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in vids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in vids])
INF = np.stack([fl(idx[v]["in_frame_by_sample"]) for v in vids])


def span(c):
    return np.maximum(c - 1, 0), np.minimum(c + 1, G - 1)


def wsize(c):                                                        # c (..., 2) → 3×3 창 크기 (가장자리 잘림)
    lo, hi = span(c); return (hi - lo + 1).prod(-1)


def woverlap(a, b):
    la, ha = span(a); lb, hb = span(b); return np.clip(np.minimum(ha, hb) - np.maximum(la, lb) + 1, 0, None).prod(-1)


geo = {}
for ai, arm in enumerate(arms):
    s, r0, nc, nf = (meta["arms"][ai][k] for k in ("stride", "raw0", "n_ctx", "n_fut"))
    fr = list(range(r0, 64, s)); tub = [(fr[2 * u], fr[2 * u + 1]) for u in range(nc + nf)]
    cx = np.stack([(X[:, a] + X[:, b]) / 2 for a, b in tub], 1); cy = np.stack([(Y[:, a] + Y[:, b]) / 2 for a, b in tub], 1)
    cell = np.stack([np.clip(np.floor(cx / CELL), 0, G - 1), np.clip(np.floor(cy / CELL), 0, G - 1)], -1).astype(int)   # (n, nc+nf, 2) = cell_at
    inf = np.stack([np.minimum(INF[:, a], INF[:, b]) for a, b in tub], 1) > 0
    cc, tc, lc = cell[:, :nc], cell[:, nc:], cell[:, nc - 1]
    dE = np.hypot(*(tc - lc[:, None]).transpose(2, 0, 1)); dC = np.abs(tc - lc[:, None]).max(-1)
    nobj = (wsize(cc)[:, None, :] - woverlap(cc[:, None, :, :], tc[:, :, None, :])).sum(-1)       # (n, nf)
    geo[arm] = dict(s=s, nc=nc, nf=nf, cc=cc, tc=tc, lc=lc, dE=dE, dC=dC, nobj=nobj, nself=wsize(tc) * nc, nlast=wsize(lc),
                    nlast_self=woverlap(lc[:, None], tc), ctx_off=(~inf[:, :nc]).sum(1), fut_off=(~inf[:, nc:]).sum(1), last_off=~inf[:, nc - 1],
                    slot=np.broadcast_to(np.arange(nf)[None], (n, nf)), disp_check=float(np.nanmax(np.abs(Dd[:, ai, :nf] - dE))))
    for t in ut:                                                    # 같은 궤적 = 같은 칸 (외형만 다름)
        ii = tix[t]; assert (tc[ii] == tc[ii[0]]).all() and (lc[ii] == lc[ii[0]]).all(), t


def cboot(fn, B=1000):
    """fn(clip 인덱스 배열, 반복 허용) → 스칼라. 궤적 군집 bootstrap."""
    m = fn(np.arange(n)); bs = []
    for _ in range(B):
        bs.append(fn(np.concatenate([tix[t] for t in rng2.choice(ut, len(ut))])))
    return [round(float(m), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


# r=∞ 위치 귀무 (P2 loc_tru argmax 를 M4 clip 에 맞춤)
null = {}; p2note = None
if (P2DIR / "loc_tru.npy").exists():
    p2m = json.load(open(P2DIR / "meta.json")); LT = np.load(P2DIR / "loc_tru.npy", mmap_mode="r"); pos = {v: i for i, v in enumerate(p2m["video_ids"])}
    p2arms = [x["name"] for x in p2m["arms"]]
    if all(v in pos for v in vids):
        mi = np.array([pos[v] for v in vids]); p2note = dict(dir=str(P2DIR), agreement={})
        for ai, arm in enumerate(arms):
            if arm not in p2arms: continue
            g = geo[arm]; nf = g["nf"]; pj = p2arms.index(arm)
            am = np.stack([LT[mi, pj, :nf, 1], LT[mi, pj, :nf, 0]], -1)                              # (n, nf, 2) = (x, y)
            hitP = (np.abs(am - g["tc"]).max(-1) <= 1).astype(float)
            trajcell = {t: g["tc"][tix[t][0]] for t in ut}; tl = {t: law[tix[t][0]] for t in ut}
            nul = np.zeros((n, nf))
            for i in range(n):
                oc = np.stack([trajcell[t] for t in ut if t != traj[i] and tl[t] == law[i]])            # (m, nf, 2)
                nul[i] = (np.abs(am[i][None] - oc).max(-1) <= 1).mean(0)
            null[arm] = nul
            p2note["agreement"][arm] = round(float((hitP == (K[:, ai, :nf, -1] > 0.5)).mean()), 4)
    else:
        p2note = dict(dir=str(P2DIR), skipped="M4 clip 이 P2 에 없음")


def arm_stats(ai, arm, M, B=1000):
    """M (n, nf) bool 관측 마스크 → 정확한 ×균등 · knockout · r=∞ 귀무 등 dict."""
    g = geo[arm]; nc, nf = g["nc"], g["nf"]; Nc = nc * S_TOK
    am = A[:, ai, :nf].mean(2); ctx = am[..., 3]; kn = K[:, ai, :nf]; hit = kn[..., -1] > 0.5
    ratio = lambda idx, col, cnt, MM=M: am[idx][MM[idx]][:, col].mean() / (ctx[idx][MM[idx]] * cnt[idx][MM[idx]] / Nc).mean()
    r = dict(n=int(M.sum()), n_traj=int(len(np.unique(traj[np.where(M)[0]]))), mean_slot=round(float(g["slot"][M].mean()), 2),
             dE_range=[round(float(g["dE"][M].min()), 2), round(float(g["dE"][M].max()), 2)], dCheb=sorted(set(g["dC"][M].astype(int).tolist())),
             obj_traj_ratio_exact=cboot(lambda idx: ratio(idx, 0, g["nobj"]), B),
             last_obj_ratio_exact=round(float(ratio(np.arange(n), 1, np.broadcast_to(g["nlast"][:, None], (n, nf)))), 2),
             last_self_overlap_frac=round(float((g["nlast_self"][M] / g["nlast"][:, None].repeat(nf, 1)[M]).mean()), 3),
             self_col_ratio_exact=round(float(ratio(np.arange(n), 2, g["nself"])), 2), self_col_mass=round(float(am[M][:, 2].mean()), 4),
             ctx_mass=round(float(ctx[M].mean()), 3),
             obj_traj_ratio_exact_by_layer=np.round(A[:, ai, :nf][M][:, :, 0].mean(0) / (A[:, ai, :nf][M][:, :, 3] * (g["nobj"][M] / Nc)[:, None]).mean(0), 2).tolist(),
             knock_ci={str(rad): cboot(lambda idx, k=k: kn[idx][M[idx]][:, k].mean(), B) for k, rad in enumerate(radii)},
             frac_clips_ctx_offframe=round(float((g["ctx_off"][np.unique(np.where(M)[0])] > 0).mean()), 3))
    for lab, hm in (("hit", hit), ("miss", ~hit)):
        mm = M & hm
        if mm.sum() >= 20:
            r[f"obj_traj_ratio_exact_{lab}"] = [int(mm.sum())] + cboot(lambda idx, mm=mm: ratio(idx, 0, g["nobj"], mm), B)
    if arm in null:
        nul = null[arm]
        r["rinf_hit"] = cboot(lambda idx: hit[idx][M[idx]].mean(), B); r["rinf_null"] = cboot(lambda idx: nul[idx][M[idx]].mean(), B)
        r["rinf_hit_minus_null"] = cboot(lambda idx: (hit[idx][M[idx]] - nul[idx][M[idx]]).mean(), B)
    return r


# Euclid 구간 (원판 구간) 에 정확한 값 추가
for ai, arm in enumerate(arms):
    g = geo[arm]
    for lo, hi in BINS:
        key = f"{lo}-{hi}"
        if key not in out["arms"][arm]: continue
        M = (g["dE"] >= lo) & (g["dE"] < hi)
        out["arms"][arm][key].update(arm_stats(ai, arm, M))

# Chebyshev 구간
CB = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 99)]
out["cheb"] = {}
for ai, arm in enumerate(arms):
    g = geo[arm]; kn = K[:, ai, :g["nf"]]; rec = {}
    for lab, lo, hi in CB:
        M = (g["dC"] >= lo) & (g["dC"] <= hi)
        if M.sum() < 30: continue
        r = arm_stats(ai, arm, M)
        if lo == hi and lo in radii: r["knock_r_eq_d"] = r["knock_ci"][str(lo)]
        h = kn[M].mean(0); ok = [rad for k, rad in enumerate(radii[:-1]) if h[k] >= h[-1] - 0.03]
        r["min_r_within_003_of_inf"] = ok[0] if ok else None
        rec[lab] = r
    exact = {}
    for dc in range(0, 11):
        M = g["dC"] == dc
        if M.sum() < 30: continue
        h = kn[M].mean(0); ok = [rad for k, rad in enumerate(radii[:-1]) if h[k] >= h[-1] - 0.03]
        exact[str(dc)] = dict(n=int(M.sum()), hits=np.round(h, 3).tolist(), min_r_within_003_of_inf=(ok[0] if ok else None))
    rec["by_exact_d"] = exact
    out["cheb"][arm] = rec

# 팔 차 (같은 clip 위, 같은 d 구간)
def arm_val(idx, arm, lo, hi, what, cheb):
    ai = arms.index(arm); g = geo[arm]; nc, nf = g["nc"], g["nf"]
    dd = g["dC"] if cheb else g["dE"]; M = (dd[idx] >= lo) & ((dd[idx] <= hi) if cheb else (dd[idx] < hi))
    if what == "att":
        am = A[idx, ai, :nf].mean(2); return am[M][:, 0].mean() / (am[M][:, 3] * g["nobj"][idx][M] / (256 * nc)).mean()
    hit = K[idx, ai, :nf, -1][M]
    if what == "hit": return hit.mean()
    return (hit - null[arm][idx][M]).mean()


out["arm_diff"] = {}
for cheb, bl in ((False, [(2, 3), (3, 4), (4, 6), (6, 30)]), (True, [(2, 2), (3, 3), (4, 5), (6, 99)])):
    for lo, hi in bl:
        key = (f"cheb{lo}" if lo == hi else f"cheb{lo}-{hi}") if cheb else f"{lo}-{hi}"; dd = {}
        for a1, a2 in (("s2_C16", "s1_C16"), ("s4_C8", "s1_C16"), ("s4_C8", "s2_C16")):
            if a1 not in arms or a2 not in arms: continue
            dd[f"{a1}-{a2}"] = {w: cboot(lambda idx, w=w: arm_val(idx, a1, lo, hi, w, cheb) - arm_val(idx, a2, lo, hi, w, cheb), 1000)
                               for w in (("att", "hit", "hit_null") if null else ("att", "hit"))}
        out["arm_diff"][key] = dd

# 법칙·팔 고정효과 회귀 (관측 = clip×slot; log 정확한 ×균등, r=∞ 적중)
Xs, Y1, Y2, CL = [], [], [], []
for ai, arm in enumerate(arms):
    g = geo[arm]; nc, nf = g["nc"], g["nf"]; am = A[:, ai, :nf].mean(2)
    ratio = am[..., 0] / (am[..., 3] * g["nobj"] / (256 * nc)); hit = K[:, ai, :nf, -1]
    for i in range(n):
        fe = np.zeros(len(arms) + len(laws) - 1); fe[ai] = 1; li = laws.index(law[i])
        if li > 0: fe[len(arms) + li - 1] = 1
        for t in range(nf):
            Xs.append(np.concatenate([fe, [g["dE"][i, t], t + 1.0, (t + 1.0) * 2 * g["s"], g["dC"][i, t]]]))
            Y1.append(np.log(max(ratio[i, t], 1e-3))); Y2.append(hit[i, t]); CL.append(i)
Xr = np.array(Xs); Yr = np.column_stack([Y1, Y2]); CL = np.array(CL); nfe = len(arms) + len(laws) - 1
rows_by_t = {t: np.where(np.isin(CL, tix[t]))[0] for t in ut}
out["regression"] = {"note": "법칙·팔 고정효과; 종속 = log(정확한 ×균등) / r=∞ 적중; d = Euclid 칸 (dCheb 판 별도); slot = 튜블릿 (1…); rawtime = raw 프레임 (slot·2s); 궤적 bootstrap 300"}
for lab, extra in (("d+slot", [0, 1]), ("d+rawtime", [0, 2]), ("dCheb+slot", [3, 1]), ("d_only", [0])):
    cols = list(range(nfe)) + [nfe + e for e in extra]
    fit = lambda ii: np.linalg.lstsq(Xr[ii][:, cols], Yr[ii], rcond=None)[0]
    b = fit(np.arange(len(Yr))); bs = np.array([fit(np.concatenate([rows_by_t[t] for t in rng2.choice(ut, len(ut))])) for _ in range(300)])
    pr = Xr[:, cols] @ b; r2 = [round(float(1 - ((Yr[:, q] - pr[:, q]) ** 2).sum() / ((Yr[:, q] - Yr[:, q].mean()) ** 2).sum()), 3) for q in (0, 1)]
    names = ["d", "slot", "rawtime", "dCheb"]
    out["regression"][lab] = dict(R2_logatt=r2[0], R2_hit=r2[1], **{
        f"{names[e]}_{yk}": [round(float(b[nfe + j, q]), 4), *np.round(np.percentile(bs[:, nfe + j, q], [2.5, 97.5]), 4).tolist()]
        for j, e in enumerate(extra) for q, yk in ((0, "logatt"), (1, "hit"))})
out["regression"]["within_arm_corr_d_slot"] = {a: round(float(np.corrcoef(geo[a]["dE"].ravel(), geo[a]["slot"].ravel())[0, 1]), 3) for a in arms}

# 같은 Cheb d 안 이른/늦은 슬롯 (중앙값 분할; 비의 평균) · 법칙별
out["early_late_same_cheb"] = {}; out["by_law"] = {}
for ai, arm in enumerate(arms):
    g = geo[arm]; nc, nf = g["nc"], g["nf"]; am = A[:, ai, :nf].mean(2)
    ratio = am[..., 0] / (am[..., 3] * g["nobj"] / (256 * nc)); hit = K[:, ai, :nf, -1]; rec = {}
    for dc in range(1, 7):
        M = g["dC"] == dc
        if M.sum() < 60: continue
        med = np.median(g["slot"][M]); E = M & (g["slot"] <= med); L = M & (g["slot"] > med)
        if E.sum() < 20 or L.sum() < 20: continue
        rr = dict(n_early=int(E.sum()), n_late=int(L.sum()), slot_early=round(float(g["slot"][E].mean()), 2), slot_late=round(float(g["slot"][L].mean()), 2),
                  ratio_early_minus_late=cboot(lambda idx: np.nanmean(ratio[idx][E[idx]]) - np.nanmean(ratio[idx][L[idx]])),
                  hit_early_minus_late=cboot(lambda idx: np.nanmean(hit[idx][E[idx]]) - np.nanmean(hit[idx][L[idx]])))
        if dc in radii:
            k = radii.index(dc); rr["knock_r_eq_d_early_minus_late"] = cboot(lambda idx: np.nanmean(K[idx, ai, :nf, k][E[idx]]) - np.nanmean(K[idx, ai, :nf, k][L[idx]]))
        rec[f"cheb{dc}"] = rr
    out["early_late_same_cheb"][arm] = rec
    lr = {}
    for lw in laws:
        for lab, lo, hi in (("cheb2", 2, 2), ("cheb3", 3, 3), ("cheb4-5", 4, 5), ("cheb6+", 6, 99)):
            M = (g["dC"] >= lo) & (g["dC"] <= hi) & (law[:, None] == lw)
            if M.sum() >= 20:
                lr[f"{lw}|{lab}"] = dict(n=int(M.sum()), ratio_mean=round(float(np.nanmean(ratio[M])), 2), hit=round(float(hit[M].mean()), 3),
                                        hit_minus_null=(round(float((hit[M] - null[arm][M]).mean()), 3) if arm in null else None))
    out["by_law"][arm] = lr

# geometry 점검
out["geometry"] = {a: dict(disp_vs_recomputed_maxabs=round(geo[a]["disp_check"], 6), frac_clips_ctx_offframe=round(float((geo[a]["ctx_off"] > 0).mean()), 3),
                           mean_ctx_offframe_tubelets=round(float(geo[a]["ctx_off"].mean()), 2), n_ctx=geo[a]["nc"],
                           frac_clips_last_ctx_offframe=round(float(geo[a]["last_off"].mean()), 3),
                           frac_obs_future_offframe=round(float((geo[a]["fut_off"] > 0).mean()), 3)) for a in arms}
out["rinf_null_source"] = p2note

# ------------------------------------------------------------------ (2) ring 귀무 (CPU fp32 부분집합)
def rn_load(arm):
    fs = sorted(glob.glob(str(RNDIR / f"ringnull_{arm}_c*of*.json")))
    if not fs: return None, None
    R, info = {}, dict(files=[Path(f).name for f in fs], complete=True, repro=None)
    for f in fs:
        d = json.load(open(f)); info["complete"] &= d["done"] == d["of"]; info["repro"] = info["repro"] or d.get("repro")
        for r in d["rows"]: R[(r["vid"], r["slot"])] = r
    return list(R.values()), info


def rn_summary(R):
    vid = sorted(set(r["vid"] for r in R)); vix = {v: i for i, v in enumerate(vid)}; cl = np.array([vix[r["vid"]] for r in R])
    by = {k: np.where(cl == k)[0] for k in range(len(vid))}
    gt = lambda k: np.array([r.get(k, np.nan) for r in R], float)

    def cb(fn, B=1000):
        m = fn(np.arange(len(R))); bs = [fn(np.concatenate([by[k] for k in rng2.choice(len(vid), len(vid))])) for _ in range(B)]
        return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
    d, dC, dp, hit, hitg = gt("d"), gt("dC"), gt("d_papp"), gt("hit"), gt("hit_gpu")
    To, Ta, Tr, Tf, Po, Pr = (gt(k) for k in ("T_obj_mean", "T_objall_mean", "T_ring_mean", "T_refl_mean", "P_obj_mean", "P_ring_mean"))
    off = gt("n_ctx_off") > 0
    s = dict(n_clips=len(vid), n_rows=len(R), hit_cpu_vs_gpu_agreement=round(float((hit == (hitg > 0.5)).mean()), 4))
    for bname, bins, dd in (("euclid", [("0-1", 0, 1), ("1-2", 1, 2), ("2-3", 2, 3), ("3-4", 3, 4), ("4-6", 4, 6), ("6+", 6, 99)], d),
                            ("cheb", [(l, lo, hi + 1) for l, lo, hi in CB], dC)):
        tab = {}
        for lab, lo, hi in bins:
            M = (dd >= lo) & (dd < hi) & np.isfinite(To)
            if M.sum() < 10: continue
            e = dict(n=int(M.sum()), obj=cb(lambda i: np.nanmean(To[i][M[i]])), ring=cb(lambda i: np.nanmean(Tr[i][M[i]])),
                     obj_minus_ring=cb(lambda i: np.nanmean(To[i][M[i]] - Tr[i][M[i]])), refl=round(float(np.nanmean(Tf[M])), 2),
                     objall=round(float(np.nanmean(Ta[M])), 2), hit_cpu=round(float(hit[M].mean()), 3))
            for hl, hm in (("hit", hit > 0.5), ("miss", hit < 0.5)):
                mm = M & hm
                if mm.sum() >= 8:
                    e[f"{hl}_obj_ring"] = [int(mm.sum()), round(float(np.nanmean(To[mm])), 2), round(float(np.nanmean(Tr[mm])), 2)]
            # head 별: 짝수 clip 에서 obj−ring 최대 (층, head) 를 고르고 홀수 clip 에서 평가 (선택 편향 제거)
            LH = [(np.array(r["T_obj_LH"]) - np.array(r["T_ring_LH"])) if "T_obj_LH" in r else None for r in R]
            ev = M & (cl % 2 == 0); od = M & (cl % 2 == 1)
            if ev.sum() >= 5 and od.sum() >= 5:
                me = np.nanmean(np.stack([LH[j] for j in np.where(ev)[0]]), 0); l_, h_ = np.unravel_index(np.nanargmax(me), me.shape)
                vals = np.array([LH[j][l_, h_] if LH[j] is not None else np.nan for j in range(len(R))])
                e["best_head_sel_even_eval_odd"] = dict(layer=int(l_), head=int(h_), sel_even=round(float(me[l_, h_]), 2),
                                                        eval_odd=cb(lambda i: np.nanmean(vals[i][od[i]])), layer_mean_odd=round(float(np.nanmean((To - Tr)[od])), 2))
            tab[lab] = e
        s[bname] = tab
    ptab = {}
    for lab, lo, hi in (("0-1", 0, 1), ("1-2", 1, 2), ("2-3", 2, 3), ("3-4", 3, 4), ("4-6", 4, 6), ("6+", 6, 99)):
        M = (dp >= lo) & (dp < hi) & np.isfinite(Po)
        if M.sum() < 10: continue
        ptab[lab] = dict(n=int(M.sum()), obj=cb(lambda i: np.nanmean(Po[i][M[i]])), ring=cb(lambda i: np.nanmean(Pr[i][M[i]])),
                         obj_minus_ring=cb(lambda i: np.nanmean(Po[i][M[i]] - Pr[i][M[i]])))
    s["P_query_by_d_papp"] = ptab
    s["d_papp_vs_true_d"] = {lab: dict(n=int(M.sum()), d_papp_median=round(float(np.median(dp[M])), 2), d_papp_mean=round(float(np.mean(dp[M])), 2),
                                       frac_d_papp_le3=round(float(np.mean(dp[M] <= 3)), 2))
                             for lab, M in (("0-2", d < 2), ("2-3", (d >= 2) & (d < 3)), ("3-4", (d >= 3) & (d < 4)), ("4-6", (d >= 4) & (d < 6)), ("6+", d >= 6)) if M.sum() >= 10}
    s["offframe_dilution"] = dict(n_rows_with_offframe_ctx=int(off.sum()),
                                  obj_inframe_only=round(float(np.nanmean(To[off])), 2) if off.any() else None,
                                  objall_with_offframe=round(float(np.nanmean(Ta[off])), 2) if off.any() else None)
    return s


out["ring_null_cpu"] = {"dir": str(RNDIR), "note": "CPU fp32, M4 clip 중 궤적당 1 clip (기본 84), clip 군집 bootstrap = 궤적 군집. ×균등 = 층별 비의 층 평균. obj 는 화면 안 문맥 튜블릿만, objall 은 GPU 정의"}
for arm in arms:
    R, info = rn_load(arm)
    if R: out["ring_null_cpu"][arm] = dict(info=info, **rn_summary(R))

json.dump(out, open(OUT / f"m4_v3{TAG}.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ 출력
print("재현 (hook vs 원래):", out["repro"], "| P2 귀무:", p2note)
print("geometry:", json.dumps(out["geometry"]))
for arm, rec in out["arms"].items():
    print(f"\n== {arm}   (Euclid 거리 구간, 칸)  ×U: 원판 9·nc 근사 → 정확 [CI]")
    for b, r in rec.items():
        rn = r.get("rinf_hit_minus_null")
        print(f" {b:6s} n{r['n']:5d} slot{r['mean_slot']:5.2f} ×U {r['obj_traj_ratio_uniform']:5.2f} → {r['obj_traj_ratio_exact']}  last {r['last_obj_ratio_exact']} (자기기둥 겹침 {r['last_self_overlap_frac']})"
              f"  self {r['self_col_mass']:.3f} | r=∞ hit−null {rn}")
for arm, rec in out["cheb"].items():
    print(f"\n== {arm}   (Chebyshev d)")
    for b, r in rec.items():
        if b == "by_exact_d": continue
        kk = " ".join(f"{v[0]:.3f}" for v in r["knock_ci"].values())
        print(f" d{b:4s} n{r['n']:5d} slot{r['mean_slot']:5.2f} ×U {r['obj_traj_ratio_exact']} hit/miss {r.get('obj_traj_ratio_exact_hit')} {r.get('obj_traj_ratio_exact_miss')}"
              f" | knock r=1,2,3,5,8,∞ {kk} | r=d {r.get('knock_r_eq_d')} min_r {r['min_r_within_003_of_inf']} | hit {r.get('rinf_hit')} null {r.get('rinf_null')} h−n {r.get('rinf_hit_minus_null')}")
print("\narm_diff", json.dumps(out["arm_diff"]))
print("\nregression", json.dumps(out["regression"]))
print("\nearly_late", json.dumps(out["early_late_same_cheb"]))
print("\nby_law", json.dumps(out["by_law"]))
for arm in arms:
    if arm in out["ring_null_cpu"]:
        s = out["ring_null_cpu"][arm]
        print(f"\n== ring 귀무 {arm}: clip {s['n_clips']} rows {s['n_rows']} complete {s['info']['complete']} repro {s['info']['repro']} hit cpu/gpu 일치 {s['hit_cpu_vs_gpu_agreement']}")
        for bn in ("euclid", "cheb"):
            for b, e in s[bn].items():
                print(f"  {bn[:3]} {b:4s} n{e['n']:4d} obj {e['obj']} ring {e['ring']} obj−ring {e['obj_minus_ring']} refl {e['refl']} objall {e['objall']} hit {e['hit_cpu']}"
                      f" {e.get('hit_obj_ring')} {e.get('miss_obj_ring')} head {e.get('best_head_sel_even_eval_odd')}")
        print("  P-query", json.dumps(s["P_query_by_d_papp"])); print("  d_papp", json.dumps(s["d_papp_vs_true_d"])); print("  offframe", s["offframe_dilution"])
