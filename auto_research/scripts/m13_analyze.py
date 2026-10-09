#!/usr/bin/env python3
"""M1 · M3 분석 (v3, C16/P32). 입력 v3_m13/. 계획: Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md §2–3.

M1  팔마다 슬롯별  Δp = mean|p_arm − p_full| / mean|p_full|,  적중 − 귀무 (템플릿 = h 진실 칸, Chebyshev ≤ 1; 귀무 = 같은 법칙 다른 궤적의 진실 칸),
    물체다움 (contrast ≥ 0.357).  CI = 궤적 군집 bootstrap.
M3  문맥 끝 속도 v = (pos[31] − pos[27]) / 4 (px/프레임, x·y) 를 특징에서 능형 회귀로 — 궤적 단위 group CV (6 겹, 안쪽 λ 선택).
    특징: z_last, h_last (문맥 마지막 물체 창), h_obj_t (진실 창, 천장), p_obj_t (p 가 스스로 둔 물체 자리), 대조: p 봉우리 위치 (y, x) 만.
    등변성: cos(h_obj_t, h_last) 대 cos(h_obj_t, 다른 궤적 h_last) ;  cos(p_obj_t, h_last).

v2 (2026-09-25, 적대 검증 Archive/VERIFY_0925_2026-09-25.md B18–B21 반영) — 기존 키·값은 그대로, 새 키만 더한다 (*_v2):
  M1_v2          팔 × 슬롯 적중 · 전수 귀무 (같은 법칙 · 다른 궤적 전부) · 적중−귀무 · 물체다움 (궤적 CI 2000),
                 팔 − full 짝 차이, hist_swap 진단 (clip 별 같은 칸 · 외형 잡음 바닥 · donor 속도 추종 · 장면 절단 비율).
  M3_v2          법칙 / (법칙×secondary) one-hot 기준선, 그룹 안 부분 R² (vx, vy, |v|), leave-one-speed-out, 셔플;
                 [B18] 봉우리 절대 위치 대조 → 마지막 관측 칸 기준 변위/경과 프레임 (진실 · p 두 템플릿) + 그룹 안 적분 이득 (기울기, 궤적 CI).
                 (선택) v3_h23 캐시가 있으면 물체 없는 창 (p@W_T, h_t@W_T) 의 속도 R² — 물체 특이성 대조.
  sufficiency_v2 [B19] z_cv / true_cv / stay / 라벨 평균 속도 (lab_cv, law_cv) / p 모두 전수 귀무 · 적중−귀무 · 궤적 CI,
                 z_cv − p, z_cv − lab_cv (z 고유 기여), 법칙별 표 (arc 역전), [B21] 중심 맞춘 판 (*_c: 위치 프레임 29 = 속도 차분 중심).
  equivariance_v2 cos(h_obj_t, h_last_j) 를 (같은 궤적 · 같은 외형) 2×2 로.
  표적: 모든 h (LOC 진실 칸 템플릿 · p_peak 외형 템플릿 · h_last · h_obj) 는 문맥 16 + 미래 32 프레임을 본 **표준 양방향** target encoder. 인과 표적 판 없음.
  환경변수: OUT_TAG (출력 접미사), M13_OUT_DIR (출력 폴더, 기본 exp_results/m13), M13_H23_DIR (h23 캐시, 기본 입력 폴더 이름에서 유도).
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m13")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("M13_OUT_DIR", str(ROOT / "auto_research/exp_results/m13"))); OUT.mkdir(parents=True, exist_ok=True)
CELL, THR, SPLIT, K = 18.0, 0.357, 32, 16
meta = json.load(open(I / "meta.json")); A = meta["arms"]; ids = meta["video_ids"]; n = len(ids)
LOC = np.load(I / "loc.npy"); DP = np.load(I / "dp.npy")
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
f = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([f(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([f(idx[v]["px_y_by_sample"]) for v in ids])
INF = np.stack([f(idx[v]["in_frame_by_sample"]) for v in ids]) > 0.5
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); scen = np.array([t[0] for t in meta["traj"]])
rng = np.random.default_rng(0); ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
other = np.array([rng.choice(np.where((scen == scen[i]) & (traj != traj[i]))[0] if ((scen == scen[i]) & (traj != traj[i])).any() else np.delete(np.arange(n), i)) for i in range(n)])


def boot(v, B=300):
    m = np.nanmean(v); bs = [np.nanmean(v[np.concatenate([tix[t] for t in rng.choice(ut, len(ut))])]) for _ in range(B)]
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


out = {"M1": {}, "M3": {}}
# ---------------- M1
for ai, arm in enumerate(A):
    rec = []
    for t in range(K):
        g0, g1 = SPLIT + 2 * t, SPLIT + 2 * t + 1
        ct = np.clip(np.floor(np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL), 0, 15)
        ok = INF[:, g0] & INF[:, g1]
        am = LOC[:, ai, t, [1, 0]]
        hit = (np.abs(am - ct).max(-1) <= 1).astype(float); nul = (np.abs(am - ct[other]).max(-1) <= 1).astype(float)
        obj = ((LOC[:, ai, t, 2] - LOC[:, ai, t, 3]) >= THR).astype(float)
        for v in (hit, nul, obj): v[~ok] = np.nan
        rec.append(dict(t=t, dp=boot(DP[:, ai, t]), hit=boot(hit), null=boot(nul), hit_null=boot(hit - nul), obj=boot(obj)))
    out["M1"][arm] = rec
print("M1 — Δp (상대) / 적중−귀무 / 물체다움, 슬롯 0 2 4 6 8 10 12 14")
for arm in A:
    r = out["M1"][arm]
    print(f"{arm:11s} Δp " + " ".join(f"{r[t]['dp'][0]:.2f}" for t in range(0, K, 2)) + " | h−n " + " ".join(f"{r[t]['hit_null'][0]:+.2f}" for t in range(0, K, 2))
          + " | obj " + " ".join(f"{r[t]['obj'][0]:.2f}" for t in range(0, K, 2)))
# ---------------- M3
P = np.load(I / "p_obj.npy").astype(np.float32); H = np.load(I / "h_obj.npy").astype(np.float32)
Z = np.load(I / "z_last.npy").astype(np.float32); HL = np.load(I / "h_last.npy").astype(np.float32) if (I / "h_last.npy").exists() else None
PK = np.load(I / "p_peak.npy")
v = np.stack([(X[:, 31] - X[:, 27]) / 4, (Y[:, 31] - Y[:, 27]) / 4], -1)
folds = {t: k % 6 for k, t in enumerate(rng.permutation(ut))}; fo = np.array([folds[t] for t in traj])
LAMS = 10.0 ** np.arange(-2, 5)


def _svd(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xs = (Xtr - mu) / sd; ym = ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    return mu, sd, ym, s, Vt, U.T @ (ytr - ym)


def _pred(fit, Xte, lam):
    mu, sd, ym, s, Vt, UtY = fit
    return ((Xte - mu) / sd) @ (Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)) + ym


def cv_r2(F, y, return_pred=False):
    """궤적 group CV (6 겹), 안쪽 3 겹으로 λ 선택. SVD 는 (겹마다) 한 번만."""
    F = F.astype(np.float64); pred = np.zeros_like(y)
    for k in range(6):
        tr, te = fo != k, fo == k
        Ftr, ytr, ttr = F[tr], y[tr], traj[tr]
        sc = np.zeros(len(LAMS))
        for kk in range(3):
            trf = np.array([folds[t] % 3 != kk for t in ttr]); fit = _svd(Ftr[trf], ytr[trf])
            for li, lam in enumerate(LAMS): sc[li] += ((_pred(fit, Ftr[~trf], lam) - ytr[~trf]) ** 2).mean()
        pred[te] = _pred(_svd(Ftr, ytr), F[te], LAMS[int(np.argmin(sc))])
    r2 = [round(float(1 - ((pred[:, j] - y[:, j]) ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum()), 3) for j in range(2)]
    return (r2, pred) if return_pred else r2


out["M3"]["z_last"], vhat = cv_r2(Z, v, return_pred=True)
# ---------------- encoder 충분성: z 에서 읽은 속도 (궤적 group CV 예측) 로 문맥 끝 위치에서 등속 외삽 → 미래 슬롯 적중
#   위치: 문맥 마지막 튜블릿 (프레임 30, 31) 의 **진실 px** (oracle — encoder 로부터 위치를 읽지 않았다).  비교: p (M1 full 팔), 진짜 속도 등속 (상한), 제자리 (복사)
#   ⚠️ 이 블록의 null_p 는 p 에만, 한 번 뽑은 귀무다. 전수 귀무 · 라벨 평균 기준선 · 법칙별은 아래 sufficiency_v2.
pl = np.stack([(X[:, 30] + X[:, 31]) / 2, (Y[:, 30] + Y[:, 31]) / 2], -1)
suff = {k: [] for k in ("z_cv", "true_cv", "stay", "p_full", "null_p")}
for t in range(K):
    g0, g1 = SPLIT + 2 * t, SPLIT + 2 * t + 1; dtf = (g0 + g1) / 2 - 30.5
    ct = np.clip(np.floor(np.stack([(X[:, g0] + X[:, g1]) / 2, (Y[:, g0] + Y[:, g1]) / 2], -1) / CELL), 0, 15)
    ok = INF[:, g0] & INF[:, g1]
    hit = lambda xy: np.where(ok, (np.abs(np.clip(np.floor(xy / CELL), 0, 15) - ct).max(-1) <= 1).astype(float), np.nan)
    suff["z_cv"].append(boot(hit(pl + vhat * dtf))); suff["true_cv"].append(boot(hit(pl + v * dtf))); suff["stay"].append(boot(hit(pl)))
    am = LOC[:, 0, t, [1, 0]]
    suff["p_full"].append(boot(np.where(ok, (np.abs(am - ct).max(-1) <= 1).astype(float), np.nan)))
    suff["null_p"].append(boot(np.where(ok, (np.abs(am - ct[other]).max(-1) <= 1).astype(float), np.nan)))
out["sufficiency"] = suff
print("\n충분성 — 미래 슬롯 적중 (1 칸 안), 슬롯 0 2 4 6 8 10 12 14")
for k_, v_ in suff.items():
    print(f"  {k_:8s} " + " ".join(f"{v_[t][0]:.2f}" for t in range(0, K, 2)))
if HL is not None: out["M3"]["h_last"] = cv_r2(HL, v)
out["M3"]["slots"] = {}
for t in (0, 2, 4, 6, 8, 10, 12, 14):
    # ⚠️ p_peakpos (봉우리 절대 칸 → 속도) 는 적분을 판별하지 못하는 무효 대조다 (B18: 진실 칸 절대 위치도 t0 에서 같은 R²). 키는 호환 때문에 남긴다 — 해석은 M3_v2.displacement / gain.
    r = dict(h_obj=cv_r2(H[:, t], v), p_obj=cv_r2(P[:, t], v), p_peakpos=cv_r2(PK[:, t, :2], v),
             p_obj_plus_pos=cv_r2(np.concatenate([P[:, t], PK[:, t, :2] * 50], 1), v))
    if HL is not None:
        c = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
        r["eq_h_same"] = round(float(np.mean(c(H[:, t], HL))), 3); r["eq_h_other"] = round(float(np.mean(c(H[:, t], HL[other]))), 3)
        r["eq_p_same"] = round(float(np.mean(c(P[:, t], HL))), 3); r["eq_p_other"] = round(float(np.mean(c(P[:, t], HL[other]))), 3)
    out["M3"]["slots"][t] = r
print("\nM3 — 문맥 끝 속도 R² (vx, vy), 궤적 group CV:  z_last", out["M3"]["z_last"], " h_last", out["M3"].get("h_last"))
for t, r in out["M3"]["slots"].items():
    print(f" t{t:2d}  h_obj {r['h_obj']}  p_obj {r['p_obj']}  p봉우리위치 {r['p_peakpos']}  p_obj+위치 {r['p_obj_plus_pos']}"
          + (f" | cos(h_t,h_last) 같은 {r['eq_h_same']} 다른 {r['eq_h_other']}  cos(p_t,h_last) 같은 {r['eq_p_same']} 다른 {r['eq_p_other']}" if "eq_h_same" in r else ""))
# =====================================================================================================================
# v2 — 적대 검증 (2026-09-25) 반영. 위의 rng 를 쓰지 않는다 (기존 키의 값·CI 가 그대로 남는다).
# =====================================================================================================================
R2G = np.random.default_rng(12345); NB = 2000
r3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
law = np.array([str(t[0]) for t in meta["traj"]]); prim = np.array([str(t[1]) for t in meta["traj"]]); sec = np.array([str(t[2]) for t in meta["traj"]])
grp = np.array([f"{a}|{c}" for a, c in zip(law, sec)])
_, tinv = np.unique(traj, return_inverse=True); NT = len(ut)
tixl = [np.where(tinv == k)[0] for k in range(NT)]; trep = np.array([ix[0] for ix in tixl]); tlaw = law[trep]
env = np.array([idx[v_]["env"] for v_ in ids]); shp = np.array([idx[v_]["shape_pre"] for v_ in ids]); col = np.array([idx[v_]["color_pre"] for v_ in ids])
iF = A.index("full")
chk = dict(n=n, n_traj=NT, px_max_dev_within_traj=r3(max(np.abs(X - X[trep][tinv]).max(), np.abs(Y - Y[trep][tinv]).max())),
           future_all_in_frame=bool(INF[:, SPLIT:SPLIT + 2 * K].all()))
print("\n[v2] 점검", chk)


def cb(vals, keep=None):
    """궤적 군집 bootstrap (NB) — nanmean 과 95 % CI. keep 이 있으면 그 clip 만 (그 clip 이 있는 궤적만 재표집)."""
    vals = np.asarray(vals, float)
    if keep is not None: vals = np.where(keep, vals, np.nan)
    s = np.array([np.nansum(vals[ix]) for ix in tixl]); c = np.array([np.sum(~np.isnan(vals[ix])) for ix in tixl], float)
    live = np.where(c > 0)[0]
    if len(live) == 0: return [None, None, None]
    s, c = s[live], c[live]; bs = R2G.integers(0, len(live), size=(NB, len(live))); bm = s[bs].sum(1) / c[bs].sum(1)
    return [r3(s.sum() / c.sum()), r3(np.percentile(bm, 2.5)), r3(np.percentile(bm, 97.5))]


def tcell(t):
    g0 = SPLIT + 2 * t
    return np.clip(np.floor(np.stack([(X[:, g0] + X[:, g0 + 1]) / 2, (Y[:, g0] + Y[:, g0 + 1]) / 2], -1) / CELL), 0, 15)


LC = np.clip(np.floor(pl / CELL), 0, 15)                                                     # 문맥 마지막 튜블릿 진실 칸 (x, y)
NM = (tlaw[None, :] == law[:, None]) & (np.arange(NT)[None, :] != tinv[:, None])            # (n, NT) 같은 법칙 · 다른 궤적
tocell = lambda xy: np.clip(np.floor(xy / CELL), 0, 15)


def hn(cell, t):
    """적중 (Chebyshev ≤ 1) 과 전수 귀무 (같은 법칙 · 다른 궤적 전부의 진실 칸에 대한 적중률)."""
    tc = tcell(t); ok = INF[:, SPLIT + 2 * t] & INF[:, SPLIT + 2 * t + 1]
    hit = (np.abs(cell - tc).max(-1) <= 1).astype(float)
    nul = ((np.abs(cell[:, None, :] - tc[trep][None]).max(-1) <= 1) & NM).sum(1) / np.maximum(NM.sum(1), 1)
    hit[~ok] = np.nan; nul = np.where(ok, nul, np.nan)
    return hit, nul


# ---------------- M1_v2
objA = lambda ai, t: ((LOC[:, ai, t, 2] - LOC[:, ai, t, 3]) >= THR).astype(float)
M1v = dict(null="exhaustive: 같은 법칙 · 다른 궤적 전부 (clip 마다 평균)", ci="궤적 군집 bootstrap 2000", arms={}, paired_vs_full={})
HNd = {}
for ai, arm in enumerate(A):
    rows = []
    for t in range(K):
        h_, n_ = hn(LOC[:, ai, t][:, [1, 0]], t); o_ = objA(ai, t); HNd[arm, t] = (h_, h_ - n_, o_)
        rows.append(dict(t=t, hit=cb(h_), null_exh=cb(n_), hit_null_exh=cb(h_ - n_), obj=cb(o_)))
    M1v["arms"][arm] = rows
for arm in A:
    if arm == "full": continue
    M1v["paired_vs_full"][arm] = [dict(t=t, d_hit_null_exh=cb(HNd[arm, t][1] - HNd["full", t][1]), d_obj=cb(HNd[arm, t][2] - HNd["full", t][2]))
                                  for t in range(K)]
if "hist_swap" in A and "donor" in meta:
    iS = A.index("hist_swap"); don = meta["donor"]
    dX = np.stack([f(idx[d]["px_x_by_sample"]) for d in don]); dY = np.stack([f(idx[d]["px_y_by_sample"]) for d in don])
    denv = np.array([idx[d]["env"] for d in don]); dshp = np.array([idx[d]["shape_pre"] for d in don]); dcol = np.array([idx[d]["color_pre"] for d in don])
    dtraj = np.array(["|".join([idx[d]["scenario"], idx[d]["primary"], idx[d]["secondary"]]) for d in don])
    c2 = lambda XX, YY, f0: np.clip(np.floor(np.stack([(XX[:, f0] + XX[:, f0 + 1]) / 2, (YY[:, f0] + YY[:, f0 + 1]) / 2], -1) / CELL), 0, 15)
    jg = np.abs(c2(dX, dY, 26) - c2(X, Y, 28)).max(-1)                                        # 이음매: donor 튜블릿 5 끝 vs 자기 튜블릿 6
    vd = np.stack([(dX[:, 31] - dX[:, 27]) / 4, (dY[:, 31] - dY[:, 27]) / 4], -1); dvn = np.linalg.norm(vd - v, axis=-1)
    mate = np.array([R2G.choice([j for j in tixl[tinv[i]] if j != i]) for i in range(n)])  # 같은 궤적 · 다른 외형 clip (외형 잡음 바닥)
    diag = dict(donor_same_law=r3((np.array([idx[d]["scenario"] for d in don]) == law).mean()), donor_diff_traj=r3((dtraj != traj).mean()),
                donor_env_mismatch=r3((denv != env).mean()), donor_env_shape_color_all_same=r3(((denv == env) & (dshp == shp) & (dcol == col)).mean()),
                junction_gap_ge2=r3((jg >= 2).mean()), junction_gap_median=r3(np.median(jg)), dv_lt_0p5=r3((dvn < 0.5).mean()), dv_median=r3(np.median(dvn)), slots=[])
    for t in range(K):
        amf = LOC[:, iF, t][:, [1, 0]]; ams = LOC[:, iS, t][:, [1, 0]]; d_ = np.abs(amf - ams).max(-1); fl = np.abs(amf - amf[mate]).max(-1)
        dtf = (SPLIT + 2 * t + 0.5) - 30.5; ps = (vd - v) * dtf / CELL; nr = np.linalg.norm(ps, axis=-1)
        prj = np.where(nr > 0.5, ((ams - amf) * ps).sum(-1) / np.maximum(nr, 1e-9) ** 2, np.nan)
        same = (d_ == 0).astype(float)
        diag["slots"].append(dict(t=t, same_cell=cb(same), within1=cb((d_ <= 1).astype(float)),
                                  floor_same_cell=cb((fl == 0).astype(float)), floor_within1=cb((fl <= 1).astype(float)),
                                  donor_shift_realised=cb(prj), n_donor_shift=int(np.isfinite(prj).sum()),
                                  same_cell_donor_same_env=cb(same, denv == env), same_cell_donor_diff_env=cb(same, denv != env),
                                  same_cell_jgap_le1=cb(same, jg <= 1), same_cell_jgap_ge3=cb(same, jg >= 3)))
    M1v["hist_swap_diag"] = diag
out["M1_v2"] = M1v
print("\n[v2] M1 — 적중−전수귀무 [CI] (슬롯 4 6 8 10) · 팔−full 짝 차이")
for arm in A:
    r_ = M1v["arms"][arm]
    print(f"  {arm:11s} " + "  ".join(f"t{t}:{r_[t]['hit_null_exh'][0]:+.3f}[{r_[t]['hit_null_exh'][1]:+.3f},{r_[t]['hit_null_exh'][2]:+.3f}]" for t in (4, 6, 8, 10))
          + ("" if arm == "full" else "  | Δ " + " ".join(f"t{t}:{M1v['paired_vs_full'][arm][t]['d_hit_null_exh']}" for t in (4, 6, 8, 10))
             + "  Δobj " + " ".join(f"t{t}:{M1v['paired_vs_full'][arm][t]['d_obj']}" for t in (4, 6))))
if "hist_swap_diag" in M1v:
    dg = M1v["hist_swap_diag"]; print("  hist_swap 진단", {k_: v_ for k_, v_ in dg.items() if k_ != "slots"})
    for r_ in dg["slots"][::2]: print("   ", {k_: r_[k_] for k_ in ("t", "same_cell", "within1", "floor_same_cell", "donor_shift_realised", "same_cell_donor_same_env", "same_cell_donor_diff_env")})

# ---------------- M3_v2 : 능형 (SVD), 궤적 group CV, 안쪽 3 겹 λ 를 열마다 고른다
LAM2 = 10.0 ** np.arange(-2, 7)
spd = np.linalg.norm(v, axis=-1)
V3 = np.stack([v[:, 0], v[:, 1], spd], -1)                                                   # 열: vx, vy, |v|


def _fitpred(Atr, ytr, Ate):
    mu, sd = Atr.mean(0), Atr.std(0) + 1e-6; A_ = (Atr - mu) / sd; B_ = (Ate - mu) / sd; ym = ytr.mean(0)
    U, s, Vt = np.linalg.svd(A_, full_matrices=False); UtY = U.T @ (ytr - ym); BV = B_ @ Vt.T
    return np.stack([BV @ ((s / (s ** 2 + l))[:, None] * UtY) + ym for l in LAM2])            # (L, n_te, c)


def _folds(gid, nf):
    if gid is None: return fo
    ug = np.unique(gid); pm = np.random.default_rng(7).permutation(ug); fm = {g: k % nf for k, g in enumerate(pm)}
    return np.array([fm[g] for g in gid])


def cvp(Fe, y, key=None, gid=None, nf=6):
    """out-of-fold 예측. key 가 있으면 train-fold 의 key 그룹 평균을 특징·표적에서 빼고 잔차를 맞춘다 (그룹 안 부분).
    반환 R² (열마다). pooled: 1 − SSE/Σ(y−ȳ)², 그룹 안: 1 − SSE/Σ(잔차)² (= 그룹 평균 기준선 대비)."""
    Fe = np.asarray(Fe, np.float64); y = np.asarray(y, np.float64).reshape(len(Fe), -1); fo_ = _folds(gid, nf)
    pred = np.zeros_like(y); yt = y.copy()
    for k in np.unique(fo_):
        tr, te = fo_ != k, fo_ == k; Fk, yk = Fe.copy(), y.copy()
        if key is not None:
            for g in np.unique(key):
                m = tr & (key == g); src = m if m.any() else tr
                Fk[key == g] -= Fe[src].mean(0); yk[key == g] -= y[src].mean(0)
        tf = np.unique(fo_[tr]); im = {g: j % 3 for j, g in enumerate(tf)}; inn = np.array([im[g] for g in fo_[tr]])
        Ftr, ytr = Fk[tr], yk[tr]; sc = np.zeros((len(LAM2), y.shape[1]))
        for kk in range(3):
            a_, b_ = inn != kk, inn == kk
            sc += ((_fitpred(Ftr[a_], ytr[a_], Ftr[b_]) - ytr[b_][None]) ** 2).sum(1)
        Pk = _fitpred(Ftr, ytr, Fk[te]); li = sc.argmin(0)
        pred[te] = np.stack([Pk[li[j], :, j] for j in range(y.shape[1])], -1); yt[te] = yk[te]
    den = ((yt ** 2).sum(0) if key is not None else ((yt - yt.mean(0)) ** 2).sum(0))
    return [r3(1 - ((pred[:, j] - yt[:, j]) ** 2).sum() / max(den[j], 1e-12)) for j in range(y.shape[1])]


def cv_groupmean(y, key):
    """라벨 기준선: train-fold 의 같은 key 그룹 평균 (특징 없음)."""
    pred = np.zeros_like(y)
    for k in np.unique(fo):
        tr, te = fo != k, fo == k
        for g in np.unique(key[te]):
            m = tr & (key == g); pred[te & (key == g)] = y[m].mean(0) if m.any() else y[tr].mean(0)
    return pred


r2c = lambda pr, y: [r3(1 - ((pr[:, j] - y[:, j]) ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum()) for j in range(y.shape[1])]
rs = np.random.default_rng(99); vperm = V3.copy()
for L_ in np.unique(law):
    ks = np.where(tlaw == L_)[0]; pk_ = rs.permutation(ks)
    for a_, b_ in zip(ks, pk_): vperm[tinv == a_] = V3[trep[b_]]
vfull = V3[trep[rs.permutation(NT)]][tinv]
nprim = len(np.unique(prim))


def wshare(key):
    gm = np.zeros_like(V3)
    for g in np.unique(key): gm[key == g] = V3[key == g].mean(0)
    return [r3(((V3[:, j] - gm[:, j]) ** 2).sum() / ((V3[:, j] - V3[:, j].mean()) ** 2).sum()) for j in range(3)]


M3v = dict(cols=["vx", "vy", "|v|"], target_def="v = (pos[31] − pos[27]) / 4 px/frame (중심 프레임 29)",
           within_var_share={"law": wshare(law), "law|sec": wshare(grp)},
           baselines={"law_onehot": r2c(cv_groupmean(V3, law), V3), "law|sec_onehot": r2c(cv_groupmean(V3, grp), V3)}, feats={})


def m3run(name, Fe, full=True):
    d = dict(pooled=cvp(Fe, V3))
    if full:
        d["within_law|sec"] = cvp(Fe, V3, key=grp)
        d["leave_one_speed_out"] = cvp(Fe, V3, gid=prim, nf=nprim)
        d["within_law_label_shuffle"] = cvp(Fe, vperm)
        d["full_label_shuffle"] = cvp(Fe, vfull)
    M3v["feats"][name] = d; print(f"  {name:28s}", d, flush=True)


print("\n[v2] M3 — 속도 R² (vx, vy, |v|): pooled / 그룹 안 (법칙×secondary) / LOSO / 셔플.  기준선", M3v["baselines"], " 그룹 안 분산 비율", M3v["within_var_share"])
m3run("z_last", Z)
if HL is not None: m3run("h_last", HL)
for t in (0, 4, 8, 10, 14): m3run(f"p_obj_t{t}", np.asarray(P[:, t], np.float32))
for t in (0, 8, 14): m3run(f"h_obj_t{t}", np.asarray(H[:, t], np.float32), full=(t != 8))
# [B18] 위치 대조: 절대 칸 (무효 대조 재현) vs 마지막 관측 칸 기준 변위 / 경과 프레임
M3v["displacement"] = {}
print("\n[v2] B18 — 변위/경과 프레임 → 속도 R² (pooled vx, vy · 그룹 안 |v|).  abs_* = 절대 칸 (무효 대조)")
for t in range(0, K, 2):
    dtf = (SPLIT + 2 * t + 0.5) - 30.5; tc = tcell(t); pk_ = PK[:, t, :2][:, [1, 0]]; am = LOC[:, iF, t][:, [1, 0]]
    row = {}
    for nm, cell in (("true", tc), ("p_app_tpl", pk_), ("p_truth_tpl", am)):
        dsp = (cell - LC) * CELL / dtf
        row[f"abs_{nm}"] = cvp(cell, V3[:, :2])
        row[f"disp_{nm}"] = cvp(dsp, V3[:, :2]); row[f"disp_{nm}_within_speed"] = cvp(dsp, spd, key=grp)
    M3v["displacement"][t] = row; print(f"  t{t:2d}", row, flush=True)
# 그룹 안 적분 이득: 진실 변위 (칸, 법칙×secondary 평균 제거) 에 대한 p 자리 변위의 기울기 (x·y 합쳐), 궤적 CI


def gdm(a, m=None):
    a = a.astype(float).copy()
    for g in np.unique(grp):
        s_ = (grp == g) if m is None else ((grp == g) & m)
        if s_.any(): a[grp == g] -= a[s_].mean(0)
    return a


def gain(td, pd_, m=None):
    keep = np.ones(n, bool) if m is None else m
    sxy = np.array([(td[ix][keep[ix]] * pd_[ix][keep[ix]]).sum() for ix in tixl]); sxx = np.array([(td[ix][keep[ix]] ** 2).sum() for ix in tixl])
    live = np.where(sxx > 0)[0]
    if len(live) < 5: return [None, None, None]
    sxy, sxx = sxy[live], sxx[live]; bs = R2G.integers(0, len(live), size=(NB, len(live))); b = sxy[bs].sum(1) / sxx[bs].sum(1)
    return [r3(sxy.sum() / sxx.sum()), r3(np.percentile(b, 2.5)), r3(np.percentile(b, 97.5))]


M3v["gain"] = []
print("\n[v2] 그룹 안 적분 이득 (p 변위 / 진실 변위, 칸)  진실 템플릿 argmax · 외형 템플릿 봉우리 · 물체다움 통과 clip · 진행률 중앙값")
for t in range(K):
    tc = tcell(t); td = gdm(tc - LC); am = LOC[:, iF, t][:, [1, 0]]; pk_ = PK[:, t, :2][:, [1, 0]]
    ob = (LOC[:, iF, t, 2] - LOC[:, iF, t, 3]) >= THR
    g_ob = gain(gdm(tc - LC, ob), gdm(am - LC, ob), ob) if ob.sum() >= 30 else [None, None, None]
    tdr = (tc - LC).astype(float); okr = (tdr ** 2).sum(-1) >= 4
    prg = lambda cell: r3(np.median(((cell - LC) * tdr).sum(-1)[okr] / (tdr ** 2).sum(-1)[okr])) if okr.any() else None
    r_ = dict(t=t, gain_truth_tpl=gain(td, gdm(am - LC)), gain_app_tpl=gain(td, gdm(pk_ - LC)), gain_truth_tpl_objpass=g_ob, n_objpass=int(ob.sum()),
              progress_median_truth_tpl=prg(am), progress_median_app_tpl=prg(pk_), n_progress=int(okr.sum()),
              p_app_within1_of_last=r3((np.abs(pk_ - LC).max(-1) <= 1).mean()), true_within1_of_last=r3((np.abs(tc - LC).max(-1) <= 1).mean()),
              app_tpl_obj=r3(((PK[:, t, 2] - PK[:, t, 3]) >= THR).mean()))
    M3v["gain"].append(r_)
    print(f"  t{t:2d} 진실템플릿 {r_['gain_truth_tpl']}  외형템플릿 {r_['gain_app_tpl']}  물체다움통과 {g_ob} (n {int(ob.sum())})  진행률 {r_['progress_median_truth_tpl']}/{r_['progress_median_app_tpl']}", flush=True)
# (선택) 물체 없는 창 대조 — v3_h23 캐시 (같은 창 C16/P32, 같은 릴리즈 추출) 가 있을 때만
h23d = os.environ.get("M13_H23_DIR") or (str(I.parent / ("v3_h23" + I.name[len("v3_m13"):])) if I.name.startswith("v3_m13") else "")
if h23d and (Path(h23d) / "vec_C16_P32.npy").exists():
    m23 = json.load(open(Path(h23d) / "meta.json")); hid = {v_: k for k, v_ in enumerate(m23["video_ids"])}
    if all(v_ in hid for v_ in ids):
        rows23 = np.array([hid[v_] for v_ in ids]); od = np.argsort(rows23); iv = np.argsort(od); sl23 = m23["vec_slots"]
        V23 = np.load(Path(h23d) / "vec_C16_P32.npy", mmap_mode="r"); get = lambda t, s: np.asarray(V23[rows23[od], t, sl23.index(s)], np.float32)[iv]
        csn = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
        xc = dict(dir=h23d, cos_z_last_vs_lnz_T=r3(csn(Z, get(0, "lnz_T@WT")).mean()), cos_h_obj_t8_vs_h_t_Wt=r3(csn(np.asarray(H[:, 8], np.float32), get(8, "h_t@Wt")).mean()), feats={})
        print("\n[v2] 물체 없는 창 대조 (v3_h23):  @W_T = 문맥 마지막 관측 칸 3×3 (먼 슬롯에서 물체는 떠났다),  @W_t = 진실 칸", {k_: xc[k_] for k_ in ("cos_z_last_vs_lnz_T", "cos_h_obj_t8_vs_h_t_Wt")})
        for t in (0, 8, 14):
            for s in ("p_L11@WT", "p_L11@Wt", "h_t@WT", "p_L0@WT"):
                Fe = get(t, s); d = dict(pooled=cvp(Fe, V3), within=cvp(Fe, V3, key=grp)); xc["feats"][f"{s}_t{t}"] = d; print(f"  t{t:2d} {s:9s}", d, flush=True)
        M3v["nonobject_xcheck_h23"] = xc
out["M3_v2"] = M3v

# ---------------- sufficiency_v2 [B19, B21]
vg = cv_groupmean(v, grp); vl = cv_groupmean(v, law)
p29 = np.stack([X[:, 29], Y[:, 29]], -1)
S2 = dict(position="oracle — 문맥 마지막 튜블릿 진실 px (프레임 30–31 평균; *_c 는 프레임 29 = 속도 차분 중심)", velocity=dict(
    z_cv="z_last 에서 궤적 group CV 능형으로 읽은 v (vhat, 위 M3 와 같음)", lab_cv="train-fold (법칙×secondary) 평균 v — encoder 없음", law_cv="train-fold 법칙 평균 v"),
    null="exhaustive", slots=[], per_law={}, excl_arc={})
cand = lambda t: dict(z_cv=pl + vhat * ((SPLIT + 2 * t + 0.5) - 30.5), true_cv=pl + v * ((SPLIT + 2 * t + 0.5) - 30.5), stay=pl,
                      lab_cv=pl + vg * ((SPLIT + 2 * t + 0.5) - 30.5), law_cv=pl + vl * ((SPLIT + 2 * t + 0.5) - 30.5),
                      z_cv_c=p29 + vhat * ((SPLIT + 2 * t + 0.5) - 29), true_cv_c=p29 + v * ((SPLIT + 2 * t + 0.5) - 29))
print("\n[v2] 충분성 — 적중 / 전수귀무 / 적중−귀무 [CI]")
HS = {}
for t in range(K):
    row = dict(t=t)
    for nm, xy in cand(t).items():
        h_, n_ = hn(tocell(xy), t); HS[nm, t] = (h_, n_); row[nm] = dict(hit=cb(h_), null_exh=cb(n_), hit_null_exh=cb(h_ - n_))
    h_, n_ = hn(LOC[:, iF, t][:, [1, 0]], t); HS["p_full", t] = (h_, n_); row["p_full"] = dict(hit=cb(h_), null_exh=cb(n_), hit_null_exh=cb(h_ - n_))
    row["z_cv_minus_p_hit_null"] = cb((HS["z_cv", t][0] - HS["z_cv", t][1]) - (h_ - n_))
    row["z_cv_minus_lab_cv_hit"] = cb(HS["z_cv", t][0] - HS["lab_cv", t][0])
    S2["slots"].append(row)
    if t % 2 == 0:
        print(f"  t{t:2d} " + "  ".join(f"{nm}:{row[nm]['hit'][0]:.2f}/{row[nm]['null_exh'][0]:.2f}/{row[nm]['hit_null_exh'][0]:+.2f}" for nm in ("z_cv", "true_cv", "lab_cv", "stay", "p_full"))
              + f"  z−p(h−n) {row['z_cv_minus_p_hit_null']}  z−lab {row['z_cv_minus_lab_cv_hit']}", flush=True)
for t in (6, 8, 10, 14):
    S2["per_law"][t] = {}
    for L_ in sorted(np.unique(law)):
        m = law == L_
        S2["per_law"][t][L_] = dict(z_cv=cb(HS["z_cv", t][0], m), true_cv=r3(np.nanmean(HS["true_cv", t][0][m])), lab_cv=r3(np.nanmean(HS["lab_cv", t][0][m])),
                                    z_cv_null=r3(np.nanmean(HS["z_cv", t][1][m])), p=cb(HS["p_full", t][0], m), p_null=r3(np.nanmean(HS["p_full", t][1][m])))
    m = law != "arc"
    S2["excl_arc"][t] = dict(z_cv=cb(HS["z_cv", t][0], m), p=cb(HS["p_full", t][0], m))
    print(f"  법칙별 t{t}: " + " | ".join(f"{L_} z{d_['z_cv'][0]:.2f} tcv{d_['true_cv']:.2f} lab{d_['lab_cv']:.2f} p{d_['p'][0]:.2f} pnull{d_['p_null']:.2f}" for L_, d_ in S2["per_law"][t].items()))
out["sufficiency_v2"] = S2

# ---------------- equivariance_v2 : cos(h_obj_t, h_last_j) 를 같은 궤적 × 같은 외형 (shape, color, env) 2×2 로
if HL is not None:
    hl = HL / (np.linalg.norm(HL, axis=-1, keepdims=True) + 1e-9); app = np.array([f"{a}|{b}|{c}" for a, b, c in zip(shp, col, env)])
    ST = traj[:, None] == traj[None]; SA = app[:, None] == app[None]; E_ = np.eye(n, dtype=bool)
    EQ = {}
    for t in (0, 8, 14):
        ht = np.asarray(H[:, t], np.float32); ht = ht / (np.linalg.norm(ht, axis=-1, keepdims=True) + 1e-9); Cm = ht @ hl.T
        mm = lambda M: r3(np.nanmean(np.where(M.any(1), (Cm * M).sum(1) / np.maximum(M.sum(1), 1), np.nan)))
        EQ[t] = dict(same_clip=r3(np.diag(Cm).mean()), same_traj_diff_app=mm(ST & ~SA), diff_traj_same_app=mm(~ST & SA & ~E_), diff_traj_diff_app=mm(~ST & ~SA))
    out["equivariance_v2"] = EQ; print("\n[v2] 등변성 cos(h_obj_t, h_last_j)", EQ)
out["v2_checks"] = chk
json.dump(out, default=float, fp=open(OUT / f"m13_v3{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1)
