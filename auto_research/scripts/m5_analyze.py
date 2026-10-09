#!/usr/bin/env python3
"""M5 분석 — 속도 조향 이득. 입력 v3_m5/.
Δ_t = (argmax 칸)_arm − (argmax 칸)_base  (x, y; 칸).  기대 이동 (속도가 s px/프레임 바뀌었을 때 진짜로 옮겨야 할 칸) = s × dt_t / 18, dt_t = 2t + 2 프레임.
gain_t = mean(Δx_t) / 기대 이동  (vx 팔),  mean(Δy_t) / 기대 (vy 팔).  대조: 무작위 방향의 |Δ| 와 vx 팔의 Δy (새는 정도).
물체다움 변화: contrast (maxcos − median) 의 팔 − base.   CI = 궤적 군집 bootstrap.

사용: python m5_analyze.py [입력 폴더 = v3_m5]      (환경변수 OUT_TAG → exp_results/m5/m5_v3{OUT_TAG}.json)
  M5_INDEX   (기본 data_csv/rollout_v3/index_probe.csv)   — 자연 속도 · 진실 칸 · 법칙
  M5_M13_DIR (기본 입력 폴더 옆 v3_m13)                     — 조향 방향의 held-out · 교차 읽기 (없으면 건너뜀)

키 (1 차, 값 불변): dirs, arms.
키 (2026-09-25 적대 검증 2 차에서 추가 — 검증자 _verify_scratch/M5/{a_steer,b_dirs,c_extra}.py 와 같은 식):
  natural_gain_x    base 팔에서 자연 vx 에 대한 Δx (argmax − 마지막 문맥 칸) 기울기 (법칙 고정효과, 궤적 평균 단위, 궤적 bootstrap 1000)
                    를 조향 이득과 같은 단위 (기울기 ÷ 경과프레임/18) 로 환산. truth_gain = 진실 칸으로 같은 식 (천장). 법칙 묶음 all6 / no_arc5 / flat3
  antisym           vx ±s 반대칭 (부호 특이) 성분 A = (arm+ − arm−)/2 의 이득과 대칭 (비특이) 성분. 궤적 bootstrap 2000
  steer_vs_natural  슬롯별 조향 반대칭 이득 (vx±3) ÷ 자연 이득 (all6)
  moved_frac        팔마다 argmax 칸이 base 와 달라진 clip 비율
  conditional_vx3   base 물체다움 ≥ 0.357 이고 진실 1 칸 안 clip 만의 반대칭 이득과 자연 이득 (먼 슬롯 판독 소실 대조)
  stall_slope_x3    X3 식 멈춘 자리: along(t5–7 평균, 진실 t7 방향 투영, 칸) = a_법칙 + b·δ (δ = 문맥 끝 2 프레임 변위, 칸). arc 제외. 궤적 bootstrap 2000
  vy_arms           vy±3 · vx±3 팔의 평균 Δx, Δy (vy 조향의 vx 교차)
  t0_dx_by_law      vx±3 의 t0 Δx 법칙별 평균
  dir_diag          dirs.npz 로 본 조향 방향: 읽기 검사 (u·W = 1), 노름, clip 간 자연 퍼짐 sqrt(Σsd²) 대비 크기, 표준화 차원별 크기
  dir_heldout_approx M13 z_last (LN 3×3 평균, 같은 C16 창) 근사로 M5 능형 W 의 held-out pooled / 법칙 안 R², 교차 읽기 변화, 법칙 지시자 교차
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m5")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/m5"; OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("OUT_TAG", "")
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; steps = meta["steps"]; keys = meta["keys"]; K = meta["K"]
S = np.load(I / "steer.npy")                                                # (n, arms, K, 4)
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); rng = np.random.default_rng(0)
ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
dirs = json.load(open(I / "dirs.json")) if (I / "dirs.json").exists() else {}


def boot(v, B=500):
    m = np.nanmean(v); bs = [np.nanmean(v[np.concatenate([tix[t] for t in rng.choice(ut, len(ut))])]) for _ in range(B)]
    return [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]


b = arms.index("base"); out = {"dirs": dirs, "arms": {}}
con = S[..., 2] - S[..., 3]
for ai, (nm, s, k) in enumerate(zip(arms, steps, keys)):
    if ai == b: continue
    dx = S[:, ai, :, 1] - S[:, b, :, 1]; dy = S[:, ai, :, 0] - S[:, b, :, 0]
    rec = []
    for t in range(K):
        exp_ = s * (2 * t + 2) / 18.0
        r = dict(t=t, expected_cells=round(exp_, 3), dx=boot(dx[:, t]), dy=boot(dy[:, t]), abs_d=boot(np.hypot(dx[:, t], dy[:, t])),
                 d_contrast=boot(con[:, ai, t] - con[:, b, t]))
        if k == "x": r["gain"] = [round(v / exp_, 3) for v in r["dx"]]
        if k == "y": r["gain"] = [round(v / exp_, 3) for v in r["dy"]]
        rec.append(r)
    out["arms"][nm] = rec

# ============================== 2026-09-25 적대 검증 2 차에서 추가한 키 (위 1 차 키는 값 불변) ==============================
CELL, G, THR = 18.0, 16, 0.357
IDX = Path(os.environ.get("M5_INDEX", str(ROOT / "data_csv/rollout_v3/index_probe.csv")))
byid = {r["video_id"]: r for r in csv.DictReader(open(IDX))}
ids = meta["video_ids"]
fl = lambda s_: np.array([float(v) for v in s_.split()])
PX = np.stack([fl(byid[v]["px_x_by_sample"]) for v in ids]); PY = np.stack([fl(byid[v]["px_y_by_sample"]) for v in ids])
law = np.array([byid[v]["scenario"] for v in ids]); laws_all = sorted(set(law))
assert all(byid[v]["plausible"] == "1" for v in ids)
Sd = S.astype(np.float64); bx, by = Sd[:, b, :, 1], Sd[:, b, :, 0]; cond_ = Sd[..., 2] - Sd[..., 3]
cell = lambda f: np.clip(np.floor(np.stack([(PX[:, f] + PX[:, f + 1]) / 2, (PY[:, f] + PY[:, f + 1]) / 2], -1) / CELL), 0, G - 1)  # (n,2) x,y
L = cell(30); T = np.stack([cell(32 + 2 * t) for t in range(K)], 1)          # 마지막 문맥 칸 (프레임 30·31), 진실 칸 (n,K,2)
vx = (PX[:, 31] - PX[:, 27]) / 4                                           # 문맥 끝 속도 (px/프레임)
ex = lambda t: (2 * t + 2) / 18.0                                          # 1 px/프레임 속도 차가 t 슬롯에서 만드는 칸 이동

rngB = np.random.default_rng(0); B2 = 2000
bidx = [np.concatenate([tix[t] for t in rngB.choice(ut, len(ut))]) for _ in range(B2)]


def ci(v, mask=None):
    v = np.asarray(v, float)
    if mask is None: mask = np.ones(len(v), bool)
    m = np.nanmean(np.where(mask, v, np.nan)); bs = [np.nanmean(np.where(mask[i], v[i], np.nan)) for i in bidx]
    return [round(float(m), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


def fe_slope(y, x, lawset, mask=None):
    """법칙 고정효과 + x 기울기, 궤적 평균 단위, 궤적 bootstrap 1000 (seed 1)."""
    if mask is None: mask = np.ones(len(y), bool)
    keep = mask & np.isin(law, lawset); tt = traj[keep]; u = np.unique(tt)
    ym = np.array([y[keep][tt == q].mean() for q in u]); xm = np.array([x[keep][tt == q].mean() for q in u])
    tl = np.array([q.split("|")[0] for q in u]); Ls = sorted(set(tl))
    A = np.concatenate([np.stack([(tl == l).astype(float) for l in Ls], 1), xm[:, None]], 1)
    co = np.linalg.lstsq(A, ym, rcond=None)[0][-1]; r1 = np.random.default_rng(1); bs = []
    for _ in range(1000):
        si = r1.integers(0, len(u), len(u))
        if len(set(tl[si])) < len(Ls): continue
        bs.append(np.linalg.lstsq(A[si], ym[si], rcond=None)[0][-1])
    return float(co), np.percentile(bs, [2.5, 97.5]), len(u)


# (1) 자연 속도 이득 (base 팔) — 조향 이득과 같은 단위
nat = {}
for lsn, lset in (("all6", laws_all), ("no_arc5", [l for l in laws_all if l != "arc"]), ("flat3", ["flat_v", "flat_a", "flat_d"])):
    rr = []
    for t in range(K):
        bp, bpci, nu = fe_slope(bx[:, t] - L[:, 0], vx, lset); bt, btci, _ = fe_slope(T[:, t, 0] - L[:, 0], vx, lset)
        rr.append(dict(t=t, p_gain=[round(bp / ex(t), 3)] + [round(float(v) / ex(t), 3) for v in bpci],
                       truth_gain=[round(bt / ex(t), 3)] + [round(float(v) / ex(t), 3) for v in btci], n_traj=nu,
                       # 환산 전 기울기 (칸 per px/프레임): 멈추면 슬롯을 따라 평평해진다 (gain 은 1/(2t+2) 로 준다)
                       p_slope=[round(bp, 3)] + [round(float(v), 3) for v in bpci], truth_slope=[round(bt, 3)] + [round(float(v), 3) for v in btci]))
    nat[lsn] = rr
out["natural_gain_x"] = nat
out["vx_stats"] = dict(vx_sd=round(float(vx.std()), 3), vx_sd_within_law=round(float(np.sqrt(np.mean([(vx[law == l] - vx[law == l].mean()).var() for l in laws_all]))), 3),
                       vx_sign_by_law={l: dict(pos=int((vx[law == l] > 0).sum()), neg=int((vx[law == l] < 0).sum())) for l in laws_all})
# (2) 반대칭 / 대칭 분해
anti = {}
for s in (1.5, 3.0):
    p_, m_ = arms.index(f"vx+{s:g}"), arms.index(f"vx-{s:g}")
    A_ = (Sd[:, p_, :, 1] - Sd[:, m_, :, 1]) / 2; Sy = (Sd[:, p_, :, 1] + Sd[:, m_, :, 1]) / 2 - bx
    anti[f"vx{s:g}"] = [dict(t=t, anti_dx=ci(A_[:, t]), gain=[round(v / (s * ex(t)), 3) for v in ci(A_[:, t])], sym_dx=ci(Sy[:, t])) for t in range(K)]
out["antisym"] = anti
out["steer_vs_natural"] = [dict(t=t, steer_gain_vx3=anti["vx3"][t]["gain"][0], natural_gain_all6=nat["all6"][t]["p_gain"][0],
                                ratio=(round(anti["vx3"][t]["gain"][0] / nat["all6"][t]["p_gain"][0], 3) if abs(nat["all6"][t]["p_gain"][0]) > 0.2 else None)) for t in range(K)]
# (3) 칸이 바뀐 clip 비율
out["moved_frac"] = {nm: [round(float((np.hypot(Sd[:, ai, t, 1] - bx[:, t], Sd[:, ai, t, 0] - by[:, t]) > 0).mean()), 3) for t in range(K)]
                     for ai, nm in enumerate(arms) if ai != b}
# (4) 조건부: 물체가 남아 있고 추적 중인 clip
rr = []
p_, m_ = arms.index("vx+3"), arms.index("vx-3"); A_ = (Sd[:, p_, :, 1] - Sd[:, m_, :, 1]) / 2
for t in range(K):
    mk = (cond_[:, b, t] >= THR) & (np.abs(np.stack([bx[:, t], by[:, t]], -1) - T[:, t]).max(-1) <= 1)
    g = ci(A_[:, t], mk); bp, bpci, _ = fe_slope(bx[:, t] - L[:, 0], vx, laws_all, mk)
    rr.append(dict(t=t, n=int(mk.sum()), steer_gain=[round(v / (3.0 * ex(t)), 3) for v in g],
                   natural_gain=[round(bp / ex(t), 3)] + [round(float(v) / ex(t), 3) for v in bpci]))
out["conditional_vx3"] = rr
# (5) X3 식 멈춘 자리 기울기 (arc 제외)
d7 = T[:, 7] - L; nrm = np.linalg.norm(d7, axis=-1); uu = d7 / np.maximum(nrm, 1e-9)[:, None]; ok = nrm >= 1
delta = np.hypot((PX[:, 30] + PX[:, 31]) / 2 - (PX[:, 28] + PX[:, 29]) / 2, (PY[:, 30] + PY[:, 31]) / 2 - (PY[:, 28] + PY[:, 29]) / 2) / CELL
rngX = np.random.default_rng(0)


def fe_ab(y, sel):
    y = y[sel]; x = delta[sel]; tt = traj[sel]; u = np.unique(tt)
    ym = np.array([y[tt == q].mean() for q in u]); xm = np.array([x[tt == q].mean() for q in u]); tl = np.array([q.split("|")[0] for q in u])
    Ls = sorted(set(tl)); A = np.concatenate([np.stack([(tl == l).astype(float) for l in Ls], 1), xm[:, None]], 1)
    co = np.linalg.lstsq(A, ym, rcond=None)[0]; bs = []
    for _ in range(2000):
        si = rngX.integers(0, len(u), len(u))
        if len(set(tl[si])) < len(Ls): continue
        bs.append(np.linalg.lstsq(A[si], ym[si], rcond=None)[0])
    bs = np.array(bs)
    return dict(b=round(float(co[-1]), 2), b_ci=[round(float(v), 2) for v in np.percentile(bs[:, -1], [2.5, 97.5])],
                a=round(float(co[:-1].mean()), 2), a_ci=[round(float(v), 2) for v in np.percentile(bs[:, :-1].mean(1), [2.5, 97.5])], n_traj=len(u))


sel = ok & (law != "arc")
tru_al = ((T - L[:, None]) * uu[:, None]).sum(-1)
x3 = {"truth_t5-7": fe_ab(tru_al[:, 5:8].mean(1), sel), "truth_t2": fe_ab(tru_al[:, 2], sel)}
for nm in ("base", "vx+3", "vx-3", "rand1"):
    ai = arms.index(nm); al = ((np.stack([Sd[:, ai, :, 1], Sd[:, ai, :, 0]], -1) - L[:, None]) * uu[:, None]).sum(-1)
    x3[f"{nm}_t5-7"] = fe_ab(al[:, 5:8].mean(1), sel); x3[f"{nm}_t0"] = fe_ab(al[:, 0], sel); x3[f"{nm}_t2"] = fe_ab(al[:, 2], sel)
out["stall_slope_x3"] = x3
# (6) vy 팔의 x 교차, 법칙별 t0 이동
out["vy_arms"] = {nm: dict(dx=[round(float((Sd[:, arms.index(nm), t, 1] - bx[:, t]).mean()), 3) for t in range(K)],
                           dy=[round(float((Sd[:, arms.index(nm), t, 0] - by[:, t]).mean()), 3) for t in range(K)]) for nm in ("vy+3", "vy-3", "vx+3", "vx-3")}
out["t0_dx_by_law"] = {nm: {l: round(float((Sd[law == l, arms.index(nm), 0, 1] - bx[law == l, 0]).mean()), 3) for l in laws_all} for nm in ("vx+3", "vx-3")}
# (7) 조향 방향 진단 (dirs.npz)
if (I / "dirs.npz").exists():
    dz = np.load(I / "dirs.npz"); ux, uy, r1v, r2v, mu, sd, W = (dz[k].astype(np.float64) for k in ("x", "y", "r1", "r2", "mu", "sd", "W"))
    Wr = W / sd[:, None]; spread = float(np.sqrt((sd ** 2).sum()))
    dd = dict(readout_check=dict(ux_Wx=float(ux @ Wr[:, 0]), ux_Wy=float(ux @ Wr[:, 1]), uy_Wy=float(uy @ Wr[:, 1]), uy_Wx=float(uy @ Wr[:, 0]),
                                 r1_Wx=float(r1v @ Wr[:, 0]), r2_Wx=float(r2v @ Wr[:, 0])),
              norm_ux=float(np.linalg.norm(ux)), norm_uy=float(np.linalg.norm(uy)), norm_mean_vector=float(np.linalg.norm(mu)),
              note_mean_token_norm="dirs.json 의 mean_token_norm (71.4) 은 개별 토큰이 아니라 3×3 평균 벡터의 노름이다",
              spread_sqrt_sum_sd2=spread, cos_ux_uy=float(ux @ uy / np.linalg.norm(ux) / np.linalg.norm(uy)))
    for nm, v in (("vx3", 3 * ux), ("vy3", 3 * uy), ("rand3", 3 * r1v)):
        zs = v / sd
        dd[f"pert_{nm}"] = dict(raw_norm=float(np.linalg.norm(v)), rel_to_natural_spread=float(np.linalg.norm(v) / spread),
                                std_norm_rel_sqrtD=float(np.linalg.norm(zs) / np.sqrt(len(zs))), std_max_abs=float(np.abs(zs).max()),
                                frac_gt1sd=float((np.abs(zs) > 1).mean()), frac_gt3sd=float((np.abs(zs) > 3).mean()))
    o = np.argsort(sd); lo = o[: len(o) // 10]
    dd["ux_energy_lowest10pct_sd"] = float((ux[lo] ** 2).sum() / (ux ** 2).sum()); dd["r1_energy_lowest10pct_sd"] = float((r1v[lo] ** 2).sum() / (r1v ** 2).sum())
    out["dir_diag"] = dd
    # (8) held-out 근사 (M13 z_last, LN 3×3 평균, 같은 C16 창) — 적합 특징 X 가 dirs.npz 에 없어 정확한 재계산은 불가
    M13 = Path(os.environ.get("M5_M13_DIR", str(I.parent / "v3_m13")))
    if (M13 / "z_last.npy").exists():
        FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
        rows = [r for r in byid.values()]
        free = [r for r in rows if r["plausible"] == "1" and r["scenario"] in FREE]
        trk = lambda r: (r["scenario"], r["primary"], r["secondary"])
        trajs = sorted({trk(r) for r in free}); fit_t = {t for k_, t in enumerate(trajs) if k_ % 3 == 0}
        m13 = json.load(open(M13 / "meta.json")); Z = np.load(M13 / "z_last.npy").astype(np.float64); i13 = m13["video_ids"]
        isfit = np.array([trk(byid[v]) in fit_t for v in i13]); law13 = np.array([byid[v]["scenario"] for v in i13])
        grp13 = np.array([byid[v]["scenario"] + "|" + byid[v]["secondary"] for v in i13])
        P13x = np.stack([fl(byid[v]["px_x_by_sample"]) for v in i13]); P13y = np.stack([fl(byid[v]["px_y_by_sample"]) for v in i13])
        v13x = (P13x[:, 31] - P13x[:, 27]) / 4; v13y = (P13y[:, 31] - P13y[:, 27]) / 4; xc13 = (P13x[:, 30] + P13x[:, 31]) / 2
        mu_ln, sd_ln = Z[isfit].mean(0), Z[isfit].std(0) + 1e-6; Xs = (Z - mu_ln) / sd_ln; tr_, te_ = np.where(isfit)[0], np.where(~isfit)[0]
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

        def r2w(p, y, g):
            pp, yy = p.copy(), y.copy()
            for l in set(g): m = g == l; pp[m] -= pp[m].mean(); yy[m] -= yy[m].mean()
            return r2(pp, yy)

        def ridge(y):
            w = np.linalg.solve(Xs[tr_].T @ Xs[tr_] + 10.0 * len(tr_) * np.eye(Xs.shape[1]), Xs[tr_].T @ (y[tr_] - y[tr_].mean())); return w, y[tr_].mean()

        A_mu = np.stack([mu_ln, np.ones_like(mu_ln)], 1); co = np.linalg.lstsq(A_mu, mu, rcond=None)[0]
        pm = Xs @ W[:, 0] + dz["ymean"][0]; pmy = Xs @ W[:, 1] + dz["ymean"][1]
        lm = {l: v13x[tr_][law13[tr_] == l].mean() for l in set(law13)}
        ho = dict(approx_mu_fit_r2=float(1 - ((A_mu @ co - mu) ** 2).sum() / ((mu - mu.mean()) ** 2).sum()),
                  approx_sd_ratio_cv=float((sd / sd_ln).std() / (sd / sd_ln).mean()),
                  n_fit_traj_clips=int(isfit.sum()), n_heldout=int((~isfit).sum()),
                  m5_eval_in_fit_traj=int(sum(tuple(map(str, t)) in fit_t for t in meta["traj"])),
                  m5_eval_traj_found_in_index=int(sum(tuple(map(str, t)) in set(trajs) for t in meta["traj"])),   # 형식 일치 검사 (784 여야 한다)
                  vx_fit_r2=r2(pm[tr_], v13x[tr_]), vx_heldout_r2=r2(pm[te_], v13x[te_]), vx_heldout_within_law=r2w(pm[te_], v13x[te_], law13[te_]),
                  vx_heldout_within_law_x_secondary=r2w(pm[te_], v13x[te_], grp13[te_]),
                  vy_heldout_r2=r2(pmy[te_], v13y[te_]), vy_heldout_within_law=r2w(pmy[te_], v13y[te_], law13[te_]),
                  law_onehot_only_heldout_r2_vx=r2(np.array([lm[l] for l in law13[te_]]), v13x[te_]))
        wv, _ = ridge(v13x); wvy, _ = ridge(v13y); wxa, _ = ridge(xc13)
        ho["cross_readout_change_s3"] = {nm: dict(vx=float((v / sd) @ wv), vy=float((v / sd) @ wvy), xabs_px=float((v / sd) @ wxa))
                                         for nm, v in (("vx+3", 3 * ux), ("vy+3", 3 * uy), ("rand1", 3 * r1v), ("rand2", 3 * r2v))}
        cr = {}
        for l in sorted(set(law13)):
            y = (law13 == l).astype(float); w, y0 = ridge(y); pr = Xs[te_] @ w
            cr[l] = dict(heldout_r2=round(r2(pr + y0, y[te_]), 3), d_vx3=round(float((3 * ux / sd) @ w), 3), d_rand=round(float((3 * r1v / sd) @ w), 3),
                         gap_in_vs_out=round(float(pr[y[te_] == 1].mean() - pr[y[te_] == 0].mean()), 3))
            cr[l]["d_vx3_over_gap"] = round(cr[l]["d_vx3"] / cr[l]["gap_in_vs_out"], 3)
        ho["law_indicator_crosstalk"] = cr
        out["dir_heldout_approx"] = ho

json.dump(out, open(OUT / f"m5_v3{TAG}.json", "w"), indent=1, default=float)
print("방향 적합:", dirs)
print("팔        | 슬롯별 이득 (gain = 실제 이동 / 속도 바뀐 만큼의 이동) t0..7            | |Δ| (칸) t0..7")
for nm, rec in out["arms"].items():
    g = " ".join(f"{r['gain'][0]:+.2f}" if "gain" in r else "  –  " for r in rec)
    print(f"{nm:9s} | {g} | " + " ".join(f"{r['abs_d'][0]:.2f}" for r in rec))
print("\nvx+3 상세 (Δx [CI] / 기대 칸 / Δy / Δcontrast):")
for r in out["arms"].get("vx+3", []):
    print(f" t{r['t']}  Δx {r['dx'][0]:+.2f} [{r['dx'][1]:+.2f},{r['dx'][2]:+.2f}]  기대 {r['expected_cells']:.2f}  Δy {r['dy'][0]:+.2f}  Δcon {r['d_contrast'][0]:+.3f}")
print("\n== 조향 반대칭 이득 (vx±3) vs 자연 속도 이득 (base, 법칙 고정효과) — 같은 단위")
for t in range(K):
    a3, n6, nr = anti["vx3"][t], nat["all6"][t], nat["no_arc5"][t]
    print(f" t{t}  조향 {a3['gain'][0]:+.3f} [{a3['gain'][1]:+.3f},{a3['gain'][2]:+.3f}] 대칭 {a3['sym_dx'][0]:+.3f}"
          f" | 자연 all6 {n6['p_gain'][0]:+.2f} [{n6['p_gain'][1]:+.2f},{n6['p_gain'][2]:+.2f}] 진실 {n6['truth_gain'][0]:+.2f}"
          f" | no_arc5 {nr['p_gain'][0]:+.2f} 진실 {nr['truth_gain'][0]:+.2f} | 칸 바뀜 vx+3 {out['moved_frac']['vx+3'][t]:.2f}"
          f" | 기울기 p {n6['p_slope'][0]:+.3f} [{n6['p_slope'][1]:+.3f},{n6['p_slope'][2]:+.3f}] 진실 {n6['truth_slope'][0]:+.3f}")
print("\n== 조건부 (물체다움 ≥ 0.357 · 진실 1 칸 안)")
for r in out["conditional_vx3"]: print(f" t{r['t']} n {r['n']} 조향 {r['steer_gain']} 자연 {r['natural_gain']}")
print("\n== X3 식 멈춘 자리 기울기 (arc 제외):", {k: (v["b"], v["b_ci"], v["a"]) for k, v in x3.items()})
if "dir_heldout_approx" in out:
    h = out["dir_heldout_approx"]
    print("\n== 조향 방향 held-out (M13 근사): vx pooled", round(h["vx_heldout_r2"], 3), "법칙 안", round(h["vx_heldout_within_law"], 3),
          "법칙×secondary 안", round(h["vx_heldout_within_law_x_secondary"], 3), "법칙 one-hot 만", round(h["law_onehot_only_heldout_r2_vx"], 3))
    print("   교차 읽기 (s=3):", h["cross_readout_change_s3"]); print("   법칙 지시자:", h["law_indicator_crosstalk"])
if "dir_diag" in out:
    print("\n== 섭동 크기:", {k: out["dir_diag"][k] for k in ("pert_vx3", "pert_vy3", "pert_rand3")})
