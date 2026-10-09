# M5 적대 검증 (r2) — 조향 방향 자체: 크기 · 분포 밖 정도 · held-out R² (M13 LN z_last 근사) · 위치 읽기와의 교차 · predictor 입력 사상.
import csv, json, os
import numpy as np

RANK0 = os.environ.get("SLURM_PROCID", "0") == "0"
C = "/data2/local_datasets/world/world_analysis/cache/auto_research/"
IDX = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v3/index_probe.csv"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/M5/b_dirs.json"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
d = np.load(C + "v3_m5/dirs.npz"); ux, uy, r1, rr2, mu, sd, W = (d[k].astype(np.float64) for k in ("x", "y", "r1", "r2", "mu", "sd", "W"))
res = {}
Wr = W / sd[:, None]
res["readout_check"] = dict(ux_dot_Wrx=float(ux @ Wr[:, 0]), ux_dot_Wry=float(ux @ Wr[:, 1]), uy_dot_Wry=float(uy @ Wr[:, 1]), uy_dot_Wrx=float(uy @ Wr[:, 0]),
                            r1_dot_Wrx=float(r1 @ Wr[:, 0]), r2_dot_Wrx=float(rr2 @ Wr[:, 0]))
res["norms"] = dict(ux=float(np.linalg.norm(ux)), uy=float(np.linalg.norm(uy)), r1=float(np.linalg.norm(r1)), mu=float(np.linalg.norm(mu)),
                    spread_sqrt_sum_sd2=float(np.sqrt((sd ** 2).sum())), cos_ux_uy=float(ux @ uy / np.linalg.norm(ux) / np.linalg.norm(uy)),
                    cos_ux_mu=float(ux @ mu / np.linalg.norm(ux) / np.linalg.norm(mu)), cos_r1_mu=float(r1 @ mu / np.linalg.norm(r1) / np.linalg.norm(mu)))
# 표준화 공간에서 본 섭동 (s=3): 각 차원이 자연 표준편차의 몇 배 움직이나
for nm, v in (("vx3", 3 * ux), ("vy3", 3 * uy), ("rand3", 3 * r1)):
    zs = v / sd
    res[f"std_pert_{nm}"] = dict(norm=float(np.linalg.norm(zs)), rel_to_sqrtD=float(np.linalg.norm(zs) / np.sqrt(len(zs))), max_abs=float(np.abs(zs).max()),
                                frac_gt1sd=float((np.abs(zs) > 1).mean()), frac_gt3sd=float((np.abs(zs) > 3).mean()),
                                raw_norm=float(np.linalg.norm(v)), raw_rel_spread=float(np.linalg.norm(v) / np.sqrt((sd ** 2).sum())))
# 방향 집중도: u_x 에너지의 몇 % 가 sd 하위 10% 차원에 몰렸나 (1/sd² 가중 효과)
o = np.argsort(sd); lo = o[: len(o) // 10]
res["ux_energy_in_lowest10pct_sd_dims"] = float((ux[lo] ** 2).sum() / (ux ** 2).sum())
res["r1_energy_in_lowest10pct_sd_dims"] = float((r1[lo] ** 2).sum() / (r1 ** 2).sum())
res["sd_ratio_max_min"] = float(sd.max() / sd.min())

# --- M13 z_last (LN(z) 3×3 평균, 같은 C16 창) 로 held-out 검사 -------------------------------------------------
rows = list(csv.DictReader(open(IDX)))
free = [i for i, r in enumerate(rows) if r["plausible"] == "1" and r["scenario"] in FREE]
tr = lambda r: (r["scenario"], r["primary"], r["secondary"])
trajs = sorted({tr(rows[i]) for i in free}); fit_t = {t for k, t in enumerate(trajs) if k % 3 == 0}
m5 = json.load(open(C + "v3_m5/meta.json")); m13 = json.load(open(C + "v3_m13/meta.json"))
byid = {r["video_id"]: r for r in rows}
res["m5_eval_in_fit_traj"] = int(sum(tuple(t) in fit_t for t in m5["traj"]))
Z = np.load(C + "v3_m13/z_last.npy").astype(np.float64); ids = m13["video_ids"]
fl = lambda s: np.array([float(v) for v in s.split()])
PX = np.stack([fl(byid[v]["px_x_by_sample"]) for v in ids]); PY = np.stack([fl(byid[v]["px_y_by_sample"]) for v in ids])
vx = (PX[:, 31] - PX[:, 27]) / 4; vy = (PY[:, 31] - PY[:, 27]) / 4
xc = (PX[:, 30] + PX[:, 31]) / 2; yc = (PY[:, 30] + PY[:, 31]) / 2
xoff = xc - (np.floor(xc / 18) * 18 + 9)                    # 칸 중심 대비 x 오프셋 (px)
law = np.array([byid[v]["scenario"] for v in ids])
isfit = np.array([tuple(byid[v][k] for k in ("scenario", "primary", "secondary")) in fit_t for v in ids])
res["m13_n_fit_traj_clips"] = int(isfit.sum()); res["m13_n_heldout"] = int((~isfit).sum())
mu_ln, sd_ln = Z[isfit].mean(0), Z[isfit].std(0) + 1e-6
# 근사 타당성: raw ≈ a·LN + c (토큰별 스칼라가 거의 상수일 때). mu / sd 의 선형 관계
A = np.stack([mu_ln, np.ones_like(mu_ln)], 1); co = np.linalg.lstsq(A, mu, rcond=None)[0]
res["approx_mu_fit"] = dict(a=float(co[0]), c=float(co[1]), r2=float(1 - ((A @ co - mu) ** 2).sum() / ((mu - mu.mean()) ** 2).sum()))
res["approx_sd_corr"] = float(np.corrcoef(sd, sd_ln)[0, 1]); res["approx_sd_ratio_cv"] = float((sd / sd_ln).std() / (sd / sd_ln).mean())
res["approx_sd_ratio_median"] = float(np.median(sd / sd_ln))
Xs = (Z - mu_ln) / sd_ln


def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def r2_within(p, y, lw):
    pp, yy = p.copy(), y.copy()
    for l in set(lw): m = lw == l; pp[m] -= pp[m].mean(); yy[m] -= yy[m].mean()
    return r2(pp, yy)


def ridge(Xtr, ytr):
    lam = 10.0 * len(Xtr); return np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean())), ytr.mean()


tr_, te_ = np.where(isfit)[0], np.where(~isfit)[0]
out = {}
for ynm, y in (("vx", vx), ("vy", vy), ("xoff", xoff), ("xabs", xc)):
    w, b0 = ridge(Xs[tr_], y[tr_])
    out[ynm] = dict(train_r2=r2(Xs[tr_] @ w + b0, y[tr_]), heldout_r2=r2(Xs[te_] @ w + b0, y[te_]), heldout_within_law_r2=r2_within(Xs[te_] @ w + b0, y[te_], law[te_]),
                    w=w)
    # 240 clip 부분표본 (M5 와 같은 n)
    rs = np.random.default_rng(0); pk = rs.choice(tr_, min(240, len(tr_)), replace=False)
    w2, b2 = ridge(Xs[pk], y[pk])
    out[ynm].update(train_r2_n240=r2(Xs[pk] @ w2 + b2, y[pk]), heldout_r2_n240=r2(Xs[te_] @ w2 + b2, y[te_]), heldout_within_law_r2_n240=r2_within(Xs[te_] @ w2 + b2, y[te_], law[te_]))
# M5 의 실제 W (표준화 공간) 를 LN-표준화 특징에 그대로 (근사)
pm5 = Xs @ W[:, 0] + d["ymean"][0]; pm5y = Xs @ W[:, 1] + d["ymean"][1]
res["m5_W_on_m13_approx"] = dict(vx_fit_r2=r2(pm5[tr_], vx[tr_]), vx_heldout_r2=r2(pm5[te_], vx[te_]), vx_heldout_within_law=r2_within(pm5[te_], vx[te_], law[te_]),
                                 vy_heldout_r2=r2(pm5y[te_], vy[te_]), vy_heldout_within_law=r2_within(pm5y[te_], vy[te_], law[te_]),
                                 corr_vx_heldout=float(np.corrcoef(pm5[te_], vx[te_])[0, 1]))
# 교차: M5 조향 (표준화 공간 섭동 = s·u/sd) 이 다른 읽기를 얼마나 바꾸나 (s=3)
cross = {}
for nm, v in (("vx+3", 3 * ux), ("vy+3", 3 * uy), ("rand1", 3 * r1), ("rand2", 3 * rr2)):
    zs = v / sd
    cross[nm] = {k: float(zs @ out[k]["w"]) for k in out}
res["cross_readout_change_s3"] = cross
res["ridge_heldout"] = {k: {kk: vv for kk, vv in v.items() if kk != "w"} for k, v in out.items()}
res["cos_std_W_m5x_vs_refit_vx"] = float(W[:, 0] @ out["vx"]["w"] / np.linalg.norm(W[:, 0]) / np.linalg.norm(out["vx"]["w"]))
res["cos_std_W_m5x_vs_xoff"] = float(W[:, 0] @ out["xoff"]["w"] / np.linalg.norm(W[:, 0]) / np.linalg.norm(out["xoff"]["w"]))
# 법칙 내 vx 부호: 방향은 법칙 안에서 바뀌나
res["vx_sign_by_law"] = {l: dict(pos=int((vx[law == l] > 0).sum()), neg=int((vx[law == l] < 0).sum())) for l in sorted(set(law))}

# --- predictor 입력 사상 (predictor_embed) 에서의 크기 -----------------------------------------------------------
try:
    import torch
    ck = torch.load("/data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth",
                    map_location="cpu", mmap=True, weights_only=False)
    pr = ck["predictor"]; kk = [k for k in pr if "predictor_embed" in k]
    We = pr[[k for k in kk if k.endswith("weight")][0]].double().numpy()
    res["predictor_embed_keys"] = kk; res["We_shape"] = list(We.shape)
    sv = np.linalg.svd(We, compute_uv=False)
    def en(v): v = v / np.linalg.norm(v); return float(np.linalg.norm(We @ v))
    rng = np.random.default_rng(5); rnd = [en(rng.standard_normal(We.shape[1])) for _ in range(200)]
    res["We_gain"] = dict(ux=en(ux), uy=en(uy), r1=en(r1), r2=en(rr2), rand200_mean=float(np.mean(rnd)), rand200_sd=float(np.std(rnd)),
                          sv_max=float(sv[0]), sv_median=float(np.median(sv)), sv_min=float(sv[-1]), mu=en(mu))
    # sd 가중 무작위 (자연 분산 모양) 방향
    sdw = [en(rng.standard_normal(We.shape[1]) * sd) for _ in range(200)]
    res["We_gain"]["sd_shaped_rand_mean"] = float(np.mean(sdw))
except Exception as ex:
    res["predictor_embed_error"] = repr(ex)
if RANK0:
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
