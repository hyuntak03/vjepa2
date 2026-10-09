# M5 적대 검증 (r2) — 추가: X3 식 멈춤 기울기 (M5 base) · vy 팔의 dx (vx 교차) · 법칙 지시자 교차 · 그룹 안 R² · t0 이동의 칸 위상
import csv, json, os
import numpy as np

RANK0 = os.environ.get("SLURM_PROCID", "0") == "0"
C = "/data2/local_datasets/world/world_analysis/cache/auto_research/"
IDX = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v3/index_probe.csv"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/M5/c_extra.json"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
CELL, G = 18.0, 16
rows = list(csv.DictReader(open(IDX))); byid = {r["video_id"]: r for r in rows}
fl = lambda s: np.array([float(v) for v in s.split()])
res = {}
# ---------- (1) X3 식: M5 base, arc 제외, stall(t5–7 along) = a_law + b·δ, 궤적 평균 단위 ----------
meta = json.load(open(C + "v3_m5/meta.json")); arms = meta["arms"]; S = np.load(C + "v3_m5/steer.npy").astype(np.float64)
ids = meta["video_ids"]; PX = np.stack([fl(byid[v]["px_x_by_sample"]) for v in ids]); PY = np.stack([fl(byid[v]["px_y_by_sample"]) for v in ids])
law = np.array([byid[v]["scenario"] for v in ids]); traj = np.array(["|".join(t) for t in meta["traj"]])
cell = lambda f: np.clip(np.floor(np.stack([(PX[:, f] + PX[:, f + 1]) / 2, (PY[:, f] + PY[:, f + 1]) / 2], -1) / CELL), 0, G - 1)
L = cell(30); T = np.stack([cell(32 + 2 * j) for j in range(8)], 1)
d7 = T[:, 7] - L; nrm = np.linalg.norm(d7, axis=-1); u = d7 / np.maximum(nrm, 1e-9)[:, None]; ok = nrm >= 1
delta = np.hypot((PX[:, 30] + PX[:, 31]) / 2 - (PX[:, 28] + PX[:, 29]) / 2, (PY[:, 30] + PY[:, 31]) / 2 - (PY[:, 28] + PY[:, 29]) / 2) / CELL
rng = np.random.default_rng(0)


def fe(y, sel):
    y = y[sel]; x = delta[sel]; tt = traj[sel]; UT = np.unique(tt)
    ym = np.array([y[tt == t].mean() for t in UT]); xm = np.array([x[tt == t].mean() for t in UT]); tl = np.array([t.split("|")[0] for t in UT])
    Ls = sorted(set(tl)); A = np.concatenate([np.stack([(tl == l).astype(float) for l in Ls], 1), xm[:, None]], 1)
    co = np.linalg.lstsq(A, ym, rcond=None)[0]; bs = []
    for _ in range(2000):
        s = rng.integers(0, len(UT), len(UT))
        if len(set(tl[s])) < len(Ls): continue
        bs.append(np.linalg.lstsq(A[s], ym[s], rcond=None)[0])
    bs = np.array(bs)
    return dict(b=round(float(co[-1]), 2), b_ci=[round(float(v), 2) for v in np.percentile(bs[:, -1], [2.5, 97.5])],
                a=round(float(co[:-1].mean()), 2), a_ci=[round(float(v), 2) for v in np.percentile(bs[:, :-1].mean(1), [2.5, 97.5])], n_traj=len(UT))


sel = ok & (law != "arc")
tru_al = ((T - L[:, None]) * u[:, None]).sum(-1)
x3 = {"truth_t5-7": fe(tru_al[:, 5:8].mean(1), sel), "truth_t2": fe(tru_al[:, 2], sel)}
for nm in ("base", "vx+3", "vx-3", "rand1"):
    a = arms.index(nm); am = np.stack([S[:, a, :, 1], S[:, a, :, 0]], -1); al = ((am - L[:, None]) * u[:, None]).sum(-1)
    x3[f"{nm}_t5-7"] = fe(al[:, 5:8].mean(1), sel); x3[f"{nm}_t0"] = fe(al[:, 0], sel); x3[f"{nm}_t2"] = fe(al[:, 2], sel)
res["x3_style_stall_slope"] = x3
# ---------- (2) vy 팔의 dx 와 dy (vy 조향은 vx 읽기를 −10 px/프레임 바꾼다) ----------
b = arms.index("base")
res["vy_arms"] = {nm: dict(dx=[round(float((S[:, arms.index(nm), t, 1] - S[:, b, t, 1]).mean()), 3) for t in range(8)],
                           dy=[round(float((S[:, arms.index(nm), t, 0] - S[:, b, t, 0]).mean()), 3) for t in range(8)]) for nm in ("vy+3", "vy-3", "vx+3", "vx-3")}
# ---------- (5) t0 이동과 진실 t0 의 칸 안 위상 (연속 이동이면 경계 가까운 clip 만 넘어간다) ----------
xt0 = (PX[:, 32] + PX[:, 33]) / 2; ph = (xt0 / CELL) % 1.0
a = arms.index("vx-3"); mv = (S[:, a, 0, 1] - S[:, b, 0, 1])
res["t0_phase_vs_move_vx-3"] = dict(phase_mean_moved_left=round(float(ph[mv < 0].mean()), 3), phase_mean_moved_right=round(float(ph[mv > 0].mean()), 3),
                                    phase_mean_all=round(float(ph.mean()), 3), n_left=int((mv < 0).sum()), n_right=int((mv > 0).sum()))
a = arms.index("vx+3"); mv = (S[:, a, 0, 1] - S[:, b, 0, 1])
res["t0_phase_vs_move_vx+3"] = dict(phase_mean_moved_left=round(float(ph[mv < 0].mean()), 3), phase_mean_moved_right=round(float(ph[mv > 0].mean()), 3),
                                    n_left=int((mv < 0).sum()), n_right=int((mv > 0).sum()))
# 이동이 법칙별로 어디서 났나
res["t0_moved_by_law_vx+3"] = {l: round(float((np.abs(S[law == l, arms.index("vx+3"), 0, 1] - S[law == l, b, 0, 1]) > 0).mean()), 3) for l in sorted(set(law))}
res["t0_dx_by_law_vx+3"] = {l: round(float((S[law == l, arms.index("vx+3"), 0, 1] - S[law == l, b, 0, 1]).mean()), 3) for l in sorted(set(law))}
res["t0_dx_by_law_vx-3"] = {l: round(float((S[law == l, arms.index("vx-3"), 0, 1] - S[law == l, b, 0, 1]).mean()), 3) for l in sorted(set(law))}
res["t2_dx_by_law_vx+3"] = {l: round(float((S[law == l, arms.index("vx+3"), 2, 1] - S[law == l, b, 2, 1]).mean()), 3) for l in sorted(set(law))}
# ---------- (3)(4) 법칙 지시자 교차 · 그룹 안 R² (M13 z_last LN 근사) ----------
d = np.load(C + "v3_m5/dirs.npz"); ux, uy, r1, sd, W = (d[k].astype(np.float64) for k in ("x", "y", "r1", "sd", "W"))
m13 = json.load(open(C + "v3_m13/meta.json")); Z = np.load(C + "v3_m13/z_last.npy").astype(np.float64); i13 = m13["video_ids"]
free = [i for i, r in enumerate(rows) if r["plausible"] == "1" and r["scenario"] in FREE]
tr = lambda r: (r["scenario"], r["primary"], r["secondary"])
trajs = sorted({tr(rows[i]) for i in free}); fit_t = {t for k, t in enumerate(trajs) if k % 3 == 0}
isfit = np.array([tr(byid[v]) in fit_t for v in i13]); law13 = np.array([byid[v]["scenario"] for v in i13])
grp13 = np.array([byid[v]["scenario"] + "|" + byid[v]["secondary"] for v in i13])
P13x = np.stack([fl(byid[v]["px_x_by_sample"]) for v in i13]); vx13 = (P13x[:, 31] - P13x[:, 27]) / 4
mu_ln, sd_ln = Z[isfit].mean(0), Z[isfit].std(0) + 1e-6; Xs = (Z - mu_ln) / sd_ln
tr_, te_ = np.where(isfit)[0], np.where(~isfit)[0]


def r2(p, y): return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def r2_within(p, y, g):
    pp, yy = p.copy(), y.copy()
    for l in set(g): m = g == l; pp[m] -= pp[m].mean(); yy[m] -= yy[m].mean()
    return r2(pp, yy)


pm5 = Xs @ W[:, 0] + d["ymean"][0]
res["m5W_heldout_within_law_x_secondary_r2"] = r2_within(pm5[te_], vx13[te_], grp13[te_])
# 법칙 one-hot 만의 R² (vx): 법칙 평균으로 예측
lm = {l: vx13[tr_][law13[tr_] == l].mean() for l in set(law13)}
res["law_onehot_only_heldout_r2_vx"] = r2(np.array([lm[l] for l in law13[te_]]), vx13[te_])
cross = {}
for l in sorted(set(law13)):
    y = (law13 == l).astype(float); lam = 10.0 * len(tr_)
    w = np.linalg.solve(Xs[tr_].T @ Xs[tr_] + lam * np.eye(Xs.shape[1]), Xs[tr_].T @ (y[tr_] - y[tr_].mean()))
    heldacc = r2(Xs[te_] @ w + y[tr_].mean(), y[te_])
    cross[l] = dict(heldout_r2=round(heldacc, 3),
                   d_vx3=round(float((3 * ux / sd) @ w), 3), d_vxm3=round(float((-3 * ux / sd) @ w), 3), d_rand=round(float((3 * r1 / sd) @ w), 3),
                   gap_mean_in_vs_out=round(float((Xs[te_] @ w)[y[te_] == 1].mean() - (Xs[te_] @ w)[y[te_] == 0].mean()), 3))
res["law_indicator_crosstalk"] = cross
if RANK0:
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
