#!/usr/bin/env python3
"""SSV2T verifier (independent code, does not import repo scripts).
Checks on ssv2_e3: NaN, proj ordering vs feats, token beta/rho/gain recompute + clip CI,
JL split-dim stability, pooled full-D vs proj, label-matched null, cross-lag rho, truth autocorr,
spatial-scale ladder, 2x2 (pooled/token x diff/dev), D_oe magnitude, e2 vs e3 identity."""
import json, sys, time
import numpy as np
t0 = time.time()
I = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e3"
I2 = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e2"
meta = json.load(open(f"{I}/meta.json")); n = meta["n"]; lab = np.array(meta["labels"])
R = {}
def pr(*a):
    print(*a, flush=True)

PR = np.load(f"{I}/proj.npy", mmap_mode="r")
pr("proj", PR.shape, PR.dtype)
# 0. NaN
nanrows = [int(np.isnan(np.asarray(PR[c], np.float32)).any()) for c in range(n)]
R["proj_nan_rows"] = int(sum(nanrows)); pr("proj rows with any NaN", sum(nanrows))
FS = np.load(f"{I}/feats.npy", mmap_mode="r")
R["feats_nan_rows"] = int(np.isnan(np.asarray(FS[:, :, :, :8], np.float32)).any((1, 2, 3)).sum())

# 1. ordering / projection validation: token-mean of proj == feats @ Rp
Rp = (np.random.default_rng(123).standard_normal((1280, 64)) / 8.0).astype(np.float32)
sub = np.arange(0, n, 17)
pm = np.asarray(PR[sub], np.float32).mean(3)                     # (s,2,25,64)
F = np.asarray(FS[sub], np.float32)                               # (s,2,56,1280)
hF = F[:, :, 8:24] @ Rp; pF = F[:, :, 32:40] @ Rp; hc7F = F[:, :, 31] @ Rp
rel = lambda a, b: float(np.abs(a - b).mean() / np.abs(b).mean())
R["order_check_rel_err"] = dict(h=rel(pm[:, :, 0:16], hF), p=rel(pm[:, :, 16:24], pF), hc7=rel(pm[:, :, 24], hc7F),
                                p_vs_wrong_slot=rel(pm[:, :, 17:24], pF[:, :, 0:7]))
pr("order check", R["order_check_rel_err"])
del F

# 2. e2 vs e3 identity of D_p
Dp3 = np.load(f"{I}/D_p.npy"); Dp2 = np.load(f"{I2}/D_p.npy")
R["Dp_e2_vs_e3_maxabs"] = float(np.nanmax(np.abs(Dp3 - Dp2))); pr("D_p e2 vs e3 max|diff|", R["Dp_e2_vs_e3_maxabs"])

rng = np.random.default_rng(11)
def boot_idx(m, B=1000):
    return [rng.integers(0, m, m) for _ in range(B)]

def parts(P, H, perm=None):
    dP = np.diff(P, axis=1); dH = np.diff(H, axis=1)
    dP -= dP.mean(0, keepdims=True); dH -= dH.mean(0, keepdims=True)
    if perm is not None: dH = dH[perm]
    ax = tuple(range(2, dP.ndim))
    num = (dP * dH).sum(ax, dtype=np.float64); nP = (dP * dP).sum(ax, dtype=np.float64); nH = (dH * dH).sum(ax, dtype=np.float64)
    return num, nP, nH

def summ(num, nP, nH):
    return dict(beta=num.sum(0) / nH.sum(0), rho=num.sum(0) / np.sqrt(nP.sum(0) * nH.sum(0)), gain=np.sqrt(nP.sum(0) / nH.sum(0)))

def ci(num, nP, nH, BI):
    bs = {k: [] for k in ("beta", "rho", "gain", "rho0_minus_rho6", "rho_slope")}
    for b in BI:
        s = summ(num[b], nP[b], nH[b])
        for k in ("beta", "rho", "gain"): bs[k].append(s[k])
        bs["rho0_minus_rho6"].append(s["rho"][0] - s["rho"][6]); bs["rho_slope"].append(np.polyfit(np.arange(7), s["rho"], 1)[0])
    return {k: (np.round(np.percentile(np.array(v), 2.5, 0), 4).tolist(), np.round(np.percentile(np.array(v), 97.5, 0), 4).tolist()) for k, v in bs.items()}

r4 = lambda x: np.round(np.asarray(x, dtype=float), 4).tolist()
BI = boot_idx(n)
store = {}
for di, dn in enumerate(("forward", "reversed")):
    A = np.asarray(PR[:, di], np.float32)                          # (n,25,256,64)
    H, P, HC = A[:, 8:16], A[:, 16:24], A[:, 24]
    rec = {}
    num, nP, nH = parts(P, H); s = summ(num, nP, nH)
    rec["token_all"] = {k: r4(v) for k, v in s.items()}; rec["token_all_ci"] = ci(num, nP, nH, BI)
    rec["identity_beta_eq_rho_x_gain_maxabs"] = float(np.abs(s["beta"] - s["rho"] * s["gain"]).max())
    store[dn] = (num, nP, nH)
    # label-matched shuffle null
    perm = np.arange(n)
    for L in (0, 1):
        ix = np.where(lab == L)[0]; perm[ix] = rng.permutation(ix)
    nu0, p0, h0 = parts(P, H, perm); rec["null_label_matched_rho"] = r4(summ(nu0, p0, h0)["rho"])
    permA = rng.permutation(n); nu1, p1, h1 = parts(P, H, permA); rec["null_any_rho"] = r4(summ(nu1, p1, h1)["rho"])
    # JL split dims: two independent 32-d projections, four 16-d
    for nm, sl in (("dims0_31", slice(0, 32)), ("dims32_63", slice(32, 64)), ("d0_15", slice(0, 16)), ("d16_31", slice(16, 32)), ("d32_47", slice(32, 48)), ("d48_63", slice(48, 64))):
        a, b, c = parts(P[..., sl], H[..., sl]); ss = summ(a, b, c); rec[f"jl_{nm}"] = dict(rho=r4(ss["rho"]), gain=r4(ss["gain"]), beta=r4(ss["beta"]))
    # approx dynamic tokens (top 25% of projected-h temporal variance)
    var = A[:, 0:16].var(1).mean(-1)                                # (n,256)
    thr = np.quantile(var, 0.75, axis=1, keepdims=True); dyn = var >= thr
    Pd = np.stack([P[c][:, dyn[c]][:, :64] for c in range(n)]); Hd = np.stack([H[c][:, dyn[c]][:, :64] for c in range(n)])
    a, b, c = parts(Pd, Hd); ss = summ(a, b, c); rec["token_dyn_approx"] = {k: r4(v) for k, v in ss.items()}
    # spatial scale ladder: block means over 16x16 grid
    Pg = P.reshape(n, 8, 16, 16, 64); Hg = H.reshape(n, 8, 16, 16, 64)
    lad = {}
    for k in (1, 2, 4, 8, 16):
        Pk = Pg.reshape(n, 8, 16 // k, k, 16 // k, k, 64).mean((3, 5)); Hk = Hg.reshape(n, 8, 16 // k, k, 16 // k, k, 64).mean((3, 5))
        a, b, c = parts(Pk, Hk); ss = summ(a, b, c); lad[f"block{k}"] = {kk: r4(v) for kk, v in ss.items()}
    rec["scale_ladder_firstdiff"] = lad
    # 2x2: token vs pooled x first-diff vs deviation-from-slot-mean (index-centred first, ratio of sums & row-mean)
    def dev_beta(X, Y):
        Xc = X - X.mean(0, keepdims=True); Yc = Y - Y.mean(0, keepdims=True)
        dX = Xc - Xc.mean(1, keepdims=True); dY = Yc - Yc.mean(1, keepdims=True)
        ax = tuple(range(1, dX.ndim)); nu = (dX * dY).sum(ax, dtype=np.float64); de = (dY * dY).sum(ax, dtype=np.float64)
        nP_ = (dX * dX).sum(ax, dtype=np.float64)
        return dict(ratio_of_sums=float(nu.sum() / de.sum()), row_mean=float((nu / de).mean()), rho=float(nu.sum() / np.sqrt(nP_.sum() * de.sum())), gain=float(np.sqrt(nP_.sum() / de.sum())))
    Pp, Hp = P.mean(2), H.mean(2)
    a, b, c = parts(Pp, Hp); ssp = summ(a, b, c)
    rec["twobytwo"] = dict(token_firstdiff_beta_pooledratio=float(num.sum() / nH.sum()),
                           pooled_firstdiff_beta_pooledratio=float(a.sum() / c.sum()), pooled_firstdiff=dict({k: r4(v) for k, v in ssp.items()}),
                           token_dev=dev_beta(P, H), pooled_dev=dev_beta(Pp, Hp),
                           dyn_pooled_dev=dev_beta(Pd.mean(2), Hd.mean(2)), dyn_token_dev=dev_beta(Pd, Hd))
    # cross-lag rho matrix  M[i,k] = corr(dp_i, dH_k)
    dP = np.diff(P, axis=1); dP -= dP.mean(0, keepdims=True); dH = np.diff(H, axis=1); dH -= dH.mean(0, keepdims=True)
    M = np.einsum('nitd,nktd->ik', dP, dH, dtype=np.float64) if False else None
    Mnum = np.zeros((7, 7));
    for i in range(7):
        for k in range(7):
            Mnum[i, k] = float((dP[:, i].astype(np.float64) * dH[:, k]).sum())
    nPi = (dP.astype(np.float64) ** 2).sum((0, 2, 3)); nHk = (dH.astype(np.float64) ** 2).sum((0, 2, 3))
    rec["crosslag_rho"] = np.round(Mnum / np.sqrt(nPi[:, None] * nHk[None]), 4).tolist()
    # truth autocorrelation of changes & persistence baselines (h7-h6 is bidirectional -> leaky)
    Mh = np.zeros((7, 7))
    for i in range(7):
        for k in range(7):
            Mh[i, k] = float((dH[:, i].astype(np.float64) * dH[:, k]).sum())
    rec["truth_change_autocorr"] = np.round(Mh / np.sqrt(nHk[:, None] * nHk[None]), 4).tolist()
    base = (A[:, 7] - A[:, 6]); base = base - base.mean(0, keepdims=True)
    rec["persistence_h7_minus_h6_rho"] = r4([(base.astype(np.float64) * dH[:, k]).sum() / np.sqrt((base.astype(np.float64) ** 2).sum() * nHk[k]) for k in range(7)])
    # p own-change autocorrelation
    Mp = np.zeros(7)
    for i in range(7): Mp[i] = float((dP[:, 0].astype(np.float64) * dP[:, i]).sum()) / np.sqrt(nPi[0] * nPi[i])
    rec["p_change_autocorr_vs_first"] = r4(Mp)
    # p change magnitude before centering vs after (how much is clip-independent index signature)
    dPraw = np.diff(P, axis=1); dHraw = np.diff(H, axis=1)
    rec["frac_energy_removed_by_centering"] = dict(p=r4(1 - nPi / (dPraw.astype(np.float64) ** 2).sum((0, 2, 3))), h=r4(1 - nHk / (dHraw.astype(np.float64) ** 2).sum((0, 2, 3))))
    del dP, dH, dPraw, dHraw, Pg, Hg, Pd, Hd
    R[dn] = rec
    pr(f"[{dn}] done {time.time()-t0:.0f}s"); pr(json.dumps({k: v for k, v in rec.items() if k in ("token_all", "null_label_matched_rho", "jl_dims0_31", "jl_dims32_63", "token_dyn_approx")}))
    del A, H, P, HC

# forward vs reversed rho difference CI (same clip resampled)
nf, pf, hf = store["forward"]; nr, prr, hr = store["reversed"]
d = []
for b in BI:
    d.append(summ(nr[b], prr[b], hr[b])["rho"] - summ(nf[b], pf[b], hf[b])["rho"])
d = np.array(d); R["rho_reversed_minus_forward"] = dict(est=r4(summ(nr, prr, hr)["rho"] - summ(nf, pf, hf)["rho"]), lo=r4(np.percentile(d, 2.5, 0)), hi=r4(np.percentile(d, 97.5, 0)))

# pooled full-D (feats, all-token mean) vs 64-d projection of same pooled vectors
F = np.asarray(FS, np.float32)
for di, dn in enumerate(("forward", "reversed")):
    Hf, Pf = F[:, di, 16:24], F[:, di, 32:40]
    a, b, c = parts(Pf, Hf); sF = summ(a, b, c)
    a, b, c = parts(Pf @ Rp, Hf @ Rp); sJ = summ(a, b, c)
    R[f"pooled_fullD_vs_JL|{dn}"] = dict(fullD={k: r4(v) for k, v in sF.items()}, jl64={k: r4(v) for k, v in sJ.items()})
del F

# D_oe magnitude matching (full-D, true dyn tokens): oracle diag L1 vs p diag L1
Doe = np.load(f"{I}/D_oe.npy"); Dp = Dp3
ok = np.isfinite(Doe).all((1, 2, 3))
dg = lambda X: np.stack([X[:, :, i, 8 + i] for i in range(8)], -1)
R["Doe_over_Dp_diag"] = dict(n_ok=int(ok.sum()), ratio_forward=r4(dg(Doe[ok])[:, 0].mean(0) / dg(Dp[ok])[:, 0].mean(0)),
                             ratio_reversed=r4(dg(Doe[ok])[:, 1].mean(0) / dg(Dp[ok])[:, 1].mean(0)))
# raw alignment of D_oe (no null subtraction) and margin: min offdiag - diag
Doe_ok = Doe[ok]
marg = []
for i in range(8):
    off = np.delete(Doe_ok[:, :, i], 8 + i, axis=-1).min(-1); marg.append(float(np.median((off - Doe_ok[:, :, i, 8 + i]) / Doe_ok[:, :, i, 8 + i])))
R["Doe_rel_margin_median"] = r4(marg)
Dp_ok = Dp[ok]; margp = []
for i in range(8):
    off = np.delete(Dp_ok[:, :, i], 8 + i, axis=-1).min(-1); margp.append(float(np.median((off - Dp_ok[:, :, i, 8 + i]) / Dp_ok[:, :, i, 8 + i])))
R["Dp_rel_margin_median"] = r4(margp)
# independent recompute of null-offset-corrected alignment
Dn = np.load(f"{I}/D_pn.npy"); Doa = np.load(f"{I}/D_oa.npy")
def align(X, N):
    out = {}
    for di, dn in enumerate(("forward", "reversed")):
        okr = np.isfinite(X[:, di]).all((1, 2)) & np.isfinite(N[:, di]).all((1, 2))
        js = (X[okr, di] - np.nanmean(N[okr, di], 0, keepdims=True)).argmin(-1)
        out[dn] = [round(float((js[:, i] == 8 + i).mean()), 3) for i in range(8)]
    return out
R["align_recomputed"] = dict(p=align(Dp, Dn), null=align(Dn, Dn), a07=align(Doa[:, :, 0], Dn), a08=align(Doa[:, :, 1], Dn), oe=align(Doe, Dn))
json.dump(R, open("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/SSV2T/s1_core.json", "w"), indent=1, default=float)
pr("DONE", f"{time.time()-t0:.0f}s")
