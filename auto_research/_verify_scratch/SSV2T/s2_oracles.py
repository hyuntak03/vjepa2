#!/usr/bin/env python3
"""SSV2T verifier part 2: smooth-trend ceiling for token beta/rho/gain, autocorr structure of p & h,
proj-space time alignment with (i) independent-error oracle, (ii) constant-a copy-shrinkage, (iii) error-magnitude-matched
copy-shrinkage (per clip, per slot), (iv) trend oracle. Own null offsets per predictor."""
import json, time
import numpy as np
t0 = time.time()
I = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e3"
meta = json.load(open(f"{I}/meta.json")); n = meta["n"]
PR = np.load(f"{I}/proj.npy", mmap_mode="r")
Dp_full = np.load(f"{I}/D_p.npy")
R = {}; rng = np.random.default_rng(5)
r4 = lambda x: np.round(np.asarray(x, dtype=float), 4).tolist()
def pr(*a): print(*a, flush=True)

def tr_stats(Pp, H):
    dP = np.diff(Pp, axis=1); dH = np.diff(H, axis=1)
    dP = dP - dP.mean(0, keepdims=True); dH = dH - dH.mean(0, keepdims=True)
    ax = (0,) + tuple(range(2, dP.ndim))
    num = (dP * dH).sum(ax, dtype=np.float64); nP = (dP * dP).sum(ax, dtype=np.float64); nH = (dH * dH).sum(ax, dtype=np.float64)
    return dict(beta=r4(num / nH), rho=r4(num / np.sqrt(nP * nH)), gain=r4(np.sqrt(nP / nH)))

def level_sim(X):
    """clip-centre per slot, then within-clip deviation from slot mean; corr matrix across slots."""
    Xc = X - X.mean(0, keepdims=True); Xd = Xc - Xc.mean(1, keepdims=True)
    k = X.shape[1]; S = np.zeros((k, k))
    nrm = np.array([np.sqrt((Xd[:, i].astype(np.float64) ** 2).sum()) for i in range(k)])
    for i in range(k):
        for j in range(k):
            S[i, j] = (Xd[:, i].astype(np.float64) * Xd[:, j]).sum() / (nrm[i] * nrm[j])
    return np.round(S, 3).tolist()

def dauto(X):
    d = np.diff(X, axis=1); d = d - d.mean(0, keepdims=True); k = d.shape[1]; S = np.zeros((k, k))
    nrm = np.array([np.sqrt((d[:, i].astype(np.float64) ** 2).sum()) for i in range(k)])
    for i in range(k):
        for j in range(k):
            S[i, j] = (d[:, i].astype(np.float64) * d[:, j]).sum() / (nrm[i] * nrm[j])
    return np.round(S, 3).tolist()

for di, dn in enumerate(("forward", "reversed")):
    A = np.asarray(PR[:, di], np.float32)
    H, P, HC = A[:, 8:16], A[:, 16:24], A[:, 24:25]
    rec = {}
    # (a)(b) structure
    Pg = P.reshape(n, 8, 16, 16, 64); Hg = H.reshape(n, 8, 16, 16, 64)
    for k in (1, 4, 16):
        Pk = Pg.reshape(n, 8, 16 // k, k, 16 // k, k, 64).mean((3, 5)); Hk = Hg.reshape(n, 8, 16 // k, k, 16 // k, k, 64).mean((3, 5))
        rec[f"block{k}"] = dict(p_level_sim=level_sim(Pk), h_level_sim=level_sim(Hk), p_change_autocorr=dauto(Pk), h_change_autocorr=dauto(Hk))
    # (c) smooth-trend ceilings (use the true future -> ceilings, not predictors)
    t = np.arange(8, dtype=np.float32); tc = t - t.mean()
    Hm = H.mean(1, keepdims=True); b = np.einsum('i,nitd->ntd', tc, H) / (tc ** 2).sum()
    Olin = Hm + tc[None, :, None, None] * b[:, None]
    V = np.stack([np.ones(8), tc, tc ** 2 - (tc ** 2).mean()], 1).astype(np.float32)   # (8,3)
    coef = np.einsum('ik,nitd->nktd', np.linalg.pinv(V).T.astype(np.float32).T @ np.eye(3, dtype=np.float32), H) if False else None
    Pinv = np.linalg.pinv(V).astype(np.float32)                                              # (3,8)
    C3 = np.einsum('ki,nitd->nktd', Pinv, H); Oquad = np.einsum('ik,nktd->nitd', V, C3)
    Hpad = np.concatenate([H[:, :1], H, H[:, -1:]], 1); Oma3 = (Hpad[:, :-2] + Hpad[:, 1:-1] + Hpad[:, 2:]) / 3
    rec["ceiling_linear_trend"] = tr_stats(Olin, H); rec["ceiling_quadratic_trend"] = tr_stats(Oquad, H); rec["ceiling_ma3"] = tr_stats(Oma3, H)
    rec["p"] = tr_stats(P, H)
    del Olin, Oquad, Oma3, C3, Hpad, Pg, Hg
    # (d) proj-space alignment on approx dyn tokens (top-64 by projected-h temporal variance), L1 over 64 dims
    var = A[:, 0:16].var(1).mean(-1); order = np.argsort(-var, 1)[:, :64]
    take = lambda X: np.take_along_axis(X, order[:, None, :, None], 2)
    Hall = take(A[:, 0:16]); Pd = take(P); Hd = Hall[:, 8:16]; HCd = take(HC)                # (n,16,64,64) etc
    def Dmat(X, Hh):   # X (n,8,T,d), Hh (n,16,T,d) -> (n,8,16) mean L1
        out = np.empty((X.shape[0], 8, 16), np.float32)
        for s in range(0, X.shape[0], 40):
            out[s:s + 40] = np.abs(X[s:s + 40, :, None] - Hh[s:s + 40, None]).mean((-1, -2))
        return out
    der = (np.arange(n) + 1 + rng.integers(0, n - 1, n)) % n; der = np.where(der == np.arange(n), (der + 1) % n, der)
    def aligned(X, own_null=True, Xoff=None):
        D = Dmat(X, Hall)
        if Xoff is None: Xoff = X
        Dn = np.empty_like(D)
        for s in range(0, n, 40):   # null: X of clip A vs h of clip B (A's dyn tokens applied to B = take with A's order)
            idx = np.arange(s, min(s + 40, n))
            HB = np.take_along_axis(A[der[idx], 0:16], order[idx][:, None, :, None], 2)
            Dn[idx] = np.abs(Xoff[idx][:, :, None] - HB[:, None]).mean((-1, -2))
        raw = D.argmin(-1); cor = (D - Dn.mean(0, keepdims=True)).argmin(-1)
        return D, dict(diag_raw=r4([(raw[:, i] == 8 + i).mean() for i in range(8)]), diag_nullcorr=r4([(cor[:, i] == 8 + i).mean() for i in range(8)]),
                       mean_diag_L1=r4(np.stack([D[:, i, 8 + i] for i in range(8)], 1).mean(0)))
    Dp_proj, rec["align_p"] = aligned(Pd)
    # agreement with full-D token argmin (D_p, true dyn tokens)
    js_full = Dp_full[:, di].argmin(-1); js_proj = Dp_proj.argmin(-1)
    rec["align_p_argmin_agreement_with_fullD"] = r4((js_full == js_proj).mean(0))
    # (i) independent error oracle
    Eb = Pd[der] - Hd[der]           # other clip's error on (its own) top tokens -- mimic by taking B's error at A's token set
    EbA = np.take_along_axis(A[der, 16:24] - A[der, 8:16], order[:, None, :, None], 2)
    _, rec["align_oracle_truth_plus_otherclip_error"] = aligned(Hd + EbA)
    # (ii) constant-a shrinkage
    for a_ in (0.7, 0.75, 0.8):
        _, rec[f"align_oracle_copy_a{a_}"] = aligned((1 - a_) * Hd + a_ * HCd)
    # (iii) magnitude-matched shrinkage: a[c,i] = L1(p_i - h_{8+i}) / L1(hc7 - h_{8+i})
    Lp = np.abs(Pd - Hd).mean((-1, -2)); Lc = np.abs(HCd - Hd).mean((-1, -2)); am = (Lp / Lc)
    rec["matched_a_mean_by_slot"] = r4(am.mean(0)); rec["matched_a_quartiles_slot0_7"] = [r4(np.percentile(am[:, 0], [25, 50, 75])), r4(np.percentile(am[:, 7], [25, 50, 75]))]
    Om = (1 - am[:, :, None, None]) * Hd + am[:, :, None, None] * HCd
    _, rec["align_oracle_copy_matched_a"] = aligned(Om)
    rec["matched_oracle_tokenbeta_alltokens_via_dyn"] = tr_stats(Om, Hd)
    rec["p_tokenbeta_dyn"] = tr_stats(Pd, Hd)
    # (iv) trend oracle alignment (linear trend fit to true future): a smooth-but-progressing predictor
    tt = np.arange(8, dtype=np.float32) - 3.5
    bb = np.einsum('i,nitd->ntd', tt, Hd) / (tt ** 2).sum(); Ol = Hd.mean(1, keepdims=True) + tt[None, :, None, None] * bb[:, None]
    _, rec["align_oracle_linear_trend"] = aligned(Ol)
    R[dn] = rec
    pr(f"[{dn}] {time.time()-t0:.0f}s")
    for k in ("ceiling_linear_trend", "ceiling_quadratic_trend", "ceiling_ma3", "p", "align_p", "align_p_argmin_agreement_with_fullD",
              "align_oracle_truth_plus_otherclip_error", "align_oracle_copy_a0.7", "align_oracle_copy_a0.75", "align_oracle_copy_a0.8",
              "matched_a_mean_by_slot", "align_oracle_copy_matched_a", "matched_oracle_tokenbeta_alltokens_via_dyn", "p_tokenbeta_dyn", "align_oracle_linear_trend"):
        pr(" ", k, rec[k])
    del A, H, P, HC, Hall, Pd, Hd, HCd, Om, Ol
json.dump(R, open("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/SSV2T/s2_oracles.json", "w"), indent=1, default=float)
pr("DONE")
