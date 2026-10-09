"""(R) Ariel epoch 포화: R50 + CI (epoch 5/19/30/41/43), per-speed, 포화 곡선 적합 + 점근 CI, lr 스케줄 교란."""
import json, numpy as np
from vcommon import *
rng = np.random.default_rng(4)
EP = {"E5": 5, "E19": 19, "E30": 30, "E41": 41, "A": 43}
res = {}
def fit_sat(e, r):
    """R(e) = Rinf − (Rinf − R0) exp(−e/τ): τ 격자 → 선형 최소제곱 (Rinf, R0)."""
    best = None
    for tau in np.linspace(2, 200, 400):
        x = np.exp(-np.asarray(e) / tau); X = np.stack([1 - x, x], 1); b, *_ = np.linalg.lstsq(X, r, rcond=None); rss = float(((X @ b - r) ** 2).sum())
        if best is None or rss < best[0]: best = (rss, tau, b[0], b[1])
    return best
def fit_log(e, r):
    X = np.stack([np.ones(len(e)), np.log(e)], 1); b, *_ = np.linalg.lstsq(X, r, rcond=None); return b
for name in ("scene_pan_arielep", "scene_nat_ssv2_arielep", "scene_nat_ek100_arielep"):
    R = Reach(name); rec = dict(epochs={}, fit={})
    bins_nums = {}; 
    for pk, ep in EP.items():
        h = R.hit(f"{pk}:base"); rec["epochs"][ep] = dict(R50=r50_boot(h, R.HE, R.D, rng, B=300), A=[c["A"][0] for c in curve(h, R.HE, R.D, rng, B=2)])
        if R.V is not None:
            rec["epochs"][ep]["per_speed_A46_69"] = {int(v): [c["A"][0] for c in curve(h, R.HE & (R.V == v)[:, None, None], R.D, rng, B=2)][4:6] for v in (2, 4, 8, 12)}
            rec["epochs"][ep]["per_speed_R50"] = {int(v): r50_boot(h, R.HE & (R.V == v)[:, None, None], R.D, rng, B=100)[0] for v in (2, 4, 8, 12)}
    # 인접 epoch 짝 차이 (같은 토큰) 4–6 · 6–9 칸, CI
    rec["paired_delta"] = {}
    order = list(EP.keys())
    for a, b in zip(order[:-1], order[1:]):
        rec["paired_delta"][f"{EP[b]}-{EP[a]}"] = [c["d"] for c in paired_delta(R.hit(f"{b}:base"), R.hit(f"{a}:base"), R.HE, R.D, rng, B=300)][3:6]
    # 적합 + clip bootstrap (R50 을 재표본마다 다시 계산)
    e = np.array(list(EP.values()), float)
    nums = {pk: np.stack([(R.hit(f"{pk}:base") & R.HE & (R.D >= lo) & (R.D < hi)).sum((1, 2)) for lo, hi in BINS], 1).astype(float) for pk in EP}
    dens = np.stack([(R.HE & (R.D >= lo) & (R.D < hi)).sum((1, 2)) for lo, hi in BINS], 1).astype(float)
    def r50s(idx):
        out = []
        for pk in EP:
            A = nums[pk][idx].sum(0) / np.maximum(dens[idx].sum(0), 1); A[dens[idx].sum(0) == 0] = np.nan; out.append(r50_from_A(A))
        return np.array(out)
    idx = np.where(dens.sum(1) > 0)[0]; r0 = r50s(idx)
    if np.all(np.isfinite(r0)):
        s0 = fit_sat(e, r0); l0 = fit_log(e, r0); bs_s, bs_l = [], []
        for _ in range(200):
            rb = r50s(rng.choice(idx, len(idx)))
            if not np.all(np.isfinite(rb)): continue
            bs_s.append(fit_sat(e, rb)[2]); bs_l.append(fit_log(e, rb)[1])
        rec["fit"] = dict(R50_points=[round(float(x), 3) for x in r0], sat_tau=round(s0[1], 1), sat_Rinf=[round(s0[2], 3), round(float(np.percentile(bs_s, 2.5)), 3), round(float(np.percentile(bs_s, 97.5)), 3)], sat_R0=round(s0[3], 3), sat_rss=round(s0[0], 5),
                          sat_R_at_90=round(float(s0[2] - (s0[2] - s0[3]) * np.exp(-90 / s0[1])), 3),
                          log_slope_per_ln_epoch=[round(float(l0[1]), 3), round(float(np.percentile(bs_l, 2.5)), 3), round(float(np.percentile(bs_l, 97.5)), 3)], log_R_at_90=round(float(l0[0] + l0[1] * np.log(90)), 3),
                          log_rss=round(float(((np.stack([np.ones(5), np.log(e)], 1) @ l0 - r0) ** 2).sum()), 5))
    res[name] = rec
# lr 스케줄 (체크포인트 config: lr 3e-4, start 1e-5, final 1e-6, warmup 1, epochs 45 → cosine)
def lr_at(ep, peak=3e-4, final=1e-6, warm=1, T=45):
    if ep < warm: return 1e-5 + (peak - 1e-5) * ep / warm
    return final + 0.5 * (peak - final) * (1 + np.cos(np.pi * (ep - warm) / (T - warm)))
res["lr_schedule"] = {ep: float(f"{lr_at(ep):.3g}") for ep in (5, 19, 30, 41, 43)}
res["lr_cum_frac"] = {ep: round(float(sum(lr_at(x) for x in range(ep)) / sum(lr_at(x) for x in range(45))), 3) for ep in (5, 19, 30, 41, 43)}
dump("epoch", res)
for name, rec in res.items():
    if not isinstance(rec, dict) or "epochs" not in rec: print(name, rec); continue
    print("==", name)
    for ep, r in rec["epochs"].items(): print(f"  ep{ep:3d} R50 {r['R50']}  A " + " ".join(f"{x:.2f}" for x in r["A"]), r.get("per_speed_R50", ""))
    print("  paired Δ (3-4,4-6,6-9):", rec["paired_delta"]); print("  fit:", rec["fit"])
