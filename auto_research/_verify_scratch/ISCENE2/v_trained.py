"""(M) release post-FT T0/T1: R50 + CI, 같은 토큰 짝 Δ (T1−T0, T0−R, T1−R) + CI."""
import json, numpy as np
from vcommon import *
rng = np.random.default_rng(5)
res = {}
for name in ("scene_pan_trained", "scene_nat_ssv2_trained", "scene_nat_ek100_trained"):
    R = Reach(name); rec = {}
    for pk in ("R", "T0", "T1", "A"):
        h = R.hit(f"{pk}:base"); rec[pk] = dict(R50=r50_boot(h, R.HE, R.D, rng, B=300), A=[c["A"] for c in curve(h, R.HE, R.D, rng, B=200)])
        if R.V is not None: rec[pk]["per_speed_R50"] = {int(v): r50_boot(h, R.HE & (R.V == v)[:, None, None], R.D, rng, B=100)[0] for v in (2, 4, 8, 12)}
    for a, b in (("T1", "T0"), ("T0", "R"), ("T1", "R"), ("T0", "A")):
        rec[f"{a}-{b}"] = [c["d"] for c in paired_delta(R.hit(f"{a}:base"), R.hit(f"{b}:base"), R.HE, R.D, rng, B=300)]
        # R50 차이 bootstrap (같은 clip 재표본)
        nums = {pk: np.stack([(R.hit(f"{pk}:base") & R.HE & (R.D >= lo) & (R.D < hi)).sum((1, 2)) for lo, hi in BINS], 1).astype(float) for pk in (a, b)}
        dens = np.stack([(R.HE & (R.D >= lo) & (R.D < hi)).sum((1, 2)) for lo, hi in BINS], 1).astype(float); idx = np.where(dens.sum(1) > 0)[0]
        def dr(ii):
            o = []
            for pk in (a, b):
                A = nums[pk][ii].sum(0) / np.maximum(dens[ii].sum(0), 1); A[dens[ii].sum(0) == 0] = np.nan; o.append(r50_from_A(A))
            return o[0] - o[1]
        bs = np.array([dr(rng.choice(idx, len(idx))) for _ in range(200)]); bs = bs[np.isfinite(bs)]
        rec[f"R50_{a}-{b}"] = [round(dr(idx), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    res[name] = rec
dump("trained", res)
for name, rec in res.items():
    print("==", name)
    for k, r in rec.items():
        if isinstance(r, dict): print(f"  {k:3s} R50 {r['R50']}  A " + " ".join(f"{x[0]:.2f}" for x in r["A"]), r.get("per_speed_R50", ""))
        elif k.startswith("R50_"): print("  ", k, r)
        else: print("  Δ", k, " ".join(f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for x in r))
