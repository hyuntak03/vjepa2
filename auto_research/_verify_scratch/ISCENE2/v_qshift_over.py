"""qshift 출력의 argmax 가 출처 기준 어디에 놓이나 (변위 방향 투영 t: 0 = 자기 칸, 1 = 출처, >1 = 출처 너머), 원거리 4–13 칸 합의 토큰."""
import json, numpy as np
from vcommon import *
res = {}
for name in ("scene_pan_cm", "scene_nat_ssv2_cm", "scene_nat_ek100_cm", "scene_nat_v3_cm"):
    R = Reach(name); rec = {}
    for a in ("R:base", "R:bias4", "R:qshift", "A:base", "A:qshift"):
        if a not in R.ai: continue
        c = R.C[:, R.ai[a]]; sx = np.arange(256)[None, None, :] % G; sy = np.arange(256)[None, None, :] // G; ux, uy = R.SRC % G, R.SRC // G; cx, cy = c % G, c // G
        dd2 = np.maximum((ux - sx) ** 2 + (uy - sy) ** 2, 1); t = ((cx - sx) * (ux - sx) + (cy - sy) * (uy - sy)) / dd2
        perp = np.abs((cx - sx) * (uy - sy) - (cy - sy) * (ux - sx)) / np.sqrt(dd2)
        for lo, hi, tag in ((1, 4, "d1-4"), (4, 13, "d4-13")):
            m = R.HE & (R.D >= lo) & (R.D < hi); hit1 = cheb(c, R.SRC) <= 1
            rec[f"{a}_{tag}"] = dict(n=int(m.sum()), hit1=round(float(hit1[m].mean()), 3), exact=round(float((cheb(c, R.SRC) == 0)[m].mean()), 3),
                                     t_median=round(float(np.median(t[m])), 2), t_mean=round(float(t[m].mean()), 2), frac_t_gt_1p1=round(float((t[m] > 1.1).mean()), 3), frac_t_lt_0p9_and_gt_0p1=round(float(((t[m] < 0.9) & (t[m] > 0.1)).mean()), 3),
                                     frac_t_le_0p1=round(float((t[m] <= 0.1).mean()), 3), perp_median=round(float(np.median(perp[m])), 2),
                                     hit1_but_not_exact_beyond=round(float(((hit1) & (t > 1.0) & (cheb(c, R.SRC) > 0))[m].mean()), 3), hit1_but_not_exact_short=round(float(((hit1) & (t < 1.0) & (cheb(c, R.SRC) > 0))[m].mean()), 3))
    res[name] = rec
dump("qshift_over", res)
for name, rec in res.items():
    print("==", name)
    for a, r in rec.items(): print(f"  {a:18s}", r)
