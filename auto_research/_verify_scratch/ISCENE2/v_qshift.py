"""(Q) qshift 재계산: A(d) + CI, Δ vs base (근거리 비용 · 원거리 이득), v3 물체 칸 argmax 행방, '정체 = 자리' 검정."""
import json, numpy as np
from vcommon import *
rng = np.random.default_rng(3)
res = {}
for name in ("scene_pan_cm", "scene_nat_ssv2_cm", "scene_nat_ek100_cm", "scene_nat_v3_cm"):
    R = Reach(name); rec = {}
    for pk in ("R", "A"):
        arms = [a for a in (f"{pk}:base", f"{pk}:bias4", f"{pk}:qshift", f"{pk}:qshift_cv") if a in R.ai]
        for a in arms:
            h = R.hit(a); rec[a] = dict(A=[c["A"] for c in curve(h, R.HE, R.D, rng)], A_exact=[c["A"][0] for c in curve(R.hit(a, 0), R.HE, R.D, rng, B=2)],
                                        R50=r50_boot(h, R.HE, R.D, rng, B=200))
            if a != f"{pk}:base":
                rec[a]["delta_vs_base"] = [c["d"] for c in paired_delta(h, R.hit(f"{pk}:base"), R.HE, R.D, rng)]
                # 정확 칸 판독의 Δ
                rec[a]["delta_exact_vs_base"] = [c["d"][0] for c in paired_delta(R.hit(a, 0), R.hit(f"{pk}:base", 0), R.HE, R.D, rng, B=2)]
                # own-cell (복사) 비율 d≥1 & 합의
                rec[a]["own_frac_d>=1"] = round(float(R.own(a)[R.HE & (R.D >= 1)].mean()), 4)
        rec[f"{pk}:base"]["own_frac_d>=1"] = round(float(R.own(f"{pk}:base")[R.HE & (R.D >= 1)].mean()), 4)
        # 정체 = 자리 검정: qshift 의 원거리 (4–13 칸) A 와 base 의 근거리 (0–1 칸) A 비교 (합의)
        if f"{pk}:qshift" in R.ai:
            far = R.HE & (R.D >= 4) & (R.D < 13); near = R.HE & (R.D < 1)
            rec[f"{pk}:identity_test"] = dict(qshift_far=rate_ci((R.hit(f"{pk}:qshift") & far).sum((1, 2)).astype(float), far.sum((1, 2)).astype(float), rng),
                                              base_near=rate_ci((R.hit(f"{pk}:base") & near).sum((1, 2)).astype(float), near.sum((1, 2)).astype(float), rng),
                                              qshift_far_exact=rate_ci((R.hit(f"{pk}:qshift", 0) & far).sum((1, 2)).astype(float), far.sum((1, 2)).astype(float), rng, B=50),
                                              base_near_exact=rate_ci((R.hit(f"{pk}:base", 0) & near).sum((1, 2)).astype(float), near.sum((1, 2)).astype(float), rng, B=50),
                                              bias4_far=rate_ci((R.hit(f"{pk}:bias4") & far).sum((1, 2)).astype(float), far.sum((1, 2)).astype(float), rng, B=50))
    # v3 물체 칸: 이동 토큰 (D ≥ 1) 의 argmax 행방 — 자기 칸 / 출처 ≤1 / 그 밖 (배경) ; 배경 토큰 (D == 0) 은 qshift 가 안 옮기므로 base 와 같아야
    if name.endswith("v3_cm"):
        for a in ("R:base", "R:qshift", "R:bias4", "A:base", "A:qshift"):
            mo = R.VAL & (R.D >= 1); c = R.C[:, R.ai[a]]
            src_hit = cheb(c, R.SRC) <= 1; own = c == np.arange(256)[None, None, :]
            # 출처와 자기 칸 사이 (직선 상)? 대신 "출처보다 더 먼 (overshoot)" vs "출처와 자기 사이" 로 분류: x 좌표 투영
            sx, ux, cx = np.arange(256)[None, None, :] % G, R.SRC % G, c % G
            sy, uy, cy = np.arange(256)[None, None, :] // G, R.SRC // G, c // G
            t = ((cx - sx) * (ux - sx) + (cy - sy) * (uy - sy)) / np.maximum((ux - sx) ** 2 + (uy - sy) ** 2, 1)
            perp = cheb(c, R.SRC)
            rec[f"v3_object_{a}"] = dict(n=int(mo.sum()), src_le1=round(float(src_hit[mo].mean()), 3), own=round(float(own[mo].mean()), 3),
                                          between=round(float(((t > 0.1) & (t < 0.9) & ~src_hit & ~own)[mo].mean()), 3), beyond=round(float((t >= 0.9)[mo].mean() - src_hit[mo].mean()), 3),
                                          elsewhere=round(float(((t <= 0.1) & ~own)[mo].mean()), 3), mean_cheb_to_src=round(float(perp[mo].mean()), 2),
                                          agree_only_src_le1=round(float(src_hit[mo & R.HE].mean()), 3), n_agree=int((mo & R.HE).sum()))
            mb = R.VAL & (R.D == 0)
            rec[f"v3_background_{a}"] = dict(n=int(mb.sum()), src_le1=round(float(src_hit[mb].mean()), 4), own=round(float(own[mb].mean()), 4))
        # v3 물체 토큰: qshift 뒤 argmax 가 base 의 argmax 와 같은 비율
        mo = R.VAL & (R.D >= 1)
        rec["v3_object_qshift_eq_base_argmax"] = round(float((R.C[:, R.ai["R:qshift"]] == R.C[:, R.ai["R:base"]])[mo].mean()), 3)
    res[name] = rec
dump("qshift", res)
for name, rec in res.items():
    print("==", name)
    for a, r in rec.items():
        if isinstance(r, dict) and "A" in r: print(f"  {a:12s} A " + " ".join(f"{x[0]:.2f}[{x[1]:.2f},{x[2]:.2f}]" for x in r["A"]), " R50", r["R50"], " own", r.get("own_frac_d>=1"), "\n      Δ", " ".join(f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for x in r["delta_vs_base"]) if "delta_vs_base" in r else "", "\n      exact", [round(x, 2) for x in r["A_exact"]])
        else: print("  ", a, r)
