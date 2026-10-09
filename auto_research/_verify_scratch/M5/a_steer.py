# M5 적대 검증 (r2) — steer.npy 독립 재계산 + 자연 속도 이득 + 물체다움 조건부. 문서 스크립트 import 안 함.
import csv, json, os, sys
import numpy as np

I = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m5/"
IDX = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v3/index_probe.csv"
OUT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_verify_scratch/M5/a_steer.json"
RANK0 = os.environ.get("SLURM_PROCID", "0") == "0"
CELL, G = 18.0, 16
meta = json.load(open(I + "meta.json")); arms = meta["arms"]; steps = meta["steps"]; keys = meta["keys"]; K = meta["K"]
S = np.load(I + "steer.npy").astype(np.float64)       # (n, arms, K, 4) = y, x, maxcos, median
n = S.shape[0]
rows = {r["video_id"]: r for r in csv.DictReader(open(IDX))}
fl = lambda s: np.array([float(v) for v in s.split()])
ids = meta["video_ids"]
PX = np.stack([fl(rows[v]["px_x_by_sample"]) for v in ids]); PY = np.stack([fl(rows[v]["px_y_by_sample"]) for v in ids])
law = np.array([rows[v]["scenario"] for v in ids])
traj = np.array(["|".join(t) for t in meta["traj"]]); UT = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in UT}
assert all(rows[v]["plausible"] == "1" for v in ids)
cell = lambda f: np.clip(np.floor(np.stack([(PX[:, f] + PX[:, f + 1]) / 2, (PY[:, f] + PY[:, f + 1]) / 2], -1) / CELL), 0, G - 1)   # (n,2) x,y
L = cell(30); T = np.stack([cell(32 + 2 * t) for t in range(K)], 1)     # (n,K,2)
vx = (PX[:, 31] - PX[:, 27]) / 4; vy = (PY[:, 31] - PY[:, 27]) / 4
rng = np.random.default_rng(0)
B = 2000
bidx = [np.concatenate([tix[t] for t in rng.choice(UT, len(UT))]) for _ in range(B)]
res = {"n": n, "n_traj": len(UT), "laws": {l: int((law == l).sum()) for l in sorted(set(law))},
       "traj_per_law": {l: int(len({t for t in UT if t.startswith(l + "|")})) for l in sorted(set(law))}}


def ci(v, mask=None):
    v = np.asarray(v, float)
    if mask is None: mask = np.ones(len(v), bool)
    m = np.nanmean(np.where(mask, v, np.nan))
    bs = [np.nanmean(np.where(mask[b], v[b], np.nan)) for b in bidx]
    return [round(float(m), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


def ratio_ci(num, den_const, mask=None):
    r = ci(num, mask); return [round(x / den_const, 3) for x in r]


b = arms.index("base")
bx, by = S[:, b, :, 1], S[:, b, :, 0]
con = S[..., 2] - S[..., 3]
# 1) 문서 표 재계산 (arm 별 gain = mean dx / 기대)
tab = {}
for ai, (nm, s, k) in enumerate(zip(arms, steps, keys)):
    if ai == b: continue
    dx = S[:, ai, :, 1] - bx; dy = S[:, ai, :, 0] - by
    r = []
    for t in range(K):
        e = s * (2 * t + 2) / 18.0
        d = dict(t=t, dx=ci(dx[:, t]), dy=ci(dy[:, t]), absd=ci(np.hypot(dx[:, t], dy[:, t])),
                 moved=round(float((np.hypot(dx[:, t], dy[:, t]) > 0).mean()), 3), dcon=ci(con[:, ai, t] - con[:, b, t]))
        if k == "x": d["gain"] = [round(x / e, 3) for x in d["dx"]]
        if k == "y": d["gain"] = [round(x / e, 3) for x in d["dy"]]
        r.append(d)
    tab[nm] = r
res["arm_table"] = tab
# 2) 반대칭 (속도 특이) / 대칭 (비특이) 성분
anti = {}
for s in (1.5, 3.0):
    p, m = arms.index(f"vx+{s:g}"), arms.index(f"vx-{s:g}")
    A = (S[:, p, :, 1] - S[:, m, :, 1]) / 2; Sy = (S[:, p, :, 1] + S[:, m, :, 1]) / 2 - bx
    anti[f"vx{s:g}"] = [dict(t=t, anti_dx=ci(A[:, t]), gain=ratio_ci(A[:, t], s * (2 * t + 2) / 18.0), sym_dx=ci(Sy[:, t])) for t in range(K)]
res["antisym"] = anti
# 3) base 는 무엇을 하나: t0 argmax == 마지막 문맥 칸 (복사) / 진실 칸 / 이동량
res["base"] = []
for t in range(K):
    bxy = np.stack([bx[:, t], by[:, t]], -1)
    cheb_tru = np.abs(bxy - T[:, t]).max(-1); cheb_L = np.abs(bxy - L).max(-1)
    res["base"].append(dict(t=t, eq_last=round(float((cheb_L == 0).mean()), 3), hit1_truth=round(float((cheb_tru <= 1).mean()), 3),
                            eq_truth=round(float((cheb_tru == 0).mean()), 3), con_mean=round(float(con[:, b, t].mean()), 3),
                            con_p10=round(float(np.percentile(con[:, b, t], 10)), 3), frac_con_ge_0357=round(float((con[:, b, t] >= 0.357).mean()), 3)))
# 4) 자연 속도 이득 — base 의 Δx (argmax − 마지막 문맥 칸) 를 자연 vx 에 회귀 (법칙 고정효과, 궤적 평균 단위, 궤적 bootstrap)
#    같은 단위: gain = 기울기 (칸 per px/프레임) / ((2t+2)/18).  진실 칸도 같은 식으로 (천장).
laws_all = sorted(set(law))


def fe_slope(y, x, lawset, mask=None):
    if mask is None: mask = np.ones(len(y), bool)
    keep = mask & np.isin(law, lawset)
    tt = traj[keep]; ut = np.unique(tt)
    ym = np.array([y[keep][tt == u].mean() for u in ut]); xm = np.array([x[keep][tt == u].mean() for u in ut])
    tl = np.array([u.split("|")[0] for u in ut]); Ls = sorted(set(tl))
    A = np.concatenate([np.stack([(tl == l).astype(float) for l in Ls], 1), xm[:, None]], 1)
    co = np.linalg.lstsq(A, ym, rcond=None)[0][-1]
    r2 = np.random.default_rng(1); bs = []
    for _ in range(1000):
        sidx = r2.integers(0, len(ut), len(ut))
        if len(set(tl[sidx])) < len(Ls): continue
        bs.append(np.linalg.lstsq(A[sidx], ym[sidx], rcond=None)[0][-1])
    return float(co), np.percentile(bs, [2.5, 97.5]), len(ut)


nat = {}
for lawset_nm, lawset in (("all6", laws_all), ("flat3", ["flat_v", "flat_a", "flat_d"]), ("no_arc5", [l for l in laws_all if l != "arc"])):
    rr = []
    for t in range(K):
        e = (2 * t + 2) / 18.0
        bp, bpci, nu = fe_slope(bx[:, t] - L[:, 0], vx, lawset)
        bt, btci, _ = fe_slope(T[:, t, 0] - L[:, 0], vx, lawset)
        rr.append(dict(t=t, p_gain=[round(bp / e, 3)] + [round(v / e, 3) for v in bpci], truth_gain=[round(bt / e, 3)] + [round(v / e, 3) for v in btci], n_traj=nu))
    nat[lawset_nm] = rr
res["natural_gain_x"] = nat
# 5) 조건부: base 물체다움 ≥ 0.357 / base 가 진실 1 칸 안 (추적 중) 인 clip 만의 조향 반대칭 이득과 자연 이득
cond = {}
for s in (3.0,):
    p, m = arms.index(f"vx+{s:g}"), arms.index(f"vx-{s:g}")
    A = (S[:, p, :, 1] - S[:, m, :, 1]) / 2
    rr = []
    for t in range(K):
        e = s * (2 * t + 2) / 18.0
        mk_c = con[:, b, t] >= 0.357
        mk_h = np.abs(np.stack([bx[:, t], by[:, t]], -1) - T[:, t]).max(-1) <= 1
        mk_mv = np.abs(T[:, t, 0] - L[:, 0]) >= 1      # 진실이 1 칸 이상 수평 이동한 clip
        d = dict(t=t, n_con=int(mk_c.sum()), gain_con=ratio_ci(A[:, t], e, mk_c), n_hit=int(mk_h.sum()), gain_hit=ratio_ci(A[:, t], e, mk_h),
                 n_both=int((mk_c & mk_h).sum()), gain_both=ratio_ci(A[:, t], e, mk_c & mk_h))
        ee = (2 * t + 2) / 18.0
        try:
            bp, bpci, nu = fe_slope(bx[:, t] - L[:, 0], vx, laws_all, mk_c & mk_h)
            d["nat_gain_both"] = [round(bp / ee, 3)] + [round(v / ee, 3) for v in bpci]
        except Exception as ex:
            d["nat_gain_both"] = str(ex)
        rr.append(d)
    cond[f"vx{s:g}"] = rr
res["conditional"] = cond
# 6) 무작위 방향 대조
res["random"] = {nm: [dict(t=t, absd=ci(np.hypot(S[:, arms.index(nm), t, 1] - bx[:, t], S[:, arms.index(nm), t, 0] - by[:, t])),
                           dx=ci(S[:, arms.index(nm), t, 1] - bx[:, t]), moved=round(float((np.hypot(S[:, arms.index(nm), t, 1] - bx[:, t], S[:, arms.index(nm), t, 0] - by[:, t]) > 0).mean()), 3)) for t in range(K)] for nm in ("rand1", "rand2")}
# 7) 조향 이동을 자연 속도 차이로 환산: t0 반대칭 이동 A0 (칸) 을 자연 기울기로 나누면 "몇 px/프레임 짜리 자연 속도 변화" 인가
res["vx_stats"] = dict(vx_sd=round(float(vx.std()), 3), vx_absmean=round(float(np.abs(vx).mean()), 3), vx_p5_p95=[round(float(v), 3) for v in np.percentile(vx, [5, 95])],
                       vx_sd_within_law=round(float(np.sqrt(np.mean([(vx[law == l] - vx[law == l].mean()).var() for l in laws_all]))), 3),
                       vy_sd=round(float(vy.std()), 3))
if RANK0:
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n", "n_traj", "laws", "traj_per_law", "vx_stats")}))
    print("\n== arm gains (recomputed) t0..7")
    for nm, r in tab.items():
        g = " ".join(f"{d['gain'][0]:+.3f}" if "gain" in d else "  -   " for d in r)
        print(f"{nm:7s} gain {g} | absd " + " ".join(f"{d['absd'][0]:.3f}" for d in r) + " | moved " + " ".join(f"{d['moved']:.2f}" for d in r))
    print("vx+3 dx:", [d["dx"] for d in tab["vx+3"]])
    print("dcon max abs:", max(abs(d["dcon"][0]) for r in tab.values() for d in r))
    print("\n== antisym")
    for k_, r in anti.items():
        print(k_, " ".join(f"t{d['t']} {d['gain'][0]:+.3f}[{d['gain'][1]:+.3f},{d['gain'][2]:+.3f}] sym{d['sym_dx'][0]:+.3f}" for d in r))
    print("\n== base")
    for d in res["base"]: print(d)
    print("\n== natural gain (x, per px/frame, law FE, traj-level)")
    for k_, r in nat.items():
        print(k_, " | ".join(f"t{d['t']} p {d['p_gain'][0]:+.2f}[{d['p_gain'][1]:+.2f},{d['p_gain'][2]:+.2f}] tru {d['truth_gain'][0]:+.2f}" for d in r))
    print("\n== conditional (vx±3 antisym)")
    for d in cond["vx3"]: print(d)
    print("\n== random")
    for nm, r in res["random"].items(): print(nm, [(d["absd"][0], d["dx"][0], d["moved"]) for d in r])
