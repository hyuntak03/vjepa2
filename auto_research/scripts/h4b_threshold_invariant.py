#!/usr/bin/env python3
"""H4b — H4 적대 검증 대응: 문턱·템플릿에 불변인 비교 + 시나리오 안 속도 분석 (CPU, 파라미터 0, GPU 없음).

왜: H4 의 "단일 질의가 먼 슬롯까지 물체를 남긴다 (k10 0.43 vs 0.10)" 는 물체다움 문턱을 **모드마다 따로** 잡은 값이었다
(모드 자기 k ≤ 2 contrast 의 10 백분위 — joint 는 근거리 contrast 가 높아 문턱이 높다). 또 진실 시각 템플릿은
"물체가 있나" 와 "그 운동·외형 상태가 진실과 맞나" 를 섞는다. 그래서 같은 원자료 (v3_h4/h4.npy) 에서
(1) 문턱을 모드 사이에 공유하고, (2) 문턱 없이 raw contrast 를 clip 단위로 짝지어 비교하고, (3) 두 템플릿
(tru = 진실 시각 h 의 진실 칸 토큰, app = 문맥 마지막 튜블릿의 물체 토큰) 을 나란히 보고, (4) 진실 적중·L1 을 같은 비중으로 싣고,
(5) 속도 효과를 시나리오 (운동 법칙) 안에서 다시 잰다.

기호 (clip i, 창 C, 모드 m, 미래 슬롯 k):
  con_{i,k}     = max_j cos(p_{k,j}, tpl) − median_j cos(p_{k,j}, tpl)          (j = 256 토큰)
  thr_m         = P10{ con_{i,k} : i ∈ free, k ≤ 2 }  (모드 m 자기 값)   ← H4 원본 "모드별 문턱"
  thr_shared    = thr_single (같은 C · 템플릿)  — 모든 모드에 같이 적용         ← 공통 문턱 (주)
  thr_shared_j  = thr_joint                     — 민감도 (더 엄격)
  obj           = 1[con ≥ thr]
  hit           = 1[Chebyshev(argmax, 진실 칸) ≤ 1]        hit_last = 1[Chebyshev(argmax, 마지막 칸) ≤ 1]
  l1            = mean_j |p_{k,j} − h_{k,j}|  (전 토큰, 템플릿 무관)
  prog          = (argmax − 마지막)·(진실 − 마지막) / |진실 − 마지막|²   (|진실 − 마지막| ≥ 3 칸)
  짝 비교       = clip 마다 k 띠 안 평균 (a − b) → **궤적 군집 부트스트랩** 95 % CI (B = 1000), 그리고 frac(con_a > con_b)
                  ⚠️ 자유 운동 2,352 clip 은 궤적 84 개 × 28 변형 (모양 7 × 배경 4) 이다 — 위치·속도·거리가 궤적 안에서 같으므로
                  재표집 단위는 clip 이 아니라 궤적 (scenario, primary, secondary) 이다
  시나리오 LPM  = obj ~ 시나리오 FE + k FE + β·거리  (β: 같은 시나리오·같은 k 에서 거리 1 칸의 효과)
                  obj ~ 시나리오 FE + 거리 구간 FE + γ·k  (γ: 같은 시나리오·같은 거리 구간에서 1 튜블릿의 효과), 궤적 군집 부트스트랩 CI

  auto_research/scripts/srun6.sh 4 32G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h4b_threshold_invariant.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE  # noqa: E402

I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h4")
OUT = Path(__file__).resolve().parents[2] / "auto_research/exp_results/h4"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; Cs, K = meta["Cs"], meta["K"]
MODES = meta["modes"]
H = np.load(I / "h4.npy")                                   # (n, 4C, K, 4 mode, 9)
R, X, Y, OK = load_index(ids)
scen = np.array([r["scenario"] for r in R]); pl = np.array([r["plausible"] == "1" for r in R])
last_xy = tub_xy(X, Y, 30); last = cell(last_xy).astype(float)
v_end = np.linalg.norm(last_xy - tub_xy(X, Y, 28), axis=1)
free = np.isin(scen, FREE) & pl
q1, q2 = np.percentile(v_end[free], [33, 67]); spd = np.digitize(v_end, [q1, q2])
traj_key = [(r["scenario"], r["primary"], r["secondary"]) for r in R]
_u = {t: i for i, t in enumerate(sorted(set(traj_key)))}
traj = np.array([_u[t] for t in traj_key])
SPD = ("slow", "mid", "fast")
n = len(ids)
TPL = {"tru": (2, 3, 0, 1), "app": (6, 7, 4, 5)}
BANDS = [(0, 2), (3, 5), (6, 7), (8, 10), (11, 13), (14, 15)]
BN = [f"k{a}-{b}" for a, b in BANDS]
rng = np.random.default_rng(0)
NB = 1000

# 슬롯별 진실 칸 · 유효 마스크 · 거리
T = np.zeros((n, K, 2)); OKk = np.zeros((n, K), bool)
for k in range(K):
    f0 = 32 + 2 * k
    T[:, k] = cell(tub_xy(X, Y, f0)).astype(float)
    OKk[:, k] = free & OK[:, f0] & OK[:, f0 + 1]
dvec = T - last[:, None, :]; dd = (dvec ** 2).sum(-1); dist = np.sqrt(dd)


def fields(ci, m, tpl):
    a, b, yc, xc = TPL[tpl]
    V = H[:, ci, :, m]
    con = V[..., a] - V[..., b]
    am = np.stack([V[..., xc], V[..., yc]], -1)              # (n, K, 2) = x, y
    hit = np.abs(am - T).max(-1) <= 1
    hitl = np.abs(am - last[:, None]).max(-1) <= 1
    prog = ((am - last[:, None]) * dvec).sum(-1) / np.maximum(dd, 1)
    return dict(con=con, hit=hit.astype(float), hitl=hitl.astype(float), prog=prog, l1=V[..., 8])


def thr(con):
    return float(np.nanpercentile(con[free][:, :3], 10))


def avail(ci, m):
    return not np.isnan(H[:, ci, :, m, 0]).all()


def mask_group(g):
    return free if g == "all" else free & (spd == SPD.index(g))


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def band_clip(A, ok, lo, hi):
    """(n, K) → clip 별 띠 평균 (유효 k 없으면 nan)."""
    s = np.where(ok[:, lo:hi + 1], A[:, lo:hi + 1], 0).sum(1); c = ok[:, lo:hi + 1].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def boot_mean(v, g=None):
    """v: free clip 값 (nan = 없음), g: 같은 길이의 궤적 id → 궤적 군집 부트스트랩."""
    if g is None:
        g = traj[free]
    ok = np.isfinite(v); v = v[ok]; g = g[ok]
    if len(v) < 10:
        return None
    ug, inv = np.unique(g, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(ug)); N = np.bincount(inv, minlength=len(ug)).astype(float)
    idx = rng.integers(0, len(ug), (NB, len(ug)))
    bs = S[idx].sum(1) / N[idx].sum(1)
    return dict(mean=r3(v.mean()), lo=r3(np.percentile(bs, 2.5)), hi=r3(np.percentile(bs, 97.5)), n=int(len(v)), n_traj=int(len(ug)))


res = dict(meta=dict(n_free=int(free.sum()), n_traj=int(len(np.unique(traj[free]))), n_speed={g: int((free & (spd == i)).sum()) for i, g in enumerate(SPD)},
                     speed_cuts=[r3(q1), r3(q2)], bands=BN, n_boot=NB), thresholds={}, curves={}, paired={},
           levels={}, scenario={}, lpm={}, progress={})
lines = []

# ---------------- 1. 문턱 · 곡선 (모드별 / 공통 문턱 / raw contrast / 적중 / L1) ----------------
for ci, C in enumerate(Cs):
    for tpl in TPL:
        F = {m: fields(ci, mi, tpl) for mi, m in enumerate(MODES) if avail(ci, mi)}
        th = {m: thr(F[m]["con"]) for m in F}
        res["thresholds"][f"C{C}|{tpl}"] = {m: r3(v) for m, v in th.items()}
        for m, f in F.items():
            for g in ("all", "slow", "mid", "fast"):
                G = mask_group(g)
                rows = []
                for k in range(K):
                    ok = G & OKk[:, k]
                    con = f["con"][ok, k]
                    o_pm = con >= th[m]; o_sh = con >= th["single"]; o_sj = con >= th["joint"]
                    rows.append(dict(k=k, n=int(ok.sum()), dist=r3(dist[ok, k].mean()), con_med=r3(np.median(con)),
                                     obj_pm=r3(o_pm.mean()), obj_sh=r3(o_sh.mean()), obj_shj=r3(o_sj.mean()),
                                     hit=r3(f["hit"][ok, k].mean()), hit_last=r3(f["hitl"][ok, k].mean()),
                                     hit_given_obj_sh=r3(f["hit"][ok, k][o_sh].mean()) if o_sh.sum() >= 20 else None,
                                     l1=r3(f["l1"][ok, k].mean())))
                res["curves"][f"C{C}|{tpl}|{m}|{g}"] = rows

# ---------------- 2. 짝 비교 (clip 부트스트랩) ----------------
PAIRS = [("single", "joint"), ("single_s4", "single"), ("single_s8", "single"), ("single_s8", "single_s4")]
for ci, C in enumerate(Cs):
    for tpl in TPL:
        F = {m: fields(ci, mi, tpl) for mi, m in enumerate(MODES) if avail(ci, mi)}
        th_s = thr(F["single"]["con"])
        for a, b in PAIRS:
            if a not in F or b not in F:
                continue
            out = {}
            for (lo, hi), bn in zip(BANDS, BN):
                ok = OKk
                d = {}
                for key, A_, B_ in (("con", F[a]["con"], F[b]["con"]),
                                    ("obj_sh", (F[a]["con"] >= th_s).astype(float), (F[b]["con"] >= th_s).astype(float)),
                                    ("hit", F[a]["hit"], F[b]["hit"]), ("l1", F[a]["l1"], F[b]["l1"])):
                    d[key] = boot_mean(band_clip(A_ - B_, ok, lo, hi)[free])
                gt = band_clip((F[a]["con"] > F[b]["con"]).astype(float), ok, lo, hi)[free]
                d["frac_con_a_gt_b"] = r3(np.nanmean(gt))
                out[bn] = d
            res["paired"][f"C{C}|{tpl}|{a}-{b}"] = out

# ---------------- 2b. 띠 수준 (C16 · C32, 모드별 값 + 궤적 군집 CI) ----------------
for C in (16, 32):
    ci = Cs.index(C)
    for tpl in TPL:
        F = {m: fields(ci, mi, tpl) for mi, m in enumerate(MODES) if avail(ci, mi)}
        th_s = thr(F["single"]["con"])
        for m, f in F.items():
            out = {}
            for (lo, hi), bn in zip(BANDS, BN):
                out[bn] = dict(con=boot_mean(band_clip(f["con"], OKk, lo, hi)[free]),
                               obj_sh=boot_mean(band_clip((f["con"] >= th_s).astype(float), OKk, lo, hi)[free]),
                               obj_pm=boot_mean(band_clip((f["con"] >= thr(f["con"])).astype(float), OKk, lo, hi)[free]),
                               hit=boot_mean(band_clip(f["hit"], OKk, lo, hi)[free]),
                               l1=boot_mean(band_clip(f["l1"], OKk, lo, hi)[free]))
            res["levels"][f"C{C}|{tpl}|{m}"] = out

# ---------------- 3. 시나리오 안 (C16) ----------------
ci16 = Cs.index(16)
SC = list(FREE)
for tpl in TPL:
    F = {m: fields(ci16, mi, tpl) for mi, m in enumerate(MODES) if avail(ci16, mi)}
    th_s = thr(F["single"]["con"])
    for m in ("single", "joint"):
        f = F[m]; obj = (f["con"] >= th_s).astype(float); obj_pm = (f["con"] >= thr(f["con"])).astype(float)
        per = {}
        for s in SC:
            S = free & (scen == s)
            vs = v_end[S]; c1, c2 = np.percentile(vs, [33, 67])
            ts = np.digitize(v_end, [c1, c2])
            row = dict(n=int(S.sum()), dist_k10=r3(dist[S & OKk[:, 10], 10].mean()),
                       obj_sh_k10=r3(obj[S & OKk[:, 10], 10].mean()), obj_pm_k10=r3(obj_pm[S & OKk[:, 10], 10].mean()),
                       hit_k10=r3(f["hit"][S & OKk[:, 10], 10].mean()), bands={}, terciles={})
            for (lo, hi), bn in zip(BANDS, BN):
                row["bands"][bn] = dict(obj_sh=r3(np.nanmean(band_clip(obj, OKk, lo, hi)[S])),
                                        hit=r3(np.nanmean(band_clip(f["hit"], OKk, lo, hi)[S])),
                                        dist=r3(np.nanmean(band_clip(dist, OKk, lo, hi)[S])))
            for t, tn in enumerate(SPD):
                St = S & (ts == t)
                row["terciles"][tn] = {bn: dict(obj_sh=r3(np.nanmean(band_clip(obj, OKk, lo, hi)[St])),
                                                dist=r3(np.nanmean(band_clip(dist, OKk, lo, hi)[St])), n=int(St.sum()),
                                                n_traj=int(len(np.unique(traj[St]))))
                                       for (lo, hi), bn in zip(BANDS, BN)}
            per[s] = row
        res["scenario"][f"C16|{tpl}|{m}"] = per

        # LPM (k = 3..15, 유효 행만)
        ks = np.arange(3, K)
        ii, kk = np.nonzero(OKk[:, ks]); kk = ks[kk]
        y = obj[ii, kk]; dv = dist[ii, kk]
        sidx = np.array([SC.index(s) for s in scen[ii]])
        Sfe = np.eye(len(SC))[sidx]
        Kfe = (kk[:, None] == ks[None, 1:]).astype(float)
        DB = np.array([0, 2, 3, 4, 5, 6, 7, 99]); dbin = np.digitize(dv, DB[1:-1])
        Dfe = (dbin[:, None] == np.arange(1, len(DB) - 1)[None]).astype(float)
        XA = np.column_stack([Sfe, Kfe, dv]); XB = np.column_stack([Sfe, Dfe, kk]); XP = np.column_stack([np.ones_like(dv), Kfe, dv])
        tr = traj[ii]; clips = np.unique(tr); pos = {c: np.nonzero(tr == c)[0] for c in clips}

        def fit(Xm, rows_):
            return np.linalg.lstsq(Xm[rows_], y[rows_], rcond=None)[0][-1]
        allr = np.arange(len(y))
        est = dict(beta_dist_scenFE=fit(XA, allr), gamma_k_scenFE=fit(XB, allr), beta_dist_pooled=fit(XP, allr))
        bs = {k_: [] for k_ in est}
        for _ in range(300):
            cs = rng.choice(clips, len(clips)); rr = np.concatenate([pos[c] for c in cs])
            bs["beta_dist_scenFE"].append(fit(XA, rr)); bs["gamma_k_scenFE"].append(fit(XB, rr)); bs["beta_dist_pooled"].append(fit(XP, rr))
        res["lpm"][f"C16|{tpl}|{m}"] = {k_: dict(est=r3(v), lo=r3(np.percentile(bs[k_], 2.5)), hi=r3(np.percentile(bs[k_], 97.5)))
                                        for k_, v in est.items()} | dict(n_rows=int(len(y)), n_traj=int(len(clips)),
                                                                         corr_dist_k_within=r3(np.corrcoef(dv - Sfe @ np.linalg.lstsq(Sfe, dv, rcond=None)[0],
                                                                                                           kk - Sfe @ np.linalg.lstsq(Sfe, kk.astype(float), rcond=None)[0])[0, 1]))

# ---------------- 4. 진행률 (C16, tru) ----------------
for m in ("single", "joint"):
    f = fields(ci16, MODES.index(m), "tru"); th_m = thr(f["con"]); th_s = thr(fields(ci16, 0, "tru")["con"])
    for g in ("all", "slow", "mid", "fast"):
        G = mask_group(g); rows = []
        for k in range(K):
            base = G & OKk[:, k] & (dist[:, k] >= 3)
            def med(sel):
                return r3(np.median(f["prog"][sel, k])) if sel.sum() >= 20 else None
            rows.append(dict(k=k, n_uncond=int(base.sum()), prog_obj_pm=med(base & (f["con"][:, k] >= th_m)),
                             prog_obj_sh=med(base & (f["con"][:, k] >= th_s)), prog_uncond=med(base)))
        res["progress"][f"C16|{m}|{g}"] = rows

json.dump(res, open(OUT / "h4b_threshold_invariant.json", "w"), indent=1, ensure_ascii=False)

# ---------------- 요약 텍스트 ----------------
def cv(key, field, ks=range(K)):
    rows = res["curves"][key]
    return " ".join("  – " if rows[k][field] is None else f"{rows[k][field]:.2f}" for k in ks)

L = [f"n_free={res['meta']['n_free']}  n_traj={res['meta']['n_traj']}  speed n={res['meta']['n_speed']}  cuts={res['meta']['speed_cuts']}", "",
     "## 문턱 (C | tpl → 모드별)"]
for k_, v in res["thresholds"].items():
    L.append(f"  {k_}: " + "  ".join(f"{m}={t:.3f}" for m, t in v.items()))
for C in Cs:
    for tpl in TPL:
        L.append(f"\n## C{C} {tpl}  (k = 0..15, all free)")
        for m in MODES:
            key = f"C{C}|{tpl}|{m}|all"
            if key not in res["curves"]:
                continue
            for fld in ("con_med", "obj_pm", "obj_sh", "obj_shj", "hit", "hit_given_obj_sh", "hit_last", "l1"):
                L.append(f"  {m:9s} {fld:16s} " + cv(key, fld))
        L.append("  -- mid tercile")
        for m in MODES:
            key = f"C{C}|{tpl}|{m}|mid"
            if key in res["curves"]:
                L.append(f"  {m:9s} obj_sh(mid)      " + cv(key, "obj_sh"))
                L.append(f"  {m:9s} obj_pm(mid)      " + cv(key, "obj_pm"))
L.append("\n## 짝 비교 (a − b), clip 띠 평균, 95% CI")
for key, out in res["paired"].items():
    L.append(f"  {key}")
    for bn, d in out.items():
        s = []
        for fld in ("con", "obj_sh", "hit", "l1"):
            x = d[fld]
            s.append(f"{fld} {x['mean']:+.3f}[{x['lo']:+.3f},{x['hi']:+.3f}]" if x else f"{fld} –")
        L.append(f"    {bn:7s} " + "  ".join(s) + f"  P(con_a>con_b)={d['frac_con_a_gt_b']}")
L.append("\n## 띠 수준 (모드별 값, 궤적 군집 95% CI)")
for key, out in res["levels"].items():
    L.append(f"  {key}")
    for bn, d in out.items():
        L.append(f"    {bn:7s} " + "  ".join(f"{fld} {d[fld]['mean']:.3f}[{d[fld]['lo']:.3f},{d[fld]['hi']:.3f}]" for fld in ("con", "obj_sh", "obj_pm", "hit", "l1")))
L.append("\n## 시나리오 (C16, 공통 문턱)")
for key, per in res["scenario"].items():
    L.append(f"  {key}")
    for s, row in per.items():
        L.append(f"    {s:7s} n={row['n']} dist@k10={row['dist_k10']} obj_sh@k10={row['obj_sh_k10']} obj_pm@k10={row['obj_pm_k10']} hit@k10={row['hit_k10']}  "
                 + " ".join(f"{bn}:{v['obj_sh']}" for bn, v in row["bands"].items()))
        for tn, bd in row["terciles"].items():
            L.append(f"        {tn:4s} " + " ".join(f"{bn}:{v['obj_sh']}(d{v['dist']})" for bn, v in bd.items()))
L.append("\n## LPM (obj 공통 문턱)")
for key, v in res["lpm"].items():
    L.append(f"  {key}: " + json.dumps(v, ensure_ascii=False))
L.append("\n## 진행률 (C16 tru)")
for key, rows in res["progress"].items():
    for fld in ("prog_obj_pm", "prog_obj_sh", "prog_uncond"):
        L.append(f"  {key:18s} {fld:12s} " + " ".join("  – " if r[fld] is None else f"{r[fld]:.2f}" for r in rows))
(OUT / "h4b_summary.txt").write_text("\n".join(L))
print("\n".join(L))
