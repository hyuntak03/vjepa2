#!/usr/bin/env python3
"""RollOut v3 운동 주장 감사 — **궤적 단위** (2026-09-25).

new_archive 의 v3_signed_error · v3_trajectory 가 적은 운동 주장을 궤적 단위로 다시 잰다.
자 = p 자기 자 (`C{c}_P{p}_p`, p 문턱 · p 치우침), 자 검사 = z (`C{c}_P{p}_z`, z 문턱 · z 치우침; z 는 창 전체 = 미래 프레임을 본다).

규칙 (CROSS_REVIEW_2026-09-24 · CLAUDE.md)
  * 문턱 하나 (자마다 thr_val_fpr5).
  * 단위 = 궤적 (scenario, primary, secondary). 법칙마다 14 궤적. 외형 복사 (28 / 14) 는 먼저 궤적 안에서 평균한다.
  * 95 % CI = 궤적 bootstrap (법칙 안 14 개 재추출; 여러 법칙을 묶을 때는 법칙별 층화).
  * 귀무 = 같은 법칙 **다른 궤적**의 진실. 위치 귀무 (자리) 와 변위 귀무 (문맥 끝에서 옮긴 양) 를 둘 다 낸다.
  * 기준선 (GT 만으로): 복사 = 문맥 마지막 튜블릿 GT 자리 x0 에 그대로 / 등속 연장 = x0 + v0·(t+1), v0 = 문맥 끝 튜블릿 속도.

정의
  X(t)  = GT 프레임 (32+2t, 33+2t) 평균 (px, 화면 좌표 = (정규화+1)·144)
  x0    = GT 프레임 30·31 평균,  v0 = x0 − GT 프레임 28·29 평균  (px/튜블릿)
  읽은 값 x̂ = readings[...,0:2]·144 − 치우침,  '있다' = logit > 문턱
  부호 s_a(t) = plot_v3_l2_error.axis_dir 과 같다 (GT 운동 방향, 멈추면 마지막 방향 = wall 은 문맥 방향; 한 번도 안 움직인 y 는 위 = +)
  부호 오차 = (x̂ − X)·s   (+ 앞섬, − 뒤처짐).  복사 = (x0 − X)·s,  연장 = (x0 + v0(t+1) − X)·s
  읽을 수 있는 칸 = 물체가 화면 안 (두 프레임) 이고 화면 가장자리에서 36 px 이상 (GT 기준, 궤적마다 같다)

  $P z_research/scripts/analysis/audit_v3_motion.py            # C16/P16 · C16/P32 · C32/P32 · C8/P32
  → z_research/RollOutV3/audit/v3/{audit_v3_motion.json, tables.md}
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, check, fingerprint  # noqa: E402

OUT = ROOT / "z_research/RollOutV3/audit/v3"
RESN, SPLIT, CELL, EDGE, STILL, W = 144.0, 32, 18.0, 36.0, 1.0, 288.0
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
FREE = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc"]
WINDOWS = [(16, 16), (16, 32), (32, 32), (8, 32)]
NB, SEED = 4000, 0
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- 데이터
Z = np.load(WIN / "readings.npz", allow_pickle=True); check(Z, PRES)
S = json.loads((PRES / "summary.json").read_text())
THR = {r: float(S["reps"][r]["attn"]["thr_val_fpr5"]) for r in ("p", "z", "h")}
BIAS = {r: np.array(v, np.float32) for r, v in json.loads((PRES / "attn_bias_px.json").read_text()).items()}
IDX = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index.csv"))}
POS = Z["role"] == "roll"
VID = Z["video_id"][POS]; SC = Z["scenario"][POS]
L = (Z["truth"][POS] + 1.0) * RESN                                   # 화면 px (n, 64, 2)
INF = Z["in_frame"][POS].astype(bool)
TKEY = np.array([f"{IDX[v]['scenario']}|{IDX[v]['primary']}|{IDX[v]['secondary']}" for v in VID])
UT = sorted(set(TKEY), key=lambda s: (LAWS.index(s.split('|')[0]), float(s.split('|')[1]), float(s.split('|')[2])))
TID = np.array([UT.index(k) for k in TKEY])
TLAW = np.array([u.split("|")[0] for u in UT])
TSPEED = np.array([float(u.split("|")[1]) for u in UT])
REP = np.array([np.where(TID == i)[0][0] for i in range(len(UT))])  # 궤적마다 대표 clip (진실은 궤적 안에서 같다)
assert all(np.abs(L[TID == i] - L[REP[i]]).max() == 0 for i in range(len(UT))), "궤적 안 진실이 다르다"
LT = {law: np.where(TLAW == law)[0] for law in LAWS}               # 법칙 → 궤적 번호 (14)
assert all(len(v) == 14 for v in LT.values())


def axis_dir(Lc, X):
    """plot_v3_l2_error.axis_dir 과 같은 규칙 (여기서 다시 적는다; 아래 main 에서 원본과 대조)."""
    prev = Lc[:, SPLIT - 1] - Lc[:, SPLIT - 3]
    Xp = np.concatenate([Lc[:, SPLIT - 2:SPLIT].mean(1)[:, None], X], 1)
    V = Xp[:, 1:] - Xp[:, :-1]
    last = np.where(np.abs(prev) >= STILL / 2, np.sign(prev), 0.0)
    s = np.zeros_like(V)
    for t in range(V.shape[1]):
        cur = np.where(np.abs(V[:, t]) >= STILL, np.sign(V[:, t]), last); s[:, t] = cur; last = cur
    never = s == 0
    s[..., 0][never[..., 0]] = 1.0; s[..., 1][never[..., 1]] = -1.0
    return s, never


# ---------------------------------------------------------------- 궤적 평균 · bootstrap
def tmean(val, m=None):
    """clip 값 (n, ...) → 궤적 평균 (112, ...). m (n, ...) 이 있으면 m 인 clip 만, 없으면 nan."""
    val = np.asarray(val, np.float64)
    if m is None: m = np.ones(val.shape, bool)
    m = np.broadcast_to(m, val.shape)
    out = np.full((len(UT),) + val.shape[1:], np.nan)
    for i in range(len(UT)):
        k = TID == i; mm = m[k]; n = mm.sum(0)
        s = np.where(mm, val[k], 0).sum(0)
        out[i] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


_BIDX = {}


def bidx(n):
    if n not in _BIDX: _BIDX[n] = rng.integers(0, n, (NB, n))
    return _BIDX[n]


def ci(a):
    """a (14, ...) 궤적 값 → (mean, lo, hi, n_finite) — nan 은 빼고 재추출 안에서 nanmean."""
    a = np.asarray(a, np.float64)
    fin = np.isfinite(a)
    if a.ndim == 1:
        a = a[fin]
        if len(a) == 0: return [None, None, None, 0]
        bs = a[bidx(len(a))].mean(1)
        return [r1(a.mean()), r1(np.percentile(bs, 2.5)), r1(np.percentile(bs, 97.5)), int(len(a))]
    return [ci(a[:, j]) for j in range(a.shape[1])]


def ci_strat(arrs):
    """법칙별 (14,) 배열 목록 → 층화 bootstrap 으로 전체 평균 (법칙 평균의 평균)."""
    arrs = [np.asarray(a, np.float64)[np.isfinite(a)] for a in arrs]
    arrs = [a for a in arrs if len(a)]
    if not arrs: return [None, None, None, 0]
    mean = np.mean([a.mean() for a in arrs])
    bs = np.mean([a[bidx(len(a))].mean(1) for a in arrs], 0)
    return [r1(mean), r1(np.percentile(bs, 2.5)), r1(np.percentile(bs, 97.5)), int(sum(len(a) for a in arrs))]


def ci_diff(a, b):
    """독립 궤적 두 묶음 평균 차 (a − b) 의 bootstrap."""
    a = np.asarray(a, float); b = np.asarray(b, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return [None, None, None, 0]
    bs = a[bidx(len(a))].mean(1) - b[rng.integers(0, len(b), (NB, len(b)))].mean(1)
    return [r1(a.mean() - b.mean()), r1(np.percentile(bs, 2.5)), r1(np.percentile(bs, 97.5)), int(min(len(a), len(b)))]


def r1(x): return None if x is None or not np.isfinite(x) else round(float(x), 2)


# ---------------------------------------------------------------- 창 하나
def window(c, p):
    tp = p // 2; T = np.arange(tp)
    X = L[:, SPLIT:SPLIT + p].reshape(-1, tp, 2, 2).mean(2)
    x0 = L[:, SPLIT - 2:SPLIT].mean(1); v0 = x0 - L[:, SPLIT - 4:SPLIT - 2].mean(1)
    COPY = np.broadcast_to(x0[:, None], X.shape)
    EXT = x0[:, None] + v0[:, None] * (T + 1)[None, :, None]
    s, never = axis_dir(L, X)
    inf = INF[:, SPLIT:SPLIT + p].reshape(-1, tp, 2).all(2)
    edge = np.minimum(np.minimum(X[..., 0], W - X[..., 0]), np.minimum(X[..., 1], W - X[..., 1]))
    readable = inf & (edge >= EDGE)

    def rd(key, head):
        v = Z[f"C{c}_P{p}_{key}"][POS]
        return (v[..., 0:2] + 1.0) * RESN - BIAS[head], v[..., 2] > THR[head]
    pxy, psay = rd("p", "p"); zxy, zsay = rd("z", "z"); hxy, hsay = rd("p@h", "h")

    # 궤적 단위 기본량
    tX, tx0, tv0 = X[REP], x0[REP], v0[REP]
    ts = s[REP]; tread = readable[REP]; tedge = edge[REP]
    disp_cells = np.linalg.norm(tX - tx0[:, None], axis=-1) / CELL      # 문맥 끝에서 옮긴 거리 (칸)
    R = {"C": c, "P": p, "tp": tp}

    # ---------------- (a) 있다 비율
    pr, zr, hr = tmean(psay), tmean(zsay), tmean(hsay)
    pres = {}
    for law in LAWS:
        u = LT[law]
        pres[law] = {
            "p": ci(pr[u]), "z": ci(zr[u]), "p_at_h": ci(hr[u]),
            "p_readable": ci(np.where(tread[u], pr[u], np.nan)),
            "z_readable": ci(np.where(tread[u], zr[u], np.nan)),
            "n_traj_readable": tread[u].sum(0).tolist(),
            "disp_cells": [r1(v) for v in disp_cells[u].mean(0)],
            "edge_px_min": [r1(v) for v in tedge[u].min(0)],
        }
    uf = np.concatenate([LT[l] for l in FREE])
    pres["FREE6"] = {
        "p": [ci_strat([pr[LT[l], t] for l in FREE]) for t in T],
        "z": [ci_strat([zr[LT[l], t] for l in FREE]) for t in T],
        "p_readable": [ci_strat([np.where(tread[LT[l], t], pr[LT[l], t], np.nan) for l in FREE]) for t in T],
        "z_readable": [ci_strat([np.where(tread[LT[l], t], zr[LT[l], t], np.nan) for l in FREE]) for t in T],
        "n_traj_readable": tread[uf].sum(0).tolist(),
        "disp_cells": [r1(v) for v in disp_cells[uf].mean(0)],
    }
    # 절벽: 궤적마다 p 있다 비율이 처음 0.5 아래로 가는 칸 (자유 운동). 거리 vs 슬롯
    cl = []
    for i in uf:
        below = np.where(pr[i] < 0.5)[0]
        tc = int(below[0]) if len(below) else None
        cl.append(dict(traj=UT[i], law=TLAW[i], speed_px_per_tub=r1(np.linalg.norm(tv0[i])),
                       t_cliff=tc, disp_cells_at_cliff=r1(disp_cells[i, tc]) if tc is not None else None,
                       readable_at_cliff=bool(tread[i, tc]) if tc is not None else None,
                       z_rate_at_cliff=r1(zr[i, tc]) if tc is not None else None))
    tcs = np.array([d["t_cliff"] if d["t_cliff"] is not None else np.nan for d in cl], float)
    dcs = np.array([d["disp_cells_at_cliff"] if d["disp_cells_at_cliff"] is not None else np.nan for d in cl], float)
    sp = np.array([d["speed_px_per_tub"] for d in cl], float)
    ok = np.isfinite(tcs)
    corr = None
    if ok.sum() > 5 and np.std(tcs[ok]) > 0:
        a, b = tcs[ok], sp[ok]
        bi = rng.integers(0, len(a), (NB, len(a)))
        bs = [np.corrcoef(a[j], b[j])[0, 1] for j in bi if np.std(a[j]) > 0 and np.std(b[j]) > 0]
        corr = [r1(np.corrcoef(a, b)[0, 1]), r1(np.percentile(bs, 2.5)), r1(np.percentile(bs, 97.5)), int(len(a))]
    R["presence"] = pres
    R["cliff"] = {"per_traj": cl, "n_with_cliff": int(ok.sum()), "n_traj": len(cl),
                  "t_cliff_mean_sd": [r1(np.nanmean(tcs)), r1(np.nanstd(tcs))] if ok.any() else None,
                  "disp_cells_at_cliff_mean_sd": [r1(np.nanmean(dcs)), r1(np.nanstd(dcs))] if ok.any() else None,
                  "cv_slot_vs_cv_disp": [r1(np.nanstd(tcs) / np.nanmean(tcs)), r1(np.nanstd(dcs) / np.nanmean(dcs))] if ok.any() and np.nanmean(tcs) > 0 else None,
                  "corr_tcliff_speed": corr,
                  "frac_readable_at_cliff": r1(np.mean([d["readable_at_cliff"] for d in cl if d["t_cliff"] is not None])) if ok.any() else None}

    # ---------------- (b) x 추적: 적중 vs 귀무 vs 복사 vs 연장 (x 축 · 2D)
    trk = {}; RAW = {}
    for law in LAWS:
        u = LT[law]; RAW[law] = {k: [] for k in ("p_own", "z_own", "copy", "ext", "own_minus_null_disp", "own_minus_null_pos", "p_minus_copy", "p_minus_ext", "gain", "copy_own_minus_null_pos")}
        out = {k: [] for k in ("p_own", "z_own", "null_pos", "null_disp", "copy", "ext", "own_minus_null_pos",
                               "own_minus_null_disp", "p_minus_copy", "p_minus_ext", "p_minus_z", "copy_own_minus_null_pos",
                               "gain", "gain_gt_ext", "frac_traj_own_lt_nulldisp", "p_own_L2", "copy_L2", "ext_L2", "null_disp_L2")}
        for t in T:
            own, npos, ndisp, gains, own2, nd2 = [], [], [], [], [], []
            for i in u:
                k = np.where((TID == i) & psay[:, t])[0]
                if len(k) == 0:
                    own.append(np.nan); npos.append(np.nan); ndisp.append(np.nan); gains.append(np.nan); own2.append(np.nan); nd2.append(np.nan); continue
                xr = pxy[k, t]                                                    # (m, 2)
                oth = [o for o in u if o != i]
                own.append(np.abs(xr[:, 0] - tX[i, t, 0]).mean())
                npos.append(np.mean([np.abs(xr[:, 0] - tX[o, t, 0]).mean() for o in oth]))
                dr = xr - tx0[i]
                ndisp.append(np.mean([np.abs(dr[:, 0] - (tX[o, t, 0] - tx0[o, 0])).mean() for o in oth]))
                own2.append(np.linalg.norm(xr - tX[i, t], axis=-1).mean())
                nd2.append(np.mean([np.linalg.norm(dr - (tX[o, t] - tx0[o]), axis=-1).mean() for o in oth]))
                gd = tX[i, t, 0] - tx0[i, 0]
                gains.append(dr[:, 0].mean() / gd if abs(gd) >= CELL else np.nan)
            own, npos, ndisp, gains, own2, nd2 = map(np.array, (own, npos, ndisp, gains, own2, nd2))
            cp = np.abs(tx0[u, 0] - tX[u, t, 0]); ex = np.abs(tx0[u, 0] + tv0[u, 0] * (t + 1) - tX[u, t, 0])
            cpn = np.array([np.mean([abs(tx0[i, 0] - tX[o, t, 0]) for o in u if o != i]) for i in u])
            zo = tmean(np.abs(zxy[:, t, 0] - X[:, t, 0]), zsay[:, t])[u]
            gd = tX[u, t, 0] - tx0[u, 0]
            out["p_own"].append(ci(own)); out["z_own"].append(ci(zo)); out["null_pos"].append(ci(npos)); out["null_disp"].append(ci(ndisp))
            out["copy"].append(ci(cp)); out["ext"].append(ci(ex))
            out["own_minus_null_pos"].append(ci(own - npos)); out["own_minus_null_disp"].append(ci(own - ndisp))
            out["p_minus_copy"].append(ci(own - cp)); out["p_minus_ext"].append(ci(own - ex)); out["p_minus_z"].append(ci(own - zo))
            out["copy_own_minus_null_pos"].append(ci(cp - cpn))
            out["gain"].append(ci(gains))
            out["gain_gt_ext"].append(ci(np.where(np.abs(gd) >= CELL, (tv0[u, 0] * (t + 1)) / np.where(gd == 0, 1, gd), np.nan)))
            fin = np.isfinite(own)
            out["frac_traj_own_lt_nulldisp"].append(r1(np.mean(own[fin] < ndisp[fin])) if fin.any() else None)
            out["p_own_L2"].append(ci(own2))
            out["copy_L2"].append(ci(np.linalg.norm(tx0[u] - tX[u, t], axis=-1)))
            out["ext_L2"].append(ci(np.linalg.norm(tx0[u] + tv0[u] * (t + 1) - tX[u, t], axis=-1)))
            out["null_disp_L2"].append(ci(nd2))
            for kk, vv in (("p_own", own), ("z_own", zo), ("copy", cp), ("ext", ex), ("own_minus_null_disp", own - ndisp),
                           ("own_minus_null_pos", own - npos), ("p_minus_copy", own - cp), ("p_minus_ext", own - ex), ("gain", gains),
                           ("copy_own_minus_null_pos", cp - cpn)):
                RAW[law][kk].append(vv)
        trk[law] = out
    trk["FREE6"] = {kk: [ci_strat([RAW[l][kk][t] for l in FREE]) for t in T] for kk in RAW["flat_v"]}
    # 속도 특이성: 법칙 안 (궤적 14) 읽은 변위 x 를 GT 변위 x 에 회귀한 기울기 (1 = 궤적마다 옮긴 양을 그대로, 0 = 궤적과 무관)
    rdx = tmean(pxy[..., 0] - x0[:, None, 0], psay); zdx = tmean(zxy[..., 0] - x0[:, None, 0], zsay)
    hdx = tmean(hxy[..., 0] - x0[:, None, 0], hsay)                           # p 를 h 자로 읽은 것 (자 의존성 검사)
    gdx = tX[..., 0] - tx0[:, None, 0]

    def slope_pooled(Y, t, laws):
        num = den = 0.0; bsn = np.zeros(NB); bsd = np.zeros(NB)
        for l in laws:
            u = LT[l]; y, x = Y[u, t], gdx[u, t]; ok_ = np.isfinite(y)
            if ok_.sum() < 4: continue
            y, x = y[ok_], x[ok_]; yc, xc = y - y.mean(), x - x.mean()
            num += (yc * xc).sum(); den += (xc * xc).sum()
            bi = bidx(len(y)); yb, xb = y[bi], x[bi]
            ybc, xbc = yb - yb.mean(1, keepdims=True), xb - xb.mean(1, keepdims=True)
            bsn += (ybc * xbc).sum(1); bsd += (xbc * xbc).sum(1)
        if den == 0: return [None, None, None, 0]
        bs = bsn / np.where(bsd == 0, np.nan, bsd)
        return [r1(num / den), r1(np.nanpercentile(bs, 2.5)), r1(np.nanpercentile(bs, 97.5)), len(laws)]
    trk["disp_slope"] = {"p_FREE6": [slope_pooled(rdx, t, FREE) for t in T], "z_FREE6": [slope_pooled(zdx, t, FREE) for t in T], "p_at_h_FREE6": [slope_pooled(hdx, t, FREE) for t in T],
                         "p_by_law": {l: [slope_pooled(rdx, t, [l]) for t in T] for l in LAWS}}
    R["tracking"] = trk

    # ---------------- (c) 부호 오차 · 기준선 · 가장 가까운 기준
    ep = (pxy - X) * s; ez = (zxy - X) * s; eh = (hxy - X) * s
    tep = tmean(ep, psay[..., None]); tez = tmean(ez, zsay[..., None]); teh = tmean(eh, hsay[..., None])
    tpz = tmean(ep - ez, (psay & zsay)[..., None])
    tcopy = (tx0[:, None] - tX) * ts; text = (tx0[:, None] + tv0[:, None] * (T + 1)[None, :, None] - tX) * ts
    sg = {}
    for law in LAWS:
        u = LT[law]; sg[law] = {}
        for a in (0, 1):
            d = {k: [] for k in ("p", "z", "p_minus_z", "p_at_h", "copy", "ext", "sep_ext_copy", "sep_ext_gt",
                                 "dist_gt", "dist_copy", "dist_ext", "dist_ext_minus_copy", "lambda_p", "lambda_pz", "lambda_z", "lambda_gt", "never_moved")}
            for t in T:
                P_, Zz, PZ, H_ = tep[u, t, a], tez[u, t, a], tpz[u, t, a], teh[u, t, a]
                Cp, Ex = tcopy[u, t, a], text[u, t, a]
                d["p"].append(ci(P_)); d["z"].append(ci(Zz)); d["p_minus_z"].append(ci(PZ)); d["p_at_h"].append(ci(H_))
                d["copy"].append(ci(Cp)); d["ext"].append(ci(Ex))
                d["sep_ext_copy"].append(r1(np.abs(Ex - Cp).mean())); d["sep_ext_gt"].append(r1(np.abs(Ex).mean()))
                d["dist_gt"].append(ci(np.abs(P_))); d["dist_copy"].append(ci(np.abs(P_ - Cp))); d["dist_ext"].append(ci(np.abs(P_ - Ex)))
                d["dist_ext_minus_copy"].append(ci(np.abs(P_ - Ex) - np.abs(P_ - Cp)))
                den = Ex - Cp; okd = np.abs(den) >= CELL
                d["lambda_p"].append(ci(np.where(okd, (P_ - Cp) / np.where(okd, den, 1), np.nan)))
                d["lambda_pz"].append(ci(np.where(okd, (PZ - Cp) / np.where(okd, den, 1), np.nan)))
                d["lambda_z"].append(ci(np.where(okd, (Zz - Cp) / np.where(okd, den, 1), np.nan)))
                d["lambda_gt"].append(ci(np.where(okd, (0 - Cp) / np.where(okd, den, 1), np.nan)))
                d["never_moved"].append(r1(never[REP][u, t, a].mean()))
            sg[law]["xy"[a]] = d
    R["signed"] = sg

    # ---------------- 자 끌림 통제: 화면 x 오차를 진실 화면 x 에 회귀 (법칙 안, 튜블릿 고정효과)
    escr_p = tmean(pxy[..., 0] - X[..., 0], psay); escr_z = tmean(zxy[..., 0] - X[..., 0], zsay)
    Tuse = [t for t in T if all(np.nanmean(pr[LT[l], t]) > 0.5 for l in FREE)]
    Xref = tX[uf][:, :, 0].mean(0)                                           # 공통 기준 화면 x (튜블릿마다, 자유 6 법칙 평균)

    def fit_beta(E, Xs, V=None):
        """E, Xs (m, tt) → 튜블릿 고정효과 + β·X (+ γ·V) 최소제곱. β (, γ)."""
        m, tt = E.shape; rows_, y = [], []
        for j in range(tt):
            for i in range(m):
                if not np.isfinite(E[i, j]): continue
                r = np.zeros(tt); r[j] = 1
                rows_.append(np.r_[r, Xs[i, j], [] if V is None else V[i, j]]); y.append(E[i, j])
        A = np.array(rows_); y = np.array(y)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return coef[tt:]

    pull = {"Tuse": [int(t) for t in Tuse], "Xref_px": [r1(v) for v in Xref], "laws": {}}
    if len(Tuse) >= 2:
        Tu = np.array(Tuse)
        adj_store = {}
        for law in FREE + ["wall"]:
            u = LT[law]
            Ep, Ez, Xs = escr_p[u][:, Tu], escr_z[u][:, Tu], tX[u][:, Tu, 0]
            sx = ts[u][:, Tu, 0]
            Vexp = (tv0[u, 0][:, None] * (Tu + 1)[None]) if law != "wall" else None
            bp = fit_beta(Ep, Xs)[0]; bz = fit_beta(Ez, Xs)[0]
            bpv = fit_beta(Ep, Xs, Vexp) if Vexp is not None else [np.nan, np.nan]
            # bootstrap: 궤적 재추출 후 재적합
            bi = bidx(14); adj_bs = []
            for j in bi[:1000]:
                try: b = fit_beta(Ep[j], Xs[j])[0]
                except Exception: continue
                adj_bs.append(np.nanmean((Ep[j] - b * (Xs[j] - Xref[Tu][None])) * sx[j], 0))
            adj_bs = np.array(adj_bs)
            adj = ((Ep - bp * (Xs - Xref[Tu][None])) * sx)                  # (14, tt) 부호 오차, 공통 화면 x 로 옮김
            raw = Ep * sx
            adj_store[law] = (adj, raw)
            pull["laws"][law] = {"beta_p": r1(bp), "beta_z": r1(bz), "beta_p_with_speed": [r1(v) for v in bpv],
                                 "raw_signed_x": ci(raw), "adj_signed_x": [[r1(np.nanmean(adj[:, j])), r1(np.nanpercentile(adj_bs[:, j], 2.5)),
                                                                           r1(np.nanpercentile(adj_bs[:, j], 97.5)), 14] for j in range(len(Tu))]}
        # 공통 β (법칙 × 튜블릿 고정효과) 도 낸다
        Eall, Xall, Vall = [], [], []
        for law in FREE:
            u = LT[law]; Eall.append(escr_p[u][:, Tu]); Xall.append(tX[u][:, Tu, 0]); Vall.append(tv0[u, 0][:, None] * (Tu + 1)[None])
        tt = len(Tu); rows_, y = [], []
        for li, (E, Xs, V) in enumerate(zip(Eall, Xall, Vall)):
            for i in range(E.shape[0]):
                for j in range(tt):
                    if not np.isfinite(E[i, j]): continue
                    r = np.zeros(len(FREE) * tt); r[li * tt + j] = 1
                    rows_.append(np.r_[r, Xs[i, j]]); y.append(E[i, j])
        coef = np.linalg.lstsq(np.array(rows_), np.array(y), rcond=None)[0]
        pull["beta_pooled_lawxt_FE"] = r1(coef[-1])
        pull["adj_pooled"] = {law: ci(((Eall[li] - coef[-1] * (Xall[li] - Xref[Tu][None])) * ts[LT[law]][:, Tu, 0])) for li, law in enumerate(FREE)}
        R["_adj_store"] = {k: (v[0].tolist(), v[1].tolist()) for k, v in adj_store.items()}
    R["pull"] = pull

    # ---------------- 주장 (c)(d)(e): 구간 평균 (궤적마다 먼저) · 기준선 · 법칙 대비 (flat_v 기준)
    bpool = pull.get("beta_pooled_lawxt_FE") or 0.0
    adjx = tep[..., 0] - bpool * (tX[..., 0] - Xref[None]) * ts[..., 0]        # 공통 β 로 화면 x 끌림을 뺀 부호 x 오차
    RANGES = {"early": [1, 2, 3, 4], "late": [5, 6, 7], "all17": [1, 2, 3, 4, 5, 6, 7]}
    if tp >= 10: RANGES["t8_9"] = [8, 9]

    # 지연 추적 기준선: GT 를 2 프레임 (= 1 튜블릿) 늦게 읽는 추적자. 공통 지연은 법칙 대비에서 거의 0 이 된다
    fr = np.arange(SPLIT, SPLIT + p) - 2.0
    Xdel = L[:, fr.astype(int)].reshape(-1, tp, 2, 2).mean(2)
    tdel = tmean((Xdel - X) * s)

    def seg(A, law, a, rng_):
        u = LT[law]; v = A[u][:, rng_, a] if A.ndim == 3 else A[u][:, rng_]
        return np.nanmean(v, 1)
    items = [("flat_v", 0), ("flat_a", 0), ("flat_d", 0), ("ramp_a", 0), ("ramp_a", 1), ("ramp_d", 0),
             ("arc", 0), ("arc", 1), ("ledge", 0), ("ledge", 1), ("wall", 0)]
    cl_ = {}
    for law, a in items:
        for rn, rr in RANGES.items():
            P_, Zz, PZ = seg(tep, law, a, rr), seg(tez, law, a, rr), seg(tpz, law, a, rr)
            Cp, Ex = seg(tcopy, law, a, rr), seg(text, law, a, rr)
            d = {"p": ci(P_), "z": ci(Zz), "p_minus_z": ci(PZ), "copy": ci(Cp), "ext": ci(Ex),
                 "p_adj_pull": ci(seg(adjx, law, 0, rr)) if a == 0 else None,
                 "dist_gt": ci(np.abs(P_)), "dist_copy": ci(np.abs(P_ - Cp)), "dist_ext": ci(np.abs(P_ - Ex)),
                 "pz_dist_gt": ci(np.abs(PZ)), "pz_dist_copy": ci(np.abs(PZ - Cp)), "pz_dist_ext": ci(np.abs(PZ - Ex)),
                 "p_minus_copy": ci(P_ - Cp), "p_minus_ext": ci(P_ - Ex), "pz_minus_copy": ci(PZ - Cp), "pz_minus_ext": ci(PZ - Ex),
                 "sep_ext_copy": r1(np.abs(Ex - Cp).mean()), "sep_ext_gt": r1(np.abs(Ex).mean()),
                 "pres_p": r1(np.nanmean(seg(pr[..., None], law, 0, rr)))}
            if law != "flat_v" and a == 0 and law != "wall":
                Pv, PZv, Ev = seg(tep, "flat_v", 0, rr), seg(tpz, "flat_v", 0, rr), seg(text, "flat_v", 0, rr)
                d["contrast_vs_flat_v"] = {"p": ci_diff(P_, Pv), "p_minus_z": ci_diff(PZ, PZv),
                                           "p_adj_pull": ci_diff(seg(adjx, law, 0, rr), seg(adjx, "flat_v", 0, rr)),
                                           "p_at_h": ci_diff(seg(teh, law, 0, rr), seg(teh, "flat_v", 0, rr)),
                                           "delayed_gt_pred": ci_diff(seg(tdel, law, 0, rr), seg(tdel, "flat_v", 0, rr)),
                                           "ext_pred": ci_diff(Ex, Ev), "gt_pred": [0.0, 0.0, 0.0, 14]}
            cl_[f"{law}|{'xy'[a]}|{rn}"] = d
    R["claims_c"] = cl_

    # ---------------- 속도 톱니: 홀짝 교대 (p, z, p@h, GT)
    def zigzag(xy, say):
        """운동 축 x 의 읽은 속도 V(t) (부호 = GT 방향) → 궤적마다 A = mean_t (−1)^t [V(t) − (V(t−1)+V(t+1))/2]."""
        Vv = (xy[:, 1:, 0] - xy[:, :-1, 0]) * s[:, 1:, 0]
        mm = say[:, 1:] & say[:, :-1]
        tV = tmean(Vv, mm)                                                   # (112, tp−1), V index j ↔ 튜블릿 j+1
        jmax = min(tp - 1, 8)                                                # t ≤ 8 까지만 (뒤는 물체를 잃는다)
        A, R2 = [], []
        for i in range(len(UT)):
            v = tV[i, :jmax]
            d2 = np.array([v[j] - 0.5 * (v[j - 1] + v[j + 1]) for j in range(1, len(v) - 1)])
            sgn = np.array([(-1) ** (j + 1) for j in range(1, len(v) - 1)])   # 튜블릿 j+1 의 홀짝
            ok_ = np.isfinite(d2)
            A.append(np.mean(sgn[ok_] * d2[ok_]) if ok_.sum() >= 3 else np.nan)
            R2.append(np.mean(np.abs(d2[ok_])) if ok_.sum() >= 3 else np.nan)
        return np.array(A), np.array(R2), tV
    zz = {}
    Xtrue = np.concatenate([x0[:, None], X], 1)                               # GT 도 같은 식 (읽은 값 대신 진실)
    for name, xy, say in (("p", pxy, psay), ("z", zxy, zsay), ("p_at_h", hxy, hsay), ("GT", X, np.ones_like(psay))):
        A, Rg, tV = zigzag(xy, say)
        zz[name] = {"A_free6": ci_strat([A[LT[l]] for l in FREE]), "rough_free6": ci_strat([Rg[LT[l]] for l in FREE]),
                    "A_by_law": {l: ci(A[LT[l]]) for l in LAWS}, "_A": A.tolist()}
    Ap, Az = np.array(zz["p"]["_A"]), np.array(zz["z"]["_A"])
    okk = np.isfinite(Ap) & np.isfinite(Az) & np.isin(np.arange(len(UT)), uf)
    zz["corr_A_p_z_free6"] = r1(np.corrcoef(Ap[okk], Az[okk])[0, 1]) if okk.sum() > 5 else None
    zz["sign_agree_p_z_free6"] = r1(np.mean(np.sign(Ap[okk]) == np.sign(Az[okk]))) if okk.sum() else None
    zz["frac_traj_A_pos"] = {n: r1(np.nanmean(np.array(zz[n]["_A"])[uf] > 0)) for n in ("p", "z", "p_at_h")}
    for n in ("p", "z", "p_at_h", "GT"): zz[n].pop("_A")
    R["zigzag"] = zz
    return R


# ---------------------------------------------------------------- 요약 표 (md)
def fmt(v):
    if v is None or v[0] is None: return "—"
    return f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"


def fmtr(v):
    if v is None or v[0] is None: return "—"
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"


def tables(allR):
    out = ["# audit_v3_motion — 자동 생성 표 (궤적 단위, 95 % CI = 궤적 bootstrap)", "",
           f"자 지문 `{fingerprint()}` · 문턱 p {THR['p']:+.3f} / z {THR['z']:+.3f} / h {THR['h']:+.3f} · 치우침 p {BIAS['p'].tolist()} z {BIAS['z'].tolist()} · NB={NB} seed={SEED}", ""]
    for key, R in allR.items():
        tp = R["tp"]
        out += [f"## {key}", "", "### (a) 있다 비율 — 자유 운동 6 법칙 (층화) · 읽을 수 있는 칸만", "",
                "| t | 옮긴 거리 (칸) | p 전체 | p 읽을 수 있는 칸 | z 읽을 수 있는 칸 | 읽을 수 있는 궤적 |", "|---|---|---|---|---|---|"]
        P = R["presence"]["FREE6"]
        for t in range(tp):
            out.append(f"| {t} | {P['disp_cells'][t]:.1f} | {fmtr(P['p'][t])} | {fmtr(P['p_readable'][t])} | {fmtr(P['z_readable'][t])} | {P['n_traj_readable'][t]}/84 |")
        cl = R["cliff"]
        out += ["", f"절벽 (궤적마다 p 있다 < 0.5 첫 칸): {cl['n_with_cliff']}/{cl['n_traj']} 궤적, t = {cl['t_cliff_mean_sd']} (평균, sd), "
                f"그때 옮긴 거리 {cl['disp_cells_at_cliff_mean_sd']} 칸, CV 슬롯/거리 {cl['cv_slot_vs_cv_disp']}, corr(t_cliff, 속도) {cl['corr_tcliff_speed']}, 그 칸 읽을 수 있음 {cl['frac_readable_at_cliff']}", ""]
        out += ["### (b) x 추적 (px, |x̂−X|) — 법칙별, 적중 − 귀무 / p − 복사 / p − 연장", "",
                "| 법칙 | t | p 적중 | z 적중 | 복사 | 연장 | 적중−위치귀무 | 적중−변위귀무 | p−복사 | p−연장 | gain (p) | gain (연장) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for law in LAWS + ["FREE6"]:
            Tk = R["tracking"][law]
            if law == "FREE6":
                for t in range(tp):
                    out.append(f"| FREE6 | {t} | {fmt(Tk['p_own'][t])} | {fmt(Tk['z_own'][t])} | {fmt(Tk['copy'][t])} | {fmt(Tk['ext'][t])} | {fmt(Tk['own_minus_null_pos'][t])} | "
                               f"{fmt(Tk['own_minus_null_disp'][t])} | {fmt(Tk['p_minus_copy'][t])} | {fmt(Tk['p_minus_ext'][t])} | {fmtr(Tk['gain'][t])} | — |")
                continue
            for t in range(tp):
                out.append(f"| {law} | {t} | {fmt(Tk['p_own'][t])} | {fmt(Tk['z_own'][t])} | {fmt(Tk['copy'][t])} | {fmt(Tk['ext'][t])} | {fmt(Tk['own_minus_null_pos'][t])} | "
                           f"{fmt(Tk['own_minus_null_disp'][t])} | {fmt(Tk['p_minus_copy'][t])} | {fmt(Tk['p_minus_ext'][t])} | {fmtr(Tk['gain'][t])} | {fmtr(Tk['gain_gt_ext'][t])} |")
        out += ["", "### (c) 부호 오차 (px, + 앞섬) — p · z · p−z · 복사 · 연장 · λ (0 = 복사, 1 = 연장)", "",
                "| 법칙 | 축 | t | p | z | p−z | 복사 | 연장 | |연장−복사| | λ_p | λ_(p−z) | λ_z | λ_GT |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for law in LAWS:
            for a in "xy":
                d = R["signed"][law][a]
                for t in range(tp):
                    out.append(f"| {law} | {a} | {t} | {fmt(d['p'][t])} | {fmt(d['z'][t])} | {fmt(d['p_minus_z'][t])} | {fmt(d['copy'][t])} | {fmt(d['ext'][t])} | {d['sep_ext_copy'][t]} | "
                               f"{fmtr(d['lambda_p'][t])} | {fmtr(d['lambda_pz'][t])} | {fmtr(d['lambda_z'][t])} | {fmtr(d['lambda_gt'][t])} |")
        pl = R["pull"]
        out += ["", f"### 자 끌림 통제 (튜블릿 {pl['Tuse']}, 공통 기준 화면 x = 자유 6 법칙 평균) — β_pooled {pl.get('beta_pooled_lawxt_FE')}", "",
                "| 법칙 | β_p | β_z | β_p (+속도) | 부호 x 원래 | 부호 x 보정 (법칙 안 β) | 부호 x 보정 (공통 β) |", "|---|---|---|---|---|---|---|"]
        for law, d in pl["laws"].items():
            raw = "; ".join(fmt(v) for v in d["raw_signed_x"]); adj = "; ".join(fmt(v) for v in d["adj_signed_x"])
            ap = "; ".join(fmt(v) for v in pl.get("adj_pooled", {}).get(law, [])) if law in pl.get("adj_pooled", {}) else "—"
            out.append(f"| {law} | {d['beta_p']} | {d['beta_z']} | {d['beta_p_with_speed']} | {raw} | {adj} | {ap} |")
        out += ["", "### 주장 (c)(d)(e) — 구간 평균 (궤적마다 먼저 평균, px, + 앞섬)", "",
                "| 법칙·축·구간 | 있다 | p | z | p−z | p (끌림 보정) | 복사 | 연장 | |p−GT| | |p−복사| | |p−연장| | 대비 p − flat_v | 대비 (p−z) | 대비 (끌림 보정) | 연장 예측 대비 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, d in R["claims_c"].items():
            cv = d.get("contrast_vs_flat_v")
            out.append(f"| {k} | {d['pres_p']} | {fmt(d['p'])} | {fmt(d['z'])} | {fmt(d['p_minus_z'])} | {fmt(d['p_adj_pull'])} | {fmt(d['copy'])} | {fmt(d['ext'])} | "
                       f"{fmt(d['dist_gt'])} | {fmt(d['dist_copy'])} | {fmt(d['dist_ext'])} | "
                       + (f"{fmt(cv['p'])} | {fmt(cv['p_minus_z'])} | {fmt(cv['p_adj_pull'])} | {fmt(cv['ext_pred'])} |" if cv else "— | — | — | — |"))
        zz = R["zigzag"]
        out += ["", "### 속도 톱니 (A = 홀짝 교대 진폭, px/튜블릿; 거칠기 = |2계 차분| 평균)", "",
                "| 읽기 | A (자유 6) | 거칠기 | A>0 궤적 비율 |", "|---|---|---|---|"]
        for n in ("p", "z", "p_at_h", "GT"):
            out.append(f"| {n} | {fmt(zz[n]['A_free6'])} | {fmt(zz[n]['rough_free6'])} | {zz['frac_traj_A_pos'].get(n, '—')} |")
        out += [f"", f"corr(A_p, A_z) 궤적 단위 = {zz['corr_A_p_z_free6']} · 부호 일치 {zz['sign_agree_p_z_free6']}", ""]
    return "\n".join(out)


def main():
    # 부호 규칙이 원본과 같은지 대조
    try:
        sys.path.insert(0, str(ROOT / "z_research/scripts/figures"))
        from plot_v3_l2_error import axis_dir as ad0
        X = L[:, SPLIT:SPLIT + 32].reshape(-1, 16, 2, 2).mean(2)
        assert np.array_equal(ad0(L, X)[0], axis_dir(L, X)[0]); sign_check = "identical to plot_v3_l2_error.axis_dir"
    except ImportError:
        sign_check = "plot_v3_l2_error not importable"
    OUT.mkdir(parents=True, exist_ok=True)
    allR = {}
    for c, p in WINDOWS:
        print(f"C{c}/P{p} ...", flush=True)
        allR[f"C{c}_P{p}"] = window(c, p)
    meta = dict(decoder_fp=fingerprint(), readings=str(WIN / "readings.npz"), thr=THR, bias={k: v.tolist() for k, v in BIAS.items()},
                n_clip_possible=int(POS.sum()), n_traj=len(UT), nb=NB, seed=SEED, edge_px=EDGE, cell_px=CELL, sign_rule=sign_check,
                windows=[f"C{c}_P{p}" for c, p in WINDOWS])
    js = {"meta": meta, "windows": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in allR.items()},
          "adj_store": {k: v.get("_adj_store") for k, v in allR.items()}}
    (OUT / "audit_v3_motion.json").write_text(json.dumps(js, ensure_ascii=False, indent=1))
    (OUT / "tables.md").write_text(tables(allR))
    print("→", OUT)


if __name__ == "__main__":
    main()
