#!/usr/bin/env python3
"""I-scene 적대 검증 — e_scene_reach_analyze.py / scene_mediation.py 와 독립 구현 (2026-09-25).
python verify_iscene.py <scene_dir> [--boot B] [--sub N]   → JSON stdout 마지막 줄 'RESULT ' + json
"""
import json, sys, argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--sub", type=int, default=400000)
a = ap.parse_args(); I = Path(a.d); meta = json.load(open(I / "meta.json")); arms = meta["arms"]; n = meta["n"]; G = 16
rng = np.random.default_rng(1234)
C = np.load(I / "arm_c.npy").astype(np.int32); CT = np.load(I / "arm_ct.npy").astype(np.float32); CC = np.load(I / "arm_cc.npy").astype(np.float32)
E = np.load(I / "enc.npy"); AT = np.load(I / "att.npy").astype(np.float32)
pan = (I / "src.npy").exists() is False
if pan:
    items = meta["items"]; V = np.array([it["v"] for it in items]); DR = np.array([it["dir"] for it in items])
    sx = np.arange(256) % G; sy = np.arange(256) // G
    SRC = np.full((n, 8, 256), -1, np.int32); D = np.zeros((n, 8, 256), np.float32)
    for i in range(8):
        xp = 16 * sx[None, :] + 8 + (DR * V)[:, None] * 2 * (i + 1)          # (n,256)
        ok = (xp >= 0) & (xp < 256)
        SRC[:, i] = np.where(ok, sy[None, :] * G + (np.clip(xp, 0, 255) // 16), -1)
        D[:, i] = (V * 2 * (i + 1) / 16.0)[:, None]
else:
    SRC = np.load(I / "src.npy").astype(np.int32); D = np.load(I / "disp.npy").astype(np.float32); D = np.where(np.isfinite(D), D, -1)
    V = None
VAL = SRC >= 0
def cheb(c):
    return np.maximum(np.abs(c % G - SRC % G), np.abs(c // G - SRC // G))
HIT = {ar: (cheb(C[:, k]) <= 1) & VAL for k, ar in enumerate(arms)}
EX = {ar: (C[:, k] == SRC) & VAL for k, ar in enumerate(arms)}
ENC_HIT = (cheb(E[:, 0].astype(np.int32)) <= 1) & VAL; ENC_EX = (E[:, 0].astype(np.int32) == SRC) & VAL
CLIP = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape); SLOT = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape)
BINS = [0, 1, 2, 3, 4, 6, 9, 13]
res = {"dir": str(I), "n": n, "pan": pan}

def ratio_boot(num, den, B):
    """num, den (n,) per clip → ratio, CI (clip bootstrap)."""
    ok = den > 0; num, den = num[ok], den[ok]
    if len(num) == 0: return [None, None, None]
    r = num.sum() / den.sum(); bs = []
    for _ in range(B):
        s = rng.integers(0, len(num), len(num)); bs.append(num[s].sum() / den[s].sum())
    return [round(float(r), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

def curve(h, m, B=0):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mm = m & (D >= lo) & (D < hi)
        out.append(ratio_boot((h & mm).sum((1, 2)).astype(float), mm.sum((1, 2)).astype(float), B) if B else
                   [round(float((h & mm).sum() / max(mm.sum(), 1)), 4), int(mm.sum())])
    return out

def regress(y, m, B):
    """y ~ 1 + slot + d, tokens m, clip bootstrap. 반환 계수 [slot, d] with CI."""
    X = np.stack([np.ones(m.sum()), SLOT[m], D[m]], 1); yy = y[m].astype(float); cl = CLIP[m]
    b0 = np.linalg.lstsq(X, yy, rcond=None)[0]
    uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}; bs = []
    for _ in range(B):
        s = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s], yy[s], rcond=None)[0])
    bs = np.array(bs)
    return {k: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, k in ((1, "per_slot"), (2, "per_cell"))}

B = a.boot
# ---------------- C1
c1 = {}
for pk in ("R", "A", "P"):
    ar = f"{pk}:base"
    c1[ar] = {"agree_cheb": regress(HIT[ar], ENC_HIT & (D >= 0), B // 2), "nofilter_cheb": regress(HIT[ar], VAL & (D >= 0), B // 2),
              "agree_exact": regress(EX[ar], ENC_EX & (D >= 0), B // 2),
              "curve_agree": curve(HIT[ar], ENC_HIT), "curve_nofilter": curve(HIT[ar], VAL), "curve_exact_agree": curve(EX[ar], ENC_EX)}
    if pan:
        c1[ar]["curve_by_speed_agree"] = {int(v): curve(HIT[ar], ENC_HIT & (V == v)[:, None, None]) for v in np.unique(V)}
        c1[ar]["curve_by_speed_nofilter"] = {int(v): curve(HIT[ar], VAL & (V == v)[:, None, None]) for v in np.unique(V)}
        # 슬롯 고정 · 거리만 변화 (속도로) vs 거리 고정 · 슬롯 변화: 슬롯 3 에서 속도별 적중 / 거리 2–3 칸에서 슬롯별 적중
        c1[ar]["slot3_by_speed"] = {int(v): round(float(HIT[ar][(V == v)[:, None, None] & ENC_HIT & (SLOT == 3)].mean()), 3) for v in np.unique(V)}
        c1[ar]["d2to3_by_slot"] = {int(sl): round(float(HIT[ar][(SLOT == sl) & ENC_HIT & (D >= 2) & (D < 3)].mean()), 3) for sl in range(8) if ((SLOT == sl) & ENC_HIT & (D >= 2) & (D < 3)).sum() > 500}
res["C1"] = c1
# ---------------- C2 / C3: 짝 차이
def paired(ar, base, m, B):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mm = m & (D >= lo) & (D < hi)
        num = ((HIT[ar] & mm).sum((1, 2)) - (HIT[base] & mm).sum((1, 2))).astype(float); den = mm.sum((1, 2)).astype(float)
        out.append(ratio_boot(num, den, B))
    return out
AG = ENC_HIT
c2 = {}
if "R:rule_full" in arms:
    c2["rule_full_eq_base_frac"] = float((C[:, arms.index("R:rule_full")] == C[:, arms.index("R:base")]).mean())
for ar in ("R:rule_prefix", "R:rule_iso", "A:rule_full"):
    if ar in arms:
        base = ar.split(":")[0] + ":base"; c2[ar] = {"delta_bins": paired(ar, base, AG, B)}
        if pan: c2[ar]["delta_by_speed_4to6"] = {int(v): paired(ar, base, AG & (V == v)[:, None, None], B // 2)[4] for v in np.unique(V)}
        c2[ar]["delta_by_slot_all_d"] = {int(sl): ratio_boot(((HIT[ar] & AG & (SLOT == sl)).sum((1, 2)) - (HIT[base] & AG & (SLOT == sl)).sum((1, 2))).astype(float), (AG & (SLOT == sl)).sum((1, 2)).astype(float), B // 2) for sl in range(8)}
res["C2"] = c2
c3 = {}
for ar in ("R:bias1", "R:bias2", "R:bias4", "R:bias4_rev", "R:anticopy2", "R:anticopy4", "R:temp1.5", "R:temp2", "R:supp_src4", "A:bias2", "A:bias4", "A:temp2", "A:anticopy4", "P:bias4", "R:bias4_cv", "A:bias4_cv"):
    if ar not in arms: continue
    base = ar.split(":")[0] + ":base"; r = {"delta_bins": paired(ar, base, AG, B)}
    m2 = AG & (D < 2); r["near_d_lt2_delta"] = ratio_boot(((HIT[ar] & m2).sum((1, 2)) - (HIT[base] & m2).sum((1, 2))).astype(float), m2.sum((1, 2)).astype(float), B)
    r["delta_exact_bins"] = [ratio_boot(((EX[ar] & mm).sum((1, 2)) - (EX[base] & mm).sum((1, 2))).astype(float), mm.sum((1, 2)).astype(float), B // 2)
                             for mm in [ENC_EX & (D >= lo) & (D < hi) for lo, hi in zip(BINS[:-1], BINS[1:])]]
    if pan: r["delta_by_speed_4to6"] = {int(v): paired(ar, base, AG & (V == v)[:, None, None], B // 2)[4] for v in np.unique(V)}
    ka, kb = arms.index(ar), arms.index(base)
    soft_a, soft_b = CT[:, ka] - CC[:, ka], CT[:, kb] - CC[:, kb]
    r["soft_delta_bins"] = [round(float(np.nanmean((soft_a - soft_b)[AG & (D >= max(lo, 1)) & (D < hi)])), 4) for lo, hi in zip(BINS[:-1], BINS[1:])]
    r["soft_pos_frac_base_vs_arm_4to6"] = [round(float((soft_b > 0)[AG & (D >= 4) & (D < 6)].mean()), 3), round(float((soft_a > 0)[AG & (D >= 4) & (D < 6)].mean()), 3)]
    c3[ar] = r
# soft vs argmax 일치 (base R): P(hit | soft>0) vs P(hit | soft<=0)
kb = arms.index("R:base"); sb = CT[:, kb] - CC[:, kb]; mm = AG & (D >= 2)
c3["R:base_hit_given_soft_pos_vs_neg_d>=2"] = [round(float(HIT["R:base"][mm & (sb > 0)].mean()), 3), round(float(HIT["R:base"][mm & (sb <= 0)].mean()), 3)]
res["C3"] = c3
# ---------------- C4: attention
c4 = {}
recp = meta.get("rec_preds", ["R", "A", "P"])
for pi, pk in enumerate(recp[:3]):
    ps, pr, pc = AT[:, pi, 0, 0], AT[:, pi, 0, 1], AT[:, pi, 0, 2]
    okp = np.isfinite(ps) & (ps > 0) & (pr > 0) & VAL
    rows = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        for nm, m in (("agree", okp & ENC_HIT & (D >= max(lo, 1)) & (D < hi)), ("all", okp & (D >= max(lo, 1)) & (D < hi)),
                      ("clean", okp & ENC_HIT & ((E[:, 1] - E[:, 2]) > 0) & (D >= max(lo, 1)) & (D < hi))):
            if m.sum() < 200: continue
            rows[f"[{lo},{hi})|{nm}"] = {"n": int(m.sum()), "p_src": round(float(ps[m].mean()), 5), "p_ring": round(float(pr[m].mean()), 5), "p_copy": round(float(pc[m].mean()), 5),
                                        "src_over_ring_meanratio": round(float(ps[m].mean() / pr[m].mean()), 2), "median_tokenwise_src/ring": round(float(np.median(ps[m] / pr[m])), 2),
                                        "src_x4096(uniform null)": round(float(ps[m].mean() * 4096), 2), "frac_src_gt_copy": round(float((ps[m] > pc[m]).mean()), 3)}
    c4[pk] = rows
res["C4"] = c4
# ---------------- C5: mediation
def irls(X, y, it=12):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w))); W = p * (1 - p) + 1e-6
        H = X.T @ (X * W[:, None]); g = X.T @ (y - p); w = w + np.linalg.solve(H + 1e-6 * np.eye(len(w)), g)
    return w
def mediation(sub_idx=None):
    parts = []
    for pi, pk in enumerate(("R", "A", "P")):
        ps = AT[:, pi, 0, 0]; pc = AT[:, pi, 0, 2]; m = ENC_HIT & (D >= 1) & np.isfinite(ps) & (ps > 0) & (pc > 0)
        parts.append((pk, HIT[f"{pk}:base"][m].astype(float), D[m], np.log(ps[m]), np.log(ps[m] / pc[m]), CLIP[m]))
    y = np.concatenate([p[1] for p in parts]); d = np.concatenate([p[2] for p in parts]); ls = np.concatenate([p[3] for p in parts]); lc = np.concatenate([p[4] for p in parts]); cl = np.concatenate([p[5] for p in parts])
    isA = np.concatenate([np.full(len(p[1]), p[0] == "A", float) for p in parts]); isP = np.concatenate([np.full(len(p[1]), p[0] == "P", float) for p in parts])
    if sub_idx is not None and len(y) > sub_idx:
        s = rng.choice(len(y), sub_idx, replace=False); y, d, ls, lc, cl, isA, isP = y[s], d[s], ls[s], lc[s], cl[s], isA[s], isP[s]
    one = np.ones_like(y); out = {"n": int(len(y))}
    X0 = np.stack([one, d, isA, isP], 1); X1 = np.stack([one, d, isA, isP, ls], 1); X1c = np.stack([one, d, isA, isP, lc], 1)
    b0, b1, b1c = (np.linalg.lstsq(X, y, rcond=None)[0] for X in (X0, X1, X1c))
    out["linear"] = {"A_M0": round(float(b0[2]), 4), "A_M1_logpsrc": round(float(b1[2]), 4), "share_logpsrc": round(float(1 - b1[2] / b0[2]), 3), "A_M1_logratio": round(float(b1c[2]), 4), "share_logratio": round(float(1 - b1c[2] / b0[2]), 3)}
    l0, l1 = irls(X0, y), irls(X1, y)
    # 로짓 계수 매개 몫 + 평균 한계효과 (평균 예측확률 차) 로도
    def ame(w, X, col):
        Xa, Xb = X.copy(), X.copy(); Xa[:, col] = 1; Xb[:, col] = 0; Xa[:, 3] = 0; Xb[:, 3] = 0
        return float((1 / (1 + np.exp(-(Xa @ w))) - 1 / (1 + np.exp(-(Xb @ w)))).mean())
    out["logistic"] = {"A_logit_M0": round(float(l0[2]), 4), "A_logit_M1": round(float(l1[2]), 4), "share_logit": round(float(1 - l1[2] / l0[2]), 3),
                       "AME_A_M0": round(ame(l0, X0, 2), 4), "AME_A_M1": round(ame(l1, X1, 2), 4)}
    out["logistic"]["share_AME"] = round(float(1 - out["logistic"]["AME_A_M1"] / out["logistic"]["AME_A_M0"]), 3)
    # 거리 구간 × log p_src 십분위 층화 (비모수): 층 안 A−R 차의 가중 평균 vs 층화 없는 A−R 차
    strat = {}; tot_raw, tot_str, tot_w = 0, 0, 0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mb = (d >= lo) & (d < hi) & (isP == 0)
        if mb.sum() < 500: continue
        yA, yR, lsA, lsR = y[mb & (isA == 1)], y[mb & (isA == 0)], ls[mb & (isA == 1)], ls[mb & (isA == 0)]
        if len(yA) < 100 or len(yR) < 100: continue
        raw = yA.mean() - yR.mean(); q = np.quantile(ls[mb], np.linspace(0, 1, 11)[1:-1]); bA, bR = np.digitize(lsA, q), np.digitize(lsR, q)
        num = wsum = 0
        for k in range(10):
            a_, r_ = yA[bA == k], yR[bR == k]
            if len(a_) >= 20 and len(r_) >= 20:
                w_ = len(a_) + len(r_); num += w_ * (a_.mean() - r_.mean()); wsum += w_
        adj = num / wsum if wsum else float("nan")
        strat[f"[{lo},{hi})"] = {"raw_A_minus_R": round(float(raw), 4), "stratified_on_logpsrc": round(float(adj), 4), "share": round(float(1 - adj / raw), 3) if raw > 0 else None, "nA": int(len(yA)), "nR": int(len(yR))}
        tot_raw += raw * mb.sum(); tot_str += adj * mb.sum(); tot_w += mb.sum()
    out["stratified"] = strat; out["stratified_pooled_share"] = round(float(1 - tot_str / tot_raw), 3) if tot_w else None
    return out
res["C5"] = {"full_linear_and_logit(sub)": mediation(a.sub)}
# 토큰 짝 (같은 clip·슬롯·칸 에서 A 와 R 둘 다 attention 정의): Δhit ~ Δlog p_src
psR, psA = AT[:, 0, 0, 0], AT[:, 1, 0, 0]; m = ENC_HIT & (D >= 1) & (psR > 0) & (psA > 0) & np.isfinite(psR) & np.isfinite(psA)
dh = (HIT["A:base"].astype(float) - HIT["R:base"].astype(float))[m]; dl = (np.log(psA) - np.log(psR))[m]; dd = D[m]
X = np.stack([np.ones(len(dh)), dd, dl], 1); b = np.linalg.lstsq(X, dh, rcond=None)[0]
res["C5"]["token_paired"] = {"n": int(m.sum()), "mean_dhit": round(float(dh.mean()), 4), "mean_dlogpsrc": round(float(dl.mean()), 4), "slope_dhit_per_dlogpsrc": round(float(b[2]), 4),
                             "explained_share": round(float(b[2] * dl.mean() / dh.mean()), 3), "corr": round(float(np.corrcoef(dh, dl)[0, 1]), 3)}
print("RESULT " + json.dumps(res, default=float))
