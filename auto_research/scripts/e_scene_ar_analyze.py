#!/usr/bin/env python3
"""e_scene_ar.py (--task reach) 산출물 분석. python e_scene_ar_analyze.py <dir>   (OUT_TAG 접미사)

e_scene_reach_analyze 와 같은 정의 (Chebyshev ≤ 1 · 합의 토큰 · 구간 · clip bootstrap · R50) 인데
**팔마다 encoder 천장이 다르다** (meta.enc_for: 표준 팔 = enc (hc_L 템플릿), AR 팔 = enc_ar (h_blk[7] 템플릿)).
추가: AR:roll − AR:tf (자기 예측 되먹임 vs 진짜 블록 되먹임 — closure 의 AR 판) 를 같은 토큰에서, 구간별 · 슬롯별.
      F:base − R:base, B:base − R:base, AR:roll − B:base (다른 합의 집합이라 교집합 토큰에서).
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; items = meta["items"]; n = len(items); EF = meta["enc_for"]
C = np.load(I / "arm_c.npy").astype(np.int64); CT = np.load(I / "arm_ct.npy").astype(np.float32); CC = np.load(I / "arm_cc.npy").astype(np.float32)
ENC = {"enc": np.load(I / "enc.npy"), "enc_ar": np.load(I / "enc_ar.npy")}
done = C[:, 0, 0, 0] >= 0; print("완료 clip", int(done.sum()), "/", n)
G = 16; rng = np.random.default_rng(0); BINS = [0, 1, 2, 3, 4, 6, 9, 13]
if (I / "src.npy").exists():
    SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.nan_to_num(np.load(I / "disp.npy").astype(np.float32), nan=-1.0)
else:
    sx = np.arange(256) % G; sy = np.arange(256) // G
    def src_of(v, dr):
        src = np.full((8, 256), -1)
        for i in range(8):
            xp = 16 * sx + 8 + dr * v * 2 * (i + 1); ok = (xp >= 0) & (xp < 256); src[i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int)
        return src
    SRC = np.stack([src_of(it["v"], it["dir"]) for it in items]); Dd = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items])
VAL = (SRC >= 0) & done[:, None, None]; SL = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape)
cheb_hit = lambda c: (np.maximum(np.abs(c % G - SRC % G), np.abs(c // G - SRC // G)) <= 1) & VAL
AG = {k: cheb_hit(v[:, 0].astype(np.int64)) for k, v in ENC.items()}
H = {a: cheb_hit(C[:, k]) for k, a in enumerate(arms)}


def boot_clip(num, den, B=500):
    ok = den > 0
    if ok.sum() == 0: return [np.nan] * 3
    r = num[ok].sum() / den[ok].sum(); idx = np.where(ok)[0]
    bs = [num[s].sum() / den[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(B))]
    return [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def curve(hit, mask):
    return [dict(lo=lo, hi=hi, n=int((mask & (Dd >= lo) & (Dd < hi)).sum()),
                 A=boot_clip((hit & mask & (Dd >= lo) & (Dd < hi)).sum((1, 2)).astype(float), (mask & (Dd >= lo) & (Dd < hi)).sum((1, 2)).astype(float)))
            for lo, hi in zip(BINS[:-1], BINS[1:])]


def r50(cv):
    a = [c["A"][0] for c in cv if c["n"] > 0]; lo = [0.5 * (c["lo"] + c["hi"]) for c in cv if c["n"] > 0]
    if not a or not np.isfinite(a[0]): return np.nan
    half = 0.5 * a[0]
    for j in range(1, len(a)):
        if a[j] < half:
            f = (a[j - 1] - half) / max(a[j - 1] - a[j], 1e-9); return round(float(lo[j - 1] + f * (lo[j] - lo[j - 1])), 2)
    return float("inf")


def delta(a, b, mask):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = mask & (Dd >= lo) & (Dd < hi)
        out.append(boot_clip(((H[a] & m).sum((1, 2)) - (H[b] & m).sum((1, 2))).astype(float), m.sum((1, 2)).astype(float)))
    return out


def by_slot(a, mask):
    return [boot_clip((H[a] & mask & (SL == i)).sum((1, 2)).astype(float), (mask & (SL == i)).sum((1, 2)).astype(float), B=200)[0] for i in range(8)]


res = dict(n_done=int(done.sum()), source=meta["source"], arms={}, enc={k: curve(v, VAL) for k, v in AG.items()})
for a in arms:
    ag = AG[EF[a]]; cv = curve(H[a], ag)
    res["arms"][a] = dict(enc=EF[a], curve=cv, R50=r50(cv), by_slot=by_slot(a, ag),
                          soft=[round(float(np.nanmean(np.where(ag & (Dd >= max(lo, 1)) & (Dd < hi), CT[:, arms.index(a)] - CC[:, arms.index(a)], np.nan))), 4)
                                for lo, hi in zip(BINS[:-1], BINS[1:])])
    m = ag & (Dd >= 0); y = H[a][m].astype(float); X = np.stack([np.ones(m.sum()), SL[m], Dd[m]], 1); cl = np.broadcast_to(np.arange(n)[:, None, None], m.shape)[m]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; bs = []
    for _ in range(200):
        s = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0])
    bs = np.array(bs)
    res["arms"][a]["slot_vs_dist"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
res["delta"] = {"AR:roll-AR:tf": delta("AR:roll", "AR:tf", AG["enc_ar"]),
                **{f"{k}:roll-{k}:tf": delta(f"{k}:roll", f"{k}:tf", AG["enc_ar"]) for k in {a.split(":")[0] for a in arms} - {"R", "F", "B", "AR"}},
                **{f"{k}:roll-AR:roll": delta(f"{k}:roll", "AR:roll", AG["enc_ar"]) for k in {a.split(":")[0] for a in arms} - {"R", "F", "B", "AR"}}, "F:base-R:base": delta("F:base", "R:base", AG["enc"]), "B:base-R:base": delta("B:base", "R:base", AG["enc"]),
                "AR:roll-B:base (교집합)": delta("AR:roll", "B:base", AG["enc"] & AG["enc_ar"]), "AR:roll_x-B:base": delta("AR:roll_x", "B:base", AG["enc"])}
json.dump(res, open(OUT / f"scene_ar{TAG}.json", "w"), indent=1, default=float)
print("구간 (칸):", " ".join(f"[{lo},{hi})" for lo, hi in zip(BINS[:-1], BINS[1:])))
for k, cv in res["enc"].items(): print(f"{k:14s}", " ".join(f"{c['A'][0]:.2f}" for c in cv), "  n", [c["n"] for c in cv])
for a in arms:
    r = res["arms"][a]; print(f"{a:12s}", " ".join(f"{c['A'][0]:.2f}" for c in r["curve"]), f"  R50 {r['R50']}", "  슬롯 " + " ".join(f"{x:.2f}" for x in r["by_slot"]),
                              f"  회귀 slot {r['slot_vs_dist']['per_slot'][0]:+.3f} cell {r['slot_vs_dist']['per_cell'][0]:+.3f}")
for k, dd in res["delta"].items(): print(f"Δ {k:26s}", " ".join(f"{d[0]:+.2f}[{d[1]:+.2f},{d[2]:+.2f}]" for d in dd))
