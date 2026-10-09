#!/usr/bin/env python3
"""R3-(A): AR ep40 ≡ ep38, ctx_ar ep16 — 독립 재계산. scene_ar2_{pan,ssv2,perm}. 자체 판독 (analyzer 코드 안 씀)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); R = ROOT / "auto_research/_stage/results_ar"; OUT = ROOT / "auto_research/_verify_scratch/R3/out"; OUT.mkdir(parents=True, exist_ok=True)
G = 16; BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)]; rng = np.random.default_rng(123)


def cheb(a, b): return np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))


def boot_ratio(num, den, B=500):
    ok = den > 0; idx = np.where(ok)[0]
    if len(idx) == 0: return [np.nan] * 3
    r = num[idx].sum() / den[idx].sum(); bs = []
    for _ in range(B):
        s = rng.choice(idx, len(idx)); bs.append(num[s].sum() / den[s].sum())
    return [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def r50(curve):
    a = [c[0] for c in curve]; mids = [0.5 * (lo + hi) for lo, hi in BINS]
    half = 0.5 * a[0]
    for j in range(1, len(a)):
        if a[j] < half:
            f = (a[j - 1] - half) / max(a[j - 1] - a[j], 1e-9); return round(mids[j - 1] + f * (mids[j] - mids[j - 1]), 2)
    return float("inf")


def load(d):
    meta = json.load(open(d / "meta.json")); arms = meta["arms"]; items = meta["items"]; n = len(items)
    C = np.load(d / "arm_c.npy").astype(np.int64); CT = np.load(d / "arm_ct.npy").astype(np.float32); CC = np.load(d / "arm_cc.npy").astype(np.float32)
    E = np.load(d / "enc.npy"); EA = np.load(d / "enc_ar.npy")
    if (d / "src.npy").exists():
        SRC = np.load(d / "src.npy").astype(np.int64); D = np.nan_to_num(np.load(d / "disp.npy").astype(np.float32), nan=-1); V = np.zeros(n)
    else:
        sx = np.arange(256) % G; sy = np.arange(256) // G; SRC = np.full((n, 8, 256), -1); D = np.zeros((n, 8, 256), np.float32); V = np.array([it["v"] for it in items])
        for c, it in enumerate(items):
            for i in range(8):
                xp = 16 * sx + 8 + it["dir"] * it["v"] * 2 * (i + 1); ok = (xp >= 0) & (xp < 256); SRC[c, i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int); D[c, i] = it["v"] * (i + 1) / 8.0
    return meta, arms, C, CT, CC, E, EA, SRC, D, V


def analyze(name):
    d = R / name; meta, arms, C, CT, CC, E, EA, SRC, D, V = load(d); n = C.shape[0]
    val = SRC >= 0; done = C[:, 0, 0, 0] >= 0; val &= done[:, None, None]
    hitE = (cheb(E[:, 0].astype(np.int64), SRC) <= 1) & val; hitEA = (cheb(EA[:, 0].astype(np.int64), SRC) <= 1) & val
    AG = {"enc": hitE, "enc_ar": hitEA, "inter": hitE & hitEA}
    ai = {a: k for k, a in enumerate(arms)}
    res = {"n_done": int(done.sum()), "arms": arms}

    def curve(hit, mask, exact=False):
        out = []
        for lo, hi in BINS:
            m = mask & (D >= lo) & (D < hi); out.append(boot_ratio((hit & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float), B=300))
        return out

    H = {a: (cheb(C[:, ai[a]], SRC) <= 1) & val for a in arms}; HX = {a: (C[:, ai[a]] == SRC) & val for a in arms}
    for a in arms:
        ag = AG[meta["enc_for"][a]]
        cv = curve(H[a], ag); cvi = curve(H[a], AG["inter"]); cvx = curve(HX[a], ag)
        res[a] = dict(curve=[c[0] for c in cv], R50=r50(cv), R50_inter=r50(cvi), curve_inter=[c[0] for c in cvi], exact_curve=[c[0] for c in cvx], R50_exact=r50(cvx))
    # CX16 read against enc agreement (hc_L template not stored for CX16 → we can only switch the agreement set)
    res["CX16_roll_on_enc_agreement"] = [c[0] for c in curve(H["CX16:roll"], AG["enc"])]
    res["AR:roll_on_enc_agreement"] = [c[0] for c in curve(H["AR:roll"], AG["enc"])]
    # deltas (same tokens)
    def delta(a, b, mask):
        out = []
        for lo, hi in BINS:
            m = mask & (D >= lo) & (D < hi); out.append(boot_ratio(((H[a] & m).sum((1, 2)) - (H[b] & m).sum((1, 2))).astype(float), m.sum((1, 2)).astype(float), B=300))
        return out
    res["delta"] = {"AR40-AR38": delta("AR40:roll", "AR:roll", AG["enc_ar"]), "CX16-AR38(enc_ar)": delta("CX16:roll", "AR:roll", AG["enc_ar"]),
                    "CX16-AR38(inter)": delta("CX16:roll", "AR:roll", AG["inter"]), "CX16-B(inter)": delta("CX16:roll", "B:base", AG["inter"]),
                    "CX16roll-CX16tf": delta("CX16:roll", "CX16:tf", AG["enc_ar"]), "AR40roll-AR40tf": delta("AR40:roll", "AR40:tf", AG["enc_ar"])}
    # per slot roll-tf
    SL = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape)
    res["slot_delta"] = {k: [boot_ratio(((H[a] & AG["enc_ar"] & (SL == i)).sum((1, 2)) - (H[b] & AG["enc_ar"] & (SL == i)).sum((1, 2))).astype(float), (AG["enc_ar"] & (SL == i)).sum((1, 2)).astype(float), B=200) for i in range(8)]
                         for k, (a, b) in {"CX16": ("CX16:roll", "CX16:tf"), "AR40": ("AR40:roll", "AR40:tf")}.items()}
    # per speed (pan only)
    if V.any():
        res["per_speed"] = {}
        for v in sorted(set(V.tolist())):
            mv = (V == v)[:, None, None]
            for a in ("AR:roll", "AR40:roll", "CX16:roll", "B:base", "R:base"):
                cv = curve(H[a], AG["inter"] & mv); res["per_speed"][f"v{int(v)}|{a}"] = dict(curve=[c[0] for c in cv], R50=r50(cv))
    # static (0-1) with CI, and exact-cell static
    for a in ("AR:roll", "AR40:roll", "CX16:roll", "B:base", "R:base"):
        m = AG["inter"] & (D >= 0) & (D < 1); res[f"static01_inter|{a}"] = boot_ratio((H[a] & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float))
        res[f"static01_exact_inter|{a}"] = boot_ratio((HX[a] & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float))
    # soft: ct - cc per slot (AR40, CX16) — sign flip slot
    for a in ("AR40:roll", "CX16:roll", "AR:roll"):
        soft = CT[:, ai[a]] - CC[:, ai[a]]; res[f"soft_by_slot|{a}"] = [round(float(np.nanmean(np.where(AG["enc_ar"] & (SL == i) & (D >= 1), soft, np.nan))), 4) for i in range(8)]
    json.dump(res, open(OUT / f"a_{name}.json", "w"), indent=1, default=float)
    print("==", name, "done", res["n_done"])
    for a in arms: print(f"  {a:11s} R50 {res[a]['R50']:>5} inter {res[a]['R50_inter']:>5} exact {res[a]['R50_exact']:>5} | curve", " ".join(f"{x:.2f}" for x in res[a]["curve"]))
    for k, v in res["delta"].items(): print("  Δ", k, " ".join(f"{d[0]:+.3f}[{d[1]:+.3f},{d[2]:+.3f}]" for d in v))
    for k, v in res["slot_delta"].items(): print("  slot roll-tf", k, " ".join(f"{d[0]:+.3f}" for d in v))
    for k in [k for k in res if k.startswith("static01")]: print(" ", k, res[k])
    for k in [k for k in res if k.startswith("soft")]: print(" ", k, res[k])
    if "per_speed" in res:
        for k, v in res["per_speed"].items(): print("  ", k, "R50", v["R50"], " ".join(f"{x:.2f}" for x in v["curve"]))
    return res


def perm():
    d = R / "scene_ar2_perm"; meta = json.load(open(d / "meta.json")); preds = meta["preds"]; items = meta["items"]; n = len(items)
    AM = np.load(d / "argmax_prev.npy").astype(np.int64); CP = np.load(d / "cos_prev.npy").astype(np.float32); CL = np.load(d / "cos_last.npy").astype(np.float32); HID = np.load(d / "hid.npy")
    K = np.array([it["k"] for it in items]); Vv = np.array([it["v"] for it in items]); DR = np.array([it["dir"] for it in items]); M = (K + 1) // 2
    sx = np.arange(256) % G; sy = np.arange(256) // G; SRC = np.full((n, 8, 256), -1); D = np.zeros((n, 8, 256), np.float32)
    for c in range(n):
        for i in range(8):
            xp = 16 * sx + 8 + DR[c] * Vv[c] * (2 * (i + 1) + 2 * M[c]); ok = (xp >= 0) & (xp < 256); SRC[c, i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int); D[c, i] = Vv[c] * (2 * (i + 1)) / 16.0
    done = AM[:, 0, 0, 0] >= 0; VAL = (SRC >= 0) & done[:, None, None]
    res = {}
    for pi, pk in enumerate(preds):
        if pk not in ("R:base", "B:base", "AR:roll", "AR40:roll", "CX16:roll"): continue
        hit = (cheb(AM[:, pi], SRC) <= 1) & VAL
        for k in (2, 4, 8):
            m = VAL & (K == k)[:, None, None] & (D < 2)
            mh, mv = m & HID, m & ~HID
            # ratio with clip bootstrap
            nh, dh = (hit & mh).sum((1, 2)).astype(float), mh.sum((1, 2)).astype(float); nv, dv = (hit & mv).sum((1, 2)).astype(float), mv.sum((1, 2)).astype(float)
            idx = np.where((dh > 0) & (dv > 0))[0]; rs = []
            for _ in range(500):
                s = rng.choice(idx, len(idx)); rs.append((nh[s].sum() / dh[s].sum()) / max(nv[s].sum() / dv[s].sum(), 1e-9))
            r = (nh[idx].sum() / dh[idx].sum()) / (nv[idx].sum() / dv[idx].sum())
            # paints occluder: cos_prev - cos_last on hid tokens
            dpl = (CP[:, pi] - CL[:, pi])[mh]; cl = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape)[mh]; uc = np.unique(cl); bs = []
            for _ in range(300):
                sc = rng.choice(uc, len(uc)); sel = np.isin(cl, sc); bs.append(float(np.nanmean(dpl[sel])))
            res[f"{pk}|k{k}"] = dict(hid=round(float(nh[idx].sum() / dh[idx].sum()), 3), vis=round(float(nv[idx].sum() / dv[idx].sum()), 3), ratio=[round(float(r), 3), round(float(np.percentile(rs, 2.5)), 3), round(float(np.percentile(rs, 97.5)), 3)],
                                     cos_prev_minus_last_hid=[round(float(np.nanmean(dpl)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
    json.dump(res, open(OUT / "a_perm.json", "w"), indent=1, default=float)
    print("== perm"); [print(" ", k, v) for k, v in res.items()]
    return res


if __name__ == "__main__":
    out = {"pan": analyze("scene_ar2_pan"), "ssv2": analyze("scene_ar2_ssv2"), "perm": perm()}
