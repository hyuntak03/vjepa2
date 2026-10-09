"""(S) 연속 판독 재계산: τ ∈ {0.02, 0.05, 0.1}, 복사 칸 지배 검사 (자기 칸 질량 · 자기 칸 제외 ρ), 회귀 clip-cluster bootstrap, 슬롯 고정 거리 삼분위 (거울)."""
import json, sys, numpy as np
from vcommon import *
name = sys.argv[1]; rng = np.random.default_rng(6)
R = Reach(name); CM = np.load(R.d / "cosmap.npy", mmap_mode="r"); preds = R.meta["rec_preds"][:CM.shape[1]]
px = CX.astype(np.float32); py = CY.astype(np.float32)
TAUS = (0.02, 0.05, 0.1)
rows = {pk: {t: dict(rho=[], rho_noself=[]) for t in TAUS} | dict(d=[], slot=[], clip=[], own_mass=[], argmax_mass=[], top_is_own=[]) for pk in preds}
eye = np.eye(256, dtype=bool)
for c in range(R.n):
    if not R.done[c]: continue
    m = R.HE[c] & (R.D[c] >= 1)
    if not m.any(): continue
    ux, uy = px[np.clip(R.SRC[c], 0, 255)], py[np.clip(R.SRC[c], 0, 255)]
    sxx, syy = np.broadcast_to(px[None], (8, 256)), np.broadcast_to(py[None], (8, 256))
    dvx, dvy = ux - sxx, uy - syy; d2 = np.maximum(dvx ** 2 + dvy ** 2, 1e-9)
    for pi, pk in enumerate(preds):
        cs = np.asarray(CM[c, pi], dtype=np.float32)
        if not np.isfinite(cs).all(): continue
        r = rows[pk]; r["d"].append(R.D[c][m]); r["slot"].append(R.SL[c][m]); r["clip"].append(np.full(int(m.sum()), c))
        for t in TAUS:
            w = np.exp((cs - cs.max(-1, keepdims=True)) / t); w /= w.sum(-1, keepdims=True)
            xh, yh = w @ px, w @ py
            rho = ((xh - sxx) * dvx + (yh - syy) * dvy) / d2; r[t]["rho"].append(rho[m])
            w2 = w * ~eye[None]; w2 /= np.maximum(w2.sum(-1, keepdims=True), 1e-12); xh2, yh2 = w2 @ px, w2 @ py
            r[t]["rho_noself"].append((((xh2 - sxx) * dvx + (yh2 - syy) * dvy) / d2)[m])
            if t == 0.05:
                r["own_mass"].append(w[..., eye][m] if False else np.take_along_axis(w, np.arange(256)[None, :, None], -1)[..., 0][m])
                am = cs.argmax(-1); r["argmax_mass"].append(np.take_along_axis(w, am[..., None], -1)[..., 0][m]); r["top_is_own"].append((am == np.arange(256)[None, :])[m])
res = {}
for pk in preds:
    r = rows[pk]
    if not r["d"]: continue
    d = np.concatenate(r["d"]); sl = np.concatenate(r["slot"]); cl = np.concatenate(r["clip"]); out = dict(n=int(len(d)))
    out["own_mass_mean"] = round(float(np.concatenate(r["own_mass"]).mean()), 4); out["argmax_mass_mean"] = round(float(np.concatenate(r["argmax_mass"]).mean()), 4); out["top_is_own_frac"] = round(float(np.concatenate(r["top_is_own"]).mean()), 4)
    for t in TAUS:
        for key in ("rho", "rho_noself"):
            rho = np.concatenate(r[t][key]); o = {}
            o["median_by_disp"] = {f"[{lo},{hi})": round(float(np.nanmedian(rho[(d >= lo) & (d < hi)])), 3) for lo, hi in BINS[1:] if ((d >= lo) & (d < hi)).sum() > 50}
            o["median_by_slot"] = [round(float(np.nanmedian(rho[sl == i])), 3) if (sl == i).sum() > 50 else None for i in range(8)]
            ok = np.isfinite(rho); y = np.clip(rho[ok], -1, 2); X = np.stack([np.ones(ok.sum()), sl[ok], d[ok]], 1); c_ = cl[ok]
            b0 = np.linalg.lstsq(X, y, rcond=None)[0]; uc = np.unique(c_); ix = {u: np.where(c_ == u)[0] for u in uc}; bs = []
            for _ in range(200):
                s = np.concatenate([ix[u] for u in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0])
            bs = np.array(bs); o["regress"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
            # 표준화 효과: 슬롯 1 SD · 거리 1 SD 당
            o["effect_per_sd"] = dict(slot=round(float(b0[1] * sl[ok].std()), 4), cell=round(float(b0[2] * d[ok].std()), 4), sd_slot=round(float(sl[ok].std()), 2), sd_cell=round(float(d[ok].std()), 2))
            # 거울: 슬롯 고정 (슬롯 2–5 각각) 거리 삼분위 ρ 중앙
            o["fixed_slot_by_disp_tercile"] = {}
            for i in (1, 3, 5, 7):
                mm = ok & (sl == i)
                if mm.sum() < 300: continue
                q = np.quantile(d[mm], [1 / 3, 2 / 3]); b = np.digitize(d[mm], q)
                o["fixed_slot_by_disp_tercile"][i] = dict(rho=[round(float(np.nanmedian(rho[mm][b == j])), 3) for j in range(3)], d_mean=[round(float(d[mm][b == j].mean()), 2) for j in range(3)])
            # 문서와 같은 거리 고정 (2–3 칸) 슬롯 삼분위
            mm = ok & (d >= 2) & (d < 3)
            if mm.sum() >= 300:
                q = np.quantile(sl[mm], [1 / 3, 2 / 3]); b = np.digitize(sl[mm], q)
                o["fixed_d23_by_slot_tercile"] = dict(rho=[round(float(np.nanmedian(rho[mm][b == j])), 3) for j in range(3)], slot_mean=[round(float(sl[mm][b == j].mean()), 1) for j in range(3)])
            out[f"tau{t}_{key}"] = o
    res[pk] = out
dump(f"cosmap_{name}", res)
for pk, o in res.items():
    print("==", name, pk, "n", o["n"], "own_mass", o["own_mass_mean"], "argmax_mass", o["argmax_mass_mean"], "top_is_own", o["top_is_own_frac"])
    for k, v in o.items():
        if k.startswith("tau"): print("  ", k, "reg", v["regress"], "perSD", v["effect_per_sd"], "\n      byd", v["median_by_disp"], "\n      byslot", v["median_by_slot"], "\n      fixslot", v["fixed_slot_by_disp_tercile"], "\n      fixd23", v.get("fixed_d23_by_slot_tercile"))
