#!/usr/bin/env python3
"""VERIFY_AR — AR_PREDICTOR_2026-09-27.md §2-1 / 2-1-b / 2-2 / 2-3 독립 재계산 (e_scene_ar_analyze 로직 불사용).
python v_ar.py  → out/verify_ar.json + 콘솔.  판독 정의는 문서와 같다 (Chebyshev ≤ 1, 합의 = encoder 판독 적중, clip bootstrap).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); R = ROOT / "auto_research/_stage/results_ar"
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
G = 16; rng = np.random.default_rng(1); EDGES = [0, 1, 2, 3, 4, 6, 9, 13]
res = {}


def cheb(a, b):
    return np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))


def boot_ratio(num, den, B=500):
    """clip 단위 합 (n,) → 비율 + CI."""
    ok = den > 0; idx = np.where(ok)[0]
    if len(idx) == 0: return [float("nan")] * 3
    r = num[idx].sum() / den[idx].sum()
    bs = []
    for _ in range(B):
        s = rng.choice(idx, len(idx)); bs.append(num[s].sum() / den[s].sum())
    return [round(float(r), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def boot_mean(v, clip, B=400):
    ok = np.isfinite(v); v, clip = v[ok], clip[ok]
    if len(v) == 0: return [float("nan")] * 3
    uc = np.unique(clip); sums = np.bincount(np.searchsorted(uc, clip), weights=v, minlength=len(uc)); cnt = np.bincount(np.searchsorted(uc, clip), minlength=len(uc)).astype(float)
    r = sums.sum() / cnt.sum(); bs = []
    for _ in range(B):
        s = rng.choice(len(uc), len(uc)); bs.append(sums[s].sum() / cnt[s].sum())
    return [round(float(r), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def curve(hit, mask, Dd, edges=EDGES):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (Dd >= lo) & (Dd < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), A=boot_ratio((hit & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float), B=300)))
    return out


def r50(cv):
    pts = [(0.5 * (c["lo"] + c["hi"]), c["A"][0]) for c in cv if c["n"] > 0 and np.isfinite(c["A"][0])]
    if not pts: return float("nan")
    half = 0.5 * pts[0][1]
    for j in range(1, len(pts)):
        if pts[j][1] < half:
            x0, y0 = pts[j - 1]; x1, y1 = pts[j]
            return round(float(x0 + (y0 - half) / max(y0 - y1, 1e-9) * (x1 - x0)), 2)
    return float("inf")


def load_reach(name):
    I = R / name; meta = json.load(open(I / "meta.json")); arms = meta["arms"]; items = meta["items"]; n = len(items)
    C = np.load(I / "arm_c.npy").astype(np.int64); CT = np.load(I / "arm_ct.npy").astype(np.float32); CC = np.load(I / "arm_cc.npy").astype(np.float32)
    E = np.load(I / "enc.npy"); EA = np.load(I / "enc_ar.npy")
    if (I / "src.npy").exists():
        SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.load(I / "disp.npy").astype(np.float32); Dd = np.where(np.isfinite(Dd), Dd, -1.0)
    else:
        sx = np.arange(256) % G; sy = np.arange(256) // G
        SRC = np.full((n, 8, 256), -1, np.int64); Dd = np.zeros((n, 8, 256), np.float32)
        for c, it in enumerate(items):
            for i in range(8):
                xp = 16 * sx + 8 + it["dir"] * it["v"] * 2 * (i + 1); ok = (xp >= 0) & (xp < 256)
                SRC[c, i, ok] = sy[ok] * G + (xp[ok] // 16); Dd[c, i] = it["v"] * 2 * (i + 1) / 16.0
    done = C[:, 0, 0, 0] >= 0
    VAL = (SRC >= 0) & done[:, None, None]
    return dict(meta=meta, arms=arms, items=items, n=n, C=C, CT=CT, CC=CC, E=E, EA=EA, SRC=SRC, Dd=Dd, VAL=VAL, done=done)


def analyze_reach(name):
    d = load_reach(name); arms = d["arms"]; SRC = d["SRC"]; Dd = d["Dd"]; VAL = d["VAL"]; n = d["n"]; C = d["C"]
    ai = {a: k for k, a in enumerate(arms)}
    hit = {a: (cheb(C[:, k], SRC) <= 1) & VAL for a, k in ai.items()}
    exact = {a: (C[:, k] == SRC) & VAL for a, k in ai.items()}
    AGe = (cheb(d["E"][:, 0].astype(np.int64), SRC) <= 1) & VAL; AGa = (cheb(d["EA"][:, 0].astype(np.int64), SRC) <= 1) & VAL
    AG = {"enc": AGe, "enc_ar": AGa, "both": AGe & AGa}
    EF = d["meta"]["enc_for"]
    SL = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape); CLIP = np.broadcast_to(np.arange(n)[:, None, None], SRC.shape)
    out = dict(n=n, n_done=int(d["done"].sum()), sanity=dict(nan_ct=int((~np.isfinite(d["CT"][d["done"]])).sum()), neg1_c=int((C[d["done"]] < 0).sum()),
                                                            agree_rate=dict(enc=round(float(AGe[VAL].mean()), 4), enc_ar=round(float(AGa[VAL].mean()), 4), both=round(float((AGe & AGa)[VAL].mean()), 4)),
                                                            valid_share=round(float(VAL[d["done"]].mean()), 4)))
    # (a) 문서 재계산: 각 팔 자기 합의 집합
    out["doc"] = {}
    for a in arms:
        cv = curve(hit[a], AG[EF[a]], Dd); out["doc"][a] = dict(curve=[c["A"][0] for c in cv], R50=r50(cv), by_slot=[round(float(hit[a][AG[EF[a]] & (SL == i)].mean()), 3) for i in range(8)])
    # (1) 템플릿 교란: 교집합 합의 · 정확 칸 · roll_x on enc
    out["intersection"] = {}
    for a in ("AR:roll", "AR:tf", "AR:roll_x", "B:base", "F:base", "R:base"):
        cv = curve(hit[a], AG["both"], Dd); out["intersection"][a] = dict(curve=[c["A"][0] for c in cv], R50=r50(cv))
    out["delta_both"] = {}
    for a, b in (("AR:roll", "B:base"), ("AR:roll_x", "B:base"), ("AR:roll", "AR:tf")):
        dd = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = AG["both"] & (Dd >= lo) & (Dd < hi)
            dd.append(boot_ratio(((hit[a] & m).sum((1, 2)) - (hit[b] & m).sum((1, 2))).astype(float), m.sum((1, 2)).astype(float), B=300))
        out["delta_both"][f"{a}-{b}"] = dd
    out["exact"] = {}
    for a in ("AR:roll", "AR:tf", "B:base", "R:base", "F:base"):
        cv = curve(exact[a], AG[EF[a]], Dd); out["exact"][a] = dict(curve=[c["A"][0] for c in cv], R50=r50(cv))
    # (3) 거리 고정 슬롯 삼분위 · 슬롯별 ct−cc · argmax 분포
    out["fixed_d_slot_terciles"] = {}
    for a in ("AR:roll", "AR:tf", "B:base", "R:base"):
        rows = []
        for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 6)):
            m = AG[EF[a]] & (Dd >= lo) & (Dd < hi)
            if m.sum() < 300: continue
            sl = SL[m]; q = np.quantile(sl, [1 / 3, 2 / 3]); b = np.digitize(sl, q)
            rows.append(dict(lo=lo, hi=hi, A=[round(float(hit[a][m][b == j].mean()), 3) for j in range(3)], slot_mean=[round(float(sl[b == j].mean()), 2) for j in range(3)], n=[int((b == j).sum()) for j in range(3)]))
        out["fixed_d_slot_terciles"][a] = rows
    out["soft_by_slot"] = {}
    for a in ("AR:roll", "AR:tf", "B:base", "R:base", "F:base"):
        k = ai[a]; sf = d["CT"][:, k] - d["CC"][:, k]; m = AG[EF[a]] & (Dd >= 1)
        out["soft_by_slot"][a] = [boot_mean(np.where(m & (SL == i), sf, np.nan).ravel(), CLIP.ravel(), B=200) for i in range(8)]
        out["soft_by_slot"][a + "_ct"] = [round(float(np.nanmean(np.where(m & (SL == i), d["CT"][:, k], np.nan))), 4) for i in range(8)]
        out["soft_by_slot"][a + "_cc"] = [round(float(np.nanmean(np.where(m & (SL == i), d["CC"][:, k], np.nan))), 4) for i in range(8)]
    # argmax 가 어디에 떨어지나 (합의 · d ≥ 2): 출처 1 칸 / 자기 칸 1 칸 / 그 밖 — 그리고 슬롯 안에서 argmax 칸의 집중도 (상위 1 칸 점유율)
    out["argmax_where"] = {}
    for a in ("AR:roll", "AR:tf", "B:base", "R:base"):
        k = ai[a]; m = AG[EF[a]] & (Dd >= 2); own = np.broadcast_to(np.arange(256)[None, None, :], SRC.shape)
        near_src = cheb(C[:, k], SRC) <= 1; near_own = cheb(C[:, k], own) <= 1
        rows = []
        for i in range(8):
            mm = m & (SL == i)
            if mm.sum() == 0: rows.append(None); continue
            cc = C[:, k][mm]; top = np.bincount(cc, minlength=256).max() / len(cc)
            rows.append(dict(near_src=round(float(near_src[mm].mean()), 3), near_own=round(float(near_own[mm].mean()), 3), elsewhere=round(float((~near_src & ~near_own)[mm].mean()), 3), top1_cell_share=round(float(top), 3), n=int(mm.sum())))
        out["argmax_where"][a] = rows
    # (2) 속도별 (팬만)
    if "v" in d["items"][0]:
        V = np.array([it["v"] for it in d["items"]]); out["by_speed"] = {}
        for v in sorted(set(V.tolist())):
            mv = (V == v)[:, None, None]; row = {}
            for a in ("AR:roll", "AR:tf", "B:base", "R:base", "F:base"):
                cv = curve(hit[a], AG[EF[a]] & mv, Dd); row[a] = dict(curve=[c["A"][0] for c in cv], n=[c["n"] for c in cv], R50=r50(cv), by_slot=[round(float(hit[a][AG[EF[a]] & mv & (SL == i)].mean()), 3) for i in range(8)])
            # 교집합 합의에서 AR:roll − B
            dd = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = AG["both"] & mv & (Dd >= lo) & (Dd < hi)
                dd.append(boot_ratio(((hit["AR:roll"] & m).sum((1, 2)) - (hit["B:base"] & m).sum((1, 2))).astype(float), m.sum((1, 2)).astype(float), B=200) if m.sum() else [float("nan")] * 3)
            row["delta_roll-B_both"] = dd; out["by_speed"][f"v{v}"] = row
    # (6) closure: roll − tf 슬롯별 (enc_ar 합의)
    out["closure_by_slot"] = []
    for i in range(8):
        m = AGa & (SL == i)
        out["closure_by_slot"].append(boot_ratio(((hit["AR:roll"] & m).sum((1, 2)) - (hit["AR:tf"] & m).sum((1, 2))).astype(float), m.sum((1, 2)).astype(float), B=300))
    k0, k1 = ai["AR:roll"], ai["AR:tf"]
    out["slot0_identical"] = dict(argmax_eq=float((C[d["done"], k0, 0] == C[d["done"], k1, 0]).mean()), max_abs_ct=float(np.nanmax(np.abs(d["CT"][d["done"], k0, 0] - d["CT"][d["done"], k1, 0]))))
    if "src" in d["meta"] or (R / name / "src.npy").exists():
        S = np.load(R / name / "src.npy"); Dr = np.load(R / name / "disp.npy").astype(np.float32)
        out["flow_validity"] = dict(src_ge0=round(float((S[d["done"]] >= 0).mean()), 4), disp_finite=round(float(np.isfinite(Dr[d["done"]]).mean()), 4))
    return out


def analyze_perm():
    I = R / "scene_ar_perm"; meta = json.load(open(I / "meta.json")); arms = meta["arms"]; items = meta["items"]; n = len(items)
    AM = np.load(I / "argmax_prev.npy").astype(np.int64); CP = np.load(I / "cos_prev.npy").astype(np.float32); CL = np.load(I / "cos_last.npy").astype(np.float32); HID = np.load(I / "hid.npy")
    EP_ = np.load(I / "enc_prev.npy").astype(np.float32); EPA = np.load(I / "enc_prev_ar.npy").astype(np.float32)
    done = AM[:, 0, 0, 0] >= 0
    K = np.array([it["k"] for it in items]); V = np.array([it["v"] for it in items]); DR = np.array([it["dir"] for it in items]); M = (K + 1) // 2
    sx = np.arange(256) % G; sy = np.arange(256) // G
    SRCp = np.full((n, 8, 256), -1, np.int64); Dd = np.zeros((n, 8, 256), np.float32)
    for c in range(n):
        for i in range(8):
            xp = 16 * sx + 8 + DR[c] * V[c] * (2 * (i + 1) + 2 * M[c]); ok = (xp >= 0) & (xp < 256)
            SRCp[c, i, ok] = sy[ok] * G + (xp[ok] // 16); Dd[c, i] = V[c] * 2 * (i + 1) / 16.0
    VAL = (SRCp >= 0) & done[:, None, None]
    ai = {a: k for k, a in enumerate(arms)}
    out = dict(n=n, n_done=int(done.sum()), sanity=dict(neg1=int((AM[done] < 0).sum()), nan_cp=int((~np.isfinite(CP[done]) & VAL[:, None][done]).sum())))
    out["ratio_0to2"] = {}; out["paints"] = {}; out["A"] = {}
    for a in arms:
        k = ai[a]; hit = (cheb(AM[:, k], SRCp) <= 1) & VAL
        out["ratio_0to2"][a] = {}; out["paints"][a] = {}; out["A"][a] = {}
        for kk in meta["ks"]:
            m = VAL & (K == kk)[:, None, None] & (Dd >= 0) & (Dd < 2)
            mh, mv = m & HID, m & ~HID
            nh, dh = (hit & mh).sum((1, 2)).astype(float), mh.sum((1, 2)).astype(float); nv, dv = (hit & mv).sum((1, 2)).astype(float), mv.sum((1, 2)).astype(float)
            idx = np.where((dh > 0) & (dv > 0))[0]; r = (nh[idx].sum() / dh[idx].sum()) / (nv[idx].sum() / dv[idx].sum()); bs = []
            for _ in range(400):
                s = rng.choice(idx, len(idx)); bs.append((nh[s].sum() / dh[s].sum()) / (nv[s].sum() / dv[s].sum()))
            out["ratio_0to2"][a][f"k{kk}"] = [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            out["A"][a][f"k{kk}"] = dict(hid=boot_ratio(nh, dh, B=300), vis=boot_ratio(nv, dv, B=300))
            # 가림막 그리기: hid 토큰에서 cos_prev − cos_last (음수 = 가림막 쪽)
            dif = np.where(mh, CP[:, k] - CL[:, k], np.nan); CLIP = np.broadcast_to(np.arange(n)[:, None, None], dif.shape)
            out["paints"][a][f"k{kk}"] = dict(cos_prev_minus_last_hid=boot_mean(dif.ravel(), CLIP.ravel(), B=200),
                                             cos_prev_hid=round(float(np.nanmean(CP[:, k][mh])), 4), cos_last_hid=round(float(np.nanmean(CL[:, k][mh])), 4),
                                             enc_prev_hid=round(float(np.nanmean((EPA if a.startswith("AR:") and a != "AR:roll_x" else EP_)[mh])), 4))
    k0, k1 = ai["AR:roll"], ai["AR:tf"]
    out["slot0_identical"] = dict(argmax_eq=float((AM[done, k0, 0] == AM[done, k1, 0]).mean()), max_abs_cp=float(np.nanmax(np.abs(CP[done, k0, 0] - CP[done, k1, 0]))))
    # tf 가 permanence 측정이 아닌지: hid 토큰 슬롯별 hit (tf 는 슬롯 1 부터 재관측)
    out["tf_by_slot_hid_k4"] = {}
    for a in ("AR:roll", "AR:tf", "B:base"):
        k = ai[a]; hit = (cheb(AM[:, k], SRCp) <= 1) & VAL; m = VAL & (K == 4)[:, None, None] & HID & (Dd < 3)
        out["tf_by_slot_hid_k4"][a] = [round(float(hit[m & (np.arange(8)[None, :, None] == i)].mean()), 3) if (m & (np.arange(8)[None, :, None] == i)).sum() > 30 else None for i in range(8)]
    return out


if __name__ == "__main__":
    res["pan"] = analyze_reach("scene_ar_pan"); print("pan done", flush=True)
    res["ssv2"] = analyze_reach("scene_ar_ssv2"); print("ssv2 done", flush=True)
    res["perm"] = analyze_perm(); print("perm done", flush=True)
    json.dump(res, open(OUT / "verify_ar.json", "w"), indent=1, default=float)
    for nm in ("pan", "ssv2"):
        r = res[nm]; print(f"\n==== {nm}  n_done {r['n_done']}/{r['n']}  sanity {r['sanity']}")
        for a, v in r["doc"].items(): print(f"  doc {a:10s} A {[round(x, 2) for x in v['curve']]} R50 {v['R50']} slot {v['by_slot']}")
        for a, v in r["intersection"].items(): print(f"  both {a:10s} A {[round(x, 2) for x in v['curve']]} R50 {v['R50']}")
        for a, v in r["exact"].items(): print(f"  exact {a:10s} A {[round(x, 2) for x in v['curve']]} R50 {v['R50']}")
        for a, v in r["delta_both"].items(): print(f"  Δboth {a:16s}", " ".join(f"{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]" for x in v))
        for a, v in r["fixed_d_slot_terciles"].items(): print(f"  fixed-d {a:8s}", [(t["lo"], t["hi"], t["A"], t["slot_mean"]) for t in v])
        for a in ("AR:roll", "AR:tf", "B:base", "R:base"): print(f"  soft {a:8s}", " ".join(f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for x in r["soft_by_slot"][a]), " ct", r["soft_by_slot"][a + "_ct"], " cc", r["soft_by_slot"][a + "_cc"])
        for a, v in r["argmax_where"].items(): print(f"  argmax {a:8s}", [(x["near_src"], x["near_own"], x["elsewhere"], x["top1_cell_share"]) if x else None for x in v])
        print("  closure roll−tf by slot", " ".join(f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for x in r["closure_by_slot"]), " slot0 identical", r["slot0_identical"])
        if "by_speed" in r:
            for v, row in r["by_speed"].items():
                print(f"  speed {v}: " + " | ".join(f"{a} A {[round(x, 2) if np.isfinite(x) else None for x in row[a]['curve']]} R50 {row[a]['R50']}" for a in ("AR:roll", "AR:tf", "B:base")))
                print(f"           Δ roll−B (both) " + " ".join(f"{x[0]:+.2f}" if np.isfinite(x[0]) else "  nan" for x in row["delta_roll-B_both"]) + f"  slots roll {row['AR:roll']['by_slot']} B {row['B:base']['by_slot']}")
        if "flow_validity" in r: print("  flow validity", r["flow_validity"])
    p = res["perm"]; print(f"\n==== perm n_done {p['n_done']}/{p['n']} sanity {p['sanity']} slot0 identical {p['slot0_identical']}")
    for a in p["ratio_0to2"]: print(f"  {a:10s} ratio hid/vis 0–2:", p["ratio_0to2"][a], " A k4:", p["A"][a]["k4"], " paints k4:", p["paints"][a]["k4"])
    print("  tf by slot hid k4:", p["tf_by_slot_hid_k4"])
