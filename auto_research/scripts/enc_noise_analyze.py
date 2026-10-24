#!/usr/bin/env python3
"""고리 1 [E] — enc_noise_collect.py 산출물 분석 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/enc_noise_analyze.py [--dir /workspace/auto_research_cache/enc_noise] [--enc target_encoder]

출력 (소스마다):
  ref       척도: 서로 다른 창 같은 칸 · 같은 블록 번호 토큰 사이 차원 평균 |Δh| (cross) · 픽셀이 크게 바뀐 칸 (≥ 16 단계) 의 블록 간 변화 (moving)
  curve     블록 간 latent 변화 d0 를 칸의 픽셀 변화 (칸 평균 |Δ픽셀|, 0–255) 구간별로
  freeze    정지 칸 (|Δ픽셀| < 2) 의 fp32 변화 = 번짐 (leak, 다른 칸 변화) vs 미세 (local, 정지 칸들 자신의 미세 픽셀 변화)
  nuisance  같은 블록 · 물리적으로 같은 내용의 다른 표현: ±1/255 반복 · bf16↔fp32 · 재압축 · 밝기 · 0.5 px
  variogram 창 내내 정지인 칸: lag k 블록 떨어진 토큰 사이 |Δh| (k = 1..7) → k→0 외삽 = 블록마다 독립인 몫 (nugget)
  share     복사 손실 (블록 간 |Δh|) 중 잡음 몫 (세 가지 추정)
  texture   정지 칸을 무늬 (칸 안 픽셀 표준편차) 로 나눠
  pool      p×p 칸 평균 토큰에서 잡음 / 장면 차이
  pca       잡음 방향 vs 장면 신호 PCA, 움직임 변화 방향"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--enc", default="target_encoder")
ap.add_argument("--sources", nargs="+", default=["phys3d", "vobench", "k710", "ssv2", "ht100m", "ek100"])
ap.add_argument("--pca_n", type=int, default=6000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
G = 16
BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 999)]
rng = np.random.default_rng(0)


def load(src):
    fs = sorted(glob.glob(os.path.join(a.dir, f"{src}_{a.enc}_s*.npz")))
    if not fs:
        return None
    D = {}
    for f in fs:
        z = np.load(f)
        for k in z.files:
            D.setdefault(k, []).append(z[k])
    D = {k: np.concatenate(v) for k, v in D.items()}
    toks = [np.load(f[:-4] + "_tok.npy", mmap_mode="r") for f in fs if os.path.exists(f[:-4] + "_tok.npy")]
    pix = [np.load(f[:-4] + "_pix.npy", mmap_mode="r") for f in fs if os.path.exists(f[:-4] + "_pix.npy")]
    meta = sum([json.load(open(f[:-4] + "_meta.json")) for f in fs], [])
    return D, toks, pix, meta


def tok_iter(toks):
    for t in toks:
        for i in range(t.shape[0]):
            yield t[i]


def cross_ref(toks, npair=200):
    H = [t[:, 0, 0] for t in toks]                      # 블록 0 원본 bf16 (n,256,D)
    H = np.concatenate([np.asarray(h, np.float32) for h in H])
    n = H.shape[0]
    d = []
    for _ in range(npair):
        i, j = rng.choice(n, 2, replace=False)
        d.append(np.abs(H[i] - H[j]).mean())
    return float(np.median(d)), H


def texture(pix):
    """칸 안 회색 픽셀 표준편차, 블록마다 첫 장 (n, nb, 256)."""
    out = []
    for p in pix:
        for i in range(p.shape[0]):
            x = np.asarray(p[i][0::2], np.float32).mean(-1)                 # (nb,256,256)
            x = x.reshape(x.shape[0], G, 16, G, 16)
            out.append(x.std(axis=(2, 4)).reshape(x.shape[0], -1))
    return np.stack(out)


def nugget(vg):
    """vg (K,) lag 1..K. lag 1–3 직선을 k=0 으로 외삽."""
    k = np.arange(1, 4)
    sl, ic = np.polyfit(k, vg[:3], 1)
    return float(ic), float(sl)


res = {}
for src in a.sources:
    L = load(src)
    if L is None:
        continue
    D, toks, pix, meta = L
    N = D["d0"].shape[0]
    R = {"n_windows": N}
    cross, H0 = cross_ref(toks)
    pixD, d0, dfp = D["pixD"], D["d0"], D["dfp"]
    mov = pixD >= 16
    moving = float(d0[mov].mean()) if mov.any() else float("nan")
    R["ref"] = {"cross": cross, "moving": moving, "frac_moving_cells": float(mov.mean())}
    # ---- curve
    R["curve"] = []
    for lo, hi in BINS:
        m = (pixD >= lo) & (pixD < hi)
        R["curve"].append({"bin": [lo, hi], "frac": float(m.mean()), "d0": float(d0[m].mean()) if m.any() else None,
                           "leak": float(D["leak"][m].mean()) if m.any() else None, "local": float(D["local"][m].mean()) if m.any() else None,
                           "dfp": float(dfp[m].mean()) if m.any() else None})
    # ---- freeze (정지 칸 = 얼린 칸)
    fz = D["frozen"].astype(bool)
    R["freeze"] = {"frac_frozen": float(fz.mean()), "total_fp32": float(dfp[fz].mean()), "leak": float(D["leak"][fz].mean()),
                   "local": float(D["local"][fz].mean()), "reduction_by_freezing": float(1 - D["leak"][fz].mean() / dfp[fz].mean())}
    # 창 전체가 거의 정지 (정지 칸 ≥ 90 %) 인 블록 쌍
    full = fz.mean(-1) >= 0.9
    R["freeze"]["pairs_mostly_static"] = int(full.sum())
    if full.any():
        R["freeze"]["mostly_static_total"] = float(dfp[full][fz[full]].mean())
        R["freeze"]["mostly_static_leak"] = float(D["leak"][full][fz[full]].mean())
    # ---- nuisance
    R["nuisance"] = {k: float(D[k].mean()) for k in ("rep", "bf", "c18", "c28", "c1828", "bright", "shift") if k in D}
    R["nuisance_pix"] = {k: float(D[k].mean()) for k in ("pix_c18", "pix_c28", "pix_shift") if k in D}
    # ---- texture split of static cells
    if pix:
        tex = texture(pix)                                                  # (N, nb, 256)
        tb = tex[:, 1:]                                                     # 블록 b (쌍의 뒤)
        R["texture"] = []
        st = pixD < 1
        for lo, hi in [(0, 2), (2, 8), (8, 999)]:
            m = st & (tb >= lo) & (tb < hi)
            mr = (tex >= lo) & (tex < hi)
            R["texture"].append({"tex": [lo, hi], "frac_of_static": float(m.sum() / max(st.sum(), 1)),
                                 "d0_static": float(d0[m].mean()) if m.any() else None,
                                 "rep": float(D["rep"][mr].mean()) if mr.any() else None,
                                 "bf": float(D["bf"][mr].mean()) if mr.any() else None})
    # ---- 무늬 있는 정지 칸만 (칸 안 픽셀 표준편차 ≥ 2: 검은 띠 · 하늘 같은 민무늬 칸 제외)
    if pix:
        tw = tex.min(1)                                                     # (N,256) 창 내내 최소 무늬
        tx = tb >= 2
        fzt = fz & tx
        R["freeze_textured"] = {"frac": float(fzt.mean()), "total_fp32": float(dfp[fzt].mean()) if fzt.any() else None,
                                "leak": float(D["leak"][fzt].mean()) if fzt.any() else None, "local": float(D["local"][fzt].mean()) if fzt.any() else None}
        st1t = (pixD < 1) & tx
        R["share_textured"] = {"static_frac": float(st1t.mean()), "d0_static": float(d0[st1t].mean()) if st1t.any() else None,
                               "static_over_cross": float(d0[st1t].mean() / cross) if st1t.any() else None,
                               "static_over_moving": float(d0[st1t].mean() / moving) if st1t.any() and mov.any() else None}
        allt = (pixD < 2).all(1) & (tw >= 2)
        if allt.sum() > 20:
            v = np.stack([D["vg0"][:, k][allt] for k in range(D["vg0"].shape[1])]).mean(1)
            vs = np.stack([D["vg0_sq"][:, k][allt] for k in range(D["vg0_sq"].shape[1])]).mean(1)
            ic, _ = nugget(v); ics, _ = nugget(vs)
            R["variogram_textured"] = {"n": int(allt.sum()), "L1": v.tolist(), "nugget_frac_L1": ic / v[0], "nugget_frac_sq": ics / vs[0]}
    # ---- variogram (창 내내 정지인 칸)
    allst = (pixD < 2).all(1)                                               # (N,256)
    vg = D["vg0"]; vgs = D["vg0_sq"]; vgf = D["vgf"]
    R["variogram"] = {"n_cells_static_whole_window": int(allst.sum())}
    if allst.sum() > 20:
        v = np.stack([vg[:, k][allst] for k in range(vg.shape[1])]).mean(1)
        vs = np.stack([vgs[:, k][allst] for k in range(vgs.shape[1])]).mean(1)
        vf = np.stack([vgf[:, k][allst] for k in range(vgf.shape[1])]).mean(1)
        R["variogram"].update({"static_L1": v.tolist(), "static_sq": vs.tolist(), "static_L1_fp32": vf.tolist()})
        ic, sl = nugget(v); ics, sls = nugget(vs)
        R["variogram"].update({"nugget_L1": ic, "nugget_frac_L1": ic / v[0], "nugget_sq": ics, "nugget_frac_sq": ics / vs[0]})
    va = vg.mean(axis=(0, 2)); vsa = vgs.mean(axis=(0, 2))
    ic, _ = nugget(va); ics, _ = nugget(vsa)
    R["variogram"].update({"all_L1": va.tolist(), "all_sq": vsa.tolist(), "all_nugget_frac_L1": ic / va[0], "all_nugget_frac_sq": ics / vsa[0]})
    # ---- share: 복사 손실 (블록 간 |Δh|, 모든 칸) 중 잡음 몫
    tot = d0.mean()
    st1 = pixD < 1
    R["share"] = {"copy_L1": float(tot),
                  "static_cells_frac": float(st1.mean()),
                  "static_cells_mass": float(d0[st1].sum() / d0.sum()),
                  "rep_over_copy": float(D["rep"].mean() / tot),
                  "bf_over_copy": float(D["bf"].mean() / tot),
                  "static_over_moving": float(d0[st1].mean() / moving) if st1.any() and mov.any() else None,
                  "static_over_cross": float(d0[st1].mean() / cross) if st1.any() else None}
    # ---- pooling + PCA (토큰)
    if toks:
        T0 = next(tok_iter(toks))
        nb = T0.shape[1]
        pools = (1, 2, 4, 16)
        PN = {p: [] for p in pools}; PS = {p: [] for p in pools}; PX = {p: [] for p in pools}
        samp = {"signal": [], "rep": [], "bf": [], "c18": [], "static_d": [], "moving_d": []}
        cap = a.pca_n
        Hs = []
        for wi, t in enumerate(tok_iter(toks)):
            if wi >= 160:
                break
            t = np.asarray(t, np.float32)                                    # (5|4, nb, 256, D)
            h0, hf, n1, n2 = t[0], t[1], t[2], t[3]
            Hs.append(h0[0])
            for p in pools:
                def pool(x):
                    s = x.shape
                    return x.reshape(s[0], G // p, p, G // p, p, s[-1]).mean(axis=(2, 4))
                PN[p].append(np.abs(pool(n1) - pool(n2)).mean())
                # 정지 칸만 이루는 p×p 영역의 블록 간 변화
                st_ = (pixD[wi] < 1).reshape(nb - 1, G // p, p, G // p, p).all(axis=(2, 4))
                dd = np.abs(pool(h0[1:]) - pool(h0[:-1])).mean(-1)
                if st_.any():
                    PX[p].append(dd[st_].mean())
            # PCA 표본
            ci = rng.choice(256, 16, replace=False)
            samp["signal"].append(h0[:, ci].reshape(-1, h0.shape[-1])); samp["rep"].append((n1 - n2)[:, ci].reshape(-1, h0.shape[-1]))
            samp["bf"].append((h0 - hf)[:, ci].reshape(-1, h0.shape[-1]))
            if t.shape[0] > 4:
                samp["c18"].append((t[4] - h0)[:, ci].reshape(-1, h0.shape[-1]))
            dh = h0[1:] - h0[:-1]
            stc = np.argwhere(pixD[wi] < 1); mvc = np.argwhere(pixD[wi] >= 16)
            if len(stc):
                s_ = stc[rng.choice(len(stc), min(48, len(stc)), replace=False)]; samp["static_d"].append(dh[s_[:, 0], s_[:, 1]])
            if len(mvc):
                s_ = mvc[rng.choice(len(mvc), min(48, len(mvc)), replace=False)]; samp["moving_d"].append(dh[s_[:, 0], s_[:, 1]])
        Hs = np.stack(Hs)
        for p in pools:
            def pool1(x):
                s = x.shape
                return x.reshape(s[0], G // p, p, G // p, p, s[-1]).mean(axis=(2, 4))
            Hp = pool1(Hs)
            cr = [np.abs(Hp[i] - Hp[j]).mean() for i, j in (rng.choice(len(Hp), 2, replace=False) for _ in range(200))]
            PS[p] = float(np.median(cr))
        R["pool"] = {str(p): {"rep": float(np.mean(PN[p])), "cross": PS[p], "rep_over_cross": float(np.mean(PN[p]) / PS[p]),
                              "static_block_change": float(np.mean(PX[p])) if PX[p] else None,
                              "static_over_cross": float(np.mean(PX[p]) / PS[p]) if PX[p] else None} for p in pools}
        # PCA
        S = {k: np.concatenate(v) for k, v in samp.items() if v}
        X = S["signal"] - S["signal"].mean(0)
        C = X.T @ X / len(X)
        lam, U = np.linalg.eigh(C); lam, U = lam[::-1], U[:, ::-1]
        ks = [10, 32, 100, 320, 640]
        R["pca"] = {"signal_PR": float(lam.sum() ** 2 / (lam ** 2).sum()), "n_samples": {k: int(len(v)) for k, v in S.items()}}
        for name, Y in S.items():
            Yc = Y if name != "signal" else X
            Cy = Yc.T @ Yc / len(Yc)
            ev = np.linalg.eigvalsh(Cy)
            proj = np.einsum("di,de,ei->i", U, Cy, U)                    # 신호 PC 마다 그 성분의 분산
            frac = {str(k): float(proj[:k].sum() / np.trace(Cy)) for k in ks}
            # 신호 PC 구간별 (잡음 분산 / 신호 분산)
            bands = [(0, 10), (10, 100), (100, 320), (320, 1280)]
            ratio = {f"{lo}-{hi}": float(proj[lo:hi].sum() / lam[lo:hi].sum()) for lo, hi in bands}
            R["pca"][name] = {"trace": float(np.trace(Cy)), "PR": float(ev.sum() ** 2 / (ev ** 2).sum()), "frac_in_top_signal_PCs": frac,
                              "var_ratio_to_signal_by_band": ratio}
        # 잡음 (±1/255 반복) 과 움직임 변화 (정지 아님) 의 방향 겹침
        def cov(Y):
            return Y.T @ Y / len(Y)
        def overlap(A, B):
            return float(np.trace(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B)))
        if "moving_d" in S and "static_d" in S:
            Cn, Cm, Cs, Cst = cov(S["rep"]), cov(S["moving_d"]), C, cov(S["static_d"])
            R["pca"]["overlap"] = {"rep_vs_moving": overlap(Cn, Cm), "rep_vs_signal": overlap(Cn, Cs), "moving_vs_signal": overlap(Cm, Cs),
                                   "static_d_vs_rep": overlap(Cst, Cn), "static_d_vs_moving": overlap(Cst, Cm),
                                   "isotropic_vs_signal": float(np.trace(Cs) / (np.sqrt(1280) * np.linalg.norm(Cs)))}
    res[src] = R
    print(f"\n==================== {src}  (창 {N})")
    print(f"  척도: 다른 창 같은 칸 {cross:.3f} · 크게 바뀐 칸 (≥16 단계) 블록 간 변화 {moving:.3f} (칸 {mov.mean():.1%})")
    print("  픽셀 변화 구간별 블록 간 latent 변화 (bf16 정답) · [fp32 번짐 / 미세]:")
    for c in R["curve"]:
        if c["d0"] is not None:
            print(f"    |Δpix| {c['bin'][0]:>4}–{c['bin'][1]:<4} 칸 {c['frac']:5.1%}  d0 {c['d0']:.3f} ({c['d0'] / cross:5.1%} of cross)  번짐 {c['leak']:.3f} · 미세 {c['local']:.3f}")
    f = R["freeze"]
    print(f"  얼리기 (정지 칸 {f['frac_frozen']:.1%}): fp32 변화 {f['total_fp32']:.3f} → 정지 칸 픽셀을 얼리면 {f['leak']:.3f} (−{f['reduction_by_freezing']:.0%}); 미세만 {f['local']:.3f}"
          + (f" | 거의 정지 창 쌍 {f['pairs_mostly_static']}: {f.get('mostly_static_total', float('nan')):.3f} → {f.get('mostly_static_leak', float('nan')):.3f}" if f["pairs_mostly_static"] else ""))
    print("  같은 내용의 다른 표현 (|Δh|, cross 대비):", " · ".join(f"{k} {v:.3f} ({v / cross:.0%})" for k, v in R["nuisance"].items()))
    print("     그 픽셀 크기:", R["nuisance_pix"])
    vgr = R["variogram"]
    if "static_L1" in vgr:
        print(f"  변이도 (창 내내 정지 칸 {vgr['n_cells_static_whole_window']}): L1 lag1..7 " + " ".join(f"{x:.3f}" for x in vgr["static_L1"])
              + f" → nugget {vgr['nugget_L1']:.3f} ({vgr['nugget_frac_L1']:.0%} of lag1) · 제곱 nugget {vgr['nugget_frac_sq']:.0%}")
    print(f"  변이도 (모든 칸): L1 " + " ".join(f"{x:.3f}" for x in vgr["all_L1"]) + f" → nugget {vgr['all_nugget_frac_L1']:.0%} · 제곱 {vgr['all_nugget_frac_sq']:.0%}")
    s = R["share"]
    print(f"  복사 손실 {s['copy_L1']:.3f} | 정지 칸 (|Δpix|<1) {s['static_cells_frac']:.0%} 가 손실 질량 {s['static_cells_mass']:.0%} | "
          f"±1/255 반복 / 복사 {s['rep_over_copy']:.0%} · bf16 / 복사 {s['bf_over_copy']:.0%} | 정지 칸 변화 / 움직인 칸 변화 {s['static_over_moving'] or float('nan'):.0%}")
    if "freeze_textured" in R:
        ft, sh = R["freeze_textured"], R["share_textured"]
        print(f"  [무늬 있는 정지 칸만] 얼리기: {ft['total_fp32'] or float('nan'):.3f} → {ft['leak'] or float('nan'):.3f} (미세만 {ft['local'] or float('nan'):.3f}) | "
              f"|Δpix|<1 칸 {sh['static_frac']:.0%}: d0 {sh['d0_static'] or float('nan'):.3f} = cross 의 {sh['static_over_cross'] or float('nan'):.0%} · 움직인 칸의 {sh['static_over_moving'] or float('nan'):.0%}"
              + (f" | 변이도 ({R['variogram_textured']['n']} 칸) " + " ".join(f"{x:.3f}" for x in R['variogram_textured']['L1']) + f" → nugget {R['variogram_textured']['nugget_frac_L1']:.0%} (제곱 {R['variogram_textured']['nugget_frac_sq']:.0%})" if "variogram_textured" in R else ""))
    if "texture" in R:
        print("  정지 칸 무늬별:", " | ".join(f"std {t['tex']} {t['frac_of_static']:.0%}: d0 {t['d0_static'] if t['d0_static'] is not None else float('nan'):.3f} rep {t['rep'] if t['rep'] is not None else float('nan'):.3f}" for t in R["texture"]))
    if "pool" in R:
        print("  풀링 p×p: " + " | ".join(f"{p}: 반복잡음/장면차 {v['rep_over_cross']:.0%} · 정지영역 블록변화/장면차 {(v['static_over_cross'] or float('nan')):.0%}" for p, v in R["pool"].items()))
    if "pca" in R:
        P = R["pca"]
        print(f"  PCA: 신호 PR {P['signal_PR']:.1f}")
        for name in ("signal", "rep", "bf", "c18", "static_d", "moving_d"):
            if name in P:
                q = P[name]
                print(f"    {name:9s} PR {q['PR']:6.1f} | 신호 상위 k PC 에 든 몫 " + " ".join(f"{k}:{v:.2f}" for k, v in q["frac_in_top_signal_PCs"].items())
                      + " | 분산비(÷신호) " + " ".join(f"{k}:{v:.2f}" for k, v in q["var_ratio_to_signal_by_band"].items()))
        if "overlap" in P:
            print("    겹침 (공분산 cos):", " · ".join(f"{k} {v:.2f}" for k, v in P["overlap"].items()))
out = a.out or os.path.join(os.path.dirname(__file__), f"../results/enc_noise_{a.enc}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1, default=float)
print("\n→", out)
