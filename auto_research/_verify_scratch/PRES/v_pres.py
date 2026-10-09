#!/usr/bin/env python3
"""VERIFY_PRESENCE — AR_PREDICTOR §2-6 검정 결과 (presence-without-placement) 독립 재계산.
원자료 auto_research/_stage/results_ar/v11_presence/{M,Mc,Dh}.npy (672, 5, 8, 256) fp16, meta.json.
M = |p−h_imp| − |p−h_pos| (+ = 가능 쪽), Mc = 복사판, Dh = |h_pos − h_imp|. preds [R,F,B,AR,CX]; R/F/B 표준 공간, AR/CX 블록별 공간.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); I = ROOT / "auto_research/_stage/results_ar/v11_presence"
OUTJ = ROOT / "auto_research/exp_results/verify/verify_presence.json"; OUTD = Path(__file__).parent / "out"; OUTD.mkdir(exist_ok=True)
meta = json.load(open(I / "meta.json")); P = meta["preds"]; pairs = meta["pairs"]; n = len(pairs)
M = np.load(I / "M.npy").astype(np.float32); Mc = np.load(I / "Mc.npy").astype(np.float32); Dh = np.load(I / "Dh.npy").astype(np.float32)
assert np.isfinite(M).all() and np.isfinite(Dh).all(), "NaN 잔존"
KIND = np.array([p["kind"] for p in pairs]); COND = np.array([p["condition"] for p in pairs]); SK = np.array([p["sym_k"] for p in pairs]); BLK = np.array([p["block"] for p in pairs])
OBJ = KIND == "obj"; EMP = KIND == "empty"
rng = np.random.default_rng(12345); B = 2000
G = 16


def boot_mean(v, blk):
    """블록 부트스트랩 (B=2000) 평균 CI. v (m,), blk (m,)."""
    ub = np.unique(blk); ix = {b: np.where(blk == b)[0] for b in ub}
    # 블록당 합과 개수로 벡터화
    s = np.array([v[ix[b]].sum() for b in ub]); c = np.array([len(ix[b]) for b in ub])
    R = rng.integers(0, len(ub), (B, len(ub))); bs = s[R].sum(1) / c[R].sum(1)
    return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def top_mask(dh, q, mode):
    """dh (m, 8, 256) → bool (m, 8, 256). mode 'slot' = 슬롯별 상위 q, 'pair' = 쌍 안 2048 칸 통합 상위 q."""
    if mode == "slot":
        k = max(1, int(round(256 * q))); idx = np.argsort(-dh, axis=-1)[..., :k]
        mk = np.zeros_like(dh, bool); np.put_along_axis(mk, idx, True, axis=-1); return mk
    k = max(1, int(round(2048 * q))); flat = dh.reshape(len(dh), -1); idx = np.argsort(-flat, axis=-1)[:, :k]
    mk = np.zeros_like(flat, bool); np.put_along_axis(mk, idx, True, axis=-1); return mk.reshape(dh.shape)


def stats(m, dh, sel, q=0.05, mode="slot"):
    """m, dh (n,8,256) 한 predictor; sel (n,) → dict (share, margin_top, margin_rest, acc)."""
    mm, dd, blk = m[sel], dh[sel], BLK[sel]
    tk = top_mask(dd, q, mode)
    mp_ = np.maximum(mm, 0); share = (mp_ * tk).sum((1, 2)) / (mp_.sum((1, 2)) + 1e-9)
    mtop = (mm * tk).sum((1, 2)) / tk.sum((1, 2)); mrest = (mm * ~tk).sum((1, 2)) / (~tk).sum((1, 2))
    acc = (mm.mean((1, 2)) > 0).astype(float)
    return dict(n=int(sel.sum()), acc=boot_mean(acc, blk), share=boot_mean(share, blk), margin_top=boot_mean(mtop, blk), margin_rest=boot_mean(mrest, blk),
                frac_top_neg=round(float((mtop < 0).mean()), 3), frac_top_neg_when_correct=round(float((mtop[acc > 0] < 0).mean()), 3) if acc.sum() else None)


res = dict(n=n, preds=P, B=B)
# ---------- 0. 문서 표 재현 (obj 쌍, 조건별; 복사 정확도 포함)
res["doc_table"] = {}
for pi, pk in enumerate(P):
    for cond in sorted(set(COND)) + ["all"]:
        sel = OBJ & ((COND == cond) if cond != "all" else True)
        d = stats(M[:, pi], Dh[:, pi], sel); d["acc_copy"] = boot_mean((Mc[:, pi][sel].mean((1, 2)) > 0).astype(float), BLK[sel])
        res["doc_table"][f"{pk}|obj|{cond}"] = d
    sel = EMP; d = stats(M[:, pi], Dh[:, pi], sel); d["acc_copy"] = boot_mean((Mc[:, pi][sel].mean((1, 2)) > 0).astype(float), BLK[sel]); res["doc_table"][f"{pk}|empty|all"] = d

# ---------- 1. 재등장 칸 정의 강건성 (AR, obj 쌍): q ∈ {1,2,5,10 %} × {slot, pair} × Dh 공간 {자기, 릴리즈}
res["robust_topcells"] = {}
for pk in ("AR", "R", "B", "CX"):
    pi = P.index(pk)
    for dh_name, dh in (("own", Dh[:, pi]), ("release_space", Dh[:, 0])):
        for q in (0.01, 0.02, 0.05, 0.10):
            for mode in ("slot", "pair"):
                for cond in ("moving_occlusion", "moving_occlusion_flat", "static_occlusion", "all"):
                    sel = OBJ & ((COND == cond) if cond != "all" else True)
                    d = stats(M[:, pi], dh, sel, q, mode)
                    res["robust_topcells"][f"{pk}|{dh_name}|q{q}|{mode}|{cond}"] = dict(share=d["share"], margin_top=d["margin_top"], margin_rest=d["margin_rest"], frac_top_neg=d["frac_top_neg"])

# ---------- 2. 공간 지도 (AR obj 쌍): 슬롯 평균 margin 16×16, 국소화
res["spatial"] = {}
for pk in ("AR", "R", "B", "CX"):
    pi = P.index(pk)
    for kind, sel in (("obj", OBJ), ("empty", EMP)):
        mm = M[:, pi][sel]; dd = Dh[:, pi][sel]
        mmap = mm.mean((0, 1)); dmap = dd.mean((0, 1))                        # (256,)
        pos = np.maximum(mmap, 0); top10 = np.sort(pos)[::-1][:10].sum() / (pos.sum() + 1e-9)
        srt = np.sort(pos); gini = float((2 * np.arange(1, 257) - 257).dot(srt) / (256 * srt.sum() + 1e-9))
        # 나머지 (Dh 상위 5 % 제외) margin 의 공간 분포: 쌍마다 top 을 빼고 평균
        tk = top_mask(dd, 0.05, "slot"); rest = np.where(tk, np.nan, mm); rmap = np.nanmean(rest, (0, 1))
        rpos = np.maximum(rmap, 0); rtop10 = np.sort(rpos)[::-1][:10].sum() / (rpos.sum() + 1e-9)
        top_cells = np.argsort(-mmap)[:10]; dh_top = np.argsort(-dmap)[:10]
        corr = float(np.corrcoef(mmap, dmap)[0, 1])
        # 가장자리 vs 안쪽, 상/하 반
        yy, xx = np.arange(256) // G, np.arange(256) % G; border = (yy == 0) | (yy == 15) | (xx == 0) | (xx == 15)
        res["spatial"][f"{pk}|{kind}"] = dict(top10_share_of_pos_mass=round(float(top10), 3), gini_pos=round(gini, 3), rest_top10_share=round(float(rtop10), 3),
                                              corr_margin_vs_Dh_map=round(corr, 3), top_cells_yx=[[int(c // G), int(c % G)] for c in top_cells], Dh_top_cells_yx=[[int(c // G), int(c % G)] for c in dh_top],
                                              margin_border=round(float(mmap[border].mean()), 4), margin_interior=round(float(mmap[~border].mean()), 4),
                                              margin_rows_mean=[round(float(mmap.reshape(G, G)[r].mean()), 4) for r in range(G)],
                                              margin_map=[round(float(v), 4) for v in mmap], Dh_map=[round(float(v), 4) for v in dmap])
        np.save(OUTD / f"map_{pk}_{kind}.npy", np.stack([mmap.reshape(G, G), dmap.reshape(G, G), rmap.reshape(G, G)]))

# ---------- 3. 슬롯 의존 (obj vs empty, sym_k 별): 슬롯별 전체 margin · 재등장 칸 margin · 정답 방향
res["slot"] = {}
for pk in P:
    pi = P.index(pk)
    for kind, sel0 in (("obj", OBJ), ("empty", EMP)):
        for sk in ("all", 1, 2, 3, 4):
            sel = sel0 & ((SK == sk) if sk != "all" else True)
            mm = M[:, pi][sel]; tk = top_mask(Dh[:, pi][sel], 0.05, "slot")
            per_slot = mm.mean((0, 2)); top_slot = (mm * tk).sum((0, 2)) / tk.sum((0, 2)); rest_slot = (mm * ~tk).sum((0, 2)) / (~tk).sum((0, 2))
            acc_slot = (mm.mean(2) > 0).mean(0)                                            # 슬롯 하나만으로 채점했을 때 정답률
            res["slot"][f"{pk}|{kind}|k{sk}"] = dict(n=int(sel.sum()), margin=[round(float(v), 4) for v in per_slot], margin_top=[round(float(v), 4) for v in top_slot],
                                                   margin_rest=[round(float(v), 4) for v in rest_slot], acc_by_slot=[round(float(v), 3) for v in acc_slot])

# ---------- 4. M vs Mc (복사) 상관 · M vs Dh 상관 (쌍별 Pearson, 2048 칸)
def pcorr(a, b):
    a = a.reshape(len(a), -1) - a.reshape(len(a), -1).mean(1, keepdims=True); b = b.reshape(len(b), -1) - b.reshape(len(b), -1).mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-9)
res["corr"] = {}
for pk in P:
    pi = P.index(pk)
    for kind, sel in (("obj", OBJ), ("empty", EMP)):
        r_mc = pcorr(M[:, pi][sel], Mc[:, pi][sel]); r_dh = pcorr(M[:, pi][sel], Dh[:, pi][sel]); r_mc_dh = pcorr(Mc[:, pi][sel], Dh[:, pi][sel])
        # 잔차: M 에서 Mc 를 선형으로 뺀 뒤 재등장 칸 margin 이 남나
        res["corr"][f"{pk}|{kind}"] = dict(M_vs_Mc=boot_mean(r_mc, BLK[sel]), M_vs_Dh=boot_mean(r_dh, BLK[sel]), Mc_vs_Dh=boot_mean(r_mc_dh, BLK[sel]))

# ---------- 5. sym_k 별 핵심 수치 (AR obj: acc, share, top, rest)
res["by_symk"] = {}
for pk in ("AR", "R", "B", "CX"):
    pi = P.index(pk)
    for sk in (1, 2, 3, 4):
        for kind, sel0 in (("obj", OBJ), ("empty", EMP)):
            sel = sel0 & (SK == sk); d = stats(M[:, pi], Dh[:, pi], sel); res["by_symk"][f"{pk}|{kind}|k{sk}"] = d

# ---------- 6. CX empty 쌍: 편향인가 — obj vs empty 의 margin 지도 상관, 부호
pi = P.index("CX")
mo = M[:, pi][OBJ].mean((0, 1)); me = M[:, pi][EMP].mean((0, 1))
res["cx_bias"] = dict(acc_obj=res["doc_table"]["CX|obj|all"]["acc"], acc_empty=res["doc_table"]["CX|empty|all"]["acc"],
                      mean_margin_obj=round(float(M[:, pi][OBJ].mean()), 4), mean_margin_empty=round(float(M[:, pi][EMP].mean()), 4),
                      corr_maps_obj_vs_empty=round(float(np.corrcoef(mo, me)[0, 1]), 3),
                      note="empty 쌍에서 M<0 = 물체가 나타나는 (불가능) 미래 쪽을 고름")
# AR 도 같은 검사 (empty 쌍에서의 margin)
pi = P.index("AR"); res["ar_empty"] = dict(mean_margin_empty=round(float(M[:, pi][EMP].mean()), 4), mean_margin_obj=round(float(M[:, pi][OBJ].mean()), 4),
                                          corr_maps_obj_vs_empty=round(float(np.corrcoef(M[:, pi][OBJ].mean((0, 1)), M[:, pi][EMP].mean((0, 1)))[0, 1]), 3))
json.dump(res, open(OUTJ, "w"), indent=1, default=float)
print("saved", OUTJ)
