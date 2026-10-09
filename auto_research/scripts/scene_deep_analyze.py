#!/usr/bin/env python3
"""I-scene 심층 분석 (CPU, 저장 배열만) — 2026-09-25 저녁 "분석 먼저 철저히".  python scene_deep_analyze.py <dir> [OUT_TAG]

B. reach 밖의 출력: 합의 토큰 (encoder 판독 = 출처) 에서 d 구간별로 predictor argmax 가 (i) 출처 1 칸 안, (ii) 자기 칸 (복사) 1 칸 안, (iii) 둘 다 아님 중 어디로 가나.
   + cos(p, hc_L[src]) 와 cos(p, hc_L[s]) 의 평균 (흐림이면 둘 다 낮다).
C. 실영상 내용의 reach 밖 비율: 유효 토큰 중 변위 ≥ R50 (predictor 별) 인 비율, 그리고 "움직이는 토큰 (d ≥ 1)" 중 비율. 슬롯별로도.
D. 방향을 아는 토큰: 선택성 r = p_src / p_ring (층 평균) 이 ≥ 2 인 토큰의 비율을 (a) 변위 구간, (b) 슬롯, (c) 국소 flow 일관성
   (3×3 이웃 출처 변위의 표준편차; 카메라 팬처럼 일관되면 작다) 구간, (d) 층 (앞/중/뒤) 으로 나눠 본다.
F. Ariel − release 우위의 분해 (합의 토큰, 같은 토큰 짝): Δhit 를 (1) 출처 attention 이 오른 토큰 vs 안 오른 토큰, (2) 복사 attention 이 내린 토큰 vs 아닌 토큰으로 2×2 분해.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", "")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; recp = meta.get("rec_preds", ["R", "A", "P"]); items = meta["items"]; n = len(items); G = 16
C = np.load(I / "arm_c.npy").astype(np.int64); CT = np.load(I / "arm_ct.npy").astype(np.float32); CC = np.load(I / "arm_cc.npy").astype(np.float32)
E_ = np.load(I / "enc.npy"); AT = np.load(I / "att.npy").astype(np.float32)
if (I / "src.npy").exists():
    SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.nan_to_num(np.load(I / "disp.npy").astype(np.float32), nan=-1)
else:
    def src_of(v, dr):
        src = np.full((8, 256), -1); sx = np.arange(256) % G; sy = np.arange(256) // G
        for i in range(8):
            xp = 16 * sx + 8 + dr * v * 2 * (i + 1); ok = (xp >= 0) & (xp < 256); src[i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int)
        return src
    SRC = np.stack([src_of(it["v"], it["dir"]) for it in items]); Dd = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items])
VAL = (SRC >= 0) & (C[:, 0, 0, 0] >= 0)[:, None, None]
S_IDX = np.broadcast_to(np.arange(256)[None, None], SRC.shape); SL = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape)
cheb = lambda a, b: np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))
HE = (cheb(E_[:, 0].astype(np.int64), SRC) <= 1) & VAL
BINS = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)]
res = {"n": int(VAL[:, 0, 0].sum())}

# ---- B. reach 밖의 출력
res["B_output_beyond"] = {}
for pk in recp:
    k = arms.index(f"{pk}:base"); c = C[:, k]; near_src = cheb(c, SRC) <= 1; near_copy = cheb(c, S_IDX) <= 1
    rows = []
    for lo, hi in BINS:
        m = HE & (Dd >= lo) & (Dd < hi)
        if m.sum() < 100: continue
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), to_src=round(float(near_src[m].mean()), 3), to_copy=round(float((near_copy & ~near_src)[m].mean()), 3),
                         elsewhere=round(float((~near_src & ~near_copy)[m].mean()), 3), cos_src=round(float(np.nanmean(CT[:, k][m])), 3), cos_copy=round(float(np.nanmean(CC[:, k][m])), 3),
                         enc_cos_src=round(float(np.nanmean(E_[:, 1][m])), 3), enc_cos_copy=round(float(np.nanmean(E_[:, 2][m])), 3)))
    res["B_output_beyond"][pk] = rows

# ---- C. reach 밖 비율
res["C_coverage"] = {}
r50 = {}
try:
    rj = json.load(open(OUT / f"scene_reach{TAG}.json")); r50 = {a.split(":")[0]: rj["arms"][a]["R50"] for a in rj["arms"] if a.endswith(":base")}
except Exception:
    pass
mv = VAL & (Dd >= 0)
for pk, R in r50.items():
    if R is None or not np.isfinite(R): continue
    beyond = mv & (Dd >= R)
    res["C_coverage"][pk] = dict(R50=R, frac_tokens_beyond=round(float(beyond.sum() / mv.sum()), 4),
                                frac_moving_beyond=round(float(beyond.sum() / max((mv & (Dd >= 1)).sum(), 1)), 4),
                                disp_weighted_beyond=round(float(np.where(beyond, Dd, 0).sum() / np.where(mv, Dd, 0).sum()), 4),
                                by_slot=[round(float((beyond & (SL == i)).sum() / max((mv & (SL == i)).sum(), 1)), 3) for i in range(8)])
res["C_coverage"]["disp_quantiles_moving"] = [round(float(x), 2) for x in np.nanpercentile(Dd[mv & (Dd >= 1)], [50, 75, 90, 95])] if (mv & (Dd >= 1)).any() else None

# ---- D. 방향을 아는 토큰
def local_consistency():
    # 출처 변위 벡터 (x, y) 의 3×3 이웃 표준편차 (칸). 유효 이웃 ≥ 4 일 때만.
    dx = (SRC % G - S_IDX % G).astype(float); dy = (SRC // G - S_IDX // G).astype(float); dx[~VAL] = np.nan; dy[~VAL] = np.nan
    out = np.full(SRC.shape, np.nan, np.float32)
    X = dx.reshape(n, 8, G, G); Y = dy.reshape(n, 8, G, G)
    acc = np.zeros((n, 8, G, G, 9, 2), np.float32); acc[:] = np.nan; q = 0
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            sl_y = slice(max(0, oy), G + min(0, oy)); sl_x = slice(max(0, ox), G + min(0, ox)); ty = slice(max(0, -oy), G + min(0, -oy)); tx = slice(max(0, -ox), G + min(0, -ox))
            acc[:, :, ty, tx, q, 0] = X[:, :, sl_y, sl_x]; acc[:, :, ty, tx, q, 1] = Y[:, :, sl_y, sl_x]; q += 1
    cnt = np.isfinite(acc[..., 0]).sum(-1); sd = np.sqrt(np.nanvar(acc[..., 0], -1) + np.nanvar(acc[..., 1], -1))
    sd[cnt < 4] = np.nan
    return sd.reshape(n, 8, 256)
LC = local_consistency()
res["D_direction_knowers"] = {}
for pk in recp:
    pi = recp.index(pk); ps, pr = AT[:, pi, 0, 0], AT[:, pi, 0, 1]; sel = np.where(pr > 0, ps / np.maximum(pr, 1e-9), np.nan)
    know = (sel >= 2.0); base = HE & (Dd >= 1) & np.isfinite(sel)
    d = dict(frac_know=round(float(know[base].mean()), 3), median_sel=round(float(np.nanmedian(sel[base])), 2))
    d["by_disp"] = [dict(lo=lo, hi=hi, frac=round(float(know[base & (Dd >= lo) & (Dd < hi)].mean()), 3)) for lo, hi in BINS if (base & (Dd >= lo) & (Dd < hi)).sum() > 100]
    d["by_slot"] = [round(float(know[base & (SL == i)].mean()), 3) if (base & (SL == i)).any() else None for i in range(8)]
    qs = np.nanpercentile(LC[base], [25, 50, 75]) if np.isfinite(LC[base]).any() else None
    if qs is not None:
        lc = LC; d["consistency_quartiles_cells"] = [round(float(x), 2) for x in qs]
        d["by_consistency"] = [round(float(know[base & (lc >= a) & (lc < b)].mean()), 3) for a, b in ((0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 99))]
        d["hit_by_consistency"] = [round(float((cheb(C[:, arms.index(f'{pk}:base')], SRC) <= 1)[base & (lc >= a) & (lc < b)].mean()), 3) for a, b in ((0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 99))]
    d["by_layer_group_median_sel"] = [round(float(np.nanmedian(np.where(AT[:, pi, g, 1] > 0, AT[:, pi, g, 0] / np.maximum(AT[:, pi, g, 1], 1e-9), np.nan)[base])), 2) for g in (1, 2, 3)]
    res["D_direction_knowers"][pk] = d

# ---- F. Ariel − release 분해 (같은 토큰)
if "R" in recp and "A" in recp:
    kR, kA = arms.index("R:base"), arms.index("A:base"); hR, hA = cheb(C[:, kR], SRC) <= 1, cheb(C[:, kA], SRC) <= 1
    dsrc = AT[:, 1, 0, 0] - AT[:, 0, 0, 0]; dcopy = AT[:, 1, 0, 2] - AT[:, 0, 0, 2]
    rows = []
    for lo, hi in BINS:
        m = HE & (Dd >= lo) & (Dd < hi) & np.isfinite(dsrc)
        if m.sum() < 200: continue
        dh = (hA.astype(float) - hR.astype(float))
        up, dn = dsrc > 0, dcopy < 0
        cell = {}
        for nm, mm in (("src_up&copy_down", up & dn), ("src_up&copy_notdown", up & ~dn), ("src_notup&copy_down", ~up & dn), ("neither", ~up & ~dn)):
            mmm = m & mm; cell[nm] = dict(frac=round(float(mmm.sum() / m.sum()), 3), dhit=round(float(dh[mmm].mean()), 3) if mmm.any() else None,
                                          share_of_gain=round(float(dh[mmm].sum() / max(dh[m].sum(), 1e-9)), 3))
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), dhit_total=round(float(dh[m].mean()), 3), cells=cell,
                         corr_dhit_dsrc=round(float(np.corrcoef(dh[m], dsrc[m])[0, 1]), 3), corr_dhit_dcopy=round(float(np.corrcoef(dh[m], dcopy[m])[0, 1]), 3)))
    res["F_ariel_gap_decomp"] = rows

json.dump(res, open(OUT / f"scene_deep{TAG}.json", "w"), indent=1, default=float)
print(f"== {I.name}  n {res['n']}")
for pk, rows in res["B_output_beyond"].items():
    print(f"B {pk}: d→ " + "  ".join(f"[{r['lo']},{r['hi']}) src {r['to_src']:.2f} copy {r['to_copy']:.2f} else {r['elsewhere']:.2f} cos(src/copy) {r['cos_src']:.2f}/{r['cos_copy']:.2f} enc {r['enc_cos_src']:.2f}/{r['enc_cos_copy']:.2f}" for r in rows))
print("C:", json.dumps(res["C_coverage"]))
for pk, d in res["D_direction_knowers"].items():
    print(f"D {pk}: know≥2 {d['frac_know']}, median sel {d['median_sel']}, by_disp {[x['frac'] for x in d['by_disp']]}, by_slot {d['by_slot']}, consistency q {d.get('consistency_quartiles_cells')} know {d.get('by_consistency')} hit {d.get('hit_by_consistency')}, layer med sel {d['by_layer_group_median_sel']}")
for r in res.get("F_ariel_gap_decomp", []):
    print(f"F [{r['lo']},{r['hi']}) Δhit {r['dhit_total']:+.3f} corr(Δhit,Δsrc) {r['corr_dhit_dsrc']:+.2f} corr(Δhit,Δcopy) {r['corr_dhit_dcopy']:+.2f} | " + " ".join(f"{k}: f {v['frac']:.2f} Δ {v['dhit']} share {v['share_of_gain']}" for k, v in r["cells"].items()))
