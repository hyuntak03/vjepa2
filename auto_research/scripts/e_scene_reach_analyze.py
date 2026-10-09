#!/usr/bin/env python3
"""I-scene 분석 — 장면 단위 reach 곡선과 개입 효과. python e_scene_reach_analyze.py <dir>   (OUT_TAG 접미사)

단위: (clip, 슬롯 i, 칸 s) 토큰. d = v·(i+1)/8 칸 (팬 속도 v px/프레임; clip·슬롯마다 전 토큰 같음). 유효 = 출처가 창 안.
  hit_X = ‖ĉ_X − src‖∞ ≤ 1.  enc = encoder 천장 (h_{8+i}).  합의 토큰 = hit_enc 인 토큰 (판독이 맞는 자리).
  A_X(d) = P(hit_X | d, 합의)        soft_X(d) = mean[cos(X, hc_L[src]) − cos(X, hc_L[s])]  (출처가 복사보다 더 닮았나; d ≥ 1)
  R50 = A 가 첫 구간 값의 절반으로 떨어지는 d (선형 보간).  ΔA = 같은 토큰의 (팔 − 그 predictor 의 base).
  거리 vs 걸음: base 에서 P(hit) ~ 1 + slot + d (합의 토큰, clip bootstrap) — 속도가 clip 마다 달라 slot 과 d 가 갈린다.
CI = clip bootstrap B = 500.
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
I = Path(sys.argv[1]); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); arms = meta["arms"]; items = meta["items"]; n = len(items)
C = np.load(I / "arm_c.npy").astype(np.int64); CT = np.load(I / "arm_ct.npy").astype(np.float32); CC = np.load(I / "arm_cc.npy").astype(np.float32)
E_ = np.load(I / "enc.npy"); EX = np.load(I / "E.npy") if (I / "E.npy").exists() else np.full((n, 4), np.nan)
done = (C[:, 0, 0, 0] >= 0); print("완료 clip", int(done.sum()), "/", n)
G = 16; rng = np.random.default_rng(0)
BINS = [0, 1, 2, 3, 4, 6, 9, 13]


def src_of(v, dr):
    src = np.full((8, 256), -1); sx = np.arange(256) % G; sy = np.arange(256) // G
    for i in range(8):
        xp = 16 * sx + 8 + dr * v * 2 * (i + 1); ok = (xp >= 0) & (xp < 256)
        src[i, ok] = sy[ok] * G + (xp[ok] // 16).astype(int)
    return src


if (I / "src.npy").exists():                                                    # 자연 영상판 (e_scene_reach_nat): flow · GT 출처
    SRC = np.load(I / "src.npy").astype(np.int64); Dd = np.load(I / "disp.npy").astype(np.float32); Dd = np.nan_to_num(Dd, nan=-1.0)
else:                                                                             # 인공 팬
    SRC = np.stack([src_of(it["v"], it["dir"]) for it in items])
    Dd = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items])
VAL = SRC >= 0
SL = np.broadcast_to(np.arange(8)[None, :, None], SRC.shape)


def cheb_hit(c):
    return (np.maximum(np.abs(c % G - SRC % G), np.abs(c // G - SRC // G)) <= 1) & VAL


HE = cheb_hit(E_[:, 0].astype(np.int64)); AG = HE & done[:, None, None]
H = {a: cheb_hit(C[:, k]) for k, a in enumerate(arms)}
SOFT = {a: CT[:, k] - CC[:, k] for k, a in enumerate(arms)}
SOFTE = E_[:, 1] - E_[:, 2]


def boot_clip(num, den, B=500):
    """num, den: (n,) per clip 합 → 비율과 clip bootstrap CI."""
    ok = den > 0
    if ok.sum() == 0: return [np.nan] * 3
    r = num[ok].sum() / den[ok].sum(); idx = np.where(ok)[0]; bs = []
    for _ in range(B):
        s = rng.choice(idx, len(idx)); bs.append(num[s].sum() / den[s].sum())
    return [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def curve(hit, mask):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = mask & (Dd >= lo) & (Dd < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), A=boot_clip((hit & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float))))
    return out


def r50(cv):
    a = [c["A"][0] for c in cv if c["n"] > 0]; lo = [0.5 * (c["lo"] + c["hi"]) for c in cv if c["n"] > 0]
    if not a or not np.isfinite(a[0]): return np.nan
    half = 0.5 * a[0]
    for j in range(1, len(a)):
        if a[j] < half:
            f = (a[j - 1] - half) / max(a[j - 1] - a[j], 1e-9); return round(float(lo[j - 1] + f * (lo[j] - lo[j - 1])), 2)
    return float("inf")


def soft_curve(sf, mask):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = mask & (Dd >= max(lo, 1)) & (Dd < hi) & np.isfinite(sf)
        out.append(round(float(np.nanmean(np.where(m, sf, np.nan))), 4) if m.any() else np.nan)
    return out


res = dict(n_done=int(done.sum()), arms={}, enc={}, E={})
res["enc"]["A_unconditional"] = curve(HE, VAL & done[:, None, None])
res["enc"]["soft"] = soft_curve(SOFTE, VAL & done[:, None, None])
for a in arms:
    cv = curve(H[a], AG); res["arms"][a] = dict(curve=cv, R50=r50(cv), soft=soft_curve(SOFT[a], AG))
for a in arms:
    pk = a.split(":")[0]; base = f"{pk}:base"
    if a == base: continue
    dd = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = AG & (Dd >= lo) & (Dd < hi)
        num = ((H[a] & m).sum((1, 2)) - (H[base] & m).sum((1, 2))).astype(float); den = m.sum((1, 2)).astype(float)
        ok = den > 0; idx = np.where(ok)[0]
        if not len(idx): dd.append([np.nan] * 3); continue
        r = num[ok].sum() / den[ok].sum(); bs = [num[s].sum() / den[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(500))]
        dd.append([round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
    res["arms"][a]["delta_vs_base"] = dd
# 거리 vs 걸음 (base 셋)
for pk in meta.get("rec_preds", ["R", "A", "P"]):
    a = f"{pk}:base"; m = AG & (Dd >= 0)
    y = H[a][m].astype(float); X = np.stack([np.ones(m.sum()), SL[m], Dd[m]], 1); cl = np.broadcast_to(np.arange(n)[:, None, None], m.shape)[m]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; bs = []
    for _ in range(200):
        s = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0])
    bs = np.array(bs)
    res.setdefault("slot_vs_dist", {})[a] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)]
                                             for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
# attention 관측 (base 셋): 미래 질의 → 진짜 출처 key · 같은 거리 ring · 복사 key 질량 (층 평균 · 세 구간), d 구간별 (합의 토큰 — 자연 영상에서 flow 출처 오류를 거른다)
if (I / "att.npy").exists():
    AT = np.load(I / "att.npy").astype(np.float32)                                  # (n, 3, 4, 3, 8, 256)
    res["attention"] = {}
    for pi, pk in enumerate(meta.get("rec_preds", ["R", "A", "P"])):
        for gi, gname in enumerate(("mean", "L_first", "L_mid", "L_last")):
            rows = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = AG & (Dd >= max(lo, 1)) & (Dd < hi)                                      # 판독 합의 토큰만 (출처가 맞는 자리)
                if not m.any(): rows.append(None); continue
                ps, pr, pc = (AT[:, pi, gi, j][m] for j in range(3))
                rows.append(dict(lo=lo, hi=hi, p_src=round(float(np.nanmean(ps)), 5), p_ring=round(float(np.nanmean(pr)), 5), p_copy=round(float(np.nanmean(pc)), 5),
                                 src_over_ring=round(float(np.nanmean(ps) / max(np.nanmean(pr), 1e-9)), 2),
                                 frac_src_gt_ring=round(float(np.nanmean(ps > pr)), 3)))
            res["attention"][f"{pk}:{gname}"] = rows
ok = np.isfinite(EX[:, 0])
if ok.any():
    res["E"] = dict(n=int(ok.sum()), z_aligned=round(float(EX[ok, 0].mean()), 4), z_null=round(float(EX[ok, 1].mean()), 4),
                    h_aligned=round(float(EX[ok, 2].mean()), 4), h_null=round(float(EX[ok, 3].mean()), 4))
json.dump(res, open(OUT / f"scene_reach{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)

print("E (encoder 등변성):", res["E"])
print("구간 (칸):", " ".join(f"[{lo},{hi})" for lo, hi in zip(BINS[:-1], BINS[1:])))
print(f"{'enc (무조건)':14s}", " ".join(f"{c['A'][0]:.2f}" for c in res["enc"]["A_unconditional"]), "  n", [c["n"] for c in res["enc"]["A_unconditional"]])
for a in arms:
    r = res["arms"][a]
    print(f"{a:14s}", " ".join(f"{c['A'][0]:.2f}" for c in r["curve"]), f"  R50 {r['R50']}", "  soft " + " ".join(f"{x:+.3f}" for x in r["soft"]))
    if "delta_vs_base" in r: print(f"{'':14s} Δ " + " ".join(f"{x[0]:+.2f}[{x[1]:+.2f},{x[2]:+.2f}]" for x in r["delta_vs_base"]))
print("거리 vs 걸음:", json.dumps(res.get("slot_vs_dist")))
for k, rows in res.get("attention", {}).items():
    if not k.endswith(":mean") and not k.startswith("R:"): continue
    print(f"attn {k:10s} " + "  ".join(f"[{r['lo']},{r['hi']}) src {r['p_src']:.4f} ring {r['p_ring']:.4f} copy {r['p_copy']:.4f} s/r {r['src_over_ring']:.1f}" for r in rows if r))
