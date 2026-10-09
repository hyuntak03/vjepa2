#!/usr/bin/env python3
"""H7 분석 — v11_full: (A) predictor 층별 렌즈로 채점 · (B) 층별 linear probing (모양·색). CPU.

(A) 층 l 렌즈 surprise  S_l(clip, T) = mean_{t∈T} mean|p^(l)_t − LN(h)_t|   (T = 슬롯 집합)
    matched pair = (block_id, pair_id) 의 가능·불가능 (문맥 픽셀 동일) → 정답 = S_l(imp) > S_l(pos).
    검증: l = 11, T = 전부 → per_video surprise 가 표준 채점 (surprise_c16t32__v11_full_vith/per_block.json) 과 같아야 하고 정확도 ≈ 76.00.
(B) probe_type == obj (가능·물체 있음) clip.  표적 shape_pre (7), color_pre (8).  ridge 분류 (primal, λ 5-fold), block 단위 반반 split × 3 seed.
    특징: 렌즈 층 l 의 미래 슬롯 t 풀링 (t = 0, 3, 7) · encoder 층 {3..31} 의 문맥 마지막 튜블릿 풀링 · h 미래 슬롯 (천장).
    프로토콜: self (같은 timing 안) · 이식 visible → {early, mid, late}.
"""
import collections, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h7")
OUT = ROOT / "auto_research/exp_results/h7"; OUT.mkdir(parents=True, exist_ok=True)
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
meta = json.load(open(I / "meta.json")); n = meta["n"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
l1 = np.load(I / "l1.npy")                                                  # (n, 12, 8)
out = {}
# ---------------- (A)
ref = json.load(open(REF))["per_video_surprise"]
s11 = l1[:, 11].mean(1); r = np.array([ref[v] for v in M["video_id"]])
out["validation"] = {"max_abs": float(np.abs(s11 - r).max()), "median_abs": float(np.median(np.abs(s11 - r)))}
print("검증 L11 vs 표준 per_video: max", out["validation"]["max_abs"], "median", out["validation"]["median_abs"])
pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
P = np.array([(d["pos"], d["imp"]) for d in pairs.values() if len(d) == 2]); ps, im = P[:, 0], P[:, 1]
SLOTS = {"all": list(range(8)), "t0-3": [0, 1, 2, 3], "t4-7": [4, 5, 6, 7], "t0": [0], "t7": [7]}
groups = {"ALL": np.ones(len(P), bool)}
for t in ("visible", "early", "mid", "late"):
    groups[t] = tim[ps] == t
    for mo in ("static", "moving_flat", "moving"):
        groups[f"{t}|{mo}"] = (tim[ps] == t) & (M["motion"][ps] == mo)
    for vt in ("vanish", "shape", "color"):
        groups[f"{t}|{vt}"] = (tim[ps] == t) & (M["violation_type"][ps] == vt)
A = {}
for sn, sl in SLOTS.items():
    S = l1[:, :, sl].mean(2)                                                # (n, 12)
    ok = (S[im] > S[ps]).astype(float) + 0.5 * (S[im] == S[ps])
    for g, m in groups.items():
        A[f"{sn}|{g}"] = (100 * ok[m].mean(0)).round(2).tolist()
out["A_layer_scoring"] = A
print("\n(A) 층별 렌즈 채점 정확도 (%) — 행: 조건, 열: L0 L2 L4 L6 L8 L9 L10 L11")
for sn in ("all", "t0-3", "t4-7"):
    for g in ("ALL", "visible", "early", "mid", "late", "late|moving", "late|vanish", "visible|moving", "visible|shape"):
        v = A[f"{sn}|{g}"]; print(f"  {sn:5s} {g:15s} " + " ".join(f"{v[l]:5.1f}" for l in (0, 2, 4, 6, 8, 9, 10, 11)))
# ---------------- (B)
obj = M["probe_type"] == "obj"
lens = np.load(I / "lens.npy", mmap_mode="r"); enc = np.load(I / "enc.npy", mmap_mode="r"); hh = np.load(I / "h.npy", mmap_mode="r")
oi = np.where(obj)[0]
blocks = M["block_id"][oi]; ub = np.unique(blocks)
def ridge_cls(Xtr, ytr, Xte, ncls):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Y = np.eye(ncls)[ytr]; best = None
    rng = np.random.default_rng(0); f = rng.permutation(len(Xtr)) % 5
    for lam in (1e1, 1e2, 1e3, 1e4):
        err = 0
        for k in range(5):
            tr, va = f != k, f == k
            W = np.linalg.solve(Xtr[tr].T @ Xtr[tr] + lam * np.eye(Xtr.shape[1]), Xtr[tr].T @ Y[tr])
            err += ((Xtr[va] @ W).argmax(1) != ytr[va]).mean()
        if best is None or err < best[0]: best = (err, lam)
    W = np.linalg.solve(Xtr.T @ Xtr + best[1] * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return (Xte @ W).argmax(1)
feats = {}
for l in range(12):
    for t in (0, 3, 7):
        feats[f"lens{l}_t{t}"] = (lambda l=l, t=t: np.asarray(lens[oi, l, t], np.float32))
for li, L in enumerate(meta["enc_layers"]):
    feats[f"enc{L}_ctxlast"] = (lambda li=li: np.asarray(enc[oi, li, 7], np.float32))
for t in (0, 3, 7):
    feats[f"h_t{t}"] = (lambda t=t: np.asarray(hh[oi, 8 + t], np.float32))
B = {}
for tgt in ("shape_pre", "color_pre"):
    cls = sorted(set(M[tgt][oi])); y = np.array([cls.index(c) for c in M[tgt][oi]])
    for fn, fx in feats.items():
        X = fx(); rs = collections.defaultdict(list)
        for sd in range(3):
            rng = np.random.default_rng(sd); trb = set(rng.permutation(ub)[: len(ub) // 2])
            tr = np.array([b in trb for b in blocks])
            for t in ("visible", "early", "mid", "late"):
                m = tim[oi] == t
                pr = ridge_cls(X[tr & m], y[tr & m], X[~tr & m], len(cls)); rs[f"self|{t}"].append((pr == y[~tr & m]).mean())
            mv = tim[oi] == "visible"
            for t in ("early", "mid", "late"):
                m = tim[oi] == t
                pr = ridge_cls(X[tr & mv], y[tr & mv], X[~tr & m], len(cls)); rs[f"xfer|{t}"].append((pr == y[~tr & m]).mean())
        B[f"{tgt}|{fn}"] = {k: round(100 * float(np.mean(v)), 1) for k, v in rs.items()}
        print(f"(B) {tgt:9s} {fn:16s} " + " ".join(f"{k} {v:5.1f}" for k, v in B[f"{tgt}|{fn}"].items()), flush=True)
out["B_probe"] = B
json.dump(out, open(OUT / "h7_v11.json", "w"), indent=1)
