#!/usr/bin/env python3
"""H7c — v11_full 풀링 선형 probe 의 비가림 → 가림 이식을 영역별 평균 중심화로 다시 잰다 (CPU 전용, vll6 에서 srun6.sh).

무엇
  H7 (B) 와 같은 ridge 분류 (primal, 학습 표준화, λ ∈ {1e1..1e4} 5-fold, block 단위 반반 split × seed 0,1,2 — split·fold 난수까지 동일)
  를 visible 에서 학습하고 early / mid / late 에 시험한다. 시험 특징을 세 방식으로 넣는다:
    raw     x̃ = (x − μ_vis) / σ_vis                       (H7 원판; 원판 수치 재현이 검증)
    center  x̃ = (x − μ_D)   / σ_vis                       μ_D = 시험 영역 D 의 **train-block** clip 평균 (라벨 안 씀)
    zscore  x̃ = (x − μ_D)   / σ_D                         σ_D 도 같은 unlabeled 표본에서
  center 는 "영역 평균 이동" 한 성분만 빼는 최소 적응이다. 남는 격차는 평균 이동으로 설명되지 않는 부분.
  특징 (전 토큰 평균 풀링, H7 추출물):
    p  = 렌즈 L0..L11 의 미래 슬롯 t ∈ {0, 3, 7}, 그리고 L11 의 8 슬롯 평균 (p_all)
    z  = LN(encoder block l) 문맥 마지막 튜블릿 (l ∈ {3..31}), 그리고 block 31 의 문맥 8 튜블릿 평균 (z_all)
    h  = LN(target_encoder) 미래 슬롯 t ∈ {0, 3, 7}, 그리고 미래 8 슬롯 평균 (h_all)
  p 특이 격차 = acc(h) − acc(p), acc(z) − acc(p) 를 seed 평균으로 내고 block 군집 bootstrap 95% CI (B = 2000; 각 seed 의 시험 block 만
  기여, 세 seed 를 같은 재표집 가중치로 평균).
  같은 팔 이식 (CLAUDE.md §5-2 attentive 판과 같은 구도): motion 팔마다 visible → late (raw / center), 그리고 팔 안 late self.
왜
  적대 검증: 원판 §0-3 "평균 풀링 선형 probe 로는 p 특이성이 안 나온다" 는 원 수치에서도 h > p 였고, 평균 중심화 뒤 p < z ≈ h 가 남는다는 지적.
  원인으로 적은 "가림막 유무" 도 검정되지 않았다 (early/mid 에도 가림막이 있는데 이식은 훨씬 높다).
출력  auto_research/exp_results/h7/h7c_centered_transfer.json
"""
import collections, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h7")
OUT = ROOT / "auto_research/exp_results/h7"; OUT.mkdir(parents=True, exist_ok=True)
REFJ = json.load(open(OUT / "h7_v11.json"))["B_probe"]
NB = 2000
t0 = time.time()
meta = json.load(open(I / "meta.json")); n = meta["n"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
oi = np.where(M["probe_type"] == "obj")[0]; no = len(oi)
blocks = M["block_id"][oi]; ub = np.unique(blocks)
tim_o, mot_o = tim[oi], M["motion"][oi]

# ---------------- 특징 적재 (obj clip 만)
lens = np.load(I / "lens.npy", mmap_mode="r"); enc = np.load(I / "enc.npy", mmap_mode="r"); hh = np.load(I / "h.npy", mmap_mode="r")
LS = np.zeros((no, 12, 3, 1280), np.float16); P_ALL = np.zeros((no, 1280), np.float32)
EC = np.zeros((no, 8, 1280), np.float16); Z_ALL = np.zeros((no, 1280), np.float32)
CH = 1024
for i0 in range(0, no, CH):
    idx = oi[i0:i0 + CH]
    X = np.asarray(lens[idx])                                                 # (ch, 12, 8, 1280)
    LS[i0:i0 + CH] = X[:, :, [0, 3, 7]]; P_ALL[i0:i0 + CH] = X[:, 11].astype(np.float32).mean(1)
    E = np.asarray(enc[idx])                                                  # (ch, 8 층, 8 튜블릿, 1280)
    EC[i0:i0 + CH] = E[:, :, 7]; Z_ALL[i0:i0 + CH] = E[:, 7].astype(np.float32).mean(1)
    if (i0 // CH) % 5 == 0:
        print(f"   적재 {i0}/{no}  {time.time()-t0:.0f}s", flush=True)
HH = np.asarray(hh[oi], np.float32)                                            # (no, 16, 1280)
H3 = HH[:, [8, 11, 15]]; H_ALL = HH[:, 8:].mean(1); del HH
print(f"적재 끝 {time.time()-t0:.0f}s", flush=True)

feats = {}
for l in range(12):
    for j, t in enumerate((0, 3, 7)):
        feats[f"lens{l}_t{t}"] = (lambda l=l, j=j: LS[:, l, j].astype(np.float32))
feats["lens11_all"] = lambda: P_ALL
for li, L in enumerate(meta["enc_layers"]):
    feats[f"enc{L}_ctxlast"] = (lambda li=li: EC[:, li].astype(np.float32))
feats["enc31_all"] = lambda: Z_ALL
for j, t in enumerate((0, 3, 7)):
    feats[f"h_t{t}"] = (lambda j=j: H3[:, j])
feats["h_all"] = lambda: H_ALL
KEY = ["lens11_t0", "lens11_t3", "lens11_t7", "lens11_all", "enc31_ctxlast", "enc31_all", "enc15_ctxlast", "h_t0", "h_t3", "h_t7", "h_all",
       "lens0_t7", "lens6_t7"]


def fit(Xtr, ytr, ncls):
    """H7 원판과 같은 ridge: 학습 표준화, λ 5-fold (rng 0), 최소 오류 λ (동률이면 작은 λ)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = (Xtr - mu) / sd
    Y = np.eye(ncls)[ytr]; f = np.random.default_rng(0).permutation(len(Z)) % 5; best = None
    errs = collections.defaultdict(float)
    for k_ in range(5):
        tr, va = f != k_, f == k_
        G = Z[tr].T.astype(np.float64) @ Z[tr].astype(np.float64); b = Z[tr].T.astype(np.float64) @ Y[tr]
        for lam in (1e1, 1e2, 1e3, 1e4):
            Wk = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
            errs[lam] += ((Z[va] @ Wk).argmax(1) != ytr[va]).mean()
    for lam in (1e1, 1e2, 1e3, 1e4):
        if best is None or errs[lam] < best[0]:
            best = (errs[lam], lam)
    G = Z.T.astype(np.float64) @ Z.astype(np.float64)
    Wf = np.linalg.solve(G + best[1] * np.eye(G.shape[0]), Z.T.astype(np.float64) @ Y)
    return mu, sd, Wf


def predict(X, mu, sd, Wf):
    return (((X - mu) / sd) @ Wf).argmax(1)


splits = []
for s in range(3):
    trb = set(np.random.default_rng(s).permutation(ub)[: len(ub) // 2]); splits.append(np.array([b in trb for b in blocks]))
DOMS = ("early", "mid", "late"); ARMS = ("static", "moving_flat", "moving")
R = {}; CORR = {}                                                              # CORR[(tgt, feat, dom, var)] = [ (seed) bool 배열 (no,) 시험 위치만 의미 ]
for tgt in ("shape_pre", "color_pre"):
    cls = sorted(set(M[tgt][oi])); y = np.array([cls.index(c) for c in M[tgt][oi]])
    for fn, fx in feats.items():
        X = fx(); rs = collections.defaultdict(list)
        for s, tr in enumerate(splits):
            mv = tim_o == "visible"
            mu, sd, Wf = fit(X[tr & mv], y[tr & mv], len(cls))
            for D in DOMS:
                mD = tim_o == D; te = ~tr & mD; unl = tr & mD
                muD, sdD = X[unl].mean(0), X[unl].std(0) + 1e-6
                preds = {"raw": predict(X[te], mu, sd, Wf),
                         "center": predict(X[te] - muD + mu, mu, sd, Wf),
                         "zscore": (((X[te] - muD) / sdD) @ Wf).argmax(1)}
                for var, pr in preds.items():
                    c = pr == y[te]; rs[f"xfer|{D}|{var}"].append(c.mean())
                    if fn in KEY:
                        full = np.zeros(no, bool); full[te] = c; CORR.setdefault((tgt, fn, D, var), []).append((te, full))
            # 같은 팔 이식 · 팔 안 late self (KEY 만)
            if fn in KEY:
                for arm in ARMS:
                    ma = mot_o == arm
                    mva = ma & (tim_o == "visible"); mla = ma & (tim_o == "late")
                    mu2, sd2, W2 = fit(X[tr & mva], y[tr & mva], len(cls))
                    te = ~tr & mla; unl = tr & mla; muD = X[unl].mean(0)
                    for var, pr in (("raw", predict(X[te], mu2, sd2, W2)), ("center", predict(X[te] - muD + mu2, mu2, sd2, W2))):
                        c = pr == y[te]; rs[f"arm_xfer|{arm}|late|{var}"].append(c.mean())
                        full = np.zeros(no, bool); full[te] = c; CORR.setdefault((tgt, fn, f"arm:{arm}", var), []).append((te, full))
                    mu3, sd3, W3 = fit(X[tr & mla], y[tr & mla], len(cls))
                    rs[f"arm_self|{arm}|late"].append((predict(X[te], mu3, sd3, W3) == y[te]).mean())
        R[f"{tgt}|{fn}"] = {k: [round(100 * float(np.mean(v)), 2), round(100 * float(np.min(v)), 2), round(100 * float(np.max(v)), 2)] for k, v in rs.items()}
        ref = REFJ.get(f"{tgt}|{fn}", {})
        chk = " ".join(f"{D}: raw {R[f'{tgt}|{fn}'][f'xfer|{D}|raw'][0]:.1f} (원판 {ref.get(f'xfer|{D}', float('nan'))})" for D in DOMS)
        print(f"{tgt:9s} {fn:14s} {chk} | late center {R[f'{tgt}|{fn}']['xfer|late|center'][0]:.1f}  {time.time()-t0:.0f}s", flush=True)

# ---------------- p 특이 격차: seed 평균 + block 군집 bootstrap
bid_o = np.unique(blocks, return_inverse=True)[1]; nbk = bid_o.max() + 1
Wb = np.stack([np.bincount(np.random.default_rng(1000 + b).integers(0, nbk, nbk), minlength=nbk) for b in range(NB)]).astype(np.float64)


def gap(tgt, fa, fb, dom, var):
    """acc(fa) − acc(fb) (%p), seed 평균, block bootstrap CI."""
    A_, B_ = CORR[(tgt, fa, dom, var)], CORR[(tgt, fb, dom, var)]
    pt, bs = [], np.zeros(NB)
    for (te, ca), (te2, cb) in zip(A_, B_):
        assert (te == te2).all()
        d = ca.astype(float) - cb.astype(float)
        s = np.bincount(bid_o[te], weights=d[te], minlength=nbk); c = np.bincount(bid_o[te], minlength=nbk).astype(float)
        pt.append(d[te].mean()); bs += (Wb @ s) / np.maximum(Wb @ c, 1)
    bs /= len(A_)
    return [round(100 * float(np.mean(pt)), 2), round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]


GAPS = {}
COMPS = [("h_t7", "lens11_t7"), ("h_t3", "lens11_t3"), ("h_t0", "lens11_t0"), ("h_all", "lens11_all"),
         ("enc31_ctxlast", "lens11_t7"), ("enc31_all", "lens11_all"), ("enc15_ctxlast", "lens11_t7"), ("h_all", "enc31_all")]
for tgt in ("shape_pre", "color_pre"):
    for fa, fb in COMPS:
        for dom in list(DOMS) + [f"arm:{a}" for a in ARMS]:
            for var in ("raw", "center") + (("zscore",) if not dom.startswith("arm") else ()):
                GAPS[f"{tgt}|{fa}-{fb}|{dom}|{var}"] = gap(tgt, fa, fb, dom, var)
print(f"격차 끝 {time.time()-t0:.0f}s", flush=True)
json.dump({"n_obj": int(no), "seeds": 3, "bootstrap_B": NB, "results": R, "gaps": GAPS,
           "note": "results 값 = [seed 평균, seed 최소, seed 최대] %. center 의 μ_D 는 시험 영역의 train-block clip (라벨 미사용). "
                   "gaps = [seed 평균 차 %p, 95% CI 하, 상] (block 군집 bootstrap)."},
          open(OUT / "h7c_centered_transfer.json", "w"), indent=1)
print(f"끝 {time.time()-t0:.0f}s", flush=True)
