#!/usr/bin/env python3
"""IntPhysGen v11 — z · p · h 가 **표현으로서** 어떻게 다른가 (visible vs late 가림), clip 당 풀링 벡터의 기하.

입력: cache/v11_pooled_vith/{z,p,h}.npy · meta.npz  (`v11_pooled_features.py` 산출, 가능 변이 · 물체 있는 9,408 clip)
clip 당 풀링 벡터 (D = 1280, 튜블릿 평균들의 평균 = (T·N, D) 토큰 평균):
  z_all  z 문맥 8 튜블릿 평균   context encoder (online), **문맥 16 샘플만** (ctx_masked = predictor 입력), 토큰별 LN
  p_fut  p 미래 8 튜블릿 평균   predictor 출력 (LN(target) 추정치, 다시 LN 하지 않음)
  h_fut  h 튜블릿 8–15 평균     target (EMA) encoder, 창 32 샘플 전체, 토큰별 LN — p 가 맞추도록 학습된 공간 (대조군)
  h_ctx  h 튜블릿 0–7 평균      대조: z 와 **같은 프레임**, 다른 encoder (또 target 은 미래 샘플도 본다)
조건: visible = sym_k '0' (visible 3 조건) / late = sym_k '1'..'4' (가림 3 조건, 문맥 끝 가림).
split: block 단위, train 0.5, seed 0, condition 층화 — 레포 하네스 `WMADataset._split` 를 그대로 부른다 (CLAUDE.md §1-5).

 1 PCA        joint (세 표현 합집합, 균형 부표본) · 표현별 · 설명분산 곡선 + participation ratio
 2 t-SNE      exact t-SNE (torch 구현) — **합성 데이터로 먼저 검증** (클러스터 분리 + 구조 없는 대조)
 3 CKA        linear CKA (+ block bootstrap CI, 순열 기준선) · debiased CKA (unbiased HSIC, Song et al. 2012)
 4 domain gap 중심 거리 / 표현 안 평균 쌍거리, 짝지은 제곱거리의 중심 이동 몫, 선형 domain 분류기 (block split)
 5 클래스 기하 shape · color: NCC (train 중심 → test), Fisher 비 tr(S_B)/tr(S_W), visible → late 이식, 선형 probe (참고)
 6 부분공간   클래스 평균 부분공간 (rank 6 / 7) 의 principal angle — 무작위 · 순열 (공유/독립) 기준선, split-half 천장
 0 분산 분해  env · condition · k · 장면 칸 · shape · color 가 설명하는 분산 몫

두 view: raw = 풀링 벡터 그대로 / resid = 장면 칸 (env × condition × sym_k, 60 칸) 평균을 뺀 것.
  칸 평균은 train clip 으로만 추정한다 (클래스 라벨은 안 쓴다). raw 분산의 ~76–80 % 가 장면 칸이라
  CKA · PCA · t-SNE · NCC · 부분공간이 전부 장면 구조에 지배된다 → 물체 구조는 resid 에서 본다.
  ⚠️ resid 의 visible→late 이식은 late 쪽 칸 평균 (라벨 없는 late train clip) 을 쓴다 = 조건별 중심 맞춤이 들어간 이식.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/v11_pooled_geometry.py --device cuda:1          # 전체 (~몇 분)
  $P z_research/scripts/analysis/v11_pooled_geometry.py --tables                  # results.json 에서 표만 다시 찍기
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
# 2026-09-25: 다른 predictor 의 p 로 돌리려고 환경변수로 덮을 수 있게 했다 (기본값 그대로). TrainingEffects/README.md
CACHE = Path(__import__("os").environ.get("V11P_CACHE", "/local_datasets/world/world_analysis/cache/v11_pooled_vith"))
OUT = ROOT / "z_research/RollOutV3/figures/v11_geometry"

N, D = 9408, 1280
REPS = ["z_all", "p_fut", "h_fut", "h_ctx"]
MAIN = ["z_all", "p_fut", "h_fut"]
CONDS = ["static_visible", "moving_visible_flat", "moving_visible",
         "static_occlusion", "moving_occlusion_flat", "moving_occlusion"]
MOTION = {"static": ("static_visible", "static_occlusion"), "flat": ("moving_visible_flat", "moving_occlusion_flat"),
          "ramp": ("moving_visible", "moving_occlusion")}
COLOR_ORDER = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
PAIRS = [("z_all", "p_fut"), ("h_fut", "p_fut"), ("z_all", "h_fut"), ("z_all", "h_ctx"), ("h_ctx", "h_fut"), ("h_ctx", "p_fut")]

# 팔레트 (CLAUDE.md §8-4): 세 slot + 회색/검정. 클래스 7–8 개는 명도 변형 + 마커 + 중앙값 직접 라벨로 구분한다
REP_COL = {"z_all": "#2a78d6", "p_fut": "#eb6834", "h_fut": "#1baf7a", "h_ctx": "#6e6e6e"}
CLS_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#111111", "#8c8c8c", "#9cc3ef", "#f4b393", "#8fdcbf"]
CLS_MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
F64 = torch.float64


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, torch.Tensor):
        return r4(x.item() if x.ndim == 0 else x.tolist())
    return x


# ============================================================================= data
def load():
    m = np.load(CACHE / "meta.npz")
    meta = {k: m[k] for k in m.files}
    z, p, h = (np.load(CACHE / f"{r}.npy") for r in ("z", "p", "h"))
    assert z.shape == (N, 8, D) and p.shape == (N, 8, D) and h.shape == (N, 16, D), (z.shape, p.shape, h.shape)
    X = {"z_all": z.astype(np.float32).mean(1), "p_fut": p.astype(np.float32).mean(1),
         "h_fut": h[:, 8:].astype(np.float32).mean(1), "h_ctx": h[:, :8].astype(np.float32).mean(1)}
    for k, v in X.items():
        assert np.isfinite(v).all(), k
    return X, meta


def repo_split(meta, frac=0.5, seed=0):
    """레포 하네스의 split (`WMADataset._split`) 을 그대로 부른다 → bool (train). block 단위, condition 층화."""
    from evals.world_model_analysis.data import WMADataset
    rows = [dict(video_id=str(v), block_id=str(b), condition=str(c))
            for v, b, c in zip(meta["video_id"], meta["block_id"], meta["condition"])]
    s = WMADataset._split(rows, dict(mode="ratio", group_by="block_id", train_frac=frac, seed=seed,
                                     stratify_by="condition"))
    tr = np.array([s[r["video_id"]] == "train" for r in rows])
    btr, bte = set(meta["block_id"][tr]), set(meta["block_id"][~tr])
    assert not (btr & bte), "block 이 train 과 test 양쪽에 있다"
    return tr


# ============================================================================= helpers (torch)
def eig_cov(X):
    """X (n,d) f64 → 고유값 내림차순 (공분산), 고유벡터 (d,d)."""
    Xc = X - X.mean(0)
    ev, V = torch.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    return ev.flip(0).clamp_min(0), V.flip(1)


def pr(ev):
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def eta2(scores, lab):
    """PC 점수 (n,k) 에서 라벨 평균이 설명하는 분산 비율 (PC 별)."""
    tot = ((scores - scores.mean(0)) ** 2).sum(0)
    bet = torch.zeros_like(tot)
    for u in np.unique(lab):
        s = scores[torch.as_tensor(lab == u, device=scores.device)]
        bet += len(s) * (s.mean(0) - scores.mean(0)) ** 2
    return (bet / tot).tolist()


def knn_acc(E, lab, k=10):
    """leave-one-out k-NN 다수결 정확도 (동률은 가장 가까운 이웃의 라벨로)."""
    E = torch.as_tensor(E, dtype=F64, device=DEV)
    _, inv = np.unique(lab, return_inverse=True)
    y = torch.as_tensor(inv, device=DEV)
    Dm = torch.cdist(E, E)
    Dm.fill_diagonal_(float("inf"))
    nn = Dm.topk(k, largest=False).indices
    C = int(y.max()) + 1
    votes = torch.nn.functional.one_hot(y[nn], C).sum(1).to(F64)
    votes += 1e-3 * torch.nn.functional.one_hot(y[nn[:, 0]], C)
    return float((votes.argmax(1) == y).double().mean())


def mean_pdist(X, chunk=1024):
    """i≠j 평균 유클리드 쌍거리."""
    n, s = len(X), 0.0
    for i in range(0, n, chunk):
        s += torch.cdist(X[i:i + chunk], X, compute_mode="donot_use_mm_for_euclid_dist").sum().item()
    return s / (n * (n - 1))


def logreg(Xtr, ytr, Xte, yte, lam=1e-3, iters=300):
    """softmax 회귀 (train 표준화, L2 λ 고정 — test 로 고르지 않는다). → (test acc, train acc)."""
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=DEV)
    u = np.unique(ytr); ytr_i = torch.as_tensor(np.searchsorted(u, ytr), device=DEV)
    yte_i = torch.as_tensor(np.array([np.searchsorted(u, v) if v in u else -1 for v in yte]), device=DEV)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(A.shape[1], len(u), device=DEV, requires_grad=True)
    b = torch.zeros(len(u), device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, ytr_i) + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        te = float(((B @ W + b).argmax(1) == yte_i).float().mean())
        tr = float(((A @ W + b).argmax(1) == ytr_i).float().mean())
    return te, tr


# ============================================================================= t-SNE (exact)
def tsne(X, perplexity=30.0, n_iter=1000, lr=200.0, exag=12.0, exag_iter=250, n_pca=50, seed=0):
    """exact t-SNE (van der Maaten 2008; sklearn 과 같은 최적화 규약).
    입력 → PCA-50, 점마다 σ 이분탐색 (엔트로피 = log perplexity), P 대칭화, 초기값 = PCA 2 성분 (PC1 std 1e-4),
    기울기 4 Σ (p−q)(1+d²)^-1 (y_i − y_j), momentum 0.5 → 0.8, gains (+0.2 / ×0.8, 최소 0.01),
    early exaggeration 12 (첫 250 반복), 총 1000 반복, lr 200."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=F64, device=DEV)
    X = X - X.mean(0)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    X50 = X @ Vh[:min(n_pca, X.shape[1])].T
    n = len(X50)
    Dm = torch.cdist(X50, X50) ** 2
    Dm.fill_diagonal_(0)
    Ds = Dm - torch.where(torch.eye(n, dtype=torch.bool, device=DEV), torch.inf, Dm).min(1).values[:, None]
    Ds.fill_diagonal_(0)                                          # 대각은 W 에서 0 으로 막는다 (exp 넘침 방지)
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    target = math.log(perplexity)
    lo, hi = torch.full((n,), -60.0, dtype=F64, device=DEV), torch.full((n,), 60.0, dtype=F64, device=DEV)
    for _ in range(200):                                          # log β 이분탐색 (행마다 독립)
        mid = (lo + hi) / 2
        b = mid.exp()[:, None]
        W = torch.exp(-b * Ds) * off
        sW = W.sum(1)
        H = sW.log() + (b[:, 0] * (W * Ds).sum(1) / sW)
        big = H > target                                           # 엔트로피가 크면 β 를 키운다
        lo, hi = torch.where(big, mid, lo), torch.where(big, hi, mid)
    b = ((lo + hi) / 2).exp()[:, None]
    W = torch.exp(-b * Ds) * off
    Pc = W / W.sum(1, keepdim=True)
    Hc = -(Pc * torch.log(Pc.clamp_min(1e-300))).sum(1)
    ent_err = float((Hc - target).abs().max())
    P = ((Pc + Pc.T) / (2 * n)).clamp_min(1e-12)

    Y = X50[:, :2] / X50[:, 0].std() * 1e-4
    upd, gains = torch.zeros_like(Y), torch.ones_like(Y)
    for it in range(n_iter):
        sq = (Y * Y).sum(1)
        num = 1.0 / (1.0 + (sq[:, None] + sq[None, :] - 2 * Y @ Y.T).clamp_min(0))
        num.fill_diagonal_(0)
        Q = (num / num.sum()).clamp_min(1e-12)
        PQ = ((P * exag if it < exag_iter else P) - Q) * num
        grad = 4.0 * (PQ.sum(1)[:, None] * Y - PQ @ Y)
        mom = 0.5 if it < exag_iter else 0.8
        gains = torch.where(upd * grad < 0, gains + 0.2, gains * 0.8).clamp_min(0.01)
        upd = mom * upd - lr * gains * grad
        Y = Y - (Y + upd).mean(0) + upd                            # 평행이동 제거
    sq = (Y * Y).sum(1)
    num = 1.0 / (1.0 + (sq[:, None] + sq[None, :] - 2 * Y @ Y.T).clamp_min(0)); num.fill_diagonal_(0)
    Q = (num / num.sum()).clamp_min(1e-12)
    kl = float((P * torch.log(P / Q)).sum())
    return Y.cpu().numpy(), dict(n=n, kl=kl, max_entropy_err=ent_err, perplexity=perplexity, n_iter=n_iter, lr=lr,
                                 early_exaggeration=exag, exag_iter=exag_iter, n_pca=int(X50.shape[1]))


def sep_ratio(E, lab):
    """임베딩에서 (클러스터 중심 사이 최소 거리) / (클러스터 안 중심까지 거리의 최대 90 분위)."""
    E = np.asarray(E); us = np.unique(lab)
    cen = np.stack([E[lab == u].mean(0) for u in us])
    dc = np.linalg.norm(cen[:, None] - cen[None], axis=-1); dc[np.eye(len(us), dtype=bool)] = np.inf
    rad = max(np.percentile(np.linalg.norm(E[lab == u] - cen[i], axis=1), 90) for i, u in enumerate(us))
    return float(dc.min() / rad)


def validate_tsne(seed=0):
    rng = np.random.default_rng(seed)
    k, m = 6, 300
    s = 15.0 / math.sqrt(2 * D)                                    # 중심 사이 거리 ≈ 15, 안쪽 쌍거리 ≈ √(2D) ≈ 50.6
    cen = rng.normal(0, s, (k, D))
    Xa = np.concatenate([cen[i] + rng.normal(0, 1, (m, D)) for i in range(k)])
    la = np.repeat(np.arange(k), m)
    Ya, ia = tsne(Xa, seed=seed)
    A = dict(desc=f"{k} Gaussian clusters x {m} pts in {D}-d, within sd 1, center dist ~15 (within pairwise ~50.6)",
             knn10_input=knn_acc(Xa, la), knn10_tsne=knn_acc(Ya, la), sep_ratio_tsne=sep_ratio(Ya, la), **ia)
    s2 = 6.0 / math.sqrt(2 * D)                                   # 겹치는 클러스터: 중심 거리 ≈ 6 (입력 kNN 이 1 미만이 되게)
    cen2 = rng.normal(0, s2, (k, D))
    Xh = np.concatenate([cen2[i] + rng.normal(0, 1, (m, D)) for i in range(k)])
    Yh, ih = tsne(Xh, seed=seed)
    Hd = dict(desc=f"{k} overlapping Gaussian clusters x {m} pts, center dist ~6", knn10_input=knn_acc(Xh, la),
              knn10_tsne=knn_acc(Yh, la), sep_ratio_tsne=sep_ratio(Yh, la), **ih)
    Xb = rng.normal(0, 1, (k * m, D))                             # 구조 없음 + 무작위 라벨
    lb = rng.permutation(la)
    Yb, ib = tsne(Xb, seed=seed)
    B = dict(desc=f"null: one isotropic Gaussian ({k*m} pts), random labels", chance=1 / k,
             knn10_input=knn_acc(Xb, lb), knn10_tsne=knn_acc(Yb, lb), sep_ratio_tsne=sep_ratio(Yb, lb), **ib)
    return dict(clusters=A, overlapping=Hd, null=B), (Ya, la, Yb, lb)


# ============================================================================= CKA
def cka_linear(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).pow(2).sum() / torch.sqrt((X.T @ X).pow(2).sum() * (Y.T @ Y).pow(2).sum()))


def gram0(X):
    X = X - X.mean(0)
    K = X @ X.T
    K.fill_diagonal_(0)
    return K


def hsic_u(K, L):
    """unbiased HSIC (Song et al. 2012), K·L 대각 0."""
    n = K.shape[0]
    k1, l1 = K.sum(1), L.sum(1)
    return float(((K * L).sum() + k1.sum() * l1.sum() / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * (k1 @ l1)) / (n * (n - 3)))


def cka_section(Xg, masks, blocks, n_boot, n_perm, rng):
    out = {}
    for sub in ["all", "visible", "late"] + CONDS:
        idx = torch.as_tensor(np.where(masks[sub])[0], device=DEV)
        Xs = {r: Xg[r][idx] for r in REPS}
        lin = {f"{a}|{b}": cka_linear(Xs[a], Xs[b]) for i, a in enumerate(REPS) for b in REPS[i + 1:]}
        G = {r: gram0(Xs[r]) for r in REPS}
        hs = {r: hsic_u(G[r], G[r]) for r in REPS}
        deb = {f"{a}|{b}": hsic_u(G[a], G[b]) / math.sqrt(hs[a] * hs[b]) for i, a in enumerate(REPS) for b in REPS[i + 1:]}
        del G
        torch.cuda.empty_cache()
        o = dict(n=len(idx), linear=lin, debiased=deb)
        if sub in ("all", "visible", "late"):
            # 순열 기준선: clip 짝을 섞는다
            perm = {}
            for key in lin:
                a, b = key.split("|")
                A32, B32 = Xs[a].float(), Xs[b].float()
                vals = [cka_linear(A32, B32[torch.as_tensor(rng.permutation(len(idx)), device=DEV)]) for _ in range(n_perm)]
                perm[key] = dict(mean=float(np.mean(vals)), max=float(np.max(vals)))
            o["linear_perm"] = perm
            # block bootstrap (clip 가중치 = 뽑힌 block 횟수), fp32
            bl = blocks[masks[sub]]
            ub, inv = np.unique(bl, return_inverse=True)
            X32 = {r: Xs[r].float() for r in REPS}
            boots = {k: [] for k in lin}
            for _ in range(n_boot):
                cnt = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub))
                w = torch.as_tensor(cnt[inv], dtype=torch.float32, device=DEV)
                Gw, self_ = {}, {}
                for r in REPS:
                    mu = (w[:, None] * X32[r]).sum(0) / w.sum()
                    Gw[r] = (X32[r] - mu) * w.sqrt()[:, None]
                    self_[r] = (Gw[r].T @ Gw[r]).pow(2).sum()
                for key in lin:
                    a, b = key.split("|")
                    boots[key].append(float((Gw[a].T @ Gw[b]).pow(2).sum() / torch.sqrt(self_[a] * self_[b])))
            o["linear_ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        out[sub] = o
        log(f"  CKA {sub:22s} n={len(idx)}  " + "  ".join(f"{k}={v:.3f}/{deb[k]:.3f}" for k, v in lin.items()))
    return out


# ============================================================================= domain gap
def domain_section(Xg, masks, train):
    out = {"norms": {}, "pairs": {}}
    for sub in ("all", "visible", "late"):
        idx = torch.as_tensor(np.where(masks[sub])[0], device=DEV)
        out["norms"][sub] = {r: dict(mean_norm=float(Xg[r][idx].norm(dim=1).mean()),
                                     rms_radius=float(((Xg[r][idx] - Xg[r][idx].mean(0)) ** 2).sum(1).mean().sqrt()),
                                     mean_pdist=mean_pdist(Xg[r][idx])) for r in REPS}
    for a, b in PAIRS:
        for sub in ("all", "visible", "late"):
            m = masks[sub]
            idx = torch.as_tensor(np.where(m)[0], device=DEV)
            A, B = Xg[a][idx], Xg[b][idx]
            mA, mB = A.mean(0), B.mean(0)
            dc = float((mA - mB).norm())
            nA, nB = out["norms"][sub][a], out["norms"][sub][b]
            wpd = (nA["mean_pdist"] + nB["mean_pdist"]) / 2
            rms = math.sqrt((nA["rms_radius"] ** 2 + nB["rms_radius"] ** 2) / 2)
            paired = float(((A - B) ** 2).sum(1).mean())
            o = dict(centroid_dist=dc, within_mean_pdist=wpd, ratio_centroid_over_pdist=dc / wpd,
                     ratio_centroid_over_rms=dc / rms, centroid_cos=float(mA @ mB / (mA.norm() * mB.norm())),
                     paired_sqdist=paired, offset_share_of_paired_sqdist=dc ** 2 / paired)
            # 선형 domain 분류기 (block split): train clip 의 A·B → test clip 의 A·B
            tr, te = np.where(m & train)[0], np.where(m & ~train)[0]
            XA, XB = Xg[a].cpu().numpy(), Xg[b].cpu().numpy()
            Xtr = np.concatenate([XA[tr], XB[tr]]); ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
            Xte = np.concatenate([XA[te], XB[te]]); yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
            o["clf_raw"], o["clf_raw_train"] = logreg(Xtr, ytr, Xte, yte)
            un = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
            o["clf_unitnorm"], _ = logreg(un(Xtr), ytr, un(Xte), yte)
            o["clf_norm_only"], _ = logreg(np.linalg.norm(Xtr, axis=1, keepdims=True), ytr,
                                           np.linalg.norm(Xte, axis=1, keepdims=True), yte)
            d = XA[tr].mean(0) - XB[tr].mean(0); mid = (XA[tr].mean(0) + XB[tr].mean(0)) / 2
            o["clf_centroid_dir"] = float((((Xte - mid) @ d > 0) == (yte == 0)).mean())
            out["pairs"][f"{a}|{b}|{sub}"] = o
            log(f"  domain {a}|{b} {sub:8s} dc/pd={o['ratio_centroid_over_pdist']:.3f} offset={o['offset_share_of_paired_sqdist']:.3f} "
                f"clf raw={o['clf_raw']:.4f} unit={o['clf_unitnorm']:.4f} norm={o['clf_norm_only']:.4f} cdir={o['clf_centroid_dir']:.4f}")
    return out


# ============================================================================= class geometry
def ncc(Xtr, ytr, Xte, yte, classes):
    mu = torch.stack([Xtr[torch.as_tensor(ytr == c, device=DEV)].mean(0) for c in classes])
    pred = torch.cdist(Xte, mu).argmin(1).cpu().numpy()
    yi = np.array([classes.index(v) for v in yte])
    acc = float((pred == yi).mean())
    bacc = float(np.mean([(pred[yi == i] == i).mean() for i in range(len(classes)) if (yi == i).any()]))
    return acc, bacc


def onehot(yi, C):
    return torch.nn.functional.one_hot(torch.as_tensor(yi, device=DEV), C).to(F64)


def fisher(X, yi, C):
    """tr(S_B) / tr(S_W). yi = 정수 라벨 (np)."""
    Y = onehot(yi, C); cnt = Y.sum(0)
    M = (Y.T @ X) / cnt.clamp_min(1)[:, None]
    mu = X.mean(0)
    sb = float((cnt * ((M - mu) ** 2).sum(1)).sum())
    sw = float(((X - Y @ M) ** 2).sum())
    return sb / sw


def class_section(Xg, meta, masks, train, rng, n_perm_fisher=50):
    out = {}
    labs = {"shape": (meta["shape_pre"].astype(str), sorted(set(meta["shape_pre"].astype(str)))),
            "color": (meta["color_pre"].astype(str), COLOR_ORDER)}
    for lname, (y, classes) in labs.items():
        assert set(y) == set(classes)
        for r in REPS:
            X = Xg[r]
            for sub in ["visible", "late"] + CONDS:
                m = masks[sub]
                tr, te = m & train, m & ~train
                acc, bacc = ncc(X[torch.as_tensor(tr, device=DEV)], y[tr], X[torch.as_tensor(te, device=DEV)], y[te], classes)
                Xs = X[torch.as_tensor(m, device=DEV)]
                yi = np.array([classes.index(v) for v in y[m]])
                fr = fisher(Xs, yi, len(classes))
                perm = float(np.mean([fisher(Xs, rng.permutation(yi), len(classes)) for _ in range(n_perm_fisher)]))
                lp, lp_tr = logreg(Xg[r][torch.as_tensor(tr, device=DEV)].cpu().numpy(), y[tr],
                                   Xg[r][torch.as_tensor(te, device=DEV)].cpu().numpy(), y[te])
                cnt = np.unique(y[te], return_counts=True)[1]
                out[f"{lname}|{r}|{sub}"] = dict(ncc=acc, ncc_bacc=bacc, fisher=fr, fisher_perm=perm, linprobe=lp,
                                                 linprobe_train=lp_tr, n_train=int(tr.sum()), n_test=int(te.sum()),
                                                 chance=1 / len(classes), majority=float(cnt.max() / cnt.sum()))
            # 이식: 한 조건 train 중심 → 다른 조건 test
            for src, dst in [("visible", "late"), ("late", "visible")] + [MOTION[k] for k in ("static", "flat", "ramp")] + \
                            [(MOTION[k][1], MOTION[k][0]) for k in ("static", "flat", "ramp")]:
                trm, tem = masks[src] & train, masks[dst] & ~train
                acc, bacc = ncc(X[torch.as_tensor(trm, device=DEV)], y[trm], X[torch.as_tensor(tem, device=DEV)], y[tem], classes)
                lp, _ = logreg(Xg[r][torch.as_tensor(trm, device=DEV)].cpu().numpy(), y[trm],
                               Xg[r][torch.as_tensor(tem, device=DEV)].cpu().numpy(), y[tem])
                out[f"{lname}|{r}|{src}->{dst}"] = dict(ncc=acc, ncc_bacc=bacc, linprobe=lp)
            log(f"  class {lname} {r}: NCC vis={out[f'{lname}|{r}|visible']['ncc']:.3f} late={out[f'{lname}|{r}|late']['ncc']:.3f} "
                f"vis->late={out[f'{lname}|{r}|visible->late']['ncc']:.3f}  lin vis={out[f'{lname}|{r}|visible']['linprobe']:.3f} "
                f"late={out[f'{lname}|{r}|late']['linprobe']:.3f} vis->late={out[f'{lname}|{r}|visible->late']['linprobe']:.3f}")
    return out


# ============================================================================= subspace overlap
def class_means(X, yi, C):
    """중심화한 클래스 평균 (C, d) — 중심 = 클래스 평균들의 (가중 없는) 평균. 부분공간은 중심 선택과 무관 (차이들의 span)."""
    Y = onehot(yi, C)
    M = (Y.T @ X) / Y.sum(0)[:, None]
    return M - M.mean(0)


def basis(Mc, r):
    _, _, Vh = torch.linalg.svd(Mc, full_matrices=False)
    return Vh[:r].T                                                 # (d, r)


def overlap(UA, UB):
    s = torch.linalg.svdvals(UA.T @ UB).clamp(0, 1)
    return float((s ** 2).mean()), [float(v) for v in torch.rad2deg(torch.arccos(s))]


def captured(Mc, U):
    return float((Mc @ U).pow(2).sum() / Mc.pow(2).sum())


def subspace_section(Xg, meta, masks, train, rng, n_perm):
    out = {"random": {}}
    labs = {"shape": (meta["shape_pre"].astype(str), sorted(set(meta["shape_pre"].astype(str)))),
            "color": (meta["color_pre"].astype(str), COLOR_ORDER)}
    for lname, (ys, classes) in labs.items():
        r, C = len(classes) - 1, len(classes)
        y = np.array([classes.index(v) for v in ys])
        rs = []
        for _ in range(1000):
            Qa = torch.linalg.qr(torch.randn(D, r, dtype=F64, device=DEV))[0]
            Qb = torch.linalg.qr(torch.randn(D, r, dtype=F64, device=DEV))[0]
            rs.append(overlap(Qa, Qb)[0])
        out["random"][lname] = dict(rank=r, analytic=r / D, mean=float(np.mean(rs)), p95=float(np.percentile(rs, 95)))
        Ms, Us = {}, {}
        for rep in REPS:
            for sub in ("visible", "late"):
                m = masks[sub]
                Ms[rep, sub] = class_means(Xg[rep][torch.as_tensor(m, device=DEV)], y[m], C)
                Us[rep, sub] = basis(Ms[rep, sub], r)
                sv = torch.linalg.svdvals(Ms[rep, sub])
                # split-half 천장: train block 의 부분공간 vs test block 의 부분공간 (clip 이 겹치지 않는다)
                for h, mh in (("tr", m & train), ("te", m & ~train)):
                    Ms[rep, sub, h] = class_means(Xg[rep][torch.as_tensor(mh, device=DEV)], y[mh], C)
                    Us[rep, sub, h] = basis(Ms[rep, sub, h], r)
                out[f"{lname}|{rep}|{sub}|self"] = dict(
                    split_half=overlap(Us[rep, sub, "tr"], Us[rep, sub, "te"])[0],
                    split_half_captured=(captured(Ms[rep, sub, "te"], Us[rep, sub, "tr"]) + captured(Ms[rep, sub, "tr"], Us[rep, sub, "te"])) / 2,
                    class_mean_sv=sv[:r].tolist())

        def xfit(MA, UA, MB, UB):
            """교차 적합: A 의 한 절반 vs B 의 다른 절반 (clip 이 안 겹친다 → 공유 표집 잡음 없음)."""
            cos2 = (overlap(UA["tr"], UB["te"])[0] + overlap(UA["te"], UB["tr"])[0]) / 2
            cap = (captured(MA["te"], UB["tr"]) + captured(MA["tr"], UB["te"]) + captured(MB["te"], UA["tr"]) + captured(MB["tr"], UA["te"])) / 4
            return cos2, cap

        def halves(rep, sub, yy=None):
            M, U = {}, {}
            for h, mh in (("tr", masks[sub] & train), ("te", masks[sub] & ~train)):
                lab = y[mh] if yy is None else yy[h]
                M[h] = class_means(Xg[rep][torch.as_tensor(mh, device=DEV)], lab, C); U[h] = basis(M[h], r)
            return M, U

        def xfit_perm(a, sa, b, sb, n=50):
            c2, cp = [], []
            for _ in range(n):
                ya = {h: rng.permutation(y[masks[sa] & (train if h == "tr" else ~train)]) for h in ("tr", "te")}
                yb = {h: rng.permutation(y[masks[sb] & (train if h == "tr" else ~train)]) for h in ("tr", "te")}
                v = xfit(*halves(a, sa, ya), *halves(b, sb, yb))
                c2.append(v[0]); cp.append(v[1])
            return dict(cos2_mean=float(np.mean(c2)), cos2_p95=float(np.percentile(c2, 95)),
                        captured_mean=float(np.mean(cp)), captured_p95=float(np.percentile(cp, 95)))

        def perm_overlap(repA, subA, repB, subB, shared):
            vals = []
            mA, mB = masks[subA], masks[subB]
            XA, XB = Xg[repA][torch.as_tensor(mA, device=DEV)], Xg[repB][torch.as_tensor(mB, device=DEV)]
            for _ in range(n_perm):
                pa = rng.permutation(y[mA])
                pb = pa if shared else rng.permutation(y[mB])
                vals.append(overlap(basis(class_means(XA, pa, C), r), basis(class_means(XB, pb, C), r))[0])
            return dict(mean=float(np.mean(vals)), p95=float(np.percentile(vals, 95)))

        combos = [(a, s, b, s) for s in ("visible", "late") for a, b in PAIRS] + [(rep, "visible", rep, "late") for rep in REPS]
        for a, sa, b, sb in combos:
            ov, ang = overlap(Us[a, sa], Us[b, sb])
            o = dict(mean_cos2=ov, angles_deg=ang, captured_A_in_B=captured(Ms[a, sa], Us[b, sb]),
                     captured_B_in_A=captured(Ms[b, sb], Us[a, sa]), perm_indep=perm_overlap(a, sa, b, sb, False))
            if sa == sb:
                o["perm_shared"] = perm_overlap(a, sa, b, sb, True)
            MA = {h: Ms[a, sa, h] for h in ("tr", "te")}; UA = {h: Us[a, sa, h] for h in ("tr", "te")}
            MB = {h: Ms[b, sb, h] for h in ("tr", "te")}; UB = {h: Us[b, sb, h] for h in ("tr", "te")}
            c2, cp = xfit(MA, UA, MB, UB)
            sA, sB = out[f"{lname}|{a}|{sa}|self"], out[f"{lname}|{b}|{sb}|self"]
            o.update(xfit_cos2=c2, xfit_captured=cp,
                     xfit_cos2_norm=c2 / math.sqrt(sA["split_half"] * sB["split_half"]),
                     xfit_captured_norm=cp / math.sqrt(sA["split_half_captured"] * sB["split_half_captured"]),
                     xfit_perm=xfit_perm(a, sa, b, sb))
            out[f"{lname}|{a}:{sa}|{b}:{sb}"] = o
            log(f"  subspace {lname} {a}:{sa} vs {b}:{sb}  cos2={ov:.3f}  capA={o['captured_A_in_B']:.3f} xfit={c2:.3f}/{cp:.3f} "
                f"norm={o['xfit_cos2_norm']:.3f}/{o['xfit_captured_norm']:.3f} "
                f"perm_indep={o['perm_indep']['mean']:.3f}" + (f" perm_shared={o['perm_shared']['mean']:.3f}" if 'perm_shared' in o else ""))
    return out


# ============================================================================= figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
                         "xtick.color": "#555555", "ytick.color": "#555555", "legend.frameon": False})
    return plt


def panel_label(ax, s, y=-0.13):
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top", fontsize=10)


def scatter_cond(ax, E, col, vis, s=7, alpha=0.45, label=None):
    ax.scatter(E[vis, 0], E[vis, 1], s=s, c=col, alpha=alpha, lw=0, marker="o", label=None if label is None else f"{label} visible")
    ax.scatter(E[~vis, 0], E[~vis, 1], s=s + 3, facecolors="none", edgecolors=col, alpha=alpha + 0.1, lw=0.6, marker="^",
               label=None if label is None else f"{label} late")


def scatter_cls(ax, E, lab, classes, s=7, alpha=0.55, vis=None):
    for i, c in enumerate(classes):
        k = lab == c
        if vis is None:
            ax.scatter(E[k, 0], E[k, 1], s=s, c=CLS_COL[i], alpha=alpha, lw=0, marker=CLS_MK[i], label=str(c))
        else:
            ax.scatter(E[k & vis, 0], E[k & vis, 1], s=s, c=CLS_COL[i], alpha=alpha, lw=0, marker="o", label=str(c))
            ax.scatter(E[k & ~vis, 0], E[k & ~vis, 1], s=s + 3, facecolors="none", edgecolors=CLS_COL[i], alpha=alpha, lw=0.6, marker="^")


def cls_legend(fig, plt, classes, x, title, y=0.0, filled_only=False):
    hs = [plt.Line2D([], [], marker="o" if filled_only else CLS_MK[i], color=CLS_COL[i], lw=0, ms=6) for i in range(len(classes))]
    fig.legend(hs, classes, loc="lower center", ncol=len(classes), fontsize=8, bbox_to_anchor=(x, y), title=title, title_fontsize=8)


def fig_pca_joint(plt, P, res, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    E, rep, vis, env = P["joint_scores"], P["joint_rep"], P["joint_vis"], P["joint_env"]
    evr = res["pca"]["joint"]["explained_ratio"]
    for ax, (i, j), lab in zip(axs[:2], [(0, 1), (2, 3)], ["(a)", "(b)"]):
        for r in MAIN:
            k = rep == r
            scatter_cond(ax, E[k][:, [i, j]], REP_COL[r], vis[k], label=r)
        ax.set_xlabel(f"PC{i+1} ({100*evr[i]:.1f}%)"); ax.set_ylabel(f"PC{j+1} ({100*evr[j]:.1f}%)")
        ax.set_title(f"joint PCA | representation x condition | PC{i+1}-PC{j+1}", fontsize=9)
        panel_label(ax, lab)
    envs = sorted(set(env))
    scatter_cls(axs[2], E[:, [0, 1]], env, envs, s=6, vis=vis)
    axs[2].set_xlabel(f"PC1 ({100*evr[0]:.1f}%)"); axs[2].set_ylabel(f"PC2 ({100*evr[1]:.1f}%)")
    axs[2].set_title("joint PCA | background env (o visible, ^ late)", fontsize=9)
    axs[2].legend(fontsize=7, markerscale=2, loc="upper center", ncol=2, frameon=True, framealpha=0.9, edgecolor="none")
    panel_label(axs[2], "(c)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, markerscale=2.2, fontsize=8, bbox_to_anchor=(0.36, 1.01))
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out / "fig_pca_joint.png", dpi=160); plt.close(fig)


def fig_pca_by_rep(plt, P, res, out):
    fig, axs = plt.subplots(6, 4, figsize=(15, 21))
    shapes, colors = P["shape_classes"], COLOR_ORDER
    vis = P["rep_vis"]; k = 0
    for view_i, view in enumerate(("raw", "resid")):
        for ri, r in enumerate(MAIN):
            E = P["rep_scores"][view][r]
            evr = res["pca"]["per_rep"][view][r]["explained_ratio_top5"]
            row = view_i * 3 + ri
            for ci, (lname, classes, lab) in enumerate([("shape", shapes, P["rep_shape"]), ("color", colors, P["rep_color"])]):
                for vi, (vname, vm) in enumerate([("visible", vis), ("late", ~vis)]):
                    ax = axs[row, 2 * ci + vi]
                    scatter_cls(ax, E[vm], lab[vm], classes, s=5)
                    lo0, hi0 = np.percentile(E[:, 0], [0.3, 99.7]); lo1, hi1 = np.percentile(E[:, 1], [0.3, 99.7])
                    pad0, pad1 = 0.06 * (hi0 - lo0), 0.06 * (hi1 - lo1)
                    ax.set_xlim(lo0 - pad0, hi0 + pad0); ax.set_ylim(lo1 - pad1, hi1 + pad1)
                    ax.set_title(f"{r} [{view}] | {lname} | {vname}", fontsize=8.5)
                    ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}%)", fontsize=8); ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}%)", fontsize=8)
                    ax.tick_params(labelsize=7)
                    panel_label(ax, f"({chr(97 + k)})" if k < 26 else f"({k})", y=-0.24); k += 1
    cls_legend(fig, plt, shapes, 0.27, "shape")
    cls_legend(fig, plt, colors, 0.74, "color")
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=3.2)
    fig.savefig(out / "fig_pca_by_rep.png", dpi=120); plt.close(fig)


def fig_explained(plt, res, out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for r in REPS:
        for sub, ls in (("visible", "-"), ("late", "--")):
            for view, ax_i in (("raw", (0, 1)), ("resid", (2,))):
                e = res["pca"]["explained"][view][f"{r}|{sub}"]
                ev = np.array(e["ratio_top50"])
                kw = dict(color=REP_COL[r], ls=ls, lw=1.6 if r != "h_ctx" else 1.1,
                          label=f"{r} {sub} (PR {e['participation_ratio']:.1f})")
                if view == "raw":
                    axs[0].plot(np.arange(1, 51), ev, **kw)
                    axs[1].plot(np.arange(1, 51), np.cumsum(ev), **kw)
                else:
                    axs[2].plot(np.arange(1, 51), np.cumsum(ev), **kw)
    axs[0].set_yscale("log"); axs[0].set_ylabel("explained variance ratio")
    axs[1].set_ylabel("cumulative explained variance"); axs[2].set_ylabel("cumulative explained variance")
    axs[0].set_title("raw | per-PC ratio", fontsize=9); axs[1].set_title("raw | cumulative", fontsize=9)
    axs[2].set_title("env x condition x k cell means removed | cumulative", fontsize=9)
    for ax, lab in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.set_xlabel("principal component"); ax.grid(alpha=0.25, lw=0.5); panel_label(ax, lab)
    axs[1].set_ylim(0, 1); axs[2].set_ylim(0, 1)
    axs[1].legend(fontsize=7, loc="lower right"); axs[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig_explained_variance.png", dpi=160); plt.close(fig)


def fig_tsne(plt, P, out):
    fig, axs = plt.subplots(4, 3, figsize=(14, 18.5))
    Ya, la, Yb, lb = P["tsne_val"]
    scatter_cls(axs[0, 0], Ya, la, list(range(6)), s=6)
    axs[0, 0].set_title("validation | 6 Gaussian clusters (1280-d)", fontsize=9)
    scatter_cls(axs[0, 1], Yb, lb, list(range(6)), s=6)
    axs[0, 1].set_title("validation | null: 1 Gaussian, random labels", fontsize=9)
    E, rep, vis = P["tsne_joint"], P["tsne_rep"], P["tsne_vis"]
    for r in MAIN:
        k = rep == r
        scatter_cond(axs[0, 2], E[k], REP_COL[r], vis[k], s=6, label=r)
    axs[0, 2].set_title("joint t-SNE | representation x condition", fontsize=9)
    axs[0, 2].legend(fontsize=7, markerscale=2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    scatter_cls(axs[1, 0], E, P["tsne_shape"], P["shape_classes"], s=6)
    axs[1, 0].set_title("joint t-SNE | shape", fontsize=9)
    scatter_cls(axs[1, 1], E, P["tsne_color"], COLOR_ORDER, s=6)
    axs[1, 1].set_title("joint t-SNE | color", fontsize=9)
    envs = sorted(set(P["tsne_env"]))
    scatter_cls(axs[1, 2], E, P["tsne_env"], envs, s=6)
    axs[1, 2].set_title("joint t-SNE | background env", fontsize=9)
    axs[1, 2].legend(fontsize=7, markerscale=2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    for row, view in ((2, "raw"), (3, "resid")):
        for ri, r in enumerate(MAIN):
            ax = axs[row, ri]
            scatter_cls(ax, P["tsne_per_rep"][view][r], P["tsne_rep_shape"], P["shape_classes"], s=7, vis=P["tsne_rep_vis"])
            ax.set_title(f"{r} alone [{view}] | shape (o visible, ^ late)", fontsize=9)
    for k, ax in enumerate(axs.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        panel_label(ax, f"({chr(97 + k)})")
    cls_legend(fig, plt, P["shape_classes"], 0.28, "shape")
    cls_legend(fig, plt, COLOR_ORDER, 0.75, "color (panel e)")
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=3)
    fig.savefig(out / "fig_tsne.png", dpi=130); plt.close(fig)


def fig_cka(plt, res, out):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#9cc3ef", "#2a78d6", "#0e3b75"])
    fig, axs = plt.subplots(3, 3, figsize=(12, 11.4))
    for ri, (view, kind) in enumerate((("raw", "linear"), ("raw", "debiased"), ("resid", "linear"))):
        for ci, sub in enumerate(("all", "visible", "late")):
            ax = axs[ri, ci]
            c = res["cka"][view][sub]
            M = np.eye(len(REPS))
            for i, a in enumerate(REPS):
                for j, b in enumerate(REPS):
                    if i < j:
                        M[i, j] = M[j, i] = c[kind][f"{a}|{b}"]
            im = ax.imshow(M, cmap=cmap, vmin=0.9, vmax=1.0)          # 값이 0.92–1.00 에 몰려 있어 색 범위를 0.9–1 로 둔다
            for i in range(len(REPS)):
                for j in range(len(REPS)):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                            color="white" if M[i, j] > 0.965 else "#111111")
            ax.set_xticks(range(len(REPS))); ax.set_xticklabels(REPS, fontsize=8)
            ax.set_yticks(range(len(REPS))); ax.set_yticklabels(REPS, fontsize=8)
            ax.spines[:].set_visible(False)
            vt = "raw" if view == "raw" else "cell means removed"
            ax.set_title(f"{kind} CKA | {vt} | {sub} (n={c['n']})", fontsize=9)
            panel_label(ax, f"({chr(97 + ri*3 + ci)})")
    fig.tight_layout(rect=(0, 0.02, 0.91, 1), h_pad=3)
    cax = fig.add_axes([0.925, 0.3, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="CKA")
    fig.savefig(out / "fig_cka.png", dpi=160); plt.close(fig)


# ============================================================================= tables
def tables(res):
    w = print
    w("### variance share (between-group / total), raw; resid = share of residual variance")
    fs = list(next(iter(res["variance"].values())).keys())
    w("| rep|sub | " + " | ".join(fs) + " |\n|---|" + "---:|" * len(fs))
    for k, v in res["variance"].items():
        w(f"| {k} | " + " | ".join(f"{v[f]:.4f}" for f in fs) + " |")
    for view in ("raw", "resid"):
        w(f"\n### CKA [{view}] (linear [95% block bootstrap] / debiased; linear perm mean)")
        for sub in ("all", "visible", "late"):
            c = res["cka"][view][sub]
            w(f"\n{sub} (n={c['n']})\n| pair | linear | 95% CI | debiased | linear perm mean |\n|---|---:|---:|---:|---:|")
            for k in c["linear"]:
                w(f"| {k} | {c['linear'][k]:.3f} | {c['linear_ci95'][k][0]:.3f}–{c['linear_ci95'][k][1]:.3f} | {c['debiased'][k]:.3f} | {c['linear_perm'][k]['mean']:.3f} |")
        w(f"\nper condition [{view}] (linear / debiased)\n| condition | " + " | ".join(res["cka"][view]["all"]["linear"].keys()) + " |\n|---|" + "---:|" * 6)
        for sub in CONDS:
            c = res["cka"][view][sub]
            w(f"| {sub} | " + " | ".join(f"{c['linear'][k]:.3f} / {c['debiased'][k]:.3f}" for k in c["linear"]) + " |")
    w("\n### norms")
    w("| rep | sub | mean ‖x‖ | RMS radius | mean pairwise dist |\n|---|---|---:|---:|---:|")
    for sub in ("visible", "late"):
        for r in REPS:
            n = res["domain"]["norms"][sub][r]
            w(f"| {r} | {sub} | {n['mean_norm']:.3f} | {n['rms_radius']:.3f} | {n['mean_pdist']:.3f} |")
    w("\n### domain gap")
    w("| pair | sub | ‖μA−μB‖ | within pdist | ratio | offset share of paired sq | cos(μA,μB) | clf raw | clf unit-norm | clf norm-only | clf centroid-dir |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, o in res["domain"]["pairs"].items():
        a, b, sub = k.split("|")
        w(f"| {a} vs {b} | {sub} | {o['centroid_dist']:.3f} | {o['within_mean_pdist']:.3f} | {o['ratio_centroid_over_pdist']:.3f} | "
          f"{o['offset_share_of_paired_sqdist']:.3f} | {o['centroid_cos']:.3f} | {o['clf_raw']:.4f} | {o['clf_unitnorm']:.4f} | "
          f"{o['clf_norm_only']:.4f} | {o['clf_centroid_dir']:.4f} |")
    for view in ("raw", "resid"):
        C = res["class"][view]
        for lname in ("shape", "color"):
            w(f"\n### class geometry [{view}] — {lname}  (NCC acc / linear probe acc; Fisher [perm])")
            w("| rep | visible | late | vis→late | late→vis | Fisher vis | Fisher late |\n|---|---:|---:|---:|---:|---:|---:|")
            for r in REPS:
                v, l = C[f"{lname}|{r}|visible"], C[f"{lname}|{r}|late"]
                vl, lv = C[f"{lname}|{r}|visible->late"], C[f"{lname}|{r}|late->visible"]
                w(f"| {r} | {100*v['ncc']:.1f} / {100*v['linprobe']:.1f} | {100*l['ncc']:.1f} / {100*l['linprobe']:.1f} | "
                  f"{100*vl['ncc']:.1f} / {100*vl['linprobe']:.1f} | {100*lv['ncc']:.1f} / {100*lv['linprobe']:.1f} | "
                  f"{v['fisher']:.4f} [{v['fisher_perm']:.4f}] | {l['fisher']:.4f} [{l['fisher_perm']:.4f}] |")
            w(f"\nper condition NCC / linprobe [{view}] ({lname})\n| rep | " + " | ".join(CONDS) + " |\n|---|" + "---:|" * 6)
            for r in REPS:
                w(f"| {r} | " + " | ".join(f"{100*C[f'{lname}|{r}|{c}']['ncc']:.1f} / {100*C[f'{lname}|{r}|{c}']['linprobe']:.1f}" for c in CONDS) + " |")
            w(f"\nper motion transfer visible→occlusion NCC / linprobe [{view}] ({lname}); (within occ)\n| rep | static | flat | ramp |\n|---|---:|---:|---:|")
            for r in REPS:
                cells = []
                for mo in ("static", "flat", "ramp"):
                    s_, d_ = MOTION[mo]
                    o = C[f"{lname}|{r}|{s_}->{d_}"]; o2 = C[f"{lname}|{r}|{d_}"]
                    cells.append(f"{100*o['ncc']:.1f} / {100*o['linprobe']:.1f} ({100*o2['ncc']:.1f} / {100*o2['linprobe']:.1f})")
                w(f"| {r} | " + " | ".join(cells) + " |")
    for view in ("raw", "resid"):
        S = res["subspace"][view]
        for lname in ("shape", "color"):
            rnd = S["random"][lname]
            w(f"\n### subspace [{view}] — {lname} (rank {rnd['rank']}; random mean {rnd['mean']:.4f}, p95 {rnd['p95']:.4f}, analytic r/d {rnd['analytic']:.4f})")
            w("| A | B | mean cos² | captured A in B | captured B in A | perm shared (mean/p95) | perm indep (mean/p95) | angles (deg) |\n|---|---|---:|---:|---:|---:|---:|---|")
            for k, o in S.items():
                if not k.startswith(lname + "|") or k.endswith("|self"):
                    continue
                _, A, B = k.split("|")
                ps = f"{o['perm_shared']['mean']:.3f} / {o['perm_shared']['p95']:.3f}" if "perm_shared" in o else "—"
                w(f"| {A} | {B} | {o['mean_cos2']:.3f} | {o['captured_A_in_B']:.3f} | {o['captured_B_in_A']:.3f} | {ps} | "
                  f"{o['perm_indep']['mean']:.3f} / {o['perm_indep']['p95']:.3f} | {', '.join(f'{x:.0f}' for x in o['angles_deg'])} |")
            w("split-half ceiling cos² / captured: " + " · ".join(f"{r}:{s} {S[f'{lname}|{r}|{s}|self']['split_half']:.3f}/{S[f'{lname}|{r}|{s}|self']['split_half_captured']:.3f}" for r in REPS for s in ("visible", "late")))
            w(f"\ncross-fitted (disjoint block halves) [{view}] {lname}\n| A | B | xfit cos² | / ceiling | xfit captured | / ceiling | perm cos² (mean/p95) | perm captured (mean/p95) |\n|---|---|---:|---:|---:|---:|---:|---:|")
            for k, o in S.items():
                if not k.startswith(lname + "|") or k.endswith("|self"):
                    continue
                _, A, B = k.split("|"); q = o["xfit_perm"]
                w(f"| {A} | {B} | {o['xfit_cos2']:.3f} | {o['xfit_cos2_norm']:.3f} | {o['xfit_captured']:.3f} | {o['xfit_captured_norm']:.3f} | "
                  f"{q['cos2_mean']:.3f} / {q['cos2_p95']:.3f} | {q['captured_mean']:.3f} / {q['captured_p95']:.3f} |")
    w("\n### PCA")
    j = res["pca"]["joint"]
    w(f"joint: explained top5 {[round(x, 4) for x in j['explained_ratio'][:5]]}, PR {j['participation_ratio']:.2f}, between-rep var share {j['between_rep_share']:.4f}")
    for kk in ("eta2_rep", "eta2_cond", "eta2_env"):
        w(f"  {kk} PC1-6 {[round(x, 3) for x in j[kk][:6]]}")
    for view in ("raw", "resid"):
        w(f"\n[{view}]\n| rep | sub | PR | n PCs 50% | 90% | 95% | PC1 ratio |\n|---|---|---:|---:|---:|---:|---:|")
        for k, e in res["pca"]["explained"][view].items():
            r, sub = k.split("|")
            w(f"| {r} | {sub} | {e['participation_ratio']:.2f} | {e['n50']} | {e['n90']} | {e['n95']} | {e['ratio_top50'][0]:.4f} |")
        w(f"per-rep PCA eta2 (PC1, PC2) [{view}]")
        for k, e in res["pca"]["per_rep"][view].items():
            w(f"  {k}: " + ", ".join(f"{kk}={[round(x, 3) for x in vv]}" for kk, vv in e.items() if kk.startswith("eta2")))
    w("\n### t-SNE")
    w(json.dumps(res["tsne"], indent=1))


# ============================================================================= main
def var_share(X, lab):
    _, inv = np.unique(lab, return_inverse=True)
    Y = onehot(inv, int(inv.max()) + 1); cnt = Y.sum(0)
    M = (Y.T @ X) / cnt[:, None]; mu = X.mean(0)
    return float((cnt * ((M - mu) ** 2).sum(1)).sum() / ((X - mu) ** 2).sum())


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pca", type=int, default=1000, help="joint PCA · 그림 부표본: 표현 × 조건당 clip 수")
    ap.add_argument("--n-tsne", type=int, default=300, help="t-SNE 부표본: 표현 × 조건당 clip 수")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tables", action="store_true", help="results.json 에서 표만 다시 찍는다")
    a = ap.parse_args()
    out = Path(a.out)
    if a.tables:
        tables(json.loads((out / "results.json").read_text())); return
    out.mkdir(parents=True, exist_ok=True)
    DEV = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    X, meta = load()
    sk = meta["sym_k"].astype(str); cond = meta["condition"].astype(str); env = meta["env"].astype(str)
    vis = sk == "0"
    assert set(cond[vis]) == {c for c in CONDS if "visible" in c} and set(cond[~vis]) == {c for c in CONDS if "occlusion" in c}
    assert (meta["occ_timing"][~vis] == "late").all()
    train = repo_split(meta)
    masks = {"all": np.ones(N, bool), "visible": vis, "late": ~vis, **{c: cond == c for c in CONDS}}
    blocks = meta["block_id"].astype(str)
    shape, color = meta["shape_pre"].astype(str), meta["color_pre"].astype(str)
    shape_classes = sorted(set(shape))
    Xg = {r: torch.as_tensor(X[r], dtype=F64, device=DEV) for r in REPS}
    # 장면 칸 (env × condition × k) 평균 제거 — 칸 평균은 **train clip 으로만** 추정해 모든 clip 에서 뺀다 (클래스 라벨 안 씀)
    cell = np.array([f"{e}|{c}|{k}" for e, c, k in zip(env, cond, sk)])
    ucell, cinv = np.unique(cell, return_inverse=True)
    Ytr = onehot(cinv[train], len(ucell)); ccnt = Ytr.sum(0)
    assert (ccnt > 0).all()
    Xr = {}
    for r in REPS:
        M = (Ytr.T @ Xg[r][torch.as_tensor(train, device=DEV)]) / ccnt[:, None]
        Xr[r] = Xg[r] - M[torch.as_tensor(cinv, device=DEV)]
    VIEWS = {"raw": Xg, "resid": Xr}
    res = dict(meta=dict(n_clip=N, D=D, cache=str(CACHE), device=a.device, seed=a.seed,
                         n_visible=int(vis.sum()), n_late=int((~vis).sum()),
                         split=dict(n_train_clip=int(train.sum()), n_test_clip=int((~train).sum()),
                                    n_train_block=len(set(blocks[train])), n_test_block=len(set(blocks[~train])),
                                    per_condition_train_frac={c: float(train[cond == c].mean()) for c in CONDS}),
                         resid=dict(cells=len(ucell), def_="env x condition x sym_k cell mean (train clips only) subtracted",
                                    train_clips_per_cell_min=int(ccnt.min()), train_clips_per_cell_max=int(ccnt.max())),
                         reps=dict(z_all="z context 8 tubelets mean (context encoder, 16 context samples, LN)",
                                   p_fut="p future 8 tubelets mean (predictor, not re-LN)",
                                   h_fut="h tubelets 8-15 mean (target encoder, full 32-sample window, LN)",
                                   h_ctx="h tubelets 0-7 mean (target encoder, full window, LN; same frames as z)")))
    log(f"data {N} clip · visible {vis.sum()} late {(~vis).sum()} · train {train.sum()} test {(~train).sum()} · cells {len(ucell)}")
    P = {"shape_classes": shape_classes}

    # ---------------------------------------------------------------- 0 variance decomposition
    factors = dict(env=env, condition=cond, sym_k=sk, cond_k=np.char.add(cond, sk), cell=cell, shape=shape, color=color,
                   travel_dir=meta["travel_dir"].astype(str))
    res["variance"] = {}
    for r in REPS:
        for sub in ("all", "visible", "late"):
            m = masks[sub]; mt = torch.as_tensor(m, device=DEV)
            o = {f: var_share(Xg[r][mt], lab[m]) for f, lab in factors.items()}
            o["resid_total_over_raw"] = float(((Xr[r][mt] - Xr[r][mt].mean(0)) ** 2).sum() / ((Xg[r][mt] - Xg[r][mt].mean(0)) ** 2).sum())
            o["resid_shape"] = var_share(Xr[r][mt], shape[m]); o["resid_color"] = var_share(Xr[r][mt], color[m])
            o["resid_travel_dir"] = var_share(Xr[r][mt], factors["travel_dir"][m])
            res["variance"][f"{r}|{sub}"] = o
    log("variance " + "  ".join(f"{k}: cell={v['cell']:.3f} shape={v['shape']:.3f} rshape={v['resid_shape']:.3f}" for k, v in res["variance"].items() if k.endswith("|all")))

    # ---------------------------------------------------------------- 1 PCA
    pca_idx = np.sort(np.concatenate([rng.choice(np.where(m)[0], a.n_pca, replace=False) for m in (vis, ~vis)]))
    pit = torch.as_tensor(pca_idx, device=DEV)
    Jx = torch.cat([Xg[r][pit] for r in MAIN])
    Jrep = np.repeat(MAIN, len(pca_idx)); Jvis = np.tile(vis[pca_idx], len(MAIN)); Jenv = np.tile(env[pca_idx], len(MAIN))
    ev, V = eig_cov(Jx)
    Js = (Jx - Jx.mean(0)) @ V[:, :10]
    mu = Jx.mean(0)
    between = sum(float(((Jx[torch.as_tensor(Jrep == r, device=DEV)].mean(0) - mu) ** 2).sum()) * (Jrep == r).sum() for r in MAIN)
    total = float(((Jx - mu) ** 2).sum())
    res["pca"] = {"joint": dict(n_per_rep_cond=a.n_pca, explained_ratio=(ev[:10] / ev.sum()).tolist(),
                                participation_ratio=pr(ev), between_rep_share=between / total,
                                eta2_rep=eta2(Js, Jrep), eta2_cond=eta2(Js, Jvis), eta2_env=eta2(Js, Jenv))}
    P.update(joint_scores=Js[:, :4].cpu().numpy(), joint_rep=Jrep, joint_vis=Jvis, joint_env=Jenv)
    res["pca"]["per_rep"], P["rep_scores"] = {"raw": {}, "resid": {}}, {"raw": {}, "resid": {}}
    res["pca"]["explained"] = {"raw": {}, "resid": {}}
    for view, XV in VIEWS.items():
        for r in REPS:
            ev, V = eig_cov(XV[r])
            S = (XV[r] - XV[r].mean(0)) @ V[:, :5]
            o = dict(explained_ratio_top5=(ev[:5] / ev.sum()).tolist())
            for sub in ("visible", "late"):
                mk = torch.as_tensor(masks[sub], device=DEV)
                for lname, lab in (("shape", shape), ("color", color), ("env", env), ("cell", cell)):
                    o[f"eta2_{lname}_{sub}"] = eta2(S[mk][:, :2], lab[masks[sub]])
            res["pca"]["per_rep"][view][r] = o
            P["rep_scores"][view][r] = S[pit, :2].cpu().numpy()
            for sub in ("visible", "late", "all"):
                ev, _ = eig_cov(XV[r][torch.as_tensor(masks[sub], device=DEV)])
                ratio = (ev / ev.sum()).cpu().numpy(); cum = np.cumsum(ratio)
                res["pca"]["explained"][view][f"{r}|{sub}"] = dict(
                    n=int(masks[sub].sum()), ratio_top50=ratio[:50].tolist(), participation_ratio=pr(ev),
                    n50=int(np.searchsorted(cum, .5) + 1), n90=int(np.searchsorted(cum, .9) + 1), n95=int(np.searchsorted(cum, .95) + 1))
    P.update(rep_vis=vis[pca_idx], rep_shape=shape[pca_idx], rep_color=color[pca_idx])
    log("PCA done " + "  ".join(f"{v}:{k}:PR={e['participation_ratio']:.1f}" for v in VIEWS for k, e in res["pca"]["explained"][v].items() if k.endswith("|all")))

    # ---------------------------------------------------------------- 2 t-SNE
    val, P["tsne_val"] = validate_tsne(a.seed)
    log(f"t-SNE validation: clusters knn_in={val['clusters']['knn10_input']:.3f} knn_tsne={val['clusters']['knn10_tsne']:.3f} "
        f"sep={val['clusters']['sep_ratio_tsne']:.2f} ent_err={val['clusters']['max_entropy_err']:.2e} | "
        f"overlap knn_in={val['overlapping']['knn10_input']:.3f} knn_tsne={val['overlapping']['knn10_tsne']:.3f} | "
        f"null knn_tsne={val['null']['knn10_tsne']:.3f} (chance {val['null']['chance']:.3f})")
    ts_idx = np.sort(np.concatenate([rng.choice(np.intersect1d(pca_idx, np.where(m)[0]), a.n_tsne, replace=False) for m in (vis, ~vis)]))
    Tx = np.concatenate([X[r][ts_idx] for r in MAIN])
    Trep = np.repeat(MAIN, len(ts_idx)); Tvis = np.tile(vis[ts_idx], len(MAIN))
    Tsh, Tco, Tenv, Tcell = (np.tile(v[ts_idx], len(MAIN)) for v in (shape, color, env, cell))
    Tgrp = np.array([f"{r}|{'vis' if v else 'late'}" for r, v in zip(Trep, Tvis)])
    E, info = tsne(Tx, seed=a.seed)
    joint = dict(n_per_rep_cond=a.n_tsne, **info)
    for nm, lab in (("rep_cond", Tgrp), ("rep", Trep), ("shape", Tsh), ("color", Tco), ("env", Tenv), ("cell", Tcell)):
        joint[f"knn10_tsne_{nm}"] = knn_acc(E, lab); joint[f"knn10_input_{nm}"] = knn_acc(Tx, lab)
    res["tsne"] = dict(validation=val, joint=joint, per_rep={"raw": {}, "resid": {}})
    P.update(tsne_joint=E, tsne_rep=Trep, tsne_vis=Tvis, tsne_shape=Tsh, tsne_color=Tco, tsne_env=Tenv,
             tsne_per_rep={"raw": {}, "resid": {}}, tsne_rep_shape=shape[ts_idx], tsne_rep_vis=vis[ts_idx])
    for view, XV in VIEWS.items():
        for r in MAIN:
            Xin = XV[r][torch.as_tensor(ts_idx, device=DEV)].cpu().numpy()
            Er, info = tsne(Xin, seed=a.seed)
            o = dict(info)
            for nm, lab in (("shape", shape[ts_idx]), ("color", color[ts_idx]), ("vis_late", vis[ts_idx].astype(int)),
                            ("env", env[ts_idx]), ("cell", cell[ts_idx])):
                o[f"knn10_tsne_{nm}"] = knn_acc(Er, lab); o[f"knn10_input_{nm}"] = knn_acc(Xin, lab)
            res["tsne"]["per_rep"][view][r] = o
            P["tsne_per_rep"][view][r] = Er
    log("t-SNE done " + json.dumps({v: {k: {kk: round(vv, 3) for kk, vv in o.items() if kk.startswith("knn10_tsne")} for k, o in d.items()}
                                    for v, d in res["tsne"]["per_rep"].items()}))

    # ---------------------------------------------------------------- 3 CKA · 4 domain · 5 class · 6 subspace
    res["cka"] = {v: cka_section(XV, masks, blocks, a.n_boot, 20, rng) for v, XV in VIEWS.items()}
    res["domain"] = domain_section(Xg, masks, train)
    res["class"] = {v: class_section(XV, meta, masks, train, rng) for v, XV in VIEWS.items()}
    res["subspace"] = {v: subspace_section(XV, meta, masks, train, rng, a.n_perm) for v, XV in VIEWS.items()}

    res["meta"]["runtime_min"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(r4(res), indent=1, ensure_ascii=False))
    log(f"→ {out / 'results.json'}")

    plt = setup_mpl()
    fig_pca_joint(plt, P, res, out); fig_pca_by_rep(plt, P, res, out); fig_explained(plt, res, out)
    fig_tsne(plt, P, out); fig_cka(plt, res, out)
    log(f"figures → {out}  ({(time.time()-t0)/60:.1f} min)")
    tables(json.loads((out / "results.json").read_text()))


DEV = torch.device("cpu")
if __name__ == "__main__":
    main()
