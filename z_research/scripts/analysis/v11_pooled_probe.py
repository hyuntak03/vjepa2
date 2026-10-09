#!/usr/bin/env python3
"""IntPhysGen v11 — 평균 풀링 특징 위 **선형 (multinomial logistic) probe** 로 shape_pre · color_pre 를 읽는다.

입력 (v11_pooled_features.py 산출, 가능 변이 · 물체 있는 clip 9,408 개 · block 5,376 개):
  z (n, 8, D)   context encoder, 문맥 16 샘플만 (ctx_masked = predictor 입력), LN, 튜블릿 평균
  p (n, 8, D)   predictor 출력, 미래 8 튜블릿 (LN(target) 추정치, 다시 LN 하지 않음)
  h (n, 16, D)  target encoder 창 전체 32 샘플, LN — 튜블릿 0-7 문맥 / 8-15 미래. 대조군
  hidden (n, 32) 가려진 샘플. late k 는 샘플 16-k .. 16+k-1

규약
  split : block 단위, train_frac 0.5, seed 0, stratify_by condition — 레포 하네스의
          `evals.world_model_analysis.data.WMADataset._split` 을 그대로 불러 쓴다 (같은 split).
          λ 선택용 hold-out = train block 의 20 % (같은 함수, train_frac 0.8, seed 1, condition 층화).
  probe : 표준화 (source train 평균·표준편차) + multinomial logistic regression,
          손실 = mean CE + λ/2·||W||² (bias 는 벌점 없음), full-batch L-BFGS (strong Wolfe, float64).
          λ ∈ {1e-4, 1e-3, 1e-2, 1e-1} 를 hold-out 정확도로 고르고 (동률은 hold-out NLL) train 전체로 다시 맞춘다.
  CI    : test block bootstrap 1,000 회 (block 을 복원추출, 정확도 = Σ정답/Σclip), 95 % percentile.

실험 (타깃 shape / color 따로)
  (2)  z_all  = z 문맥 8 튜블릿 평균. train 전체 (visible + late) 로 학습.
  (3)  z_hid  = 완전히 가려진 문맥 튜블릿 (두 샘플 모두 hidden) 평균, late k>=2 만. 대조 (3c1) visible t7 · t6-7,
       (3c2) late k>=2 의 가려지지 않은 문맥 튜블릿 (hidden 샘플이 하나도 없는 것) 평균.
  (4)  (2) 머리 → z_hid (late k>=2 test) / (4b) (3) 머리 → z_all (같은 clip).
  (5)  p_fut  = p 미래 8 튜블릿 평균. visible · late (k>=1) 머리 따로 + 교차.
  (5b) 미래 튜블릿 t 하나씩 (p[:,t], h[:,8+t]).
  (6)  encoder 머리 → p_fut : raw / mean-shift / Procrustes (train visible clip 짝, 라벨 없음). 대조 (6h) h_fut 머리.
       추가: Procrustes 를 train late 짝 · train 전체 짝으로 맞춘 것 (procrustes_fit_late / _all),
             (5x) 교차 이식에 라벨 없는 평균 이동 · 재표준화 (조건이 다르면 clip 짝이 없어 Procrustes 불가),
             (5c) 같은 visible↔late 교차를 z_all · h_fut 에서, (4c) (4) 의 풀링 대조 (visible t7 · t6-7 에 같은 머리),
             Procrustes 짝 섞기 대조, (6e) encoder ↔ encoder (h_fut ↔ z_all) 에 같은 Procrustes.
  위생 : train/test block 겹침 0, split 별 클래스 수, (2)·(5) 라벨 섞기 대조.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/v11_pooled_probe.py --device cuda:0          # 전부 (수 분)
  $P z_research/scripts/analysis/v11_pooled_probe.py --plot-only              # results.json 에서 그림만
  $P z_research/scripts/analysis/v11_pooled_probe.py --device cuda:0 --lams 0.1 --tag lam0.1 --skip-tubelet
                                                                              # λ 민감도 → results_lam0.1.json
  $P z_research/scripts/analysis/v11_pooled_probe.py --tables                 # PROBE.md 의 표 (results*.json 에서)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
# 2026-09-25: 다른 predictor 의 p 로 돌리려고 환경변수로 덮을 수 있게 했다 (기본값 그대로). TrainingEffects/README.md
CACHE = Path(__import__("os").environ.get("V11P_CACHE", "/local_datasets/world/world_analysis/cache/v11_pooled_vith"))
OUT = Path(__import__("os").environ.get("V11P_PROBE_OUT", str(ROOT / "z_research/RollOutV3/figures/v11_probe")))
LAMS = [1e-4, 1e-3, 1e-2, 1e-1]
TARGETS = {"shape": "shape_pre", "color": "color_pre"}
MOTION = {"static": "static", "moving_flat": "flat", "moving": "ramp"}
MOTIONS = ["static", "flat", "ramp"]
NBOOT = 1000
SKIP_TUBELET = False
C_BLUE, C_ORANGE, C_GREEN = "#2a78d6", "#eb6834", "#1baf7a"


# ----------------------------------------------------------------------------- data
def load():
    m = np.load(CACHE / "meta.npz", allow_pickle=False)
    meta = {k: m[k] for k in m.files}
    z = np.load(CACHE / "z.npy").astype(np.float32)
    p = np.load(CACHE / "p.npy").astype(np.float32)
    h = np.load(CACHE / "h.npy").astype(np.float32)
    assert z.shape[1:] == (8, 1280) and p.shape[1:] == (8, 1280) and h.shape[1:] == (16, 1280)
    return meta, z, p, h


def repo_split(meta, frac, seed, subset=None):
    """레포 하네스의 split 함수를 그대로 부른다 → bool (train)."""
    from evals.world_model_analysis.data import WMADataset
    idx = np.arange(len(meta["video_id"])) if subset is None else np.where(subset)[0]
    rows = [dict(video_id=str(meta["video_id"][i]), block_id=str(meta["block_id"][i]),
                 condition=str(meta["condition"][i])) for i in idx]
    cfg = dict(mode="ratio", group_by="block_id", train_frac=frac, seed=seed, stratify_by="condition")
    s = WMADataset._split(rows, cfg)
    out = np.zeros(len(meta["video_id"]), bool)
    out[idx] = [s[r["video_id"]] == "train" for r in rows]
    return out


# ----------------------------------------------------------------------------- probe
class Head:
    def __init__(self, mu, sd, W, b, lam, info):
        self.mu, self.sd, self.W, self.b, self.lam, self.info = mu, sd, W, b, lam, info

    def logits(self, X):
        return ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.W + self.b

    def predict(self, X):
        return self.logits(X).argmax(1)


def _stats(X):
    mu = X.mean(0)
    sd = X.std(0)
    return mu, np.where(sd < 1e-6, 1.0, sd)


def fit_lr(Xs, y, n_cls, lam, dev, max_steps=12, iters_per_step=500, gtol=1e-6):
    X = torch.as_tensor(Xs, dtype=torch.float64, device=dev)
    Y = torch.as_tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(X.shape[1], n_cls, dtype=torch.float64, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters_per_step, history_size=50,
                            tolerance_grad=gtol, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, Y) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss

    n_it = 0
    for _ in range(max_steps):
        opt.step(closure)
        n_it = opt.state[opt._params[0]]["n_iter"]
        loss = closure()
        g = max(W.grad.abs().max().item(), b.grad.abs().max().item())
        if g < gtol * 10:
            break
    with torch.no_grad():
        tr_acc = ((X @ W + b).argmax(1) == Y).double().mean().item()
    return (W.detach().cpu().numpy(), b.detach().cpu().numpy(),
            dict(loss=float(loss.item()), grad_max=float(g), n_iter=int(n_it), converged=bool(g < gtol * 10),
                 train_acc=tr_acc))


def train_head(X, y, tr, inner_fit, n_cls, dev):
    """X (n, D) 전체, tr = 이 머리의 train 마스크. inner_fit = 전역 hold-out 분할 (True = fit 부분)."""
    fit, hold = tr & inner_fit, tr & ~inner_fit
    assert fit.sum() > 0 and hold.sum() > 0
    mu, sd = _stats(X[fit])
    Xf, Xh = (X[fit] - mu) / sd, (X[hold] - mu) / sd
    sel = {}
    for lam in LAMS:
        W, b, info = fit_lr(Xf, y[fit], n_cls, lam, dev)
        lg = Xh @ W + b
        lg = lg - lg.max(1, keepdims=True)
        nll = float(-(lg[np.arange(len(lg)), y[hold]] - np.log(np.exp(lg).sum(1))).mean())
        sel[lam] = dict(holdout_acc=float((lg.argmax(1) == y[hold]).mean()), holdout_nll=nll, **info)
    best = max(LAMS, key=lambda l: (sel[l]["holdout_acc"], -sel[l]["holdout_nll"]))
    mu, sd = _stats(X[tr])
    W, b, info = fit_lr((X[tr] - mu) / sd, y[tr], n_cls, best, dev)
    info = dict(lam=best, n_train=int(tr.sum()), n_fit=int(fit.sum()), n_holdout=int(hold.sum()),
                selection={f"{l:g}": v for l, v in sel.items()}, refit=info)
    return Head(mu, sd, W, b, best, info)


# ----------------------------------------------------------------------------- eval
def acc_ci(correct, block, y, mask, n_cls, seed=0):
    c = correct[mask].astype(np.float64)
    if len(c) == 0:
        return None
    ub, inv = np.unique(block[mask], return_inverse=True)
    cs, ns = np.bincount(inv, weights=c), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ub), size=(NBOOT, len(ub)))
    boot = cs[ix].sum(1) / ns[ix].sum(1)
    cnt = np.bincount(y[mask], minlength=n_cls)
    return dict(acc=round(float(c.mean()) * 100, 2), lo=round(float(np.percentile(boot, 2.5)) * 100, 2),
                hi=round(float(np.percentile(boot, 97.5)) * 100, 2), n=int(len(c)), n_blocks=int(len(ub)),
                chance=round(100 / n_cls, 2), majority=round(float(cnt.max() / cnt.sum()) * 100, 2))


def evaluate(head, X, y, block, groups, n_cls):
    pred = head.predict(X)
    corr = pred == y
    return {g: acc_ci(corr, block, y, m, n_cls) for g, m in groups.items()}, corr


def diff_ci(c1, m1, c2, m2, block, paired, seed=0):
    """정확도 차 (1 − 2), block bootstrap. paired=True 면 같은 clip 집합 (m1 == m2) 에서 같은 block 추출."""
    rng = np.random.default_rng(seed)

    def per_block(c, m):
        ub, inv = np.unique(block[m], return_inverse=True)
        return ub, np.bincount(inv, weights=c[m].astype(float)), np.bincount(inv).astype(float)
    if paired:
        assert (m1 == m2).all()
        ub, a, n = per_block(c1, m1)
        _, bb, _ = per_block(c2, m2)
        ix = rng.integers(0, len(ub), size=(NBOOT, len(ub)))
        boot = (a[ix].sum(1) - bb[ix].sum(1)) / n[ix].sum(1)
    else:
        u1, a1, n1 = per_block(c1, m1)
        u2, a2, n2 = per_block(c2, m2)
        i1 = rng.integers(0, len(u1), size=(NBOOT, len(u1)))
        i2 = rng.integers(0, len(u2), size=(NBOOT, len(u2)))
        boot = a1[i1].sum(1) / n1[i1].sum(1) - a2[i2].sum(1) / n2[i2].sum(1)
    d = c1[m1].mean() - c2[m2].mean()
    return dict(diff=round(float(d) * 100, 2), lo=round(float(np.percentile(boot, 2.5)) * 100, 2),
                hi=round(float(np.percentile(boot, 97.5)) * 100, 2), paired=paired)


# ----------------------------------------------------------------------------- main
def run(dev):
    t0 = time.time()
    meta, z, p, h = load()
    n = len(meta["video_id"])
    block = meta["block_id"].astype(str)
    k = meta["sym_k"].astype(int)
    vis, late, late2 = k == 0, k >= 1, k >= 2
    mot = np.array([MOTION[x] for x in meta["motion"]])
    hid = meta["hidden"]
    full_hid = hid[:, 0:16:2] & hid[:, 1:16:2]                 # (n, 8) 문맥 튜블릿의 두 샘플 모두 가려짐
    any_hid = hid[:, 0:16:2] | hid[:, 1:16:2]

    tr = repo_split(meta, 0.5, 0)
    te = ~tr
    inner = repo_split(meta, 0.8, 1, subset=tr)               # train 안 20 % hold-out (block 단위)

    # ---- 특징
    z_all = z.mean(1)
    p_fut = p.mean(1)
    h_fut = h[:, 8:].mean(1)
    cnt = full_hid.sum(1)
    z_hid = np.where(cnt[:, None] > 0, (z * full_hid[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], np.nan)
    vis_t = ~any_hid
    z_visctx = (z * vis_t[..., None]).sum(1) / vis_t.sum(1)[:, None]
    z_t7, z_t67 = z[:, 7], z[:, 6:8].mean(1)

    # ---- 위생
    rule = {kk: sorted({tuple(np.where(full_hid[i])[0]) for i in np.where(k == kk)[0]}) for kk in range(5)}
    rule_v = {kk: sorted({tuple(np.where(vis_t[i])[0]) for i in np.where(k == kk)[0]}) for kk in range(5)}
    assert rule[2] == [(7,)] and rule[3] == [(7,)] and rule[4] == [(6, 7)] and rule[1] == [()], rule
    san = dict(
        n_clip=int(n), n_block=int(len(np.unique(block))),
        n_train=int(tr.sum()), n_test=int(te.sum()),
        n_train_blocks=int(len(np.unique(block[tr]))), n_test_blocks=int(len(np.unique(block[te]))),
        block_overlap_train_test=int(len(set(block[tr]) & set(block[te]))),
        block_overlap_fit_holdout=int(len(set(block[tr & inner]) & set(block[tr & ~inner]))),
        n_fit=int((tr & inner).sum()), n_holdout=int((tr & ~inner).sum()),
        train_frac_by_condition={c: round(float(tr[meta["condition"] == c].mean()), 4)
                                 for c in sorted(set(meta["condition"]))},
        train_frac_by_k={str(kk): round(float(tr[k == kk].mean()), 4) for kk in range(5)},
        fully_hidden_ctx_tubelets_by_k={str(kk): [[int(x) for x in t] for t in v] for kk, v in rule.items()},
        visible_ctx_tubelets_by_k={str(kk): [[int(x) for x in t] for t in v] for kk, v in rule_v.items()},
        feature_norm_mean=dict(z_all=float(np.linalg.norm(z_all, axis=1).mean()),
                               p_fut=float(np.linalg.norm(p_fut, axis=1).mean()),
                               h_fut=float(np.linalg.norm(h_fut, axis=1).mean())),
    )
    assert san["block_overlap_train_test"] == 0 and san["block_overlap_fit_holdout"] == 0
    print(f"[split] train {san['n_train']} / test {san['n_test']} clip, 겹침 0, {time.time()-t0:.0f}s", flush=True)

    def G_all():
        g = {"all": te, "visible": te & vis, "late": te & late}
        g.update({f"late_k{kk}": te & (k == kk) for kk in range(1, 5)})
        for v, vm in (("visible", vis), ("late", late)):
            g.update({f"{v}/{mm}": te & vm & (mot == mm) for mm in MOTIONS})
        return g

    def G_vis():
        return {"visible": te & vis, **{f"visible/{mm}": te & vis & (mot == mm) for mm in MOTIONS}}

    def G_late(kmin=1):
        lm = k >= kmin
        tag = "late" if kmin == 1 else f"late_k>={kmin}"
        g = {tag: te & lm}
        g.update({f"late_k{kk}": te & (k == kk) for kk in range(kmin, 5)})
        g.update({f"{tag}/{mm}": te & lm & (mot == mm) for mm in MOTIONS})
        return g

    R = dict(meta=dict(cache=str(CACHE), script="z_research/scripts/analysis/v11_pooled_probe.py", lams=LAMS,
                       n_boot=NBOOT, device=str(dev),
                       split="WMADataset._split(block_id, train_frac 0.5, seed 0, stratify_by condition)",
                       holdout="WMADataset._split on train rows (train_frac 0.8, seed 1, stratify_by condition)",
                       loss="mean CE + lam/2 ||W||^2, LBFGS float64, standardize with source-train stats"),
             sanity=san)
    diffs = {}
    for tname, col in TARGETS.items():
        classes = sorted(set(meta[col]))
        y = np.searchsorted(classes, meta[col])
        C = len(classes)
        T = R[tname] = dict(classes=classes)
        T["class_counts"] = {s: {v: {cl: int(((y == i) & sm & vm).sum()) for i, cl in enumerate(classes)}
                                 for v, vm in (("all", np.ones(n, bool)), ("visible", vis), ("late", late),
                                               ("late_k>=2", late2))}
                             for s, sm in (("train", tr), ("test", te))}
        H = lambda X, trm: train_head(X, y, trm, inner, C, dev)                       # noqa: E731
        E = lambda hd, X, g: evaluate(hd, X, y, block, g, C)                          # noqa: E731

        def ytr_shuffled(trm):
            ys = y.copy()
            ix = np.where(trm)[0]
            ys[ix] = np.random.default_rng(0).permutation(y[ix])
            return ys

        def shuffled(X, trm, g):
            ys = ytr_shuffled(trm)
            hd = train_head(X, ys, trm, inner, C, dev)
            return dict(head=hd.info, test=evaluate(hd, X, y, block, g, C)[0])

        # (2) --------------------------------------------------------------
        h2 = H(z_all, tr)
        ev, c2 = E(h2, z_all, G_all())
        T["exp2_z_all"] = dict(head=h2.info, test=ev)
        T["exp2_shuffled"] = shuffled(z_all, tr, {"all": te, "visible": te & vis, "late": te & late})
        diffs[f"{tname}/exp2_visible_minus_late"] = diff_ci(c2, te & vis, c2, te & late, block, paired=False)
        print(f"[{tname}] (2) all {ev['all']['acc']} vis {ev['visible']['acc']} late {ev['late']['acc']}  "
              f"λ={h2.lam:g}  {time.time()-t0:.0f}s", flush=True)

        # (3) --------------------------------------------------------------
        h3 = H(z_hid, tr & late2)
        ev3, c3 = E(h3, z_hid, G_late(2))
        T["exp3_z_hid"] = dict(head=h3.info, test=ev3)
        h3a = H(z_t7, tr & vis)
        T["exp3c1_visible_t7"] = dict(head=h3a.info, test=E(h3a, z_t7, G_vis())[0])
        h3b = H(z_t67, tr & vis)
        T["exp3c1_visible_t67"] = dict(head=h3b.info, test=E(h3b, z_t67, G_vis())[0])
        h3c = H(z_visctx, tr & late2)
        ev3c, c3c = E(h3c, z_visctx, G_late(2))
        T["exp3c2_late_visible_ctx"] = dict(head=h3c.info, test=ev3c)
        diffs[f"{tname}/exp3_zhid_minus_visctx"] = diff_ci(c3, te & late2, c3c, te & late2, block, paired=True)
        print(f"[{tname}] (3) z_hid {ev3['late_k>=2']['acc']}  c1 t7 {T['exp3c1_visible_t7']['test']['visible']['acc']}"
              f"  c2 {ev3c['late_k>=2']['acc']}  {time.time()-t0:.0f}s", flush=True)

        # (4) --------------------------------------------------------------
        ev4, c4 = E(h2, z_hid, G_late(2))
        ev4b, c4b = E(h3, z_all, G_late(2))
        ev4ref, c4ref = E(h2, z_all, G_late(2))
        T["exp4_zhead_on_zhid"] = dict(test=ev4)
        T["exp4b_zhidhead_on_zall"] = dict(test=ev4b)
        T["exp4_ref_zhead_on_zall_same_clips"] = dict(test=ev4ref)
        # (4c, 추가 대조) (4) 는 '가려짐' 과 '튜블릿 1~2 개 풀링 vs 8 개 평균' 이 섞인다 → 같은 풀링을 visible 에 건다
        T["exp4c_pooling_controls"] = dict(
            zhead_on_visible_t7=E(h2, z_t7, G_vis())[0],
            zhead_on_visible_t67=E(h2, z_t67, G_vis())[0],
            zhead_on_late_visctx=E(h2, z_visctx, G_late(2))[0],
            zhidhead_on_visible_t7=E(h3, z_t7, G_vis())[0],
            zhidhead_on_visible_t67=E(h3, z_t67, G_vis())[0])
        diffs[f"{tname}/exp4_transfer_minus_inset"] = diff_ci(c4, te & late2, c3, te & late2, block, paired=True)
        diffs[f"{tname}/exp4b_transfer_minus_inset"] = diff_ci(c4b, te & late2, c4ref, te & late2, block, paired=True)

        # (5) --------------------------------------------------------------
        h5v, h5l = H(p_fut, tr & vis), H(p_fut, tr & late)
        ev5vv, c5vv = E(h5v, p_fut, G_vis())
        ev5ll, c5ll = E(h5l, p_fut, G_late(1))
        T["exp5_p_vis_head"] = dict(head=h5v.info, test_visible=ev5vv, test_late=E(h5v, p_fut, G_late(1))[0])
        T["exp5_p_late_head"] = dict(head=h5l.info, test_late=ev5ll, test_visible=E(h5l, p_fut, G_vis())[0])
        # (5x, 추가) 교차 이식의 라벨 없는 보정 — clip 짝이 없으니 (조건이 다르면 block 도 다르다) Procrustes 는 못 한다
        #   shift   : 대상 조건의 train 평균을 빼고 원래 조건의 train 평균을 더한다
        #   restd   : 머리의 표준화 대신 대상 조건 train 의 평균·표준편차로 표준화 (moment matching)
        mv, ml = p_fut[tr & vis].mean(0), p_fut[tr & late].mean(0)
        sv, sl = _stats(p_fut[tr & vis])[1], _stats(p_fut[tr & late])[1]
        rs = lambda hd, mu, sd: Head(mu, sd, hd.W, hd.b, hd.lam, {})                  # noqa: E731
        T["exp5x_cross_aligned"] = dict(
            vis_head_on_late_shift=E(h5v, p_fut - ml + mv, G_late(1))[0],
            vis_head_on_late_restd=E(rs(h5v, ml, sl), p_fut, G_late(1))[0],
            late_head_on_vis_shift=E(h5l, p_fut - mv + ml, G_vis())[0],
            late_head_on_vis_restd=E(rs(h5l, mv, sv), p_fut, G_vis())[0])
        # (5c, 추가 대조) 같은 visible↔late 교차를 z_all · h_fut 에서 — 교차 실패가 p 만의 것인지,
        #   조건 (가림막이 장면에 있다 · 물체가 가려진다) 이 바뀌면 encoder 에서도 생기는 것인지 가른다
        T["exp5c_cross_controls"] = {}
        for sn, S in (("z_all", z_all), ("h_fut", h_fut)):
            hv_, hl_ = H(S, tr & vis), H(S, tr & late)
            mv_, ml_ = S[tr & vis].mean(0), S[tr & late].mean(0)
            sv_, sl_ = _stats(S[tr & vis])[1], _stats(S[tr & late])[1]
            T["exp5c_cross_controls"][sn] = dict(
                vis_head=hv_.info, late_head=hl_.info,
                vis_on_vis=E(hv_, S, G_vis())[0], vis_on_late=E(hv_, S, G_late(1))[0],
                late_on_late=E(hl_, S, G_late(1))[0], late_on_vis=E(hl_, S, G_vis())[0],
                vis_on_late_restd=E(rs(hv_, ml_, sl_), S, G_late(1))[0],
                late_on_vis_restd=E(rs(hl_, mv_, sv_), S, G_vis())[0])
            c_ = T["exp5c_cross_controls"][sn]
            print(f"[{tname}] (5c) {sn} vis {c_['vis_on_vis']['visible']['acc']} late {c_['late_on_late']['late']['acc']}"
                  f"  vis→late {c_['vis_on_late']['late']['acc']}  late→vis {c_['late_on_vis']['visible']['acc']}", flush=True)
        T["exp5_shuffled_vis"] = shuffled(p_fut, tr & vis, {"visible": te & vis})
        T["exp5_shuffled_late"] = shuffled(p_fut, tr & late, {"late": te & late})
        diffs[f"{tname}/exp5_ownvis_minus_ownlate"] = diff_ci(c5vv, te & vis, c5ll, te & late, block, paired=False)
        print(f"[{tname}] (5) p vis {ev5vv['visible']['acc']} late {ev5ll['late']['acc']}  "
              f"vis→late {T['exp5_p_vis_head']['test_late']['late']['acc']}  "
              f"late→vis {T['exp5_p_late_head']['test_visible']['visible']['acc']}  {time.time()-t0:.0f}s", flush=True)

        # (5b) -------------------------------------------------------------
        T["exp5b_tubelet"] = {}
        for src, arr, off in ((() if SKIP_TUBELET else (("p", p, 0), ("h", h, 8)))):
            T["exp5b_tubelet"][src] = {"visible": [], "late": []}
            for t in range(8):
                X = arr[:, off + t]
                hv, hl = H(X, tr & vis), H(X, tr & late)
                T["exp5b_tubelet"][src]["visible"].append(dict(t=t, lam=hv.lam, test=E(hv, X, G_vis())[0]))
                T["exp5b_tubelet"][src]["late"].append(dict(t=t, lam=hl.lam, test=E(hl, X, G_late(1))[0]))
        if not SKIP_TUBELET: print(f"[{tname}] (5b) p vis {[d['test']['visible']['acc'] for d in T['exp5b_tubelet']['p']['visible']]}"
              f"\n          p late {[d['test']['late']['acc'] for d in T['exp5b_tubelet']['p']['late']]}"
              f"\n          h vis {[d['test']['visible']['acc'] for d in T['exp5b_tubelet']['h']['visible']]}"
              f"\n          h late {[d['test']['late']['acc'] for d in T['exp5b_tubelet']['h']['late']]}"
              f"  {time.time()-t0:.0f}s", flush=True)

        # (6) --------------------------------------------------------------
        hh = H(h_fut, tr)
        hp = H(p_fut, tr)
        Gv, Gl = G_vis(), G_late(1)
        T["exp6_ref"] = dict(
            z_head_on_z=dict(test_visible=E(h2, z_all, Gv)[0], test_late=E(h2, z_all, Gl)[0]),
            h_head=hh.info,
            h_head_on_h=dict(test_visible=E(hh, h_fut, Gv)[0], test_late=E(hh, h_fut, Gl)[0]),
            p_own_heads=dict(test_visible=ev5vv, test_late=ev5ll),
            p_all_head=hp.info,
            p_all_head_on_p=dict(test_visible=E(hp, p_fut, Gv)[0], test_late=E(hp, p_fut, Gl)[0]),
        )
        trv = tr & vis
        for hname, hd, S in (("z", h2, z_all), ("h", hh, h_fut)):
            mp_, ms_ = p_fut[tr].mean(0), S[tr].mean(0)
            mpv, msv = p_fut[trv].mean(0), S[trv].mean(0)
            A, B = (p_fut[trv] - mpv).astype(np.float64), (S[trv] - msv).astype(np.float64)
            U, _, Vt = np.linalg.svd(A.T @ B)
            Rm = U @ Vt
            rel = lambda m: float(np.linalg.norm((p_fut[m] - mpv) @ Rm - (S[m] - msv)) /  # noqa: E731
                                  np.linalg.norm(S[m] - msv))
            Xs = dict(raw=p_fut, shift=p_fut - mp_ + ms_, procrustes=(p_fut - mpv) @ Rm + msv)
            key = "exp6" if hname == "z" else "exp6h"
            T[key] = dict(procrustes_fit=dict(
                rel_residual_train_visible=rel(trv), rel_residual_test_visible=rel(te & vis),
                rel_residual_test_late=rel(te & late),
                rel_residual_identity_train_visible=float(np.linalg.norm(A - B) / np.linalg.norm(B)),
                norm_ratio_centered_p_over_src=float(np.linalg.norm(A) / np.linalg.norm(B))))
            # (추가 대조) clip 짝을 섞은 Procrustes — 짝이 곧 (암묵적) 라벨 정보를 싣는다는 점의 대조. chance 근처여야 한다
            Bp = B[np.random.default_rng(0).permutation(len(B))]
            U3, _, Vt3 = np.linalg.svd(A.T @ Bp)
            Xs["procrustes_shuffled_pairs"] = (p_fut - mpv) @ (U3 @ Vt3) + msv
            # (추가) 같은 Procrustes 를 train late clip 짝 / train 전체 짝으로 맞춘 것 — 사상이 조건마다 다른가
            for fname, fm in (("procrustes_fit_late", tr & late), ("procrustes_fit_all", tr)):
                mpf, msf = p_fut[fm].mean(0), S[fm].mean(0)
                U2, _, Vt2 = np.linalg.svd((p_fut[fm] - mpf).astype(np.float64).T @ (S[fm] - msf).astype(np.float64))
                R2 = U2 @ Vt2
                Xs[fname] = (p_fut - mpf) @ R2 + msf
                T[key]["procrustes_fit"][f"{fname}_rel_residual_test_visible"] = float(
                    np.linalg.norm((p_fut[te & vis] - mpf) @ R2 - (S[te & vis] - msf)) / np.linalg.norm(S[te & vis] - msf))
                T[key]["procrustes_fit"][f"{fname}_rel_residual_test_late"] = float(
                    np.linalg.norm((p_fut[te & late] - mpf) @ R2 - (S[te & late] - msf)) / np.linalg.norm(S[te & late] - msf))
            cs = {}
            for vn, X in Xs.items():
                evv, cv = E(hd, X, Gv)
                evl, cl = E(hd, X, Gl)
                T[key][vn] = dict(test_visible=evv, test_late=evl)
                cs[vn] = cv
            for vn in ("shift", "procrustes", "procrustes_fit_late", "procrustes_fit_all"):
                for sn, sm in (("visible", te & vis), ("late", te & late)):
                    diffs[f"{tname}/{key}_{vn}_minus_raw_{sn}"] = diff_ci(cs[vn], sm, cs["raw"], sm, block, paired=True)
            print(f"[{tname}] ({key}) " + "  ".join(
                f"{vn} v{T[key][vn]['test_visible']['visible']['acc']}/l{T[key][vn]['test_late']['late']['acc']}"
                for vn in Xs) + f"  {time.time()-t0:.0f}s", flush=True)
        # (6e, 추가 대조) encoder ↔ encoder 에 같은 Procrustes — 'visible 짝 사상이 late 에 안 맞는다' 가 p 만의 것인지,
        #   두 공간 사이의 관계가 조건마다 바뀌는 일반 현상인지 가른다 (h_fut → z_all 에 z 머리, z_all → h_fut 에 h 머리)
        T["exp6e_encoder_pair_control"] = {}
        for nm, hd, Src, Dst in (("h_to_z", h2, h_fut, z_all), ("z_to_h", hh, z_all, h_fut)):
            res = dict(raw=dict(test_visible=E(hd, Src, Gv)[0], test_late=E(hd, Src, Gl)[0]))
            for fname, fm in (("fit_visible", tr & vis), ("fit_late", tr & late)):
                ms, md = Src[fm].mean(0), Dst[fm].mean(0)
                U4, _, Vt4 = np.linalg.svd((Src[fm] - ms).astype(np.float64).T @ (Dst[fm] - md).astype(np.float64))
                R4 = U4 @ Vt4
                X = (Src - ms) @ R4 + md
                rr = lambda m: float(np.linalg.norm(X[m] - Dst[m]) / np.linalg.norm(Dst[m] - md))  # noqa: E731
                res[fname] = dict(test_visible=E(hd, X, Gv)[0], test_late=E(hd, X, Gl)[0],
                                  rel_residual_test_visible=rr(te & vis), rel_residual_test_late=rr(te & late))
            T["exp6e_encoder_pair_control"][nm] = res
            print(f"[{tname}] (6e) {nm} " + "  ".join(
                f"{vn} v{res[vn]['test_visible']['visible']['acc']}/l{res[vn]['test_late']['late']['acc']}" for vn in res), flush=True)
        # p 와 h 가 같은 공간인지 (학습 목표) — clip 별 코사인
        cos = lambda a, b: float((np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).mean())  # noqa: E731
        T["geometry"] = dict(cos_pfut_hfut_test_visible=cos(p_fut[te & vis], h_fut[te & vis]),
                             cos_pfut_hfut_test_late=cos(p_fut[te & late], h_fut[te & late]),
                             cos_pfut_zall_test_visible=cos(p_fut[te & vis], z_all[te & vis]),
                             cos_pfut_zall_test_late=cos(p_fut[te & late], z_all[te & late]))
    R["diffs"] = diffs
    R["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    return R


# ----------------------------------------------------------------------------- figures
def _bar_rows(T):
    """(label, visible stats|None, late stats|None)."""
    e = T
    return [
        ("(2) z head\non z_all", e["exp2_z_all"]["test"]["visible"], e["exp2_z_all"]["test"]["late"]),
        ("(3) same pos.\nvis t7 | late z_hid", e["exp3c1_visible_t7"]["test"]["visible"],
         e["exp3_z_hid"]["test"]["late_k>=2"]),
        ("(3c2) late\nvisible ctx", None, e["exp3c2_late_visible_ctx"]["test"]["late_k>=2"]),
        ("(4) z head\n-> z_hid", None, e["exp4_zhead_on_zhid"]["test"]["late_k>=2"]),
        ("(4b) z_hid head\n-> z_all", None, e["exp4b_zhidhead_on_zall"]["test"]["late_k>=2"]),
        ("(5) p own\nhead", e["exp5_p_vis_head"]["test_visible"]["visible"], e["exp5_p_late_head"]["test_late"]["late"]),
        ("(5) p cross head\nlate->vis | vis->late", e["exp5_p_late_head"]["test_visible"]["visible"], e["exp5_p_vis_head"]["test_late"]["late"]),
        ("(5c) z_all cross head\nlate->vis | vis->late", e["exp5c_cross_controls"]["z_all"]["late_on_vis"]["visible"],
         e["exp5c_cross_controls"]["z_all"]["vis_on_late"]["late"]),
        ("(5c) h_fut cross head\nlate->vis | vis->late", e["exp5c_cross_controls"]["h_fut"]["late_on_vis"]["visible"],
         e["exp5c_cross_controls"]["h_fut"]["vis_on_late"]["late"]),
        ("(6a) z head\n-> p raw", e["exp6"]["raw"]["test_visible"]["visible"], e["exp6"]["raw"]["test_late"]["late"]),
        ("(6b) z head\n-> p shift", e["exp6"]["shift"]["test_visible"]["visible"], e["exp6"]["shift"]["test_late"]["late"]),
        ("(6c) z head\n-> p Procr.", e["exp6"]["procrustes"]["test_visible"]["visible"],
         e["exp6"]["procrustes"]["test_late"]["late"]),
        ("(6c-late) z head\n-> p Procr. fit late", e["exp6"]["procrustes_fit_late"]["test_visible"]["visible"],
         e["exp6"]["procrustes_fit_late"]["test_late"]["late"]),
        ("(6e) z head -> h_fut\nProcr. fit visible", e["exp6e_encoder_pair_control"]["h_to_z"]["fit_visible"]["test_visible"]["visible"],
         e["exp6e_encoder_pair_control"]["h_to_z"]["fit_visible"]["test_late"]["late"]),
        ("h head\non h_fut", e["exp6_ref"]["h_head_on_h"]["test_visible"]["visible"],
         e["exp6_ref"]["h_head_on_h"]["test_late"]["late"]),
        ("(6h) h head\n-> p raw", e["exp6h"]["raw"]["test_visible"]["visible"], e["exp6h"]["raw"]["test_late"]["late"]),
        ("(6h) h head\n-> p shift", e["exp6h"]["shift"]["test_visible"]["visible"], e["exp6h"]["shift"]["test_late"]["late"]),
        ("(6h) h head\n-> p Procr.", e["exp6h"]["procrustes"]["test_visible"]["visible"],
         e["exp6h"]["procrustes"]["test_late"]["late"]),
        ("(6h c-late) h head\n-> p Procr. fit late", e["exp6h"]["procrustes_fit_late"]["test_visible"]["visible"],
         e["exp6h"]["procrustes_fit_late"]["test_late"]["late"]),
    ]


def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    # ---- fig_probe
    fig, axes = plt.subplots(2, 1, figsize=(20, 8.8), sharex=True)
    for ax, tname in zip(axes, TARGETS):
        rows = _bar_rows(R[tname])
        x = np.arange(len(rows))
        w = 0.38
        for j, (lab, col, off) in enumerate((("visible test", C_BLUE, -w / 2), ("late test", C_ORANGE, w / 2))):
            for i, r in enumerate(rows):
                s = r[1 + j]
                if s is None:
                    continue
                ax.bar(x[i] + off, s["acc"], w * 0.92, color=col, label=lab if i == 0 or (j == 1 and i == 0) else None)
                ax.errorbar(x[i] + off, s["acc"], yerr=[[s["acc"] - s["lo"]], [s["hi"] - s["acc"]]],
                            color="black", lw=0.9, capsize=2)
                ax.text(x[i] + off, s["hi"] + 1.2, f"{s['acc']:.0f}", ha="center", va="bottom", fontsize=7, color="#333333")
        ch = 100 / len(R[tname]["classes"])
        ax.axhline(ch, color="#555555", ls="--", lw=1)
        ax.text(len(rows) - 0.35, ch, f"chance\n{ch:.1f}", ha="left", va="center", fontsize=8, color="#555555")
        for sep in (0.5, 4.5, 8.5, 12.5, 13.5, 14.5):
            ax.axvline(sep, color="#cccccc", lw=0.8)
        ax.set_ylim(0, 112)
        ax.set_ylabel(f"{tname} test accuracy (%)")
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=C_BLUE, label="visible test (k=0)"),
                           Patch(color=C_ORANGE, label="late test (k>=1; k>=2 for (3),(3c2),(4),(4b))")],
                  loc="upper right", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(1, 1.08))
        ax.set_xlabel("(a) shape (7-way)" if tname == "shape" else "(b) color (8-way)", fontsize=10)
        ax.set_xlim(-0.6, len(rows) + 0.2)
    axes[-1].set_xticks(np.arange(len(rows)))
    axes[-1].set_xticklabels([r[0] for r in rows], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe.png", dpi=150)
    plt.close(fig)

    # ---- fig_probe_tubelet
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    t = np.arange(8)
    allv = [d["test"][v]["lo"] for tn in TARGETS for s_ in ("p", "h") for v in ("visible", "late")
            for d in R[tn]["exp5b_tubelet"][s_][v]]
    allv += [d["test"][f"late_k{kk}"]["acc"] for tn in TARGETS for s_ in ("p", "h")
             for d in R[tn]["exp5b_tubelet"][s_]["late"] for kk in range(1, 5)]
    ylo = max(0.0, np.floor(min(allv) / 5) * 5 - 5)
    for ci, tname in enumerate(TARGETS):
        T = R[tname]["exp5b_tubelet"]
        ax = axes[0, ci]
        for src, ls, mk in (("p", "-", "o"), ("h", "--", "s")):
            for v, col in (("visible", C_BLUE), ("late", C_ORANGE)):
                a = np.array([[d["test"][v]["acc"], d["test"][v]["lo"], d["test"][v]["hi"]] for d in T[src][v]])
                ax.plot(t, a[:, 0], ls=ls, marker=mk, ms=5, lw=2, color=col, label=f"{src} {v}")
                ax.fill_between(t, a[:, 1], a[:, 2], color=col, alpha=0.12, lw=0)
        ax.set_ylim(ylo, 100.5)
        ax.set_xlabel(f"future tubelet t\n({'ab'[ci]}) {tname}: single future tubelet, visible vs late head", fontsize=10)
        ax.set_ylabel("test accuracy (%)")
        ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
        ax = axes[1, ci]
        grays = ["#bbbbbb", "#888888", "#555555", "#111111"]
        for src, ls, mk in (("p", "-", "o"), ("h", "--", "s")):
            for kk in range(1, 5):
                a = [d["test"][f"late_k{kk}"]["acc"] for d in T[src]["late"]]
                ax.plot(t, a, ls=ls, marker=mk, ms=4, lw=1.6, color=grays[kk - 1], label=f"{src} late k={kk}")
        ax.set_ylim(ylo, 100.5)
        ax.set_ylabel("test accuracy (%)")
        ax.set_xlabel(f"future tubelet t\n({'cd'[ci]}) {tname}: late head, late test split by k", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    for ax in axes.flat:
        ax.set_xticks(t)
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_tubelet.png", dpi=150)
    plt.close(fig)


def tables(R, L=None):
    """results.json (+ results_lam0.1.json) → PROBE.md 에 옮기는 마크다운 표 (stdout)."""
    import json
    TG = ("shape", "color")


    def f(s, n=False):
        if s is None:
            return "—"
        out = f"{s['acc']:.1f} [{s['lo']:.1f}, {s['hi']:.1f}]"
        return out + (f" (n={s['n']})" if n else "")


    def row(lab, getter, n=False):
        cells = []
        for t in TG:
            try:
                cells.append(f(getter(R[t]), n))
            except KeyError:
                cells.append("—")
        return f"| {lab} | " + " | ".join(cells) + " |"


    def hdr(first="test 부분집합"):
        return f"| {first} | shape (chance 14.3) | color (chance 12.5) |\n|---|---|---|"


    out = []
    P = out.append
    # sanity
    S = R["sanity"]
    P("### sanity\n")
    P(json.dumps({k: S[k] for k in ("n_clip", "n_block", "n_train", "n_test", "n_train_blocks", "n_test_blocks",
                                    "block_overlap_train_test", "block_overlap_fit_holdout", "n_fit", "n_holdout")}))
    P(json.dumps(S["train_frac_by_condition"]))
    P(json.dumps(S["train_frac_by_k"]))
    for t in TG:
        cc = R[t]["class_counts"]
        P(f"\n{t} class counts")
        P("| split / 부분 | " + " | ".join(R[t]["classes"]) + " |")
        P("|---|" + "---|" * len(R[t]["classes"]))
        for sp in ("train", "test"):
            for v in ("all", "visible", "late", "late_k>=2"):
                P(f"| {sp} / {v} | " + " | ".join(str(cc[sp][v][c]) for c in R[t]["classes"]) + " |")
    # convergence
    conv = []
    def walk(o, path=""):
        if isinstance(o, dict):
            if "refit" in o and "selection" in o:
                conv.append((path, o["lam"], o["refit"]["converged"], o["refit"]["grad_max"], o["refit"]["n_iter"],
                             all(v["converged"] for v in o["selection"].values())))
            for k, v in o.items():
                walk(v, path + "/" + k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + f"[{i}]")
    walk({t: R[t] for t in TG})
    P(f"\nheads: {len(conv)}, refit converged all: {all(c[2] for c in conv)}, selection converged all: {all(c[5] for c in conv)}, "
      f"max grad {max(c[3] for c in conv):.2e}, max iter {max(c[4] for c in conv)}")
    P("not converged: " + str([c for c in conv if not (c[2] and c[5])][:10]))
    from collections import Counter
    P("lam counts: " + str(Counter(c[1] for c in conv)))

    # (2)
    P("\n### (2)\n" + hdr())
    for g in ["all", "visible", "late", "late_k1", "late_k2", "late_k3", "late_k4",
              "visible/static", "visible/flat", "visible/ramp", "late/static", "late/flat", "late/ramp"]:
        P(row(g, lambda T, g=g: T["exp2_z_all"]["test"][g], n=False))
    P(row("λ", lambda T: {"acc": T["exp2_z_all"]["head"]["lam"], "lo": 0, "hi": 0}))
    P(row("shuffled all", lambda T: T["exp2_shuffled"]["test"]["all"]))
    P(row("shuffled visible", lambda T: T["exp2_shuffled"]["test"]["visible"]))
    P(row("shuffled late", lambda T: T["exp2_shuffled"]["test"]["late"]))
    P("n: " + str({g: R["shape"]["exp2_z_all"]["test"][g]["n"] for g in R["shape"]["exp2_z_all"]["test"]}))
    P("majority: " + str({t: {g: R[t]["exp2_z_all"]["test"][g]["majority"] for g in ("all", "visible", "late")} for t in TG}))

    # (3)
    P("\n### (3)\n| 부분집합 | (3) z_hid shape | (3c2) visible ctx shape | (3) z_hid color | (3c2) visible ctx color |\n|---|---|---|---|---|")
    for g in ["late_k>=2", "late_k2", "late_k3", "late_k4", "late_k>=2/static", "late_k>=2/flat", "late_k>=2/ramp"]:
        P(f"| {g} | {f(R['shape']['exp3_z_hid']['test'][g])} | {f(R['shape']['exp3c2_late_visible_ctx']['test'][g])} | "
          f"{f(R['color']['exp3_z_hid']['test'][g])} | {f(R['color']['exp3c2_late_visible_ctx']['test'][g])} |")
    P("n: " + str({g: R["shape"]["exp3_z_hid"]["test"][g]["n"] for g in R["shape"]["exp3_z_hid"]["test"]}))
    P("\n| visible 부분 | (3c1) t7 shape | (3c1) t6-7 shape | (3c1) t7 color | (3c1) t6-7 color |\n|---|---|---|---|---|")
    for g in ["visible", "visible/static", "visible/flat", "visible/ramp"]:
        P(f"| {g} | {f(R['shape']['exp3c1_visible_t7']['test'][g])} | {f(R['shape']['exp3c1_visible_t67']['test'][g])} | "
          f"{f(R['color']['exp3c1_visible_t7']['test'][g])} | {f(R['color']['exp3c1_visible_t67']['test'][g])} |")

    # (4)
    P("\n### (4)\n| 부분집합 (late k>=2 test) | (4) z head→z_hid shape | (4b) z_hid head→z_all shape | ref z head→z_all shape | (4) color | (4b) color | ref color |\n|---|---|---|---|---|---|---|")
    for g in ["late_k>=2", "late_k2", "late_k3", "late_k4", "late_k>=2/static", "late_k>=2/flat", "late_k>=2/ramp"]:
        P(f"| {g} | " + " | ".join(f(R[t][k]["test"][g]) for t in TG for k in
                                     ("exp4_zhead_on_zhid", "exp4b_zhidhead_on_zall", "exp4_ref_zhead_on_zall_same_clips")) + " |")
    if "exp4c_pooling_controls" in R["shape"]:
        P("\n(4c)\n| 머리 → 특징 (test) | shape | color |\n|---|---|---|")
        for k, lab in (("zhead_on_visible_t7", "z_all head → visible z t7"), ("zhead_on_visible_t67", "z_all head → visible z t6-7"),
                       ("zhead_on_late_visctx", "z_all head → late k>=2 visible-ctx pool"),
                       ("zhidhead_on_visible_t7", "z_hid head → visible z t7"), ("zhidhead_on_visible_t67", "z_hid head → visible z t6-7")):
            g = "late_k>=2" if "late" in k else "visible"
            P(row(lab, lambda T, k=k, g=g: T["exp4c_pooling_controls"][k][g]))
        for k in ("zhead_on_visible_t7", "zhidhead_on_visible_t7"):
            P(k + " by motion: " + str({t: {g: R[t]["exp4c_pooling_controls"][k][g]["acc"] for g in R[t]["exp4c_pooling_controls"][k]} for t in TG}))

    # (5)
    P("\n### (5)\n" + hdr("머리 → test"))
    for lab, get in (
            ("visible 머리 → visible", lambda T: T["exp5_p_vis_head"]["test_visible"]["visible"]),
            ("late 머리 → late", lambda T: T["exp5_p_late_head"]["test_late"]["late"]),
            ("visible 머리 → late (교차)", lambda T: T["exp5_p_vis_head"]["test_late"]["late"]),
            ("late 머리 → visible (교차)", lambda T: T["exp5_p_late_head"]["test_visible"]["visible"]),
            ("(5x) visible 머리 → late, shift", lambda T: T["exp5x_cross_aligned"]["vis_head_on_late_shift"]["late"]),
            ("(5x) visible 머리 → late, restd", lambda T: T["exp5x_cross_aligned"]["vis_head_on_late_restd"]["late"]),
            ("(5x) late 머리 → visible, shift", lambda T: T["exp5x_cross_aligned"]["late_head_on_vis_shift"]["visible"]),
            ("(5x) late 머리 → visible, restd", lambda T: T["exp5x_cross_aligned"]["late_head_on_vis_restd"]["visible"]),
            ("shuffled visible", lambda T: T["exp5_shuffled_vis"]["test"]["visible"]),
            ("shuffled late", lambda T: T["exp5_shuffled_late"]["test"]["late"])):
        P(row(lab, get))
    P("\n| 부분 | vis→vis shape | late→late shape | vis→late shape | vis→vis color | late→late color | vis→late color |\n|---|---|---|---|---|---|---|")
    for m in ("static", "flat", "ramp"):
        P(f"| {m} | " + " | ".join(
            f"{R[t]['exp5_p_vis_head']['test_visible'][f'visible/{m}']['acc']:.1f} | {R[t]['exp5_p_late_head']['test_late'][f'late/{m}']['acc']:.1f} | "
            f"{R[t]['exp5_p_vis_head']['test_late'][f'late/{m}']['acc']:.1f}" for t in TG) + " |")
    for kk in range(1, 5):
        P(f"| k={kk} | — | " + " | ".join(
            f"{R[t]['exp5_p_late_head']['test_late'][f'late_k{kk}']['acc']:.1f} | {R[t]['exp5_p_vis_head']['test_late'][f'late_k{kk}']['acc']:.1f}"
            + (" | —" if t == "shape" else "") for t in TG) + " |")
    P("late→vis by motion: " + str({t: {m: R[t]["exp5_p_late_head"]["test_visible"][f"visible/{m}"]["acc"] for m in ("static", "flat", "ramp")} for t in TG}))
    if "exp5c_cross_controls" in R["shape"]:
        P("\n(5c)\n| 특징 | 머리 → test | shape | color |\n|---|---|---|---|")
        for sn in ("z_all", "h_fut"):
            for k, g, lab in (("vis_on_vis", "visible", "visible → visible"), ("late_on_late", "late", "late → late"),
                              ("vis_on_late", "late", "visible → late"), ("late_on_vis", "visible", "late → visible"),
                              ("vis_on_late_restd", "late", "visible → late, restd"), ("late_on_vis_restd", "visible", "late → visible, restd")):
                P(f"| {sn} | {lab} | " + " | ".join(f(R[t]["exp5c_cross_controls"][sn][k][g]) for t in TG) + " |")
        for sn in ("z_all", "h_fut"):
            P(sn + " vis→late by motion/k: " + str({t: {g: R[t]["exp5c_cross_controls"][sn]["vis_on_late"][g]["acc"] for g in R[t]["exp5c_cross_controls"][sn]["vis_on_late"]} for t in TG}))

    # (5b)
    P("\n### (5b)")
    for t in TG:
        P(f"\n{t}\n| 특징 · 조건 | " + " | ".join(f"t{i}" for i in range(8)) + " |\n|---|" + "---|" * 8)
        for src in ("p", "h"):
            for v in ("visible", "late"):
                P(f"| {src} {v} | " + " | ".join(f"{d['test'][v]['acc']:.1f}" for d in R[t]["exp5b_tubelet"][src][v]) + " |")
            for kk in range(1, 5):
                P(f"| {src} late k={kk} | " + " | ".join(f"{d['test'][f'late_k{kk}']['acc']:.1f}" for d in R[t]["exp5b_tubelet"][src]["late"]) + " |")
        P("CI widths p late: " + str([round(d['test']['late']['hi'] - d['test']['late']['lo'], 1) for d in R[t]["exp5b_tubelet"]["p"]["late"]]))

    # (6)
    P("\n### (6)\n| 머리 · 변형 | shape visible | shape late | color visible | color late |\n|---|---|---|---|---|")
    refs = (("z head → z_all (상한)", lambda T, v: T["exp6_ref"]["z_head_on_z"][f"test_{v}"][v]),
            ("h head → h_fut (상한)", lambda T, v: T["exp6_ref"]["h_head_on_h"][f"test_{v}"][v]),
            ("p 자기 머리 (5, 조건별)", lambda T, v: T["exp6_ref"]["p_own_heads"][f"test_{v}"][v]),
            ("p 머리 (train 전체)", lambda T, v: T["exp6_ref"]["p_all_head_on_p"][f"test_{v}"][v]))
    for lab, get in refs:
        P(f"| {lab} | " + " | ".join(f(get(R[t], v)) for t in TG for v in ("visible", "late")) + " |")
    for key, hn in (("exp6", "z"), ("exp6h", "h")):
        for vn, lab in (("raw", "raw"), ("shift", "mean-shift"), ("procrustes", "Procrustes (fit visible)"),
                        ("procrustes_shuffled_pairs", "Procrustes, 짝 섞기"), ("procrustes_fit_late", "Procrustes (fit late)"),
                        ("procrustes_fit_all", "Procrustes (fit 전체)")):
            P(f"| {hn} head → p, {lab} | " + " | ".join(f(R[t][key][vn][f"test_{v}"][v]) for t in TG for v in ("visible", "late")) + " |")
    if "exp6e_encoder_pair_control" in R["shape"]:
        for nm, lab in (("h_to_z", "(6e) z head → h_fut"), ("z_to_h", "(6e) h head → z_all")):
            for vn, vl in (("raw", "raw"), ("fit_visible", "Procrustes (fit visible)"), ("fit_late", "Procrustes (fit late)")):
                P(f"| {lab}, {vl} | " + " | ".join(
                    f(R[t]["exp6e_encoder_pair_control"][nm][vn][f"test_{v}"][v]) for t in TG for v in ("visible", "late")) + " |")
        P("6e residuals: " + json.dumps({nm: {vn: {k: round(x, 3) for k, x in R["shape"]["exp6e_encoder_pair_control"][nm][vn].items()
                                                  if k.startswith("rel")} for vn in ("fit_visible", "fit_late")}
                                         for nm in ("h_to_z", "z_to_h")}))
    P("\nby motion (acc):")
    for key in ("exp6", "exp6h"):
        for vn in ("raw", "shift", "procrustes", "procrustes_fit_late"):
            P(f"{key} {vn}: " + str({t: {v: {m: R[t][key][vn][f'test_{v}'][f'{v}/{m}']['acc'] for m in ('static', 'flat', 'ramp')} for v in ('visible', 'late')} for t in TG}))
    P("procrustes fit: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in R["shape"][k]["procrustes_fit"].items()} for k in ("exp6", "exp6h")}))
    P("geometry: " + json.dumps({k: round(v, 3) for k, v in R["shape"]["geometry"].items()}))
    P("\ndiffs")
    for k, v in R["diffs"].items():
        P(f"| {k} | {v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}] |")

    if L:
        P("\n### λ=0.1 민감도 (results_lam0.1.json)\n| 수치 | shape 선택 λ | shape λ=0.1 | color 선택 λ | color λ=0.1 |\n|---|---|---|---|---|")
        items = (("(2) z head → z_all, late", lambda T: T["exp2_z_all"]["test"]["late"]),
                 ("(3) z_hid 머리 (late k>=2)", lambda T: T["exp3_z_hid"]["test"]["late_k>=2"]),
                 ("(4) z head → z_hid", lambda T: T["exp4_zhead_on_zhid"]["test"]["late_k>=2"]),
                 ("(4b) z_hid head → z_all", lambda T: T["exp4b_zhidhead_on_zall"]["test"]["late_k>=2"]),
                 ("(4c) z head → visible t7", lambda T: T["exp4c_pooling_controls"]["zhead_on_visible_t7"]["visible"]),
                 ("(5) p visible 머리 → late", lambda T: T["exp5_p_vis_head"]["test_late"]["late"]),
                 ("(5) p late 머리 → visible", lambda T: T["exp5_p_late_head"]["test_visible"]["visible"]),
                 ("(5c) z_all visible 머리 → late", lambda T: T["exp5c_cross_controls"]["z_all"]["vis_on_late"]["late"]),
                 ("(5c) h_fut visible 머리 → late", lambda T: T["exp5c_cross_controls"]["h_fut"]["vis_on_late"]["late"]),
                 ("(6a) z head → p raw, visible", lambda T: T["exp6"]["raw"]["test_visible"]["visible"]),
                 ("(6b) z head → p shift, visible", lambda T: T["exp6"]["shift"]["test_visible"]["visible"]),
                 ("(6c) z head → p Procrustes, visible", lambda T: T["exp6"]["procrustes"]["test_visible"]["visible"]),
                 ("(6c) z head → p Procrustes, late", lambda T: T["exp6"]["procrustes"]["test_late"]["late"]),
                 ("(6h) h head → p raw, visible", lambda T: T["exp6h"]["raw"]["test_visible"]["visible"]),
                 ("(6h) h head → p Procrustes, late", lambda T: T["exp6h"]["procrustes"]["test_late"]["late"]),
                 ("(6c-late) z head → p Procrustes fit late, late", lambda T: T["exp6"]["procrustes_fit_late"]["test_late"]["late"]),
                 ("(6e) z head → h_fut Procrustes fit visible, late",
                  lambda T: T["exp6e_encoder_pair_control"]["h_to_z"]["fit_visible"]["test_late"]["late"]),
                 ("(6e) h head → z_all Procrustes fit visible, late",
                  lambda T: T["exp6e_encoder_pair_control"]["z_to_h"]["fit_visible"]["test_late"]["late"]))
        for lab, get in items:
            cells = []
            for t in TG:
                for X in (R, L):
                    try:
                        cells.append(f"{get(X[t])['acc']:.1f}")
                    except KeyError:
                        cells.append("—")
            P(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    global LAMS, SKIP_TUBELET
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--lams", type=float, nargs="+", default=None,
                    help="λ 후보를 바꾼다 (민감도 점검). --tag 와 같이 쓴다")
    ap.add_argument("--tag", default="", help="results_<tag>.json 으로 쓰고 그림은 안 그린다")
    ap.add_argument("--skip-tubelet", action="store_true", help="(5b) 를 건너뛴다")
    ap.add_argument("--tables", action="store_true", help="results.json 에서 PROBE.md 용 표를 찍는다")
    a = ap.parse_args()
    if a.tables:
        lp = OUT / "results_lam0.1.json"
        print(tables(json.loads((OUT / "results.json").read_text()),
                     json.loads(lp.read_text()) if lp.exists() else None))
        return
    if a.lams:
        assert a.tag, "--lams 는 --tag 와 같이 (정본 results.json 을 덮지 않게)"
        LAMS = a.lams
    SKIP_TUBELET = a.skip_tubelet
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        plot(json.loads((OUT / "results.json").read_text()))
        return
    torch.set_num_threads(8)
    R = run(torch.device(a.device))
    def _np(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    R["meta"]["lams"] = LAMS
    R["meta"]["skip_tubelet"] = SKIP_TUBELET
    if a.tag:
        (OUT / f"results_{a.tag}.json").write_text(json.dumps(R, indent=1, default=_np))
        print(f"→ {OUT / f'results_{a.tag}.json'}", flush=True)
        return
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=_np))
    plot(R)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
