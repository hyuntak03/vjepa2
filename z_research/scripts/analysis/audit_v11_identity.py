#!/usr/bin/env python3
"""감사 (2026-09-25): v11 정체 (shape · color) probe 와 풀링 기하 주장을 두 probe 로 다시 낸다.

대상 문서: z_research/RollOutV3/figures/v11_probe/PROBE.md · v11_geometry/GEOMETRY.md
입력 캐시: /local_datasets/world/world_analysis/cache/v11_pooled_vith/{z,p,h}.npy, meta.npz (v11_pooled_features.py)
출력    : z_research/RollOutV3/audit/v11_identity/{audit_probe.json (1)(2)(3), audit_geom.json (4); AUDIT.md 는 사람이 씀}

(1) 궤적 반복 — meta.truth (n,32,2) 로 궤적을 식별해 block 사이 반복 · split 누수 · 라벨과의 교차를 센다.
    누수가 정체 probe 에 영향을 주는지 보려고 두 가지를 한다.
      combo split : (block ∪ 외형조합) 연결성분 단위 split. block 이 pos_a/pos_b (다른 shape 또는 color) 를 묶어서
                    성분이 (운동·궤적·env) 장면 통째가 된다 → 사실상 **장면 hold-out** (test clip 의 63 % 가 train 에 없는 궤적×env).
      twin (--part twin) : block split 그대로 두고, test clip 을 '같은 외형조합이 train 에 있다 / 없다' 로 나눠 같은 머리로 읽는다.
(2) 두 probe 로 재도출
      logistic : 레포 v11_pooled_probe.train_head 그대로 (표준화 + multinomial logistic, λ ∈ {1e-4..1e-1} block hold-out 선택)
      ridge    : 닫힌 해 ridge (표준화, one-hot 중심화 목표, λ ∈ {1..1e5} 같은 hold-out 선택) — 레포 probe 코드와 독립
      ncm      : 표준화 공간 최근접 클래스 평균 (참고용, 파라미터 없음)
    (2) z_all, (3) z_hid vs z_visctx (late k>=2), (5) p_fut late 머리 + 튜블릿별, (6) z/h 머리 → p raw, encoder 끼리 raw,
    Procrustes (같은 조건 train 짝) → p, visible 짝 → late, encoder 쌍 대조, 짝 섞기 귀무.
(3) 복사 기준선 — z[:,7] (문맥 마지막 튜블릿, late k>=2 에서는 물체가 완전히 가려짐) · z_t67 · z_all 을 late 머리로 읽어 p_fut 와 짝 비교.
    [z_t7, p_fut] 결합 머리가 z_t7 머리보다 나은지 (p 가 복사 이상을 싣는가).
(4) 기하 — 장면 칸 분산 몫, 상수 이동 몫 · 중심 방향 분류, resid 클래스 평균 부분공간 교차 적합 (split-half 정규화)
    를 독립 numpy 코드로 다시 계산 + block bootstrap CI + block 단위 라벨 순열 귀무 + clip 단위노름 변형.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/audit_v11_identity.py --part probe --device cuda:5   # (1)(2)(3) → audit_probe.json (~75 분, 로지스틱이 대부분)
  $P z_research/scripts/analysis/audit_v11_identity.py --part twin --device cuda:1    # (1b) → audit_twin.json (~6 분)
  $P z_research/scripts/analysis/audit_v11_identity.py --part geom                    # (4) → audit_geom.json (CPU ~3 분)
  $P z_research/scripts/analysis/audit_v11_identity.py --part xfer --device cuda:6    # (5) 풀링 대조 4c · Procrustes 복사 점검 A3 · G5 이식 CI A4 (운동별) → audit_xfer.json
  (공용 노드에서는 OMP/OPENBLAS/MKL_NUM_THREADS=12 를 준다 — 안 주면 BLAS 스레드 과다구독으로 수십 배 느려진다)
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import v11_pooled_probe as P  # noqa: E402  (함수만 쓴다. 그 스크립트의 OUT 에는 쓰지 않는다)

OUT = ROOT / "z_research/RollOutV3/audit/v11_identity"
REF = ROOT / "z_research/RollOutV3/figures"
RIDGE_LAMS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
TARGETS = {"shape": "shape_pre", "color": "color_pre"}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


# ============================================================================ probes
class RidgeHead:
    def __init__(self, mu, sd, W, b, lam, info):
        self.mu, self.sd, self.W, self.b, self.lam, self.info = mu, sd, W, b, lam, info

    def predict(self, X):
        return (((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.W + self.b).argmax(1)


def _ridge_solve(Xs, y, C, lams):
    """Xs 표준화됨. 목표 = one-hot 중심화, 절편 = 평균. eigh 한 번으로 λ 격자 전체."""
    Y = np.eye(C)[y]
    xm, ym = Xs.mean(0), Y.mean(0)
    Xc, Yc = Xs - xm, Y - ym
    e, V = np.linalg.eigh(Xc.T @ Xc)
    VtXY = V.T @ (Xc.T @ Yc)
    out = {}
    for lam in lams:
        W = V @ (VtXY / (e + lam)[:, None])
        out[lam] = (W, ym - xm @ W)
    return out


def fit_ridge(X, y, trm, inner, C):
    fit, hold = trm & inner, trm & ~inner
    mu, sd = P._stats(X[fit].astype(np.float64))
    sols = _ridge_solve((X[fit] - mu) / sd, y[fit], C, RIDGE_LAMS)
    Xh = (X[hold] - mu) / sd
    sel = {}
    for lam, (W, b) in sols.items():
        S = Xh @ W + b
        sel[lam] = (float((S.argmax(1) == y[hold]).mean()), -float(((S - np.eye(C)[y[hold]]) ** 2).mean()))
    best = max(RIDGE_LAMS, key=lambda l: sel[l])
    mu, sd = P._stats(X[trm].astype(np.float64))
    W, b = _ridge_solve((X[trm] - mu) / sd, y[trm], C, [best])[best]
    return RidgeHead(mu, sd, W, b, best, dict(lam=best, holdout_acc={f"{l:g}": round(v[0], 4) for l, v in sel.items()}))


class NcmHead:
    def __init__(self, X, y, trm, C):
        self.mu, self.sd = P._stats(X[trm].astype(np.float64))
        Xs = (X[trm] - self.mu) / self.sd
        self.M = np.stack([Xs[y[trm] == c].mean(0) for c in range(C)])
        self.lam = None

    def predict(self, X):
        Xs = (np.asarray(X, np.float64) - self.mu) / self.sd
        return (-2 * Xs @ self.M.T + (self.M ** 2).sum(1)).argmin(1)


def rec(correct, unit, y, mask, C):
    r = P.acc_ci(correct, unit, y, mask, C)
    return None if r is None else dict(acc=r["acc"], lo=r["lo"], hi=r["hi"], n=r["n"])


def dci(c1, c2, mask, unit):
    r = P.diff_ci(c1, mask, c2, mask, unit, paired=True)
    return dict(diff=r["diff"], lo=r["lo"], hi=r["hi"])


# ============================================================================ (1) trajectories
def trajectory_part(meta, tr):
    n = len(meta["video_id"])
    T = meta["truth"].reshape(n, -1)
    ut, traj = np.unique(np.round(T, 3), axis=0, return_inverse=True)
    dev_max = max(float(np.abs(T[traj == j] - T[traj == j].mean(0)).max()) for j in range(len(ut)))
    cond, blk = meta["condition"].astype(str), meta["block_id"].astype(str)
    tc = np.char.add(np.char.add(cond, "|"), traj.astype(str))
    utc, tci = np.unique(tc, return_inverse=True)
    out = dict(n_clip=n, n_traj_ignoring_condition=int(len(ut)), n_traj_with_condition=int(len(utc)),
               max_within_traj_deviation_norm=dev_max,
               traj_by_motion={mo: int(len(set(traj[meta["motion"] == mo]))) for mo in sorted(set(meta["motion"]))},
               blocks_per_traj_with_condition=sorted(Counter(len(set(blk[tci == j])) for j in range(len(utc))).items()),
               traj_in_train_and_test=int(len(set(tci[tr]) & set(tci[~tr]))),
               test_clips_whose_traj_in_train=float(np.mean(np.isin(tci[~tr], tci[tr]))))
    for tname, col in TARGETS.items():
        lab = meta[col].astype(str)
        classes = sorted(set(lab)); y = np.searchsorted(classes, lab)
        # 궤적 → 라벨 조회 (train 다수결) 의 test 정확도 = 궤적이 라벨에 대해 주는 정보의 상한 (선형 아님, lookup)
        maj = {j: Counter(y[tr & (tci == j)]).most_common(1)[0][0] for j in range(len(utc))}
        look = np.array([maj[j] for j in tci])
        # traj×env (장면 key) 조회
        se = np.char.add(tc, np.char.add("|", meta["env"].astype(str)))
        use, sei = np.unique(se, return_inverse=True)
        maj2 = {j: Counter(y[tr & (sei == j)]).most_common(1)[0][0] for j in range(len(use)) if (tr & (sei == j)).any()}
        look2 = np.array([maj2.get(j, -1) for j in sei])
        # Cramér's V (궤적 × 라벨)
        tab = np.zeros((len(utc), len(classes)))
        np.add.at(tab, (tci, y), 1)
        ex = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        chi2 = float(((tab - ex) ** 2 / ex).sum())
        V = float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))
        out[tname] = dict(labels_per_traj=sorted(Counter(int((tab[j] > 0).sum()) for j in range(len(utc))).items()),
                          traj_lookup_test_acc=round(float((look[~tr] == y[~tr]).mean()) * 100, 2),
                          traj_env_lookup_test_acc=round(float((look2[~tr] == y[~tr]).mean()) * 100, 2),
                          chance=round(100 / len(classes), 2), cramers_v_traj_label=round(V, 4))
    # 외형조합 (운동 · 궤적 · shape · color · env) — k 를 달리해 다른 block 으로 되풀이된다
    key = np.array([f"{a}|{b}|{c}|{d}|{e}" for a, b, c, d, e in
                    zip(meta["motion"], traj, meta["shape_pre"], meta["color_pre"], meta["env"])])
    uk, ki = np.unique(key, return_inverse=True)
    out["appearance_combo"] = dict(n_combo=int(len(uk)),
                                   test_clips_with_same_combo_in_train=float(np.mean(np.isin(ki[~tr], ki[tr]))))
    return out, key


def combo_split(meta, key, seed=0):
    """block ∪ 외형조합 연결성분 단위 split — test clip 의 (궤적·외형·env) 쌍둥이가 train 에 없다."""
    n = len(key)
    par = np.arange(n)

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for arr in (meta["block_id"].astype(str), key):
        first = {}
        for i, v in enumerate(arr):
            if v in first:
                par[f(i)] = f(first[v])
            else:
                first[v] = i
    comp = np.array([f(i) for i in range(n)])
    uc = np.unique(comp)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uc)
    # 크기 균형: 섞은 순서로 train 이 n/2 에 닿을 때까지
    sizes = {c: int((comp == c).sum()) for c in uc}
    trc, acc = set(), 0
    for c in perm:
        if acc < n / 2:
            trc.add(c); acc += sizes[c]
    tr = np.isin(comp, list(trc))
    # inner hold-out: train 성분의 20 %
    trl = [c for c in perm if c in trc]
    rng2 = np.random.default_rng(seed + 1)
    trl = list(rng2.permutation(trl))
    hold, a2 = set(), 0
    for c in trl:
        if a2 < 0.2 * tr.sum():
            hold.add(c); a2 += sizes[c]
    inner = tr & ~np.isin(comp, list(hold))
    return tr, inner, comp


# ============================================================================ (2)(3) probes
def probe_part(meta, z, p, h, tr, inner, unit, dev, kinds, tag):
    n = len(meta["video_id"])
    k = meta["sym_k"].astype(int)
    vis, late, late2 = k == 0, k >= 1, k >= 2
    te = ~tr
    mot = np.array([P.MOTION[x] for x in meta["motion"]])
    hid = meta["hidden"]
    full_hid = hid[:, 0:16:2] & hid[:, 1:16:2]
    any_hid = hid[:, 0:16:2] | hid[:, 1:16:2]
    assert (full_hid[late2, 7]).all() and not full_hid[vis].any()
    cnt = full_hid.sum(1)
    F = dict(z_all=z.mean(1), p_fut=p.mean(1), h_fut=h[:, 8:].mean(1), z_t7=z[:, 7], z_t67=z[:, 6:8].mean(1))
    F["z_hid"] = np.where(cnt[:, None] > 0, (z * full_hid[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], 0.0)
    vt = ~any_hid
    F["z_visctx"] = (z * vt[..., None]).sum(1) / vt.sum(1)[:, None]
    F["z_t7+p_fut"] = np.concatenate([F["z_t7"], F["p_fut"]], 1)
    F["z_all+p_fut"] = np.concatenate([F["z_all"], F["p_fut"]], 1)
    R = {}
    for tname, col in TARGETS.items():
        classes = sorted(set(meta[col])); y = np.searchsorted(classes, meta[col]); C = len(classes)
        ysh = y.copy(); ix = np.where(tr)[0]; ysh[ix] = np.random.default_rng(0).permutation(y[ix])
        for kind in kinds:
            def FIT(X, m, yy=y):
                if kind == "logistic":
                    return P.train_head(X, yy, m, inner, C, dev)
                if kind == "ridge":
                    return fit_ridge(X, yy, m, inner, C)
                return NcmHead(X, yy, m, C)

            def EV(hd, X, masks):
                c = hd.predict(X) == y
                return {g: rec(c, unit, y, mm, C) for g, mm in masks.items()}, c
            T = R.setdefault(tname, {}).setdefault(kind, {})
            lam = T.setdefault("lam", {})
            # (2)
            h2 = FIT(F["z_all"], tr); lam["z_all"] = h2.lam
            T["E2_z_all"], _ = EV(h2, F["z_all"], {"visible": te & vis, "late": te & late})
            if kind != "ncm":
                hs = FIT(F["z_all"], tr, ysh)
                T["E2_null_label_shuffle"], _ = EV(hs, F["z_all"], {"all": te})
            # (3)
            h3 = FIT(F["z_hid"], tr & late2); lam["z_hid"] = h3.lam
            h3c = FIT(F["z_visctx"], tr & late2); lam["z_visctx"] = h3c.lam
            T["E3_z_hid"], c3 = EV(h3, F["z_hid"], {"late2": te & late2, **{f"late2/{mm}": te & late2 & (mot == mm) for mm in P.MOTIONS}})
            T["E3_z_visctx"], c3c = EV(h3c, F["z_visctx"], {"late2": te & late2})
            T["E3_diff_zhid_minus_visctx"] = dci(c3, c3c, te & late2, unit)
            # (5)
            h5 = FIT(F["p_fut"], tr & late); lam["p_fut_late"] = h5.lam
            T["E5_p_fut_late"], c5 = EV(h5, F["p_fut"], {"late": te & late, "late2": te & late2})
            if kind != "ncm":
                h5s = FIT(F["p_fut"], tr & late, ysh)
                T["E5_null_label_shuffle"], _ = EV(h5s, F["p_fut"], {"late": te & late})
            # (3 of task) 복사 기준선
            hc = {}
            for nm in ("z_t7", "z_t67", "z_all"):
                hc[nm] = FIT(F[nm], tr & late); lam[f"copy_{nm}_late"] = hc[nm].lam
                T[f"COPY_{nm}_late"], cc = EV(hc[nm], F[nm], {"late": te & late, "late2": te & late2})
                T[f"COPY_diff_p_fut_minus_{nm}_late2"] = dci(c5, cc, te & late2, unit)
                T[f"COPY_diff_p_fut_minus_{nm}_late"] = dci(c5, cc, te & late, unit)
                if nm == "z_t7":
                    c7 = cc
            m2 = te & late2
            T["COPY_error_overlap_late2"] = dict(p_right_z7_wrong=int((c5 & ~c7 & m2).sum()),
                                                 p_wrong_z7_right=int((~c5 & c7 & m2).sum()),
                                                 both_wrong=int((~c5 & ~c7 & m2).sum()), n=int(m2.sum()))
            if kind != "ncm":
                for cn, base in (("z_t7+p_fut", "z_t7"), ("z_all+p_fut", "z_all")):
                    hcat = FIT(F[cn], tr & late); lam[f"concat_{cn}"] = hcat.lam
                    T[f"CONCAT_{cn}"], ccat = EV(hcat, F[cn], {"late": te & late, "late2": te & late2})
                    cb = hc[base].predict(F[base]) == y
                    T[f"CONCAT_diff_{cn}_minus_{base}_late2"] = dci(ccat, cb, te & late2, unit)
                    T[f"CONCAT_diff_{cn}_minus_{base}_late"] = dci(ccat, cb, te & late, unit)
            if tag != "main":
                continue
            # (5b) 튜블릿별
            if kind != "ncm":
                T["E5b_p_tubelet_late"], T["E5b_h_tubelet_late"] = [], []
                for t in range(8):
                    for src, X in (("p", p[:, t]), ("h", h[:, 8 + t])):
                        hd = FIT(X, tr & late)
                        T[f"E5b_{src}_tubelet_late"].append(dict(t=t, lam=hd.lam, **EV(hd, X, {"late": te & late})[0]["late"]))
            # (6)
            hh = FIT(F["h_fut"], tr); lam["h_fut"] = hh.lam
            V, L = {"visible": te & vis}, {"late": te & late}
            both = {"visible": te & vis, "late": te & late}
            E6 = T["E6"] = {}
            E6["ref_zhead_on_z_all"] = EV(h2, F["z_all"], both)[0]
            E6["ref_hhead_on_h_fut"] = EV(hh, F["h_fut"], both)[0]
            E6["raw_zhead_on_p"] = EV(h2, F["p_fut"], both)[0]
            E6["raw_hhead_on_p"] = EV(hh, F["p_fut"], both)[0]
            E6["raw_zhead_on_h_fut"] = EV(h2, F["h_fut"], both)[0]
            E6["raw_hhead_on_z_all"] = EV(hh, F["z_all"], both)[0]
            E6["raw_zhead_on_z_t7"] = EV(h2, F["z_t7"], both)[0]
            for pname, hd, Src, Dst in (("p->z", h2, F["p_fut"], F["z_all"]), ("p->h", hh, F["p_fut"], F["h_fut"]),
                                        ("h_fut->z", h2, F["h_fut"], F["z_all"]), ("z->h", hh, F["z_all"], F["h_fut"]),
                                        ("copy_z_t7->z", h2, F["z_t7"], F["z_all"])):
                o = E6[pname] = {}
                for fn, fm, masks in (("fit_vis->vis", tr & vis, V), ("fit_late->late", tr & late, L),
                                      ("fit_vis->late", tr & vis, L), ("fit_all->both", tr, both)):
                    X, rr = procrustes(Src, Dst, fm)
                    o[fn] = dict(**{g: v for g, v in EV(hd, X, masks)[0].items()},
                                 rel_resid_test={g: round(rr(mm), 4) for g, mm in masks.items()})
                X, _ = procrustes(Src, Dst, tr & vis, shuffle=True)
                o["null_shuffled_pairs_vis->vis"] = EV(hd, X, V)[0]
            log(f"{tag} {tname} {kind} done")
    return R


def procrustes(Src, Dst, fm, shuffle=False):
    A = Src[fm].astype(np.float64); B = Dst[fm].astype(np.float64)
    if shuffle:
        B = B[np.random.default_rng(0).permutation(len(B))]
    ms, md = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ms).T @ (B - md))
    Rm = U @ Vt
    X = (Src - ms) @ Rm + md

    def rr(m):
        return float(np.linalg.norm(X[m] - Dst[m]) / np.linalg.norm(Dst[m] - Dst[m].mean(0)))
    return X, rr


# ============================================================================ (4) geometry
def geom_part(meta, z, p, h, tr, n_boot=200, n_perm=50):
    n = len(meta["video_id"])
    X = dict(z_all=z.mean(1).astype(np.float64), p_fut=p.mean(1).astype(np.float64),
             h_fut=h[:, 8:].mean(1).astype(np.float64), h_ctx=h[:, :8].mean(1).astype(np.float64))
    reps = list(X)
    vis = meta["sym_k"].astype(str) == "0"
    cell = np.array([f"{e}|{c}|{k}" for e, c, k in zip(meta["env"], meta["condition"], meta["sym_k"])])
    uc, ci = np.unique(cell, return_inverse=True)
    blk = meta["block_id"].astype(str)
    G = {}

    def vshare(Xr, lab):
        _, inv = np.unique(lab, return_inverse=True)
        mu = Xr.mean(0); tot = ((Xr - mu) ** 2).sum()
        s = 0.0
        for j in range(inv.max() + 1):
            m = inv == j
            s += m.sum() * ((Xr[m].mean(0) - mu) ** 2).sum()
        return float(s / tot)
    G["variance_share_all"] = {r: dict(cell=round(vshare(X[r], cell), 4), shape=round(vshare(X[r], meta["shape_pre"]), 4),
                                       color=round(vshare(X[r], meta["color_pre"]), 4)) for r in reps}
    # resid = 장면 칸 평균 (train clip 만) 제거
    Xr = {}
    for r in reps:
        M = np.stack([X[r][tr & (ci == j)].mean(0) for j in range(len(uc))])
        Xr[r] = X[r] - M[ci]
    G["variance_share_resid"] = {r: dict(shape=round(vshare(Xr[r], meta["shape_pre"]), 4),
                                         color=round(vshare(Xr[r], meta["color_pre"]), 4)) for r in reps}
    # 상수 이동
    pairs = [("z_all", "p_fut"), ("h_fut", "p_fut"), ("h_ctx", "p_fut"), ("z_all", "h_fut"), ("z_all", "h_ctx"), ("h_ctx", "h_fut")]
    G["shift"] = {}
    for a, b in pairs:
        for sub, m in (("visible", vis), ("late", ~vis)):
            A, B = X[a][m], X[b][m]
            dc2 = float(((A.mean(0) - B.mean(0)) ** 2).sum())
            share = dc2 / float(((A - B) ** 2).sum(1).mean())
            mt, mte = m & tr, m & ~tr
            d = X[a][mt].mean(0) - X[b][mt].mean(0); mid = (X[a][mt].mean(0) + X[b][mt].mean(0)) / 2
            accA = ((X[a][mte] - mid) @ d > 0).mean(); accB = ((X[b][mte] - mid) @ d < 0).mean()
            G["shift"][f"{a}|{b}|{sub}"] = dict(offset_share=round(share, 4), centroid_dir_test_acc=round(float((accA + accB) / 2), 4))
    # 부분공간
    def cmeans(Xf, y, C, w):
        """가중 클래스 평균 (중심화). w = clip 가중치 (0 이면 제외, bootstrap 이면 복원추출 횟수)."""
        Y = np.eye(C)[y] * w[:, None]
        M = (Y.T @ Xf) / Y.sum(0)[:, None]
        return M - M.mean(0)

    def basis(M, r):
        return np.linalg.svd(M, full_matrices=False)[2][:r].T

    def cap(M, U):
        return float(((M @ U) ** 2).sum() / (M ** 2).sum())

    def norm_xfit(XA, XB, y, C, mA, mB, trm, w=None):
        """A 는 mA clip, B 는 mB clip. 두 절반 = trm / ~trm. 반환 (xfit captured, split-half A, B, 정규화)."""
        w = np.ones(len(y)) if w is None else w
        r = C - 1
        MA = {hf: cmeans(XA, y, C, w * (mA & s)) for hf, s in (("tr", trm), ("te", ~trm))}
        MB = {hf: cmeans(XB, y, C, w * (mB & s)) for hf, s in (("tr", trm), ("te", ~trm))}
        UA = {k_: basis(v, r) for k_, v in MA.items()}; UB = {k_: basis(v, r) for k_, v in MB.items()}
        x = (cap(MA["te"], UB["tr"]) + cap(MA["tr"], UB["te"]) + cap(MB["te"], UA["tr"]) + cap(MB["tr"], UA["te"])) / 4
        sa = (cap(MA["te"], UA["tr"]) + cap(MA["tr"], UA["te"])) / 2
        sb = (cap(MB["te"], UB["tr"]) + cap(MB["tr"], UB["te"])) / 2
        return x, sa, sb, x / np.sqrt(sa * sb)
    G["subspace"] = {}
    rng = np.random.default_rng(0)
    ublk, binv = np.unique(blk, return_inverse=True)
    members = [[] for _ in ublk]
    for i, bb in enumerate(binv):
        members[bb].append(i)
    by_size = {sz: [bb for bb in range(len(ublk)) if len(members[bb]) == sz] for sz in (1, 2)}
    assert sum(len(v) for v in by_size.values()) == len(ublk)
    boots = [np.bincount(rng.integers(0, len(ublk), len(ublk)), minlength=len(ublk))[binv].astype(float) for _ in range(n_boot)]

    def block_perm(y):
        yp = y.copy()
        for sz, bl in by_size.items():
            src = rng.permutation(bl)
            to = np.concatenate([members[bb] for bb in bl]); fr = np.concatenate([members[bb] for bb in src])
            yp[to] = y[fr]
        return yp
    for view, XX in (("resid", Xr), ("resid_unitnorm", {r: Xr[r] / np.linalg.norm(Xr[r], axis=1, keepdims=True) for r in reps})):
        for tname, col in TARGETS.items():
            classes = sorted(set(meta[col])); y = np.searchsorted(classes, meta[col]); C = len(classes)
            yperms = [block_perm(y) for _ in range(n_perm)] if view == "resid" else []
            for sub, m in (("visible", vis), ("late", ~vis)):
                for a, b in pairs:
                    x, sa, sb, nx = norm_xfit(XX[a], XX[b], y, C, m, m, tr)
                    o = dict(xfit=round(x, 4), split_half_A=round(sa, 4), split_half_B=round(sb, 4), norm=round(nx, 4))
                    if view == "resid":
                        bs = [norm_xfit(XX[a], XX[b], y, C, m, m, tr, w)[3] for w in boots]
                        o["norm_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
                        o["_boot"] = bs
                        pv = [norm_xfit(XX[a], XX[b], yp, C, m, m, tr)[0] for yp in yperms]
                        o["xfit_null_block_perm_p95"] = round(float(np.percentile(pv, 95)), 4)
                    G["subspace"][f"{view}|{tname}|{sub}|{a}|{b}"] = o
                for r in reps:
                    x, sa, sb, nx = norm_xfit(XX[r], XX[r], y, C, vis, ~vis, tr)
                    G["subspace"][f"{view}|{tname}|vis-late|{r}"] = dict(xfit=round(x, 4), norm=round(nx, 4))
            log(f"geom {view} {tname} done")
    # 같은 bootstrap 표본에서 (encoder 쌍 최소) − (p 쌍 최대) 의 분포
    for tname in TARGETS:
        for sub in ("visible", "late"):
            K = lambda a, b: G["subspace"][f"resid|{tname}|{sub}|{a}|{b}"]  # noqa: E731
            enc = [K(a, b) for a, b in pairs if "p_fut" not in (a, b)]
            pp = [K(a, b) for a, b in pairs if "p_fut" in (a, b)]
            gap = np.min([e["_boot"] for e in enc], 0) - np.max([q["_boot"] for q in pp], 0)
            G["subspace"][f"summary|{tname}|{sub}"] = dict(
                encoder_pairs_norm=[e["norm"] for e in enc], p_pairs_norm=[q["norm"] for q in pp],
                gap_minenc_minus_maxp=round(float(min(e["norm"] for e in enc) - max(q["norm"] for q in pp)), 4),
                gap_ci95=[round(float(np.percentile(gap, 2.5)), 4), round(float(np.percentile(gap, 97.5)), 4)])
    for kk in list(G["subspace"]):
        G["subspace"][kk].pop("_boot", None)
    return G



# ============================================================================ (1b) twin 유무로 test 를 나눠 읽기
def twin_part(meta, z, p, h, tr, inner, key, dev):
    """block split 그대로. test clip 을 '같은 외형조합 (운동·궤적·shape·color·env) 이 train 에 있다 (twin)' 과 '없다' 로 나눠 같은 머리로 읽는다.
    combo split 은 block 이 pos_a/pos_b 를 묶어 장면 (궤적×env) 통째 hold-out 이 되므로, 장면은 보고 쌍둥이만 없는 경우를 이것으로 본다."""
    k = meta["sym_k"].astype(int); vis, late, late2 = k == 0, k >= 1, k >= 2
    te = ~tr; block = meta["block_id"].astype(str)
    twin = np.isin(key, key[tr])
    hid = meta["hidden"]; full_hid = hid[:, 0:16:2] & hid[:, 1:16:2]; cnt = full_hid.sum(1)
    F = dict(z_all=z.mean(1), p_fut=p.mean(1), z_t7=z[:, 7])
    F["z_hid"] = np.where(cnt[:, None] > 0, (z * full_hid[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], 0.0)
    items = [("z_all", tr, {"visible": vis, "late": late}), ("z_hid", tr & late2, {"late2": late2}),
             ("p_fut", tr & late, {"late": late, "late2": late2}), ("z_t7", tr & late, {"late": late, "late2": late2})]
    R = dict(n_test=int(te.sum()), n_test_twin=int((te & twin).sum()), n_test_notwin=int((te & ~twin).sum()),
             notwin_by_k={str(kk): int((te & ~twin & (k == kk)).sum()) for kk in range(5)})
    for tname, col in TARGETS.items():
        classes = sorted(set(meta[col])); y = np.searchsorted(classes, meta[col]); C = len(classes)
        for kind in ("ridge", "logistic"):
            T = R.setdefault(tname, {}).setdefault(kind, {})
            for nm, trm, subs in items:
                hd = fit_ridge(F[nm], y, trm, inner, C) if kind == "ridge" else P.train_head(F[nm], y, trm, inner, C, dev)
                c = hd.predict(F[nm]) == y
                o = T[nm] = {}
                for sn, sm in subs.items():
                    o[sn] = dict(twin=rec(c, block, y, te & sm & twin, C), notwin=rec(c, block, y, te & sm & ~twin, C),
                                 diff_notwin_minus_twin=P.diff_ci(c, te & sm & ~twin, c, te & sm & twin, block, paired=False))
            log(f"twin {tname} {kind} done")
    return R

# ============================================================================ (5) xfer — 스크래치 감사 A3 · A4 · (4) 를 레포로 옮김 (2026-09-25)
def _logreg_fixed(Xtr, ytr, Xte, dev, lam=1e-3, iters=300):
    """GEOMETRY §5-2 와 같은 규칙: 표준화 + multinomial logistic, λ=1e-3 고정, L-BFGS."""
    A = torch.as_tensor(Xtr, dtype=torch.float32, device=dev); B = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    u = np.unique(ytr); yi = torch.as_tensor(np.searchsorted(u, ytr), device=dev)
    mu, sd = A.mean(0), A.std(0).clamp_min(1e-6); A, B = (A - mu) / sd, (B - mu) / sd
    W = torch.zeros(A.shape[1], len(u), device=dev, requires_grad=True)
    b = torch.zeros(len(u), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, yi) + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return u[(B @ W + b).argmax(1).cpu().numpy()]


def xfer_part(meta, z, p, h, tr, inner, dev):
    """(4c) 풀링 대조 · (A3) h_fut 로의 Procrustes 상대 잔차 (자 없는 복사 점검) · (A4) G5 visible→late 선형 이식과 p − 다른 표현 차의 block CI (운동별 포함).
    스크래치 `v11_probe_geom_audit.py` 의 A3 · A4 · 4c 와 같은 식이다. 결과는 audit_xfer.json 에만 쓴다."""
    n = len(meta["video_id"]); block = meta["block_id"].astype(str); te = ~tr
    k = meta["sym_k"].astype(int); vis, late, late2 = k == 0, k >= 1, k >= 2
    cond = meta["condition"].astype(str)
    motion = np.where(np.char.startswith(cond, "static"), "static", np.where(np.char.find(cond, "flat") >= 0, "flat", "ramp"))
    hid = meta["hidden"]; full_hid = hid[:, 0:16:2] & hid[:, 1:16:2]; any_hid = hid[:, 0:16:2] | hid[:, 1:16:2]
    z_all, p_fut, h_fut, h_ctx = z.mean(1), p.mean(1), h[:, 8:].mean(1), h[:, :8].mean(1)
    z_t7, z_t67 = z[:, 7], z[:, 6:8].mean(1)
    cnt = full_hid.sum(1)
    z_hid = np.where(cnt[:, None] > 0, (z * full_hid[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], 0.0)
    vis_t = ~any_hid
    z_visctx = (z * vis_t[..., None]).sum(1) / np.maximum(vis_t.sum(1), 1)[:, None]
    R = dict(note="4c: 머리를 한 풀링에서 배워 다른 풀링에 건다 (풀링 불일치 대조). A3: 라벨 없는 Procrustes (train 짝) 뒤 test 상대 잔차 "
                  "‖X̂−h_fut‖/‖h_fut−mean‖. A4: visible train → late test 선형 probe (λ=1e-3 고정), resid = (env×condition×k) 칸 평균 (train clip) 을 뺌.")
    # ---- (4c) 풀링 대조: z_all 머리 (8 튜블릿 평균에서 배움) 를 튜블릿 1–2 개 풀링에 건다
    C4 = R["4c_pooling_control"] = {}
    for tname, col in TARGETS.items():
        classes = sorted(set(meta[col])); y = np.searchsorted(classes, meta[col]); C = len(classes)
        for kind in ("logistic", "ridge"):
            FIT = (lambda X, m: P.train_head(X, y, m, inner, C, dev)) if kind == "logistic" else \
                  (lambda X, m: fit_ridge(X, y, m, inner, C))
            h_all, h_hid = FIT(z_all, tr), FIT(z_hid, tr & late2)

            def ev(hd, X, m):
                return rec(hd.predict(X) == y, block, y, te & m, C)
            C4.setdefault(tname, {})[kind] = dict(
                lam=dict(z_all=h_all.lam, z_hid=h_hid.lam),
                zall_head_on_zall_late2=ev(h_all, z_all, late2), zall_head_on_zhid_late2=ev(h_all, z_hid, late2),
                zall_head_on_visctx_late2=ev(h_all, z_visctx, late2), zall_head_on_t7_visible=ev(h_all, z_t7, vis),
                zall_head_on_t67_visible=ev(h_all, z_t67, vis), zhid_head_on_t7_visible=ev(h_hid, z_t7, vis))
            log(f"xfer 4c {tname} {kind} done")
    # ---- (A3) 풀링 공간 복사 점검: X → h_fut 직교 Procrustes 뒤 test 상대 잔차
    A3 = R["A3_procrustes_to_hfut_rel_residual"] = {}
    for nm, Src in (("p_fut", p_fut), ("z_all (ctx copy)", z_all), ("z_t7 (last ctx tubelet)", z_t7), ("h_ctx", h_ctx)):
        A3[nm] = {}
        for fname, fm in (("fit_visible", tr & vis), ("fit_late", tr & late), ("fit_all", tr)):
            _, rr = procrustes(Src, h_fut, fm)
            A3[nm][fname] = dict(test_vis=round(rr(te & vis), 4), test_late=round(rr(te & late), 4))
    log("xfer A3 done")
    # ---- (A4) G5: visible → late 이식, 운동 합침 + 운동별 (그 운동의 visible → 같은 운동의 가림)
    env = meta["env"].astype(str)
    cell = np.array([f"{e}|{c}|{s}" for e, c, s in zip(env, cond, k)])
    uc, cinv = np.unique(cell, return_inverse=True)

    def resid(X):
        Xr = X.astype(np.float64).copy()
        for c in range(len(uc)):
            m = cinv == c
            Xr[m] -= X[m & tr].mean(0)
        return Xr
    reps = dict(z_all=z_all, p_fut=p_fut, h_fut=h_fut, h_ctx=h_ctx)
    views = {"raw": reps, "resid": {r: resid(X) for r, X in reps.items()}}
    A4 = R["A4_geometry_vis_to_late_linprobe"] = {}
    splits = [("pooled", vis, late)] + [(mo, vis & (motion == mo), late & (motion == mo)) for mo in ("static", "flat", "ramp")]
    for tname, col in TARGETS.items():
        ys = meta[col].astype(str)
        for view, XS in views.items():
            for sname, src, dst in splits:
                if view == "raw" and sname != "pooled":
                    continue
                trm, tem = src & tr, dst & te
                corr = {}
                for rn, X in XS.items():
                    pred = np.full(n, "", dtype=object)
                    pred[tem] = _logreg_fixed(X[trm], ys[trm], X[tem], dev)
                    corr[rn] = pred == ys
                key = f"{tname}|{view}|{sname}"
                A4[key] = {rn: round(100 * float(c[tem].mean()), 2) for rn, c in corr.items()}
                A4[key + "|diff"] = {f"p_fut-{rn}": P.diff_ci(corr["p_fut"], tem, corr[rn], tem, block, paired=True)
                                     for rn in reps if rn != "p_fut"}
                A4[key + "|n_test"] = int(tem.sum())
                log(f"xfer A4 {key} {A4[key]}")
    return R


# ============================================================================ main
def small(R):
    return json.loads(json.dumps(R, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--part", default="all", choices=["all", "traj", "probe", "geom", "twin", "xfer"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    meta, z, p, h = P.load()
    tr = P.repo_split(meta, 0.5, 0)
    inner = P.repo_split(meta, 0.8, 1, subset=tr)
    block = meta["block_id"].astype(str)
    assert not (set(block[tr]) & set(block[~tr]))
    fn = OUT / f"audit_{a.part}.json"
    R = json.loads(fn.read_text()) if fn.exists() else {}
    R["meta"] = dict(script="z_research/scripts/analysis/audit_v11_identity.py", date="2026-09-25",
                     split="WMADataset._split(block_id, 0.5, seed 0, stratify condition); inner 0.8 seed 1",
                     ridge_lams=RIDGE_LAMS, logistic_lams=P.LAMS, n_boot=P.NBOOT)
    traj, key = trajectory_part(meta, tr)
    if a.part in ("all", "traj", "probe"):
        R["1_trajectory"] = traj
        ctr, cinner, comp = combo_split(meta, key)
        R["1_trajectory"]["combo_split"] = dict(n_components=int(len(np.unique(comp))), n_train=int(ctr.sum()),
                                                n_test=int((~ctr).sum()),
                                                block_overlap=int(len(set(block[ctr]) & set(block[~ctr]))),
                                                combo_overlap=int(len(set(key[ctr]) & set(key[~ctr]))))
        log("traj", json.dumps(R["1_trajectory"], ensure_ascii=False)[:600])
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
    if a.part in ("all", "probe"):
        R["2_probe_main"] = small(probe_part(meta, z, p, h, tr, inner, block, dev, ["ridge", "ncm", "logistic"], "main"))
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
        R["1_combo_split_probe"] = small(probe_part(meta, z, p, h, ctr, cinner, comp.astype(str), dev, ["ridge", "logistic"], "combo"))
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
    if a.part == "twin":
        R["1b_twin"] = small(twin_part(meta, z, p, h, tr, inner, key, dev))
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
    if a.part in ("all", "geom"):
        R["4_geometry"] = small(geom_part(meta, z, p, h, tr))
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
    if a.part == "xfer":
        R["5_xfer"] = small(xfer_part(meta, z, p, h, tr, inner, dev))
        fn.write_text(json.dumps(small(R), indent=1, ensure_ascii=False))
    log("wrote", fn)


if __name__ == "__main__":
    main()
