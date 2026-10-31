#!/usr/bin/env python3
r"""[X1] R2 E17 판별 부분공간 이식 + 클래스 평균 대비 — 이식 실패의 몫을 크기 · 방향 · 정보로 나눈다 (H1-S · H1-D · H1-I · H1-F) (2026-10-30).

특징 (phase feats): u_{i,t}(X; O) = LN( mean_{c∈O} LN(X_{i,t,c}) ) ∈ R^1280. O = 화면 전체 (주) · O참 (block 팔 A 의 참 물체 외곽 상자 + 1 칸, 보조).
  X ∈ {h (1440 × 16), p 릴리즈 (1440 × 8), pf PFT (1440 × 8), z late 문맥 (432 × 8, 온라인)}.
LDA (phase fit, CPU): 축소 LDA — 클래스 내 공분산 S_w + λ (tr S_w / d) I, λ ∈ {1e-3 … 10} 는 원천 train block 의 20 % 보류로 고름.
  판별 부분공간 W_X = (S_b, S_w,λ) 의 위 K−1 일반 고유벡터. 분류 = 부분공간 안 공유 공분산 가우스 (LDA 판별).
  a = 그대로 이식 (W_h · 클래스 평균 · 공분산 모두 h train) → p test
  b = 부분공간 이식 (W_h 고정, K−1 차원 안의 클래스 평균 · 공분산만 p train)
  c = p 천장 (W · 평균 · 공분산 모두 p train) · d = h 기준 (h train → 같은 표본의 h test)
  D = (b − 우연) / (c − 우연) · C = (b − a) / (c − a) · I = (c − 우연) / (d − 우연)
파라미터 없는 짝 (test 표본, 같은 (영상, 걸음) 의 h · p): 클래스 평균 대비 벡터 M^X_k = m^X_k − m̄^X.
  κ_cls = mean_k cos(M^p_k, M^h_k) · 반쪽 신뢰도 r_X = 같은 X 의 block 반쪽끼리 κ_cls (무작위 반쪽 20 번 평균) · κ_cls* = κ_cls / √(r_h r_p)
  ρ_cls = ‖M^p‖_F / ‖M^h‖_F
시나리오 (split v2 train → test, 물체 있는 영상): S1 보임 조건 · 이동 · t8–10 보임 · S2 t11–15 · S4 late · 이동 · 다시 보이는 걸음 ·
  S5 encoder 보임 → 가림 (h 의 가려진 튜블릿, 모든 조건) · PFT S2 · PFT S4. 원천 W 는 h 의 보이는 튜블릿 전부 (train block, h 판독기와 같은 학습 표본).
late 판: W 원천을 넷으로 — h_t (h 보이는 튜블릿) · z_7 (온라인 문맥 t7, late 에서는 가려짐) · h_7 (EMA 의 같은 t7) · z_끝 (온라인 마지막 보인 문맥 튜블릿).
  원천 표본 = late · 이동 · 물체 영상 (train block) 의 해당 튜블릿 하나. D_X = (b_X − 우연) / (c − 우연), c = late p 천장.
대조: 라벨 섞기 (b 를 섞은 p train 라벨로) · h 반쪽 D_hh (원천 W = h train 의 block 반쪽, 목표 = 같은 시나리오 h) ≥ 0.9 여야 지표가 쓸 만함.
과제: primitive 모양 (6) · primitive 색 (8) · realistic 모양 (6). 우연 = 1/K.
판정 (PLAN_R2 §2): D ≥ 0.8 · C ≥ 0.5 & κ* ≥ 0.7 & ρ ≤ 0.5 → 크기 / D ≥ 0.8 · I ≤ 0.7 & κ* ≥ 0.7 → 크기 + 정보 감소 / D ≤ 0.4 & κ* ≤ 0.4 → 방향 다름.
  두 열 (학습 · 파라미터 없음) 이 어긋나면 판정 안 함. ±0.05 안이면 경계.
출력: TG/fisher_feats.npz · results/q1q2_r2/E17_fisher.json

  $P z_research/scripts/analysis/transfer_gap_fisher.py feats --gpu 0     # sbatch GPU 1 (IO 가 대부분)
  $P z_research/scripts/analysis/transfer_gap_fisher.py fit                # sbatch CPU
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = C.REPO / "z_analysis/transfer_gap/results/q1q2_r2"
FEATS = C.TG / "fisher_feats.npz"


def feats(gpu):
    import torch
    dev = torch.device(f"cuda:{gpu}") if gpu >= 0 else torch.device("cpu")
    M = C.meta(); bmap, UV, HID, EXT = C.load_tracks()
    LN = lambda x: torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))
    Omask = np.zeros((len(M), NT, S), bool)
    for j, m in enumerate(M):
        bi = bmap[m["block_id"]]
        for t in range(NT):
            Omask[j, t, C.region(UV[bi, 0][t], EXT[bi, 0][t])] = True
    zm = json.load(open(C.TG / "z_late_meta.json"))
    srcs = {"h": (np.load(C.TUBE / "h.npy", mmap_mode="r"), list(range(len(M))), 0),
            "p": (np.load(C.TUBE / "p.npy", mmap_mode="r"), list(range(len(M))), NF),
            "pf": (np.load(C.TG / "p_relftg.npy", mmap_mode="r"), list(range(len(M))), NF),
            "z": (np.load(C.TG / "z_late_ctx.npy", mmap_mode="r"), zm["tp"], 0)}
    out = {"z_tp": np.array(zm["tp"])}
    t0 = time.time()
    for nm, (A, tps, off) in srcs.items():
        n, T = A.shape[0], A.shape[1]
        ua = np.zeros((n, T, D), np.float16); uo = np.zeros((n, T, D), np.float16)
        for q in range(n):
            x = torch.from_numpy(np.asarray(A[q], np.float32)).to(dev)
            xl = LN(x)
            ua[q] = LN(xl.mean(1)).cpu().numpy()
            om = torch.from_numpy(Omask[tps[q], off:off + T]).to(dev).float()          # (T, 256)
            uo[q] = LN((xl * om[..., None]).sum(1) / om.sum(1, keepdim=True).clamp_min(1)).cpu().numpy()
            if q % 200 == 0:
                print(f"{nm} {q}/{n} {time.time() - t0:.0f}s", flush=True)
        out[f"u_{nm}_all"] = ua; out[f"u_{nm}_O"] = uo
    np.savez(FEATS, **out)
    print("→", FEATS, f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------ LDA
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def lda_fit(X, y, K, lam):
    mu = X.mean(0); Xc = X - mu
    means = np.stack([X[y == k].mean(0) if (y == k).any() else mu for k in range(K)])
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for k in range(K):
        if (y == k).any():
            R = X[y == k] - means[k]; Sw += R.T @ R
    Sw /= max(len(X) - K, 1)
    Sw += lam * np.trace(Sw) / X.shape[1] * np.eye(X.shape[1])
    Mb = means - mu
    cnt = np.array([(y == k).sum() for k in range(K)], float)
    Sb = (Mb * cnt[:, None]).T @ Mb / len(X)
    L = np.linalg.cholesky(Sw); Li = np.linalg.inv(L)
    ev, V = np.linalg.eigh(Li @ Sb @ Li.T)
    o = np.argsort(ev)[::-1][: K - 1]
    W = Li.T @ V[:, o]                     # (d, K−1)
    return W


def sub_clf(Z, y, K):
    """부분공간 안 LDA 판별: 클래스 평균 + 공유 공분산."""
    means = np.stack([Z[y == k].mean(0) if (y == k).any() else np.full(Z.shape[1], 1e6) for k in range(K)])
    R = Z - means[y]
    Sg = R.T @ R / max(len(Z) - K, 1) + 1e-6 * np.eye(Z.shape[1])
    Si = np.linalg.inv(Sg)
    def pred(Zt):
        sc = Zt @ Si @ means.T - 0.5 * np.einsum("kd,de,ke->k", means, Si, means)[None]
        return sc.argmax(1)
    return pred


def choose_lam(X, y, K, grp, seed=0):
    ub = np.unique(grp); rng = np.random.default_rng(seed); rng.shuffle(ub)
    ho = np.isin(grp, ub[: max(1, len(ub) // 5)])
    best, bl = -1, LAMS[0]
    for lam in LAMS:
        W = lda_fit(X[~ho], y[~ho], K, lam)
        acc = (sub_clf(X[~ho] @ W, y[~ho], K)(X[ho] @ W) == y[ho]).mean()
        if acc > best:
            best, bl = acc, lam
    return bl


def lda_full(X, y, K, grp):
    lam = choose_lam(X, y, K, grp)
    W = lda_fit(X, y, K, lam)
    return W, lam


def class_contrast(X, y, K):
    m = np.stack([X[y == k].mean(0) if (y == k).any() else np.zeros(X.shape[1]) for k in range(K)])
    present = np.array([(y == k).any() for k in range(K)])
    return m - X.mean(0), present


def kappa(A, B, present):
    a, b = A[present], B[present]
    return float(np.mean([np.dot(x, z) / max(np.linalg.norm(x) * np.linalg.norm(z), 1e-12) for x, z in zip(a, b)]))


def half_rel(X, y, K, grp, n=20, seed=0):
    ub = np.unique(grp); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        p = rng.permutation(ub); h1 = np.isin(grp, p[: len(ub) // 2])
        A, pa = class_contrast(X[h1], y[h1], K); B, pb = class_contrast(X[~h1], y[~h1], K)
        vals.append(kappa(A, B, pa & pb))
    return float(np.mean(vals))


def paired_effects(X, y, K, keys):
    """장면을 뺀 클래스 효과: 같은 (block, 걸음) 의 두 팔 차이 Δ = u_A − u_B ≈ β_kA − β_kB 를 최소 노름 최소제곱으로 푼다 (Σβ = 0).
    keys[i] = (block, t, variant). 반환 β (K, d) · 쓰인 클래스 · block 목록 · 행 (Δ, 설계) — 반쪽 신뢰도용."""
    pos = {k: i for i, k in enumerate(keys)}
    rows, des, blks = [], [], []
    for (b, t, v), i in pos.items():
        if v != "pos_a" or (b, t, "pos_b") not in pos:
            continue
        i2 = pos[(b, t, "pos_b")]
        if y[i] == y[i2]:
            continue
        d = np.zeros(K); d[y[i]] += 1; d[y[i2]] -= 1
        rows.append(X[i] - X[i2]); des.append(d); blks.append(b)
    if len(rows) < 10:
        return None
    Dm = np.stack(des); R = np.stack(rows); blks = np.array(blks)
    return Dm, R, blks


def beta_of(Dm, R):
    return np.linalg.pinv(Dm) @ R          # (K, d), 최소 노름 → Σβ = 0


def paired_stats(Xp, Xh, y, K, keys, n=20, seed=0):
    a = paired_effects(Xp, y, K, keys); b = paired_effects(Xh, y, K, keys)
    if a is None or b is None:
        return None
    Dm, Rp, blks = a; _, Rh, _ = b
    present = np.abs(Dm).sum(0) > 0
    Bp, Bh = beta_of(Dm, Rp), beta_of(Dm, Rh)
    kc = kappa(Bp, Bh, present)
    rho = float(np.linalg.norm(Bp[present]) / max(np.linalg.norm(Bh[present]), 1e-12))
    ub = np.unique(blks); rng = np.random.default_rng(seed); rp, rh = [], []
    for _ in range(n):
        pr = rng.permutation(ub); h1 = np.isin(blks, pr[: len(ub) // 2])
        pres = (np.abs(Dm[h1]).sum(0) > 0) & (np.abs(Dm[~h1]).sum(0) > 0)
        rp.append(kappa(beta_of(Dm[h1], Rp[h1]), beta_of(Dm[~h1], Rp[~h1]), pres))
        rh.append(kappa(beta_of(Dm[h1], Rh[h1]), beta_of(Dm[~h1], Rh[~h1]), pres))
    rp, rh = float(np.mean(rp)), float(np.mean(rh))
    return dict(kappa_pair=kc, r_p_pair=rp, r_h_pair=rh, kappa_pair_star=kc / np.sqrt(rp * rh) if rp > 0 and rh > 0 else float("nan"),
                rho_pair=rho, n_pairs=len(Dm), n_block=len(ub))


def fit():
    Fz = np.load(FEATS)
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    ztp = list(Fz["z_tp"]); zslot = {int(j): q for q, j in enumerate(ztp)}
    t0 = time.time()
    tasks = [("primitive", "shape"), ("primitive", "color"), ("realistic", "shape")]
    out = {"definition": __doc__, "tasks": {}}
    for pool in ("all", "O"):
        U = {nm: Fz[f"u_{nm}_{pool}"].astype(np.float64) for nm in ("h", "p", "pf", "z")}
        for objset, attr in tasks:
            key = f"{objset}|{attr}|{pool}"
            idx = [j for j, m in enumerate(M) if m["objset"] == objset and C.label_of(m) != "empty"]
            lab = lambda j: C.label_of(M[j]).split("/")[0 if attr == "shape" else 1]
            classes = sorted({lab(j) for j in idx}); K = len(classes); ci = {c: i for i, c in enumerate(classes)}
            chance = 1.0 / K
            st = {j: C.tub_state(M[j]["samples"]) for j in idx}
            tr = lambda j: M[j]["block_id"] in trb
            mv = lambda j: M[j]["motion"] != "flat_static"
            late = lambda j: M[j]["occlusion"] == "late"
            # 원천: h 보이는 튜블릿 (train)
            src = [(j, t) for j in idx if tr(j) for t in range(NT) if st[j][t] == "vis"]
            Xh = np.stack([U["h"][j, t] for j, t in src]); yh = np.array([ci[lab(j)] for j, _ in src]); gh = np.array([M[j]["block_id"] for j, _ in src])
            Wh, lam_h = lda_full(Xh, yh, K, gh)
            clf_h = sub_clf(Xh @ Wh, yh, K)
            scen = {
                "S1": ("p", lambda j, t: not late(j) and mv(j) and 8 <= t <= 10 and st[j][t] == "vis"),
                "S2": ("p", lambda j, t: not late(j) and mv(j) and t >= 11 and st[j][t] == "vis"),
                "S4": ("p", lambda j, t: late(j) and mv(j) and t >= 8 and st[j][t] == "vis"),
                "PFT_S2": ("pf", lambda j, t: not late(j) and mv(j) and t >= 11 and st[j][t] == "vis"),
                "PFT_S4": ("pf", lambda j, t: late(j) and mv(j) and t >= 8 and st[j][t] == "vis"),
                "S5": ("hocc", lambda j, t: st[j][t] == "occ"),
            }
            res = {"K": K, "chance": chance, "lam_h": lam_h, "n_src": len(src), "scen": {}}
            for sn, (xs, f) in scen.items():
                trs = [(j, t) for j in idx if tr(j) for t in range(NT) if f(j, t)]
                tes = [(j, t) for j in idx if not tr(j) for t in range(NT) if f(j, t)]
                if len(tes) < 20 or len(trs) < 20:
                    continue
                if xs == "hocc":
                    get = lambda j, t: U["h"][j, t]
                else:
                    get = (lambda arr: (lambda j, t: U[arr][j, t - NF]))(xs)
                Xtr = np.stack([get(j, t) for j, t in trs]); ytr = np.array([ci[lab(j)] for j, _ in trs]); gtr = np.array([M[j]["block_id"] for j, _ in trs])
                Xte = np.stack([get(j, t) for j, t in tes]); yte = np.array([ci[lab(j)] for j, _ in tes]); gte = np.array([M[j]["block_id"] for j, _ in tes])
                Hte = np.stack([U["h"][j, t] for j, t in tes]); Htr = np.stack([U["h"][j, t] for j, t in trs])
                a = float((clf_h(Xte @ Wh) == yte).mean())
                b = float((sub_clf(Xtr @ Wh, ytr, K)(Xte @ Wh) == yte).mean())
                Wp, lam_p = lda_full(Xtr, ytr, K, gtr)
                c = float((sub_clf(Xtr @ Wp, ytr, K)(Xte @ Wp) == yte).mean())
                d = float((clf_h(Hte @ Wh) == yte).mean())
                rng = np.random.default_rng(0)
                b_shuf = float(np.mean([(sub_clf(Xtr @ Wh, rng.permutation(ytr), K)(Xte @ Wh) == yte).mean() for _ in range(5)]))
                # h 반쪽 D_hh: 원천 W = h train block 반쪽, 목표 = 같은 시나리오의 h (다른 반쪽 train 으로 맞춤)
                ubt = np.unique(gh); rr = np.random.default_rng(1).permutation(ubt); h1 = set(rr[: len(rr) // 2])
                s1 = np.isin(gh, list(h1))
                Wh1, _ = lda_full(Xh[s1], yh[s1], K, gh[s1])
                t2 = ~np.isin(gtr, list(h1))
                bhh = float((sub_clf(Htr[t2] @ Wh1, ytr[t2], K)(Hte @ Wh1) == yte).mean()) if t2.sum() > 20 else float("nan")
                Wh2, _ = lda_full(Htr[t2], ytr[t2], K, gtr[t2]) if t2.sum() > 20 else (None, None)
                chh = float((sub_clf(Htr[t2] @ Wh2, ytr[t2], K)(Hte @ Wh2) == yte).mean()) if Wh2 is not None else float("nan")
                Dd = lambda bb, cc: (bb - chance) / (cc - chance) if abs(cc - chance) > 1e-9 else float("nan")
                # 파라미터 없음
                Mp, pp = class_contrast(Xte, yte, K); Mh, ph = class_contrast(Hte, yte, K)
                kc = kappa(Mp, Mh, pp & ph)
                rp = half_rel(Xte, yte, K, gte); rh = half_rel(Hte, yte, K, gte)
                rho = float(np.linalg.norm(Mp[pp & ph]) / max(np.linalg.norm(Mh[pp & ph]), 1e-12))
                vio = [j for j, _ in tes]
                keep = np.array([M[j]["violation"] == attr for j in vio])
                kk = [(M[j]["block_id"], t, M[j]["variant"]) for (j, t), q in zip(tes, keep) if q]
                ps = paired_stats(Xte[keep], Hte[keep], yte[keep], K, kk) if keep.sum() > 10 else None
                e = dict(n_train=len(trs), n_test=len(tes), n_test_block=len(np.unique(gte)), lam_p=lam_p,
                         a=a, b=b, c=c, d=d, b_shuffled=b_shuf, D=Dd(b, c), C=(b - a) / (c - a) if abs(c - a) > 1e-9 else float("nan"),
                         I=Dd(c, d), D_hh=Dd(bhh, chh), b_hh=bhh, c_hh=chh,
                         kappa_cls=kc, r_p=rp, r_h=rh, kappa_cls_star=kc / np.sqrt(max(rp, 1e-6) * max(rh, 1e-6)) if rp > 0 and rh > 0 else float("nan"),
                         rho_cls=rho, **(ps or {}))
                # ⚠️ 클래스 평균 대비 (kappa_cls · r · κ*) 는 장면 구성에 지배된다 (반쪽 신뢰도 ≈ 0.04) → 판정은 쌍 차이 판 (kappa_pair_star · rho_pair) 으로
                # 판정
                Dv, Cv, Iv, ks = e["D"], e["C"], e["I"], e.get("kappa_pair_star", float("nan"))
                rho = e.get("rho_pair", float("nan"))
                lab_ = "판정 안 함 (두 열 어긋남)"
                if Dv >= 0.8 and Cv >= 0.5 and ks >= 0.7 and rho <= 0.5:
                    lab_ = "크기 (순수)"
                elif Dv >= 0.8 and Iv <= 0.7 and ks >= 0.7:
                    lab_ = "크기 + 정보 감소"
                elif Dv <= 0.4 and ks <= 0.4:
                    lab_ = "방향 다름"
                elif Dv >= 0.8 and ks < 0.7:
                    lab_ = "판정 안 함 (D 는 같은 방향, κ* 는 아님)"
                elif Dv <= 0.4 and ks > 0.4:
                    lab_ = "판정 안 함 (D 는 다른 방향, κ* 는 아님)"
                elif 0.4 < Dv < 0.8:
                    lab_ = "중간 (D 0.4–0.8)"
                e["verdict"] = lab_
                res["scen"][sn] = e
                print(key, sn, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k in ("a", "b", "c", "d", "D", "C", "I", "D_hh", "kappa_cls", "r_p", "kappa_pair", "r_p_pair", "r_h_pair", "kappa_pair_star", "rho_pair", "b_shuffled", "verdict")},
                      f"{time.time() - t0:.0f}s", flush=True)
            # late 판 — W 원천 넷
            lt = [j for j in idx if late(j) and mv(j)]
            lt_tr = [j for j in lt if tr(j) and j in zslot]
            trs = [(j, t) for j in idx if tr(j) for t in range(NF, NT) if late(j) and mv(j) and st[j][t] == "vis"]
            tes = [(j, t) for j in idx if not tr(j) for t in range(NF, NT) if late(j) and mv(j) and st[j][t] == "vis"]
            if trs and tes and lt_tr:
                Xtr = np.stack([U["p"][j, t - NF] for j, t in trs]); ytr = np.array([ci[lab(j)] for j, _ in trs]); gtr = np.array([M[j]["block_id"] for j, _ in trs])
                Xte = np.stack([U["p"][j, t - NF] for j, t in tes]); yte = np.array([ci[lab(j)] for j, _ in tes])
                Wp, _ = lda_full(Xtr, ytr, K, gtr); c = float((sub_clf(Xtr @ Wp, ytr, K)(Xte @ Wp) == yte).mean())
                def tl_of(j):
                    bi = bmap[M[j]["block_id"]]; arm = 0 if M[j]["variant"] == "pos_a" else 1
                    return C.last_visible_ctx(HID[bi, arm])
                srcs = {"h_t": (Xh, yh, gh),
                        "z_7": (np.stack([U["z"][zslot[j], 7] for j in lt_tr]),),
                        "h_7": (np.stack([U["h"][j, 7] for j in lt_tr]),),
                        "z_end": (np.stack([U["z"][zslot[j], tl_of(j)] for j in lt_tr]),)}
                ylt = np.array([ci[lab(j)] for j in lt_tr]); glt = np.array([M[j]["block_id"] for j in lt_tr])
                lres = {"c_late_p": c, "chance": chance, "n_train": len(trs), "n_test": len(tes), "n_src_late": len(lt_tr)}
                for sname, tup in srcs.items():
                    Xs, ys, gs = (tup if len(tup) == 3 else (tup[0], ylt, glt))
                    Ws, lam_s = lda_full(Xs, ys, K, gs)
                    bX = float((sub_clf(Xtr @ Ws, ytr, K)(Xte @ Ws) == yte).mean())
                    # 원천 자기 판독 (원천 표본 5 겹 대신 train 적합도) — 원천에 정체 정보가 있는지
                    lres[sname] = dict(b=bX, D=(bX - chance) / (c - chance) if abs(c - chance) > 1e-9 else float("nan"), lam=lam_s, n_src=len(Xs))
                # late 원천의 encoder 자기 판독 (test block 의 같은 튜블릿, W · 분류 모두 원천 train)
                lt_te = [j for j in lt if not tr(j) and j in zslot]
                if lt_te:
                    yte2 = np.array([ci[lab(j)] for j in lt_te])
                    for sname, getf in (("z_7", lambda j: U["z"][zslot[j], 7]), ("h_7", lambda j: U["h"][j, 7]), ("z_end", lambda j: U["z"][zslot[j], tl_of(j)])):
                        Xs = np.stack([getf(j) for j in lt_tr]); Ws, _ = lda_full(Xs, ylt, K, glt)
                        Xt2 = np.stack([getf(j) for j in lt_te])
                        lres[sname]["self_acc"] = float((sub_clf(Xs @ Ws, ylt, K)(Xt2 @ Ws) == yte2).mean())
                res["late_sources"] = lres
                print(key, "late", {k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)}) for k, v in lres.items()}, flush=True)
            out["tasks"][key] = res
            json.dump(out, open(OUT / "E17_fisher.json", "w"), indent=1, ensure_ascii=False, default=float)
    print("→", OUT / "E17_fisher.json", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["feats", "fit"])
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feats(a.gpu) if a.phase == "feats" else fit()
