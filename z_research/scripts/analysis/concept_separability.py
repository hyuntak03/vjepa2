#!/usr/bin/env python3
"""토큰 캐시에서 개념 분리도를 **닫힌 형식**으로 잰다. 최적화가 개입하지 않는다.

왜 필요한가
-----------
attentive probe 는 AdamW 상수 lr 로 학습한다. v10 실측에서 pred/static_occlusion/color
head 는 30 epoch 동안 loss 가 ln(8)=2.079 근처(균등해)에 앉아 있다가 ep33 부터 겨우
탈출하는 중에 학습이 끝났다 (train_acc 0.32). 그 18.8% 는 표현의 성질이 아니라
**정체(plateau)에서 잘린 값**이다. 그래서 optimizer 가 없는 지표로 다시 잰다.

재는 것 (전부 결정론적)
  fisher   tr(S_B)/tr(S_W)   클래스 간 분산 / 클래스 내 분산. 학습 없음.
  ridge    W = (X^T X + lam I)^-1 X^T Y 의 argmax 정확도. 역행렬 한 번.
  cos      클래스 평균 방향(= 개념 벡터) 쌍들의 평균 코사인. 낮을수록 잘 갈라짐.

⚠️ 토큰을 평균으로 접는다. 공간·시간축을 버리므로 attentive probe 의 **하한**이다.
   여기서 이미 잘 갈라지면 "정보는 있다"가 확정된다. 반대로 낮게 나와도
   attentive 가 못 읽는다는 뜻은 아니다 (풀링이 버린 것일 수 있다).

  python z_research/scripts/analysis/concept_separability.py --base predictor --targets shape color
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]


def pooled(cache: Path, base: str, rows_idx, chunk=64, half=None):
    """half='future' 면 뒤 절반 토큰만 (target 은 32프레임 전부라 p 와 맞추려면 필요)."""
    m = json.loads((cache / "meta.json").read_text())
    n, t, d = len(m["video_ids"]), m["base_counts"][base], m["embed_dim"]
    mm = np.memmap(cache / f"{base}.npy", dtype=np.float16, mode="r", shape=(n, t, d))
    sl = slice(t // 2, t) if half == "future" else slice(0, t)
    out = np.empty((len(rows_idx), d), np.float32)
    for s in range(0, len(rows_idx), chunk):
        sel = rows_idx[s:s + chunk]
        out[s:s + chunk] = np.asarray(mm[sel, sl], np.float32).mean(1)
    return out


def concept_vectors(X, y, K):
    """개념 벡터 = 클래스 평균 - 전체 평균, 단위 정규화."""
    mu = X.mean(0)
    V = np.full((K, X.shape[1]), np.nan, np.float32)
    for k in range(K):
        if (y == k).sum():
            v = X[y == k].mean(0) - mu
            V[k] = v / (np.linalg.norm(v) + 1e-9)
    return V


def cos_clf(V, X, y):
    """개념 벡터에 대한 코사인 최근접 분류. 공분산(백색화)을 전혀 안 쓴다.
    ridge 이식이 무너져도 이게 살아 있으면 '방향은 공유, 백색화만 어긋남' 이다."""
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    ok = ~np.isnan(V[:, 0])
    return float((np.asarray(np.nonzero(ok)[0])[(Z @ V[ok].T).argmax(1)] == y).mean())


def ridge_fit(X, y, K, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ np.eye(K)[y])


def ridge_apply(W, X, y):
    return float(((np.c_[X, np.ones(len(X))] @ W).argmax(1) == y).mean())


def align_mode(a, rows, idx, cond, block, lab, K):
    """h(target enc, 미래 절반) 와 p(predictor) 의 개념 방향이 같은가.

    두 지표 모두 학습 반복이 없다.
      cos diag : 같은 클래스의 개념 벡터끼리 코사인 (높을수록 정렬)
          off  : 다른 클래스끼리 (기준선 — diag 가 이보다 유의하게 커야 의미가 있다)
      h -> p   : h 로 닫힌 해 ridge 를 풀어 p 에 그대로 적용한 정확도 (= 이식)
                 p self 와의 격차가 곧 정렬손실이다.
    ⚠️ 두 공간을 각자 표준화한 뒤 비교한다. p 는 LN 을 안 받아 |p|/|h| ~ 0.60 이라
       (§5-4b) 크기를 맞추지 않으면 이식이 크기 차이만 재게 된다.
    """
    Xh = pooled(a.cache, "target", idx, half="future")
    Xp = pooled(a.cache, "predictor", idx)
    if a.std == "none":
        # 채점 규칙(latent L1)은 표준화를 안 한다. "왜 채점이 실패하나"를 물을 때는
        # 이쪽이 맞다. 대신 ViT 의 소수 거대 활성 차원이 코사인·Fisher·ridge 를
        # 지배한다 (표준화는 그걸 눌러 정보 기하를 보게 해 준다).
        mu, sd = np.zeros(Xh.shape[1], np.float32), np.ones(Xh.shape[1], np.float32)
    else:
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
    Xh = (Xh - mu) / sd
    if a.restd:
        # ⚠️ p 를 **자기** 통계로 다시 표준화하면 차원별 재척도가 공짜로 들어간다.
        #    그건 §5-4b 가 지목한 진폭 압축(|p|/|h| ~ 0.60)을 지워 주는 것이라
        #    h->p 이식이 관대해진다. 기본은 h 의 통계를 그대로 쓴다.
        Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)
    else:
        Xp = (Xp - mu) / sd
    for t in a.targets:
        print(f"\n### {t}   (chance {100/K[t]:.1f}%)   h = target enc 미래절반, p = predictor")
        print(f"  {'condition':<20}{'cos diag':>10}{'cos off':>9}"
              f"{'h self':>9}{'p self':>9}{'h -> p':>9}{'p -> h':>9}"
              f"{'cosH@h':>8}{'cosH@p':>8}{'cosP@p':>8}")
        for c in list(CONDS) + ["ALL"]:
            m = np.ones(len(rows), bool) if c == "ALL" else (cond == c)
            yh = lab[t][m]
            Vh = concept_vectors(Xh[m], yh, K[t])
            Vp = concept_vectors(Xp[m], yh, K[t])
            C = Vh @ Vp.T
            iu = ~np.eye(K[t], dtype=bool)
            g = np.array(block)[m]
            uq = np.unique(g); rng = np.random.default_rng(0)
            part = {b: i % 5 for i, b in enumerate(rng.permutation(uq))}
            fold = np.array([part[b] for b in g])
            hs = ps = hp = ph = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                Wh = ridge_fit(Xh[m][tr], yh[tr], K[t], a.lam)
                Wp = ridge_fit(Xp[m][tr], yh[tr], K[t], a.lam)
                hs += ridge_apply(Wh, Xh[m][te], yh[te]) / 5
                ps += ridge_apply(Wp, Xp[m][te], yh[te]) / 5
                hp += ridge_apply(Wh, Xp[m][te], yh[te]) / 5
                ph += ridge_apply(Wp, Xh[m][te], yh[te]) / 5
            # 개념 벡터는 train fold 에서만 만들어 test fold 에서 평가한다 (누수 차단)
            ch = cp = pp = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                VhT = concept_vectors(Xh[m][tr], yh[tr], K[t])
                VpT = concept_vectors(Xp[m][tr], yh[tr], K[t])
                ch += cos_clf(VhT, Xh[m][te], yh[te]) / 5
                cp += cos_clf(VhT, Xp[m][te], yh[te]) / 5      # h 개념벡터 -> p 표본
                pp += cos_clf(VpT, Xp[m][te], yh[te]) / 5
            print(f"  {c:<20}{np.nanmean(np.diag(C)):>10.3f}{np.nanmean(C[iu]):>9.3f}"
                  f"{100*hs:>9.2f}{100*ps:>9.2f}{100*hp:>9.2f}{100*ph:>9.2f}"
                  f"{100*ch:>8.2f}{100*cp:>8.2f}{100*pp:>8.2f}")


def fisher(X, y, K):
    mu = X.mean(0)
    sb = sw = 0.0
    for k in range(K):
        Z = X[y == k]
        if len(Z) < 2:
            continue
        mk = Z.mean(0)
        sb += len(Z) * float(((mk - mu) ** 2).sum())
        sw += float(((Z - mk) ** 2).sum())
    return sb / max(sw, 1e-9)


def ridge_cv(X, y, K, groups, lam=1e2, folds=5):
    """block 단위 k-fold. 닫힌 해라 fold 를 늘려도 싸다."""
    g = np.array(groups)
    uq = np.unique(g)
    rng = np.random.default_rng(0)
    part = {b: i % folds for i, b in enumerate(rng.permutation(uq))}
    fold = np.array([part[b] for b in g])
    acc = []
    for f in range(folds):
        tr, te = fold != f, fold == f
        Xtr = np.c_[X[tr], np.ones(tr.sum())]
        Y = np.eye(K)[y[tr]]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ Y)
        pred = (np.c_[X[te], np.ones(te.sum())] @ W).argmax(1)
        acc.append(float((pred == y[te]).mean()))
    return float(np.mean(acc))


def cos_sep(X, y, K):
    """개념 벡터 = 클래스 평균 - 전체 평균. 쌍 코사인의 평균."""
    mu = X.mean(0)
    V = np.stack([X[y == k].mean(0) - mu for k in range(K) if (y == k).sum()])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    C = V @ V.T
    iu = np.triu_indices(len(V), 1)
    return float(C[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/local_datasets/world/world_analysis/cache/v10_vith"))
    ap.add_argument("--index", type=Path,
                    default=Path("data_csv/intphysgen_v10/index_probe.csv"))
    ap.add_argument("--bases", nargs="*", default=["ctx_masked", "target", "predictor"])
    ap.add_argument("--targets", nargs="*", default=["shape", "color"])
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--std", choices=["h", "none"], default="h",
                    help="h = h 통계로 두 공간을 표준화(기본) / none = 원본 그대로")
    ap.add_argument("--restd", action="store_true",
                    help="p 를 자기 통계로 다시 표준화 (이식이 관대해진다 — 비교용)")
    ap.add_argument("--align", action="store_true",
                    help="h(target enc) 와 p(predictor) 의 개념 방향 정렬을 잰다")
    a = ap.parse_args()

    vids = json.loads((a.cache / "meta.json").read_text())["video_ids"]
    pos = {v: i for i, v in enumerate(vids)}
    rows = [r for r in csv.DictReader(a.index.open()) if r["probe_type"] == "obj"]
    idx = np.array([pos[r["video_id"]] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    block = [r["block_id"] for r in rows]
    COL = {"shape": "shape_pre", "color": "color_pre"}
    lab, K = {}, {}
    for t in a.targets:
        cls = sorted({r[COL[t]] for r in rows})
        K[t] = len(cls)
        ci = {c: i for i, c in enumerate(cls)}
        lab[t] = np.array([ci[r[COL[t]]] for r in rows])

    print(f"obj {len(rows)}행 | 토큰 평균 풀링 | ridge lam={a.lam:g}, block 5-fold")
    if a.align:
        align_mode(a, rows, idx, cond, block, lab, K)
        return
    print()
    for base in a.bases:
        X = pooled(a.cache, base, idx)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)           # 차원별 표준화
        for t in a.targets:
            print(f"{base} / {t}   (chance {100/K[t]:.1f}%)")
            print(f"  {'condition':<20}{'fisher':>9}{'ridge acc':>11}{'mean cos':>10}{'n':>6}")
            for c in list(CONDS) + ["ALL"]:
                m = np.ones(len(rows), bool) if c == "ALL" else (cond == c)
                Xa, ya = X[m], lab[t][m]
                print(f"  {c:<20}{fisher(Xa, ya, K[t]):>9.4f}"
                      f"{100*ridge_cv(Xa, ya, K[t], np.array(block)[m], a.lam):>11.2f}"
                      f"{cos_sep(Xa, ya, K[t]):>10.3f}{m.sum():>6}")
            print()


if __name__ == "__main__":
    main()
