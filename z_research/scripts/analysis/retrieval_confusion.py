#!/usr/bin/env python
"""predictor 가 실제로 무엇을 만들었나 — 7(8)-way retrieval.

채점(matched pairing)은 후보가 **둘뿐**이다: 그 block 이 마침 짝지어 준 A 와 B.
그래서 "torus 를 못 지킨다" 가 "cube 를 만든다" 인지 "그냥 아무 쪽으로나 샌다" 인지
구분되지 않는다.

여기서는 후보를 **전부** 준다:
    문맥이 S 인 clip 의 predictor 출력 p 에 대해
    d(S') = mean |p - h_bar(S')|   를 7(8)개 전부 계산하고 argmin 을 고른다.
argmin == S 면 "지켰다". 아니면 **어느 쪽으로 샜는지** 가 나온다 -> 진짜 confusion matrix.

h_bar(S') 는 같은 (조건, k) 안에서 미래 물체가 S' 인 clip 들의 LN(h)[미래] 평균이다.
같은 그룹 안에서 만들므로 배경·위치·가림막 같은 성분이 상쇄된다 (v11 은 배경을 균등 교차).
질의 clip 자신은 자기 프로토타입에서 **뺀다** (leave-one-out).

⚠️ 캐시의 `target` 은 이미 affine-free LN 이 적용돼 있다 (forward.py `_ln`).
   `predictor` 는 원본이다 — 학습이 그렇게 회귀했으므로 다시 정규화하지 않는다.

⚠️ 전역 평균 mu 도 후보에 넣어 함께 보고한다. p 가 어느 물체보다 mu 에 가깝다면
   "특정 물체를 만든다" 가 아니라 "평균으로 붕괴한다" 이다.

  python z_research/scripts/analysis/retrieval_confusion.py --target shape \
      --conditions moving_visible moving_occlusion
"""
from __future__ import annotations
import argparse, collections, csv, json, os, sys, time
import numpy as np

CACHE = "/local_datasets/world/world_analysis/cache/v11_vith"
INDEX = "data_csv/intphysgen_v11/index_probe.csv"
COL = {"shape": "shape_pre", "color": "color_pre"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--target", default="shape", choices=list(COL))
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--by-k", action="store_true", help="조건 안에서 k 로 더 쪼갠다")
    ap.add_argument("--limit-per-group", type=int, default=0, help="0 = 전부")
    ap.add_argument("--query", default="p", choices=["p", "h", "z"],
                    help="무엇을 질의로 쓸까. h = 실제 미래(대조군, 100%에 가까워야 한다), "
                         "z = 문맥 인코더(문맥 잔상 검사)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(f"{a.cache}/meta.json"))
    pos = {v: i for i, v in enumerate(meta["video_ids"])}
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")     # (N, 2048, 1280)
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")        # (N, 4096, 1280) LN 완료
    Z = np.load(f"{a.cache}/ctx_masked.npy", mmap_mode="r")    # (N, 2048, 1280) 문맥
    off = H.shape[1] - P.shape[1]                              # 미래 토큰 시작
    QS = {"p": lambda i: P[i], "h": lambda i: H[i, off:], "z": lambda i: Z[i]}[a.query]
    print(f"cache {a.cache}   질의 = {a.query}\n  predictor {P.shape}  target {H.shape}"
          f"  -> 미래 = h[{off}:]")

    rows = [r for r in csv.DictReader(open(a.index))
            if r["probe_type"] == "obj" and r["violation_type"] == a.target]
    groups = collections.defaultdict(list)
    for r in rows:
        key = (r["condition"], r["sym_k"]) if a.by_k else (r["condition"],)
        if a.conditions and r["condition"] not in a.conditions:
            continue
        groups[key].append(r)

    classes = sorted({r[COL[a.target]] for r in rows})
    ci = {c: i for i, c in enumerate(classes)}
    out = {}
    for key in sorted(groups):
        g = groups[key]
        idx = np.array([pos[r["video_id"]] for r in g])
        lab = np.array([ci[r[COL[a.target]]] for r in g])
        if a.limit_per_group:
            keep = np.concatenate([np.where(lab == c)[0][: a.limit_per_group]
                                   for c in range(len(classes))])
            idx, lab, g = idx[keep], lab[keep], [g[i] for i in keep]
        t0 = time.time()
        # --- 1st pass: 프로토타입 (미래 물체별 LN(h) 평균) + 전역 평균
        S = np.zeros((len(classes), P.shape[1], P.shape[2]), dtype=np.float64)
        cnt = np.zeros(len(classes))
        for j, i in enumerate(idx):
            S[lab[j]] += np.asarray(H[i, off:], dtype=np.float32)
            cnt[lab[j]] += 1
        mu = (S.sum(0) / cnt.sum()).astype(np.float32)
        print(f"  {key}  n={len(idx)}  클래스별 {cnt.astype(int).tolist()}  "
              f"프로토타입 {time.time()-t0:.0f}s")
        # --- 2nd pass: argmin (자기 자신은 자기 프로토타입에서 뺀다)
        M = np.zeros((len(classes), len(classes)), dtype=int)
        mu_wins = 0
        dsum = np.zeros(len(classes)); dmu = 0.0
        for j, i in enumerate(idx):
            q = np.asarray(QS(i), dtype=np.float32)
            h = np.asarray(H[i, off:], dtype=np.float32)
            c = lab[j]
            d = np.empty(len(classes), dtype=np.float64)
            for s in range(len(classes)):
                proto = (S[s] - h) / (cnt[s] - 1) if s == c else S[s] / cnt[s]
                d[s] = np.abs(q - proto.astype(np.float32)).mean()
            M[c, int(d.argmin())] += 1
            dm = float(np.abs(q - mu).mean())
            dsum += d; dmu += dm
            if dm < d.min():
                mu_wins += 1
        acc = 100 * np.trace(M) / M.sum()
        out[" / ".join(key) + f" [{a.query}]"] = dict(query=a.query,
            classes=classes, matrix=M.tolist(), acc=acc,
            mu_wins=100 * mu_wins / len(idx),
            mean_d=(dsum / len(idx)).tolist(), mean_d_mu=dmu / len(idx))
        print(f"    retrieval acc {acc:5.1f}%  (chance {100/len(classes):.1f})   "
              f"mu 가 이기는 비율 {100*mu_wins/len(idx):5.1f}%   "
              f"평균거리 물체 {(dsum/len(idx)).min():.4f}~{(dsum/len(idx)).max():.4f} / "
              f"mu {dmu/len(idx):.4f}   ({time.time()-t0:.0f}s)")
        print(f"    {'true\\pred':12s}" + "".join(f"{c[:8]:>9s}" for c in classes) + "   지킴")
        for t, c in enumerate(classes):
            keep_r = 100 * M[t, t] / max(M[t].sum(), 1)
            print(f"    {c:12s}" + "".join(f"{M[t, q]:9d}" for q in range(len(classes)))
                  + f"{keep_r:7.0f}%")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
