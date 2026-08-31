#!/usr/bin/env python
"""predictor 가 무엇을 만들었나 — 토큰평균 표현으로 7(8)-way retrieval.

⚠️ 먼저 실패한 것부터 적는다. `retrieval_confusion.py` 는 **토큰 단위 L1** 로 물체
   프로토타입까지의 거리를 쟀는데, **대조군이 통과하지 못했다**:
       h(실제 미래) 조차 자기 프로토타입 retrieval 이 30.4% (chance 14.3)
       거리 자체: 물체별 0.5458~0.5503,  전역평균까지 0.5388  -> 폭이 1% 미만
   거리의 99% 가 배경·위치·가림막이고 물체 정체성은 1% 다. block 이 다르면 그 99% 가
   상쇄되지 않는다. (채점이 작동하는 이유가 이것이다 — matched pair 는 **같은 block** 이라
   문맥이 픽셀 단위로 같고 nuisance 가 정확히 상쇄된다.)

그래서 여기서는 토큰축을 접고(2048 -> 1) 차원별로 표준화한다. CLAUDE.md §5-5 에서
v1 을 56.1% -> 92.4% 로 올린 그 지표다.

    v = mean_tokens(x)            (1280,)
    z = (v - mu) / sigma          그룹 안에서 차원별 표준화
    d(S) = ||z_q - z_bar(S)||     L2

대조군: --query h 가 높게 나와야 이 측정에 해상도가 있다는 뜻이다.
        --query z 는 문맥 잔상 검사다.

  python z_research/scripts/analysis/retrieval_pooled.py --target shape --by-k
"""
from __future__ import annotations
import argparse, collections, csv, json, time
import numpy as np

CACHE = "/local_datasets/world/world_analysis/cache/v11_vith"
COL = {"shape": "shape_pre", "color": "color_pre"}


def load_pooled(cache, vids, want):
    """video_id -> 토큰평균 (1280,). p / h / z 를 한 번의 순회로 모은다."""
    meta = json.load(open(f"{cache}/meta.json"))
    pos = {v: i for i, v in enumerate(meta["video_ids"])}
    P = np.load(f"{cache}/predictor.npy", mmap_mode="r")
    H = np.load(f"{cache}/target.npy", mmap_mode="r")
    Z = np.load(f"{cache}/ctx_masked.npy", mmap_mode="r")
    off = H.shape[1] - P.shape[1]
    out = {k: np.zeros((len(vids), P.shape[2]), dtype=np.float32) for k in want}
    t0 = time.time()
    for j, v in enumerate(vids):
        i = pos[v]
        if "p" in out: out["p"][j] = np.asarray(P[i], dtype=np.float32).mean(0)
        if "h" in out: out["h"][j] = np.asarray(H[i, off:], dtype=np.float32).mean(0)
        if "z" in out: out["z"][j] = np.asarray(Z[i], dtype=np.float32).mean(0)
        if j % 1000 == 0:
            print(f"    {j}/{len(vids)}  {time.time()-t0:.0f}s", flush=True)
    print(f"    풀링 완료 {len(vids)}개 {time.time()-t0:.0f}s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--index", default="data_csv/intphysgen_v11/index_probe.csv")
    ap.add_argument("--target", default="shape", choices=list(COL))
    ap.add_argument("--by-k", action="store_true")
    ap.add_argument("--queries", nargs="*", default=["h", "p", "z"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.index))
            if r["probe_type"] == "obj" and r["violation_type"] == a.target]
    vids = [r["video_id"] for r in rows]
    print(f"{a.target}: obj clip {len(vids)}개")
    V = load_pooled(a.cache, vids, a.queries)

    classes = sorted({r[COL[a.target]] for r in rows})
    ci = {c: i for i, c in enumerate(classes)}
    lab = np.array([ci[r[COL[a.target]]] for r in rows])
    grp = np.array([(r["condition"], r["sym_k"]) if a.by_k else (r["condition"], "*")
                    for r in rows], dtype=object)
    keys = sorted({tuple(g) for g in grp})
    res = {}
    for q in a.queries:
        print(f"\n=== 질의 {q}  ({'실제 미래 — 대조군' if q=='h' else '문맥 인코더' if q=='z' else 'predictor'})")
        for key in keys:
            m = np.array([tuple(g) == key for g in grp])
            X, y = V[q][m], lab[m]
            Hm = V["h"][m] if "h" in V else None
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Zq = (X - mu) / sd
            # 프로토타입은 **미래 표현 h** 로 만든다 (질의가 무엇이든 대상은 같다)
            base = (Hm - Hm.mean(0)) / (Hm.std(0) + 1e-6) if Hm is not None else Zq
            proto = np.stack([base[y == c].mean(0) for c in range(len(classes))])
            cnt = np.array([(y == c).sum() for c in range(len(classes))])
            M = np.zeros((len(classes), len(classes)), dtype=int)
            for j in range(len(y)):
                pr = proto.copy()
                if q == "h":                      # leave-one-out
                    pr[y[j]] = (proto[y[j]] * cnt[y[j]] - base[j]) / (cnt[y[j]] - 1)
                M[y[j], int(((Zq[j] - pr) ** 2).sum(1).argmin())] += 1
            acc = 100 * np.trace(M) / M.sum()
            res[f"{'/'.join(key)} [{q}]"] = dict(acc=acc, matrix=M.tolist(), classes=classes)
            print(f"  {'/'.join(key):26s} n={M.sum():5d}  acc {acc:5.1f}%  (chance {100/len(classes):.1f})")
            if key == keys[-1] or len(keys) <= 2:
                print(f"    {'true\\pred':11s}" + "".join(f"{c[:8]:>9s}" for c in classes))
                for t, c in enumerate(classes):
                    print(f"    {c:11s}" + "".join(f"{M[t,x]:9d}" for x in range(len(classes))))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2); print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
