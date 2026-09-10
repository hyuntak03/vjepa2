#!/usr/bin/env python3
"""순서(cylinder > ... > torus)가 'p 가 평균 미래 근처에 있어서' 생기는 것인가.

채점은  |p - h(A)|  vs  |p - h(B)|  하나다. 만약 p 가 모든 미래의 평균 mu 근처라면
비교는 사실상 |mu - h(A)| vs |mu - h(B)| 가 되고, 그러면 **문맥이 무엇이든 결과가 같다** —
즉 물체 하나짜리 고정 순서가 나온다. 관측된 것이 정확히 그것이다.

세 가지를 잰다.
  1) p 로 채점  (report.json 재현 확인)
  2) p 자리에 mu 를 넣고 채점   -> 같은 답이 나오면 p 는 채점에 관해 mu 와 구별되지 않는다
  3) 물체별 d(X) = mean|h(X) - mu|  -> 이 값의 순서가 지킴 순서와 같은가

⚠️ mu 는 **가능 변이의 미래**만으로, 그리고 **채점하지 않는 블록**으로만 만든다.
   채점 대상 clip 이 mu 에 들어가면 mu 가 그 clip 쪽으로 당겨져 정확도가 부풀려진다
   (실측: 누수 판 70.0% -> 분리 판을 볼 것).
⚠️ 표본 추출이다. 블록 수는 --blocks 로 정한다 (절반은 mu 용, 절반은 채점용).

  python z_research/scripts/analysis/is_p_the_mean.py --condition moving_occlusion --blocks 40
"""
from __future__ import annotations
import argparse, collections, csv, json
import numpy as np

CACHE = "/local_datasets/world/world_analysis/cache/v11_vith"
INDEX = "data_csv/intphysgen_v11/index_probe.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--condition", default="moving_occlusion")
    ap.add_argument("--violation", default="shape")
    ap.add_argument("--blocks", type=int, default=40, help="블록 수 (블록당 2 matched pair)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    vids = json.load(open(f"{a.cache}/meta.json"))["video_ids"]
    ix = {v: i for i, v in enumerate(vids)}
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")      # (N, 4096, 1280)
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")   # (N, 2048, 1280)
    C = H.shape[1] // 2                                      # future = 뒷 절반

    rows = [r for r in csv.DictReader(open(a.index))
            if r["condition"] == a.condition and r["violation_type"] == a.violation]
    by_blk = collections.defaultdict(dict)
    for r in rows:
        by_blk[r["source_block"]][r["variant"]] = r
    blks = sorted(b for b, d in by_blk.items()
                  if {"pos_a", "pos_b", "imp_ab", "imp_ba"} <= set(d))
    rng = np.random.default_rng(a.seed)
    blks = [blks[i] for i in rng.permutation(len(blks))[:a.blocks]]
    half = len(blks) // 2
    mu_blks, sc_blks = blks[:half], blks[half:]          # mu 용 / 채점용 분리
    print(f"  {a.condition} / {a.violation} — mu {len(mu_blks)} 블록 / 채점 {len(sc_blks)} 블록"
          f"  (전체 {len(by_blk)})")

    def fut(v):                       # 미래 절반, float32
        return np.asarray(H[ix[v]][C:], dtype=np.float32)

    # ---- mu : 가능 변이의 미래 평균만 쓴다 (라벨 누수 방지)
    pos_ids = [by_blk[b][w]["video_id"] for b in mu_blks for w in ("pos_a", "pos_b")]
    mu = np.zeros((C, H.shape[2]), dtype=np.float64)
    for v in pos_ids:
        mu += fut(v)
    mu = (mu / len(pos_ids)).astype(np.float32)

    # ---- 1·2) matched pair 채점:  (pos_a, imp_ab) 와 (pos_b, imp_ba)
    keep_p = collections.Counter(); keep_m = collections.Counter(); tot = collections.Counter()
    agree = 0; n_pair = 0
    for b in sc_blks:
        d = by_blk[b]
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            vp, vi = d[pos]["video_id"], d[imp]["video_id"]
            hp, hi = fut(vp), fut(vi)
            p = np.asarray(P[ix[vp]], dtype=np.float32)
            okp = np.abs(p - hp).mean() < np.abs(p - hi).mean()
            okm = np.abs(mu - hp).mean() < np.abs(mu - hi).mean()
            s = d[pos]["shape_pre"]
            keep_p[s] += okp; keep_m[s] += okm; tot[s] += 1
            agree += (okp == okm); n_pair += 1

    print(f"\n  1) p 로 채점   : {100*sum(keep_p.values())/n_pair:5.1f}%   (n={n_pair})")
    print(f"  2) mu 로 채점  : {100*sum(keep_m.values())/n_pair:5.1f}%")
    print(f"     두 채점이 같은 답을 낸 비율: {100*agree/n_pair:5.1f}%")

    # ---- 3) 물체별 평균까지의 거리
    shp = collections.defaultdict(list)
    for b in sc_blks:
        for w in ("pos_a", "pos_b"):
            shp[by_blk[b][w]["shape_pre"]].append(by_blk[b][w]["video_id"])
    dist = {s: float(np.mean([np.abs(fut(v) - mu).mean() for v in vs]))
            for s, vs in shp.items()}
    print(f"\n  3) {'물체':10s}{'|h-mu|':>9s}{'p 지킴%':>9s}{'mu 지킴%':>9s}{'n':>5s}")
    order = sorted(dist, key=lambda s: dist[s])
    for s in order:
        print(f"     {s:10s}{dist[s]:9.4f}"
              f"{100*keep_p[s]/tot[s]:9.1f}{100*keep_m[s]/tot[s]:9.1f}{tot[s]:5d}")
    kp = np.array([keep_p[s] / tot[s] for s in order])
    dd = np.array([dist[s] for s in order])
    print(f"\n     상관 r(|h-mu|, p 지킴) = {np.corrcoef(dd, kp)[0,1]:+.3f}"
          f"   (음수이고 크면 '평균에 가까운 물체가 이긴다')")


if __name__ == "__main__":
    main()
