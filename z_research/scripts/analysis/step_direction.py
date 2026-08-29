#!/usr/bin/env python3
"""predictor 의 '걸음' 이 옳은 미래 쪽인가.

|p-z|/|h-z| = 79% 는 **거리 비율**이라 방향을 말해 주지 않는다. 엉뚱한 쪽으로 79%
갔을 수도 있다. 그래서 걸음 벡터의 코사인을 직접 잰다 (문맥 z 를 원점으로).

    step   = p - z          predictor 가 실제로 간 방향
    right  = h_pos - z      옳은 미래로 가는 방향
    wrong  = h_imp - z      불가능한 미래로 가는 방향
    avg    = mu - z         전역 평균으로 가는 방향

  cos(step, right) ~ cos(step, wrong)  -> 두 후보의 중간으로 간다 (alpha ~ 0.5 와 일치)
  cos(step, avg) 가 제일 크다          -> 평균으로 간다
  proj = <step,right>/|right|^2        -> 옳은 변위의 몇 배를 갔나

⚠️ z 는 online encoder(문맥 시점), h 는 EMA(미래 시점)라 인코더도 시각도 다르다.
   그래서 "옳은 방향" 은 두 인코더 차이까지 포함한 값이다. 절대값보다 **right 와
   wrong 의 차이**, 그리고 avg 와의 비교를 읽을 것.
"""
from __future__ import annotations
import argparse, collections, csv, json
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
PAIRS = {"shape": ("pos_A", "imp_A_to_B"), "color": ("pos_A", "imp_A_to_B"),
         "vanish": ("pos_obj", "imp_vanish")}


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/local_datasets/world/world_analysis/cache/v10_vith")
    ap.add_argument("--index", default="data_csv/intphysgen_v10/index_probe.csv")
    ap.add_argument("--game", default="shape", choices=list(PAIRS))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--n-mu", type=int, default=12)
    a = ap.parse_args()

    vids = json.load(open(f"{a.cache}/meta.json"))["video_ids"]
    idx = {v: i for i, v in enumerate(vids)}
    Z = np.load(f"{a.cache}/ctx_masked.npy", mmap_mode="r")
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")
    off = H.shape[1] - P.shape[1]
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.index)):
        by[r["block_id"]][r["role"]] = r
    pr, ir = PAIRS[a.game]
    bl = [d for d in by.values() if next(iter(d.values()))["game_name"] == a.game
          and {pr, ir} <= set(d)]
    pool = [np.asarray(H[idx[d[r]["video_id"]], off:], np.float32).mean(0)
            for c in CONDS
            for d in [x for x in bl if x[pr]["condition"] == c][:a.n_mu]
            for r in (pr, ir)]
    mu = np.mean(pool, 0)

    print(f"game={a.game}   원점 = z(context enc).  step = p - z\n")
    print(f"  {'조건':<19}{'cos(step,right)':>16}{'cos(step,wrong)':>16}"
          f"{'cos(step,avg)':>14}{'cos(right,avg)':>15}{'proj/right':>11}")
    for c in CONDS:
        sel = [d for d in bl if d[pr]["condition"] == c][:a.n]
        R = collections.defaultdict(list)
        for d in sel:
            v = d[pr]["video_id"]
            z = np.asarray(Z[idx[v]], np.float32)
            p = np.asarray(P[idx[v]], np.float32)
            hp = np.asarray(H[idx[v], off:], np.float32)
            hi = np.asarray(H[idx[d[ir]["video_id"]], off:], np.float32)
            step, right, wrong, avg = p - z, hp - z, hi - z, mu[None, :] - z
            R["r"].append(cos(step, right)); R["w"].append(cos(step, wrong))
            R["a"].append(cos(step, avg));   R["ra"].append(cos(right, avg))
            R["pj"].append(float((step * right).sum() / (right * right).sum()))
        print(f"  {c:<19}{np.mean(R['r']):>16.3f}{np.mean(R['w']):>16.3f}"
              f"{np.mean(R['a']):>14.3f}{np.mean(R['ra']):>15.3f}{np.mean(R['pj']):>11.3f}")


if __name__ == "__main__":
    main()
