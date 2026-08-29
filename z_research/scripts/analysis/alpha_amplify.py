#!/usr/bin/env python3
"""p 를 평균에서 멀어지는 쪽으로 gamma 배 늘리면 채점이 살아나는가.

alpha 는 p 에 대해 아핀이라 p' = mu + gamma*(p - mu) 를 넣으면
    alpha(p') = alpha(mu) + gamma * (alpha(p) - alpha(mu))
              = alpha(mu) + gamma * 보정
가 된다. **alpha(mu) 는 그대로 남고 보정만 gamma 배**다. 정답은 alpha > 0.5 이므로

    보정 > 0 인 쌍  -> gamma 를 키우면 결국 맞는다
    보정 < 0 인 쌍  -> 키울수록 더 틀린다

따라서 **증폭의 천장 = 보정 부호가 맞는 쌍의 비율**이다. 그걸 쌍 단위로 센다.

--anchor 로 기준점을 고른다. **어디서 미느냐가 천장을 정한다.**
   mu : p' = mu + g(p-mu)   전역 평균에서 밀어냄  -> 천장 = (alpha(p) > alpha(mu)) 비율
   z  : p' = z  + g(p-z)    문맥 표현에서 밀어냄  -> 천장 = (alpha(p) > alpha(z))  비율
        predictor 는 z 에서 h 쪽으로 79% 만 간다(실측). z 기준 외삽은 그 걸음을
        더 크게 하는 것이라 "미래 쪽으로 밀어준다" 에 해당한다.

⚠️ 라벨을 안 쓰므로 둘 다 정당한 채점 규칙 수정이다. shape 개념 방향만 골라 키우는
   개입(steering)은 별개다 — held-out 가능 영상에서 부분공간을 배우면 그것도 정당하다.

  python z_research/scripts/analysis/alpha_amplify.py --game shape
"""
from __future__ import annotations

import argparse, collections, csv, json
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
PAIRS = {"shape":  [("pos_A", "imp_A_to_B"), ("pos_B", "imp_B_to_A")],
         "color":  [("pos_A", "imp_A_to_B"), ("pos_B", "imp_B_to_A")],
         "vanish": [("pos_obj", "imp_vanish"), ("pos_empty", "imp_appear")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/local_datasets/world/world_analysis/cache/v10_vith")
    ap.add_argument("--index", default="data_csv/intphysgen_v10/index_probe.csv")
    ap.add_argument("--game", default="shape", choices=list(PAIRS))
    ap.add_argument("--n", type=int, default=128, help="조건당 block 수")
    ap.add_argument("--n-mu", type=int, default=12)
    ap.add_argument("--anchor", choices=["mu", "z"], default="mu")
    a = ap.parse_args()

    vids = json.load(open(f"{a.cache}/meta.json"))["video_ids"]
    idx = {v: i for i, v in enumerate(vids)}
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")
    Z = np.load(f"{a.cache}/ctx_masked.npy", mmap_mode="r")
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")
    off = H.shape[1] - P.shape[1]
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.index)):
        by[r["block_id"]][r["role"]] = r

    GAM = [1, 2, 3, 5, 10, 30, 100]
    anc = "mu" if a.anchor == "mu" else "z"
    print(f"game={a.game}  cache={a.cache.split('/')[-1]}  "
          f"p' = {anc} + g*(p - {anc})   기준선 = alpha({anc})\n")
    for pos_role, imp_role in PAIRS[a.game]:
        bl = [d for d in by.values()
              if next(iter(d.values()))["game_name"] == a.game and {pos_role, imp_role} <= set(d)]
        pool = [np.asarray(H[idx[d[r]["video_id"]], off:], np.float32).mean(0)
                for c in CONDS
                for d in [x for x in bl if x[pos_role]["condition"] == c][:a.n_mu]
                for r in (pos_role, imp_role)]
        mu = np.mean(pool, 0)

        print(f"[{pos_role} vs {imp_role}]")
        print(f"  {'조건':<19}{'n':>5}{'기여>0':>8}" + "".join(f"{'g='+str(g):>8}" for g in GAM))
        for c in CONDS:
            sel = [d for d in bl if d[pos_role]["condition"] == c][:a.n]
            ap_, am_ = [], []
            for d in sel:
                p = np.asarray(P[idx[d[pos_role]["video_id"]]], np.float32)
                hp = np.asarray(H[idx[d[pos_role]["video_id"]], off:], np.float32)
                hi = np.asarray(H[idx[d[imp_role]["video_id"]], off:], np.float32)
                D = hp - hi; dd = (D * D).sum()
                ap_.append(((p - hi) * D).sum() / dd)
                base = (mu[None, :] if a.anchor == "mu"
                        else np.asarray(Z[idx[d[pos_role]["video_id"]]], np.float32))
                am_.append(((base - hi) * D).sum() / dd)
            ap_, am_ = np.array(ap_), np.array(am_)
            corr = ap_ - am_
            accs = [100 * ((am_ + g * corr) > 0.5).mean() for g in GAM]
            print(f"  {c:<19}{len(sel):>5}{100*(corr>0).mean():>7.1f}%"
                  + "".join(f"{x:>8.1f}" for x in accs))
        print()


if __name__ == "__main__":
    main()
