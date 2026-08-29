#!/usr/bin/env python3
"""p 안의 물체 정보는 '문맥이 묻어나온 것' 인가, predictor 가 미래로 옮긴 것인가.

p = predictor(context) 라 물체 정보의 출처는 문맥뿐이다(정의상). 물어야 할 것은
**predictor 가 그걸 미래 쪽으로 실제로 옮겼는가** 다. z/p/h 가 전부 (2048,1280) 로
같은 모양이라 p 자리에 z 를 넣어 같은 채점을 해 볼 수 있다.

  alpha(z) ~ alpha(p)  ->  predictor 가 채점에 관해서는 아무것도 더하지 않았다
  alpha(z) <  alpha(p) ->  옮기긴 했다

거리도 같이 본다:  |p-z| / |h-z| = predictor 가 문맥에서 미래로 간 비율.

⚠️ z 는 online encoder, h 는 EMA(target) 라 인코더가 다르다. 그리고 z 의 8 tubelet 은
   문맥 시점, p 는 미래 시점이라 토큰끼리 시각이 다르다. 그래서 alpha(z) 는
   "문맥 표현을 그대로 미래 자리에 놓았을 때" 라는 가상 실험이지 공정한 대조군이 아니다.

  python z_research/scripts/is_p_just_context.py --game shape
"""
from __future__ import annotations
import argparse, collections, csv, json
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
PAIRS = {"shape": ("pos_A", "imp_A_to_B"), "color": ("pos_A", "imp_A_to_B"),
         "vanish": ("pos_obj", "imp_vanish")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/local_datasets/world/world_analysis/cache/v10_vith")
    ap.add_argument("--index", default="data_csv/intphysgen_v10/index_probe.csv")
    ap.add_argument("--game", default="shape", choices=list(PAIRS))
    ap.add_argument("--n", type=int, default=64)
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
    pos_role, imp_role = PAIRS[a.game]
    bl = [d for d in by.values() if next(iter(d.values()))["game_name"] == a.game
          and {pos_role, imp_role} <= set(d)]
    pool = [np.asarray(H[idx[d[r]["video_id"]], off:], np.float32).mean(0)
            for c in CONDS
            for d in [x for x in bl if x[pos_role]["condition"] == c][:a.n_mu]
            for r in (pos_role, imp_role)]
    mu = np.mean(pool, 0)

    print(f"game={a.game}   z=context enc / p=predictor / h=target enc(미래절반)\n")
    print(f"  {'조건':<19}{'a(z)':>8}{'a(p)':>8}{'a(mu)':>8}"
          f"{'acc(z)':>8}{'acc(p)':>8}   {'|p-z|':>7}{'|h-z|':>7}{'p이동%':>8}{'|z-mu|':>8}")
    for c in CONDS:
        sel = [d for d in bl if d[pos_role]["condition"] == c][:a.n]
        az, apз, am, dpz, dhz, dzm = [], [], [], [], [], []
        for d in sel:
            v = d[pos_role]["video_id"]
            z = np.asarray(Z[idx[v]], np.float32)
            p = np.asarray(P[idx[v]], np.float32)
            hp = np.asarray(H[idx[v], off:], np.float32)
            hi = np.asarray(H[idx[d[imp_role]["video_id"]], off:], np.float32)
            D = hp - hi; dd = (D * D).sum()
            az.append(((z - hi) * D).sum() / dd)
            apз.append(((p - hi) * D).sum() / dd)
            am.append(((mu[None, :] - hi) * D).sum() / dd)
            dpz.append(np.linalg.norm(p - z, axis=-1).mean())
            dhz.append(np.linalg.norm(hp - z, axis=-1).mean())
            dzm.append(np.linalg.norm(z - mu[None, :], axis=-1).mean())
        az, apз = np.array(az), np.array(apз)
        print(f"  {c:<19}{az.mean():>8.3f}{apз.mean():>8.3f}{np.mean(am):>8.3f}"
              f"{100*(az>.5).mean():>8.1f}{100*(apз>.5).mean():>8.1f}   "
              f"{np.mean(dpz):>7.2f}{np.mean(dhz):>7.2f}"
              f"{100*np.mean(dpz)/np.mean(dhz):>7.1f}%{np.mean(dzm):>8.2f}")


if __name__ == "__main__":
    main()
