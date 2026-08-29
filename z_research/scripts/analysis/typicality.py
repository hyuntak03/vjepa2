#!/usr/bin/env python3
"""방향 비대칭의 기하학적 정체: "전형성(typicality)".

보정이 ~0 이면 정답 조건은 alpha(mu) > 0.5 로 환원되는데, 이를 중점 기준으로 풀면

    alpha(mu) - 0.5 = -( |mu - h_pos|^2 - |mu - h_imp|^2 ) / (2|D|^2)

즉  alpha(mu) > 0.5  <=>  h_pos 가 h_imp 보다 전역 평균 mu 에 가깝다.

전이 A->B 는 h_pos=A미래, h_imp=B미래 이므로

    **A->B 는 B 가 A 보다 비전형적(mu 에서 멂)일 때 맞는다.**

역방향 B->A 는 조건이 정확히 뒤집히므로 **둘 중 하나만 성립한다** — 방향 비대칭이
모델의 성질이 아니라 데이터 기하의 산술적 귀결이라는 뜻이다.

검증: 모양 s 마다 d_s = |h_s - mu| (가능 영상 미래표현의 평균) 를 재고,
      sign(d_B - d_A) 가 채점 정확도를 예측하는지 본다.
"""
from __future__ import annotations
import argparse, collections, csv, json
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/local_datasets/world/world_analysis/cache/v10_vith")
    ap.add_argument("--index", default="data_csv/intphysgen_v10/index_probe.csv")
    ap.add_argument("--surprise", default="z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith")
    ap.add_argument("--n-mu", type=int, default=12)
    a = ap.parse_args()

    vids = json.load(open(f"{a.cache}/meta.json"))["video_ids"]
    idx = {v: i for i, v in enumerate(vids)}
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")
    off = H.shape[1] - P.shape[1]
    rows = list(csv.DictReader(open(a.index)))
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["block_id"]][r["role"]] = r

    # 채점 방향별 정확도
    S = json.load(open(f"{a.surprise}/per_block.json"))["per_video_surprise"]
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["block_id"], r["pair_id"])].append(r)
    sur = collections.defaultdict(list)
    for v in grp.values():
        pos = [x for x in v if x["plausible"] == "1"]; imp = [x for x in v if x["plausible"] != "1"]
        if len(pos) != 1 or len(imp) != 1 or imp[0]["violation_type"] != "shape":
            continue
        p, i = pos[0], imp[0]
        sur[(p["condition"], p["shape_pre"], i["shape_post"])].append(
            1.0 if S[i["video_id"]] > S[p["video_id"]] else 0.0)

    bl = [d for d in by.values() if next(iter(d.values()))["game_name"] == "shape"
          and {"pos_A", "imp_A_to_B"} <= set(d)]
    pool = [np.asarray(H[idx[d[r]["video_id"]], off:], np.float32).mean(0)
            for c in CONDS
            for d in [x for x in bl if x["pos_A"]["condition"] == c][:a.n_mu]
            for r in ("pos_A", "imp_A_to_B")]
    mu = np.mean(pool, 0)

    # 모양별 전형성 d_s = |h_s - mu|  (조건별로 따로)
    acc_all, gap_all = [], []
    for c in CONDS:
        hs = collections.defaultdict(list)
        for d in bl:
            if d["pos_A"]["condition"] != c:
                continue
            hs[d["pos_A"]["shape_pre"]].append(
                np.asarray(H[idx[d["pos_A"]["video_id"]], off:], np.float32).mean(0))
            hs[d["imp_A_to_B"]["shape_post"]].append(
                np.asarray(H[idx[d["imp_A_to_B"]["video_id"]], off:], np.float32).mean(0))
        dist = {s: float(np.linalg.norm(np.mean(v, 0) - mu)) for s, v in hs.items()}
        print(f"\n### {c}")
        print("  모양별 |h_s - mu| (클수록 비전형):")
        for s, v in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"     {s:<12}{v:8.3f}")
        print(f"  {'전이':<24}{'채점':>8}{'d(B)-d(A)':>11}{'예측':>7}{'맞나':>6}")
        ok = 0; tot = 0
        for (cc, A, B), v in sorted(sur.items()):
            if cc != c or A not in dist or B not in dist:
                continue
            gap = dist[B] - dist[A]; acc = 100 * np.mean(v)
            pred = "정답" if gap > 0 else "오답"
            hit = (gap > 0) == (acc > 50)
            ok += hit; tot += 1
            acc_all.append(acc); gap_all.append(gap)
            print(f"  {A+' -> '+B:<24}{acc:>8.1f}{gap:>11.4f}{pred:>7}{'O' if hit else 'X':>6}")
        print(f"  부호 일치 {ok}/{tot}")
    r = np.corrcoef(gap_all, acc_all)[0, 1]
    print(f"\n전체 (n={len(gap_all)}):  corr(d(B)-d(A), 채점) = {r:.3f}"
          f"   부호 일치 {sum((np.array(gap_all)>0)==(np.array(acc_all)>50))}/{len(gap_all)}")


if __name__ == "__main__":
    main()
