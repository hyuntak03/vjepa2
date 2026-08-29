#!/usr/bin/env python3
"""probe confusion 과 surprise 방향 비대칭이 같은 것을 재는가 (v10).

v8 에서 이미 한 번 물었고 답은 "무관" 이었다 (§5-4c: 상관 0.00, recall 과는 -0.549).
v10 은 클래스가 균형(클래스당 384)이라 그 교란 없이 다시 잰다.

전이 A->B 하나마다 네 값을 붙인다.
  surprise   그 방향의 채점 정확도            (per_block.json + index)
  conf       probe 가 A 를 B 로 헷갈리는 비율  (predictions.json, p self)
  recall     probe 가 A 를 A 로 읽는 비율      (같음)
  alpha_mu   p 자리에 전역 평균을 넣은 기준선  (토큰 캐시)

⚠️ probe 와 채점은 서로 다른 질문이다 — probe 는 가능 영상만 보고 "p 에 A 가 있나",
   채점은 "p 가 h(A) 와 h(B) 중 어디에 가까운가". 어긋나는 게 모순은 아니다.

  python z_research/scripts/confusion_vs_surprise.py
"""
from __future__ import annotations

import argparse, collections, csv, json
import numpy as np

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surprise", default="z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith")
    ap.add_argument("--predictions", default="z_research/IntPhysGenV10/exp_results/attn_probe__v10_vith/predictions.json")
    ap.add_argument("--index", default="data_csv/intphysgen_v10/index_probe.csv")
    ap.add_argument("--cache", default="/local_datasets/world/world_analysis/cache/v10_vith")
    ap.add_argument("--n-mu", type=int, default=12)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.index)))
    # ---- 채점 방향별 정확도 -------------------------------------------------
    S = json.load(open(f"{a.surprise}/per_block.json"))["per_video_surprise"]
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["block_id"], r["pair_id"])].append(r)
    sur = collections.defaultdict(list)          # (cond, A, B) -> [0/1]
    for v in grp.values():
        pos = [x for x in v if x["plausible"] == "1"]
        imp = [x for x in v if x["plausible"] != "1"]
        if len(pos) != 1 or len(imp) != 1 or imp[0]["violation_type"] != "shape":
            continue
        p, i = pos[0], imp[0]
        ok = 1.0 if S[i["video_id"]] > S[p["video_id"]] else 0.0
        sur[(p["condition"], p["shape_pre"], i["shape_post"])].append(ok)

    # ---- probe confusion (p self, shape) -----------------------------------
    M = json.load(open(a.predictions))
    cls = M["targets"]["shape"]["classes"]
    gold = np.array(M["targets"]["shape"]["gold"])
    gcond = np.array(M["val_groups"])
    head = next(h for h in M["heads"] if h["fit"] == "pred__f17to32"
                and h["target"] == "shape" and h["eval"] == "pred__f17to32"
                and not h.get("groups"))
    pred = np.array(head["pred"])

    def conf_recall(cond, A, B):
        m = (gcond == cond) & (gold == cls.index(A))
        if not m.sum():
            return np.nan, np.nan
        return float((pred[m] == cls.index(B)).mean()), float((pred[m] == cls.index(A)).mean())

    # ---- alpha(mu) ----------------------------------------------------------
    vids = json.load(open(f"{a.cache}/meta.json"))["video_ids"]
    idx = {v: i for i, v in enumerate(vids)}
    P = np.load(f"{a.cache}/predictor.npy", mmap_mode="r")
    H = np.load(f"{a.cache}/target.npy", mmap_mode="r")
    off = H.shape[1] - P.shape[1]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["block_id"]][r["role"]] = r
    bl = [d for d in by.values()
          if next(iter(d.values()))["game_name"] == "shape" and {"pos_A", "imp_A_to_B"} <= set(d)]
    pool = [np.asarray(H[idx[d[r]["video_id"]], off:], np.float32).mean(0)
            for c in CONDS
            for d in [x for x in bl if x["pos_A"]["condition"] == c][:a.n_mu]
            for r in ("pos_A", "imp_A_to_B")]
    mu = np.mean(pool, 0)
    amu = collections.defaultdict(list)
    for d in bl:
        p_, i_ = d["pos_A"], d["imp_A_to_B"]
        hp = np.asarray(H[idx[p_["video_id"]], off:], np.float32)
        hi = np.asarray(H[idx[i_["video_id"]], off:], np.float32)
        D = hp - hi
        amu[(p_["condition"], p_["shape_pre"], i_["shape_post"])].append(
            float(((mu[None, :] - hi) * D).sum() / (D * D).sum()))

    # ---- 표 -----------------------------------------------------------------
    recs = []
    print(f"{'condition':<18}{'전이':<24}{'surprise':>9}{'conf A->B':>11}{'recall A':>10}{'a(mu)':>8}{'n':>5}")
    for c in CONDS:
        for (cc, A, B), v in sorted(sur.items()):
            if cc != c:
                continue
            cf, rc = conf_recall(c, A, B)
            am = float(np.mean(amu[(c, A, B)])) if amu.get((c, A, B)) else np.nan
            recs.append(dict(cond=c, A=A, B=B, sur=100 * np.mean(v),
                             conf=100 * cf, rec=100 * rc, amu=am, n=len(v)))
            print(f"{c:<18}{A+' -> '+B:<24}{100*np.mean(v):>9.1f}{100*cf:>11.1f}"
                  f"{100*rc:>10.1f}{am:>8.4f}{len(v):>5}")
        print()

    def corr(k, sel=None):
        R = [r for r in recs if sel is None or r["cond"] == sel]
        x = np.array([r[k] for r in R]); y = np.array([r["sur"] for r in R])
        m = ~np.isnan(x)
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 and x[m].std() > 0 else np.nan

    print(f"{'surprise 와의 상관':<22}{'conf':>9}{'recall':>9}{'a(mu)':>9}   n")
    for sel in [None] + CONDS:
        n = len([r for r in recs if sel is None or r["cond"] == sel])
        print(f"  {(sel or '전체'):<20}{corr('conf', sel):>9.3f}{corr('rec', sel):>9.3f}"
              f"{corr('amu', sel):>9.3f}   {n}")


if __name__ == "__main__":
    main()
