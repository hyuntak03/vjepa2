#!/usr/bin/env python3
"""binding 학습 평가 — 표준 surprise 채점을 조건 × 위반 × 방향 (A · B) 으로 (2026-10-31, z_analysis/release/03_binding_train).
입력: 각 모델의 per_block.json (per_video_surprise) — 릴리즈 = 03_binding_train/eval/surprise_c16t32__vobench_v2_test_lv_vith,
      팔 = z_training/runs/bind_v2_<팔>/eval/surprise_c16t32__vobench_v2_test_lv_latest.  판 = VOBench_v2 test late + visible (궤적 묶음 반대쪽).
A = (pos_a, imp_ab) 짝 = 문맥 A (permanence 에서 물체 있음) · B = (pos_b, imp_ba). 정답 = surprise(불가능) > surprise(가능). CI = block 재표집 95 %.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/release_bind_score_table.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODELS = {"release": REPO / "z_analysis/release/03_binding_train/eval/surprise_c16t32__vobench_v2_test_lv_vith",
          **{a: REPO / f"z_training/runs/bind_v2_{a}/eval/surprise_c16t32__vobench_v2_test_lv_latest" for a in ("a0", "a1", "a2")}}
OUT = REPO / "z_analysis/release/03_binding_train/results/score_table.json"


def boot(v, g, n=2000):
    v = np.asarray(v, float); g = np.asarray(g); ub = np.unique(g); idx = {b: np.where(g == b)[0] for b in ub}
    s = np.array([v[idx[b]].sum() for b in ub]); c = np.array([len(idx[b]) for b in ub]); r = np.random.default_rng(0)
    bs = [s[k].sum() / c[k].sum() for k in r.integers(0, len(ub), (n, len(ub)))]
    return [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]


def main():
    rows = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v2_split/index_test_lv.csv"))}
    blocks = {}
    for r in rows.values():
        blocks.setdefault(r["block_id"], {})[r["variant"]] = r["video_id"]
    res = {}
    for name, d in MODELS.items():
        S = json.load(open(d / "per_block.json"))["per_video_surprise"]
        recs = []
        for b, v in blocks.items():
            m = rows[v["pos_a"]]
            for dirn, pos, imp in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
                recs.append(dict(block=b, occ=m["occlusion"], k=m["k"], viol=m["violation"], objset=m["objset"], motion=m["motion"], dirn=dirn,
                                 ok=float(S[v[imp]] > S[v[pos]])))
        out = {}
        for occ in ("late", "visible"):
            for viol in ("permanence", "shape", "color", "all"):
                for dirn in ("A", "B", "AB"):
                    sel = [x for x in recs if x["occ"] == occ and (viol == "all" or x["viol"] == viol) and (dirn == "AB" or x["dirn"] == dirn)]
                    out[f"{occ}|{viol}|{dirn}"] = dict(n=len(sel), acc=100 * float(np.mean([x["ok"] for x in sel])),
                                                       ci=boot([x["ok"] for x in sel], [x["block"] for x in sel]))
            if occ == "late":
                for k in ("1", "2", "3"):
                    sel = [x for x in recs if x["occ"] == "late" and x["k"] == k and x["viol"] == "permanence" and x["dirn"] == "A"]
                    out[f"late_k{k}|permanence|A"] = dict(n=len(sel), acc=100 * float(np.mean([x["ok"] for x in sel])))
        res[name] = dict(table=out, recs=recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v["table"] for k, v in res.items()}, open(OUT, "w"), indent=1, ensure_ascii=False)
    keys = ["late|permanence|A", "late|permanence|B", "late|shape|AB", "late|color|AB", "late|all|AB", "visible|permanence|A", "visible|shape|AB", "visible|color|AB", "visible|all|AB"]
    print("model   " + " ".join(f"{k:>22s}" for k in keys))
    for name, v in res.items():
        t = v["table"]; print(f"{name:7s} " + " ".join(f"{t[k]['acc']:6.1f} [{t[k]['ci'][0]:3.0f},{t[k]['ci'][1]:3.0f}]" for k in keys))
    # 짝 차이 (같은 block): 팔 − a0
    for a in ("a1", "a2"):
        for key in ("late|permanence|A", "late|shape|AB", "late|color|AB", "late|all|AB", "visible|all|AB"):
            occ, viol, dirn = key.split("|")
            f = lambda x: x["occ"] == occ and (viol == "all" or x["viol"] == viol) and (dirn == "AB" or x["dirn"] == dirn)
            ra = [x for x in res[a]["recs"] if f(x)]; r0 = [x for x in res["a0"]["recs"] if f(x)]
            dlt = [x["ok"] - y["ok"] for x, y in zip(ra, r0)]
            ci = boot(np.array(dlt) + 0.0, [x["block"] for x in ra])
            print(f"  {a} − a0 {key:20s} {100 * np.mean(dlt):+5.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")


if __name__ == "__main__":
    main()
