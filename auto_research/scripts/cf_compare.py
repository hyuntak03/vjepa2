#!/usr/bin/env python3
"""[T] 반사실 이어 학습: 팔 사이 문맥 항 차이 (장면 짝 bootstrap) (CPU) (auto_research, 2026-10-22 KST).
  /venv/vjepa/bin/python auto_research/scripts/cf_compare.py --base cf_a --arms cf_b cf_c
문맥 항 i = ½ (Δ1 + Δ2), Δk = 상대 차 (s(k 불가능) − s(k 가능)) / 평균 — 장면마다. 두 모델의 i 차이를 장면 bootstrap (2000) 으로 CI.
세트: 반사실 test (굴림 · TF), 외형 바꾼 test (굴림), IntPhys 2 전체 · 고체성."""
import argparse, collections, json, os, random
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--base", default="cf_a"); ap.add_argument("--arms", nargs="+", default=["cf_b", "cf_c"])
ap.add_argument("--res", default=os.path.join(os.path.dirname(__file__), "../results/cf_eval.json"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/cf_compare.json"))
a = ap.parse_args()
R = json.load(open(a.res))


def ivec(rows, key, cond=None):
    sc = collections.defaultdict(dict)
    for r in rows:
        if cond and r.get("condition") != cond:
            continue
        sc[r["scene"]][("1" if r["pair"] == "1" else "2") + ("I" if r["imp"] else "P")] = r[key]
    f = lambda x, y: (x - y) / ((x + y) / 2)
    return {s: 0.5 * (f(v["1I"], v["1P"]) + f(v["2I"], v["2P"])) for s, v in sc.items() if len(v) == 4}


SETS = [("반사실 test 굴림", "cf_test", "roll", None), ("반사실 test TF", "cf_test", "tfs", None), ("외형 바꾼 test 굴림", "cf_test_style", "roll", None),
        ("IntPhys 2 전체", "ip2", "s", None), ("IntPhys 2 고체성", "ip2", "s", "solidity")]
out = {}
for arm in a.arms:
    for nm, part, key, cond in SETS:
        if part not in R.get(arm, {}) or part not in R.get(a.base, {}):
            continue
        A = ivec(R[arm][part]["rows"], key, cond); B = ivec(R[a.base][part]["rows"], key, cond)
        ss = sorted(set(A) & set(B)); d = np.array([A[s] - B[s] for s in ss])
        rng = random.Random(0); bs = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in range(len(d))])) for _ in range(2000))
        out[f"{arm}|{nm}"] = dict(n=len(d), diff=float(d.mean()), ci=(bs[50], bs[1949]))
        print(f"{arm} − {a.base} · {nm}: 문맥 항 차 {d.mean():+.4f} [{bs[50]:+.4f}, {bs[1949]:+.4f}] (장면 {len(d)})")
json.dump(out, open(a.out, "w"), indent=1)
