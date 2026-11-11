#!/usr/bin/env python3
"""실제 영상 벤치 (real world permanence, 95 장면 · 190 쌍) 를 출처 · 사건 종류 · 쌍 (사라짐 / 나타남) 으로 나눈 정확도 (2026-11-07). GPU 불필요.
채점 산출물 per_block.json 만 읽는다 (시트 점수와 같은 판정: 불가능 surprise > 가능 surprise).
  /venv/vjepa/bin/python z_research/scripts/analysis/realbench_by_case.py vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4f_negimp
"""
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import RES, boards  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def main():
    models = sys.argv[1:]
    pat, blk = boards()["real world (all 95)"]
    idx = {r["block_id"]: r for r in csv.DictReader(open(f"{REPO}/data_csv/real_world_permanence_uniform/index.csv"))}
    res = {}
    for m in models:
        S = {k: (v["all"] if isinstance(v, dict) else v) for k, v in json.load(open(f"{RES}/{pat.format(m=m)}/per_block.json"))["per_video_surprise"].items()}
        g = defaultdict(lambda: [0, 0])
        for b, v in blk.items():
            r = idx[b]
            src = r["file"].split("/")[0].split("_", 1)[1]
            for pair, key in (("사라짐", "A"), ("나타남", "B")):
                ok = S[v[key][1]] > S[v[key][0]]
                for k in (("전체", "", pair), (src, "", pair), (src, r["outcome"], pair), ("전체", "", "합"), (src, r["outcome"], "합")):
                    g[k][0] += int(ok); g[k][1] += 1
        res[m] = g
    keys = sorted(set().union(*[set(g) for g in res.values()]), key=lambda k: (k[0] != "전체", k[0], k[1], k[2]))
    print("| 출처 | 사건 | 쌍 | n | " + " | ".join(models) + " |")
    print("|---|---|---|---|" + "---|" * len(models))
    for k in keys:
        n = res[models[0]][k][1]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {n} | " + " | ".join(f"{100 * res[m][k][0] / max(res[m][k][1], 1):.1f}" for m in models) + " |")


if __name__ == "__main__":
    main()
