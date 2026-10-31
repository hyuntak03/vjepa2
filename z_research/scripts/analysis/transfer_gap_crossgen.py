#!/usr/bin/env python3
"""transfer_gap Q3 · E26 — 교차 생성기 자연 실험: 다른 데이터로 이어 학습한 predictor 가 VOBench late 사라짐을 바꾸나. 2026-10-30, [X2] R2.

왜: VOBench PFT 는 VOBench late 이동 사라짐 A 를 0 → 100 % 로 올린다. 같은 '문맥 끝 가림' 을 담은 다른 생성기 (v11) 로
    학습한 PFT 도 VOBench 에서 그것을 올리면 (H3-X) 효과는 생성기 계열을 넘는다. 아니면 'VOBench 학습에서만'.
정의 (CPU, 기존 per_block.json 만 읽는다; v11 PFT 의 VOBench 채점은 하네스 surprise_c16t32 로 이번에 새로 냄)
  짝 A = (pos_a, imp_ab) · 짝 B = (pos_b, imp_ba)  (같은 문맥, matched). 판정 = s(imp) > s(pos).
  permanence 에서 A 는 '물체가 있던 문맥 → 빈 미래' 가 불가능 쪽 (사라짐), B 는 '빈 문맥 → 물체 미래' (나타남).
  칸: late 이동 (motion ≠ flat_static) permanence A · B / 위반 셋 × 가림 시점 / objset / 배경 (v11 · assets).
  G = acc(model) − acc(릴리즈) 같은 짝. CI = block ((objset, block_id)) bootstrap 2,000.
  복사 = 문맥 encoder 마지막 튜블릿을 미래 8 칸에 복사한 surprise (te 판, copy__vobench_v1_vith).
  ⚠️ VOBench PFT · IntPhys 2 쌍 PFT · IntPhys 1 PFT 는 split v2 test 만 채점돼 있다 → 비교는 test 만 (주), v11 PFT · IntPhys 1 late 창 PFT 는 전부도 낸다.
출력: z_analysis/transfer_gap/results/q3_r2/E26_crossgen.json

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_crossgen.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E26_crossgen.json"
NB = 2000
SRC = {
    "release": "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/per_block.json",
    "v11_pft": "cvpr/results/eval/testbed/vobench_v1_vith_pft_v11_e10/per_block.json",
    "ip1late_pft": "cvpr/results/eval/testbed/vobench_v1_vith_pft_ip1late_f0/per_block.json",
    "ip2pair_pft": "cvpr/results/eval/testbed/vobench_v1_split2_test_vith_pft_ip2pair/per_block.json",
    "intphys1_pft": "cvpr/results/eval/testbed/vobench_v1_split2_test_vith_pft_intphys1_e40/per_block.json",
    "vob_pft": "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json",
}
COPY = "z_research/VOBench_v1/exp_results/copy__vobench_v1_vith"


def main():
    rows = list(csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")))
    test_ids = {r["video_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    by = defaultdict(dict)
    for r in rows:
        by[(r["objset"], r["block_id"])][r["variant"]] = r
    pairs = []
    for (os_, b), v in by.items():
        for d, (p, q) in (("A", ("pos_a", "imp_ab")), ("B", ("pos_b", "imp_ba"))):
            if p in v and q in v:
                r = v[p]
                pairs.append({"pos": r["video_id"], "imp": v[q]["video_id"], "dir": d, "block": f"{os_}/{b}", "objset": os_,
                              "viol": r["violation"], "occ": r["occlusion"], "k": r["k"], "motion": r["motion"],
                              "bg": r["background"], "test": r["video_id"] in test_ids})
    S = {}
    for m, p in SRC.items():
        S[m] = json.load(open(REPO / p))["per_video_surprise"]
    cm = json.load(open(REPO / COPY / "meta.json"))
    cp = np.load(REPO / COPY / "copy.npy").mean(1)
    S["copy"] = {v: float(x) for v, x in zip(cm["video_id"], cp)}

    def judge(m):
        s = S[m]
        x = np.full(len(pairs), np.nan)
        for i, q in enumerate(pairs):
            if q["pos"] in s and q["imp"] in s:
                x[i] = 1.0 if s[q["imp"]] > s[q["pos"]] else (0.5 if s[q["imp"]] == s[q["pos"]] else 0.0)
        return x

    J = {m: judge(m) for m in S}
    coverage = {m: {"n_pairs": int((~np.isnan(x)).sum()), "n_test_pairs": int((~np.isnan(x) & np.array([q["test"] for q in pairs])).sum())}
                for m, x in J.items()}
    rng = np.random.default_rng(0)
    blocks = np.array([q["block"] for q in pairs])

    def stat(mask, m, base=None):
        x = J[m]
        ok = mask & ~np.isnan(x)
        if base is not None:
            ok &= ~np.isnan(J[base])
        if ok.sum() == 0:
            return None
        v = x[ok] - (J[base][ok] if base is not None else 0)
        bl = blocks[ok]
        u = np.unique(bl)
        idx = {b: np.flatnonzero(bl == b) for b in u}
        bs = []
        for _ in range(NB):
            pick = rng.choice(len(u), len(u))
            ii = np.concatenate([idx[u[k]] for k in pick])
            bs.append(v[ii].mean())
        return {"n": int(ok.sum()), "v": round(100 * float(v.mean()), 1),
                "ci": [round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1)]}

    P = lambda f: np.array([bool(f(q)) for q in pairs])  # noqa: E731
    cells = {
        "late_move_perm_A": P(lambda q: q["occ"] == "late" and q["motion"] != "flat_static" and q["viol"] == "permanence" and q["dir"] == "A"),
        "late_move_perm_B": P(lambda q: q["occ"] == "late" and q["motion"] != "flat_static" and q["viol"] == "permanence" and q["dir"] == "B"),
        "late_static_perm_A": P(lambda q: q["occ"] == "late" and q["motion"] == "flat_static" and q["viol"] == "permanence" and q["dir"] == "A"),
        "all": P(lambda q: True),
    }
    for occ in ("visible", "early", "mid", "late"):
        cells[f"occ={occ}"] = P(lambda q, o=occ: q["occ"] == o)
        for v in ("permanence", "shape", "color"):
            cells[f"occ={occ}|viol={v}"] = P(lambda q, o=occ, vv=v: q["occ"] == o and q["viol"] == vv)
    for os_ in ("primitive", "realistic"):
        cells[f"late_move_perm_A|objset={os_}"] = cells["late_move_perm_A"] & P(lambda q, o=os_: q["objset"] == o)
    for bg in ("v11", "assets"):
        cells[f"late_move_perm_A|bg={bg}"] = cells["late_move_perm_A"] & P(lambda q, b=bg: q["bg"] == b)
        cells[f"late_move_perm_B|bg={bg}"] = cells["late_move_perm_B"] & P(lambda q, b=bg: q["bg"] == b)
        cells[f"late|bg={bg}"] = cells["occ=late"] & P(lambda q, b=bg: q["bg"] == b)
    test = P(lambda q: q["test"])
    res = {"definition": __doc__.split("정의")[1].split("출력:")[0].strip(), "sources": SRC, "copy": COPY, "coverage": coverage,
           "test": {}, "all_blocks": {}}
    models = ["release", "v11_pft", "ip1late_pft", "ip2pair_pft", "intphys1_pft", "vob_pft", "copy"]
    for split, base_mask, ms in (("test", test, models), ("all_blocks", np.ones(len(pairs), bool), ["release", "v11_pft", "ip1late_pft", "copy"])):
        R = {}
        for c, m in cells.items():
            R[c] = {"acc": {mm: stat(m & base_mask, mm) for mm in ms},
                    "gain_vs_release": {mm: stat(m & base_mask, mm, "release") for mm in ms if mm != "release"}}
        # 배경 사이 이득 차 (late 이동 A): G(v11 배경) − G(assets 배경), 판정 '배경에 안 묶임' = |차| ≤ 15
        R["bg_gain_gap_late_move_perm_A"] = {mm: (None if not (R["late_move_perm_A|bg=v11"]["gain_vs_release"].get(mm) and R["late_move_perm_A|bg=assets"]["gain_vs_release"].get(mm))
                                                 else round(R["late_move_perm_A|bg=v11"]["gain_vs_release"][mm]["v"] - R["late_move_perm_A|bg=assets"]["gain_vs_release"][mm]["v"], 1))
                                             for mm in ms if mm != "release"}
        res[split] = R
    G = res["test"]["late_move_perm_A"]["gain_vs_release"]["v11_pft"]
    res["verdict_H3X"] = ("지지 (G_A ≥ +30, CI 하한 > +10)" if G["v"] >= 30 and G["ci"][0] > 10 else
                          "기각 → 'VOBench 학습에서만' (G_A ≤ +10)" if G["v"] <= 10 else "경계 (+10 < G_A < +30 또는 CI 하한 ≤ +10)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    for split in ("test", "all_blocks"):
        for c in ("late_move_perm_A", "late_move_perm_B", "late_move_perm_A|bg=v11", "late_move_perm_A|bg=assets", "occ=late", "occ=visible", "all"):
            print(split, c, {m: (v["v"], v["n"]) for m, v in res[split][c]["acc"].items() if v})
    print("H3-X:", res["verdict_H3X"], G)
    print("→", OUT)


if __name__ == "__main__":
    main()
