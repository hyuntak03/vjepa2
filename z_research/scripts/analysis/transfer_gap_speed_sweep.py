#!/usr/bin/env python3
"""속도 sweep — 장면은 그대로, 물체 속도만 바꿔 PFT 의 뒤쪽 미래 자리가 속도를 따라가나 (2026-11-01, 사용자 요청).

왜: VOBench PFT 가 "끝 자리 shortcut" (뒤쪽 미래의 가로 자리를 문맥 속도가 아니라 학습셋 끝 자리에 둠) 을 했나를 속도 세 점으로 본다.
    R3 E35 · R4 는 두 점 (간격 3 · 2) 만 봤다.
판: VOBench split v2 test late 이동 permanence block (E35r4 와 같은 block · 같은 256² 사본). 문맥 끝은 원본 45 로 고정, 간격만 바꾼다.
  간격 3 (원판, 학습 속도) = 원본 0, 3, …, 93 · 간격 2 = 15, 17, …, 77 (속도 2/3) · 간격 1 = 30, 31, …, 61 (속도 1/3, 이번에 뽑음)
측정 (학습된 판독기 없음, E35 와 같은 정의):
  존재 대비 지도 E = ‖x(물체 문맥) − x(빈 장면 문맥)‖₂ (칸별), 자리 = 3×3 상자 합 최대 칸. 참 자리 = 실제 미래 (목표 encoder) 의 자리.
  마지막 미래 튜블릿 (t15) 의 열로 본다.
  속도 따라감 비 = (예측 열[간격 1] − 예측 열[간격 3]) / (참 열[간격 1] − 참 열[간격 3]) — 1 = 속도를 따라감, 0 = 끝 자리 고정.
  두 참 열 차가 1.5 칸 이상인 block 만. 중앙값 + block bootstrap 95 % CI.
모델: 릴리즈 · VOBench PFT · T0 (원 레시피 seed 2) · T1 (끝 자리 흩뜨리기) · T2 (생성기 섞기) — E35r4 와 같은 체크포인트.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_speed_sweep.py extract     (GPU 1, 간격 1 판만 새로 뽑는다)
  $P z_research/scripts/analysis/transfer_gap_speed_sweep.py report      (→ z_analysis/transfer_gap/results/r4_x2/speed_sweep.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_endprior as r4  # noqa: E402

ep = r4.ep
r4.SETS["S1"] = ("vobench_v1_split2_test", r4.VOB + ["data.frames_start=30", "data.frames_stride=1"])
SPEEDS = [("S1", 1, "1/3"), ("T1", 2, "2/3"), ("orig", 3, "1 (train)")]


def report():
    out = {"def": __doc__.split("측정")[1].split("모델")[0].strip(), "sets": {}}
    D = {name: ep.load(name) for name, _, _ in SPEEDS}
    va = [b["vid_a"] for b in D["orig"][1]["blocks"]]
    for name, _, _ in SPEEDS:
        assert [b["vid_a"] for b in D[name][1]["blocks"]] == va, name
    tags = D["orig"][1]["tags"]
    col = {}
    for name, s, lab in SPEEDS:
        Z, M = D[name]
        nc, nf = M["nc"], M["nf"]
        _, ch, _, _ = ep.loc(Z["Eh"])
        col[(name, "true")] = ch[:, nc + nf - 1].astype(float)
        for t in tags:
            _, cp, _, _ = ep.loc(Z[f"E_{t}"])
            col[(name, t)] = cp[:, nf - 1].astype(float)
    t1, t3 = col[("S1", "true")], col[("orig", "true")]
    keep = np.abs(t1 - t3) >= 1.5
    n = int(keep.sum())
    rng = np.random.default_rng(0)
    B = rng.integers(0, n, (2000, n))
    res = {"n_block": len(t1), "n_block_used": n, "true_col_median": {lab: float(np.median(col[(nm, "true")][keep])) for nm, _, lab in SPEEDS},
           "pred_col_median": {}, "within1col": {}, "speed_follow_ratio": {}}
    for t in tags:
        res["pred_col_median"][t] = {lab: float(np.median(col[(nm, t)][keep])) for nm, _, lab in SPEEDS}
        res["within1col"][t] = {lab: round(float((np.abs(col[(nm, t)] - col[(nm, "true")]) <= 1)[keep].mean()), 3) for nm, _, lab in SPEEDS}
        r = (col[("S1", t)] - col[("orig", t)])[keep] / (t1 - t3)[keep]
        bs = np.median(r[B], 1)
        res["speed_follow_ratio"][t] = {"median": round(float(np.median(r)), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    out["result"] = res
    json.dump(out, open(r4.RES / "speed_sweep.json", "w"), indent=1, ensure_ascii=False)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "report"
    r4.setup("all")
    if part == "extract":
        ep.extract(["S1"], 0)
    report()


if __name__ == "__main__":
    main()
