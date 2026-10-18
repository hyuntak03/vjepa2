"""VOBench split test block (z_training/data/build_vobench_split_index.py) 만으로 모델을 나란히 — post-FT 전후 비교 (2026-10-12).

  python z_research/scripts/analysis/vobench_split_compare.py                  # 표준 창 c16t32
  python z_research/scripts/analysis/vobench_split_compare.py --window c16t24   # 문맥 16 · 예측 8 프레임
  → z_research/VOBench_v1/ariel_ar_state_ep40/postft_vobench/[<창>/]compare.{md,json}

모델 = 전체 채점 실행에서 test block 만 고른 것 (release · ar · ar_state) + split_test 로 채점한 post-FT.
복사 = AR 공간 copy_blk (copy__vobench_v1_vith_ariel_ar_ep40, encoder 만으로 정해짐) — AR 계열과 같은 공간.
칸 = objset × violation × occlusion, 그리고 late × motion. 쌍 정확도 % (동점 0.5) · A / B 방향.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EX = REPO / "z_research/VOBench_v1/exp_results"
TEST = REPO / "data_csv/vobench_v1_split/index_test.csv"
OUT = REPO / "z_research/VOBench_v1/ariel_ar_state_ep40/postft_vobench"
RUNS = {
    "release": EX / "surprise_c16t32__vobench_v1_vith",
    "ar_ep40": EX / "surprise_c16t32__vobench_v1_vith_ariel_ar_ep40",
    "ar_state": EX / "surprise_c16t32__vobench_v1_vith_ariel_ar_state_ep40",
    "ar_state+FT": REPO / "z_training/runs/vobench_ar_state_postft/eval/surprise_c16t32__vobench_v1_split_test_latest",
}
VIOL, OCC = ("permanence", "shape", "color"), ("visible", "early", "mid", "late")
MOT = ("flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up")


def set_window(w):
    """창별 실행 폴더로 바꾼다 (c16t32 = 기본). 복사는 AR 공간 copy_last 를 채점기로 잰 실행 (ar_baseline)."""
    global RUNS, OUT, COPY_RUN
    if w == "c16t32":
        return
    RUNS = {"release": EX / f"surprise_c16t32__vobench_v1_vith_w{w}",
            "ar_ep40": EX / f"surprise_c16t32__vobench_v1_vith_ariel_ar_ep40_w{w}",
            "ar_state": EX / f"surprise_c16t32__vobench_v1_vith_ariel_ar_state_ep40_w{w}",
            "ar_state+FT": REPO / f"z_training/runs/vobench_ar_state_postft/eval/surprise_{w}__vobench_v1_split_test_latest"}
    COPY_RUN = EX / f"surprise_c16t32__vobench_v1_vith_ariel_ar_ep40_arcopy_w{w}"
    OUT = OUT / w


COPY_RUN = None


def surprise(run):
    S = json.load(open(run / "per_block.json"))["per_video_surprise"]
    return {k: (v["all"] if isinstance(v, dict) else v) for k, v in S.items()}


def copy_blk():
    if COPY_RUN is not None:
        return surprise(COPY_RUN)
    d = EX / "copy__vobench_v1_vith_ariel_ar_ep40"
    meta = json.load(open(d / "meta.json")); cb = np.load(d / "copy_blk.npy")
    return {v: float(cb[i].mean()) for i, v in enumerate(meta["video_id"])}


def set_split(sp):
    """v2 (2026-10-14, 누수 없는 장면 묶음 split) — test index 와 post-FT 실행을 v2 로. 창 c16t32 만."""
    global TEST, OUT, RUNS
    if sp == "v1":
        return
    TEST = REPO / "data_csv/vobench_v1_split_v2/index_test.csv"
    OUT = REPO / "z_research/VOBench_v1/vjepa2_pft/surprise"          # v2 비교표 (release+FT · ar_state+FT 둘 다)
    RUNS = {k: v for k, v in RUNS.items() if not k.endswith("+FT")}
    RUNS["release+FT"] = REPO / "z_training/runs/vobench_release_postft/eval/surprise_c16t32__vobench_v1_split2_test_latest"
    RUNS["ar_state+FT"] = REPO / "z_training/runs/vobench_ar_state_postft_v2/eval/surprise_c16t32__vobench_v1_split2_test_latest"


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--window", default="c16t32"); ap.add_argument("--split", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    set_window(a.window); set_split(a.split)
    rows = list(csv.DictReader(TEST.open())); by = defaultdict(dict)
    for r in rows: by[r["block_id"]][r["variant"]] = r
    S = {m: surprise(p) for m, p in RUNS.items() if (p / "per_block.json").exists()}
    if COPY_RUN is None or (COPY_RUN / "per_block.json").exists():
        S["copy (AR space)"] = copy_blk()
    if "release+FT" in RUNS:                                          # v2: 릴리즈 공간 복사도 (release · release+FT 와 같은 공간)
        import vobench_tables as VT
        S["copy (release space)"] = VT.copy_scores(EX / "copy__vobench_v1_vith")["copy"]
    P = []
    for b, v in by.items():
        r = v["pos_a"]
        for d, p, q in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            hit = {m: (1.0 if s[v[q]["video_id"]] > s[v[p]["video_id"]] else 0.5 if s[v[q]["video_id"]] == s[v[p]["video_id"]] else 0.0) for m, s in S.items()}
            P.append(dict(objset=r["objset"], viol=r["violation"], occ=r["occlusion"], mot=r["motion"], dir=d, block=b, **hit))
    models = list(S)
    acc = lambda ps, m: 100 * np.mean([p[m] for p in ps]) if ps else float("nan")
    R = {"n_pairs": len(P), "models": models, "cells": {}}
    md = [f"# VOBench split test block — post-FT 전후 · 창 {a.window} ({len(P)} 쌍, {len(by)} block)", "",
          "모델 = 전체 채점에서 test block 만 고름 · post-FT = split_test 로 채점. 복사 = AR 공간 copy_blk (AR 계열과 같은 공간). 칸 = 쌍 정확도 %.", ""]
    def table(title, keys, filt):
        nonlocal md
        md += [f"## {title}", "", "| 칸 | 쌍 | " + " | ".join(models) + " |", "|---|---:|" + "---:|" * len(models)]
        for k in keys:
            ps = [p for p in P if filt(p, k)]
            R["cells"][f"{title}/{k}"] = {m: acc(ps, m) for m in models} | {"n": len(ps)}
            md.append(f"| {k} | {len(ps)} | " + " | ".join(f"{acc(ps, m):.1f}" for m in models) + " |")
        md.append("")
    for o in ("primitive", "realistic"):
        table(f"{o} · overall", ["all"], lambda p, k, o=o: p["objset"] == o)
        table(f"{o} · 위반 × 가림", [f"{v}/{oc}" for v in VIOL for oc in OCC],
              lambda p, k, o=o: p["objset"] == o and f"{p['viol']}/{p['occ']}" == k)
        table(f"{o} · late × 운동 (위반 셋 합침)", list(MOT), lambda p, k, o=o: p["objset"] == o and p["occ"] == "late" and p["mot"] == k)
        table(f"{o} · late permanence 방향", ["A", "B"], lambda p, k, o=o: p["objset"] == o and p["occ"] == "late" and p["viol"] == "permanence" and p["dir"] == k)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.md").write_text("\n".join(md) + "\n"); json.dump(R, open(OUT / "compare.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
