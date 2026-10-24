#!/usr/bin/env python3
"""intphys2_late 결과 모으기 (2026-10-22) — 모델마다 쌍 정확도 (전체 · 위반 종류 · 가림 뒤 · d 출처) + 복사 기준선 (있으면).

IntPhys 2 로 학습한 PFT (vith_pft_ip2pair*) 는 학습에 안 쓴 쌍만 센다: 학습 = 장면마다 가능 영상 하나
  (data_csv/IntPhys2/pair_split/index_s0.csv, 253 개) → 그 가능 영상이 속한 쌍을 빼고 다른 쌍만 (held-out).
  python cvpr/eval/intphys2_late/collect.py [모델 …]   (없으면 결과 폴더 전부)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
import os
DS = os.environ.get("DS", "intphys2_late_uni")              # DS=intphys2_late_uni_m10 → d = 라벨 − 10 판
RES = REPO / "cvpr/results/eval/intphys2_late"
IDX = REPO / f"data_csv/{DS}/index.csv"
TRAIN = REPO / "data_csv/IntPhys2/pair_split/index_s0.csv"
HELDOUT_MODELS = ("vith_pft_ip2pair",)


def pairs():
    by = defaultdict(dict)
    for r in csv.DictReader(open(IDX)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    return by


def score(S, P, keep=None):
    sc = lambda v: S[v] if isinstance(S[v], (int, float)) else list(S[v].values())[0]
    acc = defaultdict(list)
    for k, p in P.items():
        if keep is not None and k not in keep:
            continue
        if p["pos"]["video_id"] not in S or p["imp"]["video_id"] not in S:
            continue
        a, b = sc(p["pos"]["video_id"]), sc(p["imp"]["video_id"])
        c = 1.0 if b > a else 0.5 if b == a else 0.0
        q = p["pos"]
        for g in ("all", q["condition"], f"occluded={q['occluded']}", f"d_src={q['d_src']}"):
            acc[g].append(c)
    return {g: (round(100 * sum(v) / len(v), 1), len(v)) for g, v in acc.items()}


def main():
    P = pairs()
    train = {r["video_id"] for r in csv.DictReader(open(TRAIN))}
    held = {k for k, p in P.items() if Path(p["pos"]["file"]).stem not in train}
    models = sys.argv[1:] or sorted(d.name[len(DS) + 1:] for d in RES.glob(f"{DS}_*"))
    out = {}
    for m in models:
        d = RES / f"{DS}_{m}"
        if not (d / "per_block.json").exists():
            print(f"{m:34s} ⏳"); continue
        keep = held if m.startswith(HELDOUT_MODELS) else None
        S = json.load(open(d / "per_block.json"))["per_video_surprise"]
        r = {"model": score(S, P, keep), "split": "heldout" if keep else "all"}
        if (d / "copy_baseline.json").exists():
            cb = json.load(open(d / "copy_baseline.json"))["per_video"]
            r.update({k: score(cb[k], P, keep) for k in ("copy_z", "copy_h")})
        out[m] = r
        line = " · ".join(f"{k} {v[0]}" for k, v in sorted(r["model"].items()) if k != "all")
        print(f"{m:34s} {r['split']:7s} {r['model']['all'][0]:5.1f} (n {r['model']['all'][1]})"
              + (f"  copy_z {r['copy_z']['all'][0]} copy_h {r['copy_h']['all'][0]}" if "copy_z" in r else "") + f"  | {line}")
    json.dump(out, open(RES / f"collect_{DS}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
