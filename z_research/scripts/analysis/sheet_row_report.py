#!/usr/bin/env python3
"""시트 (Overall 탭) 한 행에 들어갈 값을 cvpr/results 산출물에서 다시 센다 (2026-10-19, 사용자 지시 "끝나는 대로 sheet 에 기록").

열 정의는 평가 행렬 (eval_matrix_breakdown.py) 과 같다:
  VOBench P / R   : split v2 test 360 block, late 가림 · primitive / realistic 쌍 정확도 (matched, 동점 0.5)
  Gravity P / R   : gravity test 108 block, 고른 미래 = 진짜 (block 안 surprise 최소가 가능 영상) — primitive / realistic (ledge + arc 합침, chance 37.0)
  IntPhys 1       : Garrido A.8 — 창 w16 · w32 를 합쳐 Filtered (C 최솟값) → property macro 최고 combo (garrido_rescore.select)
  InfLevel        : property 마다 최고 C → 3 property macro 최고 combo (garrido_rescore.collect_inflevel 과 같은 셈)
  python z_research/scripts/analysis/sheet_row_report.py vith_pft_ip2pair vith_pft_ip2pair_lora [--vob <dir>] [--grav <dir>]
없는 산출물은 ⏳.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import garrido_rescore as G  # noqa: E402

TB = REPO / "cvpr/results/eval/testbed"
IP = REPO / "cvpr/results/eval/intphys1"


def vob_late(d: Path):
    f = d / "per_block.json"
    if not f.exists():
        return None
    meta = {r["block_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    acc = defaultdict(lambda: [0.0, 0])
    for b in json.load(open(f))["per_block"]:
        r = meta.get(b["block_id"])
        if r is not None and r["occlusion"] == "late":
            acc[r["objset"]][0] += b["acc"] * b["n_pairs"]; acc[r["objset"]][1] += b["n_pairs"]
    return {k: round(100 * c / n, 1) for k, (c, n) in acc.items()}


def gravity(d: Path):
    f = d / "per_block.json"
    if not f.exists():
        return None
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity_split/index_test.csv"))}
    by = defaultdict(list)
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity/index.csv")):
        if r["block_id"] in test: by[r["block_id"]].append(r)
    S = json.load(open(f))["per_video_surprise"]
    sc = lambda v: S[v] if isinstance(S[v], (int, float)) else S[v].get("all", list(S[v].values())[0])
    acc = defaultdict(lambda: [0, 0])
    for rs in by.values():
        pos = [r for r in rs if r["plausible"] == "1"][0]; imp = [r for r in rs if r["plausible"] == "0"]
        ok = int(sc(pos["video_id"]) < min(sc(r["video_id"]) for r in imp))
        acc[pos["objset"]][0] += ok; acc[pos["objset"]][1] += 1
    return {k: round(100 * c / n, 1) for k, (c, n) in acc.items()}


def dirs(ds, m):
    return [d for d in sorted(IP.glob(f"{ds}_{m}_wgarrido_w[0-9]*")) if (d / "per_window.json").exists() and (d / "summary.json").exists()]


def intphys1(m):
    ds = dirs("intphys1_dev", m)
    if len(ds) < 2:
        return None
    combos = {}
    for d in ds: combos.update(G.rescore(d)["combos"])
    b = G.select(combos, "intphys")
    return {"acc": round(b["accuracy"], 1), "combo": b["combo"]}


def inflevel(m):
    per = {}
    for p in G.INFLEVEL_PROPS:
        ds = dirs(f"inflevel_{p}", m)
        if len(ds) < 2:
            return None
        per[p] = {}
        for d in ds: per[p].update(G.rescore(d)["combos"])
    best = None
    for combo in sorted(set.intersection(*(set(c) for c in per.values()))):
        det = {}
        for p, cb in per.items():
            R = cb[combo]["reductions"]; g = next(iter(R["filtered"]["per_group"]))
            opts = {r: R[r]["per_group"][g]["acc"] for r in G._cands(R, "inflevel")}
            br = max(opts, key=opts.get); det[p] = round(opts[br], 1)
        mac = sum(det.values()) / 3
        if best is None or mac > best[0]:
            best = (mac, combo, det)
    return {"macro": round(best[0], 1), "combo": best[1], **best[2]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+"); ap.add_argument("--vob"); ap.add_argument("--grav"); a = ap.parse_args()
    for m in a.models:
        v = vob_late(Path(a.vob) if a.vob else TB / f"vobench_v1_split2_test_{m}")
        g = gravity(Path(a.grav) if a.grav else TB / f"vobench_v1_gravity_{m}")
        out = {"model": m,
               "VOBench_P/R": (v["primitive"], v["realistic"]) if v else "⏳",
               "Gravity_P/R": (g["primitive"], g["realistic"]) if g else "⏳",
               "IntPhys1": intphys1(m) or "⏳", "InfLevel": inflevel(m) or "⏳"}
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
