#!/usr/bin/env python3
"""시트 점수 표 (2026-11-06, 자동 탐색용 표준 표). GPU 불필요.
시트 점수 = matched pair 쌍 정확도 (사라짐 쌍 A 와 나타남 쌍 B 의 평균). 칸마다 '점수 (A/B)'.
보드: VOBench v3 (판 · 지나가는 × 움직임 · 정지) · Surrogate (움직임 · 정지) · IntPhys1 late (가림) · IntPhys2 late · real 95 · real 보류 48.
  /venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py <모델>[:<접미사>] ...      예: vast_ar_vw_state_v4b vast_ar_vw_state_v4b:_e11
  --md  마크다운 표로
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, RES

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
COLS = [("VOB판움직", "VOBench v3 late · panel", "vobench_v3", True), ("VOB판정지", "VOBench v3 late · panel", "vobench_v3", False),
        ("VOB지나움직", "VOBench v3 late · moving occluder", "vobench_v3", True), ("VOB지나정지", "VOBench v3 late · moving occluder", "vobench_v3", False),
        ("Sur움직", "IntPhys1 Surrogate", "intphys1_surrogate", True), ("Sur정지", "IntPhys1 Surrogate", "intphys1_surrogate", False),
        ("IP1late", "IntPhys1 late · O1 occluded", None, None), ("IP2late", "IntPhys2 late · permanence (A=pair 1, B=pair 2)", None, None),
        ("real95", "real world (all 95)", None, None), ("real48", "real world (held-out 48)", None, None)]


def cell(B, mv, m, suf, bn, ds, mov):
    pat, blk = B[bn]
    if ds:
        blk = {b: v for b, v in blk.items() if (b in mv[ds]) == mov}
    f = f"{RES}/{pat.format(m=m)}{suf}/per_block.json"
    if not os.path.exists(f):
        return None
    S = {k: (v["all"] if isinstance(v, dict) else v) for k, v in json.load(open(f))["per_video_surprise"].items()}
    r = scene_stats(S, blk)
    if r is None:
        return None
    s = summarize(r)
    return (s["A"] + s["B"]) / 2, s["A"], s["B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--json", default=None, help="결과를 json 으로도 저장")
    a = ap.parse_args()
    B = boards()
    mv = {d: {r["block_id"] for r in csv.DictReader(open(f"{REPO}/data_csv/{d}/index.csv")) if r["motion"] != "flat_static"} for d in ("vobench_v3", "intphys1_surrogate")}
    out = {}
    if a.md:
        print("| 모델 | " + " | ".join(c[0] for c in COLS) + " |\n|---|" + "---|" * len(COLS))
    for ms in a.models:
        m, suf = (ms.split(":", 1) + [""])[:2]
        vals = [cell(B, mv, m, suf, bn, ds, mov) for _, bn, ds, mov in COLS]
        out[ms] = {c[0]: v for c, v in zip(COLS, vals)}
        txt = [("—" if v is None else f"{v[0]:.1f} ({v[1]:.0f}/{v[2]:.0f})") for v in vals]
        print(("| " + ms + " | " + " | ".join(txt) + " |") if a.md else (ms + "  " + "  ".join(f"{c[0]}={t}" for c, t in zip(COLS, txt))))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
