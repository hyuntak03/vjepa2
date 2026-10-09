#!/usr/bin/env python3
"""IntPhys 2 — 창 (16/32/48) 마다 나눠 돈 실행을 **하나의 격자**로 합쳐 열별 최고를 고른다. 2026-09-27.

`intphys2_column_best.py` 는 실행 하나 (창 하나) 안에서만 C 를 고른다. 논문 Table 2 는 창 × C 전체에서
열마다 최고 실행을 보고하므로 (캡션 "best run for a given column") 여기서 창을 합친다.
쌍 채점 (`pair_acc`) · 열 정의는 `intphys2_column_best.py` 의 것을 그대로 쓴다.

격자는 **공식 비율로 제한**한다: C ∈ 창 × {1/4, 3/8, 1/2, 5/8, 3/4, 7/8} (README §2, run_grid.sh).
옛 실행 중에는 더 넓은 sweep 을 돈 것이 있어 (`bench_intphys2_main_vith` 는 창 48 에서 C 4…42) 걸러야 모델끼리 같은 격자가 된다.

  python z_research/scripts/analysis/intphys2_grid_best.py \
      --model vith=bench_intphys2_main_vith_w16,bench_intphys2_main_vith_w32,bench_intphys2_main_vith \
      --model ar_ep18=bench_intphys2_main_ariel_ar_ep18_w16,bench_intphys2_main_ariel_ar_ep18_w32,bench_intphys2_main_ariel_ar_ep18_w48
출력: stdout 표, `--out` 을 주면 json.
⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값이라 descriptive 다 (CLAUDE.md §1-3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intphys2_column_best import PAPER_VJEPA2_H, pair_acc  # noqa: E402

BASE = Path(__file__).resolve().parents[2] / "Benchmarks/exp_results/intphys2"
COLS = ["Easy", "Medium", "Hard", "Overall"]
FRACS = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


def load(run: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(open(run / "config.resolved.yaml"))
    w = int(cfg["surprise"]["window_size"])
    df = pd.read_csv(run / "per_video.csv")
    df["is_impossible"] = df["is_impossible"].astype(str) == "True"
    grid = {int(w * f) for f in FRACS}
    df = df[df["context_length"].isin(grid)].copy()
    df["setting"] = [f"w{w}/C{c}" for c in df["context_length"]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="이름=실행1,실행2,… (BASE 기준 폴더 이름)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {}
    for spec in a.model:
        name, runs = spec.split("=", 1)
        df = pd.concat([load(BASE / r) for r in runs.split(",")], ignore_index=True)
        masks = {"Easy": df["difficulty"] == "Easy", "Medium": df["difficulty"] == "Medium",
                 "Hard": df["difficulty"] == "Hard", "Overall": df["difficulty"].notna()}
        res = {}
        for col in COLS:
            sub = df[masks[col]]
            per = {s: pair_acc(g, "surprise_avg") for s, g in sub.groupby("setting")}
            best = max(per, key=lambda k: per[k][0])
            res[col] = {"best": best, "acc": per[best][0], "n_pairs": per[best][1],
                        "per": {k: {"acc": v[0], "n_pairs": v[1], "ties": v[2]} for k, v in per.items()}}
        ob = res["Overall"]["best"]
        res["overall_setting_cols"] = {c: res[c]["per"][ob]["acc"] for c in COLS}
        out[name] = res
    print("| 모델 | " + " | ".join(COLS) + " | Overall 최고 설정 |")
    print("|---|" + "---:|" * len(COLS) + "---|")
    for name, r in out.items():
        print(f"| {name} | " + " | ".join(f"{r[c]['acc']:.2f}" for c in COLS) + f" | {r['Overall']['best']} |")
    print("| 논문 V-JEPA 2-h | " + " | ".join(f"{PAPER_VJEPA2_H[c]:.2f}" for c in COLS) + " | — |")
    for name, r in out.items():
        print(f"\n[{name}] Overall 설정별: " + ", ".join(f"{k} {v['acc']:.2f}" for k, v in sorted(
            r["Overall"]["per"].items(), key=lambda kv: (int(kv[0].split('/')[0][1:]), int(kv[0].split('C')[1])))))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
