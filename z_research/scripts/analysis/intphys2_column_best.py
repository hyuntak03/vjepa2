#!/usr/bin/env python3
"""IntPhys 2 — 논문 Table 2 · 3 방식 채점: 열 (난이도 · 조건 × 카메라) 마다 최고 하이퍼파라미터를 따로 고른다. 2026-09-19.

논문 Table 2 캡션: "For a given model, we only report its best run for a given column (except the Held Out ...)".
우리 채점기 (analysis/intphys2/eval.py) 는 overall 최고 C 하나를 고르고 그 C 의 세부값을 쓴다 → 여기서 per_video.csv 로 다시 계산.
짝 = (scene_index, pair_id) 안의 가능 1 + 불가능 1, 동점 0.5 (analysis/intphys2/metrics.py:pairwise_accuracy 와 같은 규칙).
하이퍼파라미터 격자 = context_length × 창. 집계는 **avg 하나**다 — 쌍 비교이므로
(논문 D.1 "We use average surprise unless specified otherwise"). Max 는 2026-09-21 철회. ⚠️ 열마다 고른 최고값은 같은 평가셋에서 고른 것이라 descriptive 다.

  python z_research/scripts/analysis/intphys2_column_best.py --run z_exp/intphys2/vjepa2_vith_intphys2_main_repro [--agg avg max]
출력: <run>/column_best.{json,md}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

PAPER_VJEPA2_H = {"Easy": 54.00, "Medium": 58.50, "Hard": 59.38, "Overall": 57.51}
PAPER_T3 = {("permanence", "Fixed"): 59.62, ("permanence", "Moving"): 57.35, ("immutability", "Fixed"): 55.77, ("immutability", "Moving"): 58.72,
            ("continuity", "Fixed"): 57.69, ("continuity", "Moving"): 75.00, ("solidity", "Fixed"): 46.15, ("solidity", "Moving"): 58.51}


def pair_acc(df, col):
    n = c = t = 0
    for _, g in df.groupby(["scene_index", "pair_id"]):
        p, i = g[~g["is_impossible"]], g[g["is_impossible"]]
        if len(p) != 1 or len(i) != 1:
            continue
        sp, si = float(p[col].iloc[0]), float(i[col].iloc[0]); n += 1
        c += si > sp; t += si == sp
    return (100 * (c + 0.5 * t) / n if n else float("nan")), n, t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--agg", nargs="+", default=["avg"])   # 쌍 비교 = AvgSurprise 하나 (2026-09-21 Max 철회)
    ap.add_argument("--scenes", default=None, help="장면 목록 txt (예: data_csv/IntPhys2/main_split/test_scenes.txt) — 그 장면만 채점")
    ap.add_argument("--out", default="column_best", help="출력 파일 이름 (확장자 제외)")
    a = ap.parse_args(); run = Path(a.run)
    df = pd.read_csv(run / "per_video.csv"); df["is_impossible"] = df["is_impossible"].astype(str) == "True"
    if a.scenes:
        keep = {int(s) for s in open(a.scenes).read().split()}; df = df[df["scene_index"].astype(int).isin(keep)]
    cols = [("Easy", df["difficulty"] == "Easy"), ("Medium", df["difficulty"] == "Medium"), ("Hard", df["difficulty"] == "Hard"),
            ("Unknown", df["difficulty"] == "Unknown"), ("Overall", df["difficulty"].notna())]
    cols += [(f"{c}|{cam}", (df["condition"] == c) & (df["camera"] == cam)) for c, cam in PAPER_T3]
    Cs = sorted(df["context_length"].unique()); grid = [(C, g) for g in a.agg for C in Cs]
    out = {"run": str(run), "scenes": a.scenes, "grid": [f"C{C}/{g}" for C, g in grid], "columns": {}}
    for name, m in cols:
        sub = df[m]; per = {}
        for C, g in grid:
            acc, n, t = pair_acc(sub[sub["context_length"] == C], f"surprise_{g}"); per[f"C{C}/{g}"] = {"acc": acc, "n_pairs": n, "ties": t}
        best = max(per, key=lambda k: per[k]["acc"])
        out["columns"][name] = {"best": best, "best_acc": per[best]["acc"], "n_pairs": per[best]["n_pairs"], "per": per}
    # overall 최고 설정 하나로 모든 열을 읽은 값 (우리 채점기 방식) 도 같이
    ob = out["columns"]["Overall"]["best"]; out["overall_best_setting"] = ob
    json.dump(out, open(run / f"{a.out}.json", "w"), indent=1)
    L = [f"# IntPhys 2 열별 최고 선택 — `{run.name}`" + (f" (장면 {a.scenes})" if a.scenes else "") + "\n", f"격자 {len(grid)} 개: " + ", ".join(out["grid"]) + "\n",
         "| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (" + ob + ") 로 읽은 값 | 논문 V-JEPA 2-h |", "|---|---:|---:|---:|---:|"]
    for name, _ in cols:
        c = out["columns"][name]; key = tuple(name.split("|")) if "|" in name else None
        paper = PAPER_T3.get(key) if key else PAPER_VJEPA2_H.get(name)
        L.append(f"| {name} | {c['n_pairs']} | {c['best_acc']:.2f} ({c['best']}) | {c['per'][ob]['acc']:.2f} | {'' if paper is None else f'{paper:.2f}'} |")
    L.append("\n## C 별 Overall\n\n| 설정 | Overall | 동점 |\n|---|---:|---:|")
    L += [f"| {k} | {v['acc']:.2f} | {v['ties']} |" for k, v in out["columns"]["Overall"]["per"].items()]
    L.append("\n⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).\n\n## 재현\n\n```\n"
             f"python z_research/scripts/analysis/intphys2_column_best.py --run {a.run}" + (f" --scenes {a.scenes} --out {a.out}" if a.scenes else "") + "\n```")
    (run / f"{a.out}.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
