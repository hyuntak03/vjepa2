"""IntPhys 2 Main 쌍 정확도를 장면 속성별로 나눈다 (condition · camera · occluder · difficulty · game).

  python z_research/scripts/analysis/intphys2_breakdown.py \
      --run "ar_state=cvpr/results/eval/intphys2/intphys2_main_vith_ariel_ar_state_ep40_wip2_w*" \
      --run "release=z_research/previous/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith" \
      [--cell 24] [--out <dir>]

입력 = 채점 실행 폴더 (창 하나 = 폴더 하나, per_video.csv). 칸 = (창, C). 모델마다
  best   : 전체 정확도가 가장 높은 칸 (Garrido D.3 의 보고 방식, descriptive)
  --cell : 모든 모델에 같은 C (창은 그 C 를 가진 가장 큰 창) — best 선택 편향 대조
CI = 장면 (SceneIndex, 쌍 2 개) 부트스트랩 95 %, 2,000 회. per_video.row_index = metadata.csv 행.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

META = "/data2/local_datasets/world/Benchmark/IntPhys2/Main/metadata.csv"
AXES = ["condition", "camera", "condition×camera", "occluder", "condition×occluder", "difficulty", "game"]


def load_run(pattern: str) -> pd.DataFrame:
    dirs = sorted(d for d in glob.glob(pattern) if Path(d, "per_video.csv").exists())
    if not dirs:
        raise FileNotFoundError(pattern)
    out = []
    for d in dirs:
        df = pd.read_csv(Path(d, "per_video.csv"))
        df["window"] = int(yaml.safe_load(open(Path(d, "config.resolved.yaml")))["surprise"]["window_size"])
        df["run"] = Path(d).name
        out.append(df)
    return pd.concat(out, ignore_index=True)


def pair_table(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """(run, C, 장면, 쌍) 마다 맞춤 여부 + 속성."""
    df = df.join(meta[["occluder", "game_name"]], on="row_index")
    key = ["run", "window", "context_length", "scene_index", "pair_id"]
    piv = df.pivot_table(index=key, columns="is_impossible", values="surprise_avg", aggfunc="first")
    attr = df[~df.is_impossible].set_index(key)[["condition", "camera", "difficulty", "occluder", "game_name"]]
    t = piv.join(attr).reset_index()
    t["hit"] = np.where(t[True] > t[False], 1.0, np.where(t[True] == t[False], 0.5, 0.0))
    t["occluder"] = np.where(t["occluder"].isna(), "none", "occluder")
    t["condition×camera"] = t["condition"] + "/" + t["camera"]
    t["condition×occluder"] = t["condition"] + "/" + t["occluder"]
    t["game"] = t["condition"] + "/" + t["game_name"]
    return t


def ci(hits: np.ndarray, scenes: np.ndarray, rng, n=2000):
    u, inv = np.unique(scenes, return_inverse=True)
    s = np.bincount(inv, hits); c = np.bincount(inv)
    idx = rng.integers(0, len(u), (n, len(u)))
    b = s[idx].sum(1) / c[idx].sum(1) * 100
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def breakdown(t: pd.DataFrame, rng) -> dict:
    out = {"overall": {"acc": t.hit.mean() * 100, "n": len(t), "ci": ci(t.hit.values, t.scene_index.values, rng)}}
    for ax in AXES:
        out[ax] = {g: {"acc": x.hit.mean() * 100, "n": len(x), "ci": ci(x.hit.values, x.scene_index.values, rng)}
                   for g, x in t.groupby(ax)}
    return out


def pick(t: pd.DataFrame, C=None):
    acc = t.groupby(["run", "window", "context_length"]).hit.mean()
    if C is None:
        r, w, c = acc.idxmax()
    else:
        cand = [k for k in acc.index if k[2] == C]
        if not cand:
            return None, None
        r, w, c = max(cand, key=lambda k: k[1])
    return t[(t.run == r) & (t.context_length == c)], f"w{w} C{c}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=glob (창별 결과 폴더)")
    ap.add_argument("--cell", type=int, default=None, help="공통 C")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = pd.read_csv(META)
    rng = np.random.default_rng(0)
    R = {}
    for spec in a.run:
        label, pattern = spec.split("=", 1)
        t = pair_table(load_run(pattern), meta)
        R[label] = {}
        for mode, C in (("best", None),) + ((("cell", a.cell),) if a.cell else ()):
            sel, name = pick(t, C)
            if sel is not None:
                R[label][mode] = {"cell": name, **breakdown(sel, rng)}
    labels = list(R)
    md = []
    for mode in ("best", "cell"):
        cols = [l for l in labels if mode in R[l]]
        if not cols:
            continue
        md += [f"## {'모델마다 최고 칸' if mode == 'best' else f'공통 C = {a.cell}'}", "",
               "| 축 | 값 | 쌍 | " + " | ".join(f"{l} ({R[l][mode]['cell']})" for l in cols) + " |",
               "|---|---|---:|" + "---:|" * len(cols)]
        f = lambda d: f"{d['acc']:.1f} [{d['ci'][0]:.0f}, {d['ci'][1]:.0f}]"
        md.append("| overall | | " + str(R[cols[0]][mode]["overall"]["n"]) + " | " + " | ".join(f(R[l][mode]["overall"]) for l in cols) + " |")
        for ax in AXES:
            for g in R[cols[0]][mode][ax]:
                md.append(f"| {ax} | {g} | {R[cols[0]][mode][ax][g]['n']} | "
                          + " | ".join(f(R[l][mode][ax][g]) if g in R[l][mode][ax] else "—" for l in cols) + " |")
        md.append("")
    text = "\n".join(md)
    print(text)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "breakdown.md").write_text(text + "\n")
        (o / "breakdown.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
