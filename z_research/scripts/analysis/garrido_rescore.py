#!/usr/bin/env python
"""Garrido 프로토콜 채점을 **원시 window loss 에서 다시 계산한다** (GPU 불필요).

    python z_research/scripts/analysis/garrido_rescore.py <결과폴더> [--json out.json]
    python z_research/scripts/analysis/garrido_rescore.py --all          # 전 칸

왜 따로 필요한가 — **공식 코드는 데이터셋마다 context 축약이 다르다.**

  공식 `evals/intuitive_physics/eval.py` 가 내는 두 가지
    · context 길이별 행   `for i,context in enumerate(all_context_lengths)`   (eval.py:236)
    · "Filtered" 행       `all_losses.min(1)[0]`  = 시작점마다 C 들의 최솟값   (eval.py:260)

  논문 A.8
    "For all properties, we choose the context size which gives us the best performance.
     ... **For IntPhys**, we find that using the minimal surprise over all windows ... can be used"

  -> **IntPhys = Filtered(min)** · **GRASP·InfLevel = property 마다 최고 C**

채점기(`evals/world_model_analysis/eval.py`)는 `context_reduce: min` 하나로만 돌기 때문에
GRASP·InfLevel 이 과소평가된다. `per_window.json` 에 (combo, 창 시작, C, surprise) 가
전부 남으므로 여기서 재실행 없이 다시 채점한다.

쌍 짓기: **(block_id, pair_id)** 가 한 쌍이다. IntPhys1 은 `pair_id` 가 block 안 1/2 라
단독으로는 키가 안 된다 (2 개 값밖에 없다). GRASP·InfLevel 은 한 줄이 한 쌍이라 둘이 같다.
`possible` 1 개 + `impossible` 1 개가 아닌 묶음은 버리고 그 수를 보고한다.

집계: 창에 대한 평균 = **AvgSurprise** 하나다. 지표는 공식 `Relative Accuracy (avg)` 뿐이고
(노트북 key_metric 14 곳 전부, 본문 §2 "pairwise classification"), Max 는 **단일 영상 + AUROC**
용이라 (Table S5) 쌍 비교에는 쓰지 않는다. 분기를 두지 않는다 (2026-09-21 Max 철회).
A.8 격자 (skip x window) 는 이미 한 실행 안에 다 들어 있으므로 combo 마다 채점하고 최고를 고른다.
"""
from __future__ import annotations
import argparse, csv, json, os, pathlib, re, sys
from collections import defaultdict

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]


def kind_of(name: str) -> str:
    if "intphys1" in name: return "intphys"
    if "grasp" in name: return "grasp"
    if "inflevel" in name: return "inflevel"
    raise ValueError(f"데이터셋 종류를 모르겠다: {name}")


def load_index(res: pathlib.Path):
    cfg = yaml.safe_load(open(res / "_resolved.yaml"))
    d = cfg["data"]
    rows = list(csv.DictReader(open(os.path.join(d["root"], d.get("index_csv", "index.csv")))))
    meta = {r["video_id"]: r for r in rows}
    return cfg, meta


def pair_key(r):
    return (r.get("block_id", ""), r.get("pair_id", ""))


def score_table(windows: dict, combo: str):
    """-> (videos, starts, Cs, arr[v, start, C])  해당 combo 의 원시 loss."""
    per = defaultdict(dict)
    starts, Cs = set(), set()
    for vid, rec in windows.items():
        for cb, st, C, s in rec:
            if cb != combo:
                continue
            per[vid][(int(st), int(C))] = float(s)
            starts.add(int(st)); Cs.add(int(C))
    starts, Cs = sorted(starts), sorted(Cs)
    vids = sorted(per)
    arr = np.full((len(vids), len(starts), len(Cs)), np.nan)
    si = {s: i for i, s in enumerate(starts)}
    ci = {c: i for i, c in enumerate(Cs)}
    for i, v in enumerate(vids):
        for (st, C), s in per[v].items():
            arr[i, si[st], ci[C]] = s
    return vids, starts, Cs, arr


def pairwise(scores: dict, meta: dict, group_by="block_type"):
    """쌍마다 possible < impossible 인지. -> {group: (n, correct, ties)}"""
    pairs = defaultdict(list)
    for vid, s in scores.items():
        r = meta.get(vid)
        if r is None:
            continue
        pairs[pair_key(r)].append((int(r["plausible"]), s, r.get(group_by, "all")))
    out, bad = defaultdict(lambda: [0, 0, 0]), 0
    for _, members in pairs.items():
        pos = [m for m in members if m[0] == 1]
        imp = [m for m in members if m[0] == 0]
        if len(pos) != 1 or len(imp) != 1:
            bad += 1
            continue
        g = pos[0][2]
        out[g][0] += 1
        out[g][1] += int(pos[0][1] < imp[0][1])
        out[g][2] += int(pos[0][1] == imp[0][1])
    return out, bad


def rescore(res: pathlib.Path):
    cfg, meta = load_index(res)
    pw = json.load(open(res / "per_window.json"))["windows"]
    name = res.name.split("__", 1)[1]
    kind = kind_of(name)
    combos = sorted({w[0] for rec in pw.values() for w in rec})

    report = {"dir": str(res), "dataset_kind": kind, "combos": {}}
    for combo in combos:
        vids, starts, Cs, arr = score_table(pw, combo)
        entry = {"n_starts": len(starts), "context_lengths": Cs, "reductions": {}}
        # 후보 축약: Filtered(min over C) + C 각각
        cand = {"filtered": np.nanmin(arr, axis=2)}
        for j, C in enumerate(Cs):
            cand[f"C{C}"] = arr[:, :, j]
        for red, m in cand.items():
            sc = {v: float(np.nanmean(m[i])) for i, v in enumerate(vids)}   # AvgSurprise
            g, bad = pairwise(sc, meta)
            tot = sum(x[0] for x in g.values()); cor = sum(x[1] for x in g.values())
            entry["reductions"][red] = {
                "overall": cor / tot * 100 if tot else float("nan"),
                "n_pair": tot, "n_ties": sum(x[2] for x in g.values()), "dropped_groups": bad,
                "per_group": {k: {"n": v[0], "acc": v[1] / v[0] * 100} for k, v in sorted(g.items())}}
        report["combos"][combo] = entry

    return report


# 데이터셋마다 **어떤 축약을 후보로 삼는가** — 논문 텍스트가 정본이다.
#
#   A.7 (870-872) "For each property, we select the value of C (C+M being fixed) which maximizes
#                  performance ... **On IntPhys, we are able to get rid of this sweep on context
#                  lengths** by computing the minimal surprise over all context lengths C for each
#                  starting frame t."
#   A.8 (992-996) "For all properties, we choose the context size which gives us the best
#                  performance ... For IntPhys ... **removes one hyperparameter to optimize**."
#
# -> IntPhys 는 C 스윕을 **Filtered 로 대체**한다. GRASP·InfLevel 은 property 마다 최고 C.
#
# ⚠️ 공식 `figures.ipynb` 의 `df[df["Block"]==prop][key_metric].max()` 는 Filtered 행과 C 행을
#    **함께** 놓고 최댓값을 고른다 — IntPhys 에 대해 위 텍스트와 어긋난다 (스윕을 도로 집어넣는다).
#    우리는 **텍스트를 따른다.** 노트북 규칙 값도 같이 계산해 `alt_notebook` 으로 남긴다.
CANDIDATES = {"intphys": ("filtered",), "grasp": "C", "inflevel": "C"}


def _cands(R: dict, kind: str):
    spec = CANDIDATES.get(kind, "C")
    return list(spec) if isinstance(spec, tuple) else [r for r in R if r != "filtered"]


def _macro(R: dict, allowed) -> tuple:
    """property 마다 allowed 중 최고를 고르고 macro 평균. -> (macro, {property: {...}})"""
    pick = {}
    for g in R["filtered"]["per_group"]:
        opts = {r: R[r]["per_group"][g]["acc"] for r in allowed}
        br = max(opts, key=opts.get)
        pick[g] = {"reduction": br, "acc": opts[br], "n": R[br]["per_group"][g]["n"]}
    return (sum(v["acc"] for v in pick.values()) / len(pick) if pick else float("nan")), pick


def select(combos: dict, kind: str):
    """논문 텍스트 규칙으로 combo 마다 점수를 내고 최고를 고른다.

      1. 축약 후보  — IntPhys: Filtered 하나 / GRASP·InfLevel: C 각각  (위 CANDIDATES)
      2. property 마다 그중 최고
      3. 전체 = property 들의 **macro 평균** (`figures.ipynb  np.mean(perfs)`, 쌍 수 가중 아님)
      4. (skip, window) 는 데이터셋 수준에서 고른다 (Table S3)

    `alt_notebook` 에 노트북 규칙({Filtered} ∪ {C} 최고) 값을 같이 남긴다.
    """
    best = None
    for combo, e in combos.items():
        R = e["reductions"]
        acc, pick = _macro(R, _cands(R, kind))
        alt, _ = _macro(R, list(R))
        e["selected"] = {"rule": f"{kind}: " + ("Filtered" if kind == "intphys" else "property별 최고 C")
                                 + " -> macro 평균",
                         "per_group_choice": {g: v["reduction"] for g, v in pick.items()},
                         "accuracy": acc, "alt_notebook": alt,
                         "n_pair": sum(v["n"] for v in pick.values()), "per_group": pick}
        if best is None or acc > best[1]:
            best = (combo, acc, e["selected"])
    return {"combo": best[0], **best[2]}


def collect(base: pathlib.Path, dataset: str, model: str):
    """`<dataset>_<model>_w16` / `_w32` 를 **합쳐** A.8 최고를 고른다.

    창(C+M)마다 실행을 나눈 것은 공식 코드가 창마다 모델을 그 프레임 수로 짓기 때문이다
    (PROTOCOLS.md §2). A.8 은 그 창들을 **하나의 탐색 공간**으로 보므로 여기서 합친다.
    """
    dirs = sorted(base.glob(f"intphys1_sliding__{dataset}_{model}_w*"))
    dirs = [d for d in dirs if (d / "per_window.json").exists()]
    if not dirs:
        return None
    combos, kind = {}, None
    for d in dirs:
        r = rescore(d)
        kind = r["dataset_kind"]
        combos.update(r["combos"])          # combo 이름에 skip/창이 다 들어 있어 충돌하지 않는다
    return {"dataset": dataset, "model": model, "dataset_kind": kind,
            "dirs": [str(d) for d in dirs], "combos": combos,
            "best": select(combos, kind)}


INFLEVEL_PROPS = ["continuity", "solidity", "gravity"]      # 공식 utils.py PROPERTIES_BY_DATASET


def collect_inflevel(base: pathlib.Path, model: str):
    """InfLevel-lab 은 공식에서 **한 데이터셋 3 property** 다
    (`utils.py  PROPERTIES_BY_DATASET['inflevel_lab'] = ['continuity','solidity','gravity']`).

    우리는 영상 길이가 달라 property 마다 데이터셋을 쪼갰으므로 여기서 도로 합친다.
    (skip, window) 는 **데이터셋 수준**에서 고르므로 (Table S3 가 InfLevel-lab/V-JEPA 에
    skip 5 / window 32 하나를 싣는다) combo 마다 3 property 의 macro 를 내고 최고 combo 를 고른다.

    ⚠️ gravity·solidity 는 원리적으로 못 푸는 property 다 (Garrido App. E). 공식 InfLevel 값은
    그 둘을 포함한 macro 라서 구조적으로 낮다. **property 별 값을 항상 같이 낸다.**
    """
    per_ds = {}
    for prop in INFLEVEL_PROPS:
        dirs = sorted((base).glob(f"intphys1_sliding__inflevel_{prop}_{model}_w*"))
        dirs = [d for d in dirs if (d / "per_window.json").exists()]
        if not dirs:
            return None                      # 셋이 다 있어야 공식 값이 된다
        combos = {}
        for d in dirs:
            combos.update(rescore(d)["combos"])
        per_ds[prop] = combos

    shared = set.intersection(*(set(c) for c in per_ds.values()))
    if not shared:
        return None
    best = None
    for combo in sorted(shared):
        accs, detail = [], {}
        for prop, combos in per_ds.items():
            R = combos[combo]["reductions"]
            g = next(iter(R["filtered"]["per_group"]))          # 하위 데이터셋마다 property 1 개
            opts = {r: R[r]["per_group"][g]["acc"] for r in _cands(R, "inflevel")}
            br = max(opts, key=opts.get)
            accs.append(opts[br])
            detail[prop] = {"reduction": br, "acc": opts[br], "n": R[br]["per_group"][g]["n"]}
        macro = sum(accs) / len(accs)
        if best is None or macro > best[1]:
            best = (combo, macro, detail)
    return {"dataset": "inflevel_lab", "model": model, "dataset_kind": "inflevel",
            "best": {"combo": best[0], "accuracy": best[1],
                     "rule": "property별 최고(Filtered 포함) -> 3 property macro 평균",
                     "n_pair": sum(v["n"] for v in best[2].values()),
                     "per_group_choice": {k: v["reduction"] for k, v in best[2].items()},
                     "per_group": best[2]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="결과 폴더 (창별). 비우면 --all")
    ap.add_argument("--all", action="store_true", help="(데이터셋, 모델) 마다 창을 합쳐 채점")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    base = REPO / "z_research/Benchmarks/exp_results"
    out = {}

    if a.dirs:
        for d in map(pathlib.Path, a.dirs):
            if not (d / "per_window.json").exists():
                print(f"-- {d.name}: per_window.json 없음, 건너뜀"); continue
            r = rescore(d); r["best"] = select(r["combos"], r["dataset_kind"])
            out[d.name] = r
            b = r["best"]
            print(f"{d.name:50s} {b['combo']:12s} {b['rule']:22s} {b['accuracy']:6.2f}  n={b['n_pair']}")
    else:
        pairs = sorted({re.sub(r"_w\d+$", "", d.name.split("__", 1)[1])
                        for d in base.glob("intphys1_sliding__*_w*")})
        for key in pairs:
            for m in ("vith", "vjepa21g", "videomae2g"):
                if not key.endswith("_" + m):
                    continue
                ds = key[: -len(m) - 1]
                r = collect(base, ds, m)
                if r is None:
                    continue
                out[key] = r
                b = r["best"]
                print(f"{ds:22s} {m:11s} {b['combo']:12s} {b['rule']:22s} {b['accuracy']:6.2f}  n={b['n_pair']}"
                      f"   (창 {len(r['dirs'])}개 합침)")
                if b["per_group_choice"]:
                    top = list(b["per_group_choice"].items())[:6]
                    print(f"{'':35s}property별 C: {dict(top)}{' …' if len(b['per_group_choice'])>6 else ''}")
        for m in ("vith", "vjepa21g", "videomae2g"):
            r = collect_inflevel(base, m)
            if r is None:
                continue
            out[f"inflevel_lab_{m}"] = r
            b = r["best"]
            print(f"{'inflevel_lab (3 property)':22s} {m:11s} {b['combo']:12s} "
                  f"{'macro':22s} {b['accuracy']:6.2f}  n={b['n_pair']}")
            print(f"{'':35s}property별: " +
                  "  ".join(f"{k}:{v['acc']:.2f}({v['reduction']})" for k, v in b["per_group"].items()))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
        print("저장:", a.json)


if __name__ == "__main__":
    main()
