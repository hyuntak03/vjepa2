#!/usr/bin/env python3
"""평가 행렬 (z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md) 의 하위 열 표 — 모델 4 × 벤치 하위 목록 (2026-10-15).

  python z_research/scripts/analysis/eval_matrix_breakdown.py            # 마크다운 표를 stdout 으로 (없는 칸은 ⏳)

VOBench  : split v2 test 360 block (matched, 동점 0.5) — 가림 4 × 위반 3 (p + r 합친 쌍 정확도) + late p / r
gravity  : gravity test 108 block, 고른 미래 = 진짜 (argmin) — ledge / arc × p / r (vobench_gravity_pick.py 와 같은 셈)
InfLevel : property 마다 공식 규칙 (A.8: 창 · skip combo 중 property 별 최고 C 를 고르고 3 property macro 최고 combo) 으로 고른
           (combo, C) 그대로 쌍 유형별로 나눈다 — 하위 열마다 C 를 다시 고르지 않는다. 쌍 유형 = 가능 영상 이름의 코드 (vv/vi · ii/iv · ui/uv · cv/ci).
           ⚠️ 코드의 뜻 (v/i · u/c) 은 레포 문서에 없다 — 확인 전까지 코드 그대로 적는다.
GRASP    : 16 시나리오를 이름 앞부분으로 7 묶음 (Gravity* · Inertia* · ObjectPermanence* · Solidity* · Unchangeableness* · Collision · Continuity).
           공식 규칙 (property 별 최고 C) 은 16 시나리오 단위로 고르고, 묶음은 그 값들의 평균 (macro) 이다.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import garrido_rescore as G  # noqa: E402

EX = REPO / "z_research/VOBench_v1/exp_results"
BENCH = REPO / "z_research/previous/Benchmarks/exp_results"
CVPR = REPO / "cvpr/results/eval/intphys1"
MODELS = [("release", "vith"), ("release + PFT", "vith_release_vobftg"),
          ("vast AR", "vith_vast_ar_state_ep70"), ("vast AR + PFT", "vith_vast_ar_state_ep70_vobftg"),
          ("ariel AR-state pixdiff ep59", "vith_ariel_ar_state_pixdiff_ep59")]      # 2026-10-17 추가 행 (행렬 밖, 사용자 요청)
VOB_RUN = {"vith": EX / "surprise_c16t32__vobench_v1_vith",
           "vith_release_vobftg": REPO / "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest",
           "vith_vast_ar_state_ep70": EX / "surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70",
           "vith_vast_ar_state_ep70_vobftg": REPO / "z_training/runs/vobench_vast_ar_state_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest",
           "vith_ariel_ar_state_pixdiff_ep59": EX / "surprise_c16t32__vobench_v1_vith_ariel_ar_state_pixdiff_ep59"}
WAIT = "⏳"


def fmt(c, n):
    return f"{100 * c / n:.1f}" if n else WAIT


# ---------------------------------------------------------------- VOBench
def vobench():
    meta = {r["block_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    occ, vio = ("visible", "early", "mid", "late"), ("permanence", "shape", "color")
    print("### VOBench — split v2 test (matched 쌍 정확도 %, chance 50) · 칸 = p + r 합침 · 마지막 두 열 = late p / r\n")
    head = [f"{o[:3]}·{v[:4]}" for o in occ for v in vio]
    print("| 모델 | " + " | ".join(head) + " | late p | late r | 전체 |")
    print("|---|" + "---:|" * (len(head) + 3))
    for name, m in MODELS:
        f = VOB_RUN[m] / "per_block.json"
        if not f.exists():
            print(f"| {name} | " + " | ".join([WAIT] * (len(head) + 3)) + " |"); continue
        acc = defaultdict(lambda: [0.0, 0])
        for b in json.load(open(f))["per_block"]:
            r = meta.get(b["block_id"])
            if r is None:
                continue
            for k in ((r["occlusion"], r["violation"]), ("late", r["objset"]) if r["occlusion"] == "late" else None, ("all",)):
                if k:
                    acc[k][0] += b["acc"] * b["n_pairs"]; acc[k][1] += b["n_pairs"]
        cells = [fmt(*acc[(o, v)]) for o in occ for v in vio] + [fmt(*acc[("late", "primitive")]), fmt(*acc[("late", "realistic")]), fmt(*acc[("all",)])]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n칸 크기 (쌍): visible 위반마다 24 · early / mid / late 위반마다 72 · late p / r 108\n")


# ---------------------------------------------------------------- gravity
def gravity():
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity_split/index_test.csv"))}
    rows = [r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity/index.csv")) if r["block_id"] in test]
    by = defaultdict(list)
    for r in rows: by[r["block_id"]].append(r)
    print("### VOBench gravity — gravity test 108 block, 고른 미래 = 진짜 % (chance ledge 50 · arc 11.1)\n")
    print("| 모델 | ledge p | ledge r | arc p | arc r | 전체 (chance 37.0) |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m in MODELS:
        f = EX / f"surprise_c16t32__vobench_v1_gravity_{m}/per_block.json"
        if not f.exists():
            print(f"| {name} | " + " | ".join([WAIT] * 5) + " |"); continue
        S = json.load(open(f))["per_video_surprise"]
        acc = defaultdict(lambda: [0, 0])
        for b, rs in by.items():
            pos = [r for r in rs if r["plausible"] == "1"][0]; imp = [r for r in rs if r["plausible"] == "0"]
            ok = int(S[pos["video_id"]] < min(S[r["video_id"]] for r in imp))
            for k in ((pos["law"], pos["objset"]), ("all",)):
                acc[k][0] += ok; acc[k][1] += 1
        cells = [fmt(*acc[(l, o)]) for l in ("ledge", "arc") for o in ("primitive", "realistic")] + [fmt(*acc[("all",)])]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\n칸 크기 (block): ledge p / r 36 · arc p / r 18\n")


# ---------------------------------------------------------------- sliding (InfLevel · GRASP)
def sliding_dirs(dataset, model):
    """옛 Benchmarks 폴더 (intphys1_sliding__<ds>_<model>_w*) 와 cvpr 폴더 (<ds>_<model>_wgarrido_w*) 를 다 찾는다 (끝난 것만)."""
    ds = sorted(BENCH.glob(f"intphys1_sliding__{dataset}_{model}_w[0-9]*")) + sorted(CVPR.glob(f"{dataset}_{model}_wgarrido_w[0-9]*"))
    return [d for d in ds if (d / "per_window.json").exists() and (d / "summary.json").exists()]


def combos_of(d, group_by, derive=None):
    """garrido_rescore.rescore 와 같은 셈 (combo × 축약 → per_group), group_by 열로 나눈다."""
    _, meta = G.load_index(d)
    if derive:
        for r in meta.values(): r["_sub"] = derive(r)
    pw = json.load(open(d / "per_window.json"))["windows"]
    out = {}
    for combo in sorted({w[0] for rec in pw.values() for w in rec}):
        vids, _, Cs, arr = G.score_table(pw, combo)
        R = {}
        for j, C in enumerate(Cs):
            sc = {v: float(np.nanmean(arr[i, :, j])) for i, v in enumerate(vids)}
            R[f"C{C}"] = {g: G.pairwise(sc, meta, group_by=gb)[0] for g, gb in (("prop", "block_type"), ("sub", group_by))}
        out[combo] = R
    return out


INF_SUB = {"continuity": ("vv", "ii"), "gravity": ("ui", "cv"), "solidity": ("ui", "cv")}


def inflevel():
    print("### InfLevel — 공식 규칙으로 고른 (combo, C) 에서 쌍 유형별 쌍 정확도 % (chance 50)\n")
    cols = [(p, c) for p in ("continuity", "gravity", "solidity") for c in INF_SUB[p]]
    pair_name = {"vv": "vv/vi", "ii": "ii/iv", "ui": "ui/uv", "cv": "cv/ci"}
    print("| 모델 | " + " | ".join(f"{p[:4]} {pair_name[c]}" for p, c in cols) + " | cont | grav | sol | 고른 combo |")
    print("|---|" + "---:|" * (len(cols) + 3) + "---|")
    derive = lambda r: r["video_id"].split("__")[4] if r["plausible"] == "1" else None
    for name, m in MODELS:
        per = {}
        for p in ("continuity", "gravity", "solidity"):
            ds = sliding_dirs(f"inflevel_{p}", m)
            if len(ds) < 2:                        # 창 16 · 32 가 다 있어야 공식 탐색이 된다
                per = None; break
            per[p] = {}
            for d in ds: per[p].update(combos_of(d, "_sub", derive))
        if per is None:
            done = [p for p in ("continuity", "gravity", "solidity") if len(sliding_dirs(f"inflevel_{p}", m)) >= 2]
            print(f"| {name} | " + " | ".join([WAIT] * (len(cols) + 3)) + f" | ⏳ (끝난 property: {', '.join(done) or '없음'}) |"); continue
        shared = set.intersection(*(set(v) for v in per.values()))
        best = None
        for combo in sorted(shared):
            pick = {}
            for p, cb in per.items():
                R = cb[combo]
                acc = {C: (lambda g: 100 * g[p][1] / g[p][0])(R[C]["prop"]) for C in R}
                C = max(acc, key=acc.get); pick[p] = (C, acc[C])
            macro = np.mean([v[1] for v in pick.values()])
            if best is None or macro > best[1]:
                best = (combo, macro, pick)
        combo, _, pick = best
        cells = []
        for p, c in cols:
            g = per[p][combo][pick[p][0]]["sub"][c]
            cells.append(f"{100 * g[1] / g[0]:.1f}")
        cells += [f"{pick[p][1]:.1f}" for p in ("continuity", "gravity", "solidity")]
        print(f"| {name} | " + " | ".join(cells) + f" | {combo} ({', '.join(f'{p[:4]} {pick[p][0]}' for p in pick)}) |")
    print("\n칸 크기 (쌍): continuity 558 · 558 / gravity 591 · 591 / solidity 225 · 225\n")


GRASP_GROUP = [("영속", "ObjectPermanence"), ("연속", "Continuity"), ("중력", "Gravity"), ("관성", "Inertia"),
               ("고체성", "Solidity"), ("충돌", "Collision"), ("불변성", "Unchangeableness")]


def grasp_group(s):
    for g, pre in GRASP_GROUP:
        if s.startswith(pre) and not (pre == "Continuity" and s != "Continuity"):
            return g
    return "?"


def grasp():
    print("### GRASP level2 — 시나리오 16 을 7 묶음 (묶음 = 시나리오별 최고 C 값의 평균) · 쌍 정확도 % (chance 50)\n")
    print("| 모델 | " + " | ".join(g for g, _ in GRASP_GROUP) + " | macro 16 | 고른 combo |")
    print("|---|" + "---:|" * (len(GRASP_GROUP) + 1) + "---|")
    for name, m in MODELS:
        ds = sliding_dirs("grasp_level2", m)
        if len(ds) < 2:
            print(f"| {name} | " + " | ".join([WAIT] * (len(GRASP_GROUP) + 1)) + " | ⏳ |"); continue
        cb = {}
        for d in ds: cb.update(combos_of(d, "block_type"))
        best = None
        for combo, R in cb.items():
            scen = sorted({s for C in R for s in R[C]["prop"]})
            pick = {s: max(100 * R[C]["prop"][s][1] / R[C]["prop"][s][0] for C in R) for s in scen}
            macro = np.mean(list(pick.values()))
            if best is None or macro > best[1]:
                best = (combo, macro, pick)
        combo, macro, pick = best
        grp = defaultdict(list)
        for s, a in pick.items(): grp[grasp_group(s)].append(a)
        print(f"| {name} | " + " | ".join(f"{np.mean(grp[g]):.1f}" for g, _ in GRASP_GROUP) + f" | {macro:.1f} | {combo} |")
    print("\n묶음 크기 (시나리오 수 × 128 쌍): 영속 3 · 연속 1 · 중력 5 · 관성 2 · 고체성 2 · 충돌 1 · 불변성 2\n")


if __name__ == "__main__":
    vobench(); gravity(); inflevel(); grasp()
