"""다중 가설 3 의 집합 채점 — 가설마다 · 가설 중 최소 · 가설 평균 (2026-10-25, 학습 없음, GPU 없음).
사용자: "다중 가설로 했을 때 IntPhys2, IntPhys2 late, IntPhys1 측정해봐 … InfLevel 도".

가설 체크포인트는 z_training/tools/bake_wta_hypotheses.py, 채점은 z_training/chain_wta3_eval.sh (기존 채점기 그대로).
집합 점수는 **창 단위** 로 합친다: 같은 영상 · 같은 창 시작 · 같은 문맥 길이의 surprise 를 가설끼리 min (또는 mean) → 그 뒤 각 벤치마크의 원래 집계.
  IntPhys1 A.8   (combo, 시작, C) 마다 합침 → garrido_rescore.collect (Filtered = C 최소, 시작 평균, (skip, 창) 격자 최고)
  InfLevel       같은 합침 → garrido_rescore.collect_inflevel (property 별 최고 C, 3 property macro)
  IntPhys2 Main  (영상, C) 의 창별 surprise 를 합침 → 창 평균 → 쌍 정확도 (동점 0.5), (창 크기, C) 격자 최고 (시트와 같은 descriptive 값)
  IntPhys2 late  영상마다 surprise 하나 (고정 창 c16t32) → 합침 → (block_id, pair_id) 쌍
합친 IntPhys1 · InfLevel 창 파일은 runs/natural_ar_vw_state_wta3_{min,mean}/eval/ 에 쓴다 (collect 가 폴더를 읽어서).

  /venv/vjepa/bin/python z_research/scripts/analysis/wta_set_rescore.py [--check]   # --check = 기준 AR 세 벌로 합쳐도 원래 값이 나오는지
출력: auto_research/results/wta_set_rescore.json
"""
import argparse, csv, json, os, pathlib, shutil, sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from garrido_rescore import collect, collect_inflevel  # noqa: E402

RUNS = REPO / "z_training/runs"
LATE = REPO / "cvpr/results/eval/intphys1_late"
HYP = ["natural_ar_vw_state_wta3_h0", "natural_ar_vw_state_wta3_h1", "natural_ar_vw_state_wta3_h2"]
SINGLE = {"base": "natural_ar_vw_state", "ftl1": "natural_ar_vw_state_ftl1", "h0": HYP[0], "h1": HYP[1], "h2": HYP[2]}
AGG = {"min": np.min, "mean": np.mean}
I1_DIRS = ["intphys1_sliding__intphys1_dev_latest_w16", "intphys1_sliding__intphys1_dev_latest_w32"] + \
    [f"intphys1_sliding__inflevel_{p}_latest_w{w}" for p in ("continuity", "solidity", "gravity") for w in (16, 32)]


def merge_garrido(hyps, out_run):
    """IntPhys1 · InfLevel 창 파일을 가설끼리 합쳐 out_run/eval/<같은 이름>/ 에 쓴다."""
    for agg, f in AGG.items():
        for dn in I1_DIRS:
            srcs = [RUNS / h / "eval" / dn for h in hyps]
            if not all((s / "per_window.json").exists() for s in srcs):
                continue
            pws = [json.load(open(s / "per_window.json")) for s in srcs]
            merged = {}
            for vid in pws[0]["windows"]:
                maps = [{(w[0], w[1], w[2]): w[3] for w in pw["windows"].get(vid, [])} for pw in pws]
                keys = [k for k in maps[0] if all(k in m for m in maps[1:])]
                merged[vid] = [[*k, float(f([m[k] for m in maps]))] for k in keys]
            od = RUNS / f"{out_run}_{agg}" / "eval" / dn
            od.mkdir(parents=True, exist_ok=True)
            shutil.copy(srcs[0] / "_resolved.yaml", od / "_resolved.yaml")
            json.dump({"schema": pws[0].get("schema"), "windows": merged, "merged_from": [str(s) for s in srcs], "agg": agg},
                      open(od / "per_window.json", "w"))


def garrido(run):
    E = RUNS / run / "eval"
    r1 = collect(E, "intphys1_dev", "latest")
    ri = collect_inflevel(E, "latest")
    out = {}
    if r1:
        out["IntPhys1"] = (round(r1["best"]["accuracy"], 2), r1["best"]["combo"])
    if ri:
        out["InfLevel"] = (round(ri["best"]["accuracy"], 2), ri["best"]["combo"],
                           {k: round(v["acc"], 1) for k, v in ri["best"]["per_group"].items()})
    return out


def i2_load(run):
    """창 크기 → {(row_index, C): 창별 surprise 배열}, 메타 (row_index → (scene, pair, impossible))."""
    out, meta = {}, {}
    for w in (16, 32, 48):
        d = RUNS / run / "eval" / f"intphys2_main_latest_w{w}"
        if not (d / "per_window.jsonl").exists():
            continue
        out[w] = {(r["row_index"], r["context_length"]): np.asarray(r["surprise"], dtype=np.float64)
                  for r in map(json.loads, open(d / "per_window.jsonl"))}
        for r in csv.DictReader(open(d / "per_video.csv")):
            meta[int(r["row_index"])] = (r["scene_index"], r["pair_id"], r["is_impossible"] == "True")
    return out, meta


def i2_acc(sc, meta):
    """sc: {(row, C): 평균 surprise} → C 마다 쌍 정확도 (동점 0.5)."""
    byC = defaultdict(lambda: defaultdict(dict))
    for (row, C), s in sc.items():
        scene, pid, imp = meta[row]
        byC[C][(scene, pid)][imp] = s
    res = {}
    for C, pairs in byC.items():
        v = [(p[True] > p[False]) + 0.5 * (p[True] == p[False]) for p in pairs.values() if True in p and False in p]
        res[C] = (100 * float(np.mean(v)), len(v))
    return res


def intphys2(runs, agg=None):
    loads = [i2_load(r) for r in runs]
    meta = loads[0][1]
    best = None
    for w in (16, 32, 48):
        if not all(w in L[0] for L in loads):
            continue
        keys = [k for k in loads[0][0][w] if all(k in L[0][w] for L in loads[1:])]
        sc = {}
        for k in keys:
            arrs = [L[0][w][k] for L in loads]
            sc[k] = float(np.mean(AGG[agg](np.stack(arrs), 0) if agg else arrs[0]))
        for C, (acc, n) in i2_acc(sc, meta).items():
            if best is None or acc > best[0]:
                best = (round(acc, 2), f"w{w} C{C}", n)
    return best


def late(runs, agg=None):
    pvs = []
    for r in runs:
        sfx = "" if r == "natural_ar_vw_state" else "_" + r[len("natural_ar_vw_state_"):]
        f = LATE / f"intphys2_violation_uni_test_vast_ar_vw_state{sfx}" / "per_block.json"
        if not f.exists():
            return None
        pv = json.load(open(f))["per_video_surprise"]
        pvs.append({k: float(v["all"] if isinstance(v, dict) else v) for k, v in pv.items()})
    pairs = defaultdict(dict)
    for row in csv.DictReader(open(REPO / "data_csv/intphys2_violation_uni/index_test.csv")):
        v = row["video_id"]
        if all(v in p for p in pvs):
            s = [p[v] for p in pvs]
            pairs[(row["block_id"], row.get("pair_id", ""))][int(row["plausible"])] = float(AGG[agg](s)) if agg else s[0]
    ok = [p[0] > p[1] for p in pairs.values() if 0 in p and 1 in p]
    return (round(100 * float(np.mean(ok)), 2), len(ok))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    if a.check:   # 기준 AR 세 벌을 합치면 (min = mean = 원래) 원래 값과 같아야 한다
        b = ["natural_ar_vw_state"] * 3
        merge_garrido(b, "_check_base3")
        print("base 그대로   ", garrido("natural_ar_vw_state"), intphys2(["natural_ar_vw_state"]), late(["natural_ar_vw_state"]))
        print("base×3 min    ", garrido("_check_base3_min"), intphys2(b, "min"), late(b, "min"))
        for agg in AGG:
            shutil.rmtree(RUNS / f"_check_base3_{agg}", ignore_errors=True)
        return
    merge_garrido(HYP, "natural_ar_vw_state_wta3")
    res = {}
    for nm, r in SINGLE.items():
        res[nm] = {**garrido(r), "IntPhys2": intphys2([r]), "IntPhys2_late": late([r])}
    for agg in AGG:
        res[f"set_{agg}"] = {**garrido(f"natural_ar_vw_state_wta3_{agg}"), "IntPhys2": intphys2(HYP, agg), "IntPhys2_late": late(HYP, agg)}
    out = REPO / "auto_research/results/wta_set_rescore.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=str)
    for nm, v in res.items():
        print(f"{nm:9s}", {k: (x[0] if isinstance(x, (tuple, list)) else x) for k, x in v.items()})
    print("saved", out)


if __name__ == "__main__":
    main()
