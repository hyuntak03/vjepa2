#!/usr/bin/env python3
"""시트 AR_full_tabl 칸 계산기 (2026-11-09, 사용자 "Intuitive Physics 탭처럼 permanence 말고 다른 능력도, 사라짐 A / B 표시할 수 있는 것들은"). GPU 불필요.

모든 칸은 이 기계 (vast) 의 채점 산출물에서 다시 센다. 판정은 쌍 비교 s(불가능) > s(가능), 같으면 0.5.
  VOBench late       shape · color = testbed/vobench_v1_<m> (또는 vobench_v1_late_<m>), occlusion = late, objset (primitive · realistic) 별.
                     permanence = testbed/vobench_v3_<m> (11-09 22:4x 사용자 "permanence 는 v3, shape · color 는 v1"), objset 별 A = 사라짐 · B = 나타남 (420 쌍씩).
                     Cosmos 의 v1 은 cosmos/vobench_late_cosmos_ar4b/pairs_surprise_avg.json (v3 결과 없음 → permanence 비움).
  (VOBench gravity 는 11-09 사용자 지시로 표에서 뺐다. 함수는 남겨 둠.)
  IntPhys 1          Garrido A.8 (intphys1_sliding, Filtered), 창 16 · 32 중 전체 최고 창의 O1 · O2 · O3.
  IntPhys1 late occ  intphys1_late/intphys1_late_uni_<m>: 움직임 · 가림 O1 (A = 문맥에서 물체를 봄 → 사라짐, B = 나타남) · O2 · O3, 정지 · 가림 O1–O3 합.
  IntPhys2           Main, 창 (16 · 32 · 48) × 문맥 C 격자에서 전체 최고 (창, C) 의 조건별 쌍 정확도.
  IntPhys2 late occ  testbed/intphys2_late_final_<m>, 조건별 (A/B 라벨 없음).
  Real world perm.   testbed/real_world_permanence_uniform_<m>, A = 사라짐 · B = 나타남.
  GRASP              공식 Garrido 규칙 (garrido_rescore.py): 창마다 종류별 최고 C → 16 종류 macro 가 높은 창. 7 능력 = 종류 평균,
                     두 속성 이름이 붙은 종류는 양쪽에 넣는다 (GravitySupport 는 gravity 만). Intuitive Physics 탭의 V-JEPA 2 · AR 값을 칸마다 재현함 (11-09).
                     ⚠️ summary.json 의 best_avg (Filtered + 쌍 합) 로 내면 다르다 — 11-09 처음 판이 그랬다.
  InfLevel           공식 규칙: 세 세트가 같은 (skip, 창) 을 쓰고 세트마다 최고 C, 세 세트 macro 가 높은 칸 (garrido_rescore.collect_inflevel).
  VOBench gravity    testbed/vobench_v1_gravity_<m>, block 단위 (가능 미래가 그 block 의 모든 불가능보다 낮아야 정답; ledge 우연 50 · arc 우연 11.1).
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_full_table.py [--json out.json]
"""
import argparse
import csv
import sys
import glob
import json
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = f"{REPO}/cvpr/results/eval"
BEN = f"{REPO}/z_research/Benchmarks/exp_results"
RUNS = f"{REPO}/z_training/runs"

# 행: 시트 이름 → (채점 모델 이름, 학습 run 폴더 또는 None)
MODELS = {
    "V-JEPA 2 ViT-H": ("vith", None),
    "Cosmos-1.0-AR-4B": ("cosmos_ar4b", None),
    "VideoMAEv2 ViT-g": ("videomae2g", None),
    "V-JEPA 2.1 ViT-g": ("vjepa21g", None),
    "AR predictor": ("vast_ar_vw_state", "natural_ar_vw_state"),
    "AR + blender_v4bc4 (L1)": ("vast_ar_vw_state_v4bc4", "v4bc4_ar_vw_state_postft"),
    "AR + blender_v4bc4 w/ negative loss (p = 0.2)": ("vast_ar_vw_state_v4bc4_negimp_vanish_app02", "v4bc4_ar_vw_state_negimp_vanish_app02"),
}
GRASP_GROUP = {"permanence": ["ObjectPermanence", "ObjectPermanence2", "ObjectPermanence3"],
               "continuity": ["Continuity", "GravityContinuity", "SolidityContinuity", "SolidityContinuity2"],
               "gravity": ["Gravity", "GravityContinuity", "GravityInertia", "GravityInertia2", "GravitySupport"],
               "inertia": ["Inertia", "Inertia2", "GravityInertia", "GravityInertia2"], "solidity": ["SolidityContinuity", "SolidityContinuity2"],
               "collision": ["Collision"], "unchangeableness": ["Unchangeableness", "Unchangeableness2"]}


def judge(si, sp):
    return 1.0 if si > sp else (0.5 if si == sp else 0.0)


def pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def pv_of(path):
    if not os.path.exists(path):
        return None
    return {k: (v["all"] if isinstance(v, dict) else v) for k, v in json.load(open(path))["per_video_surprise"].items()}


def idx_pairs(index):
    by = defaultdict(dict)
    for r in csv.DictReader(open(index)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    return [(v["pos"], v["imp"]) for v in by.values() if "pos" in v and "imp" in v]


def vob_late(m):
    out = {}
    if m == "cosmos_ar4b":
        L = json.load(open(f"{RES}/cosmos/vobench_late_cosmos_ar4b/pairs_surprise_avg.json"))
        rows = [(r["objset"], r["violation"], r["pair_id"], judge(r["s_imp"], r["s_pos"])) for r in L]
    else:
        pv = pv_of(f"{RES}/testbed/vobench_v1_{m}/per_block.json") or pv_of(f"{RES}/testbed/vobench_v1_late_{m}/per_block.json")
        if pv is None:
            return out
        rows = [(p["objset"], p["violation_type"], p["pair_id"], judge(pv[i["video_id"]], pv[p["video_id"]]))
                for p, i in idx_pairs(f"{REPO}/data_csv/vobench_v1/index.csv") if p["occlusion"] == "late"]
    for os_ in ("primitive", "realistic"):
        tag = "prim" if os_ == "primitive" else "real"
        for v in ("permanence", "shape", "color"):
            sel = [x for o, vt, ab, x in rows if o == os_ and vt == v]
            out[f"vob_{tag}_{v}"] = pct(sel)
            if v == "permanence":
                for ab in ("A", "B"):
                    out[f"vob_{tag}_perm_{ab}"] = pct([x for o, vt, a, x in rows if o == os_ and vt == v and a == ab])
    return out


def vob3_perm(m):
    pv = pv_of(f"{RES}/testbed/vobench_v3_{m}/per_block.json")
    if pv is None:
        return {}
    P = idx_pairs(f"{REPO}/data_csv/vobench_v3/index.csv")
    out = {}
    for os_, tag in (("primitive", "prim"), ("realistic", "real")):
        for ab in ("A", "B"):
            out[f"vob_{tag}_perm_{ab}"] = pct([judge(pv[i["video_id"]], pv[p["video_id"]]) for p, i in P if p["objset"] == os_ and p["pair_id"] == ab])
        out[f"vob_{tag}_permanence"] = pct([judge(pv[i["video_id"]], pv[p["video_id"]]) for p, i in P if p["objset"] == os_])
    return out


def a8_best(dirs):
    """창별 summary.json → 전체 최고 창의 (전체, by_block_type 정확도 dict)."""
    best = None
    for d in dirs:
        f = f"{d}/summary.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        k = (s.get("best_avg") or {}).get("key")
        if not k:
            continue
        cell = s["surprise"][k]
        acc = cell["overall"]["block_pairwise"]
        if best is None or acc > best[0]:
            best = (acc, {t: v["block_pairwise"] for t, v in cell.get("by_block_type", {}).items()}, d)
    return best


def a8_dirs(m, run, bench):
    if run:
        return [f"{RUNS}/{run}/eval/intphys1_sliding__{bench}_latest_w{w}" for w in (16, 32)]
    return [f"{BEN}/intphys1_sliding__{bench}_{m}_w{w}" for w in (16, 32)]


def intphys1(m, run):
    b = a8_best(a8_dirs(m, run, "intphys1_dev"))
    return {} if b is None else {"ip1_O1": round(100 * b[1]["O1"], 1), "ip1_O2": round(100 * b[1]["O2"], 1), "ip1_O3": round(100 * b[1]["O3"], 1)}


def grasp(m, run):
    import pathlib
    import garrido_rescore as GR
    best = None
    for d in a8_dirs(m, run, "grasp_level2"):
        if not os.path.exists(f"{d}/per_window.json"):
            continue
        b = GR.select(GR.rescore(pathlib.Path(d))["combos"], "grasp")
        if best is None or b["accuracy"] > best["accuracy"]:
            best = b
    if best is None:
        return {}
    pg = {k: v["acc"] for k, v in best["per_group"].items()}
    return {f"grasp_{g}": round(sum(pg[t] for t in ts) / len(ts), 1) for g, ts in GRASP_GROUP.items()}


def inflevel(m, run):
    import pathlib
    import garrido_rescore as GR
    base = pathlib.Path(f"{RUNS}/{run}/eval") if run else pathlib.Path(BEN)
    r = GR.collect_inflevel(base, "latest" if run else m)
    if r is None:
        return {}
    return {f"inf_{k}": round(v["acc"], 1) for k, v in r["best"]["per_group"].items()}


def ip1_late(m):
    pv = pv_of(f"{RES}/intphys1_late/intphys1_late_uni_{m}/per_block.json")
    if pv is None:
        return {}
    P = idx_pairs(f"{REPO}/data_csv/intphys1_late_uni/index.csv")
    j = lambda p, i: judge(pv[i["video_id"]], pv[p["video_id"]])
    mo = [(p, i) for p, i in P if p["group"] == "moving/occluded"]
    isA = lambda p: p["event_kind"] != "not_seen_before_d"
    return {"ip1l_mov_perm_A": pct([j(p, i) for p, i in mo if p["block_type"] == "O1" and isA(p)]),
            "ip1l_mov_perm_B": pct([j(p, i) for p, i in mo if p["block_type"] == "O1" and not isA(p)]),
            "ip1l_mov_perm": pct([j(p, i) for p, i in mo if p["block_type"] == "O1"]),
            "ip1l_mov_shape": pct([j(p, i) for p, i in mo if p["block_type"] == "O2"]),
            "ip1l_mov_cont": pct([j(p, i) for p, i in mo if p["block_type"] == "O3"]),
            "ip1l_static": pct([j(p, i) for p, i in P if p["group"] == "static/occluded"])}


def ip2_late(m):
    pv = pv_of(f"{RES}/testbed/intphys2_late_final_{m}/per_block.json")
    if pv is None:
        return {}
    P = idx_pairs(f"{REPO}/data_csv/intphys2_late_final/index.csv")
    return {f"ip2l_{c}": pct([judge(pv[i["video_id"]], pv[p["video_id"]]) for p, i in P if p["condition"] == c])
            for c in ("permanence", "immutability", "continuity", "solidity")}


def rwp(m):
    pv = pv_of(f"{RES}/testbed/real_world_permanence_uniform_{m}/per_block.json")
    if pv is None:
        return {}
    P = idx_pairs(f"{REPO}/data_csv/real_world_permanence_uniform/index.csv")
    return {f"rwp_{ab}": pct([judge(pv[i["video_id"]], pv[p["video_id"]]) for p, i in P if p["pair_id"] == ab]) for ab in ("A", "B")}


def ip2_main(m, run):
    if run:
        dirs = [f"{RUNS}/{run}/eval/intphys2_main_latest_w{w}" for w in (16, 32, 48)]
    else:
        dirs = sorted(glob.glob(f"{RES}/intphys2/intphys2_main_{m}*"))
    best = None
    for d in dirs:
        f = f"{d}/summary.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        if "overall" not in s or "breakdown" not in s:
            continue
        acc = s["overall"]["pairwise_accuracy"]
        if best is None or acc > best[0]:
            best = (acc, s)
    if best is None:
        return {}
    out = {}
    for c in ("permanence", "immutability", "continuity", "solidity"):
        n = k = 0.0
        for key, v in best[1]["breakdown"].items():
            if key.startswith(f"condition={c}|"):
                n += v["n_pairs"]; k += v["n_correct"] + 0.5 * v.get("n_tied", 0)
        out[f"ip2_{c}"] = round(100 * k / n, 1) if n else None
    return out


def gravity(m):
    pv = pv_of(f"{RES}/testbed/vobench_v1_gravity_{m}/per_block.json")
    if pv is None:
        return {}
    blk = defaultdict(lambda: {"pos": [], "imp": [], "objset": None, "law": None})
    for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1_gravity/index.csv")):
        b = blk[r["block_id"]]
        b["pos" if r["plausible"] == "1" else "imp"].append(r["video_id"]); b["objset"] = r["objset"]; b["law"] = r["law"]
    out = {}
    for os_ in ("primitive", "realistic"):
        for law in ("ledge", "arc"):
            ok = []
            for b in blk.values():
                if b["objset"] != os_ or law not in b["law"]:
                    continue
                sp = min(pv[v] for v in b["pos"])
                ok.append(1.0 if all(pv[v] > sp for v in b["imp"]) else 0.0)
            out[f"grav_{'prim' if os_ == 'primitive' else 'real'}_{law}"] = pct(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    table = {}
    for name, (m, run) in MODELS.items():
        row = {}
        for fn in (lambda: {k: v for k, v in vob_late(m).items() if "perm" not in k}, lambda: vob3_perm(m), lambda: intphys1(m, run), lambda: ip1_late(m), lambda: ip2_main(m, run),
                   lambda: ip2_late(m), lambda: rwp(m), lambda: grasp(m, run), lambda: inflevel(m, run)):
            try:
                row.update(fn())
            except FileNotFoundError:
                pass
        table[name] = row
        print(name, json.dumps(row, ensure_ascii=False))
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
