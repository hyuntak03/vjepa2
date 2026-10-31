#!/usr/bin/env python3
"""transfer_gap R1 — 사용자 전제 넷을 작은 산출물에서 다시 센다 (2026-10-30, 검토 [R]). GPU · 큰 캐시 없음, CPU 1 분 안쪽.

다시 세는 것 (전부 이미 있는 산출물에서, 새 학습 없음):
  probe      튜블릿 정체 probe (릴리즈, h 판독기 / p 판독기 / 16 층) — 표본 단위 예측에서 정확도 · t8 · t15 · 정지/이동 · realistic 원색 몫
  occluded   문맥 encoder 의 완전히 가려진 문맥 튜블릿 정체 probe (모양 · 색, 가림 시점별)
  late_id    late 정체 ridge (late 안에서 학습, 블록 평균 vs 참 자리 3×3, 은닉 384 vs 출력 1280) — 릴리즈 · Ariel AR40 · AR-state40
  tempgen    시점 하나로 학습한 판독기 → late p
  readout    VOB_readout 자 (모델마다 p 자 · h 자 · h→h) 를 VOBench primitive 에, split v2 test 만 — 릴리즈 · +PFT · vast AR · +PFT · Ariel AR40
  velocity   vast AR 평균 풀링 속도 probe 의 걸음 공유 기울기 비 (보임 / late)
  late_acc   VOBench late 쌍 정확도 (split v2 test) — 네 모델
  bench      IntPhys 1 · GRASP · InfLevel 공식 규칙 재채점 (창별 per_window.json)

  python z_research/scripts/analysis/transfer_gap_r1_facts.py   # → z_analysis/transfer_gap/results/R1_facts_recount.json
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/R1_facts_recount.json"
VOB = REPO / "z_research/VOBench_v1"
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]


def mot(clip):
    return next((m for m in MOT if f"_{m}_" in clip), "?")


def pct(xs):
    return round(100 * float(np.mean(xs)), 1) if len(xs) else None


def probe():
    R = VOB / "vjepa2/probing/results"
    runs = ["e35_s0_user", "e35_s1", "e10_s0", "e10_s1", "p_only_e35_s0", "p_only_e35_s1", "p_only_e10_s0", "p_only_e10_s1",
            "L16_e35_s0", "L16_e35_s1"]
    out = {}
    for r in runs:
        for obj in ("primitive", "realistic"):
            p = R / r / f"{obj}.json"
            if not p.exists():
                continue
            d = json.load(open(p)); cls = d["classes"]; row = {}
            for cat, v in d["cats"].items():
                s = v.get("samples", [])
                if not s:
                    continue
                ok = [x[4] == x[5] for x in s]
                e = {"n": len(s), "acc": pct(ok), "none": round(v.get("none", float("nan")), 1),
                     "t8": pct([x[4] == x[5] for x in s if x[1] == 8]), "t15": pct([x[4] == x[5] for x in s if x[1] == 15]),
                     "static": pct([x[4] == x[5] for x in s if mot(x[0]) == "flat_static"]),
                     "moving": pct([x[4] == x[5] for x in s if mot(x[0]) != "flat_static"]),
                     "visible_t8": pct([x[4] == x[5] for x in s if "_visible_" in x[0] and x[1] == 8]),
                     "visible_t15": pct([x[4] == x[5] for x in s if "_visible_" in x[0] and x[1] == 15])}
                if "none" not in cls[s[0][5]]:
                    e["shape_only"] = pct([cls[x[4]].split("/")[0] == cls[x[5]].split("/")[0] for x in s])
                    if obj == "realistic":   # realistic 은 80 % 가 원색 (native) — 다수 클래스 효과를 분리
                        nat = [x for x in s if cls[x[5]].endswith("/native")]; pai = [x for x in s if not cls[x[5]].endswith("/native")]
                        e["native_frac"] = round(100 * len(nat) / len(s), 1)
                        e["acc_native"] = pct([x[4] == x[5] for x in nat]); e["acc_painted"] = pct([x[4] == x[5] for x in pai])
                row[cat] = e
            out[f"{r}/{obj}"] = row
    return out


def occluded():
    d = json.load(open(REPO / "z_analysis/occluded_identity_probe/result.json")); out = {}
    for obj, v in d.items():
        for cond, w in v.items():
            if not isinstance(w, dict):
                continue
            for a, x in w.items():
                if isinstance(x, dict) and "test_acc_mean" in x:
                    out[f"{obj}/{cond}/{a}"] = {"acc": round(100 * x["test_acc_mean"], 1), "sd": round(100 * x["test_acc_sd"], 1),
                                                "chance": round(100 * x["chance"], 1), "n_clips": w["n_clips"],
                                                "by_timing": {k: round(100 * t["acc"], 1) for k, t in x.get("by_timing", {}).items()}}
    return out


def late_id():
    d = json.load(open(VOB / "bottleneck/late_identity/late_identity.json")); out = {}
    for m, cells in d.items():
        for c, v in cells.items():
            out[f"{m}/{c}"] = {"n": v["n"], **{k: round(100 * float(np.mean(v[k][1:8])), 1) for k in ("st_g", "st_l", "r_g", "r_l", "h_g", "h_l") if k in v},
                               "step1": {k: round(100 * float(v[k][0]), 1) for k in ("st_l", "r_l", "h_l") if k in v}}
    return out


def tempgen():
    d = json.load(open(VOB / "vjepa2/probing/temporal_generalization/summary.json"))
    return {k: {"chance": round(v["chance_object"], 1), **{h: round(x["mean"], 1) for h, x in v["late"].items()}} for k, v in d.items()}


def readout():
    sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
    import vobench_ar_readout as V
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    rows = [r for r in V.rows_primitive() if r["variant"] in ("pos_a", "pos_b")]
    res = {}
    for name in ("rel", "relft", "vast70", "vast70ft", "ar40"):
        Z = V.load(name); vid = {v: i for i, v in enumerate(Z["video_id"])}
        G = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if r["video_id"] not in vid or r["block_id"] not in test:
                continue
            arm = "A" if r["variant"] == "pos_a" else "B"; obj = r["obj_a"] if arm == "A" else r["obj_b"]
            if obj == "empty":
                continue
            sh, co = obj.split("/"); cls = V.SHAPES.index(sh) * len(V.COLORS) + V.COLORS.index(co)
            i = vid[r["video_id"]]; m = "static" if r["motion"] == "flat_static" else "moving"
            occ = "late" if r["occlusion"] == "late" else "seen"
            for src, key in (("pp", "pp_prob"), ("ph", "ph_prob"), ("hh", "hh_prob")):
                a = Z[key][i].argmax(-1)
                G[(occ, m)][src + "_id"].append(a == cls); G[(occ, m)][src + "_none"].append(a == V.NONE)
        res[name] = {f"{o}/{m}": {"n": len(d["pp_id"]), **{k: pct(v) for k, v in d.items()},
                                  **{k + "_t": [round(100 * float(x)) for x in np.mean(v, 0)] for k, v in d.items() if k.endswith("_id")}}
                     for (o, m), d in G.items()}
    return res


def velocity():
    d = json.load(open(REPO / "z_analysis/ar_layer_velocity_probe/vast70/probe.json")); out = {}
    for rep in ("p", "L6", "h(천장)"):
        for g in ("not_late", "late"):
            t = d["shared"][rep]["traj"][g]
            out[f"{rep}/{g}"] = {m: round((v["read"][-1] - v["read"][0]) / (v["true"][-1] - v["true"][0]), 2)
                                 for m, v in t.items() if m in ("flat_acc", "flat_dec", "ramp_down", "ramp_up")}
    return out


def late_acc():
    test = {r["block_id"]: r["objset"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))
            if r["occlusion"] == "late"}
    runs = {"release": VOB / "exp_results/surprise_c16t32__vobench_v1_vith/per_block.json",
            "release+PFT(g)": REPO / "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json",
            "vastAR": VOB / "exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70/per_block.json",
            "vastAR+PFT(g)": REPO / "z_training/runs/vobench_vast_ar_state_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json"}
    out = {}
    for name, p in runs.items():
        acc = collections.defaultdict(lambda: [0.0, 0])
        for b in json.load(open(p))["per_block"]:
            o = test.get(str(b["block_id"]))
            if o:
                acc[o][0] += b["acc"] * b["n_pairs"]; acc[o][1] += b["n_pairs"]
        out[name] = {o: (round(100 * a / n, 1), n) for o, (a, n) in acc.items()}
    return out


def bench():
    spec = importlib.util.spec_from_file_location("gr", REPO / "z_research/scripts/analysis/garrido_rescore.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    base = REPO / "cvpr/results/eval/intphys1"; out = {}

    def comb(prefix):
        dirs = [d for d in sorted(base.glob(prefix + "_w*")) if (d / "per_window.json").exists() and not d.name.endswith(("arcopy", "smoke24"))]
        if not dirs:
            return None
        combos, kind = {}, None
        for d in dirs:
            r = gr.rescore(d); kind = r["dataset_kind"]; combos.update(r["combos"])
        b = gr.select(combos, kind); return {"combo": b["combo"], "acc": round(b["accuracy"], 2), "n_pair": b["n_pair"]}

    models = ["vith", "vith_release_vobftg", "vith_vast_ar_state_ep70", "vith_vast_ar_state_ep70_vobftg", "vith_pft_intphys1_e40",
              "vith_pft_ip2pair", "vith_pft_ip2pair_lora", "vith_phys2d_futureA", "vith_pft_single_nosup"]
    for ds in ("intphys1_dev", "grasp_level2"):
        for m in models:
            out[f"{ds}/{m}"] = comb(f"{ds}_{m}_wgarrido")
    for m in models[:4]:
        per = {}
        for prop in ("continuity", "gravity", "solidity"):
            dirs = [d for d in sorted(base.glob(f"inflevel_{prop}_{m}_wgarrido_w*")) if (d / "per_window.json").exists()]
            if not dirs:
                per = None; break
            per[prop] = {}
            for d in dirs:
                per[prop].update(gr.rescore(d)["combos"])
        if not per:
            out[f"inflevel/{m}"] = None; continue
        best = None
        for combo in sorted(set.intersection(*(set(c) for c in per.values()))):
            det = {}
            for prop, combos in per.items():
                R = combos[combo]["reductions"]; g = next(iter(R["filtered"]["per_group"]))
                opts = {r: R[r]["per_group"][g]["acc"] for r in gr._cands(R, "inflevel")}; det[prop] = round(max(opts.values()), 1)
            mac = round(sum(det.values()) / 3, 1)
            if best is None or mac > best["macro"]:
                best = {"combo": combo, "macro": mac, **det}
        out[f"inflevel/{m}"] = best
    return out


def main():
    res = {}
    for f in (probe, occluded, late_id, tempgen, readout, velocity, late_acc, bench):
        try:
            res[f.__name__] = f()
        except Exception as e:                              # 산출물이 하나 없어도 나머지는 낸다
            res[f.__name__] = {"error": repr(e)}
        print(f"[{f.__name__}] done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
    print("→", OUT)


if __name__ == "__main__":
    main()
