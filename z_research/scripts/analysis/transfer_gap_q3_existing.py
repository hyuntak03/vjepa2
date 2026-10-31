#!/usr/bin/env python3
"""transfer_gap Q3 · E9 — 이미 있는 산출물에서 PFT 비전이를 다시 센다 (CPU, 새 학습 · GPU 없음). 2026-10-30, 실험 B [X2].

(a) 전이 행렬   행 = predictor {릴리즈, PFT v11 e10, PFT IntPhys 1 e40, PFT IntPhys 2 쌍, PFT VOBench+gravity (vobftg), PFT 처음 보는 물체}
                열 = {VOBench split v2 test (late · 전체), v11 split test, IntPhys 1 (Garrido 규칙), GRASP, InfLevel continuity, IntPhys 2 Main}
                값은 per_video / per_window 에서 다시 센다. 없는 칸은 'missing' 목록.
(b) ε           가능 영상 평균 surprise 의 비 = mean_가능(PFT) / mean_가능(릴리즈), 같은 창 · 같은 C · 같은 영상.
                IntPhys 1 = combo 마다 Filtered (C 최솟값) AvgSurprise / GRASP · InfLevel = combo × C 마다 / IntPhys 2 = C 마다 / VOBench · v11 = 표준 surprise.
(c) IntPhys 1 뒤집힘  릴리즈 → vobftg 에서 맞던 쌍이 틀린 / 틀리던 쌍이 맞은 수를 성질 (O1–O3) × 운동 × 가림 칸으로 (각자 선택된 combo, 그리고 같은 combo).
(d) 갱신 기하   predictor 행렬마다 ‖ΔW‖/‖W‖ (ΔW = PFT − 릴리즈) 와 PFT 끼리 ΔW 의 cos (층별 · 전체).
판정 (사전 등록, PLAN_R1 §5 E9)
  H3-F: ε ≥ 1.05 가 분포 밖 네 곳 (IntPhys 1 · GRASP · InfLevel continuity · IntPhys 2) 중 셋 이상, 분포 안 (VOBench test) ε < 0.95.
  갱신: 다른 데이터 cos ≤ 0.2 × 같은 데이터 cos → '데이터별 갱신'.

출력  z_analysis/transfer_gap/results/q3/E9_existing.json  (경로 · 수치 · 규칙 전부)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python;  $P z_research/scripts/analysis/transfer_gap_q3_existing.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/q3/E9_existing.json"
CI1 = REPO / "cvpr/results/eval/intphys1"
BX = REPO / "z_research/previous/Benchmarks/exp_results"
TB = REPO / "cvpr/results/eval/testbed"
IP2 = REPO / "cvpr/results/eval/intphys2"
ZE = REPO / "z_exp/intphys2"
ZR = REPO / "z_training/runs"
VOBX = REPO / "z_research/VOBench_v1/exp_results"
V11X = REPO / "z_research/previous/IntPhysGenV11/exp_results"

spec = importlib.util.spec_from_file_location("gr", REPO / "z_research/scripts/analysis/garrido_rescore.py")
gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)

MODELS = ["vith", "vith_pft_v11_e10", "vith_pft_intphys1_e40", "vith_pft_ip2pair", "vith_release_vobftg", "objholdout"]
TRAIN_DOMAIN = {"vith": None, "vith_pft_v11_e10": "v11", "vith_pft_intphys1_e40": "intphys1", "vith_pft_ip2pair": "intphys2",
                "vith_release_vobftg": "vobench", "objholdout": "vobench"}

# ---- 출처 (모델 → 열 → 폴더/파일). 없는 칸은 None
SRC = {
    "vobench": {"vith": VOBX / "surprise_c16t32__vobench_v1_vith/per_block.json",
                "vith_pft_intphys1_e40": TB / "vobench_v1_split2_test_vith_pft_intphys1_e40/per_block.json",
                "vith_pft_ip2pair": TB / "vobench_v1_split2_test_vith_pft_ip2pair/per_block.json",
                "vith_release_vobftg": ZR / "vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json"},
    "v11": {"vith": [V11X / "surprise_c16t32__v11_vith/per_block.json"],
            "vith_pft_v11_e10": [ZR / "v11_postft/eval/surprise_c16t32__v11_split_test_e10/per_block.json"],
            "vith_pft_intphys1_e40": [ZR / "intphys1_postft/eval/surprise_c16t32__v11_split_test_e40/per_block.json"]},
    "intphys1": {"vith": [CI1 / "intphys1_dev_vith_wgarrido_w16", CI1 / "intphys1_dev_vith_wgarrido_w32"],
                 "vith_pft_v11_e10": [BX / "intphys1_sliding__intphys1_dev_vith_pft_v11_e10_w16", BX / "intphys1_sliding__intphys1_dev_vith_pft_v11_e10_w32"],
                 "vith_pft_intphys1_e40": [CI1 / "intphys1_dev_vith_pft_intphys1_e40_wgarrido_w16", CI1 / "intphys1_dev_vith_pft_intphys1_e40_wgarrido_w32"],
                 "vith_pft_ip2pair": [CI1 / "intphys1_dev_vith_pft_ip2pair_wgarrido_w16", CI1 / "intphys1_dev_vith_pft_ip2pair_wgarrido_w32"],
                 "vith_release_vobftg": [CI1 / "intphys1_dev_vith_release_vobftg_wgarrido_w16", CI1 / "intphys1_dev_vith_release_vobftg_wgarrido_w32"],
                 "objholdout": [REPO / "z_analysis/transfer_gap/results/q3/runs/E9_intphys1_objholdout_w16",
                                REPO / "z_analysis/transfer_gap/results/q3/runs/E9_intphys1_objholdout_w32"]},
    "grasp": {"vith": [CI1 / "grasp_level2_vith_wgarrido_w16", CI1 / "grasp_level2_vith_wgarrido_w32"],
              "vith_pft_intphys1_e40": [CI1 / "grasp_level2_vith_pft_intphys1_e40_wgarrido_w16", CI1 / "grasp_level2_vith_pft_intphys1_e40_wgarrido_w32"],
              "vith_release_vobftg": [CI1 / "grasp_level2_vith_release_vobftg_wgarrido_w16", CI1 / "grasp_level2_vith_release_vobftg_wgarrido_w32"]},
    "inflevel_continuity": {"vith": [BX / "intphys1_sliding__inflevel_continuity_vith_w16", BX / "intphys1_sliding__inflevel_continuity_vith_w32"],
                            "vith_pft_intphys1_e40": [CI1 / "inflevel_continuity_vith_pft_intphys1_e40_wgarrido_w16", CI1 / "inflevel_continuity_vith_pft_intphys1_e40_wgarrido_w32"],
                            "vith_pft_ip2pair": [CI1 / "inflevel_continuity_vith_pft_ip2pair_wgarrido_w16", CI1 / "inflevel_continuity_vith_pft_ip2pair_wgarrido_w32"],
                            "vith_release_vobftg": [CI1 / "inflevel_continuity_vith_release_vobftg_wgarrido_w16", CI1 / "inflevel_continuity_vith_release_vobftg_wgarrido_w32"]},
    "intphys2": {"vith": ZE / "vjepa2_vith_intphys2_main_repro",
                 "vith_pft_intphys1_e40": IP2 / "intphys2_main_vith_pft_intphys1_e40",
                 "vith_release_vobftg": IP2 / "intphys2_main_vith_release_vobftg",
                 "vith_pft_v11_e10@MainTest": ZE / "vjepa2_vith_intphys2_maintest_v11_postft_e10",
                 "vith_pft_ip2pair@MainPair": ZE / "vjepa2_vith_intphys2_pair_test_postft_pair_vll5"},
}


def rd(p):
    return list(csv.DictReader(open(p)))


def pv(p):
    return json.load(open(p))["per_video_surprise"]


def matched(rows, s):
    g = defaultdict(dict)
    for r in rows:
        g[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["video_id"]
    h = [1.0 if s[v["pos"]] < s[v["imp"]] else (0.5 if s[v["pos"]] == s[v["imp"]] else 0.0)
         for v in g.values() if "pos" in v and "imp" in v and v["pos"] in s and v["imp"] in s]
    return (round(100 * float(np.mean(h)), 2), len(h)) if h else (None, 0)


# ----------------------------------------------------------------------------- (a) + (b) VOBench · v11
def vob():
    rows = rd(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")
    late = [r for r in rows if r["occlusion"] == "late"]
    out, S = {}, {}
    for m, p in SRC["vobench"].items():
        S[m] = pv(p)
        out[m] = {"all": matched(rows, S[m]), "late": matched(late, S[m]), "src": str(p.relative_to(REPO))}
    pos = [r["video_id"] for r in rows if r["plausible"] == "1"]
    eps = {m: round(float(np.mean([S[m][v] for v in pos]) / np.mean([S["vith"][v] for v in pos])), 4) for m in S if m != "vith"}
    return out, eps


def v11():
    rows = rd(REPO / "data_csv/intphysgen_v11_split/index_test.csv")
    out, S = {}, {}
    for m, ps in SRC["v11"].items():
        s = {}
        for p in ps:
            if p.exists():
                s.update(pv(p))
        S[m] = s
        a, n = matched(rows, s)
        out[m] = {"all": (a, n), "src": [str(p.relative_to(REPO)) for p in ps], "n_video_found": sum(r["video_id"] in s for r in rows)}
    # 릴리즈의 기존 행렬 값 (early/mid 조건은 다른 실행에 있다)
    f = ZR / "release_vith/eval/v11_split_test_from_existing_runs.json"
    if f.exists():
        out["vith"]["from_existing_runs_json"] = json.load(open(f)).get("overall", json.load(open(f)))
    common = set(S["vith"])
    pos = [r["video_id"] for r in rows if r["plausible"] == "1" and r["video_id"] in common]
    eps = {}
    for m in S:
        if m == "vith":
            continue
        pp = [v for v in pos if v in S[m]]
        if pp:
            eps[m] = {"eps": round(float(np.mean([S[m][v] for v in pp]) / np.mean([S["vith"][v] for v in pp])), 4), "n_pos": len(pp)}
    return out, eps


# ----------------------------------------------------------------------------- (a) + (b) Garrido 벤치
def garrido(bench):
    out, raw = {}, {}
    for m, dirs in SRC[bench].items():
        dirs = [d for d in dirs if (d / "per_window.json").exists()]
        if not dirs:
            out[m] = None
            continue
        combos, kind, meta = {}, None, None
        pw = {}
        for d in dirs:
            r = gr.rescore(d); kind = r["dataset_kind"]; combos.update(r["combos"])
            _, meta = gr.load_index(d)
            w = json.load(open(d / "per_window.json"))["windows"]
            for vid, rec in w.items():
                pw.setdefault(vid, []).extend(rec)
        if bench == "inflevel_continuity":
            best = None
            for combo, e in combos.items():
                R = e["reductions"]
                g = next(iter(R["filtered"]["per_group"]))
                opts = {r: R[r]["per_group"][g]["acc"] for r in gr._cands(R, "inflevel")}
                br = max(opts, key=opts.get)
                if best is None or opts[br] > best["acc"]:
                    best = {"combo": combo, "reduction": br, "acc": round(opts[br], 2), "n_pair": R[br]["per_group"][g]["n"]}
            out[m] = best
        else:
            b = gr.select(combos, kind)
            out[m] = {"combo": b["combo"], "acc": round(b["accuracy"], 2), "n_pair": b["n_pair"]}
        out[m]["src"] = [str(d.relative_to(REPO)) for d in dirs]
        raw[m] = (pw, meta, kind)
    # ε: 같은 combo · 같은 C (IntPhys 1 은 Filtered) 에서 가능 영상 평균
    eps = {}
    if "vith" in raw:
        pw0, meta0, kind = raw["vith"]
        pos = [v for v, r in meta0.items() if r["plausible"] == "1"]

        def table(pw):
            t = defaultdict(lambda: defaultdict(list))                 # (combo, C) → vid → [s over starts]
            for vid, rec in pw.items():
                for cb, st, C, s in rec:
                    t[(cb, int(C))][vid].append(float(s))
            return t
        t0 = table(pw0)
        for m, (pw, meta, _) in raw.items():
            if m == "vith":
                continue
            t1 = table(pw)
            per = {}
            for key in sorted(set(t0) & set(t1)):
                vids = [v for v in pos if v in t0[key] and v in t1[key]]
                if len(vids) < 5:
                    continue
                a = np.mean([np.mean(t1[key][v]) for v in vids]); b = np.mean([np.mean(t0[key][v]) for v in vids])
                per[f"{key[0]}|C{key[1]}"] = round(float(a / b), 4)
            # IntPhys 1: Filtered (시작점마다 C 최솟값) — 공식 축약
            filt = {}
            if kind == "intphys":
                for combo in sorted({k[0] for k in t0} & {k[0] for k in t1}):
                    def fl(pw, combo):
                        d = defaultdict(dict)
                        for vid, rec in pw.items():
                            for cb, st, C, s in rec:
                                if cb == combo:
                                    d[vid][int(st)] = min(d[vid].get(int(st), math.inf), float(s))
                        return {v: np.mean(list(x.values())) for v, x in d.items()}
                    f0, f1 = fl(pw0, combo), fl(pw, combo)
                    vids = [v for v in pos if v in f0 and v in f1]
                    filt[combo] = round(float(np.mean([f1[v] for v in vids]) / np.mean([f0[v] for v in vids])), 4)
            vals = list(per.values())
            eps[m] = {"median_over_combo_C": round(float(np.median(vals)), 4) if vals else None,
                      "min": min(vals) if vals else None, "max": max(vals) if vals else None, "per_combo_C": per,
                      "filtered_per_combo": filt or None}
    return out, eps


# ----------------------------------------------------------------------------- (a) + (b) IntPhys 2
def ip2():
    out, raw = {}, {}
    for m, d in SRC["intphys2"].items():
        f = d / "per_video.csv"
        if not f.exists():
            out[m] = None; continue
        rows = rd(f)
        byC = defaultdict(dict)
        for r in rows:
            byC[int(r["context_length"])][(r["scene_index"], r["pair_id"], r["is_impossible"])] = float(r["surprise_avg"])
        best = None
        for C, s in byC.items():
            pr = defaultdict(dict)
            for (sc, pid, imp), v in s.items():
                pr[(sc, pid)][imp] = v
            h = [1.0 if x["False"] < x["True"] else 0.0 for x in pr.values() if "False" in x and "True" in x]
            a = 100 * float(np.mean(h))
            if best is None or a > best["acc"]:
                best = {"C": C, "acc": round(a, 2), "n_pair": len(h)}
        cfg = (d / "config.resolved.yaml").read_text() if (d / "config.resolved.yaml").exists() else ""
        best["split"] = next((ln.split(":")[1].strip() for ln in cfg.splitlines() if ln.strip().startswith("split:")), "?")
        best["src"] = str(d.relative_to(REPO))
        out[m] = best
        raw[m] = byC
    eps = {}
    if "vith" in raw:
        for m, byC in raw.items():
            if m == "vith" or "@" in m:
                continue
            per = {}
            for C in sorted(set(byC) & set(raw["vith"])):
                k = [x for x in byC[C] if x[2] == "False" and x in raw["vith"][C]]
                per[f"C{C}"] = round(float(np.mean([byC[C][x] for x in k]) / np.mean([raw["vith"][C][x] for x in k])), 4)
            eps[m] = {"median_over_C": round(float(np.median(list(per.values()))), 4), "per_C": per}
    return out, eps


# ----------------------------------------------------------------------------- (c) IntPhys 1 뒤집힘
def ip1_flips():
    vids = json.load(open(REPO / "z_research/previous/OcclusionStats/intphys1/videos.json"))
    scene_group = {v["scene"]: v["group"] for v in vids.values()}

    def per_pair(dirs, combo):
        pw = {}
        meta = None
        for d in dirs:
            _, meta = gr.load_index(d)
            for vid, rec in json.load(open(d / "per_window.json"))["windows"].items():
                pw.setdefault(vid, []).extend(rec)
        f = defaultdict(dict)
        for vid, rec in pw.items():
            for cb, st, C, s in rec:
                if cb == combo:
                    f[vid][int(st)] = min(f[vid].get(int(st), math.inf), float(s))
        sc = {v: float(np.mean(list(x.values()))) for v, x in f.items()}
        pr = defaultdict(dict)
        for v, r in meta.items():
            if v in sc:
                pr[gr.pair_key(r)]["pos" if r["plausible"] == "1" else "imp"] = (v, r)
        res = {}
        for k, x in pr.items():
            if "pos" in x and "imp" in x:
                r = x["pos"][1]
                scene = f"{r['block']}_{r['quadruplet']}"
                res[k] = {"hit": sc[x["pos"][0]] < sc[x["imp"][0]], "prop": r["block_type"], "group": scene_group.get(scene, "?")}
        return res

    rel_d, ft_d = SRC["intphys1"]["vith"], SRC["intphys1"]["vith_release_vobftg"]
    out = {}
    for name, (c0, c1) in {"selected (rel skip2_w32 / pft skip2_w16)": ("skip2_w32", "skip2_w16"),
                           "same skip2_w16": ("skip2_w16", "skip2_w16"), "same skip2_w32": ("skip2_w32", "skip2_w32")}.items():
        a, b = per_pair(rel_d, c0), per_pair(ft_d, c1)
        cell = defaultdict(lambda: {"n": 0, "rel": 0, "pft": 0, "lost": 0, "gained": 0})
        for k in a:
            if k not in b:
                continue
            for key in (f"{a[k]['prop']}|{a[k]['group']}", f"*|{a[k]['group']}", f"{a[k]['prop']}|*", "*|*"):
                c = cell[key]
                c["n"] += 1; c["rel"] += a[k]["hit"]; c["pft"] += b[k]["hit"]
                c["lost"] += a[k]["hit"] and not b[k]["hit"]; c["gained"] += (not a[k]["hit"]) and b[k]["hit"]
        out[name] = {k: {**v, "rel_acc": round(100 * v["rel"] / v["n"], 1), "pft_acc": round(100 * v["pft"] / v["n"], 1)} for k, v in sorted(cell.items())}
    return out


# ----------------------------------------------------------------------------- (d) 갱신 기하
def weights():
    import torch
    base = torch.load(ZR / "release_vith/latest.pt", map_location="cpu", weights_only=False)["predictor"]
    ck = {"vobft (VOBench)": ZR / "vobench_release_postft/latest.pt", "vobftg (VOBench+gravity)": ZR / "vobench_release_postft_g/latest.pt",
          "objholdout (VOBench, 물체 뺌)": ZR / "vobench_release_postft_objholdout/latest.pt",
          "v11_e10": ZR / "v11_postft/e10.pt", "ip1_e40": ZR / "intphys1_postft/e40.pt", "ip2pair": ZR / "intphys2_postft_pair/latest.pt"}
    D, info = {}, {}
    for n, p in ck.items():
        st = torch.load(p, map_location="cpu", weights_only=False)
        sd = st["predictor"]
        D[n] = {k: (sd[k].float() - base[k].float()) for k in base if k in sd and sd[k].shape == base[k].shape}
        info[n] = {"path": str(p.relative_to(REPO)), "epoch": st.get("epoch"), "n_keys": len(D[n]),
                   "num_frames": (st.get("arch") or {}).get("num_frames")}
    def layer_of(k):
        if k.startswith("predictor_blocks."):
            return f"block{int(k.split('.')[1]):02d}"
        return k.split(".")[0]
    rel = {}
    for n, d in D.items():
        per = defaultdict(lambda: [0.0, 0.0])
        for k, v in d.items():
            if base[k].ndim < 2:
                continue
            per[layer_of(k)][0] += float(v.pow(2).sum()); per[layer_of(k)][1] += float(base[k].float().pow(2).sum())
        tot = [sum(x[0] for x in per.values()), sum(x[1] for x in per.values())]
        rel[n] = {"total": round(math.sqrt(tot[0] / tot[1]), 5), **{l: round(math.sqrt(a / b), 5) for l, (a, b) in sorted(per.items())}}
    names = list(D)
    cos = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            per = defaultdict(lambda: [0.0, 0.0, 0.0])
            for k in D[a]:
                if k not in D[b] or base[k].ndim < 2:
                    continue
                l = layer_of(k)
                per[l][0] += float((D[a][k] * D[b][k]).sum()); per[l][1] += float(D[a][k].pow(2).sum()); per[l][2] += float(D[b][k].pow(2).sum())
            t = [sum(x[j] for x in per.values()) for j in range(3)]
            cos[f"{a} ~ {b}"] = {"total": round(t[0] / math.sqrt(t[1] * t[2]), 4),
                                 **{l: round(x[0] / math.sqrt(x[1] * x[2]), 4) for l, x in sorted(per.items()) if x[1] > 0 and x[2] > 0}}
    same = ["vobft (VOBench) ~ vobftg (VOBench+gravity)", "vobft (VOBench) ~ objholdout (VOBench, 물체 뺌)",
            "vobftg (VOBench+gravity) ~ objholdout (VOBench, 물체 뺌)"]
    other = [k for k in cos if k not in same]
    vob_vs_other = [k for k in other if any(s in k for s in ("vobft", "objholdout"))]
    sc = float(np.mean([cos[k]["total"] for k in same]))
    oc = float(np.mean([cos[k]["total"] for k in vob_vs_other]))
    return {"checkpoints": info, "rel_norm": rel, "cos": cos,
            "summary": {"same_data_mean_cos": round(sc, 4), "vobench_vs_other_data_mean_cos": round(oc, 4),
                        "other_data_pairs_mean_cos": round(float(np.mean([cos[k]["total"] for k in other])), 4),
                        "ratio_other_over_same": round(oc / sc, 3), "rule": "다른 데이터 cos ≤ 0.2 × 같은 데이터 cos → 데이터별 갱신",
                        "data_specific_update": bool(oc <= 0.2 * sc)}}


def main():
    res = {"rules": {"H3-F": "ε ≥ 1.05 가 분포 밖 네 곳 중 셋 이상 + 분포 안 (VOBench test) ε < 0.95", "update": "다른 데이터 cos ≤ 0.2 × 같은 데이터 cos"}}
    a = {}
    a["vobench"], e_vob = vob()
    a["v11"], e_v11 = v11()
    eps = {"vobench_split2_test (in-dist for vobftg)": e_vob, "v11_split_test": e_v11}
    for b in ("intphys1", "grasp", "inflevel_continuity"):
        a[b], eps[b] = garrido(b)
    a["intphys2"], eps["intphys2"] = ip2()
    res["matrix"] = a
    res["eps"] = eps
    miss = []
    for col in ("vobench", "v11", "intphys1", "grasp", "inflevel_continuity", "intphys2"):
        for m in MODELS:
            if not a[col].get(m) and not any(k.startswith(m + "@") for k in a[col]):
                miss.append(f"{m} × {col}")
    res["missing"] = miss
    # H3-F 판정 (vobftg)
    ood = {"intphys1": eps["intphys1"].get("vith_release_vobftg", {}), "grasp": eps["grasp"].get("vith_release_vobftg", {}),
           "inflevel_continuity": eps["inflevel_continuity"].get("vith_release_vobftg", {}), "intphys2": eps["intphys2"].get("vith_release_vobftg", {})}
    vals = {k: (v.get("median_over_combo_C") or v.get("median_over_C")) for k, v in ood.items()}
    ind = e_vob.get("vith_release_vobftg")
    res["H3F_vobftg"] = {"eps_ood_median": vals, "eps_in_dist": ind,
                         "n_ood_ge_1.05": sum(1 for v in vals.values() if v is not None and v >= 1.05),
                         "verdict": ("지지" if sum(1 for v in vals.values() if v is not None and v >= 1.05) >= 3 and ind is not None and ind < 0.95
                                     else "기각 (≈1)" if all(v is not None and 0.98 <= v <= 1.02 for v in vals.values()) else "판정 보류/부분")}
    res["intphys1_flips"] = ip1_flips()
    try:
        res["weights"] = weights()
    except Exception as e:                                             # noqa: BLE001
        res["weights"] = {"error": repr(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: res[k] for k in ("matrix", "missing", "H3F_vobftg")}, indent=0, ensure_ascii=False, default=str)[:6000])
    print("eps", json.dumps({k: {m: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('per_combo_C', 'per_C')}) for m, v in e.items()} for k, e in eps.items()}, ensure_ascii=False))
    print("weights", json.dumps(res["weights"].get("summary"), ensure_ascii=False))
    print("→", OUT)


if __name__ == "__main__":
    main()
