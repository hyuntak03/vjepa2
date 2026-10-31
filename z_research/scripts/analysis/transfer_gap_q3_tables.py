#!/usr/bin/env python3
"""transfer_gap Q3 — E10 · E11 · E12 · E13 집계 (CPU, 몇 초). 2026-10-30, 실험 B [X2].

정의
  쌍 = (block_id, pair_id) 의 (가능, 불가능) 하나씩 (matched). 정답 = s_pos < s_imp, 동점 0.5 (하네스 score_blocks 와 같다).
  s = 영상 surprise = 미래 슬롯 평균 L1 (하네스 per_video_surprise, 또는 te 배열 l1.mean(슬롯)).
  A 방향 = pair_id A = (pos_a, imp_ab). permanence 에서 A = 물체 → 빈 장면 (사라짐).
  이득 G = acc_PFT − acc_릴리즈 (같은 쌍). 유지율 r_T = G(T) / G(T0, 같은 block).
  CI = block 단위 bootstrap 2,000 회 (같은 재표집을 PFT · 릴리즈 · T0 에 함께 쓴다), 2.5–97.5 백분위.

입력  (부록 경로) — E10: 하네스 per_block.json (PFT) + 원 VOBench 릴리즈 per_block.json + 복사 te 배열
       E11 · E12 · E13: transfer_gap_te.py 산출 (/data2/.../cache/transfer_gap/te_*/) + 하네스 per_block.json (AR)
출력  z_analysis/transfer_gap/results/q3/E10_objholdout.json · E11_neardomain.json · E12_transforms.json · E13_epochs.json · NF_noise_floor.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_q3_tables.py E10      # E11 · E12 · E13 도 같은 꼴
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/q3"
RUNS = OUT / "runs"
CACHE = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
VOBX = REPO / "z_research/VOBench_v1/exp_results"
NB = 2000
RNG = np.random.default_rng(0)


def rd(p):
    return list(csv.DictReader(open(p)))


def pv_harness(p):
    return json.load(open(p))["per_video_surprise"]


def pv_te(d, tag=None):
    """te 폴더 → {tag: {video_id: s}} (+ 'copy')."""
    d = Path(d)
    m = json.load(open(d / "meta.json"))
    l1 = np.load(d / "l1.npy").mean(2)
    out = {t: dict(zip(m["video_id"], l1[:, k].tolist())) for k, t in enumerate(m["pred_tags"])}
    out["copy"] = dict(zip(m["video_id"], np.load(d / "copy.npy").mean(1).tolist()))
    if (d / "copy_blk.npy").exists():
        out["copy_blk"] = dict(zip(m["video_id"], np.load(d / "copy_blk.npy").mean(1).tolist()))
    return out if tag is None else out[tag]


def pairs(rows):
    """index 행 → [(block, pair_id, pos_vid, imp_vid, meta_row_of_pos)]."""
    g = defaultdict(dict)
    for r in rows:
        g[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    out = []
    for (b, pid), v in g.items():
        if "pos" in v and "imp" in v:
            out.append((b, pid, v["pos"]["video_id"], v["imp"]["video_id"], v["pos"]))
    return out


def hits(P, s):
    return np.array([1.0 if s[p] < s[i] else (0.5 if s[p] == s[i] else 0.0) for _, _, p, i, _ in P])


def boot_G(P, sa, sb, nb=NB):
    """acc(sa) − acc(sb) 의 block bootstrap. → dict."""
    if not P:
        return None
    ha, hb = hits(P, sa), hits(P, sb)
    blocks = sorted({b for b, *_ in P})
    bi = {b: i for i, b in enumerate(blocks)}
    idx = np.array([bi[b] for b, *_ in P])
    nbk = len(blocks)
    da = np.bincount(idx, ha, nbk); db = np.bincount(idx, hb, nbk); n = np.bincount(idx, None, nbk)
    W = RNG.multinomial(nbk, np.ones(nbk) / nbk, size=nb)                    # (nb, nbk) block 재표집 횟수
    A = (W @ da) / (W @ n); B = (W @ db) / (W @ n)
    g = 100 * (A - B)
    return {"n_pair": len(P), "n_block": nbk, "acc_a": round(100 * ha.mean(), 2), "acc_b": round(100 * hb.mean(), 2),
            "G": round(100 * (ha.mean() - hb.mean()), 2), "G_ci": [round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]}


def acc(P, s):
    if not P:
        return None
    return {"n_pair": len(P), "acc": round(100 * hits(P, s).mean(), 2)}


def boot_ratio(P, sT_a, sT_b, P0, s0_a, s0_b, nb=NB):
    """r = G(T)/G(T0), 같은 block 재표집 (block id 가 두 쌍 목록에 공통이어야 한다)."""
    blocks = sorted({b for b, *_ in P} & {b for b, *_ in P0})
    P = [x for x in P if x[0] in blocks]; P0 = [x for x in P0 if x[0] in blocks]
    if not P or not P0:
        return None
    bi = {b: i for i, b in enumerate(blocks)}
    nbk = len(blocks)

    def agg(PP, sa, sb):
        idx = np.array([bi[b] for b, *_ in PP])
        return np.bincount(idx, hits(PP, sa) - hits(PP, sb), nbk), np.bincount(idx, None, nbk)
    dT, nT = agg(P, sT_a, sT_b); d0, n0 = agg(P0, s0_a, s0_b)
    W = RNG.multinomial(nbk, np.ones(nbk) / nbk, size=nb)
    gT = (W @ dT) / (W @ nT); g0 = (W @ d0) / (W @ n0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = gT / g0
    GT, G0 = dT.sum() / nT.sum(), d0.sum() / n0.sum()
    return {"n_block": nbk, "G_T": round(100 * GT, 2), "G_T0": round(100 * G0, 2), "r": round(float(GT / G0), 3) if G0 else None,
            "r_ci": [round(float(np.nanpercentile(r, 2.5)), 3), round(float(np.nanpercentile(r, 97.5)), 3)],
            "frac_G0_le0": round(float((g0 <= 0).mean()), 3)}


def sel(P, **kw):
    """meta 열 조건으로 쌍을 고른다. 값이 callable 이면 술어."""
    out = []
    for x in P:
        r = x[4]
        ok = True
        for k, v in kw.items():
            if k == "pair_id":
                ok &= x[1] == v
            elif callable(v):
                ok &= bool(v(r.get(k)))
            else:
                ok &= r.get(k) == v
        if ok:
            out.append(x)
    return out


MOVING = lambda m: m not in (None, "", "flat_static")     # noqa: E731


def vob_cells(P):
    """VOBench 쌍 목록 → 표준 칸들."""
    return {"all": P, "late": sel(P, occlusion="late"), "late_moving": sel(P, occlusion="late", motion=MOVING),
            "late_moving_perm_A": sel(P, occlusion="late", motion=MOVING, violation="permanence", pair_id="A"),
            "late_moving_perm_B": sel(P, occlusion="late", motion=MOVING, violation="permanence", pair_id="B"),
            "late_perm_A": sel(P, occlusion="late", violation="permanence", pair_id="A"),
            "visible": sel(P, occlusion="visible"), "early": sel(P, occlusion="early"), "mid": sel(P, occlusion="mid"),
            "late_shape": sel(P, occlusion="late", violation="shape"), "late_color": sel(P, occlusion="late", violation="color"),
            "late_perm": sel(P, occlusion="late", violation="permanence")}


# ----------------------------------------------------------------------------- E10
def E10():
    rows = rd(REPO / "data_csv/vobench_v1_objholdout/index_test.csv")
    P = pairs(rows)
    pft = pv_harness(RUNS / "E10_objholdout_pft/per_block.json")
    rel = pv_harness(VOBX / "surprise_c16t32__vobench_v1_vith/per_block.json")
    cp = pv_te(VOBX / "copy__vobench_v1_vith", "copy")
    miss = [v for _, _, p, i, _ in P for v in (p, i) if v not in pft or v not in rel or v not in cp]
    res = {"definition": "G = acc(처음 보는 물체 PFT) − acc(릴리즈), 같은 쌍, block bootstrap 2000",
           "n_pair": len(P), "missing_videos": len(miss),
           "gate": "test 1,688 영상 = VOBench_v1 원본과 같은 file_name·block·variant·pair (차이 0), 학습 사본 표본 프레임 md5 60/60 같음 "
                   "→ 릴리즈 per_video (원 VOBench 채점) 와 복사 (te 배열) 재사용", "cells": {}}
    cats = ["seen", "color", "shape", "mesh", "both"]
    for objset in ("all", "primitive", "realistic"):
        PO = P if objset == "all" else sel(P, objset=objset)
        for cat in ["ALL", "unseen_any"] + cats:
            PC = PO if cat == "ALL" else sel(PO, holdout=(lambda h: h != "seen")) if cat == "unseen_any" else sel(PO, holdout=cat)
            for cell, PP in vob_cells(PC).items():
                if cell not in ("all", "late", "late_moving", "late_moving_perm_A", "late_perm_A", "visible", "late_shape", "late_color", "late_perm"):
                    continue
                if not PP:
                    continue
                e = boot_G(PP, pft, rel)
                e["acc_copy"] = acc(PP, cp)["acc"]
                res["cells"][f"{objset}|{cat}|{cell}"] = e
    # 판정: 안 본 범주 이득 / 본 것 이득 · / 색만 안 본 것 이득 (전체 · late)
    ver = {}
    for cell in ("all", "late"):
        g = {c: res["cells"].get(f"all|{c}|{cell}", {}).get("G") for c in cats + ["unseen_any"]}
        ver[cell] = {"G": g,
                     "ratio_shape_mesh_both_over_seen": {c: (round(g[c] / g["seen"], 2) if g.get(c) is not None and g.get("seen") else None) for c in ("shape", "mesh", "both", "color")},
                     "ratio_shape_mesh_both_over_color": {c: (round(g[c] / g["color"], 2) if g.get(c) is not None and g.get("color") else None) for c in ("shape", "mesh", "both")}}
    res["verdict_inputs"] = ver
    json.dump(res, open(OUT / "E10_objholdout.json", "w"), indent=1, ensure_ascii=False)
    for k, v in res["cells"].items():
        if k.startswith("all|") and k.split("|")[2] in ("all", "late", "late_moving_perm_A", "visible"):
            print(f"{k:42s} n={v['n_pair']:4d}  rel {v['acc_b']:6.2f}  pft {v['acc_a']:6.2f}  G {v['G']:+6.2f} {v['G_ci']}  copy {v['acc_copy']}")
    print(json.dumps(ver, ensure_ascii=False))


# ----------------------------------------------------------------------------- E11
V11_COND = ("moving_occlusion", "moving_occlusion_flat", "static_occlusion", "moving_visible", "moving_visible_flat")


def E11():
    res = {"definition": "v11 split test vanish × 5 조건 (k 0–4) · v11_realistic. A = pos_a vs imp_ab (물체→빈). "
                         "G = acc(PFT) − acc(기준), 같은 쌍, block bootstrap", "sets": {}}
    for name, idx, te_name, ar_tag in (("v11sub", REPO / "data_csv/transfer_gap_v11sub/index.csv", "te_E11_v11sub", "v11sub"),
                                       ("v11_realistic", REPO / "data_csv/intphysgen_v11_realistic/index.csv", "te_E11_v11real", "v11real")):
        rows = rd(idx)
        P = pairs(rows)
        S = {}
        if (CACHE / te_name / "summary.json").exists():
            t = pv_te(CACHE / te_name)
            S.update({"release": t["release"], "release_pft": t["vobftg"], "copy": t["copy"]})
        for m, key in (("vith_vast_ar_state_ep70", "vastAR"), ("vith_vast_ar_state_ep70_vobftg", "vastAR_pft")):
            f = RUNS / f"E11_{ar_tag}_{m}/per_block.json"
            if f.exists():
                S[key] = pv_harness(f)
        # 기존 하네스 릴리즈 per_video 와 te 릴리즈 대조 (관문)
        gate = {}
        ref = {"v11sub": REPO / "z_research/previous/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith/per_block.json",
               "v11_realistic": REPO / "z_research/previous/v11_realistic/exp_results/surprise_c16t32__v11_realistic_vith/per_block.json"}[name]
        if "release" in S and ref.exists():
            rp = pv_harness(ref)
            common = [v for v in S["release"] if v in rp]
            d = np.array([abs(S["release"][v] - rp[v]) / rp[v] for v in common])
            Pc = [x for x in P if x[2] in rp and x[3] in rp]
            agree = float(np.mean(hits(Pc, S["release"]) == hits(Pc, rp))) if Pc else None
            gate = {"ref": str(ref.relative_to(REPO)), "n_common": len(common), "max_rel": float(d.max()) if len(d) else None,
                    "mean_rel": float(d.mean()) if len(d) else None, "pair_agree": agree}
        cells = {}
        for cond in sorted({r["condition"] for r in rows}):
            for k in ["all"] + sorted({r["sym_k"] for r in rows if r["condition"] == cond}):
                for pid in ("all", "A", "B"):
                    PP = sel(P, condition=cond)
                    if k != "all":
                        PP = sel(PP, sym_k=k)
                    if pid != "all":
                        PP = sel(PP, pair_id=pid)
                    if not PP:
                        continue
                    e = {"n_pair": len(PP)}
                    for key, s in S.items():
                        e[key] = round(100 * hits(PP, s).mean(), 2)
                    if "release" in S and "release_pft" in S:
                        e["G_release"] = boot_G(PP, S["release_pft"], S["release"])
                    if "vastAR" in S and "vastAR_pft" in S:
                        e["G_vastAR"] = boot_G(PP, S["vastAR_pft"], S["vastAR"])
                    cells[f"{cond}|k{k}|{pid}"] = e
        # 묶음: late 이동 (경사 + 평지) A, k 1–3 / k 4, 보임 칸
        grp = {}
        mov_occ = lambda c: c in ("moving_occlusion", "moving_occlusion_flat")   # noqa: E731
        for gname, PP in (("occ_moving_A_k1to3", [x for x in sel(P, condition=mov_occ, pair_id="A") if x[4]["sym_k"] in ("1", "2", "3")]),
                          ("occ_moving_A_k4", sel(P, condition=mov_occ, pair_id="A", sym_k="4")),
                          ("occ_moving_A_all", sel(P, condition=mov_occ, pair_id="A")),
                          ("occ_moving_B_all", sel(P, condition=mov_occ, pair_id="B")),
                          ("occ_static_A", sel(P, condition="static_occlusion", pair_id="A")),
                          ("visible_moving_all", sel(P, condition=lambda c: c in ("moving_visible", "moving_visible_flat"))),
                          ("visible_static_all", sel(P, condition="static_visible")),
                          ("all", P)):
            if not PP:
                continue
            e = {"n_pair": len(PP)}
            for key, s in S.items():
                e[key] = round(100 * hits(PP, s).mean(), 2)
            if "release" in S and "release_pft" in S:
                e["G_release"] = boot_G(PP, S["release_pft"], S["release"])
            if "vastAR" in S and "vastAR_pft" in S:
                e["G_vastAR"] = boot_G(PP, S["vastAR_pft"], S["vastAR"])
            grp[gname] = e
        res["sets"][name] = {"n_pair": len(P), "models": sorted(S), "gate_release_vs_harness": gate, "groups": grp, "cells": cells}
        print(name, json.dumps(grp, ensure_ascii=False)[:3000])
    json.dump(res, open(OUT / "E11_neardomain.json", "w"), indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------- E12
TRANS = {"T0": "index_T0.csv", "T1": "index_T0.csv", "T3_x0y0": "index_T3.csv", "T3_x64y0": "index_T3.csv",
         "T3_x0y64": "index_T3.csv", "T3_x64y64": "index_T3.csv", "T4": "index_T0.csv", "T2": "index_T2.csv", "T5": "index_T5.csv"}


def E12():
    """⚠️ 2026-10-30 정정: T0 (작성기) 관문이 실패했다 (반올림 ≤ 0.5/255 만으로 릴리즈 쌍 판정 10 % 가 뒤집힘).
    그래서 기준을 둘로 나눈다 — 작성기로 쓴 T3 · T4 는 T0 (작성기) 대비, 원본 프레임을 읽는 T1 · T2 · T5 는 Torig (원본, 같은 te 경로) 대비.
    T0 대 Torig 의 차이는 그 자체로 '잡음 바닥' (눈에 안 보이는 입력 차이로 바뀌는 쌍 판정) 으로 낸다."""
    idx = REPO / "data_csv/transfer_gap_vob"
    res = {"definition": "G(T) = acc_PFT(T) − acc_rel(T); r_T = G(T) / G(기준, 같은 block). 기준 = T0 (작성기, T3·T4) 또는 Torig (원본 프레임, T1·T2·T5). "
                         "T5 는 source_block 으로 split v2 block 에 맞춘다. block bootstrap 2000", "transforms": {}}
    have = {T: CACHE / f"te_E12_{T}" for T in list(TRANS) + ["Torig"] if (CACHE / f"te_E12_{T}" / "summary.json").exists()}
    if "T0" not in have:
        print("T0 없음"); return
    P0 = pairs(rd(idx / TRANS["T0"]))
    REF = {"T0": "Torig", "T1": "Torig", "T2": "Torig", "T5": "Torig", "T4": "T0",
           "T3_x0y0": "T0", "T3_x64y0": "T0", "T3_x0y64": "T0", "T3_x64y64": "T0"}
    SS = {T: pv_te(d) for T, d in have.items()}
    # 관문 · 잡음 바닥: T0 (작성기) vs 공식 per_video, T0 vs Torig, Torig vs 공식
    rel_off = pv_harness(VOBX / "surprise_c16t32__vobench_v1_vith/per_block.json")
    ft_off = pv_harness(REPO / "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json")
    gate = {}
    for name, A, B in (("T0_vs_official", "T0", None), ("Torig_vs_official", "Torig", None), ("T0_vs_Torig", "T0", "Torig")):
        if A not in SS or (B and B not in SS):
            continue
        g = {}
        for tag, off in (("release", rel_off), ("vobftg", ft_off)):
            ref = SS[B][tag] if B else off
            d = np.array([abs(SS[A][tag][v] - ref[v]) / ref[v] for v in SS[A][tag]])
            ha, hb = hits(P0, SS[A][tag]), hits(P0, ref)
            g[tag] = {"max_rel": round(float(d.max()), 5), "mean_rel": round(float(d.mean()), 6),
                      "pair_agree": round(float(np.mean(ha == hb)), 4), "acc_A": round(100 * ha.mean(), 2), "acc_B": round(100 * hb.mean(), 2),
                      "pass": bool(d.max() <= 1e-3 and np.mean(ha == hb) >= 0.99)}
            for cell, pp in vob_cells(P0).items():
                if cell in ("all", "late", "visible", "late_moving_perm_A"):
                    g[tag][f"agree_{cell}"] = round(float(np.mean(hits(pp, SS[A][tag]) == hits(pp, ref))), 4)
        if B:
            g["copy"] = {"pair_agree": round(float(np.mean(hits(P0, SS[A]["copy"]) == hits(P0, SS[B]["copy"]))), 4)}
        gate[name] = g
    res["gate_and_noise_floor"] = gate
    src2blk = {r["source_block"]: r["block_id"] for r in rd(idx / "index_T0.csv")}
    for T, d in have.items():
        if T == "Torig":
            continue
        S = SS[T]
        rT = REF[T]
        if rT not in SS:
            print(T, "기준", rT, "없음 — 건너뜀"); continue
        S0 = SS[rT]
        rows = rd(idx / TRANS[T])
        if T == "T5":
            for r in rows:
                r["block_id"] = src2blk[r["source_block"]]
        P = pairs(rows)
        cT, c0 = vob_cells(P), vob_cells(P0)
        e = {"n_pair": len(P), "reference": rT}
        for cell in ("all", "late", "late_moving", "late_moving_perm_A", "late_moving_perm_B", "late_perm_A", "visible", "early", "mid",
                     "late_shape", "late_color", "late_perm"):
            if not cT[cell]:
                continue
            x = boot_G(cT[cell], S["vobftg"], S["release"])
            x["acc_copy"] = acc(cT[cell], S["copy"])["acc"]
            blocks = {b for b, *_ in cT[cell]}
            p0 = [y for y in c0[cell] if y[0] in blocks]
            x["ref_same_blocks"] = {"rel": acc(p0, S0["release"])["acc"], "pft": acc(p0, S0["vobftg"])["acc"], "copy": acc(p0, S0["copy"])["acc"]} if p0 else None
            x["ratio"] = boot_ratio(cT[cell], S["vobftg"], S["release"], c0[cell], S0["vobftg"], S0["release"])
            e[cell] = x
        for objset in ("primitive", "realistic"):
            for cell in ("all", "late", "late_moving_perm_A"):
                pp = vob_cells(sel(P, objset=objset))[cell]
                if pp:
                    x = boot_G(pp, S["vobftg"], S["release"]); x["acc_copy"] = acc(pp, S["copy"])["acc"]
                    x["ratio"] = boot_ratio(pp, S["vobftg"], S["release"], vob_cells(sel(P0, objset=objset))[cell], S0["vobftg"], S0["release"])
                    e[f"{objset}|{cell}"] = x
        a = e["all"]
        if a.get("ref_same_blocks"):
            e["validity"] = {"rel_change": round(a["acc_b"] - a["ref_same_blocks"]["rel"], 2),
                             "copy_change": round(a["acc_copy"] - a["ref_same_blocks"]["copy"], 2),
                             "valid": bool(abs(a["acc_b"] - a["ref_same_blocks"]["rel"]) <= 10)}
        res["transforms"][T] = e
        print(T, "ref", rT, {c: (e[c]["acc_b"], e[c]["acc_a"], e[c]["G"], e[c]["G_ci"], (e[c].get("ratio") or {}).get("r"), (e[c].get("ratio") or {}).get("r_ci"))
                            for c in ("all", "late", "late_moving_perm_A") if c in e}, e.get("validity"))
    print("gate", json.dumps(gate, ensure_ascii=False))
    json.dump(res, open(OUT / "E12_transforms.json", "w"), indent=1, ensure_ascii=False, default=float)


# ----------------------------------------------------------------------------- E13 · NF (IntPhys 1 Garrido)
def _gr():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gr", REPO / "z_research/scripts/analysis/garrido_rescore.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    return gr


def ip1_eval(dirs):
    """IntPhys 1 Garrido 규칙 (Filtered → combo 최고) + combo 마다 쌍 판정 (pair key → hit)."""
    gr = _gr()
    dirs = [Path(d) for d in dirs if (Path(d) / "per_window.json").exists()]
    if not dirs:
        return None, None
    combos, pw, meta = {}, {}, None
    for d in dirs:
        r = gr.rescore(d); combos.update(r["combos"])
        _, meta = gr.load_index(d)
        for vid, rec in json.load(open(d / "per_window.json"))["windows"].items():
            pw.setdefault(vid, []).extend(rec)
    best = gr.select(combos, "intphys")
    per = {}
    for combo in combos:
        f = defaultdict(dict)
        for vid, rec in pw.items():
            for cb, st, C, sc in rec:
                if cb == combo:
                    f[vid][int(st)] = min(f[vid].get(int(st), 1e9), float(sc))
        S = {v: float(np.mean(list(x.values()))) for v, x in f.items()}
        pr = defaultdict(dict)
        for v, r in meta.items():
            if v in S:
                pr[gr.pair_key(r)]["pos" if r["plausible"] == "1" else "imp"] = v
        per[combo] = {k: (S[x["pos"]] < S[x["imp"]]) for k, x in pr.items() if "pos" in x and "imp" in x}
    return {"combo": best["combo"], "acc": round(best["accuracy"], 2),
            "per_combo": {c: round(100 * float(np.mean(list(h.values()))), 2) for c, h in per.items()}}, per


def E13():
    res = {}
    d = CACHE / "te_E13_epochs"
    if (d / "summary.json").exists():
        S = pv_te(d)
        rows = rd(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")
        P = pairs(rows)
        res["vobench_split2_test"] = {tag: {c: acc(pp, S[tag])["acc"] for c, pp in vob_cells(P).items() if pp and c in ("all", "late", "late_moving_perm_A", "visible", "early", "mid")}
                                      for tag in S if tag != "copy"}
        pos = [r["video_id"] for r in rows if r["plausible"] == "1"]
        res["vobench_eps_vs_release"] = {tag: round(float(np.mean([S[tag][v] for v in pos]) / np.mean([S["release"][v] for v in pos])), 4) for tag in S if tag not in ("copy",)}
    CI1 = REPO / "cvpr/results/eval/intphys1"
    ip = {"release(e0)": [CI1 / "intphys1_dev_vith_wgarrido_w16", CI1 / "intphys1_dev_vith_wgarrido_w32"],
          "e10": [RUNS / "E13_intphys1_vobftg_e10_w16", RUNS / "E13_intphys1_vobftg_e10_w32"],
          "e20": [RUNS / "E13_intphys1_vobftg_e20_w16", RUNS / "E13_intphys1_vobftg_e20_w32"],
          "e30(latest)": [CI1 / "intphys1_dev_vith_release_vobftg_wgarrido_w16", CI1 / "intphys1_dev_vith_release_vobftg_wgarrido_w32"],
          "objholdout(e30)": [RUNS / "E9_intphys1_objholdout_w16", RUNS / "E9_intphys1_objholdout_w32"]}
    res["intphys1"] = {k: ip1_eval(v)[0] for k, v in ip.items()}
    json.dump(res, open(OUT / "E13_epochs.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=0, ensure_ascii=False)[:3000])


def NF():
    """잡음 바닥: IntPhys 1 dev 프레임을 같은 작성기로 재양자화 (≤ 0.5/255) 한 판 vs 원본 — 같은 모델의 쌍 판정 일치 · 정확도 변화."""
    CI1 = REPO / "cvpr/results/eval/intphys1"
    res = {"definition": "원본 (288 px PNG → 로더 256 px) vs 재양자화 (256 px uint8 PNG, 화소 차 ≤ 0.5/255). combo 마다 Filtered AvgSurprise 쌍 판정"}
    for m, orig in (("vith", "intphys1_dev_vith"), ("vith_release_vobftg", "intphys1_dev_vith_release_vobftg")):
        a, pa = ip1_eval([CI1 / f"{orig}_wgarrido_w16", CI1 / f"{orig}_wgarrido_w32"])
        b, pb = ip1_eval([RUNS / f"NF_intphys1_requant_{m}_w16", RUNS / f"NF_intphys1_requant_{m}_w32"])
        if a is None or b is None:
            res[m] = None; continue
        agree = {c: round(float(np.mean([pa[c][k] == pb[c][k] for k in pa[c] if k in pb.get(c, {})])), 4) for c in pa if c in pb}
        res[m] = {"orig": a, "requant": b, "pair_agree_per_combo": agree}
    VOB = json.load(open(OUT / "E12_transforms.json")).get("gate_and_noise_floor") if (OUT / "E12_transforms.json").exists() else None
    res["vobench_T0_vs_Torig"] = (VOB or {}).get("T0_vs_Torig")
    json.dump(res, open(OUT / "NF_noise_floor.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=0, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["E10", "E11", "E12", "E13", "NF"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"E10": E10, "E11": E11, "E12": E12, "E13": E13, "NF": NF}[a.what]()
