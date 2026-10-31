#!/usr/bin/env python3
"""transfer_gap R4 [X2] — 진단 PFT 전이 표 (사전 등록 판정 포함). 2026-10-30.

왜: 한 축씩 바꾼 PFT (T1 끝 자리 흩뜨리기 · T2 생성기 섞기) 가 학습 생성기 밖 late 영속 지표를 T0 (같은 레시피 두 seed) 보다 올리나.
정의 (CPU, transfer_gap_te.py 산출물만 읽는다)
  s_k(clip) = 슬롯 평균 l1 = mean_j mean|p_k[j] − LN(target_encoder(32 장))[문맥 뒤 j]| (표준 surprise) · s_copy = 슬롯 평균 copy
  쌍 = (block_id, pair_id) 안 (가능 1, 불가능 1). 정답 = s(불가능) > s(가능) (같으면 0.5). 정확도 = 평균 × 100.
  칸 (판마다): VOBench late — permanence × 이동 (motion ≠ flat_static) × 방향 (A = 물체 있던 문맥 → 빈 미래가 불가능 = 사라짐, B = 나타남)
                · permanence 정지 · shape · color · late 전체
             IntPhys 1 late uni — 이동 · 가림 60 쌍 (주) · 짝 번호 1 / 2 · 성질별
             IntPhys 2 late uni — permanence 120 쌍 (주) · 전체
             v11 · v11_realistic — 이동 가림 (moving_occlusion · moving_occlusion_flat) 사라짐 A / B (주 A) · 정지 가림 A / B
  CI = 장면 (block_id) bootstrap 2,000 (같은 표본으로 모든 차를 같이 재표집), 2.5–97.5 분위.
  판정 (BOARD "[X2] R4 — 사전 등록" 그대로): X ∈ {t1, t2}, 기준 T0 = (t0s0 + t0s1)/2, 잡음 폭 N = |t0s0 − t0s1|.
    이득 = (X − T0) > N 이고 CI 하한 (X − t0s0) > 0 · (X − t0s1) > 0; 단 A/B 가 있는 판에서 B 가 10 pt 넘게 떨어지면 '편향 이동'.
    손해 = 부호를 뒤집은 것. 그 밖 = 차이 없음. 분포 안 유지 = VOBench 사라짐 A 에서 X ≥ T0 − max(N, 10).
    학습 생성기 밖 이득 = 밖 판 (IntPhys 1 · 2 는 모두, v11 · v11_realistic 은 T1 만) 에서 이득이고 (X − 복사) CI 하한 > 0.
  관문 (gate): 같은 원본 화소 판은 기존 vll5 하네스 per_video_surprise 와 clip 별 차 · 쌍 판정 일치율 (하드웨어 차만 남아야 함).
입력: z_analysis/transfer_gap/results/r4_x2/te/<판>_<묶음>/ {meta.json, l1.npy, copy.npy, done.npy}
출력: z_analysis/transfer_gap/results/r4_x2/R4_transfer.json

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_r4_report.py --groups all
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_r4_report.py --garrido   (점검: IntPhys 1 Garrido skip2_w32)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TE = REPO / "z_analysis/transfer_gap/results/r4_x2/te"
OUT = REPO / "z_analysis/transfer_gap/results/r4_x2/R4_transfer.json"
NB = 2000
PANELS = ("vob_late", "ip1uni", "ip2uni", "v11late", "v11real")
PRIMARY = {"vob_late": ("perm_move_A", "perm_move_B"), "ip1uni": ("move_occ", None), "ip2uni": ("permanence", None),
           "v11late": ("move_A", "move_B"), "v11real": ("move_A", "move_B")}
OUTSIDE = {"t1": ("ip1uni", "ip2uni", "v11late", "v11real"), "t2": ("ip1uni", "ip2uni")}   # T2: v11 = 같은 생성기, v11_realistic = 반쯤 (따로 표시)
GATES = {  # 판 → {태그: 기존 vll5 하네스 per_block.json}
    "ip1uni": {"release": "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith/per_block.json",
               "vobftg": "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_release_vobftg/per_block.json"},
    "ip2uni": {"release": "cvpr/results/eval/intphys2_late/intphys2_late_uni_vith/per_block.json",
               "vobftg": "cvpr/results/eval/intphys2_late/intphys2_late_uni_vith_release_vobftg/per_block.json"},
    "v11real": {"release": "z_research/previous/v11_realistic/exp_results/surprise_c16t32__v11_realistic_vith/per_block.json"},
    "vob_late": {"vobftg": "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json"},
}


def load_panel(pn: str, tag: str):
    d = TE / f"{pn}_{tag}"
    if not (d / "done.npy").exists() or not np.load(d / "done.npy").all():
        return None
    meta = json.load(open(d / "meta.json"))
    l1 = np.load(d / "l1.npy").mean(-1)              # (n, K)
    cp = np.load(d / "copy.npy").mean(-1)            # (n,)
    idx = {r["video_id"]: r for r in csv.DictReader(open(meta["index_csv"], newline="", encoding="utf-8"))}
    rows = [idx[v] for v in meta["video_id"]]
    S = {t: l1[:, k] for k, t in enumerate(meta["pred_tags"])}
    S["copy"] = cp
    return {"rows": rows, "S": S, "tags": meta["pred_tags"], "vid": meta["video_id"], "sets": meta.get("sets")}


def merge(parts):
    """같은 판의 여러 묶음 (base · tg4) 을 video_id 로 합친다 — 같은 노드 · 같은 프레임이어야 한다."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    base = parts[0]
    for p in parts[1:]:
        if p["vid"] != base["vid"]:
            raise SystemExit("video_id 순서가 다르다")
        for t in p["tags"]:
            key = t if t not in base["S"] else f"{t}@{len(base['S'])}"
            base["S"][key] = p["S"][t]
        base["copy_check"] = float(np.max(np.abs(p["S"]["copy"] - base["S"]["copy"])))
    return base


def pairs(P, cell_fn):
    by = defaultdict(dict)
    for i, r in enumerate(P["rows"]):
        if not cell_fn(r):
            continue
        by[(r["block_id"], r["pair_id"])][r["plausible"]] = i
    out = [(k[0], v["1"], v["0"]) for k, v in by.items() if "1" in v and "0" in v]
    return out


def cell_defs(pn):
    if pn == "vob_late":
        mv = lambda r: r["motion"] != "flat_static"  # noqa: E731
        return {"perm_move_A": lambda r: r["violation"] == "permanence" and mv(r) and r["pair_id"] == "A",
                "perm_move_B": lambda r: r["violation"] == "permanence" and mv(r) and r["pair_id"] == "B",
                "perm_static_A": lambda r: r["violation"] == "permanence" and not mv(r) and r["pair_id"] == "A",
                "perm_static_B": lambda r: r["violation"] == "permanence" and not mv(r) and r["pair_id"] == "B",
                "shape": lambda r: r["violation"] == "shape", "color": lambda r: r["violation"] == "color",
                "late_all": lambda r: True}
    if pn == "ip1uni":
        mo = lambda r: r["group"] == "moving/occluded"  # noqa: E731
        return {"move_occ": mo, "move_occ_pair1": lambda r: mo(r) and r["pair_id"] == "1",
                "move_occ_pair2": lambda r: mo(r) and r["pair_id"] == "2",
                **{f"move_occ_{p}": (lambda r, p=p: mo(r) and r["principle"] == p) for p in ("O1", "O2", "O3")},
                "all": lambda r: True}
    if pn == "ip2uni":
        return {"permanence": lambda r: r["condition"] == "permanence", "all": lambda r: True}
    mv = lambda r: r["condition"] in ("moving_occlusion", "moving_occlusion_flat")  # noqa: E731
    return {"move_A": lambda r: mv(r) and r["pair_id"] == "A", "move_B": lambda r: mv(r) and r["pair_id"] == "B",
            "static_A": lambda r: r["condition"] == "static_occlusion" and r["pair_id"] == "A",
            "static_B": lambda r: r["condition"] == "static_occlusion" and r["pair_id"] == "B"}


def acc_vec(P, pr, tag):
    s = P["S"][tag]
    pos, imp = np.array([p[1] for p in pr]), np.array([p[2] for p in pr])
    return np.where(s[imp] > s[pos], 1.0, np.where(s[imp] == s[pos], 0.5, 0.0))


def boot(blocks, vals: dict, fns: dict, seed=0):
    """blocks (n_pair,) · vals {tag: (n_pair,)} · fns {name: f(dict of means) -> float} → {name: [lo, hi]}."""
    ub = sorted(set(blocks))
    bi = {b: i for i, b in enumerate(ub)}
    g = np.array([bi[b] for b in blocks])
    sums = {t: np.bincount(g, v, len(ub)) for t, v in vals.items()}
    cnt = np.bincount(g, minlength=len(ub)).astype(float)
    rng = np.random.default_rng(seed)
    res = {k: [] for k in fns}
    for _ in range(NB):
        w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
        n = (w * cnt).sum()
        m = {t: 100 * (w * s).sum() / n for t, s in sums.items()}
        for k, f in fns.items():
            res[k].append(f(m))
    return {k: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)] for k, v in res.items()}


def gate(pn, P):
    out = {}
    for tag, f in GATES.get(pn, {}).items():
        f = REPO / f
        if not f.exists() or tag not in P["S"]:
            continue
        ref = json.load(open(f))["per_video_surprise"]
        s = P["S"][tag]
        ix = [i for i, v in enumerate(P["vid"]) if v in ref]
        if not ix:
            continue
        r = np.array([ref[P["vid"][i]] for i in ix])
        d = np.abs(s[ix] - r)
        # 쌍 판정 일치
        sub = {P["vid"][i]: s[i] for i in ix}
        agree, n = 0, 0
        for c in (lambda r_: True,):
            for b, ip, ii in pairs(P, c):
                vp, vi = P["vid"][ip], P["vid"][ii]
                if vp in ref and vi in ref:
                    n += 1
                    agree += int((sub[vi] > sub[vp]) == (ref[vi] > ref[vp]))
        out[tag] = {"ref": str(f.relative_to(REPO)), "n_clip": len(ix), "max_abs_diff": round(float(d.max()), 6),
                    "mean_abs_diff": round(float(d.mean()), 6), "pair_decision_agree": round(agree / max(n, 1), 4), "n_pair": n}
    return out


def verdicts(res):
    out = {}
    for X in ("t1", "t2"):
        o = {}
        for pn in PANELS:
            c = res.get(pn, {}).get("cells", {})
            pa, pb = PRIMARY[pn]
            if pa not in c or X not in c[pa]["acc"] or "t0s0" not in c[pa]["acc"] or "t0s1" not in c[pa]["acc"]:
                continue
            A = c[pa]
            x, a0, a1 = A["acc"][X], A["acc"]["t0s0"], A["acc"]["t0s1"]
            t0, N = (a0 + a1) / 2, abs(a0 - a1)
            lo0, lo1 = A["ci"][f"{X}-t0s0"][0], A["ci"][f"{X}-t0s1"][0]
            hi0, hi1 = A["ci"][f"{X}-t0s0"][1], A["ci"][f"{X}-t0s1"][1]
            v = "차이 없음 (검출 못 함)"
            if x - t0 > N and lo0 > 0 and lo1 > 0:
                v = "이득"
            elif t0 - x > N and hi0 < 0 and hi1 < 0:
                v = "손해"
            dB = None
            if pb and pb in c:
                B = c[pb]
                dB = B["acc"][X] - (B["acc"]["t0s0"] + B["acc"]["t0s1"]) / 2
                if v == "이득" and dB < -10:
                    v = "편향 이동 (B 가 10 pt 넘게 하락)"
            vs_copy = A["ci"].get(f"{X}-copy")
            o[pn] = {"cell": pa, "X": round(x, 2), "T0_mean": round(t0, 2), "N_seed": round(N, 2), "X_minus_T0": round(x - t0, 2),
                     "ci_X_minus_t0s0": A["ci"][f"{X}-t0s0"], "ci_X_minus_t0s1": A["ci"][f"{X}-t0s1"],
                     "B_change_vs_T0": round(dB, 2) if dB is not None else None, "X_minus_copy_ci": vs_copy, "verdict": v,
                     "outside_generator": pn in OUTSIDE[X] if pn != "vob_late" else False}
            if pn == "vob_late":
                o[pn]["in_dist_maintained"] = bool(x >= t0 - max(N, 10))
        outside_gain = [pn for pn, r in o.items() if r["outside_generator"] and r["verdict"] == "이득"
                        and r["X_minus_copy_ci"] and r["X_minus_copy_ci"][0] > 0]
        o["_outside_generator_gain_panels"] = outside_gain
        out[X] = o
    return out


def garrido():
    """점검 (결론 축 아님): IntPhys 1 dev Garrido skip2_w32 칸 (avg surprise, matched 180 쌍) — 모델별 정확도 · 성질별 ·
    T0 s0 대비 차의 장면 (사분 묶음) bootstrap. 입력 results/r4_x2/garrido/ip1dev_skip2w32_<모델>/per_block.json → garrido_check.json"""
    G = REPO / "z_analysis/transfer_gap/results/r4_x2/garrido"
    idx = list(csv.DictReader(open(REPO / "z_pixel_decoder/runs/_frames/tg4_IntPhys1_dev_videos/index.csv", newline="")))
    models = {"release": "release", "t0s0": "tg4_t0_s0", "t1": "tg4_t1_endscatter", "t2": "tg4_t2_genmix"}
    S = {}
    for t, m in models.items():
        f = G / f"ip1dev_skip2w32_{m}/per_block.json"
        if f.exists():
            S[t] = {v: x["skip2_w32/avg"] for v, x in json.load(open(f))["per_video_surprise"].items()}
    by = defaultdict(dict)
    for r in idx:
        by[(r["block_id"], r["pair_id"])][r["plausible"]] = r
    prs = [(k[0], v["1"]["video_id"], v["0"]["video_id"], v["1"]["block_type"]) for k, v in by.items() if "1" in v and "0" in v]
    out = {"n_pair": len(prs), "acc": {}, "by_type": {}, "diff_ci": {}}
    vals = {t: np.array([1.0 if s[i] > s[p] else 0.5 if s[i] == s[p] else 0.0 for _, p, i, _ in prs]) for t, s in S.items()}
    for t, v in vals.items():
        out["acc"][t] = round(100 * float(v.mean()), 2)
        out["by_type"][t] = {bt: round(100 * float(v[[j for j, x in enumerate(prs) if x[3] == bt]].mean()), 2) for bt in ("O1", "O2", "O3")}
    blocks = [x[0] for x in prs]
    fns = {f"{a}-{b}": (lambda m, a=a, b=b: m[a] - m[b]) for a in ("t1", "t2", "release") for b in ("t0s0",) if a in vals and b in vals}
    fns.update({f"{a}-release": (lambda m, a=a: m[a] - m["release"]) for a in ("t0s0", "t1", "t2") if a in vals and "release" in vals})
    out["diff_ci"] = boot(blocks, vals, fns)
    json.dump(out, open(REPO / "z_analysis/transfer_gap/results/r4_x2/garrido_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="all,base", help="te 묶음 접미사 (앞이 우선)")
    ap.add_argument("--garrido", action="store_true", help="IntPhys 1 Garrido 점검 표만 (garrido_check.json)")
    a = ap.parse_args()
    if a.garrido:
        return garrido()
    res = {"definition": __doc__.split("정의 (CPU")[1].split("입력:")[0].strip()}
    for pn in PANELS:
        parts = [load_panel(pn, g) for g in a.groups.split(",")]
        P = merge(parts[:1]) if parts[0] else merge(parts[1:])
        if P is None:
            continue
        tags = [t for t in P["S"] if t != "copy" and "@" not in t] + ["copy"]
        o = {"models": tags, "n_clip": len(P["vid"]), "gate": gate(pn, P), "cells": {}}
        for cn, fn in cell_defs(pn).items():
            pr = pairs(P, fn)
            if not pr:
                continue
            blocks = [p[0] for p in pr]
            vals = {t: acc_vec(P, pr, t) for t in tags}
            accs = {t: round(float(100 * v.mean()), 2) for t, v in vals.items()}
            fns = {}
            for t in tags:
                if t != "copy":
                    fns[f"{t}-copy"] = (lambda m, t=t: m[t] - m["copy"])
                    if t != "release" and "release" in tags:
                        fns[f"{t}-release"] = (lambda m, t=t: m[t] - m["release"])
            for X in ("t1", "t2", "t0s1", "vobftg"):
                for ref in ("t0s0", "t0s1"):
                    if X in tags and ref in tags and X != ref:
                        fns[f"{X}-{ref}"] = (lambda m, X=X, ref=ref: m[X] - m[ref])
            for X in ("t1", "t2"):
                if X in tags and "t0s0" in tags and "t0s1" in tags:
                    fns[f"{X}-T0mean"] = (lambda m, X=X: m[X] - (m["t0s0"] + m["t0s1"]) / 2)
            acc_ci = boot(blocks, vals, {t: (lambda m, t=t: m[t]) for t in tags}, seed=1)
            o["cells"][cn] = {"n_pair": len(pr), "n_block": len(set(blocks)), "acc": accs, "acc_ci": acc_ci,
                              "ci": boot(blocks, vals, fns) if fns else {}}
        # 보조 지표 ε (BOARD "사전 등록 추가" 16:35): 가능 영상 평균 surprise 와 같은 clip 비, 장면 bootstrap
        pos = [i for i, r in enumerate(P["rows"]) if r["plausible"] == "1"]
        bl = [P["rows"][i]["block_id"] for i in pos]
        ub = sorted(set(bl)); bi = {b: j for j, b in enumerate(ub)}; g = np.array([bi[b] for b in bl])
        sums = {t: np.bincount(g, P["S"][t][pos], len(ub)) for t in tags}
        cnt = np.bincount(g, minlength=len(ub)).astype(float)
        means = {t: float(P["S"][t][pos].mean()) for t in tags}
        ratios = [(a_, b_) for a_ in ("t1", "t2", "t0s1", "vobftg", "t0s0") for b_ in ("t0s0", "t0s1", "release")
                  if a_ in tags and b_ in tags and a_ != b_]
        rng = np.random.default_rng(2)
        bs = {f"{a_}/{b_}": [] for a_, b_ in ratios}
        for _ in range(NB):
            w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
            for a_, b_ in ratios:
                bs[f"{a_}/{b_}"].append((w * sums[a_]).sum() / (w * sums[b_]).sum())
        o["eps"] = {"n_pos_clip": len(pos), "mean_s_pos": {t: round(v, 5) for t, v in means.items()},
                    "ratio": {k: {"v": round(means[k.split('/')[0]] / means[k.split('/')[1]], 4),
                                  "ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
                              for k, v in bs.items()}}
        for X in ("t1", "t2"):
            if f"{X}/t0s0" in o["eps"]["ratio"] and f"{X}/t0s1" in o["eps"]["ratio"]:
                c0, c1 = o["eps"]["ratio"][f"{X}/t0s0"]["ci"], o["eps"]["ratio"][f"{X}/t0s1"]["ci"]
                o["eps"][f"verdict_{X}"] = ("예측 오차를 T0 보다 줄였다" if c0[1] < 1 and c1[1] < 1 else
                                            "예측 오차를 T0 보다 늘렸다" if c0[0] > 1 and c1[0] > 1 else "차이 없음")
        res[pn] = o
        print(pn, {cn: c["acc"] for cn, c in o["cells"].items() if cn in PRIMARY[pn]}, o["gate"], flush=True)
    res["verdicts"] = verdicts(res)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["verdicts"], ensure_ascii=False, indent=1)[:4000])
    print("→", OUT)


if __name__ == "__main__":
    main()
