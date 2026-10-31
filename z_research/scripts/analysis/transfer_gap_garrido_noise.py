#!/usr/bin/env python3
"""transfer_gap Q3 · E36 — 잡음 판에서 Garrido 규칙을 seed 마다 다시 적용 (승자의 저주 · 정답 조건부 뒤집힘). 2026-10-30, [X2] R3.

왜: R2 E31c 는 IntPhys 1 의 한 칸 (skip2_w32 = 릴리즈가 원본 화소로 격자에서 고른 최고 칸) 만 잡음 판으로 다시 쟀다.
    보고값 (Garrido 규칙 = 격자 최고) 의 PFT − 릴리즈 를 말하려면 seed 마다 격자 전체를 채점하고 최고를 다시 골라야 한다.
정의
  판 v ∈ {원판, src1..4} (R2 E31c 정정판: 원본 화소 내용으로 seed 한 확률적 반올림, 화소 차 < 1/255, matched 문맥 비트 일치).
  칸 c ∈ {skip2_w16, skip2_w32, skip5_w16} (IntPhys 1 격자 전체). 영상 점수 G(v) = mean_t min_C S(v, t, C) (Filtered AvgSurprise).
  쌍 정답 = 1[G(pos) < G(imp)] (strict). 칸 점수 = 성질 (O1 · O2 · O3) 평균. 규칙 최고 = 칸 점수 최대 칸 (garrido_rescore.select).
  D_s = 최고(PFT, s) − 최고(릴리즈, s).  칸별 하락 Δ_c = 평균_s acc(c, s) − acc(c, 원판).
  뒤집힘: 칸 · 모델마다 원판 정답 쌍 / 오답 쌍을 나누고 쌍마다 4 seed 중 판정이 바뀐 횟수 f ∈ {0..4}.
    정답 조건부 귀무 = "정답 쌍과 오답 쌍의 f 분포가 같다" — 쌍 단위 순열 검정 (라벨 섞기 10,000, 통계 = 평균 f(오답) − 평균 f(정답)).
  CI: 장면 (block_id) bootstrap 2,000 — 매 표본에서 칸 점수를 다시 내고 최고를 다시 고른다 (고르기까지 포함한 CI).
입력  원판 cvpr/results/eval/intphys1/intphys1_dev_{vith,vith_release_vobftg}_wgarrido_w{16,32}/
      잡음 창 32 z_analysis/transfer_gap/results/q3_r2/runs/E31c2_ip1_src{s}_{m}_skip2w32/ (R2) · 창 16 …/results/q3/runs/E36_ip1_src{s}_{m}_w16/ (R3)
출력  z_analysis/transfer_gap/results/q3/E36_garrido_noise.json

  $P z_research/scripts/analysis/transfer_gap_garrido_noise.py
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/q3/E36_garrido_noise.json"
CI1 = REPO / "cvpr/results/eval/intphys1"
R2 = REPO / "z_analysis/transfer_gap/results/q3_r2/runs"
R3 = REPO / "z_analysis/transfer_gap/results/q3/runs"
MODELS = {"release": "vith", "vobpft": "vith_release_vobftg"}
COMBOS = ("skip2_w16", "skip2_w32", "skip5_w16")
SEL0 = {"release": "skip2_w32", "vobpft": "skip2_w16"}        # 원판에서 규칙이 고른 칸 (R1 NF)
NB, NPERM = 2000, 10000


def _gr():
    spec = importlib.util.spec_from_file_location("gr", REPO / "z_research/scripts/analysis/garrido_rescore.py")
    gr = importlib.util.module_from_spec(spec); spec.loader.exec_module(gr)
    return gr


GR = _gr()
GR.kind_of = lambda name: "intphys"          # 잡음 판 폴더 이름 (E36_ip1_* · E31c2_ip1_*) 에는 "intphys1" 이 없다 — 모두 IntPhys 1 이다


def dirs(m, v):
    mm = MODELS[m]
    if v == "orig":
        return [CI1 / f"intphys1_dev_{mm}_wgarrido_w16", CI1 / f"intphys1_dev_{mm}_wgarrido_w32"]
    s = v[3:]
    return [R3 / f"E36_ip1_src{s}_{mm}_w16", R2 / f"E31c2_ip1_src{s}_{mm}_skip2w32"]


def hits(ds):
    """-> {combo: {pair_key: (hit, block_type, block_id)}}, report (select)."""
    ds = [d for d in ds if (d / "per_window.json").exists()]
    if len(ds) < 2:
        return None, None
    combos, pw, meta = {}, {}, None
    for d in ds:
        r = GR.rescore(d); combos.update(r["combos"])
        _, meta = GR.load_index(d)
        for vid, rec in json.load(open(d / "per_window.json"))["windows"].items():
            pw.setdefault(vid, []).extend(rec)
    best = GR.select(combos, "intphys")
    out = {}
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
                pr[GR.pair_key(r)]["pos" if r["plausible"] == "1" else "imp"] = v
                pr[GR.pair_key(r)]["bt"] = r["block_type"]; pr[GR.pair_key(r)]["b"] = r["block_id"]
        out[combo] = {k: (S[x["pos"]] < S[x["imp"]], x["bt"], x["b"]) for k, x in pr.items() if "pos" in x and "imp" in x}
    return out, {"combo": best["combo"], "acc": round(best["accuracy"], 3),
                 "per_combo": {c: round(e["selected"]["accuracy"], 3) for c, e in combos.items()}}


def macro(H, keys):
    """H: {pair: (hit, bt, b)} · keys: 고른 쌍 목록 (중복 허용) → 성질 평균 정확도 (%)."""
    g = defaultdict(list)
    for k in keys:
        h, bt, _ = H[k]
        g[bt].append(h)
    return 100 * float(np.mean([np.mean(v) for v in g.values()]))


def main():
    vers = ["orig"] + [f"src{s}" for s in range(1, 5)]
    HH, REP = {}, {}
    for m in MODELS:
        for v in vers:
            h, r = hits(dirs(m, v))
            if h is not None:
                HH[(m, v)], REP[(m, v)] = h, r
    have = sorted({v for (_, v) in HH})
    res = {"definition": __doc__.split("정의")[1].split("입력")[0].strip(), "versions_available": have, "per_version": {}}
    for v in have:
        if all((m, v) in REP for m in MODELS):
            res["per_version"][v] = {m: REP[(m, v)] for m in MODELS}
            res["per_version"][v]["D_best"] = round(REP[("vobpft", v)]["acc"] - REP[("release", v)]["acc"], 3)
            res["per_version"][v]["D_per_combo"] = {c: round(REP[("vobpft", v)]["per_combo"][c] - REP[("release", v)]["per_combo"][c], 3) for c in COMBOS}
    seeds = [v for v in have if v != "orig" and v in res["per_version"]]
    D = [res["per_version"][v]["D_best"] for v in seeds]
    if D:
        mD = float(np.mean(D))
        verdict = ("(a) 보고된 하락이 잡음 판에서도 선다" if all(x < 0 for x in D) and mD <= -3 else
                   "(b) Garrido 기준 하락도 원본 화소 경로에서만 생긴다" if mD >= -1 else "(c) 판정 보류")
        res["D_best_seeds"] = {"values": D, "mean": round(mD, 3), "min": min(D), "max": max(D), "orig": res["per_version"]["orig"]["D_best"],
                               "n_seeds": len(D), "verdict": verdict,
                               "best_combo_by_seed": {v: {m: REP[(m, v)]["combo"] for m in MODELS} for v in seeds}}
    # 장면 bootstrap — 고르기 포함
    keys = sorted(HH[("release", "orig")]["skip2_w32"])
    blocks = sorted({HH[("release", "orig")]["skip2_w32"][k][2] for k in keys})
    kb = defaultdict(list)
    for k in keys:
        kb[HH[("release", "orig")]["skip2_w32"][k][2]].append(k)
    rng = np.random.default_rng(0)
    boot = {v: [] for v in ["orig"] + seeds}
    for _ in range(NB):
        bs = rng.choice(blocks, len(blocks))
        kk = [k for b in bs for k in kb[b]]
        for v in boot:
            best = {m: max(macro(HH[(m, v)][c], kk) for c in COMBOS) for m in MODELS}
            boot[v].append(best["vobpft"] - best["release"])
    res["D_best_scene_bootstrap_ci95"] = {v: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)] for v, x in boot.items()}
    pooled = np.mean([boot[v] for v in seeds], 0) if seeds else None
    if pooled is not None:
        res["D_best_seed_mean_scene_bootstrap_ci95"] = [round(float(np.percentile(pooled, 2.5)), 2), round(float(np.percentile(pooled, 97.5)), 2)]
    # 승자의 저주: 칸별 하락
    wc = {}
    for m in MODELS:
        o = {}
        for c in COMBOS:
            a0 = REP[(m, "orig")]["per_combo"][c]
            xs = [REP[(m, v)]["per_combo"][c] for v in seeds]
            o[c] = {"orig": a0, "seeds": xs, "mean": round(float(np.mean(xs)), 3), "delta_mean": round(float(np.mean(xs)) - a0, 3)}
        sel = SEL0[m]; uns = "skip2_w16" if sel == "skip2_w32" else "skip2_w32"
        gap = round(o[sel]["delta_mean"] - o[uns]["delta_mean"], 3)
        o["selected_orig"] = sel; o["unselected_compared"] = uns; o["delta_selected_minus_unselected"] = gap
        o["verdict"] = "승자의 저주 있음" if gap <= -3 else "승자의 저주 없음 (3 pt 문턱)"
        wc[m] = o
    res["winners_curse"] = wc
    res["unselected_cell_skip2_w16_D_by_seed"] = {"orig": res["per_version"]["orig"]["D_per_combo"]["skip2_w16"],
                                                  "seeds": [res["per_version"][v]["D_per_combo"]["skip2_w16"] for v in seeds]}
    res["selected_cell_skip2_w32_D_by_seed"] = {"orig": res["per_version"]["orig"]["D_per_combo"]["skip2_w32"],
                                                "seeds": [res["per_version"][v]["D_per_combo"]["skip2_w32"] for v in seeds]}
    # 뒤집힘 — 정답 조건부 (쌍 단위 순열)
    fl = {}
    prng = np.random.default_rng(1)
    for m in MODELS:
        for c in COMBOS:
            H0 = HH[(m, "orig")][c]
            ks = sorted(H0)
            corr = np.array([H0[k][0] for k in ks], bool)
            f = np.array([sum(HH[(m, v)][c][k][0] != H0[k][0] for v in seeds) for k in ks])
            stat = f[~corr].mean() - f[corr].mean() if (~corr).any() and corr.any() else float("nan")
            perm = []
            for _ in range(NPERM):
                pc = prng.permutation(corr)
                perm.append(f[~pc].mean() - f[pc].mean())
            perm = np.array(perm)
            p = float((np.abs(perm) >= abs(stat) - 1e-12).mean())
            L, G = int(f[corr].sum()), int(f[~corr].sum())
            nc, nw = int(corr.sum()), int((~corr).sum())
            pool = (L + G) / (len(seeds) * len(ks))
            fl[f"{m}|{c}"] = {"orig_correct": nc, "orig_wrong": nw, "lost": L, "gained": G,
                              "flip_rate_correct": round(L / (len(seeds) * nc), 3) if nc else None,
                              "flip_rate_wrong": round(G / (len(seeds) * nw), 3) if nw else None,
                              "expected_lost_gained_if_uniform": [round(pool * len(seeds) * nc, 1), round(pool * len(seeds) * nw, 1)],
                              "pairs_ever_flipped": int((f > 0).sum()), "perm_stat_wrong_minus_correct": round(float(stat), 3),
                              "perm_p_two_sided": p, "verdict": "한쪽 몰림 (p < 0.01)" if p < 0.01 else "몰림 없음"}
    res["flips_conditional"] = fl
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in res if k not in ("definition", "flips_conditional")}, ensure_ascii=False, indent=0)[:4000])
    print(json.dumps(fl, ensure_ascii=False)[:3000])
    print("→", OUT)


if __name__ == "__main__":
    main()
