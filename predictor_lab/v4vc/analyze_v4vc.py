#!/usr/bin/env python3
"""v4_velocity_control 분석 (2026-11-02) — ① 같은 장면 · 다른 속도 시험 (v4vctest) ② 바깥 벤치 전이 ③ 속도 따라감 (VOBench · trapdoor).

① v4vctest (묶음 322 × 속도 4 단계; 판정은 v3 표와 같다: s = 미래 8 튜블릿 평균 L1, 정답 = s(불가능) > s(가능), 같으면 0.5)
   사라짐   (A_k, vanish_k)       문맥 A_k · 미래 빈 장면          — 1,288 짝
   나타남   (B, appear_k)         문맥 빈 장면 · 미래 A_k          — 1,288 짝
   속도 바꿔치기 (A_k, swap_k_j)  문맥 속도 k · 미래 속도 j ≠ k    — 3,864 짝 (|k−j| = 1, 2, 3 · 미래가 더 빠름 / 느림)
   제 속도 고르기: 문맥 k 마다 미래 4 개 (제 것 + 바꿔치기 3) 중 s 가 가장 작은 것이 제 것인 비율 (우연 25 %)
   튜블릿별 바꿔치기 정답률 (미래 튜블릿 1–8) · 가림 종류별 (옆으로 빠져나옴 = 판 · 벽 · 물체 / 제자리에서 드러남 = trapdoor)
   95 % CI = 묶음 단위 bootstrap 2,000 번.
② 바깥 벤치: table_v3.panel_scores (배치 병합 규칙 그대로) — VOBench late permanence (A 사라짐 · 움직임 / 정지, B 나타남) · VO_moving_occluder ·
   Blender v3 · trapdoor 시험 · CLEVRER 식 (static 뺌) · IntPhys 1 late occ · IntPhys 2 late permanence · Rubric. 짝 비교 McNemar (양측 정확).
③ where_speed_v4vc (where_speed.py WS_MODE=v4vc): 같은 영상에서 프레임 간격만 바꿨을 때 예측 자리가 따라가는 비 (0 = 고정, 1 = 속도대로).
출력 predictor_lab/results/v4vc/analysis.json · analysis.md
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v4vc/analyze_v4vc.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "predictor_lab/v3_ablation"))
import table_v3 as T  # noqa: E402

TE = REPO / "predictor_lab/results/v3_ablation/te"
OUT = REPO / "predictor_lab/results/v4vc"
ARMS = ["copy", "release", "vobftg", "v3_full_s0", "v3t_only_s0", "v3_mix1600_s0", "v4vc_ind_s0", "v4vc_fam_s0", "v4vc_ind6k_s0", "v4vc_fam6k_s0", "v4vc_all_s0"]
NAME = {"copy": "복사", "release": "원래 (release)", "vobftg": "VOBench 학습", "v3_full_s0": "v3 전부", "v3t_only_s0": "trapdoor 만",
        "v3_mix1600_s0": "v3 섞음", "v4vc_ind_s0": "대조 2k", "v4vc_fam_s0": "v4 평면 2k", "v4vc_ind6k_s0": "대조 6k", "v4vc_fam6k_s0": "v4 평면 6k", "v4vc_all_s0": "v4 평면+빗면 4k"}
CMP = [("v4vc_fam_s0", "v4vc_ind_s0"), ("v4vc_fam6k_s0", "v4vc_ind6k_s0"), ("v4vc_fam_s0", "release"), ("v4vc_fam6k_s0", "release"),
       ("v4vc_fam6k_s0", "v4vc_fam_s0"), ("v4vc_fam6k_s0", "vobftg"), ("v4vc_all_s0", "v4vc_fam_s0"), ("v4vc_all_s0", "release"), ("v4vc_all_s0", "v3_full_s0")]
RNG = np.random.default_rng(0)


def judge(si, sp):
    return np.where(si > sp, 1.0, np.where(si == sp, 0.5, 0.0))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(vals, fam, n=2000):
    """묶음 단위 bootstrap 평균 95 % CI (vals · fam 같은 길이)."""
    fams = np.unique(fam)
    by = {f: vals[fam == f] for f in fams}
    sums = np.array([by[f].sum() for f in fams]); cnts = np.array([len(by[f]) for f in fams])
    idx = RNG.integers(0, len(fams), (n, len(fams)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------- ① v4vctest
def load_v4(panel="v4vctest"):
    dirs = sorted((d for d in TE.glob(f"{panel}_*") if (d / "done.npy").exists() and np.load(d / "done.npy").all()),
                  key=lambda d: (d / "meta.json").stat().st_mtime)
    if not dirs:
        return None
    S, vids, rows = {}, None, None
    for d in dirs:
        m = json.load(open(d / "meta.json"))
        l1 = np.load(d / "l1.npy"); cp = np.load(d / "copy.npy")
        if vids is None:
            vids = list(m["video_id"])
            idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
            rows = [idx[v] for v in vids]
        assert list(m["video_id"]) == vids
        S["copy"] = cp
        for k, t in enumerate(m["pred_tags"]):
            S[t] = l1[:, k, :]
    return S, rows


def v4_tables(S, rows):
    K = 4
    by = defaultdict(dict)
    for i, r in enumerate(rows):
        by[r["block_id"]][r["variant"]] = i
    fams = sorted(by)
    occ = {b: rows[next(iter(by[b].values()))]["occ"] + ("_up" if rows[next(iter(by[b].values()))].get("ramp_mode") == "up" else "") for b in fams}
    res = {"n_family": len(fams), "models": {}}
    for t in [a for a in ARMS if a in S]:
        s8 = S[t]; s = s8.mean(-1)
        recs = defaultdict(list)                                       # 종류 → [(값, 묶음 번호, 가림, k, j)]
        for fi, b in enumerate(fams):
            v = by[b]
            for k in range(K):
                recs["vanish"].append((judge(s[v[f"vanish_k{k}"]], s[v[f"A_k{k}"]]), fi, occ[b], k, -1))
                recs["appear"].append((judge(s[v[f"appear_k{k}"]], s[v["B"]]), fi, occ[b], k, -1))
                own = s[v[f"A_k{k}"]]
                others = [s[v[f"swap_k{k}_j{j}"]] for j in range(K) if j != k]
                mn = min([own] + others)
                recs["choice"].append(((own == mn) / sum(x == mn for x in [own] + others), fi, occ[b], k, -1))
                for j in range(K):
                    if j != k:
                        recs["swap"].append((judge(s[v[f"swap_k{k}_j{j}"]], own), fi, occ[b], k, j))
                        recs["swap_slots"].append((judge(s8[v[f"swap_k{k}_j{j}"]], s8[v[f"A_k{k}"]]), fi, occ[b], k, j))
        out = {}
        for kind in ("vanish", "appear", "swap", "choice"):
            vals = np.array([x[0] for x in recs[kind]], float); fi = np.array([x[1] for x in recs[kind]])
            oc = np.array([x[2] for x in recs[kind]]); kk = np.array([x[3] for x in recs[kind]]); jj = np.array([x[4] for x in recs[kind]])
            def cell(mask):
                if mask.sum() == 0:
                    return None
                lo, hi = boot(vals[mask], fi[mask])
                return {"acc": round(100 * float(vals[mask].mean()), 1), "ci": [round(100 * lo, 1), round(100 * hi, 1)], "n": int(mask.sum())}
            allm = np.ones(len(vals), bool)
            d = {"all": cell(allm), "exit_side": cell(~np.char.startswith(oc.astype(str), "trapdoor")), "trapdoor": cell(np.char.startswith(oc.astype(str), "trapdoor"))}
            for o in ("board", "wall", "object", "trapdoor_up"):
                d[o] = cell(oc == o)
            if kind == "swap":
                for g in (1, 2, 3):
                    d[f"gap{g}"] = cell(np.abs(kk - jj) == g)
                    d[f"gap{g}_exit_side"] = cell((np.abs(kk - jj) == g) & (oc != "trapdoor"))
                    d[f"gap{g}_trapdoor"] = cell((np.abs(kk - jj) == g) & (oc == "trapdoor"))
                d["fut_faster"] = cell(jj > kk); d["fut_slower"] = cell(jj < kk)
                d["_vals"] = vals                                       # McNemar 용 (저장 안 함)
            if kind in ("vanish", "appear"):
                for k in range(K):
                    d[f"k{k}"] = cell(kk == k)
            out[kind] = d
        sl = np.stack([x[0] for x in recs["swap_slots"]])               # (n, 8)
        oc = np.array([x[2] for x in recs["swap_slots"]])
        out["swap_by_slot"] = {"all": [round(100 * float(x), 1) for x in sl.mean(0)],
                               "exit_side": [round(100 * float(x), 1) for x in sl[oc != "trapdoor"].mean(0)],
                               "trapdoor": [round(100 * float(x), 1) for x in sl[oc == "trapdoor"].mean(0)]}
        res["models"][t] = out
    cmp = {}
    for x, y in CMP:
        if x in res["models"] and y in res["models"]:
            a, b_ = res["models"][x]["swap"]["_vals"], res["models"][y]["swap"]["_vals"]
            bb = int(((a == 1) & (b_ < 1)).sum()); cc = int(((b_ == 1) & (a < 1)).sum())
            cmp[f"{x} vs {y}"] = {"only_x": bb, "only_y": cc, "p": mcnemar(bb, cc)}
    for t in res["models"]:
        res["models"][t]["swap"].pop("_vals", None)
    res["swap_mcnemar"] = cmp
    return res


# ----------------------------------------------------------------------------- ② 바깥 벤치
CELLS = [("vob_late", "A", None), ("vob_late", "A_moving", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and r["motion"] != "flat_static"),
         ("vob_late", "A_static", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and r["motion"] == "flat_static"),
         ("vob_late", "B", None), ("vob_late", "all", None)] + [
         ("vob_late", f"A_{mo}", (lambda m_: lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and r["motion"] == m_)(mo))
         for mo in ("flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up")] + [
         ("vomo", "A", None), ("vomo", "B", None),
         ("v3test", "A", None), ("v3ttest", "A", None), ("clevrer", "all", None), ("ip1uni", "all", None), ("ip1uni", "moving", None),
         ("ip2uni", "all", None), ("rubric", "all", None)]


def transfer():
    out, cache = {}, {}
    for pn, sub, f in CELLS:
        if pn not in cache:
            try:
                cache[pn] = T.panel_scores(pn)
            except Exception as e:                                       # noqa: BLE001
                print("panel 실패", pn, e); cache[pn] = None
        if cache[pn] is None:
            continue
        S, P = cache[pn]
        f = f or dict(T.PANELS[pn])[sub]
        sel = [(a, b) for a, b, r in P if f(r)]
        J = {t: T._judge(S[t], sel) for t in ARMS if t in S}
        cell = {"n": len(sel), "acc": {t: round(100 * float(J[t].mean()), 1) for t in J}, "mcnemar": {}}
        for x, y in CMP:
            if x in J and y in J:
                b = int(((J[x] == 1) & (J[y] < 1)).sum()); c = int(((J[y] == 1) & (J[x] < 1)).sum())
                cell["mcnemar"][f"{x} vs {y}"] = {"only_x": b, "only_y": c, "p": mcnemar(b, c)}
        out[f"{pn}/{sub}"] = cell
    return out


# ----------------------------------------------------------------------------- ③ 속도 따라감
def where_speed():
    p = REPO / "predictor_lab/results/v3_ablation/where_speed_v4vc/where_speed.json"
    if not p.exists():
        p = REPO / "predictor_lab/results/v3_ablation/where_speed_v4vc.json"
    return json.load(open(p)) if p.exists() else None


def md(res):
    L = ["# v4_velocity_control 분석 (자동 생성)", ""]
    for pn, title in (("v4vctest", "평면"), ("v4vcrtest", "빗면")):
      v4 = res.get(pn)
      if v4:
        L += [f"## ① 같은 장면 · 다른 속도 시험 ({title}) — 묶음 {v4['n_family']} 개, 정답률 % [95 % CI]", "",
              "| 모델 | 사라짐 | 나타남 | 속도 바꿔치기 (전체) | 바꿔치기 · 옆으로 나옴 | 바꿔치기 · trapdoor | 바꿔치기 차 1 단계 | 차 3 단계 | 제 속도 고르기 (우연 25) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in [a for a in ARMS if a in v4["models"]]:
            m = v4["models"][t]
            def c(x):
                return "—" if x is None else f"{x['acc']} [{x['ci'][0]}, {x['ci'][1]}]"
            L.append(f"| {NAME[t]} | {c(m['vanish']['all'])} | {c(m['appear']['all'])} | {c(m['swap']['all'])} | {c(m['swap']['exit_side'])} | "
                     f"{c(m['swap']['trapdoor'])} | {c(m['swap']['gap1'])} | {c(m['swap']['gap3'])} | {c(m['choice']['all'])} |")
        L += ["", "바꿔치기 McNemar: " + " · ".join(f"{k}: {v['only_x']} 대 {v['only_y']} p={v['p']:.2g}" for k, v in v4["swap_mcnemar"].items()), ""]
    tr = res.get("transfer")
    if tr:
        ks = list(tr)
        L += ["## ② 바깥 벤치 정답률 %", "", "| 모델 | " + " | ".join(f"{k} (n {tr[k]['n']})" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for t in ARMS:
            if any(t in tr[k]["acc"] for k in ks):
                L.append(f"| {NAME[t]} | " + " | ".join(str(tr[k]["acc"].get(t, "—")) for k in ks) + " |")
        L.append("")
        for k in ks:
            if tr[k]["mcnemar"]:
                L.append(f"- {k}: " + " · ".join(f"{a}: {v['only_x']} 대 {v['only_y']} p={v['p']:.2g}" for a, v in tr[k]["mcnemar"].items()))
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for pn in ("v4vctest", "v4vcrtest"):
        L = load_v4(pn)
        if L:
            res[pn] = v4_tables(*L)
    res["transfer"] = transfer()
    res["where_speed"] = where_speed()
    json.dump(res, open(OUT / "analysis.json", "w"), ensure_ascii=False, indent=1, default=float)
    text = md(res)
    (OUT / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
