#!/usr/bin/env python3
"""H6g (v11) — H6 문서 적대 검증 대응: v11 H6f 원자료 (l1.npy) 로 칸마다 p − 복사 · p − chance 를 block bootstrap CI 와 함께,
표준 표적과 인과 표적을 나란히, A/B 방향별로, late 의 가림 중 (재등장 전) / 재등장 후 슬롯을 나눠서.

왜: H6 첫 판 §0.5 는 이동 칸 중 0 이하인 칸만 골라 "움직이는 물체에서는 0 또는 음수" 라고 썼다. 여기서는 12 칸 (timing 4 × 운동 3)
전부와 이동 합침 추정을 CI 와 함께 낸다. 또 §5.5 "v11 번짐 미측정" 을 late 의 재등장 전 슬롯으로 직접 잰다.

공식:
  L1[i, j, c]  clip i, 미래 슬롯 j (0..7, 튜블릿), c ∈ {p_full, copyz_full, p_causal, copyz_causal}   (h6f_v11_baselines.py)
  S_c(i) = mean_{j ∈ J} L1[i, j, c]          J = 전 슬롯 (표준 채점) 또는 부분 슬롯
  쌍 정답 o_c = 1[S_c(imp) > S_c(pos)] + 0.5·1[=]   (matched pair: block_id × pair_id A/B)
  p − copy = mean(o_p − o_copy)  같은 쌍 위의 차.  CI: block 단위 bootstrap (B = 4000), 선택된 칸 안에서 block 재표집.
late 재등장 전 슬롯: sym_k = k 면 미래 샘플 프레임 16 .. 16+k−1 이 가려진다 → 두 프레임이 모두 가려진 튜블릿 j < ⌊k/2⌋ = "pre".
  검증: pre 슬롯에서 인과 표적의 Δ = L1(imp) − L1(pos) 가 정확히 0 인 비율 (pos/imp 입력이 픽셀 단위로 같으면 1 이어야 한다).
  그 슬롯의 표준 표적 정답률 = **표적의 양방향 번짐만으로 나오는 판별**.

입력 (vll6 로컬): /data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f/{l1.npy, meta.json}
출력 (NFS): auto_research/exp_results/h6/h6g_v11.json

  auto_research/scripts/srun6.sh 4 16G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h6g_v11_cell_ci.py
"""
from __future__ import annotations
import collections, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f")
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
OUT = ROOT / "auto_research/exp_results/h6"
B = 4000
rng = np.random.default_rng(0)

meta = json.load(open(I / "meta.json")); n = meta["n"]; L = np.load(I / "l1.npy").astype(np.float64); cols = meta["cols"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
ref = json.load(open(REF))["per_video_surprise"]; r = np.array([ref[v] for v in M["video_id"]])
val = np.abs(L[:, :, 0].mean(1) - r)
pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
P = np.array([(d["pos"], d["imp"]) for d in pairs.values() if len(d) == 2]); ps, im = P[:, 0], P[:, 1]
blk = M["block_id"][ps]; T, MOT, VIO, AB = tim[ps], M["motion"][ps], M["violation_type"][ps], M["pair_id"][ps]
K = M["sym_k"][ps].astype(int)
CI = {c: i for i, c in enumerate(cols)}


def ok_of(S):
    return (S[im] > S[ps]).astype(float) + 0.5 * (S[im] == S[ps])


def boot(v, m):
    """block bootstrap: [추정 %, lo, hi, n 쌍, n block]."""
    v, cl = v[m], blk[m]
    if len(v) == 0: return None
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, v, len(u)); c = np.bincount(inv, None, len(u))
    idx = rng.integers(0, len(u), (B, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * v.mean(), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(len(v)), int(len(u))]


def stats(O, m):
    d = {c: boot(O[:, CI[c]], m) for c in cols}
    for tg in ("full", "causal"):
        d[f"p_minus_copy_{tg}"] = boot(O[:, CI[f"p_{tg}"]] - O[:, CI[f"copyz_{tg}"]], m)
        d[f"p_minus_chance_{tg}"] = boot(O[:, CI[f"p_{tg}"]] - 0.5, m)
        d[f"copy_minus_chance_{tg}"] = boot(O[:, CI[f"copyz_{tg}"]] - 0.5, m)
    d["p_full_minus_causal"] = boot(O[:, CI["p_full"]] - O[:, CI["p_causal"]], m)
    d["copy_full_minus_causal"] = boot(O[:, CI["copyz_full"]] - O[:, CI["copyz_causal"]], m)
    return d


O = ok_of(L.mean(1))                                                     # (npairs, 4) 전 슬롯
out = {"_doc": "값 = [추정 %, CI lo, CI hi, n 쌍, n block]. p_minus_copy = 같은 쌍 위 (o_p − o_copy) 평균 (pt).",
       "validation": {"max_abs": float(val.max()), "median_abs": float(np.median(val)), "n_clip": int(n), "n_pairs": int(len(P))},
       "abs_l1": {c: float(L[:, :, CI[c]].mean()) for c in cols}, "cells": {}, "pooled": {}, "direction": {}, "late_slots": {}}
TIMS = ("visible", "early", "mid", "late"); MOTS = ("static", "moving_flat", "moving")
mov = MOT != "static"
for t in TIMS:
    for mo in MOTS:
        m = (T == t) & (MOT == mo)
        out["cells"][f"{t}|{mo}"] = stats(O, m)
        for ab in ("A", "B"):
            out["direction"][f"{t}|{mo}|{ab}"] = stats(O, m & (AB == ab))
    for vi in ("vanish", "shape", "color"):
        for ab in ("A", "B"):
            out["direction"][f"{t}|{vi}|{ab}"] = stats(O, (T == t) & (VIO == vi) & (AB == ab))
POOL = {"ALL": np.ones(len(P), bool), "moving(8 cells)": mov, "moving_occluded(6 cells)": mov & (T != "visible"),
        "static(4 cells)": ~mov, "static_occluded(3 cells)": ~mov & (T != "visible"),
        **{f"{t}": T == t for t in TIMS}, **{f"{t}|moving(both)": (T == t) & mov for t in TIMS},
        "moving|A": mov & (AB == "A"), "moving|B": mov & (AB == "B"), "static|A": ~mov & (AB == "A"), "static|B": ~mov & (AB == "B")}
for k, m in POOL.items():
    out["pooled"][k] = stats(O, m)

# late: 재등장 전 (가려진 튜블릿) / 재등장 후 슬롯
late = T == "late"
for kk in (2, 3, 4):
    npre = kk // 2
    m = late & (K == kk)
    pre_i, post_i = list(range(npre)), list(range(npre, 8))
    Opre, Opost = ok_of(L[:, pre_i].mean(1)), ok_of(L[:, post_i].mean(1))
    dpre = L[im][:, pre_i] - L[ps][:, pre_i]                            # (npairs, npre, 4)
    zero_frac = {c: float((dpre[m][:, :, CI[c]] == 0).mean()) for c in cols}
    for mo in ("static", "moving_flat", "moving", "all"):
        mm = m & ((MOT == mo) if mo != "all" else True)
        out["late_slots"][f"k{kk}|{mo}|pre"] = stats(Opre, mm)
        out["late_slots"][f"k{kk}|{mo}|post"] = stats(Opost, mm)
    out["late_slots"][f"k{kk}|pre_exact_zero_delta_frac"] = zero_frac
# k = 2..4 합침 (pre 슬롯이 있는 late 쌍)
Opre_all = np.full((len(P), 4), np.nan); Opost_all = np.full((len(P), 4), np.nan)
for kk in (2, 3, 4):
    npre = kk // 2; m = late & (K == kk)
    Opre_all[m] = ok_of(L[:, :npre].mean(1))[m]; Opost_all[m] = ok_of(L[:, npre:].mean(1))[m]
mk = late & (K >= 2)
for mo in ("static", "moving_flat", "moving", "moving(both)", "all"):
    mm = mk & ((MOT == mo) if mo in MOTS else (mov if mo == "moving(both)" else True))
    out["late_slots"][f"k2-4|{mo}|pre"] = stats(Opre_all, mm)
    out["late_slots"][f"k2-4|{mo}|post"] = stats(Opost_all, mm)
    for ab in ("A", "B"):
        out["late_slots"][f"k2-4|{mo}|pre|{ab}"] = stats(Opre_all, mm & (AB == ab))
        out["late_slots"][f"k2-4|{mo}|post|{ab}"] = stats(Opost_all, mm & (AB == ab))

# late: 재등장 튜블릿만 (j_r = ⌊k/2⌋; k 홀수면 가려진 1 장 + 첫 보이는 1 장, 짝수면 첫 보이는 2 장) / 그 뒤 슬롯. k = 1..4 합침
#   IntPhys 1 의 '분기 튜블릿 (e0)' 과 짝을 맞추기 위한 것
Orev = np.full((len(P), 4), np.nan); Oaft = np.full((len(P), 4), np.nan)
for kk in (1, 2, 3, 4):
    jr = kk // 2; m = late & (K == kk)
    Orev[m] = ok_of(L[:, jr:jr + 1].mean(1))[m]; Oaft[m] = ok_of(L[:, jr + 1:].mean(1))[m]
for mo in ("static", "moving_flat", "moving", "moving(both)", "all"):
    mm = late & ((MOT == mo) if mo in MOTS else (mov if mo == "moving(both)" else True))
    out["late_slots"][f"k1-4|{mo}|reveal"] = stats(Orev, mm)
    out["late_slots"][f"k1-4|{mo}|after"] = stats(Oaft, mm)
    for ab in ("A", "B"):
        out["late_slots"][f"k1-4|{mo}|reveal|{ab}"] = stats(Orev, mm & (AB == ab))

OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "h6g_v11.json", "w"), indent=1, ensure_ascii=False)


def line(k, d):
    f = lambda x: f"{x[0]:5.1f}" if x else "  -  "
    g = lambda x: f"{x[0]:+5.1f} [{x[1]:+.1f},{x[2]:+.1f}]" if x else "-"
    return (f"{k:34s} p {f(d['p_full'])} c {f(d['copyz_full'])} | pc {f(d['p_causal'])} cc {f(d['copyz_causal'])} | "
            f"p−c std {g(d['p_minus_copy_full'])}  p−50 std {g(d['p_minus_chance_full'])} | p−c cau {g(d['p_minus_copy_causal'])}  n {d['p_full'][3]}")


print("validation", out["validation"], "abs_l1", out["abs_l1"])
for sec in ("cells", "pooled", "direction", "late_slots"):
    print(f"\n== {sec}")
    for k, d in out[sec].items():
        if isinstance(d, dict) and "p_full" in d and d["p_full"]:
            print(line(k, d))
        else:
            print(k, d)
