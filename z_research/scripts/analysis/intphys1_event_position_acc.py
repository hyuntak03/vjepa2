#!/usr/bin/env python3
"""IntPhys 1 Garrido sliding — 창마다 pos/imp 가 처음 갈리는 프레임이 context 안 / 예측 구간 안 / 창 뒤 중 어디인가, 그리고 그걸로 정확도를 다시 낸다.

왜: 88.89 칸 (skip2_w32) 의 창 그림 (`z_research/Benchmarks/figures/intphys1_windows/`) 에서 O1_01 쌍은 거의 모든 시작점에서
사건이 예측 구간에 있는데 start raw 20 만 Filtered 가 C=16 을 골라 사건이 context 안에 들어갔다. 이런 창은 모델이 위반을 **이미 본 뒤**
미래를 예측하므로 "예측이 어긋나서 놀란다" 는 VoE 채점의 전제 (context 가 픽셀 단위로 같은 matched pair) 가 깨진다. 그 창이 점수에 얼마나 들어가나.

정의 (GPU 불필요 — 공식 실행의 창별 surprise `per_window.json` 을 그대로 쓴다):
  d        = pos/imp PNG 가 처음 다른 raw 프레임 (0-기준, PNG 직접 비교)   ← pairs.csv first_div_sensitive − 1 과 대조
  창 (s, C) 의 context raw 인덱스 cf, 예측 raw 인덱스 tf (채점기와 같은 생성기 _intphys1_windows)
  분류  before  : d ≤ cf[0]             창 전체가 이미 다르다
        context : cf[0] < d ≤ cf[-1]    위반이 context 안 (모델이 봤다)
        future  : cf[-1] < d ≤ tf[-1]   context 는 픽셀 단위로 같고 위반은 예측 구간 안   ← VoE 의 정상 경우
        after   : d > tf[-1]            창 전체가 같다 (동점이어야 한다)
  창 정답  = 1[S_imp > S_pos] + 0.5·1[=]   (같은 (s, C) 창끼리)
  공식 점수 G(v) = mean_s min_C S(v, s, C) → 쌍 정답 → principle (O1/O2/O3) macro
  변형 (창 집합을 쌍마다 정한다 — pos 와 imp 가 같은 창 집합을 쓴다):
    V0 all            : 공식 (재현: skip2_w32 88.89)
    V1 future_only    : 분류가 future 인 창만. 시작점마다 그 C 들 중 min, 남는 시작점이 없으면 그 쌍은 빠진다 (n 병기)
    V2 no_context     : context·before 창을 뺀다 (future + after)
    V3 context_only   : context·before 창만 (위반을 본 창만으로 몇 점인가)
  Filtered 가 고른 창 (시작점마다 영상별 argmin_C) 의 분류 비율도 낸다.
CI: scene (4 중항) bootstrap, macro 는 principle 층화, B = 10000.

  python z_research/scripts/analysis/intphys1_event_position_acc.py
  → z_research/Benchmarks/exp_results/intphys1_event_position/event_position_acc.json
  문서: z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md
"""
from __future__ import annotations
import collections, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

BENCH = ROOT / "z_research/Benchmarks/exp_results"
PAIRS = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene/pairs.csv"
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
OUT = BENCH / "intphys1_event_position"
CELLS = {"skip2_w32": (2, 32, "w32"), "skip2_w16": (2, 16, "w16"), "skip5_w16": (5, 16, "w16")}
CATS = ("before", "context", "future", "after")
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
B = 10000
rng = np.random.default_rng(0)


def png(vid, raw):
    b, q, r = vid.split("_")
    return np.asarray(Image.open(FR / b / q / r / "scene" / f"scene_{raw + 1:03d}.png"))


def first_div(pos, imp):
    for f in range(100):
        if not np.array_equal(png(pos, f), png(imp, f)):
            return f
    return None


def cat(d, cf, tf):
    if d is None or d > tf[-1]:
        return "after"
    if d <= cf[0]:
        return "before"
    if d <= cf[-1]:
        return "context"
    return "future"


def boot_macro(v, scene, prin):
    est, bs = [], np.zeros(B)
    for pr in ("O1", "O2", "O3"):
        m = prin == pr
        if not m.any():
            continue
        u, inv = np.unique(scene[m], return_inverse=True)
        s = np.bincount(inv, v[m], len(u)); c = np.bincount(inv, None, len(u))
        idx = rng.integers(0, len(u), (B, len(u)))
        bs += s[idx].sum(1) / c[idx].sum(1) / 3; est.append(v[m].mean())
    return [round(100 * float(np.mean(est)), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(len(v))]


def boot_pool(v, scene):
    u, inv = np.unique(scene, return_inverse=True)
    s = np.bincount(inv, v, len(u)); c = np.bincount(inv, None, len(u))
    idx = rng.integers(0, len(u), (B, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * float(v.mean()), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(len(v))]


def main():
    pairs = list(csv.DictReader(PAIRS.open()))
    for p in pairs:
        p["d"] = first_div(p["pos"], p["imp"])
        fs = int(p["first_div_sensitive"]) - 1 if p["first_div_sensitive"] not in ("", None) else None
        p["d_matches_sensitive"] = (p["d"] == fs)
        p["group"] = f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'
    S = {}
    for w in ("w16", "w32"):
        for vid, rows in json.load(open(BENCH / f"intphys1_sliding__intphys1_dev_vith_{w}/per_window.json"))["windows"].items():
            for combo, s, C, sur in rows:
                S[(vid, combo, int(s), int(C))] = float(sur)
    out = {"_doc": __doc__.split("\n\n")[1], "n_pairs": len(pairs),
           "d_vs_first_div_sensitive_match": int(sum(p["d_matches_sensitive"] for p in pairs)),
           "d_none": int(sum(p["d"] is None for p in pairs)), "cells": {}}
    scene_all = np.array([p["scene"] for p in pairs]); prin_all = np.array([p["principle"][:2] for p in pairs])
    for cell, (sk, wz, _) in CELLS.items():
        cl = [m * wz // 16 for m in (2, 4, 6, 8, 10)]
        wins = _intphys1_windows(100, sk, wz, cl, 2, 2, "official")
        W = [(cf[0], C, cf, tf) for C, cf, tf in wins]
        # ── 창 단위 ──
        wrows = []                                                   # (pair_i, s, C, cat, correct)
        for i, p in enumerate(pairs):
            for s, C, cf, tf in W:
                a, b = S[(p["imp"], cell, s, C)], S[(p["pos"], cell, s, C)]
                wrows.append((i, s, C, cat(p["d"], cf, tf), 1.0 if a > b else (0.5 if a == b else 0.0), a == b))
        res = {"n_windows_per_video": len(W), "window_level": {}, "filtered_choice": {}, "variants": {}}
        for c in CATS:
            sel = [r for r in wrows if r[3] == c]
            if not sel:
                res["window_level"][c] = None; continue
            v = np.array([r[4] for r in sel]); sc = scene_all[[r[0] for r in sel]]
            res["window_level"][c] = {"acc_pooled": boot_pool(v, sc), "share_of_windows": round(len(sel) / len(wrows), 4),
                                      "exact_ties": int(sum(r[5] for r in sel))}
        # ── Filtered 가 고른 창의 분류 ──
        cho = collections.Counter(); cho_pair = collections.Counter(); start_rows = []
        for i, p in enumerate(pairs):
            for s in sorted({x[0] for x in W}):
                ws = [x for x in W if x[0] == s]
                cats = {}
                for role in ("pos", "imp"):
                    Cs = min(ws, key=lambda x: S[(p[role], cell, s, x[1])])
                    cats[role] = cat(p["d"], Cs[2], Cs[3])
                    cho[(role, cats[role])] += 1
                gi = min(S[(p["imp"], cell, s, x[1])] for x in ws); gp = min(S[(p["pos"], cell, s, x[1])] for x in ws)
                key = f'pos={cats["pos"]}|imp={cats["imp"]}'
                cho_pair[key] += 1
                start_rows.append((i, key, 1.0 if gi > gp else (0.5 if gi == gp else 0.0)))
        n_st = len(start_rows)
        res["filtered_choice"] = {"by_role": {f"{r}:{c}": round(n / (n_st), 4) for (r, c), n in sorted(cho.items())},
                                  "by_pair_of_choices": {}}
        for key, n in cho_pair.most_common():
            sel = [r for r in start_rows if r[1] == key]
            v = np.array([r[2] for r in sel]); sc = scene_all[[r[0] for r in sel]]
            res["filtered_choice"]["by_pair_of_choices"][key] = {"share_of_starts": round(n / n_st, 4), "start_acc": boot_pool(v, sc)}
        # ── 변형 점수 ──
        keep = {"V0_all": set(CATS), "V1_future_only": {"future"}, "V2_no_context": {"future", "after"},
                "V3_context_only": {"before", "context"}}
        by_group = {}
        for name, allowed in keep.items():
            o, pi = [], []
            for i, p in enumerate(pairs):
                G = {}
                for role in ("pos", "imp"):
                    per_s = collections.defaultdict(list)
                    for s, C, cf, tf in W:
                        if cat(p["d"], cf, tf) in allowed:
                            per_s[s].append(S[(p[role], cell, s, C)])
                    G[role] = float(np.mean([min(v) for v in per_s.values()])) if per_s else None
                if G["pos"] is None:
                    continue
                o.append(1.0 if G["imp"] > G["pos"] else (0.5 if G["imp"] == G["pos"] else 0.0)); pi.append(i)
            o = np.array(o); pi = np.array(pi)
            res["variants"][name] = {"macro": boot_macro(o, scene_all[pi], prin_all[pi]), "n_pairs": int(len(o)),
                                     "per_principle": {pr: round(100 * float(o[prin_all[pi] == pr].mean()), 2) for pr in ("O1", "O2", "O3") if (prin_all[pi] == pr).any()}}
            by_group[name] = {g: round(100 * float(o[np.array([pairs[j]["group"] for j in pi]) == g].mean()), 2)
                              for g in sorted({pairs[j]["group"] for j in pi})}
            by_group[name]["_n"] = {g: int((np.array([pairs[j]["group"] for j in pi]) == g).sum()) for g in sorted({pairs[j]["group"] for j in pi})}
        res["variants_by_group"] = by_group
        # V0 과 V1 이 쌍마다 다른 곳
        out["cells"][cell] = res
    # p 대 copy (auto_research H6h 원자료가 있으면): 같은 분류·변형을 p_full / copy_ln / copy_raw 창 surprise 에 건다
    h6h = sorted((ROOT / "auto_research/exp_results/h6").glob("h6h_copy_noln_vith_r*.npz"))
    if h6h:
        z = np.load(h6h[0]); cols = [str(c) for c in z["cols"]]; hc = [str(c) for c in z["combos"]]; hv = [str(v) for v in z["video_ids"]]
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in np.concatenate([np.load(f)["rows"] for f in h6h]):
            acc[(hv[int(r[0])], hc[int(r[1])], int(r[2]), int(r[3]))]["_"].append(r)
        out["copy_vs_p"] = {}
        for cell, (sk, wz, _) in CELLS.items():
            cl = [m * wz // 16 for m in (2, 4, 6, 8, 10)]
            W = [(cf[0], C, cf, tf) for C, cf, tf in _intphys1_windows(100, sk, wz, cl, 2, 2, "official")]
            res = {}
            for vname, allowed in (("V0_all", set(CATS)), ("V1_future_only", {"future"})):
                O = {}
                for m in ("p_full", "copy_ln", "copy_raw"):
                    ci = cols.index(m); o, pi = [], []
                    for i, p in enumerate(pairs):
                        G = {}
                        for role in ("pos", "imp"):
                            per_s = collections.defaultdict(list)
                            for s_, C, cf, tf in W:
                                if cat(p["d"], cf, tf) in allowed:
                                    per_s[s_].append(float(np.mean([x[ci] for x in acc[(p[role], cell, s_, C)]["_"]])))
                            G[role] = float(np.mean([min(v) for v in per_s.values()])) if per_s else None
                        if G["pos"] is None:
                            continue
                        o.append(1.0 if G["imp"] > G["pos"] else (0.5 if G["imp"] == G["pos"] else 0.0)); pi.append(i)
                    O[m] = (np.array(o), np.array(pi))
                o_p, pi = O["p_full"]
                res[vname] = {m: boot_macro(O[m][0], scene_all[O[m][1]], prin_all[O[m][1]]) for m in O}
                d_ = O["p_full"][0] - O["copy_ln"][0] + 0.5
                res[vname]["p_minus_copy_ln"] = [round(x - 50, 2) for x in boot_macro(d_, scene_all[pi], prin_all[pi])[:3]]
            out["copy_vs_p"][cell] = res
    # 예시 쌍 O1_01 (그림의 쌍)
    ex = next(p for p in pairs if p["pos"] == "O1_01_2" and p["imp"] == "O1_01_1")
    cell = "skip2_w32"; wins = _intphys1_windows(100, 2, 32, [4, 8, 12, 16, 20], 2, 2, "official")
    W = [(cf[0], C, cf, tf) for C, cf, tf in wins]; rows = []
    for s in sorted({x[0] for x in W}):
        ws = [x for x in W if x[0] == s]
        r = {"start": s}
        for role in ("pos", "imp"):
            m = min(ws, key=lambda x: S[(ex[role], cell, s, x[1])])
            r[f"{role}_C*"] = m[1]; r[f"{role}_cat"] = cat(ex["d"], m[2], m[3]); r[f"{role}_S"] = round(S[(ex[role], cell, s, m[1])], 5)
        r["correct"] = r["imp_S"] > r["pos_S"]
        r["per_C"] = {C: {"cat": cat(ex["d"], cf, tf), "imp>pos": S[(ex["imp"], cell, s, C)] > S[(ex["pos"], cell, s, C)]} for s_, C, cf, tf in ws for _ in [0]}
        rows.append(r)
    out["example_O1_01"] = {"d": ex["d"], "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_position_acc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "_doc"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
