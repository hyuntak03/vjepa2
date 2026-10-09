#!/usr/bin/env python3
"""H6g (IntPhys 1) — H6 문서 적대 검증 대응: Garrido 칸마다 p − 복사 · p − chance 를 scene bootstrap CI 와 함께, 표준 표적과 인과 표적을 나란히.

왜: H6 첫 판은 칸 (skip2_w32 만) 과 표적 (표준 / 인과) 을 주장마다 바꿔 골랐고 CI 가 없었다. 여기서는
  - 세 Garrido 칸 (skip2_w16 · skip2_w32 · skip5_w16) × 두 표적 × {p, copyz} 를 **같은 실행 (H6e)** 에서 전부 낸다
  - p − copyz (쌍마다 정답 차의 평균) 와 p − 50 을 scene (4 중항) bootstrap 95 % CI 로 낸다. macro 는 principle 층화 bootstrap
  - 그룹 (운동 × 가림) · 이동 합침 · 방향 (O1/O3: 사라짐 / 나타남) 으로 쪼갠다
  - H6e 재추출 (모델 window 64 하나) 이 공식 하네스 (창마다 모델) 와 어디서 어긋나는지 창 단위로 잰다
  - 인과 표적으로 §2 의 창 단위 j 표를 다시 만든다 (번짐 희석 설명의 직접 검정)
  - H6d (재등장 튜블릿 쪼개기) 를 tie = 0.5 · scene CI 로 다시 낸다

공식 (PROTOCOLS.md 와 같다):
  창 surprise  S(v, s, C) = mean_j L1_j                     L1_j = mean_tokens |x_j − LN(h)_j|,  x ∈ {p, copyz}
  영상 점수    G(v) = mean_s min_C S(v, s, C)               (Filtered → AvgSurprise)
  쌍 정답      o = 1[G(imp) > G(pos)] + 0.5·1[G(imp) = G(pos)]
  macro        = mean_{O1,O2,O3} mean_pairs o
  p − copy     = mean_pairs (o_p − o_copy)                   (같은 쌍 위의 차 → 쌍 대응 bootstrap)
  표준 표적 h = LN(target_encoder(창 전체))[u];  인과 표적 h^c_u = LN(target_encoder(창 프레임 [0, 2(u+1))))[u]
  copyz = LN(z)[문맥 마지막 튜블릿] 복사 (예측 없음).  copyh (H6b) 는 **자기 영상의 양방향 h_T** 라 미래를 본다 → 누수 기준선.

입력 (NFS): exp_results/h6/causal_l1_vith_r*.npz (H6e), tubelet_l1_vith_r*.npz (H6b), rows_vith.json 은 안 쓴다,
  z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w{16,32}/per_window.json (공식),
  _stage/IntPhys1_dev_by_scene/{pairs.csv, videos.csv, obj_vis.npz}
출력: exp_results/h6/h6g_intphys1.json

  python auto_research/scripts/h6g_intphys1_cell_ci.py        (CPU, 1 분 안)
"""
from __future__ import annotations
import collections, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H6 = ROOT / "auto_research/exp_results/h6"
OUT = ROOT / "auto_research/exp_results/h6"
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
BENCH = ROOT / "z_research/Benchmarks/exp_results"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
B = 10000
rng = np.random.default_rng(0)


def load(glob, names):
    parts = sorted(H6.glob(glob)); z = np.load(parts[0])
    combos, vids = [str(c) for c in z["combos"]], [str(v) for v in z["video_ids"]]
    R = np.concatenate([np.load(f)["rows"] for f in parts])
    T = collections.defaultdict(dict)
    for r in R:
        T[(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]))][int(r[4])] = dict(zip(names, r[5:9]))
    return T, combos


TE, combos = load("causal_l1_vith_r*.npz", ("p_causal", "copyz_causal", "p_full", "copyz_full"))
TB, _ = load("tubelet_l1_vith_r*.npz", ("p", "copyh", "copyz", "zero"))
KE, KB = collections.defaultdict(list), collections.defaultdict(list)          # 영상 → 창 키
for k in TE: KE[k[0]].append(k)
for k in TB: KB[k[0]].append(k)
OFF = {}
for w in (16, 32):
    for vid, rows in json.load(open(BENCH / f"intphys1_sliding__intphys1_dev_vith_w{w}/per_window.json"))["windows"].items():
        for combo, s, C, sur in rows:
            OFF[(vid, combo, int(s), int(C))] = float(sur)

pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
for p in pairs:
    p["group"] = f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}'
    p["pr"] = p["principle"][:2]
    d = "none"
    if p["pr"] in ("O1", "O3"):
        a_, b_ = int(V[p["imp"]]["obj_visible_frames"]), int(V[p["pos"]]["obj_visible_frames"])
        d = "disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal")
    p["dir"] = d
scene = np.array([p["scene"] for p in pairs]); prin = np.array([p["pr"] for p in pairs])
group = np.array([p["group"] for p in pairs]); dirn = np.array([p["dir"] for p in pairs])


def winmean(T, key, m):
    d = T.get(key)
    return None if d is None else float(np.mean([x[m] for x in d.values()]))


def garrido(T, combo, m):
    vs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (v, c, s, C), d in T.items():
        if c == combo:
            vs[v][s].append(np.mean([x[m] for x in d.values()]))
    return {v: float(np.mean([min(cs) for cs in st.values()])) for v, st in vs.items()}


def garrido_off(combo):
    vs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (v, c, s, C), sur in OFF.items():
        if c == combo:
            vs[v][s].append(sur)
    return {v: float(np.mean([min(cs) for cs in st.values()])) for v, st in vs.items()}


def pair_ok(score):
    o = []
    for p in pairs:
        a, b = score[p["imp"]], score[p["pos"]]
        o.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return np.array(o)


def cl_boot(v, cl, b=B):
    """클러스터 (scene) bootstrap: 평균과 95 % CI. v (n,), cl (n,)."""
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, v, len(u)); c = np.bincount(inv, None, len(u))
    idx = rng.integers(0, len(u), (b, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * v.mean(), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(len(v)), int(len(u))]


def macro_boot(v, b=B):
    """principle 층화 scene bootstrap 으로 macro (O1/O2/O3 평균) 의 CI."""
    est, bs = [], np.zeros(b)
    for pr in ("O1", "O2", "O3"):
        m = prin == pr; u, inv = np.unique(scene[m], return_inverse=True)
        s = np.bincount(inv, v[m], len(u)); c = np.bincount(inv, None, len(u))
        idx = rng.integers(0, len(u), (b, len(u)))
        bs += s[idx].sum(1) / c[idx].sum(1) / 3; est.append(v[m].mean())
    return [round(100 * np.mean(est), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(len(v)), int(len(np.unique(scene)))]


out = {"_doc": "값 = [추정 %, CI lo, CI hi, n 쌍, n scene]. diff = 같은 쌍 위 (o_p − o_copy) 평균 (pt). p_minus_chance = o_p − 0.5.",
       "cells": {}, "groups": {}, "direction": {}, "discordant": {}, "official_vs_h6e": {}, "window_j": {}, "h6d_ties05": {}, "abs_l1": {},
       "_tub_doc": "tub: [추정 %, lo, hi, n 튜블릿, n scene, tie 아닌 비율]. 사건 전 튜블릿은 pos/imp 픽셀·p 가 같다 → 표준 표적에서의 판별은 번짐."}
SEL = {"ALL": np.ones(len(pairs), bool), "moving(all)": np.char.startswith(group.astype(str), "moving"),
       **{g: group == g for g in ("moving/occluded", "moving/visible", "static/occluded", "static/visible")},
       "disappear(O1+O3)": dirn == "disappear", "appear(O1+O3)": dirn == "appear",
       "moving&disappear": np.char.startswith(group.astype(str), "moving") & (dirn == "disappear"),
       "moving&appear": np.char.startswith(group.astype(str), "moving") & (dirn == "appear")}

OK = {}
for cb in combos:
    for tgt in ("full", "causal"):
        op, oc = pair_ok(garrido(TE, cb, f"p_{tgt}")), pair_ok(garrido(TE, cb, f"copyz_{tgt}"))
        OK[(cb, tgt)] = (op, oc)
        k = f"{cb}|{tgt}"
        out["cells"][k] = {"p": macro_boot(op), "copy": macro_boot(oc), "p_minus_copy": macro_boot(op - oc), "p_minus_chance": macro_boot(op - 0.5),
                           "copy_minus_chance": macro_boot(oc - 0.5)}
        out["discordant"][k] = {"p_only": int(((op == 1) & (oc == 0)).sum()), "copy_only": int(((op == 0) & (oc == 1)).sum())}
        for gname, m in SEL.items():
            if m.sum() == 0: continue
            out["groups"][f"{k}|{gname}"] = {"p": cl_boot(op[m], scene[m]), "copy": cl_boot(oc[m], scene[m]),
                                            "p_minus_copy": cl_boot(op[m] - oc[m], scene[m]), "p_minus_chance": cl_boot(op[m] - 0.5, scene[m]),
                                            "d_p_only": int(((op == 1) & (oc == 0) & m).sum()), "d_copy_only": int(((op == 0) & (oc == 1) & m).sum())}
    # 표적 효과 (같은 방법, 표준 − 인과)
    for meth in ("p", "copyz"):
        a, b_ = OK[(cb, "full")][0 if meth == "p" else 1], OK[(cb, "causal")][0 if meth == "p" else 1]
        out["cells"][f"{cb}|{meth}_full_minus_causal"] = macro_boot(a - b_)
    # 차이의 차: (p−copy)_causal − (p−copy)_full
    (pf, cf), (pc, cc) = OK[(cb, "full")], OK[(cb, "causal")]
    out["cells"][f"{cb}|gap_causal_minus_gap_full"] = macro_boot((pc - cc) - (pf - cf))

# 공식 하네스 vs H6e 재추출 (표준 표적 p)
for cb in combos:
    diffs = [abs(winmean(TE, k, "p_full") - OFF[k]) for k in TE if k[1] == cb and k in OFF]
    g_off, g_e = pair_ok(garrido_off(cb)), OK[(cb, "full")][0]
    flips = [(pairs[i]["pos"], pairs[i]["imp"], g_off[i], g_e[i]) for i in range(len(pairs)) if g_off[i] != g_e[i]]
    out["official_vs_h6e"][cb] = {"n_windows": len(diffs), "median_abs": float(np.median(diffs)), "p99_abs": float(np.percentile(diffs, 99)),
                                  "max_abs": float(np.max(diffs)), "macro_official": macro_boot(g_off)[0], "macro_h6e": macro_boot(g_e)[0],
                                  "pair_flips": flips}
    # H6b 실행 (창마다 배치 1) vs H6e (시작점 배치) — 같은 p_full 의 튜블릿 값 차
    dd = [abs(TB[k][j]["p"] - TE[k][j]["p_full"]) for k in TE if k[1] == cb and k in TB for j in TE[k]]
    out["official_vs_h6e"][cb]["h6b_vs_h6e_tubelet_median_abs"] = float(np.median(dd))
    out["official_vs_h6e"][cb]["h6b_vs_h6e_tubelet_max_abs"] = float(np.max(dd))
    out["official_vs_h6e"][cb]["macro_h6b_p"] = macro_boot(pair_ok(garrido(TB, cb, "p")))[0]
    out["official_vs_h6e"][cb]["macro_h6b_copyh_LEAKY"] = macro_boot(pair_ok(garrido(TB, cb, "copyh")))[0]

# 쌍 단위 절대 L1 (영상 점수의 평균) — p 와 copy 가 표적에서 얼마나 떨어져 있나
for cb in combos:
    out["abs_l1"][cb] = {m: float(np.mean(list(garrido(TE, cb, m).values()))) for m in ("p_full", "copyz_full", "p_causal", "copyz_causal")}

# 창 단위 j 표 (표준 vs 인과) — §2 희석 설명의 직접 검정.  j = floor(((F−1) − (s + 2C)) / 4), F = first_div_strict
for cb in ("skip2_w16", "skip2_w32"):
    acc = collections.defaultdict(lambda: collections.defaultdict(list)); cls = collections.defaultdict(list)
    for pi, p in enumerate(pairs):
        for key in [k for k in KE[p["pos"]] if k[1] == cb]:
            ki = (p["imp"],) + key[1:]
            if ki not in TE: continue
            _, _, s, C = key; wz = int(cb.split("_w")[1]); ntub = (wz - C) // 2
            j = int(np.floor(((int(p["first_div_strict"]) - 1) - (s + 2 * C)) / 4))
            if not (0 <= j < ntub): continue
            jb = "j0-1" if j <= 1 else ("j2-3" if j <= 3 else "j4+")
            for g in (p["group"], f'dir:{p["dir"]}'):
                kk = f"{g}|{jb}"
                for m in ("p_full", "p_causal", "copyz_full", "copyz_causal"):
                    dl = winmean(TE, ki, m) - winmean(TE, key, m)
                    acc[kk][m].append(1.0 if dl > 0 else (0.5 if dl == 0 else 0.0))
                cls[kk].append(p["scene"])
    for kk, d in sorted(acc.items()):
        cl = np.array(cls[kk])
        out["window_j"][f"{cb}|{kk}"] = {m: cl_boot(np.array(v), cl, 2000) for m, v in d.items()}

# H6d 다시: 이동+가림 재등장 튜블릿, tie = 0.5, scene CI (H6b 실행: p / copyz / copyh)
ov = np.load(AUX / "obj_vis.npz")
acc = collections.defaultdict(lambda: collections.defaultdict(list)); cls = collections.defaultdict(list)
for p in pairs:
    if p["group"] != "moving/occluded": continue
    vis = ov[p["pos"]]
    for key in [k for k in KB[p["pos"]] if k[1] in ("skip2_w16", "skip2_w32")]:
        ki = (p["imp"],) + key[1:]
        if ki not in TB: continue
        _, cb, s, C = key; e0 = s + 2 * C
        jev = int(np.floor(((int(p["first_div_strict"]) - 1) - e0) / 4))
        if jev < 0 or jev not in TB[key]: continue
        ctxvis = "ctxend_visible" if (vis[max(0, e0 - 2)] > 0 or vis[max(0, e0 - 4)] > 0) else "ctxend_hidden"
        db = "d0-2" if jev <= 2 else ("d3-5" if jev <= 5 else "d6+")
        for g in (f"{ctxvis}|{p['dir']}|{db}", f"{ctxvis}|{p['dir']}|all", f"{ctxvis}|all|{db}", f"all|{p['dir']}|{db}", "all|all|all"):
            for m in ("p", "copyz", "copyh"):
                dl = TB[ki][jev][m] - TB[key][jev][m]
                acc[g][m].append(1.0 if dl > 0 else (0.5 if dl == 0 else 0.0))
            cls[g].append(p["scene"])
for g, d in sorted(acc.items()):
    cl = np.array(cls[g])
    out["h6d_ties05"][g] = {m: cl_boot(np.array(v), cl, 2000) for m, v in d.items()} | \
        {"p_minus_copyz": cl_boot(np.array(d["p"]) - np.array(d["copyz"]), cl, 2000)}

# 튜블릿 단위 사건 전 / 분기 / 이후 (보이는 분기, skip2 두 칸) — 두 실행 (H6b, H6e) 을 **같은 정의** (tie = 0.5) 로, scene CI
out["tub"] = {}
for run, TT, KK, mets in (("h6b", TB, KB, ("p", "copyz", "copyh")), ("h6e", TE, KE, ("p_full", "copyz_full", "p_causal", "copyz_causal"))):
    acc = collections.defaultdict(lambda: collections.defaultdict(list)); cls = collections.defaultdict(list); nt = collections.Counter()
    for p in pairs:
        for key in [k for k in KK[p["pos"]] if k[1] in ("skip2_w16", "skip2_w32")]:
            ki = (p["imp"],) + key[1:]
            if ki not in TT: continue
            _, cb, s, C = key; e0 = s + 2 * C
            jev = int(np.floor(((int(p["first_div_strict"]) - 1) - e0) / 4)); ntub = len(TT[key])
            if not (0 <= jev < ntub): continue
            for j in TT[key]:
                eb = "pre" if j < jev else ("e0" if j == jev else "post")
                kk = f'{p["group"]}|{eb}'
                for m in mets:
                    dl = TT[ki][j][m] - TT[key][j][m]
                    acc[kk][m].append(1.0 if dl > 0 else (0.5 if dl == 0 else 0.0))
                    if dl != 0: nt[(kk, m)] += 1
                cls[kk].append(p["scene"])
    for kk, d in sorted(acc.items()):
        cl = np.array(cls[kk])
        out["tub"][f"{run}|{kk}"] = {m: cl_boot(np.array(v), cl, 2000) + [round(nt[(kk, m)] / len(v), 4)] for m, v in d.items()}
        if run == "h6b":
            out["tub"][f"{run}|{kk}"]["p_minus_copyz"] = cl_boot(np.array(d["p"]) - np.array(d["copyz"]), cl, 2000)
        else:
            out["tub"][f"{run}|{kk}"]["p_minus_copyz_full"] = cl_boot(np.array(d["p_full"]) - np.array(d["copyz_full"]), cl, 2000)
            out["tub"][f"{run}|{kk}"]["p_minus_copyz_causal"] = cl_boot(np.array(d["p_causal"]) - np.array(d["copyz_causal"]), cl, 2000)

OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "h6g_intphys1.json", "w"), indent=1, ensure_ascii=False)

# 요약 출력
for cb in combos:
    for tgt in ("full", "causal"):
        c = out["cells"][f"{cb}|{tgt}"]; d = out["discordant"][f"{cb}|{tgt}"]
        print(f"{cb:10s} {tgt:6s} p {c['p'][0]:6.2f}  copy {c['copy'][0]:6.2f}  p−copy {c['p_minus_copy'][0]:+6.2f} [{c['p_minus_copy'][1]:+.2f},{c['p_minus_copy'][2]:+.2f}]"
              f"  p−50 {c['p_minus_chance'][0]:+6.2f} [{c['p_minus_chance'][1]:+.2f},{c['p_minus_chance'][2]:+.2f}]  disc {d['p_only']}/{d['copy_only']}")
    print(f"{cb:10s} gap(causal)−gap(full) {out['cells'][f'{cb}|gap_causal_minus_gap_full']}  p full−causal {out['cells'][f'{cb}|p_full_minus_causal']}"
          f"  copy full−causal {out['cells'][f'{cb}|copyz_full_minus_causal']}")
print()
for k, v in out["groups"].items():
    print(f"{k:45s} p {v['p'][0]:6.1f} copy {v['copy'][0]:6.1f}  p−copy {v['p_minus_copy'][0]:+6.1f} [{v['p_minus_copy'][1]:+.1f},{v['p_minus_copy'][2]:+.1f}]"
          f"  p−50 {v['p_minus_chance'][0]:+6.1f} [{v['p_minus_chance'][1]:+.1f},{v['p_minus_chance'][2]:+.1f}]  n {v['p'][3]} disc {v['d_p_only']}/{v['d_copy_only']}")
print()
for cb, v in out["official_vs_h6e"].items():
    print(cb, {k: (x if k != "pair_flips" else len(x)) for k, x in v.items()}, v["pair_flips"])
print(out["abs_l1"])
print()
for k, v in out["window_j"].items():
    print(f"{k:40s} " + "  ".join(f"{m} {x[0]:5.1f} [{x[1]:.1f},{x[2]:.1f}]" for m, x in v.items()) + f"  n {v['p_full'][3]}")
print()
for k, v in out["tub"].items():
    print(f"{k:32s} " + "  ".join(f"{m} {x[0]:5.1f} [{x[1]:.1f},{x[2]:.1f}]" + (f" nz{x[5]:.3f}" if len(x) > 5 else "") for m, x in v.items()) + f"  n {list(v.values())[0][3]}")
print()
for g, v in out["h6d_ties05"].items():
    print(f"{g:40s} " + "  ".join(f"{m} {x[0]:5.1f} [{x[1]:.1f},{x[2]:.1f}]" for m, x in v.items()) + f"  n {v['p'][3]} sc {v['p'][4]}")
