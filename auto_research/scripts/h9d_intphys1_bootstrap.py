#!/usr/bin/env python3
"""H9d — H9b (IntPhys 1 Garrido 격자에 frozen 개입) 결과에 CI 를 붙인다: 변형 − clean 의 장면 단위 bootstrap.

무엇: h9b_intphys1_intervene.py 가 쓴 ip1_intervene_r*.npz (창 × 미래 튜블릿 L1) 를 h9b_analyze.py 와 **같은 Garrido 규칙** 으로 채점하고,
      macro (O1/O2/O3 평균) · 원리 · 운동×가림 · 사라짐 방향 칸마다 Δ = acc(변형) − acc(clean) 의 95 % CI 를 낸다.
왜:  IntPhys 1 은 180 쌍 (원리당 60, 운동×가림 칸 15~60) 이라 한 쌍이 0.56~6.7 pt 다. CI 없이 칸 점수를 읽을 수 없다.
식:
  창 점수      s(v, 창) = mean_j mean|p_j − LN(h_full)_j|           (j = 창의 미래 튜블릿)
  video 점수   S(v) = mean_{시작점} min_{C} s(v, 시작점, C)          (h9b_analyze 와 같은 규칙)
  ok(pair)    = 1[S(imp) > S(pos)] + ½·1[같음];  macro = mean_{O1,O2,O3} acc(원리)
  bootstrap:  원리마다 장면 (scene, 쌍 2 개를 묶는 단위) 을 복원추출 (층화), 같은 가중치로 clean 과 변형을 계산 → Δ* 의 2.5 / 97.5 백분위.
검증: clean skip2_w32 macro = 88.89 (Garrido 기준값), clean 창 평균 = per_window.json.
실행: python h9d_intphys1_bootstrap.py   (NFS 의 작은 npz 만 읽는다, CPU 수 초)
"""
import collections, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H9 = ROOT / "auto_research/exp_results/h9"; AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
BENCH = ROOT / "z_research/Benchmarks/exp_results"
NB = 4000; SEED = 0
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}

parts = sorted(H9.glob("ip1_intervene_r*.npz")); z = np.load(parts[0])
combos, vids, VAR = list(z["combos"]), list(z["video_ids"]), list(z["variants"])
R = np.concatenate([np.load(f)["rows"] for f in parts])
T = collections.defaultdict(list)
for r in R:
    T[(vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]), int(r[5]))].append(r[6])
pw = {}
for w in (16, 32):
    for vid, rows in json.load(open(BENCH / f"intphys1_sliding__intphys1_dev_vith_w{w}/per_window.json"))["windows"].items():
        for cb, s, C, sur in rows: pw[(vid, cb, int(s), int(C))] = sur
d = [abs(np.mean(v) - pw[k[:4]]) for k, v in T.items() if k[4] == 0 and k[:4] in pw]
check = {"n_parts": len(parts), "clean_vs_per_window_max": float(max(d)), "clean_vs_per_window_median": float(np.median(d)), "n_windows": len(d)}
print(check)

V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
prin = np.array([p["principle"][:2] for p in pairs]); scene = np.array([p["scene"] for p in pairs])
lab = {}
lab["mv"] = np.array([f'{MO.get(p["label_motion"])}/{VI.get(p["label_vis"])}' for p in pairs])
dirs = []
for p in pairs:
    if p["principle"][:2] in ("O1", "O3"):
        a_, b_ = int(V[p["imp"]]["obj_visible_frames"]), int(V[p["pos"]]["obj_visible_frames"])
        dirs.append("disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal"))
    else:
        dirs.append("-")
lab["dir"] = np.array(dirs)

OK = {}
for cb in combos:
    for vi, vn in enumerate(VAR):
        vs = {}
        for (v, c, s, C, x), l in T.items():
            if c == cb and x == vi: vs.setdefault(v, {}).setdefault(s, []).append(np.mean(l))
        sc = {v: np.mean([min(cs) for cs in st.values()]) for v, st in vs.items()}
        OK[(cb, vn)] = np.array([1.0 if sc[p["imp"]] > sc[p["pos"]] else (0.5 if sc[p["imp"]] == sc[p["pos"]] else 0.0) for p in pairs])

# 기준 대조: 같은 규칙을 표준 per_window.json surprise 에 걸면 기준값 (w32 88.89 / w16 83.33) 이 나와야 한다 → clean 재계산과의 판정 차이
for cb in combos:
    vs = {}
    for (v, c, s_, C), sur in pw.items():
        if c == cb: vs.setdefault(v, {}).setdefault(s_, []).append(sur)
    scr = {v: np.mean([min(cs) for cs in st.values()]) for v, st in vs.items()}
    okr = np.array([1.0 if scr[p["imp"]] > scr[p["pos"]] else (0.5 if scr[p["imp"]] == scr[p["pos"]] else 0.0) for p in pairs])
    check[f"{cb}_standard_macro"] = round(100 * np.mean([okr[prin == pr].mean() for pr in ("O1", "O2", "O3")]), 2)
    check[f"{cb}_clean_flips_vs_standard"] = int((okr != OK[(cb, "clean")]).sum())
    check[f"{cb}_flipped_pairs"] = [f'{pairs[i]["pos"]}/{pairs[i]["imp"]}' for i in np.where(okr != OK[(cb, "clean")])[0]]
print(check)

# 층화 bootstrap 가중치: 원리마다 장면 복원추출 → 쌍 가중치
rng = np.random.default_rng(SEED)
Wp = np.zeros((NB, len(pairs)))
for pr in ("O1", "O2", "O3"):
    sc_ = np.unique(scene[prin == pr]); G = len(sc_)
    Ws = rng.multinomial(G, np.full(G, 1.0 / G), size=NB)            # (NB, G)
    pos_ = {s: k for k, s in enumerate(sc_)}
    for i in np.where(prin == pr)[0]:
        Wp[:, i] = Ws[:, pos_[scene[i]]]


def acc_b(ok, m):
    return (Wp[:, m] @ ok[m]) / Wp[:, m].sum(1)


def macro_b(ok):
    return np.mean([acc_b(ok, prin == pr) for pr in ("O1", "O2", "O3")], 0)


groups = {"macro": None} | {pr: prin == pr for pr in ("O1", "O2", "O3")} | \
         {f"mv:{g}": lab["mv"] == g for g in sorted(set(lab["mv"]))} | {f"dir:{g}": lab["dir"] == g for g in ("disappear", "appear", "equal")}
out = {"check": check, "n_boot": NB, "variants": VAR, "combos": combos, "res": {}}
for cb in combos:
    for gname, m in groups.items():
        if m is not None and not m.any(): continue
        e = {"n_pairs": int(len(pairs) if m is None else m.sum())}
        base = OK[(cb, "clean")]
        b0 = macro_b(base) if m is None else acc_b(base, m)
        for vn in VAR:
            ok = OK[(cb, vn)]
            a = 100 * (np.mean([ok[prin == pr].mean() for pr in ("O1", "O2", "O3")]) if m is None else ok[m].mean())
            bb = macro_b(ok) if m is None else acc_b(ok, m)
            lo, hi = np.percentile(bb, [2.5, 97.5])
            e[vn] = {"acc": round(a, 2), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
            if vn != "clean":
                db = bb - b0; dlo, dhi = np.percentile(db, [2.5, 97.5]); a0 = e["clean"]["acc"]
                e[vn] |= {"d": round(a - a0, 2), "d_lo": round(100 * dlo, 1), "d_hi": round(100 * dhi, 1)}
        out["res"][f"{cb}|{gname}"] = e
out["check"]["clean_skip2_w32_macro"] = out["res"]["skip2_w32|macro"]["clean"]["acc"]
json.dump(out, open(H9 / "h9d_intphys1_bootstrap.json", "w"), indent=1, ensure_ascii=False)
for k, e in out["res"].items():
    s = f"{k:28s} n={e['n_pairs']:3d}  clean {e['clean']['acc']:6.2f} [{e['clean']['lo']:5.1f},{e['clean']['hi']:5.1f}]"
    for vn in VAR[1:]:
        x = e[vn]; s += f" | {vn} {x['acc']:6.2f} Δ{x['d']:+6.2f} [{x['d_lo']:+5.1f},{x['d_hi']:+5.1f}]"
    print(s)
print("검증 clean skip2_w32 macro =", out["check"]["clean_skip2_w32_macro"], "(기준 88.89)")
