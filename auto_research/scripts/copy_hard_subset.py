#!/usr/bin/env python3
"""복사 기준선이 못 푸는 부분집합 (copy-hard) 과 복사가 천장인 부분집합 (copy-ceiling) 에서 predictor 들의 정확도 — 2026-09-29.

    python auto_research/scripts/copy_hard_subset.py      # CPU, 몇 분 → exp_results/verify/copy_hard_subset.json

정의 (쌍 단위): o_m(pair) = 1[S_m(imp) > S_m(pos)] (동점 0.5). copy-hard = {pair : o_copy(pair) < 1} (복사가 틀리거나 동점),
copy-correct = {pair : o_copy = 1}. 칸 단위: copy-hard 칸 = 복사 정확도 ≤ 50 %, 천장 칸 = 복사 ≥ 95 %.
CI = cluster bootstrap (v11 block · IntPhys 1 scene · IntPhys 2 scene_index), B = 2000, 같은 resample 을 방법끼리 공유.
원자료: v11 = training_effects/v11score_vith_curves (배열) + IntPhysGenV11/exp_results/*ariel* per_block.json (하네스);
        IntPhys 1 = training_effects/ip1score_final · ip1score_ar_copyblk (배열) + Benchmarks per_window.json (Ariel 0927 팔);
        IntPhys 2 = Benchmarks/exp_results/intphys2/<tag>/per_video.csv (predictor · IP2_COPY 복사).
"""
from __future__ import annotations
import csv, json, sys, zlib
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
OUT = ROOT / "auto_research/exp_results/verify"; OUT.mkdir(parents=True, exist_ok=True)
B = 2000


def boot(v, cl, key):
    """cluster bootstrap 비율 (%). 같은 key → 같은 resample."""
    v = np.asarray(v, np.float64)
    if v.size == 0: return None
    u, inv = np.unique(np.asarray(cl), return_inverse=True)
    s = np.bincount(inv, v, len(u)); c = np.bincount(inv, None, len(u)).astype(np.float64)
    rng = np.random.default_rng([0, zlib.crc32(("|".join(map(str, u)) + key).encode())])
    idx = rng.integers(0, len(u), (B, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * v.mean(), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2), int(v.size), int(len(u))]


def outcome(sp, si):
    d = si - sp
    return np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))


# ============================================================================ v11
def v11():
    d = CACHE / "v11score_vith_curves"; meta = json.load(open(d / "meta.json"))
    tags = meta["pred_tags"]; l1 = np.load(d / "l1.npy"); cp = np.load(d / "copy.npy")
    S = {t: l1[:, k].mean(1) for k, t in enumerate(tags)}; S["copy"] = cp.mean(1)
    keep = ["release", "v11_e10", "pv1_e15", "ip1_e40", "ip2_e80", "ariel_ep43"]
    vid = meta["video_id"]; pos_of = {v: i for i, v in enumerate(vid)}
    # 하네스 per_video_surprise (Ariel 0927 팔 + AR)
    H = {}
    for tag, folder in (("ariel_prefix_ep45", "vith_ariel_prefix_ep45"), ("ariel_full_ep40", "vith_ariel_full_ep40"),
                        ("ariel_ar_ep38", "vith_ariel_ar_ep38"), ("ariel_ar_ep18", "vith_ariel_ar_ep18"), ("ariel_ep43_harness", "vith_ariel_bc_ep43")):
        f = ROOT / f"z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_split_test_{folder}/per_block.json"
        if f.exists():
            pv = json.load(open(f))["per_video_surprise"]; H[tag] = np.array([pv.get(v, np.nan) for v in vid], np.float64)
    S.update(H)
    # 쌍
    grp = defaultdict(dict)
    for i, (b, p, pl) in enumerate(zip(meta["block_id"], meta["pair_id"], meta["plausible"])):
        grp[(str(b), str(p))]["pos" if str(pl) == "1" else "imp"] = i
    pairs = [(k, g["pos"], g["imp"]) for k, g in grp.items() if "pos" in g and "imp" in g]
    pi = np.array([p for _, p, _ in pairs]); ii = np.array([i for _, _, i in pairs]); blk = np.array([k[0] for k, _, _ in pairs])
    # 속성 (pos 영상 기준): 타이밍 · 운동 · 위반 · vanish 방향 (role pos_obj = 물체→빈, pos_empty = 빈→물체)
    idx_rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/intphysgen_v11_split/index_test.csv"))}
    def attr(i):
        r = idx_rows[vid[i]]; t = r["occ_timing"] or "visible"; vio = r["violation_type"]
        if vio == "vanish": vio = "vanish→빈" if r["role"].startswith("pos_obj") else "vanish→물체"
        return t, r["motion"], vio
    A = np.array([attr(p) for p in pi]); tim, mot, vio = A[:, 0], A[:, 1], A[:, 2]
    O = {m: outcome(S[m][pi], S[m][ii]) for m in ["copy"] + keep + list(H)}
    valid = {m: np.isfinite(S[m][pi]) & np.isfinite(S[m][ii]) for m in O}
    res = dict(n_pair=len(pairs), methods=list(O), roles=sorted(set(r["role"] for r in idx_rows.values())))
    hard = O["copy"] < 1; corr = O["copy"] == 1
    res["subset_pair_level"] = {}
    for nm, m in (("all", np.ones(len(pairs), bool)), ("copy_hard(copy<1)", hard), ("copy_correct", corr)):
        res["subset_pair_level"][nm] = {k: boot(O[k][m & valid[k]], blk[m & valid[k]], f"v11|{nm}") for k in O}
    # copy-hard 쌍의 구성
    res["copy_hard_composition"] = {f"{t}|{mo}|{v}": int(((tim == t) & (mot == mo) & (vio == v) & hard).sum()) for t in ("visible", "early", "mid", "late") for mo in ("static", "moving_flat", "moving") for v in ("vanish→빈", "vanish→물체", "shape", "color")}
    res["copy_hard_by_violation"] = {v: {k: boot(O[k][hard & (vio == v) & valid[k]], blk[hard & (vio == v) & valid[k]], f"v11|hard|{v}") for k in O} for v in ("vanish→빈", "vanish→물체", "shape", "color")}
    res["copy_hard_by_motion"] = {mo: {k: boot(O[k][hard & (mot == mo) & valid[k]], blk[hard & (mot == mo) & valid[k]], f"v11|hard|{mo}") for k in O} for mo in ("static", "moving_flat", "moving")}
    res["copy_hard_by_timing"] = {t: {k: boot(O[k][hard & (tim == t) & valid[k]], blk[hard & (tim == t) & valid[k]], f"v11|hard|{t}") for k in O} for t in ("visible", "early", "mid", "late")}
    # 칸 단위
    cells = {}
    for t in ("visible", "early", "mid", "late"):
        for mo in ("static", "moving_flat", "moving"):
            for v in ("vanish→빈", "vanish→물체", "shape", "color"):
                m = (tim == t) & (mot == mo) & (vio == v)
                if m.sum() == 0: continue
                cells[f"{t}|{mo}|{v}"] = {k: boot(O[k][m & valid[k]], blk[m & valid[k]], f"v11|cell|{t}{mo}{v}") for k in O}
    res["cells"] = cells
    hard_cells = [c for c, r in cells.items() if r["copy"][0] <= 50]; ceil_cells = [c for c, r in cells.items() if r["copy"][0] >= 95]
    res["copy_hard_cells"] = hard_cells; res["copy_ceiling_cells"] = ceil_cells
    for nm, cl in (("copy_hard_cells_union", hard_cells), ("copy_ceiling_cells_union", ceil_cells)):
        m = np.zeros(len(pairs), bool)
        for c in cl:
            t, mo, v = c.split("|"); m |= (tim == t) & (mot == mo) & (vio == v)
        res[nm] = {k: boot(O[k][m & valid[k]], blk[m & valid[k]], f"v11|{nm}") for k in O}
    res["late_vanish_to_empty"] = {f"late|{mo}|vanish→빈": cells.get(f"late|{mo}|vanish→빈") for mo in ("static", "moving_flat", "moving")}
    res["agreement_with_copy"] = {k: round(float(np.mean((O[k] == O["copy"])[valid[k]])), 3) for k in O}
    # AR 공간 복사 (copy_blk) — v11score_vith_ar_copyblk 는 360 행만 done (부분). 그 행의 쌍에서만 AR ep18 vs copy_blk.
    d2 = CACHE / "v11score_vith_ar_copyblk"; m2 = json.load(open(d2 / "meta.json")); done2 = np.load(d2 / "done.npy").astype(bool)
    l12 = np.load(d2 / "l1.npy"); cb2 = np.load(d2 / "copy_blk.npy"); vid2 = m2["video_id"]; idx2 = {v: i for i, v in enumerate(vid2)}
    Sar, Scb = l12[:, 0].mean(1), cb2.mean(1)
    ok = np.array([done2[idx2[vid[p]]] and done2[idx2[vid[i]]] for p, i in zip(pi, ii)])
    if ok.sum():
        p2 = np.array([idx2[vid[p]] for p in pi[ok]]); i2 = np.array([idx2[vid[i]] for i in ii[ok]])
        o_ar = outcome(Sar[p2], Sar[i2]); o_cb = outcome(Scb[p2], Scb[i2]); hard_cb = o_cb < 1
        sub = {"n_pair": int(ok.sum()), "note": "copy_blk 는 360 영상만 추출됨 (부분) — 그 쌍에서만",
               "all": {"copy_blk": boot(o_cb, blk[ok], "v11cb|all"), "ariel_ar_ep18(같은 배열)": boot(o_ar, blk[ok], "v11cb|all"), "copy(표준)": boot(O["copy"][ok], blk[ok], "v11cb|all"), "release": boot(O["release"][ok], blk[ok], "v11cb|all")},
               "copy_blk_hard": {"copy_blk": boot(o_cb[hard_cb], blk[ok][hard_cb], "v11cb|hard"), "ariel_ar_ep18": boot(o_ar[hard_cb], blk[ok][hard_cb], "v11cb|hard"), "copy(표준)": boot(O["copy"][ok][hard_cb], blk[ok][hard_cb], "v11cb|hard"), "release": boot(O["release"][ok][hard_cb], blk[ok][hard_cb], "v11cb|hard")},
               "copy_std_hard": {"copy_blk": boot(o_cb[hard[ok]], blk[ok][hard[ok]], "v11cb|stdhard"), "ariel_ar_ep18": boot(o_ar[hard[ok]], blk[ok][hard[ok]], "v11cb|stdhard")},
               "agreement(copy_blk, copy_std)": round(float(np.mean(o_cb == O["copy"][ok])), 3)}
        res["ar_space_partial"] = sub
    return res


# ============================================================================ IntPhys 1
def ip1():
    import te_analyze_ip1 as TA
    man, arr, lay, src = TA.load_arrays(CACHE / "ip1score_final"); tags = list(man["tags"])
    man2, arr2, lay2, _ = TA.load_arrays(CACHE / "ip1score_ar_copyblk"); z2 = np.load(CACHE / "ip1score_ar_copyblk/arrays_v360.npz")
    assert list(arr["video_ids"]) == list(arr2["video_ids"])
    pairs = TA.load_pairs(); vidx = {v: i for i, v in enumerate(arr["video_ids"])}
    pi = np.array([vidx[p["pos"]] for p in pairs]); ii = np.array([vidx[p["imp"]] for p in pairs])
    sc = np.array([p["scene"] for p in pairs]); pr = np.array([p["pr"] for p in pairs]); mv = np.array([p["mv"] for p in pairs]); dr = np.array([p["dir"] for p in pairs])
    scorer = TA.Scorer(lay)
    G = {t: scorer.G(arr["p_std"][k]) for k, t in enumerate(tags)}; G["copy"] = scorer.G(arr["copy_std"])
    G["ariel_ar_ep18"] = scorer.G(arr2["p_std"][0]); G["copy_blk"] = scorer.G(z2["copy_blk"])
    # 하네스 (Ariel 0927 팔 · AR ep38): per_window → G
    for tag, model in (("ariel_prefix_ep45", "vith_ariel_prefix_ep45"), ("ariel_full_ep40", "vith_ariel_full_ep40"), ("ariel_ar_ep38", "vith_ariel_ar_ep38"), ("release_harness", "vith")):
        pw, dirs = TA.harness_pw(model)
        if not pw: continue
        g = TA.perwindow_G(pw)
        G[tag] = {cb: np.array([g[cb].get(v, np.nan) for v in arr["video_ids"]]) for cb in g}
    combos = ["skip2_w32", "skip2_w16", "skip5_w16"]
    res = dict(n_pair=len(pairs), methods=list(G), combos=combos, per_combo={})
    for cb in combos:
        O = {m: TA.outcomes(G[m][cb], pi, ii)[0] for m in G if cb in G[m]}
        V = {m: np.isfinite(G[m][cb][pi]) & np.isfinite(G[m][cb][ii]) for m in O}
        hard = O["copy"] == 0; corr = O["copy"] == 1; hard_blk = O["copy_blk"] == 0
        e = {"subset_pair_level": {}}
        for nm, m in (("all", np.ones(len(pairs), bool)), ("copy_hard(copy_std wrong)", hard), ("copy_correct", corr), ("copy_blk_hard(AR 공간)", hard_blk)):
            e["subset_pair_level"][nm] = {k: boot(O[k][m & V[k]], sc[m & V[k]], f"ip1|{cb}|{nm}") for k in O}
        e["copy_hard_composition"] = {f"{p}|{g}": int(((pr == p) & (mv == g) & hard).sum()) for p in ("O1", "O2", "O3") for g in TA.MV}
        e["copy_hard_by_principle"] = {p: {k: boot(O[k][hard & (pr == p) & V[k]], sc[hard & (pr == p) & V[k]], f"ip1|{cb}|hard|{p}") for k in O} for p in ("O1", "O2", "O3")}
        e["copy_hard_by_dir"] = {dn: {k: boot(O[k][hard & (dr == dn) & V[k]], sc[hard & (dr == dn) & V[k]], f"ip1|{cb}|hard|{dn}") for k in O} for dn in ("appear", "disappear")}
        cells = {}
        for p in ("O1", "O2", "O3"):
            for g in TA.MV:
                m = (pr == p) & (mv == g)
                if m.sum() == 0: continue
                cells[f"{p}|{g}"] = {k: boot(O[k][m & V[k]], sc[m & V[k]], f"ip1|{cb}|cell|{p}{g}") for k in O}
        e["cells"] = cells
        e["copy_hard_cells"] = [c for c, r in cells.items() if r["copy"][0] <= 50]; e["copy_ceiling_cells"] = [c for c, r in cells.items() if r["copy"][0] >= 95]
        e["agreement_with_copy"] = {k: round(float(np.mean((O[k] == O["copy"])[V[k]])), 3) for k in O}
        e["agreement_with_copy_blk"] = {k: round(float(np.mean((O[k] == O["copy_blk"])[V[k]])), 3) for k in O}
        res["per_combo"][cb] = e
    return res


# ============================================================================ IntPhys 2
def ip2():
    BX = ROOT / "z_research/Benchmarks/exp_results/intphys2"
    def load(tag):
        rows = list(csv.DictReader(open(BX / tag / "per_video.csv"))); out = defaultdict(dict); attr = {}
        for r in rows:
            C = int(r["context_length"]); key = (int(r["scene_index"]), int(r["pair_id"]))
            out[C].setdefault(key, {})["imp" if r["is_impossible"] == "True" else "pos"] = float(r["surprise_avg"])
            attr[key] = (r["condition"], r["difficulty"], r["camera"])
        return out, attr
    tags = {"copy_w48": "bench_intphys2_main_vith_copy_w48", "release_w48": "bench_intphys2_main_vith", "ariel_prefix_ep45_w48": "bench_intphys2_main_ariel_prefix_ep45_w48",
            "ariel_full_ep40_w48": "bench_intphys2_main_ariel_full_ep40_w48", "ariel_ar_ep38_w48": "bench_intphys2_main_ariel_ar_ep38_w48", "ariel_ar_ep18_w48": "bench_intphys2_main_ariel_ar_ep18_w48",
            "copy_w32": "bench_intphys2_main_vith_copy_w32", "release_w32": "bench_intphys2_main_vith_w32"}
    D = {}; AT = {}
    for k, t in tags.items():
        if (BX / t / "per_video.csv").exists(): D[k], a = load(t); AT.update(a)
    res = {"methods": list(D), "per_setting": {}}
    for win, C in (("w48", 24), ("w32", 24)):
        cm, rm = f"copy_{win}", f"release_{win}"
        if cm not in D or C not in D[cm]: continue
        keys = sorted(k for k, v in D[cm][C].items() if "pos" in v and "imp" in v)
        meths = [m for m in D if m.endswith(win) and C in D[m]]
        O = {}
        for m in meths:
            O[m] = np.array([outcome(D[m][C][k]["pos"], D[m][C][k]["imp"]) if k in D[m][C] and "pos" in D[m][C][k] and "imp" in D[m][C][k] else np.nan for k in keys])
        sc = np.array([k[0] for k in keys]); cond = np.array([AT[k][0] for k in keys]); cam = np.array([AT[k][2] for k in keys])
        hard = O[cm] < 1; corr = O[cm] == 1
        e = {"C": C, "n_pair": len(keys), "subset_pair_level": {}}
        for nm, m in (("all", np.ones(len(keys), bool)), ("copy_hard(copy<1)", hard), ("copy_correct", corr)):
            e["subset_pair_level"][nm] = {k: boot(O[k][m & np.isfinite(O[k])], sc[m & np.isfinite(O[k])], f"ip2|{win}|{nm}") for k in O}
        e["copy_hard_composition"] = {f"{c}|{ca}": int(((cond == c) & (cam == ca) & hard).sum()) for c in sorted(set(cond)) for ca in sorted(set(cam))}
        e["copy_hard_by_condition"] = {c: {k: boot(O[k][hard & (cond == c) & np.isfinite(O[k])], sc[hard & (cond == c) & np.isfinite(O[k])], f"ip2|{win}|hard|{c}") for k in O} for c in sorted(set(cond))}
        cells = {}
        for c in sorted(set(cond)):
            for ca in sorted(set(cam)):
                m = (cond == c) & (cam == ca)
                if m.sum() == 0: continue
                cells[f"{c}|{ca}"] = {k: boot(O[k][m & np.isfinite(O[k])], sc[m & np.isfinite(O[k])], f"ip2|{win}|cell|{c}{ca}") for k in O}
        e["cells"] = cells; e["copy_hard_cells"] = [c for c, r in cells.items() if r[cm][0] <= 50]; e["copy_ceiling_cells"] = [c for c, r in cells.items() if r[cm][0] >= 95]
        e["agreement_with_copy"] = {k: round(float(np.nanmean((O[k] == O[cm])[np.isfinite(O[k])])), 3) for k in O}
        # best-C 참고: copy 의 best C (42 for w48) 에서의 copy-hard 도
        res["per_setting"][f"{win}|C{C}"] = e
    return res


if __name__ == "__main__":
    out = {"v11": v11(), "intphys1": ip1(), "intphys2": ip2(), "definition": __doc__}
    json.dump(out, open(OUT / "copy_hard_subset.json", "w"), indent=1, ensure_ascii=False, default=float)
    print("saved", OUT / "copy_hard_subset.json")
