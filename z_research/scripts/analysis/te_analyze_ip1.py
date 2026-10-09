#!/usr/bin/env python3
"""te_analyze_ip1 — IntPhys 1 dev (Garrido A.8 격자) 에서 그룹 A predictor 마다 **예측 p 가 예측 없는 복사 기준선보다 나은가**
(칸 skip2_w32 · skip2_w16 · skip5_w16 × 표준 / 인과 표적 × O1/O2/O3 · 운동×가림 4 칸 × 사건 튜블릿).  CPU 전용.

입력 (읽기만 한다):
  /data2/.../cache/training_effects/ip1score_final/{manifest.json, shards/v*.npz | arrays_v360.npz}   (te_intphys1_score.py 추출)
      p_std / p_causal (K, 925) · copy_std / copy_causal (925,)  — 행 = (combo, start, C, j) 튜블릿 L1
  auto_research/_stage/IntPhys1_dev_by_scene/{pairs.csv, videos.csv, obj_vis.npz}   라벨 (scene, principle, 운동·가림, 분기 프레임)
  z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_<model>_w{16,32}/per_window.json   공식 하네스 (검증 · 그룹 B)
  auto_research/exp_results/h6/h6g_intphys1.json   릴리즈 기준값 (H6g)

정의 (te_intphys1_score.py docstring 과 같다; 기호 PROTOCOLS.md §1):
  튜블릿 L1_j(x) = mean_{토큰, 채널} |x_j − 표적_j|,   x ∈ {p^k (predictor k), copy = LN(z)[문맥 마지막 튜블릿]}
  표적: 표준 h = LN(target_encoder(창 전부))  /  인과 h^c_u = LN(target_encoder(창 앞 2(u+1) 장))[u]
  창 S(v, combo, t, C) = mean_j L1_j  →  G(v) = mean_t min_C S   (Filtered → AvgSurprise)
  쌍 정답 o = 1[G(imp) > G(pos)]  (garrido_rescore strict: 동점 = 오답).  tie = 0.5 판은 동점이 있을 때만 따로 보고.
  macro = mean_{O1,O2,O3} mean_pairs o.   p − copy = 같은 쌍 위 (o_p − o_copy) 의 평균 (pt).  Δ vs release 도 같은 쌍.
  CI: principle 층화 scene cluster bootstrap (macro), 칸 (principle · 운동×가림) 은 scene cluster bootstrap. B = 10,000,
      resample 은 cluster 집합마다 고정 (common random numbers) → 같은 칸의 방법들이 같은 resample 을 쓴다.
  튜블릿 (H6e/H6g `tub`, H6d 정의 그대로): skip2_w16 + skip2_w32 의 창 (t, C) 마다 e0 = t + 2C (첫 미래 원 프레임),
      사건 튜블릿 jev = floor(((first_div_strict − 1) − e0) / 4), 0 ≤ jev < 창 미래 튜블릿 수인 창만.
      j < jev = pre, j = jev = e0 (드러남 / 분기 튜블릿), j > jev = post.  튜블릿 정답 = 1[L1_j(imp) > L1_j(pos)].
      문맥 끝 가시 (H6d) = pos 영상 obj_vis[max(0, e0−2)] > 0 또는 obj_vis[max(0, e0−4)] > 0.
  copy 는 encoder 만의 함수 → 그룹 A 여섯 predictor 가 같은 copy 를 공유한다 (MODELS.md).

산출: z_research/TrainingEffects/ip1_copy/{IP1_COPY.md, ip1_copy.json, fig_*.png}

  python z_research/scripts/analysis/te_analyze_ip1.py                  # 전체 (B = 10,000)
  python z_research/scripts/analysis/te_analyze_ip1.py --B 2000         # 빠르게
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import garrido_rescore as GR  # noqa: E402  (채점 식 — 읽기 전용 재사용)
import te_intphys1_score as TE  # noqa: E402  (창 배치 Layout — 읽기 전용 재사용)

CACHE_ROOT = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
OUT = ROOT / "z_research/TrainingEffects/ip1_copy"
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
BX = ROOT / "z_research/Benchmarks/exp_results"
ZR = ROOT / "z_training/runs"
H6G = ROOT / "auto_research/exp_results/h6/h6g_intphys1.json"
N_VIDEOS = 360

FINAL = ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_ep43"]
HARNESS = {"release": "vith", "v11_e10": "vith_pft_v11_e10", "ip1_e40": "vith_pft_intphys1_e40",
           "ip2_e80": "vith_pft_intphys2_e80", "pv1_e15": "vith_pft_predv1_e15", "ariel_ep43": "vith_ariel_bc_ep43"}
DESC = {"release": "릴리즈 (학습 안 함)", "v11_e10": "post-FT v11 train 절반", "ip1_e40": "post-FT IntPhys1 train (**학습 도메인 안**)",
        "ip2_e80": "post-FT IntPhys2 254 clip", "pv1_e15": "post-FT Predictor_v1", "ariel_ep43": "scratch SSv2+K400 block-causal"}
COMBOS = ["skip2_w32", "skip2_w16", "skip5_w16"]
PRS = ["O1", "O2", "O3"]
MV = ["static/visible", "static/occluded", "moving/visible", "moving/occluded"]
MV_SHORT = {"static/visible": "s/vis", "static/occluded": "s/occ", "moving/visible": "m/vis", "moving/occluded": "m/occ"}
MO = {"정지": "static", "이동": "moving"}
VI = {"눈앞": "visible", "가려짐": "occluded"}
GROUP_B = ["vitb_k400_e240", "vitb_k400_e100", "vitb_synphys_30k", "vittiny_k400_5k", "vittiny_synphys_5k",
           "vittiny_k400", "vittiny_synphys_30k_e225", "vittiny_k400ssv2_30k_e225"]
# 학습 곡선 (하네스 p 만 있는 점): run → [(epoch, 하네스 이름 or 폴더)]
CURVE_RUNS = {
    "v11": [(10, ("bx", "vith_pft_v11_e10"))],
    "pv1": [(15, ("bx", "vith_pft_predv1_e15"))],
    "ip1": [(20, ("dir", ZR / "intphys1_postft/eval/intphys1_sliding__intphys1_dev_e20")), (40, ("bx", "vith_pft_intphys1_e40"))],
    "ip2": [(80, ("bx", "vith_pft_intphys2_e80"))],
    "ariel": [(e, ("bx", f"vith_ariel_bc_ep{e}")) for e in (5, 19, 30, 41, 43)],
}
CURVES_PRESET = ["release", "v11_e1", "v11_e2", "v11_e3", "v11_e5", "v11_e10", "pv1_e1", "pv1_e3", "pv1_e5", "pv1_e10", "pv1_e15",
                 "ip1_e5", "ip1_e10", "ip1_e20", "ip1_e40", "ip2_e10", "ip2_e40", "ip2_e80",
                 "ariel_ep5", "ariel_ep19", "ariel_ep30", "ariel_ep43"]
C_BLUE, C_ORANGE, C_GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def disp(t):
    """문서 표시 이름 — IntPhys1 로 학습한 predictor 는 '학습 도메인 안' 을 붙인다."""
    return f"{t} (학습 도메인 안)" if t.startswith("ip1") else t


def figl(t):
    return f"{t} (in-domain)" if t.startswith("ip1") else t
C_GRID, C_GRAY, C_DARK = "#e4e4e4", "#9a9a9a", "#222222"


# ============================================================================ 입력
def load_arrays(cache: Path):
    """shard 를 메모리에서 쌓는다 (캐시 폴더에 아무것도 쓰지 않는다). 병합본이 있으면 그걸 읽는다."""
    man = json.load(open(cache / "manifest.json"))
    shards = sorted((cache / "shards").glob("v*.npz"))
    mg = cache / f"arrays_v{len(shards)}.npz"
    if mg.exists():
        z = np.load(mg)
        arr = {k: z[k] for k in ("p_std", "p_causal", "copy_std", "copy_causal", "video_ids")}
        src = str(mg)
    else:
        P, PC, CS, CC, V = [], [], [], [], []
        for f in shards:
            z = np.load(f)
            assert list(z["tags"]) == man["tags"], f
            P.append(z["p_std"]); PC.append(z["p_causal"]); CS.append(z["copy_std"]); CC.append(z["copy_causal"])
            V.append(str(z["video_id"]))
        arr = dict(p_std=np.stack(P, 1), p_causal=np.stack(PC, 1), copy_std=np.stack(CS), copy_causal=np.stack(CC),
                   video_ids=np.array(V))
        src = f"{cache}/shards ({len(shards)} 개)"
    lay = TE.Layout(dict(data=man["data"], surprise=man["surprise"], model=man["model"]))
    assert lay.fingerprint() == man["layout"]["fp"], "창 배치 지문 불일치"
    arr["video_ids"] = np.array([str(v) for v in arr["video_ids"]])
    return man, arr, lay, src


def load_pairs():
    pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
    V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
    for p in pairs:
        p["mv"] = f'{MO[p["label_motion"]]}/{VI[p["label_vis"]]}'
        p["pr"] = p["principle"][:2]
        d = "none"
        if p["pr"] in ("O1", "O3"):
            a_, b_ = int(V[p["imp"]]["obj_visible_frames"]), int(V[p["pos"]]["obj_visible_frames"])
            d = "disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal")
        p["dir"] = d
    return pairs


def check_pairs_vs_index(pairs, man):
    """pairs.csv 의 (pos, imp) 가 채점기 쌍 규칙 (block_id, pair_id; plausible 1/0) 과 같은지."""
    meta = TE.index_meta(man["data"])
    for p in pairs:
        a, b = meta[p["pos"]], meta[p["imp"]]
        assert GR.pair_key(a) == GR.pair_key(b), p
        assert int(a["plausible"]) == 1 and int(b["plausible"]) == 0, p
        assert a["block_type"] == p["pr"], p
    keys = collections.Counter(GR.pair_key(meta[p["pos"]]) for p in pairs)
    assert len(keys) == len(pairs) == 180
    return meta


# ============================================================================ 채점
class Scorer:
    """창 S → combo 마다 G(v) = mean_t min_C S."""

    def __init__(self, lay):
        self.lay = lay
        wc = np.array([k[0] for k in lay.win_key])
        self.M = {}
        for cb in lay.names:
            w = np.nonzero(wc == cb)[0]
            st = sorted({lay.win_key[i][1] for i in w}); Cs = sorted({lay.win_key[i][2] for i in w})
            pos = {(lay.win_key[i][1], lay.win_key[i][2]): i for i in w}
            self.M[cb] = np.array([[pos[(s, C)] for C in Cs] for s in st])        # (nS, nC) 창 번호

    def S(self, vals):
        v = np.asarray(vals, np.float64)
        return np.add.reduceat(v, self.lay.win_start, axis=-1) / self.lay.win_len

    def G(self, vals) -> dict:
        s = self.S(vals)
        return {cb: s[..., M].min(-1).mean(-1) for cb, M in self.M.items()}


def outcomes(G, pi, ii):
    d = G[..., ii] - G[..., pi]
    strict = (d > 0).astype(np.float64)
    return strict, strict + 0.5 * (d == 0), d


class Boot:
    """cluster bootstrap. resample 은 cluster 집합 (라벨 튜플) 마다 고정 → 같은 칸의 방법끼리 같은 resample."""

    def __init__(self, B: int):
        self.B = B
        self.cache = {}

    def _idx(self, labels):
        key = "\x1f".join(labels)
        if key not in self.cache:
            rng = np.random.default_rng([0, zlib.crc32(key.encode())])
            self.cache[key] = rng.integers(0, len(labels), (self.B, len(labels)))
        return self.cache[key]

    def _parts(self, v, cl):
        u, inv = np.unique(cl, return_inverse=True)
        return u, np.bincount(inv, v, len(u)), np.bincount(inv, None, len(u)).astype(np.float64)

    def cluster(self, v, cl):
        v = np.asarray(v, np.float64)
        if v.size == 0:
            return None
        u, s, c = self._parts(v, cl)
        idx = self._idx(tuple(u))
        bs = s[idx].sum(1) / c[idx].sum(1)
        return [100 * v.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), int(v.size), int(len(u))]

    def macro(self, v, cl, strata):
        """principle 층화 scene bootstrap: macro = strata 평균의 평균."""
        v = np.asarray(v, np.float64)
        est, bs, n_st = [], np.zeros(self.B), 0
        for g in sorted(set(strata)):
            m = strata == g
            u, s, c = self._parts(v[m], cl[m])
            idx = self._idx(tuple(u))
            bs += s[idx].sum(1) / c[idx].sum(1)
            est.append(v[m].mean()); n_st += 1
        bs /= n_st
        return [100 * np.mean(est), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), int(v.size),
                int(len(set(cl))), n_st]


def excl0(ci):
    return ci is not None and (ci[1] > 0 or ci[2] < 0)


def fci(ci, sign=False, d=1):
    if ci is None:
        return "—"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(ci[0])} [{f.format(ci[1])}, {f.format(ci[2])}]"


# ============================================================================ 분석 본체
def pair_level(G_all, P, boot):
    """G_all[(method, tgt)][combo] (V,) → 칸별 정답률 · p − copy · Δ vs release."""
    pi, ii = P["pi"], P["ii"]
    sc, pr, mv = P["scene"], P["pr"], P["mv"]
    O = {}
    for key, Gd in G_all.items():
        for cb, G in Gd.items():
            O[key + (cb,)] = outcomes(G, pi, ii)
    subsets = {**{p_: pr == p_ for p_ in PRS}, **{g: mv == g for g in MV}, "moving(all)": np.char.startswith(mv.astype(str), "moving")}
    if "dir" in P:                                       # 정정 2026-09-25: A/B 방향 (appear / disappear) 을 쌍 단위로 병기 (CLAUDE.md §1-7)
        dr = P["dir"]
        for dn in ("appear", "disappear"):
            subsets[f"dir={dn}"] = dr == dn
            for p_ in ("O1", "O3"):
                subsets[f"{p_}|dir={dn}"] = (pr == p_) & (dr == dn)
    res = {}
    methods = sorted({k[0] for k in O}, key=lambda m: (m == "copy", FINAL.index(m) if m in FINAL else 99, m))
    for m in methods:
        for tgt in ("std", "causal"):
            for cb in COMBOS:
                if (m, tgt, cb) not in O:
                    continue
                o, oh, d = O[(m, tgt, cb)]
                e = {"macro": boot.macro(o, sc, pr), "n_pair": int(o.size), "n_ties": int((d == 0).sum()),
                     "G_mean": float(np.mean(G_all[(m, tgt)][cb])),
                     "cells": {k: boot.cluster(o[s], sc[s]) for k, s in subsets.items() if s.any()}}
                if e["n_ties"]:
                    e["macro_tie05"] = boot.macro(oh, sc, pr)
                refs = [("copy", "minus_copy")] if m != "copy" else []
                if m not in ("copy", "release") and ("release", tgt, cb) in O:
                    refs.append(("release", "minus_release"))
                for ref, nm in refs:
                    r, rh, rd = O[(ref, tgt, cb)]
                    dd = o - r
                    e[nm] = {"macro": boot.macro(dd, sc, pr),
                             "cells": {k: boot.cluster(dd[s], sc[s]) for k, s in subsets.items() if s.any()},
                             "discordant": {"only_" + m: int(((o == 1) & (r == 0)).sum()), "only_" + ref: int(((o == 0) & (r == 1)).sum())}}
                    if e["n_ties"] or (rd == 0).any():
                        e[nm]["macro_tie05"] = boot.macro(oh - rh, sc, pr)
                if m != "copy":
                    e["minus_chance"] = boot.macro(o - 0.5, sc, pr)
                res[f"{m}|{tgt}|{cb}"] = e
    return res, O


def tubelet_level(arr, lay, man, P, pairs_used, boot, fcol="first_div_strict", drop_early=False, groups=None, drop_pairs=None, flag_out=None):
    """H6e/H6g `tub` 와 H6d 정의: skip2 두 칸, 사건 튜블릿 기준 pre / e0 / post, 방법마다 튜블릿 정답률.
    정정 2026-09-25: fcol = 사건 프레임 열 (first_div_sensitive 로 다시 bin 가능), drop_early = PNG 가
    first_div_strict 보다 먼저 갈리는 쌍 (first_div_sensitive < first_div_strict) 을 뺀다, groups = 운동×가림 칸 제한,
    drop_pairs = 뺄 (pos, imp) 집합, flag_out = 인과 표적 pre 에서 copy 가 동점이 아닌 튜블릿을 가진 쌍을 모을 list."""
    ov = np.load(AUX / "obj_vis.npz")
    rows = lay.rows
    names = lay.names
    Wof = {cb["name"]: cb["W"] for cb in lay.combos}
    sk2 = np.isin(np.array(names)[rows[:, 0]], ["skip2_w16", "skip2_w32"])
    rr = np.nonzero(sk2)[0]
    st, C, j = rows[rr, 1].astype(np.int64), rows[rr, 2].astype(np.int64), rows[rr, 3].astype(np.int64)
    W = np.array([Wof[names[c]] for c in rows[rr, 0]])
    ntub = W // 2 - C // 2
    e0 = st + 2 * C
    tags = man["tags"]
    meth = [(t, "std") for t in tags] + [(t, "causal") for t in tags] + [("copy", "std"), ("copy", "causal")]

    def vals(m, tgt, vi):
        return (arr[f"copy_{tgt}"][vi] if m == "copy" else arr[f"p_{tgt}"][tags.index(m)][vi])[rr]

    D = {mt: [] for mt in meth}
    lab = collections.defaultdict(list)
    for p, vp, vi in pairs_used:
        if drop_early and int(p["first_div_sensitive"]) < int(p["first_div_strict"]):
            continue
        if groups is not None and p["mv"] not in groups:
            continue
        if drop_pairs is not None and (p["pos"], p["imp"]) in drop_pairs:
            continue
        F = int(p[fcol])
        jev = np.floor(((F - 1) - e0) / 4).astype(np.int64)
        ok = (jev >= 0) & (jev < ntub)
        if not ok.any():
            continue
        b = np.where(j[ok] < jev[ok], "pre", np.where(j[ok] == jev[ok], "e0", "post"))
        vis = ov[p["pos"]]
        e0k = e0[ok]
        cv = (vis[np.maximum(0, e0k - 2)] > 0) | (vis[np.maximum(0, e0k - 4)] > 0)
        n = int(ok.sum())
        lab["bin"].append(b); lab["ctx"].append(np.where(cv, "ctxend_visible", "ctxend_hidden"))
        lab["mv"].append(np.full(n, p["mv"])); lab["dir"].append(np.full(n, p["dir"])); lab["scene"].append(np.full(n, p["scene"]))
        for mt in meth:
            D[mt].append(vals(*mt, vi)[ok] - vals(*mt, vp)[ok])
        if flag_out is not None and np.any((b == "pre") & (D[("copy", "causal")][-1] != 0)):
            flag_out.append((p["pos"], p["imp"], p["mv"], p["scene"]))
    L = {k: np.concatenate(v) for k, v in lab.items()}
    D = {k: np.concatenate(v) for k, v in D.items()}
    sel = {}
    for g in (MV if groups is None else groups):
        for bn in ("pre", "e0", "post"):
            sel[f"{g}|{bn}"] = (L["mv"] == g) & (L["bin"] == bn)
    mo = (L["mv"] == "moving/occluded") & (L["bin"] == "e0")
    for cvn in ("ctxend_visible", "ctxend_hidden"):
        sel[f"moving/occluded|e0|{cvn}"] = mo & (L["ctx"] == cvn)
    for dn in ("appear", "disappear", "equal", "none"):
        sel[f"moving/occluded|e0|dir={dn}"] = mo & (L["dir"] == dn)
    res = {}
    for name, s in sel.items():
        if not s.any():
            continue
        cl = L["scene"][s]
        ent = {"n_tubelets": int(s.sum()), "n_scene": int(len(set(cl)))}
        for m, tgt in meth:
            d = D[(m, tgt)][s]
            o = (d > 0).astype(np.float64); oh = o + 0.5 * (d == 0)
            e = {"acc": boot.cluster(o, cl), "nonzero": float((d != 0).mean())}
            if (d == 0).any():
                e["acc_tie05"] = boot.cluster(oh, cl)
            if m != "copy":
                dc = D[("copy", tgt)][s]
                oc = (dc > 0).astype(np.float64); och = oc + 0.5 * (dc == 0)
                e["minus_copy"] = boot.cluster(o - oc, cl)
                if (d == 0).any() or (dc == 0).any():
                    e["minus_copy_tie05"] = boot.cluster(oh - och, cl)
                if m != "release":
                    dr = D[("release", tgt)][s]
                    orl = (dr > 0).astype(np.float64)
                    e["minus_release"] = boot.cluster(o - orl, cl)
            ent[f"{m}|{tgt}"] = e
        res[name] = ent
    return res


def perwindow_G(pw, vids_keep=None):
    """하네스 per_window → {combo: {vid: G}} (Filtered, AvgSurprise) — garrido_rescore.score_table 그대로."""
    out = {}
    for cb in sorted({w[0] for rec in pw.values() for w in rec}):
        vids, _, _, a = GR.score_table(pw, cb)
        g = np.nanmean(np.nanmin(a, axis=2), axis=1)
        out[cb] = {v: float(x) for v, x in zip(vids, g) if vids_keep is None or v in vids_keep}
    return out


def harness_pw(model):
    dirs = [BX / f"intphys1_sliding__intphys1_dev_{model}_w{w}" for w in (16, 32)]
    dirs = [d for d in dirs if (d / "per_window.json").exists()]
    pw = {}
    for d in dirs:
        for v, rows in json.load(open(d / "per_window.json"))["windows"].items():
            pw.setdefault(v, []).extend(rows)
    return pw, dirs


def validation(man, arr, lay, scorer, P, pairs_used, meta):
    """arrays 의 표준 표적 Filtered macro vs 공식 하네스 per_window (Benchmarks) — 칸마다, 뒤집힌 쌍 목록."""
    vids = list(arr["video_ids"])
    vix = {v: i for i, v in enumerate(vids)}
    res = {}
    S_all = scorer.S(arr["p_std"])                        # (K, V, nwin)
    for k, tag in enumerate(man["tags"]):
        pw, dirs = harness_pw(HARNESS[tag])
        if not pw:
            continue
        Gh = perwindow_G(pw, set(vids))
        Go = scorer.G(arr["p_std"][k])
        ent = {"harness_dirs": [str(d.relative_to(ROOT)) for d in dirs]}
        coll = GR.collect(BX, "intphys1_dev", HARNESS[tag])
        for cb in COMBOS:
            if cb not in Gh:
                continue
            ref = {(v, int(s), int(C)): float(x) for v, rows in pw.items() if v in vix for c, s, C, x in rows if c == cb}
            ours = {}
            for wi, (c, s, C) in enumerate(lay.win_key):
                if c == cb:
                    for v in vix:
                        ours[(v, s, C)] = S_all[k, vix[v], wi]
            common = [q for q in ref if q in ours]
            dif = np.abs(np.array([ours[q] - ref[q] for q in common]))
            oo = [1.0 if Go[cb][vi] > Go[cb][vp] else 0.0 for p, vp, vi in pairs_used]
            oh = [1.0 if Gh[cb][p["imp"]] > Gh[cb][p["pos"]] else 0.0 for p, vp, vi in pairs_used]
            flips = [{"pos": p["pos"], "imp": p["imp"], "pr": p["pr"], "mv": p["mv"], "ours": oo[i], "harness": oh[i],
                      "margin_ours": float(Go[cb][vi] - Go[cb][vp]), "margin_harness": Gh[cb][p["imp"]] - Gh[cb][p["pos"]]}
                     for i, (p, vp, vi) in enumerate(pairs_used) if oo[i] != oh[i]]
            pr = P["pr"]
            mac = lambda o: 100 * np.mean([np.mean(np.array(o)[pr == g]) for g in PRS if (pr == g).any()])  # noqa: E731
            e = {"ours_macro": mac(oo), "harness_macro": mac(oh), "n_pairs": len(oo), "flips": flips,
                 "n_windows": len(common), "win_abs_median": float(np.median(dif)), "win_abs_max": float(dif.max())}
            if coll and cb in coll["combos"] and len(oo) == 180:
                e["garrido_rescore_macro"] = GR._macro(coll["combos"][cb]["reductions"], ["filtered"])[0]
            ent[cb] = e
        if coll:
            ent["harness_best"] = {"combo": coll["best"]["combo"], "macro": coll["best"]["accuracy"]}
        res[tag] = ent
    return res


def explain_ties(O, lay, man, pairs_used):
    """쌍 단위 동점 (G(pos) == G(imp)) 이 있으면: 두 영상의 PNG 가 어느 원 프레임에서 다른지, 그 칸의 창들이 마지막으로 쓰는 원 프레임은 몇인지."""
    from PIL import Image
    d = man["data"]
    meta = TE.index_meta(d)
    last = {cb["name"]: int(max(max(f) for f in cb["frames"])) for cb in lay.combos}
    ties = collections.defaultdict(set)
    for (m, tgt, cb), (o, oh, dd) in O.items():
        for i in np.nonzero(dd == 0)[0]:
            ties[(pairs_used[i][0]["pos"], pairs_used[i][0]["imp"])].add(f"{m}|{tgt}|{cb}")
    out = []
    for (pv, iv), where in sorted(ties.items()):
        def fr(v, k):
            r = meta[v]
            return np.asarray(Image.open(Path(d["frames_root"]) / d["frames_pattern"].format(
                block=r["block"], quadruplet=r["quadruplet"], run=r["run"], frame=k + int(d.get("frames_start", 1))))).astype(np.int16)
        diff = [k for k in range(int(d["n_frames"])) if np.abs(fr(pv, k) - fr(iv, k)).max() > 0]
        cbs = sorted({w.split("|")[2] for w in where})
        out.append({"pos": pv, "imp": iv, "n_methods_tied": len(where), "combos": cbs,
                    "frames_differ": [min(diff), max(diff)] if diff else None, "n_frames_differ": len(diff),
                    "last_raw_frame_used": {cb: last[cb] for cb in cbs}})
    return out


def builtin_check(cells, n_videos):
    """te_intphys1_score.py --score 의 칸 macro (같은 배열 · garrido_rescore 함수) 와 이 스크립트의 strict macro 가 같은가."""
    d = ROOT / "z_research/TrainingEffects/ip1score"
    fs = sorted(d.glob("ip1score_final_scores*.json"), key=lambda f: f.stat().st_mtime)
    fs = [f for f in fs if json.load(open(f)).get("n_videos") == n_videos]
    if not fs:
        return {"status": f"n_videos {n_videos} 인 내장 채점 json 없음"}
    b = json.load(open(fs[-1]))
    rows, mx = [], 0.0
    for name, bt in b["cells"].items():
        for tgt, c in bt.items():
            for cb, e in c["combos"].items():
                ours = cells.get(f"{name}|{tgt}|{cb}")
                if ours is None:
                    continue
                dd = abs(ours["macro"][0] - e["macro"])
                mx = max(mx, dd)
                rows.append([name, tgt, cb, e["macro"], ours["macro"][0]])
    return {"file": str(fs[-1].relative_to(ROOT)), "n": len(rows), "max_abs_diff": mx, "rows": rows}


def release_vs_h6g(cells, tub):
    """릴리즈 · copy 칸을 H6g (다른 실행, tie = 0.5) 와 나란히."""
    if not H6G.exists():
        return {}
    h = json.load(open(H6G))
    out = {}
    for cb in COMBOS:
        for tgt, ht in (("std", "full"), ("causal", "causal")):
            hc = h["cells"].get(f"{cb}|{ht}")
            c = cells.get(f"release|{tgt}|{cb}")
            cc = cells.get(f"copy|{tgt}|{cb}")
            if hc is None or c is None:
                continue
            out[f"{cb}|{tgt}"] = {"p_ours": c["macro"][0], "p_h6g": hc["p"][0], "copy_ours": cc["macro"][0], "copy_h6g": hc["copy"][0],
                                  "p_minus_copy_ours": c["minus_copy"]["macro"][:3], "p_minus_copy_h6g": hc["p_minus_copy"][:3]}
    for bn in ("pre", "e0", "post"):
        t = tub.get(f"moving/occluded|{bn}")
        ht = h["tub"].get(f"h6e|moving/occluded|{bn}")
        if t and ht:
            g = lambda e: (e.get("acc_tie05") or e["acc"])[0]  # noqa: E731
            out[f"tub|moving/occluded|{bn}"] = {"p_std_ours_tie05": g(t["release|std"]), "p_std_h6g": ht["p_full"][0],
                                                "copy_std_ours_tie05": g(t["copy|std"]), "copy_std_h6g": ht["copyz_full"][0],
                                                "p_causal_ours_tie05": g(t["release|causal"]), "p_causal_h6g": ht["p_causal"][0],
                                                "copy_causal_ours_tie05": g(t["copy|causal"]), "copy_causal_h6g": ht["copyz_causal"][0]}
    return out


def harness_table(models, P, pairs, boot, copy_O=None):
    """하네스 per_window 만으로 (그룹 B · 그룹 A 대조): 칸마다 macro [CI], O1/O2/O3, 운동×가림."""
    res = {}
    sc, pr, mv = P["scene"], P["pr"], P["mv"]
    for name in models:
        pw, dirs = harness_pw(name)
        if not pw:
            res[name] = {"status": "per_window 없음 (IntPhys1 미실행)"}
            continue
        Gh = perwindow_G(pw)
        ent = {"dirs": [str(d.relative_to(ROOT)) for d in dirs], "n_videos": len(pw)}
        coll = GR.collect(BX, "intphys1_dev", name)
        for cb in COMBOS:
            if cb not in Gh or any(p["pos"] not in Gh[cb] or p["imp"] not in Gh[cb] for p in pairs):
                continue
            d = np.array([Gh[cb][p["imp"]] - Gh[cb][p["pos"]] for p in pairs])
            o = (d > 0).astype(np.float64)
            e = {"macro": boot.macro(o, sc, pr), "n_ties": int((d == 0).sum()),
                 "cells": {**{g: boot.cluster(o[pr == g], sc[pr == g]) for g in PRS},
                           **{g: boot.cluster(o[mv == g], sc[mv == g]) for g in MV}}}
            if coll:
                e["garrido_rescore_macro"] = GR._macro(coll["combos"][cb]["reductions"], ["filtered"])[0]
            ent[cb] = e
        if coll:
            ent["best"] = {"combo": coll["best"]["combo"], "macro": coll["best"]["accuracy"]}
        res[name] = ent
    return res


def curves(P, pairs, boot, O_final, final_arr=None):
    """학습 곡선. ip1score_curves 가 완성돼 있으면 그 배열로, 아니면 하네스 p (+ final 배열의 copy) 로 있는 점만."""
    cdir = CACHE_ROOT / "ip1score_curves"
    status = {"cache": str(cdir)}
    n = len(list((cdir / "shards").glob("v*.npz"))) if (cdir / "shards").exists() else 0
    status["n_shards"] = n
    sc, pr = P["scene"], P["pr"]
    out = {"status": status, "points": {}}
    if n >= N_VIDEOS:
        man, arr, lay, src = load_arrays(cdir)
        scorer = Scorer(lay)
        vix = {v: i for i, v in enumerate(arr["video_ids"])}
        pi = np.array([vix[p["pos"]] for p in pairs]); ii = np.array([vix[p["imp"]] for p in pairs])
        Gc = {tgt: scorer.G(arr[f"copy_{tgt}"]) for tgt in ("std", "causal")}
        status.update(source="arrays", arrays=src, tags=list(man["tags"]))
        if final_arr is not None:                          # final 과 겹치는 태그 · copy 가 같은 값인가 (결정론)
            fv = {v: i for i, v in enumerate(final_arr["video_ids"])}
            order = np.array([fv[v] for v in arr["video_ids"]])
            chk = {}
            for tgt in ("std", "causal"):
                chk[f"copy_{tgt}"] = float(np.abs(arr[f"copy_{tgt}"] - final_arr[f"copy_{tgt}"][order]).max())
                for k, t in enumerate(man["tags"]):
                    if t in FINAL:
                        chk[f"{t}|{tgt}"] = float(np.abs(arr[f"p_{tgt}"][k] - final_arr[f"p_{tgt}"][FINAL.index(t)][order]).max())
            status["final_overlap_max_abs"] = chk
        OS = {}
        for k, tag in enumerate(man["tags"]):
            for tgt in ("std", "causal"):
                G = scorer.G(arr[f"p_{tgt}"][k])
                for cb in COMBOS:
                    o = outcomes(G[cb], pi, ii)[0]; oc = outcomes(Gc[tgt][cb], pi, ii)[0]
                    OS[(tag, tgt, cb)] = o
                    out["points"][f"{tag}|{tgt}|{cb}"] = {"macro": boot.macro(o, sc, pr), "minus_copy": boot.macro(o - oc, sc, pr)}
        for tgt in ("std", "causal"):
            for cb in COMBOS:
                out["points"][f"copy|{tgt}|{cb}"] = {"macro": boot.macro(outcomes(Gc[tgt][cb], pi, ii)[0], sc, pr)}
        # 정정 2026-09-25: 체크포인트 사이 변화를 같은 쌍 위의 paired CI 로 (release→첫 점, 첫 점→끝 점)
        out["steps"] = {}
        for run in ("v11", "pv1", "ip1", "ip2", "ariel"):
            ks = sorted([t for t in man["tags"] if t.startswith(run + "_e")],
                        key=lambda s_: int("".join(ch for ch in s_.rsplit("_e", 1)[1] if ch.isdigit())))
            if len(ks) < 2:
                continue
            for tgt in ("std", "causal"):
                for cb in COMBOS:
                    segs = ([("release", ks[0])] if run != "ariel" and ("release", tgt, cb) in OS else []) + [(ks[0], ks[-1])]
                    for a_, b_ in segs:
                        oa, ob = OS[(a_, tgt, cb)], OS[(b_, tgt, cb)]
                        out["steps"][f"{run}|{tgt}|{cb}|{a_}->{b_}"] = {
                            "delta": boot.macro(ob - oa, sc, pr),
                            "discordant": {"only_" + b_: int(((ob == 1) & (oa == 0)).sum()), "only_" + a_: int(((ob == 0) & (oa == 1)).sum())}}
        return out
    status["source"] = "harness p (Benchmarks / z_training eval per_window) + copy 는 final 배열 (표준 표적)"
    pts = {"release": ("bx", "vith")}
    for run, lst in CURVE_RUNS.items():
        for ep, src in lst:
            pts[f"{run}_ep{ep}" if run == "ariel" else f"{run}_e{ep}"] = src      # curves preset 태그와 같은 이름
    for tag, (kind, where) in pts.items():
        if kind == "bx":
            pw, _ = harness_pw(where)
        else:
            pw = json.load(open(where / "per_window.json"))["windows"] if (where / "per_window.json").exists() else {}
        if not pw:
            continue
        Gh = perwindow_G(pw)
        for cb in COMBOS:
            if cb not in Gh:
                continue
            d = np.array([Gh[cb][p["imp"]] - Gh[cb][p["pos"]] for p in pairs])
            o = (d > 0).astype(np.float64)
            e = {"macro": boot.macro(o, sc, pr), "source": str(where)}
            if O_final is not None and ("copy", "std", cb) in O_final:
                e["minus_copy"] = boot.macro(o - O_final[("copy", "std", cb)][0], sc, pr)
            out["points"][f"{tag}|std|{cb}"] = e
    if O_final is not None:
        for cb in COMBOS:
            if ("copy", "std", cb) in O_final:
                out["points"][f"copy|std|{cb}"] = {"macro": boot.macro(O_final[("copy", "std", cb)][0], sc, pr)}
    return out


# ============================================================================ 그림
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#888888",
                         "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
                         "axes.labelcolor": "#333333", "axes.titlesize": 8.5, "legend.frameon": False,
                         "legend.fontsize": 7.5, "savefig.dpi": 170})
    return plt


def plabel(ax, i, dy=-26):
    ax.annotate(f"({'abcdefghijklmnop'[i]})", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", ha="center", va="top", fontsize=9)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bgo", [C_ORANGE, "#f3c7b1", "#e6e6e6", "#b6cff0", C_BLUE])


def fig_minus_copy(cells, tags, path):
    """칸 3 개 × predictor: p − copy (macro, pt) 표준 (파랑) · 인과 (주황) 표적, 95 % CI."""
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    ys = np.arange(len(tags))[::-1]
    allv = [v for t in tags for tgt in ("std", "causal") for cb in COMBOS
            if (c := cells.get(f"{t}|{tgt}|{cb}")) for v in c["minus_copy"]["macro"][1:3]]
    lim = max(5, np.ceil(max(abs(x) for x in allv) / 5) * 5)
    for pi_, (ax, cb) in enumerate(zip(axes, COMBOS)):
        ax.axvline(0, color=C_GRAY, lw=0.8)
        for tgt, col, mk, off in (("std", C_BLUE, "o", 0.13), ("causal", C_ORANGE, "s", -0.13)):
            for y, t in zip(ys, tags):
                c = cells.get(f"{t}|{tgt}|{cb}")
                if not c:
                    continue
                e, lo, hi = c["minus_copy"]["macro"][:3]
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle="round")
                ax.plot(e, y + off, mk, color=col, ms=5.5, mec="white", mew=0.8,
                        label=("standard target" if tgt == "std" else "causal target") if y == ys[0] else None)
        ax.set_xlim(-lim, lim)
        ax.set_title(cb)
        ax.set_xlabel("acc(p) − acc(copy), macro pt")
        ax.set_yticks(ys)
        ax.set_yticklabels([figl(t) for t in tags])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
        plabel(ax, pi_, dy=-30)
        if pi_ == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.subplots_adjust(wspace=0.08, bottom=0.26, top=0.80)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cells(cells, tags, key, path, cbar_label):
    """칸 3 개 (패널) × predictor (행) × O1/O2/O3 · 운동×가림 (열) 의 차이 (pt). 굵은 글씨 = CI 가 0 을 안 포함."""
    plt = setup_mpl()
    cols = PRS + MV
    rows = [t for t in tags if any(cells.get(f"{t}|std|{cb}", {}).get(key) for cb in COMBOS)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.36 * len(rows) + 1.9), sharey=True)
    cm = div_cmap(); vmax = 40
    for pi_, (ax, cb) in enumerate(zip(axes, COMBOS)):
        Z = np.full((len(rows), len(cols)), np.nan); S = np.zeros_like(Z, bool)
        for ri, t in enumerate(rows):
            c = cells.get(f"{t}|std|{cb}", {}).get(key)
            if not c:
                continue
            for ci_, g in enumerate(cols):
                x = c["cells"].get(g)
                if x:
                    Z[ri, ci_] = x[0]; S[ri, ci_] = excl0(x)
        ax.imshow(np.clip(Z, -vmax, vmax), cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
        for ri in range(len(rows)):
            for ci_ in range(len(cols)):
                if np.isfinite(Z[ri, ci_]):
                    ax.text(ci_, ri, f"{Z[ri, ci_]:+.0f}", ha="center", va="center", fontsize=6.8,
                            color="#111111" if S[ri, ci_] else "#8a8a8a", fontweight="bold" if S[ri, ci_] else "normal")
        ax.axvline(2.5, color="white", lw=2.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(PRS + [MV_SHORT[g] for g in MV], rotation=45, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([figl(t) for t in rows])
        ax.set_title(cb)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        plabel(ax, pi_, dy=-36)
    fig.subplots_adjust(wspace=0.06, bottom=0.28, right=0.9)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(-vmax, vmax))
    cb_ = fig.colorbar(sm, cax=fig.add_axes([0.915, 0.33, 0.008, 0.5]))
    cb_.set_label(cbar_label, fontsize=7.5); cb_.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tubelet(tub, tags, path):
    """이동+가림 튜블릿 pre / e0 / post (열) × 표적 (행): predictor 별 튜블릿 정답률 (파랑) vs copy (회색 선 · 띠)."""
    plt = setup_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.0), sharey=True)
    ys = np.arange(len(tags))[::-1]
    k = 0
    for ri, tgt in enumerate(("std", "causal")):
        for ci_, bn in enumerate(("pre", "e0", "post")):
            ax = axes[ri, ci_]
            t = tub.get(f"moving/occluded|{bn}")
            if not t:
                ax.axis("off"); continue
            cp = t[f"copy|{tgt}"]
            g = lambda e: e.get("acc_tie05") or e["acc"]  # noqa: E731   (동점 = 0.5, H6g 와 같은 규칙; 동점 없는 칸은 strict 와 같다)
            c = g(cp)
            ax.axvspan(c[1], c[2], color="#dddddd", lw=0)
            ax.axvline(c[0], color=C_GRAY, lw=1.2, label="copy (no prediction)")
            ax.axvline(50, color="#cccccc", lw=0.8, ls=(0, (3, 3)))
            for y, m in zip(ys, tags):
                e = g(t[f"{m}|{tgt}"])
                col = C_DARK if m == "release" else C_BLUE
                ax.plot([e[1], e[2]], [y, y], color=col, lw=1.6)
                ax.plot(e[0], y, "o", color=col, ms=5.5, mec="white", mew=0.8)
            ax.set_xlim(30, 100)
            ax.set_title(f"{'standard' if tgt == 'std' else 'causal'} target, {bn}")
            ax.set_yticks(ys); ax.set_yticklabels([figl(t) for t in tags])
            ax.set_xlabel("tubelet acc (%, ties = 0.5)")
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
            plabel(ax, k, dy=-30); k += 1
            if k == 1:
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.subplots_adjust(wspace=0.08, hspace=0.75, top=0.88)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_curves(cv, path):
    """학습 곡선 — run 마다 한 열. 윗줄 = macro (표준 표적, skip2_w32 파랑 · skip2_w16 주황, 점선 = copy),
    아랫줄 = p − copy (skip2_w32; 표준 파랑 · 인과 주황), x = epoch (0 = release)."""
    plt = setup_mpl()
    from matplotlib.ticker import MaxNLocator
    pts = cv["points"]
    runs = ["v11", "pv1", "ip1", "ip2", "ariel"]
    title = {"v11": "v11 post-FT", "pv1": "Predictor_v1 post-FT", "ip1": "IntPhys1 post-FT (in domain)",
             "ip2": "IntPhys2 post-FT", "ariel": "Ariel scratch (no e0)"}

    def series(run, key_fn):
        xs, ys, lo, hi = [], [], [], []
        if run != "ariel":
            e = key_fn("release")
            if e:
                xs.append(0); ys.append(e[0]); lo.append(e[1]); hi.append(e[2])
        tags = sorted({k.split("|")[0] for k in pts if k.startswith(run + "_e")},
                      key=lambda t: int("".join(ch for ch in t.rsplit("_e", 1)[1] if ch.isdigit())))
        for t in tags:
            e = key_fn(t)
            if e:
                xs.append(int("".join(ch for ch in t.rsplit("_e", 1)[1] if ch.isdigit()))); ys.append(e[0]); lo.append(e[1]); hi.append(e[2])
        return xs, ys, lo, hi

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharey="row")
    k = 0
    for ci_, run in enumerate(runs):
        ax = axes[0, ci_]
        for cb, col, mk in (("skip2_w32", C_BLUE, "o"), ("skip2_w16", C_ORANGE, "s")):
            xs, ys, lo, hi = series(run, lambda t: (pts.get(f"{t}|std|{cb}") or {}).get("macro"))
            if xs:
                ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
                ax.plot(xs, ys, "-", color=col, lw=2, marker=mk, ms=5, mec="white", mew=0.8, label=cb if ci_ == 0 else None)
            cc = pts.get(f"copy|std|{cb}")
            if cc:
                ax.axhline(cc["macro"][0], color=col, lw=0.9, ls=(0, (4, 3)), alpha=0.8, label=f"copy {cb}" if ci_ == 0 else None)
        ax.set_title(title[run])
        if ci_ == 0:
            ax.set_ylabel("IntPhys1 macro (%)")
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.14), ncol=4)
        ax2 = axes[1, ci_]
        ax2.axhline(0, color=C_GRAY, lw=0.8)
        for tgt, col, mk, lab in (("std", C_BLUE, "o", "standard target"), ("causal", C_ORANGE, "s", "causal target")):
            xs, ys, lo, hi = series(run, lambda t: (pts.get(f"{t}|{tgt}|skip2_w32") or {}).get("minus_copy"))
            if xs:
                ax2.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
                ax2.plot(xs, ys, "-", color=col, lw=2, marker=mk, ms=5, mec="white", mew=0.8, label=lab if ci_ == 0 else None)
        ax2.set_xlabel("epoch")
        if ci_ == 0:
            ax2.set_ylabel("skip2_w32: acc(p) − acc(copy), pt")
            ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
        for a in (ax, ax2):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.grid(axis="y", color=C_GRID, lw=0.6); a.set_axisbelow(True)
            a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    for ri in range(2):
        for ci_ in range(5):
            plabel(axes[ri, ci_], k, dy=-30 if ri == 1 else -22); k += 1
    fig.subplots_adjust(wspace=0.08, hspace=0.62, bottom=0.12, top=0.86)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ 문서
def write_md(R, path):
    cells, tags, tub, val = R["cells"], R["meta"]["tags"], R["tubelet"], R["validation"]
    meta = R["meta"]
    L = []
    A = L.append
    complete = meta["n_videos"] == N_VIDEOS
    A("# IntPhys 1 — 예측이 복사 기준선을 넘는가 (그룹 A, 칸 · 표적 · 튜블릿별)")
    A("")
    for s_ in R.get("correction_lines", []):
        A(s_)
    if R.get("correction_lines"):
        A("")
    A(f"> {meta['generated']} · `te_analyze_ip1.py` 가 배열에서 다시 계산해 쓴 문서다 (숫자는 전부 스크립트 산출, 손으로 옮기지 않았다).")
    A(f"> 입력 `{meta['arrays_src']}` — 영상 {meta['n_videos']}/{N_VIDEOS} · 쌍 {meta['n_pairs']}/180"
      + ("" if complete else " ⚠️ **미완성 — 추출 중인 부분 데이터**") + ". 상위: [`../README.md`](../README.md) · 모델 [`../MODELS.md`](../MODELS.md)")
    A("")
    A("## 한 줄")
    A("")
    A(R.get("headline", ""))
    A("")
    for s in R["headline_lines"]:
        A(s)
    A("")
    # ---------------- 표 1 검증
    A("## 표 1 — 검증: 이 배열의 표준 표적 Filtered macro vs 공식 하네스 per_window")
    A("")
    A("하네스 = `z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_<model>_w{16,32}` (창마다 모델을 지은 공식 실행). "
      "창 |ΔS| = 같은 (영상, 시작점, C) 창 surprise 의 차. 뒤집힘 = 쌍 정오가 다른 쌍.")
    A("")
    A("| predictor | 칸 | 배열 | 하네스 | garrido_rescore | 창 \\|ΔS\\| 중앙값 / 최대 | 뒤집힘 |")
    A("|---|---|---:|---:|---:|---|---|")
    for t in tags:
        v = val.get(t, {})
        for cb in COMBOS:
            e = v.get(cb)
            if not e:
                continue
            fl = "; ".join(f"{f['pos']}/{f['imp']} ({f['mv']}, 배열 {f['margin_ours']:+.1e} · 하네스 {f['margin_harness']:+.1e})" for f in e["flips"]) or "0"
            gr = f"{e['garrido_rescore_macro']:.2f}" if "garrido_rescore_macro" in e else "—"
            A(f"| {disp(t)} | {cb} | {e['ours_macro']:.2f} | {e['harness_macro']:.2f} | {gr} | {e['win_abs_median']:.1e} / {e['win_abs_max']:.1e} | {fl} |")
    A("")
    _fm = [abs(f["margin_ours"]) for t in tags for e in val.get(t, {}).values() if isinstance(e, dict) for f in e.get("flips", [])]
    A("뒤집힘 칸의 margin = G(imp) − G(pos) (양수 = 정답). 두 실행 (영상 배치 구성이 다르다) 의 fp16 GEMM 차이가 창 값에서 중앙값 ~1e-4 다. "
      + (f"⚠️ 정정 (2026-09-25): 뒤집힌 쌍 {len(_fm)} 개의 배열 |margin| 은 최대 {max(_fm):.1e} 이고 1e-4 이상이 {sum(x >= 1e-4 for x in _fm)} 개다 — "
         "'전부 잡음 아래의 동점 근처 쌍' 은 아니다. 칸 macro 는 하네스 (SCORES.md) 와 1 pt 안팎 다르다." if _fm else ""))
    A("")
    bc = R.get("builtin_check", {})
    if "max_abs_diff" in bc:
        A(f"내장 채점기 (`{bc['file']}`, 같은 배열 · `garrido_rescore` 함수) 와 이 스크립트의 strict macro: {bc['n']} 칸 (방법 7 × 표적 2 × 칸 3) 최대 |차| = {bc['max_abs_diff']:.2e}.")
    else:
        A(f"내장 채점기 대조: {bc.get('status', '—')}.")
    A("")
    rv = R.get("release_vs_h6g", {})
    if rv:
        A("릴리즈 · copy 를 H6g (`auto_research/exp_results/h6/h6g_intphys1.json`, 다른 실행 · tie 0.5 · 다른 bootstrap 난수) 와 나란히:")
        A("")
        A("| 칸 · 표적 | p (이 배열 / H6g) | copy (이 배열 / H6g) | p − copy (이 배열) | p − copy (H6g) |")
        A("|---|---|---|---|---|")
        for k, e in rv.items():
            if k.startswith("tub|"):
                continue
            A(f"| {k} | {e['p_ours']:.2f} / {e['p_h6g']:.2f} | {e['copy_ours']:.2f} / {e['copy_h6g']:.2f} | "
              f"{fci(e['p_minus_copy_ours'], True, 2)} | {fci(e['p_minus_copy_h6g'], True, 2)} |")
        for k, e in rv.items():
            if k.startswith("tub|"):
                A(f"| {k} (tie 0.5) | 표준 p {e['p_std_ours_tie05']:.2f} / {e['p_std_h6g']:.2f} · 인과 p {e['p_causal_ours_tie05']:.2f} / {e['p_causal_h6g']:.2f} | "
                  f"표준 {e['copy_std_ours_tie05']:.2f} / {e['copy_std_h6g']:.2f} · 인과 {e['copy_causal_ours_tie05']:.2f} / {e['copy_causal_h6g']:.2f} | | |")
        A("")
    # ---------------- 표 2 주 결과
    for tgt, tname in (("std", "표준 표적 h (창 전부를 본 target encoder)"), ("causal", "인과 표적 h^c (튜블릿 u 까지만 본 target encoder)")):
        A(f"## 표 2{'a' if tgt == 'std' else 'b'} — p vs copy, {tname}")
        A("")
        A("칸마다: p macro · p − copy [95 % CI] (principle 층화 scene bootstrap, 같은 쌍 위의 차). 굵게 = CI 가 0 을 안 포함. copy 행 = 예측 없는 기준선 (여섯 predictor 공유).")
        A("")
        A("| predictor | " + " | ".join(f"{cb} p | {cb} p − copy" for cb in COMBOS) + " | Δ vs release (skip2_w32) |")
        A("|---|" + "---:|---|" * len(COMBOS) + "---|")
        for t in tags + ["copy"]:
            row = []
            for cb in COMBOS:
                c = cells.get(f"{t}|{tgt}|{cb}")
                if not c:
                    row += ["—", "—"]; continue
                row.append(f"{c['macro'][0]:.2f}" + (f" (동점 {c['n_ties']}; tie0.5 {c['macro_tie05'][0]:.2f})" if c["n_ties"] else ""))
                if t == "copy":
                    row.append("기준")
                else:
                    mc = c["minus_copy"]["macro"]
                    s = fci(mc, True)
                    row.append(f"**{s}**" if excl0(mc) else s)
            c32 = cells.get(f"{t}|{tgt}|skip2_w32", {})
            dr = c32.get("minus_release")
            drs = ("기준" if t == "release" else "—") if not dr else (f"**{fci(dr['macro'], True)}**" if excl0(dr["macro"]) else fci(dr["macro"], True))
            A(f"| {disp(t)} | " + " | ".join(row) + f" | {drs} |")
        A("")
        for tz in (R.get("ties", []) if tgt == "causal" else []):
            A(f"표 2a · 2b 의 동점 쌍 `{tz['pos']}`/`{tz['imp']}` ({', '.join(tz['combos'])}, 방법·표적 {tz['n_methods_tied']} 개 전부): 두 영상의 PNG 는 원 프레임 "
              f"{tz['frames_differ'][0]}–{tz['frames_differ'][1]} ({tz['n_frames_differ']} 장) 에서만 다른데 그 칸의 창은 원 프레임 "
              + ", ".join(f"{v}" for v in tz["last_raw_frame_used"].values()) + " 까지만 쓴다 → 위반이 창 밖이라 **구조적 동점** (스크립트가 PNG 로 확인). "
              "strict 는 오답, tie 0.5 는 반점 — p 와 copy 가 같은 쌍에서 같이 동점이라 p − copy 는 두 규칙에서 같다.")
        if R.get("ties") and tgt == "causal":
            A("")
    # ---------------- 표 3 칸
    for tgt in ("std", "causal"):
        A(f"## 표 3{'a' if tgt == 'std' else 'b'} — p − copy (pt) 를 principle · 운동×가림으로 ({'표준' if tgt == 'std' else '인과'} 표적)")
        A("")
        A("값 = 추정 [95 % CI] (scene cluster bootstrap). 굵게 = CI 가 0 을 안 포함. 쌍 수: O1 · O2 · O3 각 60, s/vis · s/occ 각 30, m/vis · m/occ 각 60. "
          "마지막 줄 묶음 = 같은 칸의 copy 정답률 (절대값).")
        A("")
        A("| predictor | 칸 | " + " | ".join(PRS + [MV_SHORT[g] for g in MV]) + " |")
        A("|---|---|" + "---|" * 7)
        for t in tags + ["copy"]:
            for cb in COMBOS:
                c = cells.get(f"{t}|{tgt}|{cb}")
                if not c:
                    continue
                src = c["cells"] if t == "copy" else c["minus_copy"]["cells"]
                vals = []
                for g in PRS + MV:
                    x = src.get(g)
                    if x is None:
                        vals.append("—"); continue
                    s = f"{x[0]:.0f}" if t == "copy" else f"{x[0]:+.0f} [{x[1]:+.0f}, {x[2]:+.0f}]"
                    vals.append(f"**{s}**" if t != "copy" and excl0(x) else s)
                A(f"| {disp(t)} | {cb} | " + " | ".join(vals) + " |")
        A("")
    # ---------------- 표 2c A/B 방향 (정정 2026-09-25)
    A("## 표 2c — A/B 방향별 쌍 정답률 (appear / disappear, CLAUDE.md §1-7)")
    A("")
    _n = {k: (cells.get(f"copy|std|skip2_w32", {}).get("cells", {}).get(k) or [0, 0, 0, 0])[3] for k in ("dir=appear", "dir=disappear", "O1|dir=appear", "O1|dir=disappear", "O3|dir=appear", "O3|dir=disappear")}
    A("O1 · O3 쌍을 imp 영상의 물체 보임 프레임 수로 나눈다 (imp 가 적으면 disappear, 많으면 appear; O2 · 같음은 제외). 값 = 정답률 % [95 % CI] (scene cluster bootstrap). "
      f"쌍 수: appear {_n['dir=appear']} · disappear {_n['dir=disappear']} (O1 {_n['O1|dir=appear']} / {_n['O1|dir=disappear']} · O3 {_n['O3|dir=appear']} / {_n['O3|dir=disappear']}). "
      "O1 disappear 한 쌍 = " + (f"{100 / _n['O1|dir=disappear']:.1f} pt." if _n['O1|dir=disappear'] else "—"))
    A("")
    for tgt in ("std", "causal"):
        A(f"**{'표준' if tgt == 'std' else '인과'} 표적**")
        A("")
        A("| 방법 | " + " | ".join(f"{cb} appear | {cb} disappear | {cb} O1 app / dis · O3 app / dis" for cb in COMBOS) + " |")
        A("|---|" + "---|---|---|" * len(COMBOS))
        for t in tags + ["copy"]:
            row = []
            for cb in COMBOS:
                cc = cells.get(f"{t}|{tgt}|{cb}", {}).get("cells", {})
                for k in ("dir=appear", "dir=disappear"):
                    row.append(fci(cc.get(k)) if cc.get(k) else "—")
                row.append(" · ".join(f"{cc[k][0]:.0f}" if cc.get(k) else "—" for k in ("O1|dir=appear", "O1|dir=disappear", "O3|dir=appear", "O3|dir=disappear")))
            A(f"| {disp(t)} | " + " | ".join(row) + " |")
        A("")
    # ---------------- 표 4 Δ vs release 칸
    A("## 표 4 — Δ vs release (pt, 표준 표적) 를 principle · 운동×가림으로")
    A("")
    A("| predictor | 칸 | macro | " + " | ".join(PRS + [MV_SHORT[g] for g in MV]) + " | 쌍 (이 모델만 / release 만) |")
    A("|---|---|---|" + "---|" * 7 + "---|")
    for t in tags[1:]:
        for cb in COMBOS:
            c = cells.get(f"{t}|std|{cb}")
            if not c or "minus_release" not in c:
                continue
            d = c["minus_release"]
            vals = []
            for g in PRS + MV:
                x = d["cells"].get(g)
                s = "—" if x is None else f"{x[0]:+.0f} [{x[1]:+.0f}, {x[2]:+.0f}]"
                vals.append(f"**{s}**" if excl0(x) else s)
            ms = fci(d["macro"], True)
            dc = d["discordant"]
            A(f"| {disp(t)} | {cb} | {'**' + ms + '**' if excl0(d['macro']) else ms} | " + " | ".join(vals)
              + f" | {dc['only_' + t]} / {dc['only_release']} |")
    A("")
    # ---------------- 표 5 튜블릿
    A("## 표 5 — 튜블릿: 이동+가림 사건 튜블릿 (H6e/H6d 정의, skip2_w16 + skip2_w32)")
    A("")
    _mo = tub.get("moving/occluded|e0", {})
    A("pre = 사건 전 · e0 = 분기 (드러남) 튜블릿 · post = 사건 뒤. 튜블릿 정답 = 1[L1_j(imp) > L1_j(pos)]. "
      f"CI = scene cluster bootstrap (이동+가림 {_mo.get('n_scene', '—')} scene · e0 튜블릿 {_mo.get('n_tubelets', '—')} 개 — 창 × C 반복이라 독립 표본이 아니다). "
      "인과 표적의 pre 는 pos/imp 입력 프레임이 대부분 같아 L1 이 **정확히 같다** (동점) → strict 는 ~0, tie = 0.5 는 ~50 — 둘 다 적는다. "
      "⚠️ 정정 (2026-09-25): 사건 bin 은 first_div_strict 기준인데 일부 쌍은 PNG 가 그보다 1–4 프레임 먼저 (first_div_sensitive) 갈린다 → 오염 쌍을 뺀 판은 표 5b, "
      "보임 칸 pre 는 표 5c.")
    A("")
    for tgt in ("std", "causal"):
        A(f"**{'표준' if tgt == 'std' else '인과'} 표적**")
        A("")
        A("| 방법 | " + " | ".join(f"{bn} acc | {bn} p − copy" for bn in ("pre", "e0", "post")) + " | e0 Δ vs release |")
        A("|---|" + "---:|---|" * 3 + "---|")
        for m in tags + ["copy"]:
            row = []
            for bn in ("pre", "e0", "post"):
                t = tub.get(f"moving/occluded|{bn}")
                if not t:
                    row += ["—", "—"]; continue
                e = t[f"{m}|{tgt}"]
                a = f"{e['acc'][0]:.1f}" + (f" (tie0.5 {e['acc_tie05'][0]:.1f})" if "acc_tie05" in e else "")
                row.append(a)
                if m == "copy":
                    row.append("기준")
                else:
                    x = e["minus_copy"]
                    s = fci(x, True)
                    if "minus_copy_tie05" in e:
                        s += f" (tie0.5 {e['minus_copy_tie05'][0]:+.1f})"
                    row.append(f"**{s}**" if excl0(x) else s)
            t0 = tub.get("moving/occluded|e0", {}).get(f"{m}|{tgt}", {})
            dr = t0.get("minus_release")
            drs = ("기준" if m == "release" else "—") if not dr else (f"**{fci(dr, True)}**" if excl0(dr) else fci(dr, True))
            A(f"| {disp(m)} | " + " | ".join(row) + f" | {drs} |")
        A("")
    A("e0 튜블릿을 H6d 처럼 **문맥 끝에서 물체가 보였나** 로 쪼갠 것 (표준 표적, p − copy; n = 튜블릿 · scene):")
    A("")
    A("| 방법 | 문맥 끝 보임 acc | p − copy | 문맥 끝 가려짐 acc | p − copy |")
    A("|---|---:|---|---:|---|")
    for m in tags + ["copy"]:
        row = []
        for cvn in ("ctxend_visible", "ctxend_hidden"):
            t = tub.get(f"moving/occluded|e0|{cvn}")
            if not t:
                row += ["—", "—"]; continue
            e = t[f"{m}|std"]
            row.append(f"{e['acc'][0]:.1f} (n {e['acc'][3]} · {e['acc'][4]})")
            row.append("기준" if m == "copy" else (f"**{fci(e['minus_copy'], True)}**" if excl0(e["minus_copy"]) else fci(e["minus_copy"], True)))
        A(f"| {disp(m)} | " + " | ".join(row) + " |")
    A("")
    td, tsn, ed = R.get("tubelet_drop_early_div", {}), R.get("tubelet_first_div_sensitive", {}), R.get("early_div_pairs", {})
    if td or tsn:
        _m = ed.get("moving/occluded", {})
        A(f"**표 5b — 이동+가림, 사건 bin 오염 보정** (정정 2026-09-25). 왼쪽 = first_div_sensitive 로 다시 bin (60 쌍 전부, 표준 표적 정답률 · p − copy). "
          f"오른쪽 = first_div_strict bin 에서 인과 pre 비동점을 가진 쌍 {_m.get('n_pairs_pre_nontie', '—')} 개 ({', '.join(_m.get('scenes_pre_nontie', []))}) 를 뺀 판. "
          "마지막 열 = sensitive bin 의 인과 pre 비동점 비율 (이동+가림 · 정지+가림; 0 이면 bin 이 입력 차이를 다 담는다).")
        A("")
        A("| 방법 | sens pre acc | sens pre p − copy | sens e0 acc | sens e0 p − copy | sens post acc | 제외판 pre acc | 제외판 e0 acc | 제외판 e0 p − copy | sens 인과 pre 비동점 |")
        A("|---|---:|---|---:|---|---:|---:|---:|---|---|")
        for m in tags + ["copy"]:
            row = []
            for bn in ("pre", "e0"):
                e = (tsn.get(f"moving/occluded|{bn}") or {}).get(f"{m}|std")
                if not e:
                    row += ["—", "—"]; continue
                row.append(f"{e['acc'][0]:.1f}")
                row.append("기준" if m == "copy" else (f"**{fci(e['minus_copy'], True)}**" if excl0(e["minus_copy"]) else fci(e["minus_copy"], True)))
            e = (tsn.get("moving/occluded|post") or {}).get(f"{m}|std")
            row.append(f"{e['acc'][0]:.1f}" if e else "—")
            for bn in ("pre", "e0"):
                e = (td.get(f"moving/occluded|{bn}") or {}).get(f"{m}|std")
                row.append(f"{e['acc'][0]:.1f}" if e else "—")
            e = (td.get("moving/occluded|e0") or {}).get(f"{m}|std")
            row.append("기준" if m == "copy" else ("—" if not e else (f"**{fci(e['minus_copy'], True)}**" if excl0(e["minus_copy"]) else fci(e["minus_copy"], True))))
            nz = " · ".join(f"{tsn[g + '|pre'][f'{m}|causal']['nonzero']:.3f}" if g + "|pre" in tsn else "—" for g in ("moving/occluded", "static/occluded"))
            A(f"| {disp(m)} | " + " | ".join(row) + f" | {nz} |")
        A("")
    A("**표 5c — 보임 칸 · 정지+가림의 pre (표준 표적 정답률)** (정정 2026-09-25): 표준 표적의 번짐이 copy 에 점수를 주는 곳. 정지+가림은 bin 오염 (위) 때문에 참고만.")
    A("")
    A("| 방법 | " + " | ".join(f"{MV_SHORT[g]} pre" for g in MV) + " |")
    A("|---|" + "---:|" * len(MV))
    for m in tags + ["copy"]:
        A(f"| {disp(m)} | " + " | ".join(f"{tub[g + '|pre'][f'{m}|std']['acc'][0]:.1f}" if g + "|pre" in tub else "—" for g in MV) + " |")
    A("")
    A("다른 운동×가림 칸의 튜블릿 표 (pre / e0 / post × 네 칸) 와 방향별 (appear / disappear / equal / none) e0 는 `ip1_copy.json` → `tubelet`.")
    A("")
    # ---------------- 표 6 그룹 B
    A("## 표 6 — 그룹 B (encoder 부터 새로 학습) — 하네스 값만, **맥락용**")
    A("")
    A("⚠️ 그룹 B 는 IntPhys 1 에서 **자기 copy 기준선을 뽑지 않았다** (이 배열은 릴리즈 ViT-H encoder 의 copy 다) — 그래서 p − copy 가 없다. "
      "그룹 A 와 encoder 가 달라 copy 를 공유하지 않는다. 값 = Filtered macro [95 % CI] (principle 층화 scene bootstrap, 하네스 per_window 에서 다시 계산).")
    A("")
    A("| 모델 | " + " | ".join(COMBOS) + " | 최고 칸 | skip2_w32 O1 · O2 · O3 | skip2_w32 s/vis · s/occ · m/vis · m/occ |")
    A("|---|" + "---|" * len(COMBOS) + "---|---|---|")
    gb = R["group_B"]
    for name in GROUP_B:
        e = gb.get(name, {})
        if "status" in e or not e:
            A(f"| {name} | " + " | ".join("—" for _ in COMBOS) + f" | {e.get('status', '—')} | | |")
            continue
        row = [fci(e[cb]["macro"]) if cb in e else "—" for cb in COMBOS]
        b = e.get("best", {})
        c32 = e.get("skip2_w32", {}).get("cells", {})
        A(f"| {name} | " + " | ".join(row) + f" | {b.get('combo', '—')} {b.get('macro', float('nan')):.2f} | "
          + " · ".join(f"{c32[g][0]:.0f}" if g in c32 else "—" for g in PRS) + " | "
          + " · ".join(f"{c32[g][0]:.0f}" if g in c32 else "—" for g in MV) + " |")
    A("")
    # ---------------- 표 7 곡선
    cv = R["curves"]
    A("## 표 7 — 학습 곡선")
    A("")
    st = cv["status"]
    if st.get("source") == "arrays":
        A(f"`{st['cache']}` 완성 ({st['n_shards']} 영상, predictor {len(st.get('tags', []))} 개) — 배열에서 표준 표적 macro 와 p − copy (같은 쌍, 표준 · 인과). "
          "e0 = release (Ariel 은 scratch 라 e0 없음). 곡선 배열과 final 배열이 겹치는 열 (final 여섯 태그 · copy) 의 최대 |차| = "
          + (f"{max(st['final_overlap_max_abs'].values()):.1e}" if st.get("final_overlap_max_abs") else "—") + " (다른 실행 · 다른 predictor 목록).")
    else:
        A(f"⚠️ `ip1score_curves` 추출은 **이 문서 시점에 끝나지 않았다** (shard {st['n_shards']}/{N_VIDEOS}; 다른 세션이 추출 중 → 끝나면 이 스크립트를 다시 돌리면 배열 판으로 바뀐다). "
          "대신 **하네스 per_window 가 있는 점만** 싣는다: p 는 하네스 (Benchmarks · z_training eval), copy 는 이 final 배열의 표준 표적 copy "
          "(copy 는 encoder 만의 함수라 어느 predictor 에도 같다). 인과 표적 곡선은 curves 추출 뒤에만 나온다.")
    A("")
    A("| 점 | " + " | ".join(f"{cb} macro | {cb} p − copy | {cb} p − copy (인과)" for cb in ("skip2_w32", "skip2_w16")) + " |")
    A("|---|" + "---:|---|---|" * 2)
    order = sorted({k.split("|")[0] for k in cv["points"]},
                   key=lambda s: (s != "release", s == "copy", s.split("_e")[0], int(''.join(ch for ch in s.split("_e")[-1] if ch.isdigit()) or 0)))
    for tg in order:
        row = []
        for cb in ("skip2_w32", "skip2_w16"):
            e = cv["points"].get(f"{tg}|std|{cb}")
            if not e:
                row += ["—", "—"]; continue
            row.append(f"{e['macro'][0]:.2f}")
            for mc in (e.get("minus_copy"), (cv["points"].get(f"{tg}|causal|{cb}") or {}).get("minus_copy")):
                row.append("—" if not mc else (f"**{fci(mc, True)}**" if excl0(mc) else fci(mc, True)))
        A(f"| {disp(tg)} | " + " | ".join(row) + " |")
    A("")
    stp = cv.get("steps", {})
    if stp:
        A("**표 7b — 체크포인트 사이 변화 (같은 쌍 paired, principle 층화 scene bootstrap)** (정정 2026-09-25). 값 = Δ macro [95 % CI] · 쌍 (나중만 맞음 / 앞만 맞음).")
        A("")
        A("| run | 구간 | " + " | ".join(f"{cb} 표준" for cb in COMBOS) + " | " + " | ".join(f"{cb} 인과" for cb in COMBOS) + " |")
        A("|---|---|" + "---|" * (2 * len(COMBOS)))
        segs = sorted({(k.split("|")[0], k.split("|")[-1]) for k in stp}, key=lambda x: (x[0], not x[1].startswith("release")))
        for run, sg in segs:
            row = []
            for tgt in ("std", "causal"):
                for cb in COMBOS:
                    v = stp.get(f"{run}|{tgt}|{cb}|{sg}")
                    if not v:
                        row.append("—"); continue
                    a_, b_ = sg.split("->")
                    x = fci(v["delta"], True)
                    row.append((f"**{x}**" if excl0(v["delta"]) else x) + f" ({v['discordant'].get('only_' + b_, 0)}/{v['discordant'].get('only_' + a_, 0)})")
            A(f"| {run} | {sg.replace('->', ' → ')} | " + " | ".join(row) + " |")
        A("")
    # ---------------- 읽기 · 단서
    A("## 읽기")
    A("")
    for s in R["reading_lines"]:
        A(s)
    A("")
    A("## 단서")
    A("")
    for s in R["caveat_lines"]:
        A(s)
    A("")
    A("## 그림")
    A("")
    for f, d in R["figures"]:
        A(f"- `{f}` — {d}")
    A("")
    A("## 재현")
    A("")
    A("```bash")
    A("cd /data/hyuntak/project/2026/2027_cvpr/vjepa2")
    A("# 추출 (GPU, 다른 세션 소유 — 이 분석은 읽기만 한다)")
    A("CUDA_VISIBLE_DEVICES=0,1,2,3 python z_research/scripts/analysis/te_intphys1_score.py --preset final --gpus 4")
    A("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python z_research/scripts/analysis/te_intphys1_score.py --preset curves --gpus 8   # 학습 곡선 (22 predictor)")
    A("# 내장 채점 · 검증 (CPU) → z_research/TrainingEffects/ip1score/ip1score_final_{scores,validation}.json")
    A("python z_research/scripts/analysis/te_intphys1_score.py --preset final --validate --score")
    A("# 이 분석 (CPU, ~1-2 분)")
    A(f"python z_research/scripts/analysis/te_analyze_ip1.py --B {meta['B']}")
    A("```")
    A("")
    A("입력: `/data2/local_datasets/world/world_analysis/cache/training_effects/ip1score_final/` (shards 또는 arrays_v360.npz) · "
      "`auto_research/_stage/IntPhys1_dev_by_scene/{pairs.csv, videos.csv, obj_vis.npz}` (읽기 전용) · "
      "`z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_*_w{16,32}/per_window.json`. "
      "config: `configs/protocols/intphys1_sliding.yaml` + `configs/protocols/models.md` (`vith`). "
      "채점 식: `z_research/scripts/analysis/garrido_rescore.py` (import), 창 배치: `te_intphys1_score.Layout` (import).")
    path.write_text("\n".join(L) + "\n")


# ============================================================================ 한 줄 · 읽기 · 단서 (결과에서 문장 생성)
def narrative(R):
    cells, tags, tub = R["cells"], R["meta"]["tags"], R["tubelet"]
    beats = collections.defaultdict(list)       # (tgt) -> [(tag, cb, ci)]
    below = collections.defaultdict(list)
    for t in tags:
        for tgt in ("std", "causal"):
            for cb in COMBOS:
                c = cells.get(f"{t}|{tgt}|{cb}")
                if not c:
                    continue
                mc = c["minus_copy"]["macro"]
                if mc[1] > 0:
                    beats[tgt].append((t, cb, mc))
                elif mc[2] < 0:
                    below[tgt].append((t, cb, mc))
    H = []
    def _inh(t, tgt, cb):                        # 정정 2026-09-25: Δ vs release 의 CI 가 0 을 포함하면 릴리즈에서 물려받은 몫
        d = (cells.get(f"{t}|{tgt}|{cb}") or {}).get("minus_release", {}).get("macro")
        return f" (릴리즈에서 물려받음 — Δ vs release {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}])" if d and not excl0(d) else ""
    lst = lambda xs, tgt="std": ", ".join(f"{disp(t)} {cb} {m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}]" + (_inh(t, tgt, cb) if m[0] > 0 else "") for t, cb, m in xs) or "없음"  # noqa: E731
    H.append(f"- **복사 기준선을 CI 로 넘는 칸 (표준 표적, macro)**: {lst(beats['std'])}.")
    H.append(f"- **인과 표적**: {lst(beats['causal'], 'causal')}.")
    if below["std"] or below["causal"]:
        H.append(f"- **copy 보다 CI 로 낮은 칸**: 표준 {lst(below['std'])} / 인과 {lst(below['causal'], 'causal')}.")
    where = []
    for t in tags:
        seg = []
        for cb in COMBOS:
            c = cells.get(f"{t}|std|{cb}")
            if not c:
                continue
            hits = [f"{MV_SHORT.get(g, g)} {x[0]:+.0f}" for g, x in c["minus_copy"]["cells"].items()
                    if g in PRS + MV and x and x[1] > 0]
            lows = [f"{MV_SHORT.get(g, g)} {x[0]:+.0f}" for g, x in c["minus_copy"]["cells"].items()
                    if g in PRS + MV and x and x[2] < 0]
            if hits or lows:
                seg.append(f"{cb}: " + ", ".join(hits + [f"({l}, copy 우세)" for l in lows]))
        where.append(f"{disp(t)} — " + ("; ".join(seg) if seg else "CI 로 갈리는 칸 없음"))
    H.append("- 칸 (principle · 운동×가림, 표준 표적) 에서 CI 로 갈리는 곳 (⚠️ 보정 없는 칸 CI 수십 개 — 5 % 면 우연히 갈리는 칸이 여럿 나온다; 정지 칸은 scene 15 개): " + " / ".join(where) + ".")
    c32 = cells.get("copy|std|skip2_w32")
    if c32:
        H.append(f"- copy 자체가 이미 높다 — skip2_w32 표준 표적 copy {c32['macro'][0]:.2f} (chance 50). 그래서 p − copy 가 작은 칸에서는 "
                 "'예측이 문맥의 마지막 장면 이상을 쓴다' 는 증거가 약하다.")
    return H


# ============================================================================ 읽기 · 단서 (숫자는 R 에서 꺼낸다 — 손으로 옮기지 않는다)
def _g(R, t, tgt, cb, *path):
    e = R["cells"].get(f"{t}|{tgt}|{cb}")
    for k in path:
        if e is None:
            return None
        e = e.get(k)
    return e


def _sig(ci):
    if ci is None:
        return "—"
    return "CI 가 0 위" if ci[1] > 0 else ("CI 가 0 아래" if ci[2] < 0 else "CI 가 0 을 포함")


def _dir_range(R, tgt="std", cb="skip2_w32"):
    """모든 방법 (predictor 여섯 + copy) 의 한 칸 appear / disappear 정답률 범위 (쌍 단위, O1+O3)."""
    out = {}
    for dn in ("appear", "disappear"):
        v = [c["cells"][f"dir={dn}"][0] for k, c in R["cells"].items()
             if k.split("|")[1] == tgt and k.split("|")[2] == cb and f"dir={dn}" in c.get("cells", {})]
        out[dn] = (min(v), max(v)) if v else None
    return out


def _steps_below(R, tgt, cb):
    st = R["curves"].get("steps", {})
    return {k: v for k, v in st.items() if k.startswith(tuple(f"{r}|{tgt}|{cb}|" for r in ("v11", "pv1", "ip1", "ip2", "ariel")))}


def HEADLINE(R):
    """한 줄 — 정정 2026-09-25: 적대 검증의 safe_headline 에 맞춰 다시 조립 (숫자는 R 에서)."""
    tags = R["meta"]["tags"]
    G = lambda t, tgt, cb, *k: _g(R, t, tgt, cb, *k)  # noqa: E731
    c32 = G("copy", "std", "skip2_w32", "macro")
    if c32 is None:
        return "*(데이터 부족)*"
    best = "skip2_w32"
    c32c = G("copy", "causal", best, "macro")
    r32 = G("release", "std", best, "minus_copy", "macro")
    s = (f"IntPhys 1 dev ({R['meta']['n_pairs']} 쌍, 90 scene) 에서 예측 없이 문맥 마지막 튜블릿 latent 를 복사하기만 해도 (copy) Garrido Filtered macro 가 "
         f"skip2_w32 {c32[0]:.1f} (표준 표적) / {c32c[0]:.1f} (인과 표적) 이다. 릴리즈 predictor 의 {G('release', 'std', best, 'macro')[0]:.2f} 는 같은 칸 copy 보다 "
         f"{fci(r32, True)} 로 " + ("CI 상 구분되지 않는다. " if not excl0(r32) else f"{_sig(r32)}. "))
    beats = [f"{cb} {'표준' if tgt == 'std' else '인과'} {fci(G('release', tgt, cb, 'minus_copy', 'macro'), True)}"
             for tgt in ("std", "causal") for cb in COMBOS if G("release", tgt, cb, "minus_copy", "macro")[1] > 0]
    s += f"릴리즈가 copy 를 CI 로 넘는 곳은 {', '.join(beats) or '없음'} 뿐이다. "
    dr = _dir_range(R)
    if dr.get("appear") and dr.get("disappear"):
        s += (f"skip2_w32 에서 모든 방법 (predictor 여섯 + copy, 표준 표적, O1 · O3 쌍) 이 appear 방향 {dr['appear'][0]:.0f}–{dr['appear'][1]:.0f} %, "
              f"disappear 방향 {dr['disappear'][0]:.0f}–{dr['disappear'][1]:.0f} % 로 방향 편향이 크다 (표 2c). ")
    both = [t for t in tags[1:] if G(t, "std", best, "minus_copy", "macro")[1] > 0 and G(t, "std", best, "minus_release", "macro")[1] > 0]
    if both:
        s += ("학습한 predictor 중 skip2_w32 에서 copy 와 릴리즈를 둘 다 CI 로 넘는 것은 "
              + ", ".join(f"{disp(t)} (vs copy {G(t, 'std', best, 'minus_copy', 'macro')[0]:+.1f} / vs release {G(t, 'std', best, 'minus_release', 'macro')[0]:+.1f})" for t in both)
              + " 이고, 학습 도메인 안의 결과라 일반화 증거가 아니다. ")
    else:
        s += "학습한 predictor 중 skip2_w32 에서 copy 와 릴리즈를 둘 다 CI 로 넘는 것은 없다. "
    down = [t for t in tags[1:] if t not in both and not t.startswith("ariel") and G(t, "std", best, "minus_release", "macro")[2] < 0]
    if down:
        dv = [G(t, "std", best, "minus_release", "macro")[0] for t in down]
        s += (", ".join(disp(t) for t in down) + f" 는 skip2_w32 에서 릴리즈보다 {min(-x for x in dv):.1f}–{max(-x for x in dv):.1f} pt 낮다 (CI, 배열 값; 하네스 값은 표 1). ")
    fall = []
    for k, v in _steps_below(R, "std", "skip2_w16").items():
        a_, b_ = k.split("|")[-1].split("->")
        if a_ != "release" and v["delta"][2] < 0:
            fall.append(f"{k.split('|')[0]} {a_.rsplit('_', 1)[1]}→{b_.rsplit('_', 1)[1]} {fci(v['delta'], True)}")
    if fall:
        s += "skip2_w16 에서 epoch 이 늘수록 더 떨어지는 것: " + ", ".join(fall) + " (첫 점→끝 점, 같은 쌍 paired CI). "
    ar = [t for t in tags if t.startswith("ariel")]
    for t in ar:
        bs = [cb for cb in COMBOS if G(t, "std", cb, "minus_copy", "macro")[2] < 0]
        bc = [cb for cb in COMBOS if G(t, "causal", cb, "minus_copy", "macro")[2] < 0]
        s += (f"Ariel scratch ({disp(t)}) 는 표준 표적 {len(bs)}/3 칸에서 copy 보다 CI 로 낮지만, 인과 표적에서는 "
              + (", ".join(f"{cb} {fci(G(t, 'causal', cb, 'minus_copy', 'macro'), True)}" for cb in bc) or "없음") + " 만 그렇다. ")
    s += "학습은 한 번씩 (seed 0) 뿐이고 칸 CI 는 scene 15–30 개 위의 값이라, 칸 단위 차이는 다중비교를 감안해 읽어야 한다."
    return "**" + s.strip() + "**"


def CORRECTIONS(R):
    """정정 블록 (2026-09-25, 적대 검증 `../Archive/verify_raw/verify_ip1.json` 반영) — 이전 문장 → 지금 문장, 숫자는 R 에서."""
    G = lambda t, tgt, cb, *k: _g(R, t, tgt, cb, *k)  # noqa: E731
    tub, ts, td = R["tubelet"], R.get("tubelet_first_div_sensitive", {}), R.get("tubelet_drop_early_div", {})
    ed, hz = R.get("early_div_pairs", {}), R.get("future_tubelets_max", {})
    L = ["> ⚠️ **정정 (2026-09-25, 적대 검증 반영)** — 무엇이 바뀌었나 (이전 문장 → 지금 문장). 숫자는 전부 이번 재실행 산출. 검증 원문: `../Archive/verify_raw/verify_ip1.json`.", ">"]
    ar = [t for t in R["meta"]["tags"] if t.startswith("ariel")]
    for t in ar:
        cz = ", ".join(f"{cb} {fci(G(t, 'causal', cb, 'minus_copy', 'macro'), True)}" for cb in COMBOS)
        L.append(f"> 1. **Ariel vs copy** — 이전: '{disp(t)} 는 세 칸 모두 copy 보다 CI 로 낮다 (예측이 복사보다 못하다)' → 지금: **표준 표적에서만** 세 칸 모두 낮다. "
                 f"인과 표적 p − copy: {cz}. 또 Ariel 은 예측 지평 최대 8 slot (12 fps) 으로 학습됐는데 창이 요구하는 미래 튜블릿은 최대 "
                 + " · ".join(f"{cb} {hz[cb]}" for cb in COMBOS if cb in hz) + " 개다 — skip2_w32 는 학습 지평 밖. '복사보다 못하다' 를 attention 규칙의 성질로 읽지 않는다.")
    st = R["curves"].get("steps", {})
    def _st(k):
        v = st.get(k)
        if not v:
            return None
        d = v["discordant"]; a_, b_ = k.split("|")[-1].split("->")
        return f"{fci(v['delta'], True)} ({d.get('only_' + b_, 0)} vs {d.get('only_' + a_, 0)} 쌍)"
    sw = [f"{k.split('|')[0]} {k.split('|')[-1].replace('->', '→')} {_st(k)}" for k in st if "|std|skip2_w16|" in k]
    s32 = [f"{k.split('|')[0]} {k.split('|')[-1].replace('->', '→')} {_st(k)}" for k in st if "|std|skip2_w32|" in k and not k.startswith("ariel")]
    L.append("> 2. **학습 곡선** — 이전: 'v11 · ip1 는 변화의 3/4 이상이 첫 체크포인트 안에서 일어나고 그 뒤로 거의 평평하다' (skip2_w32 한 칸에서 고른 결론) → "
             "지금: 체크포인트 사이 변화를 같은 쌍 paired CI 로 잰다 (표 7b). skip2_w16 표준: " + "; ".join(sw)
             + ". skip2_w32 표준: " + "; ".join(s32) + ". '더 오래 학습해서 생긴 변화가 아니다' 류의 읽기는 뺀다.")
    mo, so = ed.get("moving/occluded", {}), ed.get("static/occluded", {})
    pre_s = (ts.get("moving/occluded|pre") or {}).get("copy|causal", {}).get("nonzero")
    pre_ss = (ts.get("static/occluded|pre") or {}).get("copy|causal", {}).get("nonzero")
    pre0 = (tub.get("moving/occluded|pre") or {}).get("copy|causal", {}).get("nonzero")
    pre0s = (tub.get("static/occluded|pre") or {}).get("copy|causal", {}).get("nonzero")
    ip1 = next((t for t in R["meta"]["tags"] if t.startswith("ip1")), None)
    dpre = (td.get("moving/occluded|pre") or {})
    de0 = (td.get("moving/occluded|e0") or {})
    txt = (f"> 3. **사건 전 튜블릿 (pre)** — 이전: 'pos/imp 입력 프레임이 같다 · 문맥 프레임이 픽셀 단위로 같다 · 인과 pre 의 비동점은 fp16 잡음 (미검증)' → "
           f"지금: **틀렸다.** PNG 가 first_div_strict 보다 먼저 (first_div_sensitive 에서) 갈리는 쌍이 이동+가림 {mo.get('n_pairs_early', '—')}/{mo.get('n_pairs', '—')} 쌍, "
           f"정지+가림 {so.get('n_pairs_early', '—')}/{so.get('n_pairs', '—')} 쌍이고, 그 탓에 first_div_strict bin 의 인과 pre 에 실제로 비동점이 생기는 쌍은 이동+가림 "
           f"{mo.get('n_pairs_pre_nontie', '—')} 쌍 ({', '.join(mo.get('scenes_pre_nontie', []))}) · 정지+가림 {so.get('n_pairs_pre_nontie', '—')} 쌍 "
           f"({len(so.get('scenes_pre_nontie', []))}/{so.get('n_scene', '—')} scene) 이다. "
           f"인과 표적 pre 의 비동점 비율 (copy): first_div_strict bin 이동+가림 {pre0 if pre0 is None else f'{pre0:.3f}'} · 정지+가림 {pre0s if pre0s is None else f'{pre0s:.3f}'} "
           f"→ first_div_sensitive 로 다시 bin 하면 {pre_s if pre_s is None else f'{pre_s:.3f}'} · {pre_ss if pre_ss is None else f'{pre_ss:.3f}'}. ")
    spre, se0 = ts.get("moving/occluded|pre") or {}, ts.get("moving/occluded|e0") or {}
    if spre and se0 and ip1:
        txt += (f"first_div_sensitive 로 다시 bin 한 이동+가림 (표 5b): {disp(ip1)} pre 표준 {spre[f'{ip1}|std']['acc'][0]:.1f}, 릴리즈 e0 {se0['release|std']['acc'][0]:.1f} vs copy "
                f"{se0['copy|std']['acc'][0]:.1f}. ")
    if dpre and de0 and ip1:
        txt += (f"비동점 쌍을 뺀 이동+가림 (표 5b): {disp(ip1)} pre 표준 {dpre[f'{ip1}|std']['acc'][0]:.1f} (전체 {tub['moving/occluded|pre'][f'{ip1}|std']['acc'][0]:.1f}), "
                f"릴리즈 e0 {de0['release|std']['acc'][0]:.1f} vs copy {de0['copy|std']['acc'][0]:.1f} (전체 {tub['moving/occluded|e0']['release|std']['acc'][0]:.1f} vs "
                f"{tub['moving/occluded|e0']['copy|std']['acc'][0]:.1f}) — pre · e0 결론은 남는다. json 의 정지+가림 튜블릿 bin (first_div_strict) 은 쓰지 않는다.")
    L.append(txt)
    sv, mvv, moc = tub.get("static/visible|pre"), tub.get("moving/visible|pre"), tub.get("moving/occluded|pre")
    if sv and mvv and moc:
        L.append(f"> 4. **표준 표적 번짐의 근거** — 이전: '표 5 의 pre 칸에서 copy 에도 점수를 준다' → 지금: 표 5 (이동+가림) pre 의 copy 표준 표적은 "
                 f"{moc['copy|std']['acc'][0]:.1f} 로 50 아래 — 거기서는 번짐이 copy 를 **깎는다**. copy 가 번짐 덕을 보는 곳은 보임 칸 pre: 정지+보임 "
                 f"{sv['copy|std']['acc'][0]:.1f} · 이동+보임 {mvv['copy|std']['acc'][0]:.1f} (표 5c). 칸 macro 의 copy 표준→인과 {G('copy', 'std', 'skip2_w32', 'macro')[0]:.1f} → "
                 f"{G('copy', 'causal', 'skip2_w32', 'macro')[0]:.1f} 는 그대로.")
    c = R["cells"]
    def dd(t, cb="skip2_w32", k="dir=disappear"):
        e = c.get(f"{t}|std|{cb}", {}).get("cells", {}).get(k)
        return e
    ca, cd, ra, rd = dd("copy", k="dir=appear"), dd("copy"), dd("release", k="dir=appear"), dd("release")
    o1d = dd("copy", k="O1|dir=disappear")
    if ca and cd and ra and rd:
        L.append(f"> 5. **A/B 방향 분리 추가 (표 2c, CLAUDE.md §1-7)** — 이전: 쌍 단위 appear / disappear 분리가 없었다 → 지금: skip2_w32 표준 표적 copy appear {ca[0]:.1f} / "
                 f"disappear {cd[0]:.1f}, 릴리즈 {ra[0]:.1f} / {rd[0]:.1f} (O1+O3 쌍; disappear n = {cd[3]}). O1 disappear 는 n = {o1d[3] if o1d else '—'} 쌍이라 한 쌍이 "
                 f"{100 / o1d[3] if o1d else float('nan'):.1f} pt 다. copy 85 · p 88.89 의 상당 부분이 방향 편향과 섞여 있다.")
    if ip1:
        L.append(f"> 6. **{disp(ip1)} 의 인과 표적 이득** — 이전: '인과 표적에서도 CI 가 0 위' (skip2_w32 만 보고) → 지금: 인과 표적 Δ vs release 는 "
                 + " · ".join(f"{cb} {fci(G(ip1, 'causal', cb, 'minus_release', 'macro'), True)}" for cb in COMBOS) + " — CI 로 갈리는 칸은 "
                 + (", ".join(cb for cb in COMBOS if excl0(G(ip1, 'causal', cb, 'minus_release', 'macro'))) or "없음") + " 뿐.")
    inh = [(t, cb) for t in R["meta"]["tags"][1:] for cb in COMBOS
           if G(t, "std", cb, "minus_copy", "macro") and G(t, "std", cb, "minus_copy", "macro")[1] > 0
           and not excl0(G(t, "std", cb, "minus_release", "macro"))]
    if inh:
        L.append("> 7. **릴리즈에서 물려받은 copy 초과** — 이전: 학습 predictor 의 copy 초과 칸으로 나열 → 지금: " + ", ".join(
            f"{disp(t)} {cb} (p − copy {fci(G(t, 'std', cb, 'minus_copy', 'macro'), True)}, Δ vs release {fci(G(t, 'std', cb, 'minus_release', 'macro'), True)})" for t, cb in inh)
            + " 는 학습 효과가 아니라 릴리즈 몫이라고 표시한다.")
    fm = [abs(f["margin_ours"]) for v in R["validation"].values() if isinstance(v, dict) for e in v.values() if isinstance(e, dict) for f in e.get("flips", [])]
    if fm:
        L.append(f"> 8. **표 1 주석** — 이전: '뒤집힌 쌍은 전부 |margin| 이 ~1e-4 (창 잡음 중앙값) 보다 작은 동점 근처' → 지금: 뒤집힌 쌍 {len(fm)} 개의 배열 |margin| 최대 "
                 f"{max(fm):.1e}, 1e-4 이상 {sum(x >= 1e-4 for x in fm)} 개 — '전부 잡음 아래' 는 아니다. 배열 값과 하네스 (SCORES.md) 값이 칸 macro 에서 1 pt 안팎 다르다 (예: v11_e10 skip2_w32 "
                 f"{G('v11_e10', 'std', 'skip2_w32', 'macro')[0]:.2f} vs 하네스 {R['validation'].get('v11_e10', {}).get('skip2_w32', {}).get('harness_macro', float('nan')):.2f}).")
    L.append("> 9. **단서 추가** — tie 규칙 (정지+가림 인과 pre 는 동점이 소수다), copy 의 encoder 교차 (LN(online z) vs EMA target h), 칸 CI 다중비교, 그룹 B Tiny 의 ~54 해석.")
    L.append("> 10. **미적용** — (a) clean copy (target encoder 를 문맥 프레임에만 돌린 copy) 는 GPU 추출이 필요해 돌리지 않았다 — 교차 encoder 편차의 방향은 모른다. "
             "(b) 칸 CI 다중비교 보정 (FDR 등) 과 정지 칸 (scene 15) 의 bootstrap 반보수성 보정은 하지 않았다 — 단서로만. "
             "(c) seed 반복 학습은 재학습이 필요해 하지 않았다. (d) 튜블릿 bin 의 기본값은 first_div_strict 그대로 두고 (H6 과 비교 가능하게) sensitive 재 bin · 오염 쌍 제외 판을 옆에 붙였다.")
    return L


def READING(R):
    tags = R["meta"]["tags"]
    tub = R["tubelet"]
    L = []
    cp = {(tgt, cb): _g(R, "copy", tgt, cb, "macro") for tgt in ("std", "causal") for cb in COMBOS}
    if all(cp.values()):
        L.append("1. **복사 기준선이 이미 높다.** 예측 없이 문맥 마지막 튜블릿을 미래에 복사해도 표준 표적 macro 가 "
                 + " · ".join(f"{cb} {cp[('std', cb)][0]:.1f}" for cb in COMBOS) + " (인과 표적 "
                 + " · ".join(f"{cb} {cp[('causal', cb)][0]:.1f}" for cb in COMBOS) + "). "
                 "IntPhys 1 의 위반 (사라짐 · 나타남 · 모양 바뀜) 은 '미래가 마지막 관측에서 얼마나 멀어지나' 만으로 상당 부분 잡힌다. "
                 "그래서 p 의 점수가 높다는 것만으로는 예측이 들어갔다고 못 한다 — 비교 기준은 50 이 아니라 copy 다.")
    r = {cb: _g(R, "release", "std", cb, "minus_copy", "macro") for cb in COMBOS}
    rc = {cb: _g(R, "release", "causal", cb, "minus_copy", "macro") for cb in COMBOS}
    if all(r.values()):
        L.append(f"2. **릴리즈** — p − copy (표준 표적): " + " · ".join(f"{cb} {fci(r[cb], True)} ({_sig(r[cb])})" for cb in COMBOS)
                 + ". 인과 표적: " + " · ".join(f"{cb} {fci(rc[cb], True)} ({_sig(rc[cb])})" for cb in COMBOS)
                 + ". 88.89 로 알려진 칸 (skip2_w32, 격자 최고) 은 copy 도 가장 높은 칸이라 예측의 몫이 가장 작게 보이는 칸이기도 하다.")
    i = 3
    for t in tags[1:]:
        segs = []
        for cb in COMBOS:
            d = _g(R, t, "std", cb, "minus_release", "macro")
            m = _g(R, t, "std", cb, "minus_copy", "macro")
            mc = _g(R, t, "causal", cb, "minus_copy", "macro")
            if d is None:
                continue
            segs.append(f"{cb}: p {_g(R, t, 'std', cb, 'macro')[0]:.1f}, Δ vs release {fci(d, True)}, p − copy {fci(m, True)} / 인과 {fci(mc, True)}")
        L.append(f"{i}. **{disp(t)}** — " + "; ".join(segs) + ".")
        i += 1
    # ip1 의 이득이 인과 표적에서도 남는가
    for t in tags[1:]:
        if not t.startswith("ip1"):
            continue
        ms, mc = _g(R, t, "std", "skip2_w32", "minus_copy", "macro"), _g(R, t, "causal", "skip2_w32", "minus_copy", "macro")
        ds, dc = _g(R, t, "std", "skip2_w32", "minus_release", "macro"), _g(R, t, "causal", "skip2_w32", "minus_release", "macro")
        if ms and mc and ds and dc:
            L.append(f"{i}. **{disp(t)} 의 이득은 표적을 바꿔도 남는가** — skip2_w32 p − copy 표준 {fci(ms, True)} / 인과 {fci(mc, True)}, "
                     f"Δ vs release 표준 {fci(ds, True)} / 인과 {fci(dc, True)}. "
                     + ("인과 표적에서도 CI 가 0 위라서, 표준 표적의 사건 전 번짐 (표 5 pre) 만으로 설명되는 이득은 아니다. " if mc[1] > 0 and dc[1] > 0 else
                        "인과 표적에서는 CI 가 0 을 포함 — 표준 표적의 사건 전 번짐 (표 5 pre) 이 이득의 일부일 수 있다. ")
                     + "⚠️ 이 판단은 skip2_w32 한 칸에서만이다 — 인과 표적 Δ vs release 는 "
                     + " · ".join(f"{cb} {fci(_g(R, t, 'causal', cb, 'minus_release', 'macro'), True)}" for cb in COMBOS)
                     + " (정정 2026-09-25). 다만 학습 도메인 안 (같은 생성기의 train 장면 · 같은 창 격자) 이라 이것은 'IntPhys 장면 분포를 맞췄다' 이지 일반화가 아니다.")
            i += 1
    for t in tags[1:]:                                 # 정정 2026-09-25: Ariel 의 copy 비교를 표적별로 · 학습 지평
        if not t.startswith("ariel"):
            continue
        hz = R.get("future_tubelets_max", {})
        L.append(f"{i}. **{disp(t)} 와 copy** — 표준 표적 p − copy: " + " · ".join(f"{cb} {fci(_g(R, t, 'std', cb, 'minus_copy', 'macro'), True)}" for cb in COMBOS)
                 + " / 인과 표적: " + " · ".join(f"{cb} {fci(_g(R, t, 'causal', cb, 'minus_copy', 'macro'), True)}" for cb in COMBOS)
                 + ". '예측이 복사보다 못하다' 는 표준 표적에서만 세 칸 모두 성립하고, 인과 표적에서는 CI 로 낮은 칸이 "
                 + (", ".join(cb for cb in COMBOS if _g(R, t, "causal", cb, "minus_copy", "macro")[2] < 0) or "없음") + " 뿐이다. "
                 "Ariel 은 예측 지평 최대 8 slot (12 fps, MODELS.md) 으로 학습됐고 창이 요구하는 미래 튜블릿은 최대 "
                 + " · ".join(f"{cb} {hz[cb]}" for cb in COMBOS if cb in hz) + " 개라 skip2_w32 는 학습 지평 밖이다 — attention 규칙 · 데이터 · scratch · epoch 에 더해 "
                 "다섯째 교란이다. 이 결과를 attention 규칙의 성질로 읽지 않는다.")
        i += 1
    # 표적
    if all(cp.values()) and all(r.values()):
        L.append(f"{i}. **표적을 인과로 바꾸면** (창 안 미래를 보지 않는 target) copy 가 skip2_w32 {cp[('std', 'skip2_w32')][0]:.1f} → "
                 f"{cp[('causal', 'skip2_w32')][0]:.1f} 로 움직이고 릴리즈 p 는 {_g(R, 'release', 'std', 'skip2_w32', 'macro')[0]:.1f} → "
                 f"{_g(R, 'release', 'causal', 'skip2_w32', 'macro')[0]:.1f} 로 움직인다. 표준 표적은 사건 뒤 프레임이 사건 전 튜블릿 표적에 번져 들어간다. "
                 + (f"copy 가 그 덕을 보는 곳은 **보임** 칸 pre (copy 표준 정지+보임 {tub['static/visible|pre']['copy|std']['acc'][0]:.1f} · 이동+보임 "
                    f"{tub['moving/visible|pre']['copy|std']['acc'][0]:.1f}, 표 5c) 이고, 이동+가림 pre 에서는 copy 표준 {tub['moving/occluded|pre']['copy|std']['acc'][0]:.1f} "
                    "로 50 아래 — 거기서는 번짐이 copy 를 깎는다 (정정 2026-09-25; 이전 판은 표 5 의 pre 칸을 근거로 들었다). "
                    if all(k in tub for k in ('static/visible|pre', 'moving/visible|pre', 'moving/occluded|pre')) else "")
                 + "인과 표적의 p − copy 가 '예측의 몫' 에 더 가까운 읽기다.")
        i += 1
    e0 = tub.get("moving/occluded|e0")
    if e0:
        rel = e0["release|std"]; cpy = e0["copy|std"]
        gains = [f"{disp(t)} {fci(e0[f'{t}|std']['minus_release'], True)}" for t in tags[1:] if excl0(e0[f"{t}|std"].get("minus_release"))]
        L.append(f"{i}. **튜블릿 (이동+가림, 분기 튜블릿 e0)** — 릴리즈 p {rel['acc'][0]:.1f} vs copy {cpy['acc'][0]:.1f} "
                 f"(p − copy {fci(rel['minus_copy'], True)}, 표준 표적; 인과 {fci(e0['release|causal']['minus_copy'], True)}). "
                 "copy 가 chance 위라는 것은 (정의상) 이 튜블릿에서 imp 의 미래가 pos 의 미래보다 마지막 관측에서 더 멀다는 뜻이다. "
                 + ("Δ vs release 가 CI 로 갈리는 predictor (표준): " + ", ".join(gains) + ". " if gains else "Δ vs release 가 CI 로 갈리는 predictor 는 없다 (표준). ")
                 + ("인과 표적: " + ", ".join(f"{disp(t)} {fci(e0[f'{t}|causal']['minus_release'], True)}" for t in tags[1:]
                                             if excl0(e0[f"{t}|causal"].get("minus_release"))) + "."
                    if any(excl0(e0[f"{t}|causal"].get("minus_release")) for t in tags[1:]) else ""))
        i += 1
        pre = tub.get("moving/occluded|pre")
        _td = R.get("tubelet_drop_early_div", {})
        if pre:
            L.append(f"{i}. **사건 전 튜블릿 (pre)** 은 pos/imp 의 입력 프레임이 (오염 쌍을 빼면) 같아 정답이 나올 이유가 없다. 표준 표적에서 릴리즈 p {pre['release|std']['acc'][0]:.1f} · "
                     f"copy {pre['copy|std']['acc'][0]:.1f} 가 50 에서 벗어나는 것은 표적의 양방향 번짐 (창 안 사건 뒤 프레임을 target encoder 가 본다) 이다. "
                     f"인과 표적에서는 표적 입력도 대부분 같아 동점 (값이 다른 튜블릿 비율 {pre['copy|causal']['nonzero']:.3f}) 이라 tie = 0.5 로 ~50 이 된다. "
                     "⚠️ 정정 (2026-09-25): 비동점 몫은 fp16 잡음이 **아니다** — PNG 가 first_div_strict 보다 1–4 프레임 먼저 (first_div_sensitive 에서) 갈리는 쌍이 있어 "
                     "사건 bin 이 최대 한 튜블릿 어긋난다 (정정 블록 3 · 표 5b). "
                     + (f"first_div_sensitive 로 다시 bin 하면 이동+가림 인과 pre 비동점 {R['tubelet_first_div_sensitive']['moving/occluded|pre']['copy|causal']['nonzero']:.3f}. "
                        if 'moving/occluded|pre' in R.get('tubelet_first_div_sensitive', {}) else "")
                     + ("학습한 predictor 중 pre 가 크게 움직이는 것: "
                        + ", ".join(f"{disp(t)} {pre[f'{t}|std']['acc'][0]:.1f}" for t in tags[1:]
                                    if abs(pre[f'{t}|std']['acc'][0] - pre['release|std']['acc'][0]) >= 10)
                        + " (표준 표적). 이 튜블릿이 속한 창은 문맥 프레임이 (first_div_strict 기준) pos/imp 에서 같다 — 단 일부 쌍은 그 전에 이미 갈린다 (위 정정) — "
                          + (f"그 쌍을 빼도 {', '.join(f'{disp(t)} {_td['moving/occluded|pre'][f'{t}|std']['acc'][0]:.1f}' for t in tags[1:] if abs(pre[f'{t}|std']['acc'][0] - pre['release|std']['acc'][0]) >= 10)} (표 5b). "
                             if 'moving/occluded|pre' in R.get('tubelet_drop_early_div', {}) else "")
                          + "p 가 사건을 알아챈 것이 아니다. "
                          "p 가 '가능 영상의 h (번짐 포함)' 에 더 가까워진 것이고, 인과 표적에서는 정의상 사라지는 이득이다. 사건 탐지의 증거로 읽지 않는다."
                        if any(abs(pre[f'{t}|std']['acc'][0] - pre['release|std']['acc'][0]) >= 10 for t in tags[1:]) else ""))
            i += 1
    e0v, e0h = tub.get("moving/occluded|e0|ctxend_visible"), tub.get("moving/occluded|e0|ctxend_hidden")
    if e0v and e0h:
        L.append(f"{i}. **e0 를 문맥 끝 가시성으로 (H6d)** — 릴리즈 p − copy: 문맥 끝에 물체가 보임 {fci(e0v['release|std']['minus_copy'], True)} · 가려짐 "
                 f"{fci(e0h['release|std']['minus_copy'], True)} (copy 자체 {e0v['copy|std']['acc'][0]:.1f} / {e0h['copy|std']['acc'][0]:.1f}). "
                 + "학습 predictor: " + "; ".join(f"{disp(t)} {e0v[f'{t}|std']['minus_copy'][0]:+.1f} / {e0h[f'{t}|std']['minus_copy'][0]:+.1f}" for t in tags[1:])
                 + " (보임 / 가려짐, p − copy). 방향별 (appear · disappear) 은 json.")
        i += 1
    gb = R.get("group_B", {})
    okb = {k: v for k, v in gb.items() if "skip2_w32" in v}
    if okb:
        top = max(okb, key=lambda k: okb[k]["skip2_w32"]["macro"][0])
        L.append(f"{i}. **그룹 B (맥락, 하네스만)** — skip2_w32 macro: " + " · ".join(f"{k} {v['skip2_w32']['macro'][0]:.1f}" for k, v in okb.items())
                 + f". 최고는 {top} ({okb[top]['skip2_w32']['macro'][0]:.1f}, 합성 물리 데이터 · 렌더 도메인이 IntPhys 와 가깝다 — MODELS.md). "
                 "자기 encoder 의 copy 가 없어서 이 점수 중 얼마가 예측의 몫인지는 **모른다** — 그룹 A 의 copy (85.0) 와 비교하지 않는다.")
        i += 1
    cv = R["curves"]
    if cv["points"]:
        src = "배열" if cv["status"].get("source") == "arrays" else "하네스 점"
        pts = cv["points"]
        seq = []
        for run in ("v11", "pv1", "ip1", "ip2", "ariel"):
            ks = sorted([k.split("|")[0] for k in pts if k.startswith(run + "_e") and k.endswith("|std|skip2_w32")],
                        key=lambda s: int("".join(ch for ch in s.rsplit("_e", 1)[1] if ch.isdigit())))
            if ks:
                seq.append(f"{run}: " + " → ".join(f"{k.rsplit('_', 1)[1]} {pts[k + '|std|skip2_w32']['macro'][0]:.1f}" for k in ks))
        if "release|std|skip2_w32" in pts:
            seq.insert(0, f"release {pts['release|std|skip2_w32']['macro'][0]:.1f}")
        L.append(f"{i}. **학습 곡선 ({src}, skip2_w32 표준 표적 macro)** — " + " · ".join(seq) + ".")
        seq2 = []
        for run in ("v11", "pv1", "ip1", "ip2", "ariel"):
            ks = sorted([k.split("|")[0] for k in pts if k.startswith(run + "_e") and k.endswith("|causal|skip2_w32")],
                        key=lambda s_: int("".join(ch for ch in s_.rsplit("_e", 1)[1] if ch.isdigit())))
            if ks:
                seq2.append(f"{run}: " + " → ".join(f"{k.rsplit('_', 1)[1]} {pts[k + '|causal|skip2_w32']['minus_copy'][0]:+.1f}" for k in ks))
        if seq2:
            rc = pts.get("release|causal|skip2_w32", {}).get("minus_copy")
            L.append(f"   같은 칸 인과 표적 p − copy: " + (f"release {rc[0]:+.1f} · " if rc else "") + " · ".join(seq2)
                     + ". CI 는 표 7 (여기는 추정치만).")
        r0 = pts.get("release|std|skip2_w32", {}).get("macro")
        fs = []
        for run in ("v11", "pv1", "ip1", "ip2"):
            ks = sorted([k.split("|")[0] for k in pts if k.startswith(run + "_e") and k.endswith("|std|skip2_w32")],
                        key=lambda s_: int("".join(ch for ch in s_.rsplit("_e", 1)[1] if ch.isdigit())))
            if r0 and len(ks) >= 2:
                a1 = pts[ks[0] + "|std|skip2_w32"]["macro"][0] - r0[0]; aT = pts[ks[-1] + "|std|skip2_w32"]["macro"][0] - r0[0]
                fs.append((run, ks[0].rsplit('_', 1)[1], a1, ks[-1].rsplit('_', 1)[1], aT))
        if fs:
            L.append("   release 대비 변화 (skip2_w32 표준, pt): " + " · ".join(f"{r} 첫 점 ({e1}) {a:+.1f} / 끝 ({eT}) {b:+.1f}" for r, e1, a, eT, b in fs) + ".")
        stp = cv.get("steps", {})
        if stp:
            def _sg(cb):
                out_ = []
                for k, v in stp.items():
                    if f"|std|{cb}|" not in k or not excl0(v["delta"]):
                        continue
                    a_, b_ = k.split("|")[-1].split("->")
                    out_.append(f"{k.split('|')[0]} {a_.rsplit('_', 1)[-1] if a_ != 'release' else 'release'}→{b_.rsplit('_', 1)[1]} {fci(v['delta'], True)}")
                return ", ".join(out_) or "없음"
            L.append("   ⚠️ 정정 (2026-09-25) — 이전 판은 'v11 · ip1 는 변화가 첫 체크포인트 안에서 일어나고 그 뒤 평평하다' 를 skip2_w32 한 칸에서 골랐다. "
                     "체크포인트 사이 변화를 같은 쌍 paired CI 로 재면 (표 7b, 표준 표적) CI 로 갈리는 구간: skip2_w32 — " + _sg("skip2_w32")
                     + " / skip2_w16 — " + _sg("skip2_w16") + ". 즉 skip2_w16 에서는 첫 점 뒤에도 더 떨어지는 run 이 있고, '더 오래 학습해서 생긴 변화가 아니다' 로 읽지 않는다. "
                     "(체크포인트 간격이 run 마다 달라 첫 점의 epoch 이 다르다 — 곡선 사이 속도 비교는 하지 않는다.)")
    return L


def CAVEATS(R):
    n = R["meta"]["n_pairs"]
    L = [
        f"- **표본**: 쌍 {n} · scene 90 (principle 당 30). 운동×가림 칸은 15 (정지) · 30 (이동) scene 이다. CI 는 scene cluster bootstrap 이라 칸 CI 가 넓다 — "
        "칸 사이 차이는 CI 가 겹치면 읽지 않는다. 폭 0 인 CI (+0 [+0, +0], +50 [+50, +50]) 는 모든 scene 이 같은 값인 천장 · 바닥 구조다 (CLAUDE.md §7-3).",
        "- **격자 최고 칸은 descriptive** (CLAUDE.md §1-3) — 세 칸을 모두 적었고, 칸을 고른 뒤의 값은 held-out 추정치가 아니다. "
        "칸마다 p − copy 의 부호가 다를 수 있으므로 한 칸으로 요약하지 않는다.",
        "- **ip1_e40 은 학습 도메인 안이다** — IntPhys 2019 train 장면 (dev 와 같은 생성기 · 물체 분포) 으로 학습했고, 창 격자도 Garrido 채점 격자 그대로 학습했다 "
        "(MODELS.md). 이 predictor 의 IntPhys1 이득은 일반화 증거가 아니다. 나머지 넷은 IntPhys1 에 대해 도메인 밖이다.",
        "- **copy 는 encoder 만의 함수** — 그룹 A 여섯이 같은 copy 를 공유하므로 p − copy 의 차이는 전부 predictor 몫이다. "
        "그룹 B 는 encoder 가 달라 이 copy 를 쓸 수 없고, IntPhys1 에서 자기 copy 를 뽑지 않았다 → 그룹 B 는 하네스 점수만 (맥락용).",
        "- **표준 표적의 번짐** — h 는 창 전부를 양방향으로 본 target encoder 출력이라 사건 전 튜블릿에도 사건 뒤 정보가 섞인다. copy 도 그 덕을 본다. "
        "인과 표적은 이를 막지만 공식 채점 (88.89) 과 다른 양이다 — 두 표적을 같이 적었다.",
        "- **튜블릿 CI** — 한 쌍이 여러 창 · 여러 C 에서 같은 튜블릿을 반복해 세므로 튜블릿 n (수천) 은 독립 표본 수가 아니다. CI 는 scene 단위 (이동+가림 30 scene · 60 쌍).",
        "- **배열 vs 하네스** — 배열은 영상 하나의 시작점을 한 배치로 돌려 fp16 GEMM 선택이 공식 실행과 다르다 (창 값 중앙값 ~1e-4). "
        "그래서 동점 근처 쌍 몇 개가 뒤집혀 칸 macro 가 하네스와 1~2 pt 다를 수 있다 (표 1). p − copy 는 같은 실행 안의 비교라 이 잡음에 짝지어져 있다.",
        "- **tie 규칙** — 쌍 단위는 garrido_rescore 대로 strict (동점 = 오답); 동점이 있으면 tie = 0.5 판을 같이 적는다. 튜블릿 단위 그림은 H6g 와 같은 tie = 0.5 다. "
        "⚠️ 정정 (2026-09-25): 인과 표적 pre 의 동점 비율은 칸마다 크게 다르다 — copy 비동점 비율 "
        + " · ".join(f"{MV_SHORT[g]} {R['tubelet'][g + '|pre']['copy|causal']['nonzero']:.3f}" for g in MV if g + "|pre" in R["tubelet"])
        + " (정지+가림은 PNG 가 first_div_strict 전에 갈리는 쌍 때문에 동점이 소수다; 정정 블록 3).",
        "- **copy 의 encoder 교차** — copy = LN(online encoder z 의 마지막 문맥 튜블릿) 을 EMA target encoder 의 h 에 대고 잰 값이다. 두 encoder 사이 편차가 copy 에 섞이며, "
        "target encoder 를 문맥 프레임에만 돌린 clean copy 는 뽑지 않았다 (H6 문서와 같은 단서) — 편차가 copy 를 올리는지 내리는지 모른다.",
        "- **다중비교** — 표 3a · 3b · 4 의 칸 CI 는 수백 개이고 보정하지 않았다. 5 % 면 우연히 0 을 벗어나는 칸이 여럿 나온다. 정지 칸은 scene 15 개라 percentile cluster bootstrap 이 "
        "반보수적이다 (+13 [+0, +30] 처럼 0 에 닿는 CI 가 많다). 칸 단위 차이는 macro · 여러 칸에서 같은 방향일 때만 읽는다.",
        "- **A/B 방향** — IntPhys 1 의 O1 · O3 쌍은 appear / disappear 로 갈리고 모든 방법이 appear 쪽이 거의 천장이다 (표 2c). 칸 macro 는 두 방향의 평균이라 방향 편향을 숨긴다.",
        "- **학습은 한 번씩 (seed 0)** — 학습 분산을 모른다. 오른 · 내린 칸이 학습 설정의 차이인지 seed 운인지 가를 수 없다. "
        "Ariel 은 attention 규칙 · 데이터 · scratch · epoch 가 동시에 달라 원인을 하나로 좁히지 않는다 (MODELS.md). 원인을 목적함수로 돌리지 않는다.",
        "- **H6g 대조** — H6g 는 다른 실행 (H6e, 모델 window 64) · tie = 0.5 · 다른 bootstrap 난수라 CI 끝값이 조금 다르다. 추정치가 같은지만 본다.",
        "- **그룹 B Tiny 의 ~54** — scene 당 두 쌍 구조라 방향 편향 하나만으로 scene 당 약 1/2 을 맞춘다. 좁은 CI 의 54 는 '약한 신호' 가 아니라 방향 편향의 바닥으로 읽는다 (하네스 대조로 버그 아님 확인 — 검증 원문).",
    ]
    cv = R["curves"]
    if cv["status"].get("source") != "arrays":
        L.append(f"- **학습 곡선은 미완** — `ip1score_curves` 추출 (22 predictor) 이 이 문서 시점에 끝나지 않았다 (shard {cv['status']['n_shards']}/{N_VIDEOS}). "
                 "표 7 은 하네스 per_window 가 있는 점만 (v11 e10 · pv1 e15 · ip1 e20/e40 · ip2 e80 · Ariel ep5/19/30/41/43) 이고, p − copy 의 copy 는 final 배열 것을 "
                 "빌렸다 (같은 encoder 라 정당하지만 p 와 copy 가 다른 실행이라 fp16 잡음이 짝지어지지 않는다). 인과 표적 곡선 · 초기 epoch (e1 · e2 · e3 · e5) 는 없다.")
    return L


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(CACHE_ROOT / "ip1score_final"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no_fig", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    boot = Boot(a.B)

    man, arr, lay, src = load_arrays(Path(a.cache))
    tags = man["tags"]
    pairs = load_pairs()
    meta = check_pairs_vs_index(pairs, man)
    vix = {v: i for i, v in enumerate(arr["video_ids"])}
    pairs_used = [(p, vix[p["pos"]], vix[p["imp"]]) for p in pairs if p["pos"] in vix and p["imp"] in vix]
    P = {"pi": np.array([x[1] for x in pairs_used]), "ii": np.array([x[2] for x in pairs_used]),
         "scene": np.array([x[0]["scene"] for x in pairs_used]), "pr": np.array([x[0]["pr"] for x in pairs_used]),
         "mv": np.array([x[0]["mv"] for x in pairs_used]), "dir": np.array([x[0]["dir"] for x in pairs_used])}
    print(f"영상 {len(vix)} · 쌍 {len(pairs_used)} · predictor {tags}", flush=True)
    scorer = Scorer(lay)
    G_all = {}
    for k, t in enumerate(tags):
        for tgt in ("std", "causal"):
            G_all[(t, tgt)] = scorer.G(arr[f"p_{tgt}"][k])
    for tgt in ("std", "causal"):
        G_all[("copy", tgt)] = scorer.G(arr[f"copy_{tgt}"])

    cells, O = pair_level(G_all, P, boot)
    print("pair level 끝", flush=True)
    flagged = []
    tub = tubelet_level(arr, lay, man, P, [x for x in pairs_used], boot, flag_out=flagged)
    # 정정 2026-09-25: PNG 가 first_div_strict 보다 먼저 갈리는 쌍 → first_div_sensitive 로 다시 bin / 그 쌍을 뺀 판
    tub_sens = tubelet_level(arr, lay, man, P, pairs_used, boot, fcol="first_div_sensitive", groups=["static/occluded", "moving/occluded"])
    tub_drop = tubelet_level(arr, lay, man, P, pairs_used, boot, groups=["moving/occluded"], drop_pairs={(a_, b_) for a_, b_, _, _ in flagged})
    early = {}
    for g in MV:
        ps = [x[0] for x in pairs_used if x[0]["mv"] == g]
        ee = [q for q in ps if int(q["first_div_sensitive"]) < int(q["first_div_strict"])]
        fl = [f for f in flagged if f[2] == g]
        early[g] = {"n_pairs_early": len(ee), "n_pairs": len(ps), "scenes_early": sorted({q["scene"] for q in ee}),
                    "n_scene": len({q["scene"] for q in ps}), "n_pairs_pre_nontie": len(fl), "scenes_pre_nontie": sorted({f[3] for f in fl})}
    minC = collections.defaultdict(lambda: 10 ** 9)
    for k_ in lay.win_key:
        minC[k_[0]] = min(minC[k_[0]], int(k_[2]))
    horizon = {cb["name"]: int(cb["W"] // 2 - minC[cb["name"]] // 2) for cb in lay.combos if cb["name"] in minC}
    print("tubelet 끝", flush=True)
    val = validation(man, arr, lay, scorer, P, pairs_used, meta)
    print("validation 끝", flush=True)
    full_pairs = [p for p in pairs]
    Pfull = {"scene": np.array([p["scene"] for p in full_pairs]), "pr": np.array([p["pr"] for p in full_pairs]),
             "mv": np.array([p["mv"] for p in full_pairs])}
    gB = harness_table(GROUP_B, Pfull, full_pairs, boot)
    gA_h = harness_table([HARNESS[t] for t in FINAL], Pfull, full_pairs, boot)
    O_final = O if len(pairs_used) == 180 else None
    cv = curves(Pfull, full_pairs, boot, O_final, arr if man["tags"] == FINAL else None)
    print("group B · curves 끝", flush=True)

    R = {"meta": {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "arrays_src": src, "tags": tags,
                  "n_videos": len(vix), "n_pairs": len(pairs_used), "B": a.B,
                  "rule": "o = 1[G(imp) > G(pos)] strict (garrido_rescore); macro = mean O1/O2/O3; "
                          "CI = principle-stratified scene bootstrap (macro) / scene cluster bootstrap (cells, tubelets); "
                          "p − copy, Δ vs release = paired on identical pairs/tubelets",
                  "value_format": "[estimate %, CI lo, CI hi, n, n_scene(, n_strata)]"},
         "validation": val, "cells": cells, "tubelet": tub, "tubelet_first_div_sensitive": tub_sens,
         "tubelet_drop_early_div": tub_drop, "early_div_pairs": early, "future_tubelets_max": horizon, "group_B": gB, "group_A_harness": gA_h, "curves": cv}
    R["release_vs_h6g"] = release_vs_h6g(cells, tub)
    R["builtin_check"] = builtin_check(cells, len(vix))
    R["ties"] = explain_ties(O, lay, man, pairs_used)
    R["headline_lines"] = narrative(R)
    R["correction_lines"] = CORRECTIONS(R)
    R["headline"] = HEADLINE(R)
    R["reading_lines"] = READING(R) if "READING" in globals() else ["*(작성 중)*"]
    R["caveat_lines"] = CAVEATS(R) if "CAVEATS" in globals() else ["*(작성 중)*"]
    figs = []
    if not a.no_fig:
        fig_minus_copy(cells, tags, out / "fig_p_minus_copy.png")
        figs.append(("fig_p_minus_copy.png", "칸 3 개 × predictor: macro p − copy [95 % CI], 표준 (파랑) · 인과 (주황) 표적"))
        fig_cells(cells, tags, "minus_copy", out / "fig_cells_minus_copy.png", "acc(p) − acc(copy), pt")
        figs.append(("fig_cells_minus_copy.png", "표준 표적 p − copy 를 O1/O2/O3 · 운동×가림 칸으로 (굵게 = CI 가 0 제외)"))
        fig_cells(cells, tags, "minus_release", out / "fig_cells_minus_release.png", "acc(pred) − acc(release), pt")
        figs.append(("fig_cells_minus_release.png", "표준 표적 Δ vs release 를 같은 칸으로"))
        if tub:
            fig_tubelet(tub, tags, out / "fig_tubelet_moving_occluded.png")
            figs.append(("fig_tubelet_moving_occluded.png", "이동+가림 pre / e0 / post 튜블릿 정답률, predictor (파랑, release 검정) vs copy (회색 선 · 띠)"))
        if cv["points"]:
            fig_curves(cv, out / "fig_curves.png")
            figs.append(("fig_curves.png", "학습 곡선 — 윗줄 macro (표준 표적; skip2_w32 파랑 · skip2_w16 주황, 점선 = copy), 아랫줄 skip2_w32 p − copy (표준 파랑 · 인과 주황), 띠 = 95 % CI"
                         + ("" if cv["status"].get("source") == "arrays" else " — 하네스 점만 (curves 추출 대기)")))
    R["figures"] = figs
    json.dump(R, open(out / "ip1_copy.json", "w"), indent=1, ensure_ascii=False, default=float)
    write_md(R, out / "IP1_COPY.md")
    for s in R["headline_lines"]:
        print(s)
    print("->", out / "IP1_COPY.md")


if __name__ == "__main__":
    main()
