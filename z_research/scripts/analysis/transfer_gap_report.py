#!/usr/bin/env python3
"""[X1] E4 · E6 · E7 집계 (CPU) — 판독 회복률 R · 정렬 공간 VoE · 주입 VoE (2026-10-30).

입력: results/q1q2/E4_reads_<objset>.npz (transfer_gap_ladder.py eval) · E6E7_rows_<objset>.jsonl · E3_heads_<objset>.json
정의:
  판독 정확도 = 결합 (모양/색 + 없음) 정답률, 시드 0 · 1 평균. 모양 = 예측 클래스의 모양이 맞음. 없음 % = '없음' 이라고 답한 비율.
  R(M) = (acc_M − acc_L0)/(acc_천장 − acc_L0), 천장 = 같은 표본에서 p 판독기 (보임 조건 p 로 학습) 의 L0 정확도. 같은 시점 판독기 (S3) 는 p_tX.
  VoE: matched 쌍 (pos_a, imp_ab) = 방향 A · (pos_b, imp_ba) = 방향 B. 정답 = s(imp) > s(pos). Δ = 사상 M − L0raw (원값 p, 공식 채점).
  CI = block bootstrap 2000 (95 %). Holm 보정은 한 가족 (late · 이동 · 방향 A 의 사상들) 안에서.
출력: results/q1q2/E4_ladder.json · E6_rescore.json · E7_inject.json

  $P z_research/scripts/analysis/transfer_gap_report.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

HEADN = ["h", "p", "h_t8", "h_t11", "h_t15", "p_t8", "p_t11", "p_t15"]
STATE = ["vis", "occ", "mix"]


def boot_ratio(num_a, num_b, den_c, groups, n=2000, seed=0):
    """R = (mean a − mean b)/(mean c − mean b) 의 block bootstrap."""
    a, b, c, g = map(np.asarray, (num_a, num_b, den_c, groups))
    ub = np.unique(g); idx = {x: np.where(g == x)[0] for x in ub}
    A = np.array([a[idx[x]].sum() for x in ub]); B = np.array([b[idx[x]].sum() for x in ub]); Cc = np.array([c[idx[x]].sum() for x in ub])
    N = np.array([len(idx[x]) for x in ub])
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.integers(0, len(ub), len(ub)); nn = N[s].sum()
        ma, mb, mc = A[s].sum() / nn, B[s].sum() / nn, Cc[s].sum() / nn
        out.append((ma - mb) / (mc - mb) if abs(mc - mb) > 1e-9 else np.nan)
    out = np.array(out)
    den = c.mean() - b.mean()
    return dict(R=float((a.mean() - b.mean()) / den) if abs(den) > 1e-9 else None, lo=float(np.nanpercentile(out, 2.5)), hi=float(np.nanpercentile(out, 97.5)))


def ladder(objset, M):
    z = np.load(C.RES / f"E4_reads_{objset}.npz", allow_pickle=True)
    R = z["reads"]; maps = list(z["mapnames"]); labs = list(z["classes"]); NONE = int(z["none"])
    shape = np.array([l.split("/")[0] for l in labs])
    # 표본 키 (j, t) → 시드 평균 정답
    tab = defaultdict(lambda: defaultdict(list))   # (head, map) → (j,t) → [correct...], [shape], [none]
    for j, t, mi, hi, seed, pr, gt, stt in R:
        tab[(HEADN[hi], maps[mi])][(j, t)].append((pr == gt, shape[pr] == shape[gt], pr == NONE))
    meta = lambda j: M[j]
    def scen_keys(head, f):
        ks = [k for k in tab[(head, "L0")] if f(k)]
        return sorted(ks)
    st_of = {}
    for j, t, mi, hi, seed, pr, gt, stt in R:
        st_of[(j, t)] = STATE[stt]
    empty = lambda j: C.label_of(M[j]) == "empty"
    late = lambda j: M[j]["occlusion"] == "late"
    mov = lambda j: M[j]["motion"] != "flat_static"
    scen = {
        "S1_vis_move_t8-10": lambda k: not empty(k[0]) and not late(k[0]) and mov(k[0]) and st_of[k] == "vis" and k[1] <= 10,
        "S2_vis_move_t11-15": lambda k: not empty(k[0]) and not late(k[0]) and mov(k[0]) and st_of[k] == "vis" and k[1] >= 11,
        "S4r_late_move_reapp": lambda k: not empty(k[0]) and late(k[0]) and mov(k[0]) and st_of[k] == "vis",
        "S4o_late_move_occ": lambda k: not empty(k[0]) and late(k[0]) and mov(k[0]) and st_of[k] == "occ",
        "static_vis": lambda k: not empty(k[0]) and not late(k[0]) and not mov(k[0]) and st_of[k] == "vis",
        "static_late_reapp": lambda k: not empty(k[0]) and late(k[0]) and not mov(k[0]) and st_of[k] == "vis",
        "empty_vis_cond": lambda k: empty(k[0]) and not late(k[0]),
        "empty_late": lambda k: empty(k[0]) and late(k[0]),
    }
    out = {}
    allmaps = [m for m in maps if not m.startswith("onH_") and m != "H"]
    for sn, f in scen.items():
        ks = scen_keys("h", f)
        if not ks:
            continue
        grp = [M[j]["block_id"] for j, _ in ks]
        cell = dict(n=len(ks), n_block=len(set(grp)), maps={})
        ceil_k = [tab[("p", "L0")].get(k) for k in ks]
        ceil = np.array([np.mean([x[0] for x in v]) if v else np.nan for v in ceil_k])
        base = np.array([np.mean([x[0] for x in tab[("h", "L0")][k]]) for k in ks])
        cell["ceiling_p_reader_L0"] = 100 * float(np.nanmean(ceil))
        cell["ceiling_p_reader_shape"] = 100 * float(np.nanmean([np.mean([x[1] for x in v]) if v else np.nan for v in ceil_k]))
        cell["ceiling_p_reader_none"] = 100 * float(np.nanmean([np.mean([x[2] for x in v]) if v else np.nan for v in ceil_k]))
        for mn in allmaps:
            if (("h", mn)) not in tab or not all(k in tab[("h", mn)] for k in ks):
                continue
            acc = np.array([np.mean([x[0] for x in tab[("h", mn)][k]]) for k in ks])
            shp = np.array([np.mean([x[1] for x in tab[("h", mn)][k]]) for k in ks])
            non = np.array([np.mean([x[2] for x in tab[("h", mn)][k]]) for k in ks])
            r = dict(acc=C.bootstrap_mean(100 * acc, grp), shape=100 * float(shp.mean()), none=100 * float(non.mean()))
            if not sn.startswith("empty") and not np.isnan(ceil).any():
                r["R"] = boot_ratio(acc, base, ceil, grp)
            cell["maps"][mn] = r
        out[sn] = cell
    # S3 같은 시점 판독기
    for X in (8, 11, 15):
        hn, pn = f"h_t{X}", f"p_t{X}"
        ks = sorted(k for k in tab[(hn, "L0")] if k[1] == X and not empty(k[0]) and not late(k[0]) and st_of[k] == "vis")
        if not ks:
            continue
        grp = [M[j]["block_id"] for j, _ in ks]
        ceil = np.array([np.mean([x[0] for x in tab[(pn, "L0")][k]]) for k in ks])
        base = np.array([np.mean([x[0] for x in tab[(hn, "L0")][k]]) for k in ks])
        cell = dict(n=len(ks), ceiling_p_tX=100 * float(ceil.mean()), maps={})
        for mn in allmaps:
            if all(k in tab[(hn, mn)] for k in ks):
                acc = np.array([np.mean([x[0] for x in tab[(hn, mn)][k]]) for k in ks])
                cell["maps"][mn] = dict(acc=100 * float(acc.mean()), none=100 * float(np.mean([np.mean([x[2] for x in tab[(hn, mn)][k]]) for k in ks])),
                                        R=boot_ratio(acc, base, ceil, grp))
        out[f"S3_same_time_t{X}"] = cell
    # 대조: 사상을 h 에
    ks = sorted(k for k in tab[("h", "H")] if not empty(k[0]) and st_of[k] == "vis")
    out["control_maps_on_h"] = {mn: 100 * float(np.mean([np.mean([x[0] for x in tab[("h", mn)][k]]) for k in ks]))
                                for mn in ["H"] + [m for m in maps if m.startswith("onH_")]}
    return out


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def voe(objsets, M):
    rows = []
    for o in objsets:
        rows += [json.loads(l) for l in open(C.RES / f"E6E7_rows_{o}.jsonl")]
    for r in rows:
        r["objset"] = M[r["j"]]["objset"]
    names = sorted({k[4:] for r in rows for k in r if k.startswith("pos_")})
    names = [n for n in names if f"imp_{n}" in rows[0] or any(f"imp_{n}" in r for r in rows)]

    def corr(r, n):
        return float(r[f"imp_{n}"] > r[f"pos_{n}"]) if f"pos_{n}" in r else np.nan

    def cellstats(sel, n, base="L0raw"):
        v = np.array([corr(r, n) for r in sel]); b = np.array([corr(r, base) for r in sel]); g = [r["block"] for r in sel]
        if np.isnan(v).any():
            return None
        d = v - b
        bm = C.bootstrap_mean(100 * d, g)
        # 단측 p (Δ ≤ 0) — bootstrap
        ub = np.unique(g); idx = {x: np.where(np.array(g) == x)[0] for x in ub}
        rng = np.random.default_rng(1); bs = []
        sums = np.array([d[idx[x]].sum() for x in ub]); cn = np.array([len(idx[x]) for x in ub])
        for _ in range(2000):
            s = rng.integers(0, len(ub), len(ub)); bs.append(sums[s].sum() / cn[s].sum())
        p = float((np.array(bs) <= 0).mean())
        return dict(acc=100 * float(v.mean()), delta=bm, p_le0=p, n=len(sel), n_block=len(ub))

    cells = {
        "late_all": lambda r: r["occlusion"] == "late",
        "late_move": lambda r: r["occlusion"] == "late" and r["motion"] != "flat_static",
        "late_static": lambda r: r["occlusion"] == "late" and r["motion"] == "flat_static",
        "vis_cond_all": lambda r: r["occlusion"] != "late",
        "vis_cond_move": lambda r: r["occlusion"] != "late" and r["motion"] != "flat_static",
    }
    out6, out7 = {}, {}
    for o in list(objsets) + ["both"]:
        for split in ("test", "all"):
            for cn, f in cells.items():
                for viol in ("all", "permanence", "shape", "color"):
                    for dirn in ("all", "A", "B"):
                        sel = [r for r in rows if (o == "both" or r["objset"] == o) and f(r) and (split == "all" or r["test"])
                               and (viol == "all" or r["violation"] == viol) and (dirn == "all" or r["dirn"] == dirn)]
                        if not sel:
                            continue
                        key = f"{o}|{split}|{cn}|{viol}|{dirn}"
                        base = dict(L0raw=100 * float(np.mean([corr(r, "L0raw") for r in sel])),
                                    copy=100 * float(np.mean([r["copy_imp"] > r["copy_pos"] for r in sel])), n=len(sel))
                        e6 = dict(base)
                        for n in names:
                            if n.startswith("d") and len(n) == 2:
                                continue
                            s_ = cellstats(sel, n)
                            if s_:
                                e6[n] = s_
                        out6[key] = e6
                        if cn.startswith("late"):
                            e7 = dict(base)
                            for n in ("d1", "d2", "d3", "d4"):
                                s_ = cellstats(sel, n)
                                if s_:
                                    e7[n] = s_
                            out7[key] = e7
    # Holm: late_move A (test, both objsets, viol all) 사상 가족
    fam = out6.get("both|test|late_move|all|A", {})
    mnames = [k for k, v in fam.items() if isinstance(v, dict) and "p_le0" in v and not k.startswith("PFT")]
    if mnames:
        adj = holm(np.array([fam[k]["p_le0"] for k in mnames]))
        out6["_holm_late_move_A_test"] = {k: float(a) for k, a in zip(mnames, adj)}
    return out6, out7


if __name__ == "__main__":
    M = C.meta()
    objsets = [o for o in ("primitive", "realistic") if (C.RES / f"E4_reads_{o}.npz").exists()]
    lad = {o: ladder(o, M) for o in objsets}
    for o in objsets:
        try:
            lad[o]["_maps_meta"] = {k: v for k, v in __import__("torch").load(C.MAPS / f"{o}.pt", map_location="cpu").items()
                                    if k in ("gam", "lam_factor", "lam_mse", "polar", "n_tok", "n_tok_late", "l5_losses", "l4_train_l1")}
            lad[o]["_maps_meta"]["gam_t"] = __import__("torch").load(C.MAPS / f"{o}.pt", map_location="cpu")["gam_t"].tolist()
        except Exception as e:  # noqa
            print("maps meta", e)
    json.dump(lad, open(C.RES / "E4_ladder.json", "w"), indent=1, ensure_ascii=False, default=float)
    e6, e7 = voe(objsets, M)
    json.dump(e6, open(C.RES / "E6_rescore.json", "w"), indent=1, ensure_ascii=False, default=float)
    json.dump(e7, open(C.RES / "E7_inject.json", "w"), indent=1, ensure_ascii=False, default=float)
    print("→ E4_ladder.json · E6_rescore.json · E7_inject.json")
