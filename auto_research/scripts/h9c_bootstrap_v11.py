#!/usr/bin/env python3
"""H9c — H9 (frozen 개입 채점, v11) 의 적대 검증: 변형 − clean 차이의 block bootstrap CI.

무엇: H9 의 l1.npy (n, 4 변형, 8 슬롯) 로 matched pair 정확도를 다시 계산하고, 변형마다 clean 과의 **짝지은 차이**
      Δ = acc(변형) − acc(clean) 의 95 % CI 를 **block 단위 재표집** 으로 낸다 (block 안 A/B 두 쌍은 같은 장면을 공유하므로 단위는 block).
왜:  H9 초판은 CI 없이 셀별 점수만 냈고, 최대 향상 셀을 골라 결론을 세웠다 (다중 비교). "정지는 그대로" 도 검정 없이 말했다.
식:
  S_v(clip) = mean_{j∈창} mean|p^v_j − LN(h_full)_j|      (창 = 슬롯 0–7 · 0–3 · 4–7 · 슬롯 j 하나)
  ok_v(pair) = 1[S_v(imp) > S_v(pos)] + ½·1[같음]
  acc_v(G) = Σ_{pair∈G} ok_v / |G|;   Δ_v(G) = acc_v(G) − acc_clean(G)
  bootstrap: G 의 block 을 복원추출 (B 회, multinomial 가중치 w_b) → acc*_v = Σ_b w_b s_{v,b} / Σ_b w_b n_b
             같은 w 로 clean 과 변형을 함께 계산 (짝지은 차이). 95 % CI = 2.5 / 97.5 백분위, p_boot = 2·min(P(Δ*≤0), P(Δ*≥0)).
그룹: timing(4|*) × motion(3|*) × 위반(3|*) × 방향(A|B|*). 방향 A = pos_a vs imp_ab (vanish 에서는 물체→빈),
      B = pos_b vs imp_ba (vanish 에서는 빈→물체). shape · color 의 A/B 는 "어느 모양/색이 먼저인가" 일 뿐 의미 있는 방향이 아니다.
참고 변형: H6f 의 copyz_full (문맥 마지막 튜블릿 LN(z) 복사, 같은 표본) 이 있으면 함께 싣는다 (clean 과 같은 bootstrap 가중치).
추가: contrasts (single − 복사 · single − mmx · mm_L0-6 − clean, 복사와의 판정 일치율 mean 1[ok_v == ok_copy]),
      holm (48 칸 가족에 Holm 보정), level (가능 clip L1 과 상대 margin (S_imp − S_pos)/S_pos — 개입이 예측을 전반적으로 흐리나).
검증: clean 슬롯 평균 = 표준 per_video surprise (v11_full per_block.json), 같은 쌍 표준 채점과의 판정 차이 수; H6f 표본의 video 순서 일치.
실행: srun6.sh 8 32G python h9c_bootstrap_v11.py   (vll6, CPU)
"""
import collections, itertools, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h9")
I6F = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f")
REF = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
OUT = ROOT / "auto_research/exp_results/h9"; OUT.mkdir(parents=True, exist_ok=True)
NB = 4000; SEED = 0

t0 = time.time()
meta = json.load(open(I / "meta.json")); n = meta["n"]; L = np.load(I / "l1.npy"); V = list(meta["vars"])
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
assert np.isfinite(L).all(), "l1.npy 에 NaN (미완 실행)"
ref = json.load(open(REF))["per_video_surprise"]; r = np.array([ref[v] for v in M["video_id"]])
dv = np.abs(L[:, 0].mean(1) - r)
check = {"clean_vs_standard_max": float(dv.max()), "clean_vs_standard_median": float(np.median(dv))}
print(f"검증 clean vs 표준: max {dv.max():.2e} median {np.median(dv):.2e}")

# H6f copy 기준선 (같은 표본일 때만)
try:
    m6 = json.load(open(I6F / "meta.json")); L6 = np.load(I6F / "l1.npy")
    if list(m6["video_id"]) == list(M["video_id"]):
        L = np.concatenate([L, L6[:, :, 1][:, None, :].astype(L.dtype)], 1); V = V + ["copyz(H6f)"]
        d6 = np.abs(L6[:, :, 0].mean(1) - L[:, 0].mean(1)); check["h6f_p_vs_h9_clean_max"] = float(d6.max())
        print(f"H6f copy 기준선 붙임 (h6f p vs h9 clean max {d6.max():.2e})")
    else:
        print("H6f video 순서 불일치 — copy 기준선 생략")
except FileNotFoundError:
    print("H6f 없음 — copy 기준선 생략")

pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
P = np.array([(d["pos"], d["imp"]) for d in pairs.values() if len(d) == 2]); ps, im = P[:, 0], P[:, 1]
blk = M["block_id"][ps]; _, bidx = np.unique(blk, return_inverse=True)
WIN = {"all": slice(0, 8), "t0-3": slice(0, 4), "t4-7": slice(4, 8)} | {f"s{j}": slice(j, j + 1) for j in range(8)}
OK = []
for w, sl in WIN.items():
    S = L[:, :, sl].mean(2)
    OK.append((S[im] > S[ps]).astype(np.float64) + 0.5 * (S[im] == S[ps]))
OK = np.stack(OK, 2)                                                  # (pairs, vars, wins)
okr = (r[im] > r[ps]) + 0.5 * (r[im] == r[ps])                         # 표준 per_video surprise 로 같은 쌍 채점 (수치 재현 폭)
check["acc_standard_same_pairs"] = round(100 * float(okr.mean()), 2); check["acc_clean"] = round(100 * float(OK[:, 0, 0].mean()), 2)
check["flip_clean_vs_standard"] = int((okr != OK[:, 0, 0]).sum())
print(f"같은 쌍 표준 채점 {check['acc_standard_same_pairs']} vs clean {check['acc_clean']} (판정 뒤집힘 {check['flip_clean_vs_standard']} 쌍)")
ties = {v: int((OK[:, vi, 0] == 0.5).sum()) for vi, v in enumerate(V)}

rng = np.random.default_rng(SEED)
TIMS = ("visible", "early", "mid", "late"); MOTS = ("static", "moving_flat", "moving"); VIOS = ("vanish", "shape", "color")
out = {"check": check, "ties_all_window": ties, "n_boot": NB, "vars": V, "windows": list(WIN), "groups": {}}
for t, mo, vt, d in itertools.product(("*",) + TIMS, ("*",) + MOTS, ("*",) + VIOS, ("*", "A", "B")):
    m = np.ones(len(P), bool)
    if t != "*": m &= tim[ps] == t
    if mo != "*": m &= M["motion"][ps] == mo
    if vt != "*": m &= M["violation_type"][ps] == vt
    if d != "*": m &= M["pair_id"][ps] == d
    if not m.any(): continue
    ub, bi = np.unique(bidx[m], return_inverse=True); G = len(ub)
    s = np.zeros((G,) + OK.shape[1:]); np.add.at(s, bi, OK[m]); cnt = np.bincount(bi, minlength=G).astype(np.float64)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=NB).astype(np.float64)   # (NB, G)
    acc_b = np.einsum("bg,gvw->bvw", W, s) / (W @ cnt)[:, None, None]        # (NB, vars, wins)
    acc = s.sum(0) / cnt.sum()
    g = {"n_pairs": int(m.sum()), "n_blocks": int(G)}
    for wi, w in enumerate(WIN):
        e = {}
        for vi, v in enumerate(V):
            lo, hi = np.percentile(acc_b[:, vi, wi], [2.5, 97.5])
            e[v] = {"acc": round(100 * acc[vi, wi], 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
            if vi > 0:
                db = acc_b[:, vi, wi] - acc_b[:, 0, wi]; dlo, dhi = np.percentile(db, [2.5, 97.5])
                pb = 2 * min((db <= 0).mean(), (db >= 0).mean())
                e[v] |= {"d": round(100 * (acc[vi, wi] - acc[0, wi]), 1), "d_lo": round(100 * dlo, 1), "d_hi": round(100 * dhi, 1),
                         "p": round(float(min(pb, 1.0)), 4)}
        g[w] = e
    out["groups"][f"{t}|{mo}|{vt}|{d}"] = g

# ── 추가 대조: 변형끼리 차이 (single − copy, single − mmx) 와 copy 와의 판정 일치율 ──
# agree_v = mean_pair 1[ok_v == ok_copy]  (v 의 정답/오답 패턴이 "마지막 관측 복사" 와 얼마나 같은가)
CONTRAST = [("single", "copyz(H6f)"), ("single", "mmx_all"), ("mm_L0-6", "clean")]
KEYS = ["*|*|*|*", "*|static|*|*", "*|moving_flat|*|*", "*|moving|*|*", "visible|moving|*|*", "late|moving|*|*",
        "late|moving_flat|*|*", "late|static|*|*", "late|*|vanish|A", "late|*|vanish|B", "late|moving|vanish|*", "late|moving_flat|vanish|*",
        "visible|moving|shape|*", "late|moving|vanish|A", "late|moving_flat|vanish|A", "late|static|vanish|A"]
out["contrasts"] = {}
if "copyz(H6f)" in V:
    ci = V.index("copyz(H6f)")
    for key in KEYS:
        t, mo, vt, d = key.split("|"); m = np.ones(len(P), bool)
        if t != "*": m &= tim[ps] == t
        if mo != "*": m &= M["motion"][ps] == mo
        if vt != "*": m &= M["violation_type"][ps] == vt
        if d != "*": m &= M["pair_id"][ps] == d
        ub, bi = np.unique(bidx[m], return_inverse=True); G = len(ub)
        s = np.zeros((G,) + OK.shape[1:]); np.add.at(s, bi, OK[m]); cnt = np.bincount(bi, minlength=G).astype(np.float64)
        ag = (OK[m][:, :, 0] == OK[m][:, ci:ci + 1, 0]).astype(np.float64); sa = np.zeros((G, ag.shape[1])); np.add.at(sa, bi, ag)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=NB).astype(np.float64); den = (W @ cnt)
        acc_b = np.einsum("bg,gvw->bvw", W, s) / den[:, None, None]; ag_b = (W @ sa) / den[:, None]
        e = {}
        for a_, b_ in CONTRAST:
            for w in ("all", "t0-3", "t4-7"):
                wi = list(WIN).index(w); db = acc_b[:, V.index(a_), wi] - acc_b[:, V.index(b_), wi]
                d0 = 100 * (s[:, V.index(a_), wi].sum() - s[:, V.index(b_), wi].sum()) / cnt.sum()
                lo, hi = np.percentile(db, [2.5, 97.5])
                e[f"{a_}-{b_}@{w}"] = {"d": round(d0, 1), "d_lo": round(100 * lo, 1), "d_hi": round(100 * hi, 1)}
        for vn in ("clean", "mmx_all", "single", "mm_L0-6"):
            vi = V.index(vn); lo, hi = np.percentile(ag_b[:, vi], [2.5, 97.5])
            e[f"agree_copy:{vn}"] = {"v": round(100 * sa[:, vi].sum() / cnt.sum(), 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
        db = ag_b[:, V.index("single")] - ag_b[:, 0]; lo, hi = np.percentile(db, [2.5, 97.5])
        e["agree_copy:single-clean"] = {"d": round(100 * (sa[:, V.index("single")].sum() - sa[:, 0].sum()) / cnt.sum(), 1),
                                        "d_lo": round(100 * lo, 1), "d_hi": round(100 * hi, 1)}
        out["contrasts"][key] = e

# ── 다중 비교: single − clean (창 all) 을 표의 칸 전체 (timing×motion 12 + timing×위반×방향 36) 에 걸어 Holm 보정 ──
fam = [f"{t}|{mo}|*|*" for t in TIMS for mo in MOTS] + [f"{t}|*|{vt}|{d}" for t in TIMS for vt in VIOS for d in ("A", "B", "*")]
for vn in ("single", "mmx_all", "mm_L0-6"):
    ps_ = sorted(((max(out["groups"][k]["all"][vn]["p"], 1.0 / (NB + 1)), k) for k in fam))
    sig, mx = [], 0.0
    for r_, (pv, k) in enumerate(ps_):
        adj = min(1.0, pv * (len(ps_) - r_)); mx = max(mx, adj)
        if mx < 0.05: sig.append(k)
    out.setdefault("holm", {})[vn] = {"family": len(fam), "n_sig": len(sig), "sig": sig}
# ── 수준: 변형이 예측을 전반적으로 흐리나 — 가능 clip 의 L1 (슬롯 평균) 과 상대 margin (S_imp − S_pos)/S_pos ──
S_all = L.mean(2)
out["level"] = {}
for t in ("*",) + TIMS:
    for mo in ("*",) + MOTS:
        m = np.ones(len(P), bool)
        if t != "*": m &= tim[ps] == t
        if mo != "*": m &= M["motion"][ps] == mo
        e = {}
        for vi, v in enumerate(V):
            sp, si = S_all[ps[m], vi], S_all[im[m], vi]; rm = (si - sp) / sp
            e[v] = {"L1_pos": round(float(sp.mean()), 4), "rel_margin_mean_pct": round(100 * float(rm.mean()), 3),
                    "rel_margin_median_pct": round(100 * float(np.median(rm)), 3), "abs_rel_margin_median_pct": round(100 * float(np.median(np.abs(rm))), 3)}
        out["level"][f"{t}|{mo}"] = e
json.dump(out, open(OUT / "h9c_bootstrap_v11.json", "w"), indent=1, ensure_ascii=False)


def row(key, w="all"):
    g = out["groups"][key]; e = g[w]; c = e["clean"]
    s = f"{key:34s} {w:5s} n={g['n_pairs']:5d}/{g['n_blocks']:4d}b  clean {c['acc']:5.1f} [{c['lo']:5.1f},{c['hi']:5.1f}]"
    for v in V[1:]:
        x = e[v]; s += f" | {v} {x['acc']:5.1f} Δ{x['d']:+5.1f} [{x['d_lo']:+5.1f},{x['d_hi']:+5.1f}] p{x['p']:.3f}"
    return s


lines = []
for t in ("*",) + TIMS:
    for mo in ("*",) + MOTS:
        lines.append(row(f"{t}|{mo}|*|*"))
for t in ("*",) + TIMS:
    for vt in VIOS:
        for d in ("*", "A", "B"):
            lines.append(row(f"{t}|*|{vt}|{d}"))
for t in TIMS:
    for mo in MOTS:
        for vt in VIOS:
            lines.append(row(f"{t}|{mo}|{vt}|*"))
for key in ("*|*|*|*", "visible|moving|*|*", "visible|moving_flat|*|*", "late|moving|*|*", "late|moving_flat|*|*", "late|static|*|*",
            "late|*|vanish|A", "visible|*|*|*", "late|*|*|*"):
    for w in ["t0-3", "t4-7"] + [f"s{j}" for j in range(8)]:
        lines.append(row(key, w))
open(OUT / "h9c_bootstrap_v11.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
for k, e in out.get("contrasts", {}).items():
    print(f"[contrast] {k:28s} " + "  ".join(f"{c} {x.get('d', x.get('v'))}[{x.get('d_lo', x.get('lo'))},{x.get('d_hi', x.get('hi'))}]" for c, x in e.items()))
for vn, h in out["holm"].items():
    print(f"[holm] {vn}: {h['n_sig']}/{h['family']} 칸 유의 — {h['sig']}")
for k, e in out["level"].items():
    print(f"[level] {k:22s} " + "  ".join(f"{v} L1 {x['L1_pos']:.4f} m̄ {x['rel_margin_mean_pct']:+.2f}% |m|med {x['abs_rel_margin_median_pct']:.2f}%" for v, x in e.items()))
print(f"ties {ties}  ({time.time()-t0:.0f}s)")
