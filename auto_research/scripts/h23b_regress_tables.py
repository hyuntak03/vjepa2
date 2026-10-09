#!/usr/bin/env python3
"""H23 적대 검증 보정 (2/2) — 귀무 보정 회귀 · 적중−귀무 표 · 문맥 길이별 곡선 · 공통 문턱 층 차이 · A−Ash · 법칙별 S/B (CPU, NFS 입력만).

입력: exp_results/h23b/rows_null.npz (h23b_null_objectness_testA.py), exp_results/h23b/objectness.json,
      exp_results/h23/rows_C*_P*.npz · h23_v3.json (h23_analyze_v3.py).
왜: 옛 회귀 (logit hit ~ 슬롯 + 거리 + 문맥) 는 위치 사전확률 (귀무) 을 빼지 않았다. v3 는 한 시나리오 clip 들이 거의 같은 경로를
    지나므로 "그 시각 물체가 보통 있는 자리" 만 짚어도 적중이 오른다 — 슬롯 효과가 이 사전확률의 감소일 수 있다.
식 (L11, 자유 운동 6 법칙, 가능, 16 창 합침, 행 = clip × 창 × 슬롯):
  M0  logit P(hit) = b0 + b_t·t + b_d·d + b_C·(C/2) + 법칙 FE                  (옛 회귀 재현)
  Mn  logit E[null] = 같은 식 (분수 로지스틱)                                   (귀무 자체가 슬롯에 따라 얼마나 주나)
  M1  logit P(hit) = M0 + g·logit(null)                                          (귀무를 공변량으로)
  M1c M1 에서 C 를 범주 (C ∈ {4,8,16,32} 더미) 로
  M1x M1 + b_td·(t·d)                                                             (가법 가정 점검)
  M2  hit − null = OLS (t, d, C/2, 법칙 FE)                                      (선형 확률, 귀무 차감)
  d = 마지막 본 자리 → 진실 자리 거리 (칸, 18 px). CI = clip 단위 bootstrap (같은 clip 의 창·슬롯을 묶는다).
그 밖: 적중−귀무 (층 × 슬롯, C16/P32 · C32/P32), 거리 × 슬롯 표 (clip 수 · 법칙 수 표시), P32 문맥 길이별 적중 곡선,
      공통 문턱에서 L11 − L8 · L11 − L9 물체다움 차이 (clip bootstrap), A − Ash 층별 (clip bootstrap), 법칙별 S vs B, arc 의 적중·마지막칸 적중,
      D_t · V_t 와 복사 기준선 L1.

  auto_research/scripts/srun6.sh 8 64G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h23b_regress_tables.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "auto_research/exp_results/h23"
O = ROOT / "auto_research/exp_results/h23b"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
L = 11
RNG = np.random.default_rng(0)
out_md, out_js = [], {}
pr = out_md.append


def logit_fit(X, y, iters=40, w0=None):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1]) if w0 is None else w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        Hs = (X1 * (p * (1 - p))[:, None]).T @ X1 + 1e-4 * np.eye(len(w))
        step = np.linalg.solve(Hs, X1.T @ (y - p)); w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def ols_fit(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def boot_mean_ci(x, reps=1000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return [np.nan, np.nan]
    ix = RNG.integers(0, len(x), (reps, len(x)))
    m = x[ix].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    Z = dict(np.load(O / "rows_null.npz"))
    n = len(Z["t"])
    t = Z["t"].astype(float); d = Z["dist"] / 18.0; C = Z["C"].astype(int); P = Z["P"].astype(int)
    law = np.stack([(Z["scen"] == i).astype(float) for i in range(1, 6)], 1)
    hit = Z["hit"][:, L].astype(float); nul = Z["null"][:, L].astype(float)
    b = Z["b"].astype(int)
    uniq = np.unique(b); pos = {u: np.where(b == u)[0] for u in uniq}
    lnull = np.log(np.clip(nul, 2e-3, 1 - 2e-3) / (1 - np.clip(nul, 2e-3, 1 - 2e-3)))
    Cd = np.stack([(C == c).astype(float) for c in (8, 16, 32)], 1)
    pr(f"# H23b — 귀무 보정 회귀 · 표 (행 {n}, clip {len(uniq)}, 층 L{L})\n")
    pr(f"corr(t, d) = {np.corrcoef(t, d)[0, 1]:.3f}\n")
    # 옛 rows 와 같은 행인지 (적중 일치)
    old = []
    for f in sorted(H.glob("rows_C*_P*.npz")):
        Cw, Pw = (int(v) for v in re.findall(r"\d+", f.stem))
        q = np.load(f); m = np.isin(q["scen"], FREE) & q["plaus"]
        old.append((Cw, Pw, q["b"][m], q["t"][m], q["ht"][m][:, L]))
    agree = []
    for Cw, Pw, ob, ot, oh in old:
        mm = (C == Cw) & (P == Pw)
        key_new = dict(zip(zip(b[mm], Z["t"][mm].astype(int)), hit[mm]))
        agree.append(np.mean([key_new.get((int(x), int(y)), -1) == float(h) for x, y, h in zip(ob, ot, oh)]))
    pr(f"옛 rows (h23/rows_C*_P*.npz) 와 적중 일치율: min {min(agree):.4f} (16 창)\n")

    models = {
        "M0 raw": (np.c_[t, d, C / 2, law], hit, "logit", ["t", "d", "C/2"]),
        "Mn null": (np.c_[t, d, C / 2, law], nul, "logit", ["t", "d", "C/2"]),
        "M1 +logit(null)": (np.c_[t, d, C / 2, law, lnull], hit, "logit", ["t", "d", "C/2", "g_null"]),
        "M1c C categorical": (np.c_[t, d, Cd, law, lnull], hit, "logit", ["t", "d", "C8", "C16", "C32", "g_null"]),
        "M1x +t*d": (np.c_[t, d, C / 2, law, lnull, t * d], hit, "logit", ["t", "d", "C/2", "g_null", "t*d"]),
        "M2 OLS hit-null": (np.c_[t, d, C / 2, law], hit - nul, "ols", ["t", "d", "C/2"]),
    }
    idx_of = {"t": 1, "d": 2, "C/2": 3}
    pr("## 1. 회귀 (clip bootstrap 100 회, 95 % CI)\n")
    pr("| 모형 | 계수 | 추정 | CI |"); pr("|---|---|---:|---|")
    REG = {}
    for name, (X, y, kind, keys) in models.items():
        w = logit_fit(X, y) if kind == "logit" else ols_fit(X, y)
        cols = {}
        ncol = X.shape[1]
        for k in keys:
            if k in idx_of:
                cols[k] = idx_of[k]
            elif k in ("C8", "C16", "C32"):
                cols[k] = 3 + ("C8", "C16", "C32").index(k)
            elif k == "g_null":
                cols[k] = ncol - (1 if "t*d" not in keys else 2) + 1
            elif k == "t*d":
                cols[k] = ncol
        bs = []
        for _ in range(100):
            pick = np.concatenate([pos[u] for u in RNG.choice(uniq, len(uniq))])
            bs.append(logit_fit(X[pick], y[pick], iters=8, w0=w) if kind == "logit" else ols_fit(X[pick], y[pick]))
        bs = np.array(bs)
        REG[name] = {}
        for k in keys:
            j = cols[k]
            lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
            REG[name][k] = [round(float(w[j]), 4), round(float(lo), 4), round(float(hi), 4)]
            pr(f"| {name} | {k} | {w[j]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        print(name, REG[name], flush=True)
    out_js["regression"] = REG
    pr("")

    # ---- 2. 적중 − 귀무, 층 × 슬롯 (P32 두 창)
    HN = {}
    for Cw in (16, 32):
        mw = (C == Cw) & (P == 32)
        pr(f"## 2. 적중 / 귀무 / 적중−귀무 [95 % CI] — C{Cw}/P32, 자유 운동 (행 = clip)\n")
        pr("| 슬롯 | n clip | " + " | ".join(f"L{l}" for l in range(12)) + " |")
        pr("|---|---:|" + "---|" * 12)
        tab = {}
        for tt in range(16):
            m = mw & (Z["t"] == tt)
            if m.sum() < 30:
                continue
            cells = []; tab[tt] = {"n": int(m.sum())}
            for l in range(12):
                hh = Z["hit"][m, l].astype(float); nn = Z["null"][m, l].astype(float)
                ci = boot_mean_ci(hh - nn, 500)
                tab[tt][l] = [round(hh.mean(), 3), round(nn.mean(), 3), round((hh - nn).mean(), 3), round(ci[0], 3), round(ci[1], 3)]
                star = "**" if ci[0] > 0 else ""
                cells.append(f"{hh.mean():.2f}/{nn.mean():.2f} {star}{(hh-nn).mean():+.2f}{star}")
            pr(f"| {tt} | {int(m.sum())} | " + " | ".join(cells) + " |")
        HN[f"C{Cw}_P32"] = tab
        pr("\n굵게 = 적중−귀무 CI 하한 > 0.\n")
    out_js["hit_minus_null"] = HN

    # ---- 3. 거리 × 슬롯 (L11, 16 창 합침): 적중 · 적중−귀무 · 행 · clip · 법칙 수
    pr("## 3. 거리 × 슬롯 (L11, 16 창 합침) — 적중 % / 적중−귀무 %p (행, clip, 법칙 수)\n")
    db = np.minimum(d.astype(int), 7); ts = np.minimum(Z["t"], 15)
    pr("| 거리 칸 \\ 슬롯 | " + " | ".join(f"{i}–{i+1}" for i in range(0, 16, 2)) + " |")
    pr("|---" * 9 + "|")
    DT = {}
    for dd in range(8):
        cells = []
        for tt in range(0, 16, 2):
            m = (db == dd) & ((ts == tt) | (ts == tt + 1))
            if m.sum() < 30:
                cells.append("–"); continue
            ncl = len(np.unique(b[m])); nlaw = len(np.unique(Z["scen"][m]))
            hh = hit[m].mean(); hn = (hit[m] - nul[m]).mean()
            flag = " ⚠" if (ncl < 50 or nlaw == 1) else ""
            cells.append(f"{100*hh:.0f} / {100*hn:+.0f} ({m.sum()}, {ncl}, {nlaw}){flag}")
            DT[f"{dd}|{tt}"] = dict(hit=round(float(hh), 4), hit_minus_null=round(float(hn), 4), rows=int(m.sum()), clips=ncl, laws=nlaw)
        pr(f"| {dd}{'+' if dd == 7 else ''} | " + " | ".join(cells) + " |")
    pr("\n⚠ = clip < 50 또는 법칙 하나뿐.\n")
    out_js["dist_slot"] = DT

    # ---- 4. 문맥 길이별 곡선 (P32, L11)
    pr("## 4. 문맥 길이별 적중 · 적중−귀무 (P32, L11, 자유 운동)\n")
    pr("| 슬롯 | " + " | ".join(f"C{c} 적중 / −귀무" for c in (4, 8, 16, 32)) + " |")
    pr("|---|" + "---|" * 4)
    CC = {}
    for tt in range(16):
        cells = []
        for c in (4, 8, 16, 32):
            m = (C == c) & (P == 32) & (Z["t"] == tt)
            if m.sum() < 30:
                cells.append("–"); continue
            cells.append(f"{hit[m].mean():.2f} / {(hit[m]-nul[m]).mean():+.2f}")
            CC[f"C{c}|{tt}"] = [round(float(hit[m].mean()), 4), round(float((hit[m] - nul[m]).mean()), 4), int(m.sum())]
        pr(f"| {tt} | " + " | ".join(cells) + " |")
    out_js["per_C"] = CC
    pr("")

    # ---- 5. 공통 문턱 물체다움: L11 − L8, L11 − L9, L11 − L6 (clip bootstrap)
    OBJ = json.load(open(O / "objectness.json"))
    pr("## 5. 공통 문턱 물체다움 층 차이 (|진실−마지막| ≥ 3 칸, 행 = clip, rows_null 로 재계산)\n")
    pr("| 창 | 문턱 | 슬롯 | n | L6 | L8 | L9 | L11 | L11−L8 [CI] | L11−L6 [CI] |")
    pr("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    far = np.sqrt(((Z["tc"].astype(float) - Z["lc"].astype(float)) ** 2).sum(1)) >= 3
    CT = {}
    for Cw in (16, 32, 8):
        th = OBJ[f"C{Cw}_P32"]["theta"]
        for thname, thv in (("θ_L6", th[6]), ("θ_L11", th[11])):
            for tt in (8, 9, 10, 11, 12, 13):
                m = (C == Cw) & (P == 32) & (Z["t"] == tt) & far
                con = Z["contrast"][m].astype(float)
                ob = con >= thv
                d8 = ob[:, 11].astype(float) - ob[:, 8]; d6 = ob[:, 11].astype(float) - ob[:, 6]
                c8 = boot_mean_ci(d8, 500); c6 = boot_mean_ci(d6, 500)
                pr(f"| C{Cw} | {thname} | {tt} | {m.sum()} | {ob[:,6].mean():.2f} | {ob[:,8].mean():.2f} | {ob[:,9].mean():.2f} | {ob[:,11].mean():.2f} | "
                   f"{d8.mean():+.2f} [{c8[0]:+.2f}, {c8[1]:+.2f}] | {d6.mean():+.2f} [{c6[0]:+.2f}, {c6[1]:+.2f}] |")
                CT[f"C{Cw}|{thname}|{tt}"] = dict(n=int(m.sum()), obj=[round(float(x), 4) for x in ob.mean(0)],
                                                  L11_minus_L8=[round(float(d8.mean()), 4)] + [round(x, 4) for x in c8],
                                                  L11_minus_L6=[round(float(d6.mean()), 4)] + [round(x, 4) for x in c6])
    out_js["common_threshold_diff"] = CT
    pr("")

    # ---- 6. A − Ash 층별 (C16/P32, 옮겨 간 슬롯) + 렌즈 수렴
    q = np.load(H / "rows_C16_P32.npz")
    mv = np.isin(q["scen"], FREE) & q["plaus"] & (q["cheb"] >= 3)
    pr("## 6. A − Ash (clip 특이 몫) 층별 — C16/P32, 자유 운동, 옮겨 간 슬롯 (행 = clip) · 렌즈 수렴 cos\n")
    pr("| 슬롯 | n | " + " | ".join(f"L{l}" for l in range(12)) + " |")
    pr("|---|---:|" + "---:|" * 12)
    AA = {}
    for tt in (3, 5, 7, 8, 10, 12, 13):
        m = mv & (q["t"] == tt)
        if m.sum() < 30:
            continue
        dA = (q["A"][m] - q["Ash"][m]).astype(float)
        cis = [boot_mean_ci(dA[:, l], 500) for l in range(12)]
        AA[tt] = dict(n=int(m.sum()), mean=dA.mean(0).round(4).tolist(), ci=[[round(a, 4), round(b_, 4)] for a, b_ in cis],
                      conv=q["conv"][m].astype(float).mean(0).round(3).tolist(), A=q["A"][m].mean(0).round(3).tolist())
        pr(f"| {tt} | {int(m.sum())} | " + " | ".join(f"{dA[:, l].mean():+.3f}" for l in range(12)) + " |")
        pr(f"| {tt} CI 하한 | | " + " | ".join(f"{c[0]:+.3f}" for c in cis) + " |")
        pr(f"| {tt} conv | | " + " | ".join(f"{x:.2f}" for x in q["conv"][m].mean(0)) + " |")
    out_js["A_minus_Ash"] = AA
    pr("")

    # ---- 7. 법칙별 S vs B (L11, C16/P32, 옮겨 간 슬롯) · arc 적중
    J = json.load(open(H / "h23_v3.json"))["C16_P32"]
    pr("## 7. 법칙별 S / B (L11, C16/P32, 옮겨 간 슬롯) — S ≥ B 인 슬롯\n")
    SB = {}
    for s in FREE:
        rr = []
        for tt in range(16):
            v = J.get(f"moved|{s}|t{tt}")
            if v and v["n"] >= 20:
                rr.append((tt, v["n"], round(v["S"][L], 3), round(v["B"][L], 3)))
        SB[s] = rr
        pr(f"- {s}: " + ", ".join(f"t{a} {c:.2f}/{e:.2f}{'*' if c >= e else ''} (n{nn})" for a, nn, c, e in rr))
    out_js["S_B"] = SB
    pr("\n* = S ≥ B.\n")
    pr("arc (C16/P32, L11, 모든 가능 clip): 슬롯별 진실 적중 / 마지막 칸 적중 / 귀무")
    mA = (C == 16) & (P == 32) & (Z["scen"] == 5)
    arc = []
    for tt in range(16):
        v = J.get(f"all|arc|t{tt}"); m = mA & (Z["t"] == tt)
        if v and m.sum():
            arc.append((tt, v["n"], round(v["ht"][L], 2), round(v["hl"][L], 2), round(float(nul[m].mean()), 2)))
    pr("  " + ", ".join(f"t{a} {h:.2f}/{l_:.2f}/{nn_:.2f}" for a, _, h, l_, nn_ in arc))
    out_js["arc"] = arc
    pr("")

    # ---- 8. D_t · V_t · 복사 기준선
    D = np.array(J["diversity"]["D"]); V = np.array(J["progress"]["V"]); sh = np.array(J["progress"]["step_h"])
    m0 = (C == 16) & (P == 32)
    q_l1 = q["l1"][np.isin(q["scen"], FREE) & q["plaus"]]; q_t = q["t"][np.isin(q["scen"], FREE) & q["plaus"]]
    pr("## 8. D_t · V_t (C16/P32, L11, 전 토큰)\n")
    pr("D_t L11: " + " ".join(f"t{i} {x:.3f}" for i, x in enumerate(D[11])))
    pr("V_t L11: " + " ".join(f"t{i} {x:.2f}" for i, x in enumerate(V[11]) if not np.isnan(x)))
    pr("step_h (h 튜블릿 간 mean|Δ|): " + " ".join(f"{x:.3f}" for x in sh if not np.isnan(x)))
    pr("복사 h_T 의 L1 |h_T − h_t| (자유 운동 가능): " + " ".join(f"t{i} {q_l1[q_t == i, 12].mean():.3f}" for i in (0, 1, 3, 6, 10, 13)))
    pr("p (L11) 의 L1 |p − h_t|: " + " ".join(f"t{i} {q_l1[q_t == i, 11].mean():.3f}" for i in (0, 1, 3, 6, 10, 13)))
    trp = np.array(J["diversity"]["trp"])[11]; trh = np.array(J["diversity"]["trh_future"])
    pr("trCov p L11: " + " ".join(f"{x/1e3:.1f}k" for x in trp) + " / trCov h: " + " ".join(f"{x/1e3:.0f}k" for x in trh))
    out_js["D_V"] = dict(D_L11=D[11].round(4).tolist(), V_L11=[None if np.isnan(x) else round(float(x), 3) for x in V[11]],
                         step_h=[None if np.isnan(x) else round(float(x), 3) for x in sh],
                         copy_hT_l1={i: round(float(q_l1[q_t == i, 12].mean()), 4) for i in range(16)})
    (O / "REPORT_h23b.md").write_text("\n".join(out_md))
    json.dump(out_js, open(O / "h23b.json", "w"), indent=1, default=float)
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
