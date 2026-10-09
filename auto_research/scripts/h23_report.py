#!/usr/bin/env python3
"""H2 + H3 보고 — h23_analyze_v3.py 산출물 (h23_v3.json, rows_C*_P*.npz) → 표 · 회귀 · 그림.

회귀 (자 없는 위치 적중, 층 L11, 자유 운동 6 법칙, 가능 clip, 16 창 합침):
  logit P(hit_truth) = b0 + b_t·t + b_d·(dist/18 px) + b_C·(C/2) + 법칙 고정효과
  t = 미래 슬롯 (문맥 끝에서 튜블릿 수), dist = 마지막으로 본 자리에서 진실 자리까지 (칸),  C = 문맥 프레임
  → "지속을 정하는 것은 시간 (t) 인가 거리 (dist) 인가 문맥 길이 (C) 인가" (RollOutV2 거리 vs RollOutV3 튜블릿 모순).
  bootstrap: clip 단위 (같은 clip 의 여러 창·슬롯을 묶는다).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "auto_research/exp_results/h23"
FIG = ROOT / "auto_research/figures/h23"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
SC = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def logit_fit(X, y, iters=60):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H_ = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(len(w))
        w += np.linalg.solve(H_, X.T @ (y - p))
    return w


def load_rows():
    R = []
    for f in sorted(H.glob("rows_C*_P*.npz")):
        C, P = (int(v) for v in re.findall(r"\d+", f.stem))
        z = dict(np.load(f)); z["C"] = np.full(len(z["t"]), C); z["P"] = np.full(len(z["t"]), P); R.append(z)
    return {k: np.concatenate([r[k] for r in R]) for k in R[0]}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    J = json.load(open(H / "h23_v3.json"))
    Q = load_rows()
    L = 11
    out = []
    pr = lambda s="": out.append(s)
    # ---- 1. 회귀: 지속을 정하는 변수
    m = np.isin(Q["scen"], FREE) & Q["plaus"]
    t, d, C = Q["t"][m].astype(float), Q["dist"][m] / 18.0, Q["C"][m] / 2.0
    law = np.stack([(Q["scen"][m] == s).astype(float) for s in FREE[1:]], 1)
    X = np.c_[t, d, C, law]
    ids = Q["b"][m]; uniq = np.unique(ids)
    rng = np.random.default_rng(0)
    pos = {u: np.where(ids == u)[0] for u in uniq}
    pr("## 1. 자 없는 위치 적중 (L11) 회귀 — 시간 · 거리 · 문맥 길이")
    pr("")
    pr("| 표적 | n | b_t (슬롯당) | b_d (칸당) | b_C (문맥 튜블릿당) |")
    pr("|---|---:|---|---|---|")
    for key, lab in (("ht", "진실 칸 적중"), ("hl", "마지막 본 칸 적중"), ("hc", "등속 외삽 칸 적중")):
        y = Q[key][m][:, L].astype(float)
        w = logit_fit(X, y)
        bs = []
        for _ in range(200):
            pick = np.concatenate([pos[u] for u in rng.choice(uniq, len(uniq))])
            bs.append(logit_fit(X[pick], y[pick], 30))
        bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        f = lambda i: f"{w[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]"
        pr(f"| {lab} | {len(y)} | {f(1)} | {f(2)} | {f(3)} |")
    pr("")
    # ---- 2. 거리 × 슬롯 표 (진실 적중, L11, 자유 운동, 16 창 합침)
    pr("## 2. 진실 칸 적중 (L11) — 거리 (칸) × 슬롯, 자유 운동 6 법칙, 16 창 합침")
    pr("")
    db = np.minimum((Q["dist"] / 18.0).astype(int), 7)
    ts = np.minimum(Q["t"], 15)
    hdr = "| 거리 칸 \\ 슬롯 | " + " | ".join(str(i) for i in range(0, 16, 2)) + " |"
    pr(hdr); pr("|---" * 9 + "|")
    for dd in range(8):
        cells = []
        for tt in range(0, 16, 2):
            mm = m & (db == dd) & ((ts == tt) | (ts == tt + 1))
            cells.append(f"{100*Q['ht'][mm][:, L].mean():.0f} ({mm.sum()})" if mm.sum() >= 30 else "–")
        pr(f"| {dd}{'+' if dd == 7 else ''} | " + " | ".join(cells) + " |")
    pr("")
    # ---- 3. 층 프로파일 (C16/P32, 자유 운동)
    for win in ("C16_P32", "C32_P32", "C8_P16"):
        if win not in J:
            continue
        W = J[win]
        S = int(win.split("P")[1]) // 2
        pr(f"## 3. 층 프로파일 — {win}, 자유 운동 (옮겨 간 슬롯만: W_t 와 W_T 가 안 겹침)")
        pr("")
        pr("| 슬롯 | n | 지표 | " + " | ".join(f"L{l}" for l in range(12)) + " |")
        pr("|---|---:|---|" + "---:|" * 12)
        for tt in range(S):
            v = W.get(f"moved|FREE|t{tt}")
            if not v or v["n"] < 40:
                continue
            for k in ("A", "Ash", "S", "B", "ht"):
                pr(f"| {tt} | {v['n']} | {k} | " + " | ".join(f"{x:.2f}" for x in v[k]) + " |")
        pr("")
    # ---- 4. 시나리오 × 슬롯 (L11, C16/P32)
    W = J.get("C16_P32")
    if W:
        pr("## 4. 시나리오 × 슬롯 — C16/P32, L11 (모든 가능 clip; A·S·B 는 옮겨 간 슬롯만)")
        pr("")
        pr("| 시나리오 | 지표 | " + " | ".join(f"t{i}" for i in range(16)) + " |")
        pr("|---|---|" + "---:|" * 16)
        for sc in SC:
            for k, src in (("ht", "all"), ("hl", "all"), ("hc", "all"), ("A", "moved"), ("Ash", "moved"), ("S", "moved"), ("B", "moved")):
                row = []
                for tt in range(16):
                    v = W.get(f"{src}|{sc}|t{tt}")
                    row.append(f"{v[k][L]:.2f}" if v and v["n"] >= 20 else "–")
                pr(f"| {sc} | {k} | " + " | ".join(row) + " |")
        pr("")
        pr("## 5. 다양성 D_t = trCov(p_t)/trCov(h_t) 와 진행 V_t (C16/P32, 층 L0 · L6 · L11)")
        pr("")
        D = np.array(W["diversity"]["D"]); V = np.array(W["progress"]["V"])
        pr("| 층 | 지표 | " + " | ".join(f"t{i}" for i in range(16)) + " |")
        pr("|---|---|" + "---:|" * 16)
        for l in (0, 6, 11):
            pr(f"| L{l} | D | " + " | ".join(f"{x:.3f}" for x in D[l]) + " |")
            pr(f"| L{l} | V | " + " | ".join("–" if np.isnan(x) else f"{x:.2f}" for x in V[l]) + " |")
        pr("")
    # ---- 6. testA
    pr("## 6. 검정 A — ledge·wall 짝: p 가 이어 간 미래 (불가능) 에 더 가까운 쌍의 비율 (L11)")
    pr("")
    for win in ("C16_P32", "C32_P32", "C8_P32", "C4_P32"):
        if win not in J:
            continue
        for sc in ("ledge", "wall"):
            ta = J[win]["testA"].get(sc)
            if not ta:
                continue
            fr = np.array(ta["frac_closer_imp"])[:, L]
            pr(f"- {win} {sc} (n={ta['n_pair']}, 짝 p 최대차 {ta['p_pair_maxdiff']:.2e}): " + " ".join(f"{100*x:.0f}" for x in fr))
    pr("")
    (H / "REPORT_h23.md").write_text("\n".join(out))
    print("\n".join(out))
    # ---- 그림: 진실 적중 vs 슬롯, 속도 3 분위 (C16/P32, L11) + 층별 A
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    mm = np.isin(Q["scen"], FREE) & Q["plaus"] & (Q["C"] == 16) & (Q["P"] == 32)
    sp = Q["dist"][mm] / (Q["t"][mm] + 1)
    q1, q2 = np.percentile(sp, [33, 67])
    for (lo_, hi_), col, lab in (((-1, q1), C1, "slow"), ((q1, q2), C2, "mid"), ((q2, 1e9), C3, "fast")):
        k = (sp > lo_) & (sp <= hi_)
        tt = Q["t"][mm][k]; y = Q["ht"][mm][k][:, L]
        ax[0].plot(range(16), [y[tt == i].mean() if (tt == i).sum() > 10 else np.nan for i in range(16)], color=col, label=lab)
    ax[0].set_xlabel("future slot t (tubelets after context end)"); ax[0].set_ylabel("hit (truth cell, L11)"); ax[0].legend(frameon=False)
    for l, col in ((3, C1), (7, C2), (11, C3)):
        tt = Q["t"][mm]; y = Q["ht"][mm][:, l]
        ax[1].plot(range(16), [y[tt == i].mean() if (tt == i).sum() > 10 else np.nan for i in range(16)], color=col, label=f"lens L{l}")
    ax[1].set_xlabel("future slot t"); ax[1].legend(frameon=False)
    for i, a_ in enumerate(ax):
        a_.text(0.5, -0.32, f"({'ab'[i]})", transform=a_.transAxes, ha="center")
    fig.tight_layout(); fig.savefig(FIG / "fig_hit_slot_speed_layer.png", dpi=160); fig.savefig(FIG / "fig_hit_slot_speed_layer.pdf")


if __name__ == "__main__":
    main()
