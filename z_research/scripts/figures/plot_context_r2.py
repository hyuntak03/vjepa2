#!/usr/bin/env python3
"""RollOut v3 — **문맥이 빨랐던 clip 일수록 p 가 물체를 더 멀리 보내나**, 16 창 × 미래 프레임마다.

물리 법칙을 맞히는지가 아니라 (물리 장면은 측정 도구다, CLAUDE.md), **p 의 미래가 문맥 내용에 따라 달라지는가**
를 본다 (state evolution 의 '전이', 2026-09-19 정의).

한 법칙 · 한 튜블릿 t 에서:

    이어가기 이동  f_i(t) = V_i·(t+1) + ½·A_i·(t+1)²      문맥 끝 속도 V · 가속도 A (진실 궤적) 를 그대로 이어 갔을 때의 이동
    p 의 이동      y_i(t) = Ŷ_i(t) − X_i(−1)               p 가 문맥 끝 자리에서 물체를 얼마나 옮겨 놓았나 (px, 2 축)

    같은 속도 수준 (|V| 0.5 px 단위, 수준마다 28~56 clip) 끼리 먼저 평균 → 수준 7~11 개의 (f̄, ȳ)
    ȳ ≈ β · f̄ + b                    (β 하나, 절편은 축마다)
    R²(t) = 1 − Σ‖ȳ − ŷ‖² / Σ‖ȳ − mean‖²       속도 순서를 얼마나 지키나      → fig_context_r2.png
    β(t)                                        이어가기의 몇 배만큼 보내나 (1 = 그대로) → fig_context_slope.png

왜 평균부터 내나: 한 법칙 안 clip 들은 속도가 비슷해서 (예: flat_v 6~10.5 px/2f) clip 하나씩 회귀하면
t1 에서 clip 간 이동 차이 (SD 3 px) 가 자 위치 오차 (z 6 px, p 12 px) 보다 작다 — 미래를 직접 보는 z 조차
R² 0.05 가 나왔다 (2026-09-24, `_superseded/` 그림). 같은 속도끼리 평균하면 clip 마다 다른 오차가 상쇄된다.

  R² 실선  = R²(물체 있다고 한 clip 으로 낸 평균) × 물체 있다고 한 비율      ← 놓친 clip 은 0 (사용자 지시)
  수준에 '있다' clip 이 5 개 미만이면 그 수준은 빠지고, 수준이 4 개 미만이면 R² = 0 · β 없음.
  회색 칸   = z 의 R² (법칙 평균) 가 0.8 아래 — 이동 차이가 자가 가를 수 있는 크기보다 작아 판정 불가.
              z 는 미래를 직접 보므로 **상한이 아니라 자 검사** 로만 쓴다.

⚠️ 법칙 6 개 (flat_v · flat_a · flat_d · ramp_a · ramp_d · arc) 만. **ledge · wall 은 이 지표로 못 잰다** —
   떨어짐·멈춤은 모든 clip 에 똑같이 일어나 절편으로 들어간다 (진실 궤적도 ledge R² 0.97~1.00).
⚠️ C4 는 문맥이 두 튜블릿이라 A 를 못 본다 → f = V·(t+1) 만.
⚠️ 가장자리 칸 (총 창 48f 이상이면 마지막 튜블릿, C=4 면 첫 미래 튜블릿) 은 뺀다 — rollout3_behavior_metrics.py 와 같은 규칙.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_context_r2.py          # → figures/behavior/fig_context_{r2,slope}.png + CONTEXT_R2.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, fig, check   # noqa: E402

RESN, SPLIT, MIN_CLIP, MIN_LEV, GATE = 144.0, 32, 5, 4, 0.8
CS = PS = (4, 8, 16, 32)
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc"]
COL = {"p": "#eb6834", "z": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def fit(y, f):
    """y, f: (m, 2) 수준 평균. 열 = [β, b_x, b_y]. R² 는 축별 평균을 뺀 분산 기준."""
    Z = np.zeros((len(y), 2, 3)); Z[..., 0] = f; Z[:, 0, 1] = 1; Z[:, 1, 2] = 1
    Zr, yr = Z.reshape(-1, 3), y.reshape(-1)
    b, *_ = np.linalg.lstsq(Zr, yr, rcond=None)
    res = yr - Zr @ b; tot = (y - y.mean(0)).reshape(-1)
    return 1 - (res @ res) / max(tot @ tot, 1e-9), b[0]


def load():
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text())
    thr = {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    return z, thr, bias


def window(z, thr, bias, c, p):
    """→ {(law, rep): dict(t, r2 (놓침 = 0), r2_pres, beta, rec)}"""
    POS = z["role"] == "roll"; L = z["truth"][POS] * RESN; sc = z["scenario"][POS]; tp = p // 2
    Xc = L[:, SPLIT - c:SPLIT].reshape(-1, c // 2, 2, 2).mean(2)
    x0, V = Xc[:, -1], Xc[:, -1] - Xc[:, -2]
    A = Xc[:, -1] - 2 * Xc[:, -2] + Xc[:, -3] if c // 2 >= 3 else np.zeros_like(V)
    lev = np.round(np.abs(V[:, 0]) * 2) / 2
    vis = z["in_frame"][POS][:, SPLIT:SPLIT + p].reshape(-1, tp, 2).all(2)
    keep = np.ones(tp, bool)
    if c + p >= 48: keep[-1] = False
    if c == 4: keep[0] = False
    T = np.where(keep)[0]
    out = {}
    for r in ("p", "z"):
        v = z[f"C{c}_P{p}_{r}"][POS]
        Y = v[..., 0:2] * RESN - np.array(bias[r], np.float32); say = v[..., 2] > thr[r]
        for law in LAWS:
            k = np.where(sc == law)[0]
            r2, r2p, be, rec = [], [], [], []
            for t in T:
                kv = k[vis[k, t]]; m = kv[say[kv, t]]
                rec.append(len(m) / max(len(kv), 1))
                ys, fs = [], []
                for l in np.unique(lev[m]):
                    g = m[lev[m] == l]
                    if len(g) < MIN_CLIP: continue
                    ys.append((Y[g, t] - x0[g]).mean(0)); fs.append((V[g] * (t + 1) + 0.5 * A[g] * (t + 1) ** 2).mean(0))
                if len(ys) < MIN_LEV:
                    r2.append(0.0); r2p.append(np.nan); be.append(np.nan); continue
                R2, b = fit(np.array(ys), np.array(fs))
                r2.append(R2 * rec[-1]); r2p.append(R2); be.append(b)
            out[(law, r)] = dict(t=T, r2=np.array(r2), r2_pres=np.array(r2p), beta=np.array(be), rec=np.array(rec))
    return out


def lawmean(out, r, key):
    a = np.stack([out[(law, r)][key] for law in LAWS])
    ok = np.isfinite(a).sum(0) >= 3
    with np.errstate(all="ignore"):
        return np.where(ok, np.nanmean(a, 0), np.nan)


def grid(W, key, ylab, ylim, ref, fname, title):
    f_, ax = plt.subplots(4, 4, figsize=(13.0, 10.2), sharey=True)
    for i, c in enumerate(CS):
        for j, p in enumerate(PS):
            a = ax[i, j]; o = W[(c, p)]; T = o[(LAWS[0], "p")]["t"]; fr = 2 * (T + 1)
            zr2 = lawmean(o, "z", "r2")
            for tt, g in zip(fr, zr2):                                          # 자가 못 가르는 칸
                if not g >= GATE:
                    a.axvspan(tt - 1, tt + 1, color="#dddddd", lw=0, zorder=0)
            for law in LAWS:
                a.plot(fr, o[(law, "p")][key], color=COL["p"], lw=0.7, alpha=0.35)
            a.plot(fr, lawmean(o, "z", key), color=COL["z"], lw=2.0, marker="o", ms=2.5,
                   label="z (sees the future; readout check)" if (i, j) == (0, 0) else None)
            a.plot(fr, lawmean(o, "p", key), color=COL["p"], lw=2.2, marker="o", ms=2.5,
                   label="p, mean of 6 laws" if (i, j) == (0, 0) else None)
            if (i, j) == (0, 0):
                a.plot([], [], color=COL["p"], lw=0.7, alpha=0.5, label="p, each law")
                a.fill_between([], [], color="#dddddd", label=f"z R² < {GATE}: not resolvable")
            a.axhline(ref, color="#aaaaaa", lw=0.7, ls=":"); a.axhline(0, color="#aaaaaa", lw=0.5)
            a.set_ylim(*ylim); a.set_xlim(0, p + 1)
            a.set_xticks(range(0, p + 1, 4 if p >= 16 else 2))
            a.spines[["top", "right"]].set_visible(False)
            if i == 0: a.set_title(f"prediction {p}f", fontsize=9.5, fontweight="bold")
            if j == 0: a.set_ylabel(f"context {c}f\n{ylab}")
            if i == 3: a.set_xlabel("future frame (after context end)")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    f_.suptitle(title, fontsize=10)
    f_.tight_layout(rect=(0, 0, 1, 0.97))
    d = fig("behavior"); d.mkdir(parents=True, exist_ok=True)
    f_.savefig(d / fname, dpi=160); plt.close(f_)
    return d / fname


def table(W):
    """CONTEXT_R2.md — 창마다 법칙 평균. 자가 가를 수 있는 칸만."""
    L = ["# RollOut v3 — 문맥 속도 이어받기 (16 창)", "",
         "정의는 `z_research/scripts/figures/plot_context_r2.py` 머리말. 법칙 6 개 평균, 놓친 clip = 0 (R²), "
         f"z 의 R² < {GATE} 칸은 `·` (자가 못 가른다).", "",
         "| 창 | 미래 프레임 | p R² | z R² | p β | z β | p '있다' % |", "|---|---|---|---|---|---|---|"]
    for c in CS:
        for p in PS:
            o = W[(c, p)]; T = o[(LAWS[0], "p")]["t"]
            zr, pr = lawmean(o, "z", "r2"), lawmean(o, "p", "r2")
            zb, pb = lawmean(o, "z", "beta"), lawmean(o, "p", "beta")
            rec = np.mean([o[(law, "p")]["rec"] for law in LAWS], 0)
            for n, t in enumerate(T):
                g = zr[n] >= GATE
                s = lambda v: f"{v:.2f}" if (g and np.isfinite(v)) else "·"
                L.append(f"| C{c}/P{p} | {2*(t+1)} | {s(pr[n])} | {zr[n]:.2f} | {s(pb[n])} | {s(zb[n])} | {100*rec[n]:.0f} |")
    L += ["", "## 재현", "", "```bash", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2",
          "/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_context_r2.py", "```", ""]
    f = fig("behavior") / "CONTEXT_R2.md"; f.write_text("\n".join(L)); return f


def main():
    z, thr, bias = load()
    W = {(c, p): window(z, thr, bias, c, p) for c in CS for p in PS}
    f1 = grid(W, "r2", "R²  (speed order kept)", (-0.25, 1.05), 1.0, "fig_context_r2.png",
              "Do faster-context clips get moved farther?  R² over speed-level means, lost clips count as 0")
    f2 = grid(W, "beta", "slope  (1 = as far as context speed)", (-0.25, 1.6), 1.0, "fig_context_slope.png",
              "How far does p move the object, relative to continuing the context motion?  slope over speed-level means")
    f3 = table(W)
    print("\n".join(map(str, (f1, f2, f3))))


if __name__ == "__main__":
    main()
