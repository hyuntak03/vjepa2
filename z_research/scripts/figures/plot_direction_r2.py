#!/usr/bin/env python3
"""RollOut v3 — p 의 미래가 **문맥에서 본 속도 방향 · 가속 방향** 을 따르나, 16 창 × 미래 튜블릿마다 R².

방향만 본다 (속력·크기는 안 본다). 문맥 끝 위치 x0, 속도 V, 가속 A 는 진실 궤적의 문맥 튜블릿에서.

  속도 방향   y = unit(x̂(t) − x0)                    마지막으로 본 자리에서 옮긴 방향
              u = unit(V)                              문맥 끝 속도 방향
  가속 방향   y = unit(x̂(t) − (x0 + V·(t+1)))          본 속도 그대로 이어 갔을 때보다 더/덜 간 방향
              u = unit(A)                              문맥 끝 가속 방향 (flat_a · flat_d · ramp_a · ramp_d · arc 만)
  y ≈ β·u + b (β 하나, 절편 축마다),  R² = 1 − Σ‖y − ŷ‖² / Σ‖y − ȳ‖²

⚠️ **법칙을 모아서** 잰다 — v3 는 한 법칙 안 clip 의 방향이 전부 같아서 (ramp_d 만 좌→우, 나머지 우→좌) 법칙 안에서는 R² 가 정의되지 않는다.
⚠️ p 가 '있다' 고 한 clip 만 (없으면 방향이 없다). '있다' 비율은 회색 선으로 같이 그린다.
⚠️ `h` (target encoder, 창 전체를 본다) 에 같은 계산 = **자 검사이자 진짜 미래의 방향**. 자유 운동에서는 문맥 방향 ≈ 진짜 방향이고,
   `arc`·`ledge`·`wall` 에서는 진짜 방향이 문맥에서 벗어난다 → `h` 가 p 보다 낮으면 p 가 **문맥 방향을 연장**한 것.
⚠️ 가장자리 칸 규칙은 `rollout3_behavior_metrics.py` 와 같다. C4 는 문맥 튜블릿이 둘이라 가속 방향이 없다.

  $P z_research/scripts/figures/plot_direction_r2.py   → figures/direction/fig_direction_r2.png · DIRECTION_R2.md
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

RESN, SPLIT, MINN = 144.0, 32, 30
CS = PS = (4, 8, 16, 32)
ACC = ["flat_a", "flat_d", "ramp_a", "ramp_d", "arc"]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def fit(y, *regs):
    """y (m,2) · 회귀변수 (m,2) 여럿 → R². 계수는 두 축 공통, 절편은 축마다."""
    m = len(y); k = len(regs)
    Z = np.zeros((m, 2, k + 2))
    for j, r in enumerate(regs): Z[..., j] = r
    Z[:, 0, k] = 1; Z[:, 1, k + 1] = 1
    b, *_ = np.linalg.lstsq(Z.reshape(-1, k + 2), y.reshape(-1), rcond=None)
    res = y.reshape(-1) - Z.reshape(-1, k + 2) @ b; tot = (y - y.mean(0)).reshape(-1)
    return 1 - (res @ res) / max(tot @ tot, 1e-9), b[0]


def window(z, thr, bias, c, p):
    POS = z["role"] == "roll"; L = z["truth"][POS] * RESN; sc = z["scenario"][POS]; tp = p // 2
    Xc = L[:, SPLIT - c:SPLIT].reshape(-1, c // 2, 2, 2).mean(2)
    x0, V = Xc[:, -1], Xc[:, -1] - Xc[:, -2]
    A = Xc[:, -1] - 2 * Xc[:, -2] + Xc[:, -3] if c // 2 >= 3 else None
    vis = z["in_frame"][POS][:, SPLIT:SPLIT + p].reshape(-1, tp, 2).all(2)
    keep = np.ones(tp, bool)
    if c + p >= 48: keep[-1] = False
    if c == 4: keep[0] = False
    T = np.where(keep)[0]; acc = np.isin(sc, ACC)
    out = {"t": T}
    for r in ("p", "h"):
        v = z[f"C{c}_P{p}_{r}"][POS]
        Y = v[..., 0:2] * RESN - np.array(bias[r], np.float32); say = v[..., 2] > thr[r]
        pres, rv, bv, ra, ba = [], [], [], [], []
        for t in T:
            m = vis[:, t] & say[:, t]
            pres.append(say[vis[:, t], t].mean())
            if m.sum() >= MINN and len(np.unique(sc[m])) >= 2:
                R2, b = fit(unit(Y[m, t] - x0[m]), unit(V[m]))
            else:
                R2, b = np.nan, np.nan
            rv.append(R2); bv.append(b)
            ma = m & acc
            if A is not None and ma.sum() >= MINN and len(np.unique(sc[ma])) >= 2:
                R2, b = fit(unit(Y[ma, t] - (x0[ma] + V[ma] * (t + 1))), unit(A[ma]))
            else:
                R2, b = np.nan, np.nan
            ra.append(R2); ba.append(b)
        out[r] = dict(pres=np.array(pres), vel=np.array(rv), vel_b=np.array(bv), acc=np.array(ra), acc_b=np.array(ba))
    return out


def figure(W):
    f_, ax = plt.subplots(4, 4, figsize=(13.0, 10.2), sharey=True)
    for i, c in enumerate(CS):
        for j, p in enumerate(PS):
            a = ax[i, j]; o = W[(c, p)]; fr = 2 * (o["t"] + 1)
            a.plot(fr, o["p"]["pres"], color=GRAY, lw=1.2, label="p: fraction read as present" if (i, j) == (0, 0) else None)
            for key, col, nm in (("vel", BLUE, "velocity direction"), ("acc", ORANGE, "acceleration direction")):
                sg = lambda r: o[r][key] * np.sign(o[r][key + "_b"])          # 부호 붙은 R² (− = 반대 방향)
                a.plot(fr, sg("h"), color=col, lw=1.0, ls="--", label=f"h: {nm} (readout check)" if (i, j) == (0, 0) else None)
                a.plot(fr, sg("p"), color=col, lw=2.0, marker="o", ms=2.5, label=f"p: {nm} signed R²" if (i, j) == (0, 0) else None)
            a.axhline(0, color="#bbbbbb", lw=0.5); a.axhline(1, color="#dddddd", lw=0.5, ls=":")
            a.set_ylim(-1.05, 1.05); a.set_xlim(0, p + 1); a.set_xticks(range(0, p + 1, 4 if p >= 16 else 2))
            a.spines[["top", "right"]].set_visible(False)
            if i == 0: a.set_title(f"prediction {p}f", fontsize=9.5, fontweight="bold")
            if j == 0: a.set_ylabel(f"context {c}f\nsigned R²  /  fraction")
            if i == 3: a.set_xlabel("future frame (after context end)")
    ax[0, 0].legend(frameon=False, fontsize=6.8, loc="lower left")
    f_.suptitle("Does p move the object in the direction it saw?  R² of p's direction on the context direction (laws pooled)", fontsize=10)
    f_.tight_layout(rect=(0, 0, 1, 0.97))
    d = fig("direction"); d.mkdir(parents=True, exist_ok=True); fp = d / "fig_direction_r2.png"
    f_.savefig(fp, dpi=160); plt.close(f_); return fp


def table(W, fp):
    s = lambda v, b: "·" if not np.isfinite(v) else f"{v:.2f}" + ("⁻" if b < 0 else "")
    L = ["# RollOut v3 — 속도 방향 · 가속 방향 R² (16 창, 2026-09-24)", "",
         f"정의: `z_research/scripts/figures/plot_direction_r2.py` 머리말. 그림 `{fp.relative_to(ROOT)}`. "
         "법칙을 모아서 잰다. p 가 '있다' 고 한 clip 만. ⁻ = β < 0 (반대 방향). · = clip 30 개 미만 또는 C4 (가속 없음).", "",
         "| 창 | 미래 프레임 | p '있다' | p 속도 방향 | h 속도 방향 | p 가속 방향 | h 가속 방향 |", "|---|---|---|---|---|---|---|"]
    for c in CS:
        for p in PS:
            o = W[(c, p)]
            for n, t in enumerate(o["t"]):
                L.append(f"| C{c}/P{p} | {2*(t+1)} | {100*o['p']['pres'][n]:.0f} % | {s(o['p']['vel'][n], o['p']['vel_b'][n])} | "
                         f"{s(o['h']['vel'][n], o['h']['vel_b'][n])} | {s(o['p']['acc'][n], o['p']['acc_b'][n])} | {s(o['h']['acc'][n], o['h']['acc_b'][n])} |")
    L += ["", "## 재현", "", "```bash", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2",
          "/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_direction_r2.py", "```", ""]
    f = fig("direction") / "DIRECTION_R2.md"; f.write_text("\n".join(L)); return f


def main():
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text())
    thr = {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    W = {(c, p): window(z, thr, bias, c, p) for c in CS for p in PS}
    fp = figure(W); print(fp); print(table(W, fp))


if __name__ == "__main__":
    main()
