#!/usr/bin/env python3
"""RollOut v3 — 시나리오 8 행 x **위치·속도 4 열** 프로필. 한 장으로 운동만 본다.

  행  = 법칙 8 개
  열1 x / v0x   문맥 속도로 몇 걸음 갔나 (등속이면 기울기 1 인 직선)
  열2 y  (px)   아래가 +
  열3 vx / v0x  **1.0 = 문맥 속도를 그대로 이어감**
  열4 vy (px/tubelet)

⚠️ **왜 x 만 나누고 y 는 안 나누나** (2026-09-24 실측, 속도칸 7 개 사이의 CV):
     x  px 그대로 0.18~0.42  →  v0x 로 나누면 **0.002~0.025**
     y  px 그대로 **arc·ledge 0.000** →  v0x 로 나누면 arc 0.83 · ledge 0.27 로 **깨진다**
   수평 이동은 clip 속도에 비례하고 **낙하는 수평 속도와 무관**하다. 그래서 단위를 열마다 다르게 쓴다.
   (ramp 의 y 만 px 에서 CV 0.18 이 남는다 — 경사면에서는 y 도 속도에 매여 있다. 띠로 보인다)

  `v0x` = 문맥 **마지막 한 걸음**의 수평 이동 (px/tubelet), 진실 라벨에서 잰다.
  ⚠️ `v0x < 2` 인 clip 은 뺀다 (28 개, 전부 `arc` 의 가파른 발사각 = 7 %). 나누기가 폭발한다.

⚠️ 물체가 없다고 판정된 튜블릿은 빼고 평균한다 (presence > 학습셋 문턱). 남은 clip 이 30 % 아래면 선을 끊고,
   `p` 가 절반 아래로 떨어지는 자리에 세로 점선을 긋는다 — **거기부터는 평균이 살아남은 clip 의 것**이다.
⚠️ 속도는 **중심차분** (x(t+1) − x(t−1))/2 다. 전진차분보다 잡음이 절반이다.
⚠️ 좌표는 학습셋에서 잰 **attn 치우침을 뺀 값** (`attn_bias_px.json`).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_rollout3_profile.py                  # C16 / P32
  $P z_research/scripts/figures/plot_rollout3_profile.py --ctx 8 --prd 16 --reps z p
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
import sys; sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))   # noqa: E702
from rollout3_paths import PRES, WIN as RES_DIR, fig, check   # noqa: E402  경로·자 지문은 한 곳에서
FIG = fig("windows")
RESN, SPLIT, VMIN = 144.0, 32, 2.0
SCEN = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
NICE = {"flat_v": "flat, constant v", "flat_a": "flat, accelerating", "flat_d": "flat, decelerating",
        "ramp_a": "ramp down (speeds up)", "ramp_d": "ramp up (slows down)", "arc": "projectile arc",
        "ledge": "ledge → falls", "wall": "wall → stops"}
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a", "truth": "#222222"}
MINFRAC = 0.30
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def thresholds(d):
    """문턱을 **학습 산출물에서 읽는다.**

    ⚠️ 하드코딩하면 자를 다시 학습했을 때 **조용히 옛 문턱으로** 그려진다 (2026-09-24 에 이걸 막았다).
       값은 `summary.json` 의 `thr_val_fpr5` — 학습셋 val 음성 5 % 오탐 지점이다.
    """
    S = json.loads((d / "summary.json").read_text())
    return {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}


THR = None   # thresholds(PRES) 로 채운다


def mmean(v, m, n_all):
    """t 마다 평균 + 표준편차. 남은 clip 이 MINFRAC 아래면 NaN."""
    mu = np.full(v.shape[1], np.nan); sd = np.full(v.shape[1], np.nan)
    for t in range(v.shape[1]):
        k = m[:, t]
        if k.sum() >= max(20, MINFRAC * n_all):
            mu[t], sd[t] = np.nanmean(v[k, t]), np.nanstd(v[k, t])
    return mu, sd


def central(a):
    """중심차분 속도 (…, T) → (…, T). 양 끝은 한쪽 차분."""
    v = np.full_like(a, np.nan)
    v[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / 2.0
    v[..., 0] = a[..., 1] - a[..., 0]
    v[..., -1] = a[..., -1] - a[..., -2]
    return v


def main(c, p_len, reps):
    global THR
    THR = thresholds(PRES)
    z = np.load(RES_DIR / "readings.npz", allow_pickle=True)
    check(z, PRES)
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    reps = [r for r in reps if f"C{c}_P{p_len}_{r}" in z.files]
    L, inf, sc, role, blk = z["truth"], z["in_frame"], z["scenario"], z["role"], z["block_id"]
    tp = p_len // 2
    gt = L[:, SPLIT:SPLIT + p_len].reshape(len(L), tp, 2, 2).mean(2) * RESN
    ctx = L[:, SPLIT - c:SPLIT].reshape(len(L), c // 2, 2, 2).mean(2) * RESN
    vis = inf[:, SPLIT:SPLIT + p_len].reshape(len(L), tp, 2).all(2)
    # 문맥 마지막 한 걸음의 **수평** 속도. 문맥이 1 튜블릿뿐이면 (C=4 는 2) 진실 첫 미래 걸음으로 대신한다
    v0 = np.abs(ctx[:, -1, 0] - ctx[:, -2, 0]) if ctx.shape[1] >= 2 else np.abs(gt[:, 0, 0] - ctx[:, -1, 0])
    sgn = np.sign(ctx[:, -1, 0] - ctx[:, 0, 0]); sgn[sgn == 0] = 1
    POS = (role == "roll") & (v0 >= VMIN)
    imp_of = {b: i for i, (b, r_) in enumerate(zip(blk, role)) if r_ == "impossible_roll"}

    rd = {r: z[f"C{c}_P{p_len}_{r}"].copy() for r in reps}
    for r in rd:
        rd[r][..., 0:2] -= np.array(bias[r], np.float32) / RESN

    fig, axes = plt.subplots(len(SCEN), 4, figsize=(11.2, 2.05 * len(SCEN)), sharex=True)
    t = np.arange(tp)
    n_drop = int(((role == "roll") & (v0 < VMIN)).sum())
    for row, s in enumerate(SCEN):
        k = np.where(POS & (sc == s))[0]
        nk, V = len(k), v0[k][:, None]
        SG = sgn[k][:, None]
        # 진실 — 변위는 t0 기준
        TX = (gt[k][..., 0] - gt[k][:, :1, 0]) * SG
        TY = gt[k][..., 1] - gt[k][:, :1, 1]
        series = [("truth", TX / V, TY, central(TX) / V, central(TY), vis[k])]
        if s in ("ledge", "wall"):
            ki = np.array([imp_of[blk[i]] for i in k])
            gi = L[ki, SPLIT:SPLIT + p_len].reshape(len(ki), tp, 2, 2).mean(2) * RESN
            IX = (gi[..., 0] - gt[k][:, :1, 0]) * SG; IY = gi[..., 1] - gt[k][:, :1, 1]
            series.append(("imp", IX / V, IY, central(IX) / V, central(IY), vis[k]))
        for r in reps:
            X = (rd[r][k][..., 0] * RESN - gt[k][:, :1, 0]) * SG
            Y = rd[r][k][..., 1] * RESN - gt[k][:, :1, 1]
            say = (rd[r][k][..., 2] > THR[r]) & vis[k]
            series.append((r, X / V, Y, central(X) / V, central(Y), say))

        for cidx, (ylab, unit) in enumerate([("x / v0x", "steps"), ("y (px)", "px"),
                                             ("vx / v0x", ""), ("vy (px/tubelet)", "")]):
            ax = axes[row, cidx]
            for nm, X, Y, VX, VY, m in series:
                d = [X, Y, VX, VY][cidx]
                mu, sd = mmean(d, m, nk)
                # ⚠️ 속도 열 (2, 3) 에는 띠를 안 그린다 — 위치 오차 0.5 칸이 차분으로 증폭돼
                #    SD 가 2.2~2.8 px 이라 띠가 패널을 덮어 버린다 (2026-09-24). 평활은 걸지 않는다:
                #    깜빡임 같은 실제 구조를 뭉갠다. 띠만 걷고 평균선을 보인다.
                band = cidx < 2
                if nm == "truth":
                    ax.plot(t, mu, color=COL["truth"], lw=2.0, zorder=3, label="truth")
                    if band:
                        ax.fill_between(t, mu - sd, mu + sd, color=COL["truth"], alpha=0.10, lw=0)
                elif nm == "imp":
                    ax.plot(t, mu, color=COL["truth"], lw=1.3, ls="--", zorder=3,
                            label="truth (impossible)")
                else:
                    ax.plot(t, mu, color=COL[nm], lw=1.6, label=nm)
                    if band:
                        ax.fill_between(t, mu - sd, mu + sd, color=COL[nm], alpha=0.13, lw=0)
            if cidx == 0:
                ax.plot(t, t, color="#999999", lw=0.7, ls=":", zorder=1)      # 등속 기준선
            if cidx == 2:
                ax.axhline(1.0, color="#999999", lw=0.7, ls=":", zorder=1)    # 문맥 속도 유지
            if cidx in (1, 3):
                ax.axhline(0, color="#999999", lw=0.7, ls=":", zorder=1)
            if "p" in reps:                                                   # p 가 절반 아래로 가는 자리
                fr = ((rd["p"][k][..., 2] > THR["p"]) & vis[k]).mean(0)
                lost = np.where(fr < 0.5)[0]
                if len(lost):
                    ax.axvline(lost[0], color=COL["p"], lw=0.8, ls=":", alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if cidx == 1:
                ax.invert_yaxis()                                             # + 가 아래
            if row == 0:
                ax.set_title(["x displacement / v0x", "y displacement (px)",
                              "horizontal speed / v0x", "vertical speed (px/tubelet)"][cidx],
                             fontsize=9.5, fontweight="bold")
            if row == len(SCEN) - 1:
                ax.set_xlabel("future tubelet")
        axes[row, 0].set_ylabel(f"{s}\n{NICE[s]}", fontsize=8.5)

    h_, l_ = axes[6, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.001))
    fig.suptitle(
        f"RollOut v3 — motion profile, context {c}f / prediction {p_len}f (event at f{SPLIT}).  "
        f"readout = attn, frozen.  z = context encoder, h = target encoder, p = predictor\n"
        f"x is normalised by v0x (the clip's own horizontal step at the end of the context) so the 7 speed bins "
        f"collapse; y is left in px because falling does not scale with horizontal speed", fontsize=9.5)
    fig.text(0.5, 0.016,
             f"Readout curves use only tubelets judged 'present'; a curve is cut below {MINFRAC:.0%} of clips, and the "
             f"dotted vertical line marks where p holds the object in fewer than half of them.  "
             f"{n_drop} arc clips with v0x < {VMIN:.0f} px/tubelet are excluded.  Bands are ±1 SD across clips, "
             f"drawn for position only: differencing turns a 0.5-cell position error into a 2-3 px speed error, so a "
             f"speed band would swamp the panel.",
             ha="center", fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"fig_profile_C{c}_P{p_len}.png"
    fig.tight_layout(rect=(0, 0.033, 1, 0.962)); fig.savefig(out, dpi=165); plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16, choices=(4, 8, 16, 32))
    ap.add_argument("--prd", type=int, default=32, choices=(4, 8, 16, 32))
    ap.add_argument("--reps", nargs="+", default=["z", "h", "p"], choices=["z", "h", "p"])
    a = ap.parse_args()
    main(a.ctx, a.prd, a.reps)
