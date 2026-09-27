#!/usr/bin/env python3
"""RollOut v3 창 격자 — 시나리오별 위치-시간 / 속도 프로필.

  figures/windows/fig_xy_speed.png   행 4 개 x 시나리오 8 개:
      x(t) · y(t) 변위 (원점 = 그 clip 의 진실 미래 t0 위치, 좌상단, + 는 아래)
      speed (튜블릿당 이동 px)
      p 가 "물체 있다" 고 하는 비율

⚠️ **물체가 없어지면 좌표를 안 그린다** (2026-09-23 사용자 지시).
   자는 **attn** (attention pooling + head 2 개), 판정은 **presence 값 > 학습셋 문턱** (`summary.json`).
   v3 에는 음성이 0 개라 문턱을 재조정할 수 없지만, **라벨 없는 검증**은 된다 — v3 는 물체가 내내
   화면 안이라 `z` 는 항상 "있다" 여야 한다. 그 오작동률은 `exp_results/windows/WINDOW_SUMMARY.md`.
   ⚠️ `attn` 의 지도(top1)는 v3 에서 평평해 판정에 못 쓴다 — presence 값만 쓴다.
   **대조는 언제나 `z`** — predictor 에 입력을 주는 바로 그 encoder 다.

x 는 **문맥 진행 방향**으로 부호를 맞춘다 (좌/우 clip 이 평균에서 상쇄되지 않게).
`wall`·`ledge` 는 불가능 미래 (통과 / 부유) 의 진실도 점선으로 같이 그린다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_rollout3_windows.py                 # C=16, P=32
  $P z_research/scripts/figures/plot_rollout3_windows.py --ctx 8 --prd 16
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
RESN, CELL, SPLIT = 144.0, 18.0, 32
REF = "z"                                          # 대조는 언제나 z (2026-09-23 사용자 결정)
# 자는 **attn** 으로 확정 (2026-09-23).
# ⚠️ attn 의 지도(top1)는 v3 에서 평평해 **판정 기준으로 못 쓴다** — presence 값을 쓴다.
REPS = ("z", "h", "p")                             # 그리는 순서 (p 가 맨 위로 오게)
SCEN = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a", "truth": "#222222"}
MINFRAC = 0.30                                     # 남은 clip 이 이보다 적으면 선을 끊는다
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def thresholds(d):
    """문턱을 **학습 산출물에서 읽는다.**

    ⚠️ 하드코딩하면 자를 다시 학습했을 때 **조용히 옛 문턱으로** 그려진다 (2026-09-24 에 이걸 막았다).
       값은 `summary.json` 의 `thr_val_fpr5` — 학습셋 val 음성 5 % 오탐 지점이다.
    """
    S = json.loads((d / "summary.json").read_text())
    return {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}


THR = None   # thresholds(PRES) 로 채운다


def masked_mean(v, m, n_all):
    """(clip, t) 를 t 마다 평균. presence 로 걸러 남은 clip 이 MINFRAC 아래면 NaN."""
    out = np.full(v.shape[1], np.nan); sd = np.full(v.shape[1], np.nan)
    for t in range(v.shape[1]):
        k = m[:, t]
        if k.sum() >= max(20, MINFRAC * n_all):
            out[t], sd[t] = v[k, t].mean(), v[k, t].std()
    return out, sd


def main(c, p_len):
    global THR
    THR = thresholds(PRES)
    z = np.load(RES_DIR / "readings.npz", allow_pickle=True)
    check(z, PRES)
    L, inf, sc, role, blk = z["truth"], z["in_frame"], z["scenario"], z["role"], z["block_id"]
    tp = p_len // 2
    gt = L[:, SPLIT:SPLIT + p_len].reshape(len(L), tp, 2, 2).mean(2) * RESN   # (n, tp, 2) **px**
    vis = inf[:, SPLIT:SPLIT + p_len].reshape(len(L), tp, 2).all(2)
    POS = role == "roll"
    reps = [r for r in REPS if f"C{c}_P{p_len}_{r}" in z.files]
    if REF not in reps:
        raise SystemExit(f"대조 {REF} 가 readings.npz 에 없다 (있는 것: {reps})")
    rd = {r: z[f"C{c}_P{p_len}_{r}"] for r in reps}
    # ⚠️ **attn 의 좌표 편향을 뺀다** (학습셋 held-out 에서 잰 자 자신의 성질, v3 라벨 미사용).
    #    안 빼면 wall·ledge 에서 "p 가 사건을 적용한다" 는 **자의 평균 수축이 만든 결론**이 나온다
    #    (실측: attn p 의 Δx +11.5 px → wall 판정이 mass 와 정반대로 뒤집혔다, 2026-09-23).
    BIAS = json.loads((PRES / "attn_bias_px.json").read_text())
    rd = {r: v.copy() for r, v in rd.items()}
    for r in rd:
        rd[r][..., 0:2] -= np.array(BIAS[r], np.float32) / RESN
    imp_of = {}                                                               # block → 불가능 clip 행
    for i, (b, r_) in enumerate(zip(blk, role)):
        if r_ == "impossible_roll":
            imp_of[b] = i

    fig, axes = plt.subplots(4, len(SCEN), figsize=(2.0 * len(SCEN), 8.6), sharex=True)
    t = np.arange(tp)
    for col, s in enumerate(SCEN):
        k = np.where(POS & (sc == s))[0]
        # 진행 방향 = **문맥** 구간의 운동 방향 (wall 은 미래에서 멈추므로 미래로 정하면 부호가 죽는다)
        ctx = L[k, SPLIT - c:SPLIT, 0]
        sgn = np.sign(ctx[:, -1] - ctx[:, 0]); sgn[sgn == 0] = 1
        SG = np.stack([sgn, np.ones_like(sgn)], -1)[:, None, :]               # (nk,1,2)
        g = (gt[k] - gt[k][:, :1]) * SG
        say = {r: (rd[r][k][..., 2] > THR[r]) & vis[k] for r in reps}
        nk = len(k)

        for a in (0, 1):                                                       # x, y 변위
            ax = axes[a, col]
            mu, sd = masked_mean(g[..., a], vis[k], nk)
            ax.plot(t, mu, color=COL["truth"], lw=2, label="truth (possible)")
            ax.fill_between(t, mu - sd, mu + sd, color=COL["truth"], alpha=0.08)
            if s in ("ledge", "wall"):
                ki = np.array([imp_of[blk[i]] for i in k])
                gi = (L[ki, SPLIT:SPLIT + p_len].reshape(len(ki), tp, 2, 2).mean(2) * RESN - gt[k][:, :1]) * SG
                ax.plot(t, gi[..., a].mean(0), color=COL["truth"], lw=1.4, ls="--",
                        label="truth (impossible: pass / float)")
            for r in reps:
                d = (rd[r][k][..., 0:2] * RESN - gt[k][:, :1]) * SG
                mu, sd = masked_mean(d[..., a], say[r], nk)
                ax.plot(t, mu, color=COL[r], lw=1.7, label=r)
                ax.fill_between(t, mu - sd, mu + sd, color=COL[r], alpha=0.12)
            ax.axhline(0, color="k", lw=0.4); ax.spines[["top", "right"]].set_visible(False)
            lo, hi = ax.get_ylim(); ax.set_ylim(max(hi, 8.0), min(lo, -0.05 * max(hi, 8.0)))
            if col == 0:
                ax.set_ylabel("x from true start (px)\nalong travel, + = down" if a == 0
                              else "y from true start (px)\n+ = down")
            if a == 0:
                ax.set_title(s, fontsize=10, fontweight="bold")

        ax = axes[2, col]                                                      # 속도
        sp_t = np.linalg.norm(np.diff(gt[k], axis=1), axis=-1)
        mu, _ = masked_mean(sp_t, vis[k][:, 1:], nk)
        ax.plot(t[1:], mu, color=COL["truth"], lw=2)
        for r in reps:
            v = np.linalg.norm(np.diff(rd[r][k][..., 0:2] * RESN, axis=1), axis=-1)
            mk = say[r][:, 1:] & say[r][:, :-1]                                # 양 끝 튜블릿 모두 "있다"
            mu, _ = masked_mean(v, mk, nk)
            ax.plot(t[1:], mu, color=COL[r], lw=1.7)
        ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel("speed\n(px per tubelet)")

        ax = axes[3, col]                                                      # presence 값
        for r in reps:
            mu, _ = masked_mean(rd[r][k][..., 2], vis[k], nk)
            ax.plot(t, mu, color=COL[r], lw=1.7)
            ax.axhline(THR[r], color=COL[r], lw=0.7, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("future tubelet"); ax.set_xticks(range(0, tp, max(1, tp // 4)))
        if col == 0:
            ax.set_ylabel("presence logit\n(dotted = threshold)")

    h_, l_ = axes[0, -1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, frameon=False, fontsize=11,
               handlelength=3.0, columnspacing=2.5, bbox_to_anchor=(0.5, -0.002))
    fig.suptitle(f"RollOut v3 — context {c}f, prediction {p_len}f (event at f{SPLIT}).  "
                 f"readout = attn (attention pooling, 2 heads), applied frozen.  "
                 f"p = predictor, z = context encoder (predictor's input), h = target encoder", fontsize=11)
    fig.text(0.5, 0.062,
             "Origin = the clip's true position at future tubelet 0 (top-left); both axes grow downward, so a readout "
             "behind the truth sits above the black line.  x is signed along the CONTEXT travel direction.",
             ha="center", fontsize=8)
    fig.text(0.5, 0.040,
             f"A tubelet counts as 'object lost' when the presence logit falls below the threshold set on the readout's own val split "
             f"(dotted lines, bottom row); those are dropped, and a curve is cut when fewer than {MINFRAC:.0%} of clips remain.",
             ha="center", fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"fig_xy_speed_C{c}_P{p_len}.png"
    fig.tight_layout(rect=(0, 0.085, 1, 0.965)); fig.savefig(out, dpi=165); plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16, choices=(4, 8, 16, 32))
    ap.add_argument("--prd", type=int, default=32, choices=(4, 8, 16, 32))
    a = ap.parse_args()
    main(a.ctx, a.prd)
