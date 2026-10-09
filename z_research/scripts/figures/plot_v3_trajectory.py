#!/usr/bin/env python3
"""RollOut v3 — 시나리오마다 **위치–시간 · 속도–시간** (GT · p · z). 그림 하나 = 창 하나.

  행 = 시나리오 8 개 × (x, y)  → 16 줄        열 = 위치–시간 | 속도–시간
  위치  = 마지막으로 본 자리 (문맥 끝 튜블릿) 에서 옮긴 양 (px). y 는 **위 = +** 로 뒤집어 그린다
  속도  = 튜블릿 사이 이동 (px/튜블릿). p · z 의 첫 미래 칸은 (읽은 위치 − 문맥 끝 GT 위치)
  검정 = GT (문맥 구간 포함) · 주황 = p (p 자기 자, 미래만) · 초록 = z (context encoder, 창 전체를 본다 = 자 검사, 미래만)
  평균: GT 는 시나리오의 clip 전부, p · z 는 그 튜블릿에서 '있다' 고 한 clip 만 (속도는 앞뒤 두 칸 다 '있다')
  빨간 점 (패널 아래쪽) = clip 의 절반 넘게 p 가 '없다' 고 한 튜블릿 — 그 칸의 p 선은 끊는다
  좌표 치우침 (`attn_bias_px.json`) 은 뺐다. 가능 clip 만.

⚠️ 읽은 위치는 자의 읽기다 — 물체가 흐려지면 기본값 (학습 평균 위치) 쪽으로 끌린다 (CLAUDE.md §7-3). 후반 칸의 속도 0 을 "멈춤" 으로 읽지 말 것 (§6).
⚠️ z · h 는 추출 때 미래 튜블릿만 저장했다 — 문맥 구간은 GT 만 있다.

  $P z_research/scripts/figures/plot_v3_trajectory.py                  # 16 창 전부
  $P z_research/scripts/figures/plot_v3_trajectory.py --windows 16x32
  → z_research/RollOutV3/figures/v3_trajectory/fig_C{c}_P{p}.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, ARCHIVE, check, present_thr, is_identity, stamp       # noqa: E402
DEC = "decoder: position + identity (56 shape x colour + none)" if is_identity(PRES) else "decoder: position + presence"

OUT = ARCHIVE / "v3_trajectory"                  # R3_FIGROOT 로 바꾼다 (기본 RollOutV3/figures)
RESN, SPLIT = 144.0, 32
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
BLACK, ORANGE, GREEN = "#222222", "#eb6834", "#1baf7a"
FLIP = np.array([1.0, -1.0], np.float32)             # y 는 위 = +
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def masked_mean(a, m):
    """a (n, T), m (n, T) → (T,) m 인 것만 평균, 없으면 nan."""
    s = np.where(m, a, 0).sum(0); n = m.sum(0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def draw(z, thr, bias, c, p):
    POS = z["role"] == "roll"; L = z["truth"][POS] * RESN; sc = z["scenario"][POS]
    tc, tp = c // 2, p // 2
    Lw = L[:, SPLIT - c:SPLIT + p].reshape(-1, tc + tp, 2, 2).mean(2)            # 창 전체 GT (튜블릿)
    x0 = Lw[:, tc - 1]                                                           # 마지막으로 본 자리
    TG = np.arange(tc + tp) - (tc - 1)                                           # 시간: 문맥 끝 = 0
    TF = np.arange(tp) + 1                                                       # 미래 튜블릿 = 1..tp
    G = (Lw - x0[:, None]) * FLIP                                                # GT 위치 (n, tc+tp, 2)
    GV = np.diff(Lw, axis=1) * FLIP                                              # GT 속도 (n, tc+tp-1, 2)
    R = {}
    for r in ("p", "z"):
        v = z[f"C{c}_P{p}_{r}"][POS]
        xy = v[..., 0:2] * RESN - np.array(bias[r], np.float32)
        say = v[..., 2] > thr[r]
        pos = (xy - x0[:, None]) * FLIP
        vel = np.diff(np.concatenate([x0[:, None], xy], 1), axis=1) * FLIP        # 첫 칸 = 읽은 위치 − 문맥 끝 GT
        sv = say & np.concatenate([np.ones_like(say[:, :1]), say[:, :-1]], 1)     # 속도: 앞뒤 두 칸 다 '있다'
        R[r] = (pos, vel, say, sv)

    f_, ax = plt.subplots(16, 2, figsize=(11, 27), sharex=True)
    for j, law in enumerate(LAWS):
        k = sc == law
        absent = R["p"][2][k].mean(0) <= 0.5
        for a_ in range(2):
            row = 2 * j + a_
            for col in range(2):
                a = ax[row, col]
                if col == 0:
                    a.plot(TG, G[k][..., a_].mean(0), color=BLACK, lw=1.6)
                    for r, cl, lw in (("z", GREEN, 1.1), ("p", ORANGE, 1.8)):
                        pos, vel, say, sv = R[r]
                        m = masked_mean(pos[k][..., a_], say[k])
                        if r == "p": m[absent] = np.nan
                        a.plot(TF, m, color=cl, lw=lw, marker="o" if r == "p" else None, ms=2.5)
                else:
                    a.plot(TG[1:], GV[k][..., a_].mean(0), color=BLACK, lw=1.6)
                    for r, cl, lw in (("z", GREEN, 1.1), ("p", ORANGE, 1.8)):
                        pos, vel, say, sv = R[r]
                        m = masked_mean(vel[k][..., a_], sv[k])
                        if r == "p": m[absent] = np.nan
                        a.plot(TF, m, color=cl, lw=lw, marker="o" if r == "p" else None, ms=2.5)
                a.axvline(0.5, color="#999999", lw=0.8, ls=":")
                a.axhline(0, color="#cccccc", lw=0.6)
                a.scatter(TF[absent], np.full(absent.sum(), 0.06), transform=a.get_xaxis_transform(), s=14, color="red", zorder=4)
                a.spines[["top", "right"]].set_visible(False)
                a.tick_params(labelsize=7)
            ax[row, 0].set_ylabel(f"{law} · {'xy'[a_]}", fontsize=8.5, fontweight="bold" if a_ == 0 else "normal")
    ax[0, 0].set_title("position: displacement from last seen (px)\nx: + = right · y: + = up", fontsize=9)
    ax[0, 1].set_title("velocity (px / tubelet)\nx: + = right · y: + = up", fontsize=9)
    for col in range(2):
        ax[-1, col].set_xlabel("tubelet (0 = last context tubelet; future = 1..)")
    hd = [Line2D([], [], color=BLACK, lw=1.6, label="GT"), Line2D([], [], color=ORANGE, lw=1.8, marker="o", ms=3, label="p (own decoder)"),
          Line2D([], [], color=GREEN, lw=1.1, label="z (sees the actual frames): readout check"),
          Line2D([], [], ls="", marker="o", ms=4, color="red", label="p reads absent (>50% of clips)"),
          Line2D([], [], color="#999999", ls=":", label="context end")]
    f_.legend(handles=hd, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    f_.suptitle(f"RollOut v3 · context {c}f / prediction {p}f · position and velocity over time (mean over clips) · {DEC}", fontsize=10)
    f_.tight_layout(rect=(0, 0.012, 1, 0.985))
    f_.savefig(OUT / f"fig_C{c}_P{p}.png", dpi=110); plt.close(f_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="*", default=None, metavar="CxP", help="기본: 16 창 전부")
    a = ap.parse_args()
    wins = ([tuple(int(v) for v in w.lower().split("x")) for w in a.windows] if a.windows
            else [(c, p) for c in (4, 8, 16, 32) for p in (4, 8, 16, 32)])
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    thr = {r: present_thr(r, PRES) for r in ("p", "z")}                          # 자 종류마다 뜻이 다르다 (rollout3_paths)
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for c, p in wins:
        draw(z, thr, bias, c, p)
    stamp(OUT, PRES, [WIN / "readings.npz"])                 # 어느 자로 그렸나 (_decoder.json)
    print(OUT)


if __name__ == "__main__":
    main()
