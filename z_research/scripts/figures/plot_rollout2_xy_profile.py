#!/usr/bin/env python3
"""RollOut_v2 — 화면 x · y 속도 (기본) 또는 위치 (--pos) 를 시간에 따라 그대로 그린다 (사영 없음). 2026-09-19.

속도 = 튜블릿 사이 이동 (px/튜블릿). p · h 의 첫 값은 문맥 마지막 튜블릿 (진실) → 미래 슬롯 0.
자 잡음이 차분에서 두 배가 되므로 속도 판의 흔들림은 위치 판과 같이 읽는다.

fig_step_profile (한 축 사영 + 걸음) 이 읽기 어렵다는 피드백으로 만든 판. 운동 9 종 / 11 패널, 패널마다 위 x(t) · 아래 y(t).
  x : 문맥 끝 (튜블릿 6→7) 에서 오른쪽으로 가도록 clip 마다 좌우를 뒤집는다 (진행 방향 = +x).
  y : 화면 위쪽 = + .
  위치는 문맥 마지막 튜블릿 = 0. 모든 패널 (x · y 모두) 이 같은 px 세로 스케일을 쓴다. 시간축은 튜블릿, 0 = 문맥 마지막, 1..8 = 미래 슬롯 0..7 (회색 음영).
  선: 진실 (문맥 −7..0 + 미래), h, p, 그리고 "문맥 끝 속도를 그대로 이어 간 선" (const. v from context end, 회색 점선).
arc 는 문맥 끝에서 상승 / 하강 clip 을 나눈다. 같은 attentive 자 (v5, test 만), 가능 clip 만. 평균 ± 95 % CI.
  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_xy_profile.py          # 속도 -> fig_xy_velocity
  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_xy_profile.py --pos    # 위치 -> fig_xy_profile
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_test_readout as rt                          # noqa: E402

E = rt.RES_ROOT / "attentive_pooling"; INDEX = rt.ROOT / "data_csv/rollout_v2/index_probe.csv"; FIG = rt.FIG_ROOT / "summary"
RES, T, TC = 144.0, 8, 8
SC = [("flat_v", "flat, constant v", None), ("flat_a", "flat, accelerating", None), ("flat_d", "flat, decelerating", None),
      ("arc", "arc (thrown, bounces)", None), ("ramp_a", "ramp downhill (accel.)", None), ("ramp_d", "ramp uphill (decel.)", None),
      ("wall", "wall (stops at wall)", None), ("ledge", "rolls off ledge", None),
      ("arc", "arc, rising at context end", "rise"), ("arc", "arc, falling at context end", "sink"),
      ("fall", "thrown up then falls", None)]
C_P, C_H, C_T, C_V = "#eb6834", "#1baf7a", "#222222", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def ci(a):
    return a.mean(0), 1.96 * a.std(0, ddof=1) / np.sqrt(len(a))


def collect():
    P, H = np.load(E / "p/preds.npz"), np.load(E / "h/preds.npz")
    idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    assert (P["video_id"] == H["video_id"]).all()
    out = []
    for s, _, flt in SC:
        rec = {q: [] for q in ("ctx", "truth", "p", "h", "cv")}
        for i in np.where((P["scenario"] == s) & (P["plausible"] == 1))[0]:
            r = idx[P["video_id"][i]]
            xy = np.stack([arr(r["px_x_by_sample"]).reshape(16, 2).mean(1), arr(r["px_y_by_sample"]).reshape(16, 2).mean(1)], 1)
            v = xy[TC - 1] - xy[TC - 2]
            if flt and ((v[1] < 0) != (flt == "rise")):                    # 화면 y 아래가 + → 위로 = Δy < 0
                continue
            sgn = np.array([1.0 if v[0] >= 0 else -1.0, -1.0])               # x: 진행 방향 +, y: 위 +
            f = lambda q: (q - xy[TC - 1]) * sgn
            rec["ctx"].append(f(xy[:TC]))
            rec["truth"].append(f((P["truth"][i] + 1) * RES)); rec["p"].append(f((P["pred"][i] + 1) * RES))
            rec["h"].append(f((H["pred"][i] + 1) * RES)); rec["cv"].append(np.arange(1, T + 1)[:, None] * (v * sgn)[None])
        out.append({q: np.array(a) for q, a in rec.items()})
    return out


def main():
    VEL = "--pos" not in sys.argv; name = "fig_xy_velocity" if VEL else "fig_xy_profile"
    D = collect(); FIG.mkdir(parents=True, exist_ok=True)
    tc, tf = np.arange(-TC + 1, 1), np.arange(1, T + 1)
    fig = plt.figure(figsize=(7.0 * 1.75, 11.2)); outer = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.34)
    letters = iter("abcdefghijk"); AX = []
    for j, (s, title, flt) in enumerate(SC):
        d = D[j]; g = outer[j // 4, j % 4].subgridspec(2, 1, hspace=0.12)
        a1 = fig.add_subplot(g[0]); a2 = fig.add_subplot(g[1], sharex=a1); AX += [a1, a2]
        for c, a in enumerate((a1, a2)):
            a.axvspan(0.5, T + 0.5, color="#f1f1f1", lw=0, zorder=0); a.axhline(0, color="#ccc", lw=0.6, zorder=0)
            if VEL:                                                          # 튜블릿 사이 이동, 시각 t 에 (t−1 → t) 를 둔다
                full = np.concatenate([d["ctx"][..., c], d["truth"][..., c]], 1); tv = np.diff(full, axis=1)
                a.plot(np.r_[tc[1:], tf], tv.mean(0), color=C_T, ls=(0, (4, 2)), lw=1.4, marker="o", ms=2.5, mec="white", mew=0.4, label="truth")
                a.plot(np.r_[0, tf], np.full(T + 1, d["cv"][:, 0, c].mean()), color=C_V, ls=":", lw=1.4, label="const. v from context end")
            else:
                m, _ = ci(d["ctx"][..., c]); tr, _ = ci(d["truth"][..., c])
                a.plot(np.r_[tc, tf], np.r_[m, tr], color=C_T, ls=(0, (4, 2)), lw=1.4, marker="o", ms=2.5, mec="white", mew=0.4, label="truth")
                a.plot(np.r_[0, tf], np.r_[0, ci(d["cv"][..., c])[0]], color=C_V, ls=":", lw=1.4, label="const. v from context end")
            for q, col, lab in (("h", C_H, "h (target encoder)"), ("p", C_P, "p (predictor)")):
                s_ = np.diff(np.concatenate([np.zeros_like(d[q][:, :1, c]), d[q][..., c]], 1), axis=1) if VEL else d[q][..., c]
                mm, ee = ci(s_)
                a.fill_between(tf, mm - ee, mm + ee, color=col, alpha=0.18, lw=0)
                if VEL:
                    a.plot(tf, mm, color=col, lw=1.9 if q == "p" else 1.4, marker="o", ms=2.8, mec="white", mew=0.5, label=lab)
                else:
                    a.plot(np.r_[0, tf], np.r_[0, mm], color=col, lw=1.9 if q == "p" else 1.4, marker="o", ms=2.8, mec="white", mew=0.5, label=lab)
            a.set_xlim(-TC + 0.5, T + 0.6); a.tick_params(labelsize=6.5)
        a1.set_title(f"{title}  (n={len(d['p'])})", fontsize=8, color="#222")
        u = " (px/tubelet)" if VEL else " (px)"; a1.set_ylabel(("vx, forward" if VEL else "x, forward") + u, fontsize=6.8)
        a2.set_ylabel(("vy, up" if VEL else "y, up") + u, fontsize=6.8)
        plt.setp(a1.get_xticklabels(), visible=False); a2.set_xlabel("tubelet (0 = last context)", fontsize=7)
        a2.annotate(f"({next(letters)})", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -24), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, annotation_clip=False)
    lo = min(a.get_ylim()[0] for a in AX); hi = max(a.get_ylim()[1] for a in AX)   # 모든 패널 같은 px 스케일
    for a in AX:
        a.set_ylim(lo, hi)
    AX[0].text(4.5, lo, "future", ha="center", va="bottom", fontsize=6.5, color="#777")
    lg = fig.add_subplot(outer[2, 3]); lg.axis("off")
    lg.legend(*a1.get_legend_handles_labels(), loc="center", frameon=False, fontsize=8,
              title=("velocity = displacement per tubelet\n" if VEL else "0 = position at last context tubelet\n") + "top: screen x (motion direction = +)\nbottom: screen y (up = +)\nmean ± 95% CI", title_fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08)
    print(f"-> {FIG / (name + '.png')}")
    for j, (s, title, flt) in enumerate(SC):
        d = D[j]
        print(f"{title:32s} final x: truth {d['truth'][:, -1, 0].mean():6.1f}  cv {d['cv'][:, -1, 0].mean():6.1f}  p {d['p'][:, -1, 0].mean():6.1f}  h {d['h'][:, -1, 0].mean():6.1f}"
              f" | y: truth {d['truth'][:, -1, 1].mean():6.1f}  cv {d['cv'][:, -1, 1].mean():6.1f}  p {d['p'][:, -1, 1].mean():6.1f}  h {d['h'][:, -1, 1].mean():6.1f}")


if __name__ == "__main__":
    main()
