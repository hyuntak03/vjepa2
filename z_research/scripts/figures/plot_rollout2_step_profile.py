#!/usr/bin/env python3
"""RollOut_v2 — predictor 가 미래 8 슬롯에서 물체를 어떻게 옮기는가 (걸음 모양), 운동 9 종 / 11 패널. 2026-09-19.

같은 attentive 자 (RollOut_v2_training v5 에서 정함, test 만) 로 읽은 위치를 운동마다 한 축으로 사영한다.
  along : 문맥 끝 속도 방향 (튜블릿 6→7 의 2D 단위벡터) 으로 사영 — flat_v/a/d, ramp_a/d, wall.
  x     : 문맥 끝 x 진행 방향 (수평 성분만) — arc 의 수평 (등속).
          along / x 는 문맥 끝 속도 < 1 px/튜블릿 인 clip (정지 레벨) 을 방향이 없어 뺀다.
  up    : 화면 위쪽 (+) — ledge, fall, arc 수직. arc 는 문맥 끝에서 상승 중 / 하강 중 clip 을 나눠 그린다 (반반이라 평균하면 상쇄).
시나리오마다 두 패널 (위: 슬롯마다 움직인 거리, 아래: 문맥 마지막 튜블릿에서부터의 누적 위치). 진실 · p · h, 평균 ± 95 % CI.
가능 clip 만. 입력: exp_results/v5/attentive_pooling/{p,h}/preds.npz, data_csv/rollout_v2/index_probe.csv.
출력: figures/v5/summary/fig_step_profile.{png,pdf}. stdout 에 수치와 "p 걸음 / 문맥 끝 속도" (along 축) 를 찍는다.
(2026-09-19 첫 판은 flat 세 개 + attention 패널 — figures/v5/summary/_superseded/step_profile_flat_attn_20260919/)
  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_step_profile.py
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
RES, T = 144.0, 8
# (시나리오, 제목, 축, 필터). 축: along = 문맥 끝 2D 진행 방향, x = 문맥 끝 x 진행 방향 (수평 성분만), up = 화면 위쪽.
# 필터: rise / sink = 문맥 끝 (튜블릿 6→7) 에서 위로 / 아래로 움직이는 clip 만 (arc 는 반반이라 평균하면 상쇄된다)
SC = [("flat_v", "flat, constant v", "along", None), ("flat_a", "flat, accelerating", "along", None), ("flat_d", "flat, decelerating", "along", None),
      ("arc", "arc, horizontal (const. v)", "x", None),
      ("ramp_a", "ramp downhill (accel.)", "along", None), ("ramp_d", "ramp uphill (decel.)", "along", None), ("wall", "wall (stops at wall)", "along", None),
      ("ledge", "rolls off ledge, vertical", "up", None),
      ("arc", "arc rising at boundary, vertical", "up", "rise"), ("arc", "arc falling at boundary, vertical", "up", "sink"),
      ("fall", "thrown up then falls, vertical", "up", None)]
AXLAB = {"along": "along motion", "x": "along x motion", "up": "screen up"}
C_P, C_H, C_T = "#eb6834", "#1baf7a", "#222222"
V_MIN = 1.0
STEP_LAB = ["ctx→0"] + [f"{t-1}→{t}" for t in range(1, T)]
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
    out = {}
    for key, (s, _, ax, flt) in enumerate(SC):
        k = np.where((P["scenario"] == s) & (P["plausible"] == 1))[0]
        st = {"truth": [], "p": [], "h": []}; pos = {"truth": [], "p": [], "h": []}; ratio = []; drop = 0
        for i in k:
            r = idx[P["video_id"][i]]
            xy = np.stack([arr(r["px_x_by_sample"]).reshape(16, 2).mean(1), arr(r["px_y_by_sample"]).reshape(16, 2).mean(1)], 1)   # 튜블릿 중심 (px)
            if flt and ((xy[7, 1] - xy[6, 1] < 0) != (flt == "rise")):       # 화면 y 는 아래가 + → 위로 = Δy < 0
                continue
            if ax in ("along", "x"):
                v = xy[7] - xy[6]
                if ax == "x":
                    v = np.array([v[0], 0.0])
                sp = float(np.linalg.norm(v))
                if sp < V_MIN:
                    drop += 1; continue
                u = v / sp
            else:
                u = np.array([0.0, -1.0]); sp = None                          # 화면 위쪽 = −y
            seq = {"truth": (P["truth"][i] + 1) * RES, "p": (P["pred"][i] + 1) * RES, "h": (H["pred"][i] + 1) * RES}
            for name, q in seq.items():
                proj = np.r_[0.0, (q - xy[7]) @ u]                              # 문맥 마지막 튜블릿 = 0
                st[name].append(np.diff(proj)); pos[name].append(proj[1:])
            if sp:
                ratio.append(np.diff(np.r_[0.0, (seq["p"] - xy[7]) @ u]) / sp)
        n_used = len(st["p"])
        out[key] = {"n": n_used, "drop": drop, "step": {q: np.array(v) for q, v in st.items()},
                  "pos": {q: np.array(v) for q, v in pos.items()}, "ratio": np.array(ratio) if ratio else None}
    return out


def line(ax, x, a, color, ls, lw, band, label=None):
    m, e = ci(a)
    if band:
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, ls=ls, lw=lw, marker="o", ms=2.8, mec="white", mew=0.5, label=label)
    return m


def main():
    D = collect(); FIG.mkdir(parents=True, exist_ok=True); x = np.arange(T)
    fig = plt.figure(figsize=(7.0 * 1.75, 11.2)); outer = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.34)
    SER = (("truth", C_T, (0, (4, 2)), 1.4, False, "truth"), ("h", C_H, "-", 1.4, True, "h (target encoder)"), ("p", C_P, "-", 1.9, True, "p (predictor)"))
    print(f"{'scenario':8s} {'what':10s} " + " ".join(f"{l:>7s}" for l in STEP_LAB))
    letters = iter("abcdefghijk")
    for j, (s, title, axis, flt) in enumerate(SC):
        g = outer[j // 4, j % 4].subgridspec(2, 1, hspace=0.12); d = D[j]; s = f"{s}{'_' + flt if flt else ''}{'_x' if axis == 'x' else ''}"
        a1 = fig.add_subplot(g[0]); a2 = fig.add_subplot(g[1], sharex=a1)
        for a in (a1, a2):
            a.axhline(0, color="#bbb", lw=0.6)
        for name, col, ls, lw, band, lab in SER:
            m = line(a1, x, d["step"][name], col, ls, lw, band, lab); print(f"{s:8s} step {name:5s} " + " ".join(f"{v:7.1f}" for v in m))
            m = line(a2, x, d["pos"][name], col, ls, lw, band); print(f"{s:8s} pos  {name:5s} " + " ".join(f"{v:7.1f}" for v in m))
        if d["ratio"] is not None:
            print(f"{s:8s} p/v_ctx    " + " ".join(f"{v:7.2f}" for v in ci(d["ratio"])[0]))
        a1.set_title(f"{title}  (n={d['n']})", fontsize=8, color="#222")
        a1.set_ylabel(f"step, {AXLAB[axis]} (px)", fontsize=6.8); a2.set_ylabel(f"{AXLAB[axis]} from\nlast context (px)", fontsize=6.8)
        plt.setp(a1.get_xticklabels(), visible=False); a2.set_xticks(x); a2.set_xticklabels([str(t) for t in x]); a2.set_xlabel("future slot", fontsize=7)
        for a in (a1, a2):
            a.set_xlim(-0.3, T - 1 + 0.9); a.tick_params(labelsize=6.5)
        # 누적 위치 직접 라벨 (truth / p), 겹치면 벌린다
        yt, yp = ci(d["pos"]["truth"])[0][-1], ci(d["pos"]["p"])[0][-1]; lo, hi = a2.get_ylim(); gap = 0.1 * (hi - lo)
        if abs(yt - yp) < gap:
            yt, yp = (yt + gap / 2, yp - gap / 2) if yt >= yp else (yt - gap / 2, yp + gap / 2)
        a2.text(T - 1 + 0.15, yt, "truth", fontsize=6, color="#333", va="center"); a2.text(T - 1 + 0.15, yp, "p", fontsize=6, color="#333", va="center")
        a2.annotate(f"({next(letters)})", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -24), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, annotation_clip=False)
    lg = fig.add_subplot(outer[2, 3]); lg.axis("off")                    # 빈 칸 = 범례
    lg.legend(handles=[plt.Line2D([], [], color=c, ls=ls, lw=lw, marker="o", ms=3, mec="white") for _, c, ls, lw, _, _ in SER],
              labels=[lab for *_, lab in SER], loc="center", frameon=False, fontsize=8,
              title="top: step per slot\nbottom: position from last context\nmean \u00b1 95% CI", title_fontsize=7.5)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_step_profile.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig); print(f"-> {FIG / 'fig_step_profile.png'}")
    print("dropped static clips (ctx speed < 1 px/tubelet): " + ", ".join(f"{SC[j][0]}/{SC[j][2]} {D[j]['drop']}" for j in range(len(SC)) if D[j]["drop"]))


if __name__ == "__main__":
    main()
