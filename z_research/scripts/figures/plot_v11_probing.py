#!/usr/bin/env python3
"""v11 attentive probing 그림 일체 — predictions.json + index_probe.csv 에서 전부 재계산.

head 는 **조건(6)** 으로만 학습됐지만, 평가는 val 클립 단위 예측이 남아 있으므로
index 의 어떤 축으로도 사후에 쪼갤 수 있다 (k, 운동 팔, surface...). 재학습 없음.

  self        학습 조건 == 평가 조건
  transfer    학습 조건 != 평가 조건 (head 는 그 조건을 본 적이 없다)

  fig_self_condition    조건 6 x z/h/p x 3 타깃          — 어디서 떨어지는가
  fig_self_k            k dose (0=가림없음, 1~4) x z/h/p — 가림 길이가 표현을 깎는가
  fig_self_arm_k        운동 팔 3 x k x (타깃 x 지점) 9칸 — 전수 분해
  fig_transfer_matrix   6x6 이식 행렬 x 9 (타깃 x 지점)
  fig_confusion_cond    가림 3조건 x shape/colour, p     — 무엇과 헷갈리는가
  fig_confusion_k       sta+OCC 를 k 별로               — 혼동 구조가 k 로 변하는가
  fig_confusion_transfer 이식 실패 시 붕괴 + env 대조군

⚠️ 모든 값은 summary.json 의 per_group 과 대조 검증한다 (`--verify` 출력).

  python z_research/scripts/figures/plot_v11_probing.py
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, BAND = "#000000", "#3b3b3b", "#9a9a9a", "#efefef"
CMAP = LinearSegmentedColormap.from_list("wb", ["#ffffff", "#cfe0f4", BLUE])

CONDS = ["static_visible", "static_occlusion", "moving_visible_flat",
         "moving_occlusion_flat", "moving_visible", "moving_occlusion"]
SHORT = {"static_visible": "sta+VIS", "static_occlusion": "sta+OCC",
         "moving_visible_flat": "flat+VIS", "moving_occlusion_flat": "flat+OCC",
         "moving_visible": "ramp+VIS", "moving_occlusion": "ramp+OCC"}
# 운동 팔: (이름, 가림없음 조건, 가림 조건, 색)
ARMS = [("Static", "static_visible", "static_occlusion", BLUE),
        ("Constant velocity", "moving_visible_flat", "moving_occlusion_flat", ORANGE),
        ("Accelerating", "moving_visible", "moving_occlusion", GREEN)]
PTS = [("contextF__f1to16", "z", "context encoder"),
       ("targetF__f17to32", "h", "target encoder"),
       ("pred__f17to32", "p", "predictor")]
TGT = [("shape", "Shape", 1 / 7), ("color", "Colour", 1 / 8), ("env", "Background", 1 / 4)]
KS = ["0", "1", "2", "3", "4"]
PANEL = [f"({c})" for c in "abcdefghi"]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ----------------------------------------------------------------- 데이터
class Runs:
    """predictions.json + index_probe.csv. 어떤 마스크로도 정확도를 낸다."""

    def __init__(self, pred_path, index_path, summary_path):
        P = json.load(open(pred_path))
        idx = {r["video_id"]: r for r in csv.DictReader(open(index_path))}
        self.vids = P["val_video_ids"]
        miss = [v for v in self.vids if v not in idx]
        if miss:
            raise SystemExit(f"index 에 없는 val 클립 {len(miss)}개: {miss[:3]}")
        self.meta = [idx[v] for v in self.vids]
        self.group = np.array(P["val_groups"])
        self.k = np.array([m["sym_k"] for m in self.meta])
        self.cls = {t: P["targets"][t]["classes"] for t in P["targets"]}
        self.gold = {t: np.asarray(P["targets"][t]["gold"]) for t in P["targets"]}
        self.head = {(h["target"], h["fit"], h["groups"][0]): np.asarray(h["pred"])
                     for h in P["heads"]}
        self.summary = json.load(open(summary_path))

    def hit(self, target, fit, train):
        """(N,) bool — 그 head 가 각 val 클립을 맞혔는가."""
        return self.head[(target, fit, train)] == self.gold[target]

    def acc(self, target, fit, train, mask):
        m = np.asarray(mask)
        return float(self.hit(target, fit, train)[m].mean()) * 100 if m.sum() else float("nan")

    def self_acc(self, target, fit, cond, kmask=None):
        m = self.group == cond
        if kmask is not None:
            m = m & kmask
        return self.acc(target, fit, cond, m), int(m.sum())

    def verify(self):
        """summary.json 의 per_group 과 한 칸씩 대조."""
        bad, n = [], 0
        for r in self.summary["probing"]:
            tr = r["groups"][0]
            pg = r["evals"][r["fit"]]["per_group"]
            for c in CONDS:
                a = self.acc(r["target"], r["fit"], tr, self.group == c)
                n += 1
                if abs(a - pg[c]["acc"] * 100) > 0.02:
                    bad.append((r["target"], r["fit"], tr, c, a, pg[c]["acc"] * 100))
        print(f"  [verify] {n}칸 대조 — 불일치 {len(bad)}")
        for b in bad[:5]:
            print("    ", b)
        return not bad


# ----------------------------------------------------------------- 스타일
def frame(ax, ylim, yt):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(*ylim); ax.set_yticks(yt)
    ax.tick_params(labelsize=8.0, colors=INK2, pad=2.0)
    for sp in ax.spines.values():
        sp.set_zorder(30)
    for ax_ in (ax.xaxis, ax.yaxis):
        ax_.set_zorder(30)
        for t in ax_.get_major_ticks():
            t.tick1line.set_zorder(30); t.tick2line.set_zorder(30)


def panel_labels(fig, axs, pad=0.012, fs=9.2):
    """(a)(b)(c) 를 **실제로 그려진 axes 박스** 바로 아래에 붙인다.

    ⚠️ axes-fraction 오프셋을 쓰면 안 된다. aspect="equal" 인 히트맵은 axes 박스가
       subplot 슬롯보다 작게 줄어들어서 같은 오프셋이 그림마다 다른 거리가 된다
       (실제로 한 번은 각주와, 한 번은 아랫줄 제목과 겹쳤다).
       get_tightbbox 는 눈금 라벨·축 라벨·주석까지 포함하므로 그 아래로 내리면 항상 안전하다.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lab in zip(axs, PANEL):
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - pad, lab, ha="center", va="top",
                 fontsize=fs, color=INK)


def save(fig, out: Path, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((out / name).with_suffix(ext), dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  [saved] {name}")
    plt.close(fig)


# ============================================================ 1. self x 조건
def fig_self_condition(R, out, width=9.6, ymin=79.0):
    MK = {"z": "s", "h": "o", "p": "^"}
    COL = {"z": INK2, "h": BLUE, "p": ORANGE}
    xs, xlab, groups = [], [], []
    for gi, (gname, vis, occ, _) in enumerate(ARMS):
        b = gi * 4.3
        xs += [b, b + 1.55]; xlab += ["vis", "occ"]
        groups.append((gname, b + 0.775, b, b + 1.55))
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.75))
    for pi, (ax, (t, tl, ch)) in enumerate(zip(axes, TGT)):
        frame(ax, (ymin, 101.6), (80, 85, 90, 95, 100))
        for _, _, _, x1 in groups:
            ax.axvspan(x1 - 0.72, x1 + 0.72, color=BAND, lw=0, zorder=0)
        conds = [c for _, v, o, _ in ARMS for c in (v, o)]
        for j, (fit, sh, _) in enumerate(PTS):
            dx = (j - 1) * 0.40
            y = [R.self_acc(t, fit, c)[0] for c in conds]
            for _, _, x0, x1 in groups:
                k = [i for i, xx in enumerate(xs) if xx in (x0, x1)]
                ax.plot([xs[i] + dx for i in k], [y[i] for i in k], "-",
                        color=COL[sh], lw=1.3, zorder=3, alpha=0.9)
            ax.plot([x + dx for x in xs], y, MK[sh], color=COL[sh], ms=4.5,
                    zorder=4 + (2 - j), mec="white", mew=0.8)
            for xx, yy in zip(xs, y):
                if yy < 99.5:
                    ax.annotate(f"{yy:.1f}", (xx + dx, yy - 1.4), ha="center", va="top",
                                fontsize=6.8, color=COL[sh], zorder=5)
        ax.set_xlim(-1.0, groups[-1][3] + 1.0)
        ax.set_xticks(xs); ax.set_xticklabels(xlab, fontsize=7.6)
        for gname, gx, _, _ in groups:
            ax.annotate(gname, (gx, -0.145), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7.8, color=INK)
        ax.annotate(f"chance {ch*100:.1f}%", (0.985, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=6.8, color=MUTED, style="italic")
        ax.set_title(tl, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("Val accuracy (%)", fontsize=8.8, color=INK, labelpad=2.0)
    h = [Line2D([], [], color=COL[s], marker=MK[s], ls="-", lw=1.3, ms=4.5, mec="white",
                mew=0.8, label=f"{s}  {d}") for _, s, d in PTS]
    fig.subplots_adjust(top=0.86, bottom=0.26, wspace=0.20)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
               frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=1.8,
               handletextpad=0.5)
    panel_labels(fig, axes)
    fig.text(0.5, -0.015, "each point = one head, trained and tested on that condition. "
             f"y axis truncated at {ymin:g}%.", ha="center", va="top",
             fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_self_condition")


# ================================================================ 2. self x k
def _k_series(R, t, fit, arm=None):
    """k=0 은 가림 없음(visible), k=1~4 는 가림. arm=None 이면 세 팔을 합친다."""
    arms = ARMS if arm is None else [a for a in ARMS if a[0] == arm]
    ys, ns = [], []
    for k in KS:
        hit, tot = 0, 0
        for _, vis, occ, _ in arms:
            c = vis if k == "0" else occ
            m = (R.group == c) & (R.k == k)
            if m.sum():
                hit += int(R.hit(t, fit, c)[m].sum()); tot += int(m.sum())
        ys.append(100 * hit / tot if tot else float("nan")); ns.append(tot)
    return ys, ns


def fig_self_k(R, out, width=9.6, ymin=79.0):
    MK = {"z": "s", "h": "o", "p": "^"}
    COL = {"z": INK2, "h": BLUE, "p": ORANGE}
    xs = [0.0, 1.6, 2.6, 3.6, 4.6]
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.8))
    for pi, (ax, (t, tl, ch)) in enumerate(zip(axes, TGT)):
        frame(ax, (ymin, 101.6), (80, 85, 90, 95, 100))
        ax.axvspan(xs[1] - 0.55, xs[-1] + 0.55, color=BAND, lw=0, zorder=0)
        for j, (fit, sh, _) in enumerate(PTS):
            dx = (j - 1) * 0.10
            y, n = _k_series(R, t, fit)
            ax.plot([x + dx for x in xs[1:]], y[1:], "-", color=COL[sh], lw=1.4, zorder=3)
            ax.plot([xs[0] + dx, xs[1] + dx], y[:2], ":", color=COL[sh], lw=1.0,
                    alpha=0.55, zorder=2)
            ax.plot([x + dx for x in xs], y, MK[sh], color=COL[sh], ms=4.6,
                    zorder=4 + (2 - j), mec="white", mew=0.8)
            for xx, yy in zip(xs, y):
                if yy < 99.5:
                    ax.annotate(f"{yy:.1f}", (xx + dx, yy - 1.4), ha="center", va="top",
                                fontsize=6.5, color=COL[sh], zorder=5)
        ax.set_xlim(-0.8, xs[-1] + 0.8)
        ax.set_xticks(xs); ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=8.0)
        ax.set_xlabel("hidden frames per side  $k$", fontsize=8.4, color=INK, labelpad=1.5)
        ax.annotate(f"chance {ch*100:.1f}%", (0.985, 0.04), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=6.8, color=MUTED, style="italic")
        ax.set_title(tl, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("Val accuracy (%)", fontsize=8.8, color=INK, labelpad=2.0)
    h = [Line2D([], [], color=COL[s], marker=MK[s], ls="-", lw=1.4, ms=4.6, mec="white",
                mew=0.8, label=f"{s}  {d}") for _, s, d in PTS]
    fig.subplots_adjust(top=0.86, bottom=0.24, wspace=0.20)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
               frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=1.8,
               handletextpad=0.5)
    panel_labels(fig, axes)
    fig.text(0.5, -0.02, "three motion arms pooled; k=0 is the visible arm "
             "(no occluder in the scene at all).", ha="center", va="top",
             fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_self_k")


# ==================================================== 4. 이식 행렬 (6x6) x 9
def fig_transfer_matrix(R, out, width=8.6):
    fig, axes = plt.subplots(3, 3, figsize=(width, 8.8), squeeze=False)
    for ri, (t, tl, ch) in enumerate(TGT):
        for ci, (fit, sh, dsc) in enumerate(PTS):
            ax = axes[ri][ci]
            M = np.array([[R.acc(t, fit, tr, R.group == ev) for ev in CONDS]
                          for tr in CONDS])
            ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="equal")
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            fontsize=5.9, color="white" if M[i, j] > 55 else INK2,
                            weight="bold" if i == j else "normal")
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels([SHORT[c] for c in CONDS], rotation=45, ha="right",
                               fontsize=6.2, color=INK2)
            ax.set_yticklabels([SHORT[c] for c in CONDS], fontsize=6.2, color=INK2)
            ax.tick_params(length=0, pad=1.2)
            for s in ax.spines.values():
                s.set_color(MUTED); s.set_linewidth(0.5); s.set_zorder(30)
            off = np.array([[M[i, j] for j in range(6) if j != i] for i in range(6)])
            ax.set_title(f"{tl}  —  {sh}  {dsc}", fontsize=8.6, color=INK, pad=12.0)
            ax.annotate(f"off-diagonal mean {off.mean():.1f}%   chance {ch*100:.1f}%",
                        (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom",
                        fontsize=6.8, color=ORANGE if off.mean() < 70 else INK2,
                        style="italic")
            if ci == 0:
                ax.set_ylabel("trained on", fontsize=7.6, color=INK, labelpad=1.5)
            if ri == 2:
                ax.set_xlabel("evaluated on", fontsize=7.6, color=INK, labelpad=1.5)
    fig.subplots_adjust(wspace=0.36, hspace=0.58)
    panel_labels(fig, [a for r in axes for a in r])
    save(fig, out, "fig_transfer_matrix")


# ================================================================ confusion
def _conf(R, target, fit, train, mask):
    cls = R.cls[target]
    g, q = R.gold[target][mask], R.head[(target, fit, train)][mask]
    K = len(cls)
    M = np.zeros((K, K))
    for a, b in zip(g, q):
        M[a, b] += 1
    acc = float((g == q).mean()) * 100 if len(g) else float("nan")
    row = M.sum(1, keepdims=True)
    return np.divide(M, np.where(row == 0, 1, row)) * 100, cls, acc, int(mask.sum())


def _cpanel(ax, M, cls, title, sub, note_col=INK2, fs=6.0):
    ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="equal")
    K = len(cls)
    for i in range(K):
        for j in range(K):
            if M[i, j] < 1.0:
                continue
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=fs,
                    color="white" if M[i, j] > 55 else INK2,
                    weight="bold" if i == j else "normal")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(cls, rotation=45, ha="right", fontsize=6.2, color=INK2)
    ax.set_yticklabels(cls, fontsize=6.2, color=INK2)
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5); s.set_zorder(30)
    ax.set_title(title, fontsize=8.4, color=INK, pad=12.5)
    ax.annotate(sub, (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=6.8, color=note_col, style="italic")


def _cgrid(R, specs, out, name, ncol, figsize, hspace=0.50):
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    flat = [ax for r in axes for ax in r]
    for k, (ax, sp) in enumerate(zip(flat, specs)):
        M, cls, acc, n = _conf(R, sp["target"], sp["fit"], sp["train"], sp["mask"])
        _cpanel(ax, M, cls, sp["title"],
                f"acc {acc:.1f}%   n={n}" + sp.get("extra", ""),
                ORANGE if acc < 50 else INK2)
        if k % ncol == 0:
            ax.set_ylabel("true", fontsize=7.4, color=INK, labelpad=1.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel("predicted", fontsize=7.4, color=INK, labelpad=1.5)
    for ax in flat[len(specs):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.34, hspace=hspace)
    panel_labels(fig, flat[:len(specs)])
    save(fig, out, name)
    for sp in specs:
        M, cls, acc, n = _conf(R, sp["target"], sp["fit"], sp["train"], sp["mask"])
        top = sorted(((M[i, j], cls[i], cls[j]) for i in range(len(cls))
                      for j in range(len(cls)) if i != j), reverse=True)[:3]
        print(f"    {sp['title']:40s} acc {acc:6.2f} n={n:4d}  " +
              ", ".join(f"{a}->{b} {v:.0f}%" for v, a, b in top if v > 0))


def fig_confusion_transfer_k(R, out):
    """VIS->OCC 이식이 k 와 함께 떨어지면서 **무엇이 되는가**.

    ⚠️ 예전에는 여기에 sta+OCC 의 self confusion 을 그렸다. 버렸다 —
       (1) sta+OCC 는 채점이 오히려 살아남는 조건이라(vanish sens +20.5) 논문의
           어려운 케이스가 아니고, (2) 그 두 p head 가 하필 수렴이 안 된
           셋 중 둘이라(train_acc 0.931 / 0.954) 오분류를 표현 탓으로 읽을 수 없다.
    """
    F, vis, occ = "pred__f17to32", "moving_visible", "moving_occlusion"
    sp = [dict(target=t, fit=F, train=vis, mask=(R.group == occ) & (R.k == k),
               title=f"{tl}   p:  ramp V$\\to$O,  k={k}")
          for t, tl in (("color", "Colour"), ("shape", "Shape")) for k in KS[1:]]
    print("  [confusion] VIS->OCC 를 k 별로 — 붕괴가 깊어지는가")
    _cgrid(R, sp, out, "fig_confusion_transfer_k", 4, (10.4, 6.2))


def fig_confusion_transfer(R, out):
    """팔 안 VIS->OCC 의 오분류. p 와 h 는 같은 미래 구간을 다루는데 결과가 갈린다."""
    vis, occ = "moving_visible", "moving_occlusion"
    S = [("shape", "pred__f17to32", "Shape   p:  ramp VIS $\\to$ OCC"),
         ("shape", "targetF__f17to32", "Shape   h:  ramp VIS $\\to$ OCC"),
         ("color", "pred__f17to32", "Colour   p:  ramp VIS $\\to$ OCC"),
         ("color", "targetF__f17to32", "Colour   h:  ramp VIS $\\to$ OCC")]
    sp = [dict(target=t, fit=f, train=vis, mask=R.group == occ, title=ti)
          for t, f, ti in S]
    print("  [confusion] 팔 안 VIS->OCC — p 는 무너지고 h 는 안 무너진다")
    _cgrid(R, sp, out, "fig_confusion_transfer", 2, (7.0, 7.0), hspace=0.55)


# ============================ 2b. self x k x 운동 타입
# fig_self_k 는 세 팔을 합친다. 그런데 shape·colour 의 `p` 는 **Static 에서만** 떨어지고
# (80~90) 나머지 두 팔은 98~99 라, 합산(92~95)은 어느 쪽도 아닌 값을 만든다.
# 패널 축은 채점 그림(fig_occlusion / fig_k_dose_arm)과 같게 **운동 타입**으로 잡는다.
# 그래야 등속 vs 등가속 비교가 (b) vs (c) 열 비교가 된다.
MOTION = [("Static", "static_visible", "static_occlusion"),
          ("Moving (flat)", "moving_visible_flat", "moving_occlusion_flat"),
          ("Moving (ramp)", "moving_visible", "moving_occlusion")]


def fig_self_k_motion(R, out, width=9.6, ymin=74.0):
    MK = {"z": "s", "h": "o", "p": "^"}
    COL = {"z": INK2, "h": BLUE, "p": ORANGE}
    xs = [0.0, 1.6, 2.6, 3.6, 4.6]
    fig, axes = plt.subplots(3, 3, figsize=(width, 6.1), squeeze=False)
    for ri, (t, tl, ch) in enumerate(TGT):
        for ci, (mname, vis, occ) in enumerate(MOTION):
            ax = axes[ri][ci]
            frame(ax, (ymin, 102.5), (80, 90, 100))
            ax.axvspan(xs[1] - 0.55, xs[-1] + 0.55, color=BAND, lw=0, zorder=0)
            for j, (fit, sh, _) in enumerate(PTS):
                dx = (j - 1) * 0.10
                y = [R.acc(t, fit, vis if k == "0" else occ,
                           (R.group == (vis if k == "0" else occ)) & (R.k == k))
                     for k in KS]
                ax.plot([x + dx for x in xs[1:]], y[1:], "-", marker=MK[sh], color=COL[sh],
                        lw=1.3, ms=4.4, zorder=3, mec="white", mew=0.8)
                ax.plot([xs[0] + dx, xs[1] + dx], y[:2], ":", color=COL[sh], lw=1.0,
                        alpha=0.5, zorder=2)
                ax.plot([xs[0] + dx], y[:1], marker=MK[sh], color=COL[sh], ms=4.8,
                        ls="none", zorder=3, mec="white", mew=0.8)
                for xx, yy in zip(xs, y):
                    if yy < 99.5:
                        ax.annotate(f"{yy:.1f}", (xx + dx, yy - 1.4), ha="center", va="top",
                                    fontsize=6.3, color=COL[sh], zorder=5)
            ax.set_xlim(-0.8, xs[-1] + 0.8)
            ax.set_xticks(xs); ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=7.6)
            ax.set_title(f"{tl}  —  {mname}", fontsize=8.8, color=INK, pad=3.0)
            if ci == 0:
                ax.set_ylabel("Val accuracy (%)", fontsize=8.2, color=INK, labelpad=2.0)
            if ri == 2:
                ax.set_xlabel("hidden frames per side  $k$", fontsize=8.2, color=INK,
                              labelpad=1.5)
            ax.annotate(f"chance {ch*100:.1f}%", (0.98, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=6.6, color=MUTED, style="italic")
    h = [Line2D([], [], color=COL[sh], marker=MK[sh], ls="-", lw=1.3, ms=4.4, mec="white",
                mew=0.8, label=f"{sh}  {d}") for _, sh, d in PTS]
    fig.subplots_adjust(top=0.915, bottom=0.085, wspace=0.20, hspace=0.44)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.008), ncol=3,
               frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=2.0,
               handletextpad=0.5)
    panel_labels(fig, [a for r in axes for a in r])
    fig.text(0.5, -0.012, f"y axis truncated at {ymin:g}%. k=0 is that arm's visible "
             "condition; k$\\geq$1 is its occluded condition sliced by k.",
             ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_self_k_motion")
    print("  self x k x 운동 타입 (p 만)")
    for t, tl, _ in TGT:
        for mname, vis, occ in MOTION:
            y = [R.acc(t, "pred__f17to32", vis if k == "0" else occ,
                       (R.group == (vis if k == "0" else occ)) & (R.k == k)) for k in KS]
            print(f"    {tl:11s} {mname:14s} " + "  ".join(f"k{k}={v:6.2f}"
                                                           for k, v in zip(KS, y)))


# ================= 0. 핵심 그림 — 같은 클립에서 잰 두 측정을 나란히
# 이 논문의 주장은 "표현엔 정보가 있는데 채점 규칙이 못 읽는다" 이고, 그건 **대비**다.
# 그런데 probing 그림만 보면 전부 100 이라 "그래서 뭐" 가 되고, 채점 그림만 보면
# "정보가 없나 보다" 로 읽힌다. 둘을 같은 x 축에 올려야 논증이 그림이 된다.
#
# 위: probing p self (그 조건의 head 를 그 조건에서 평가)  — 정보가 있는가
# 아래: 채점 sensitivity = (fwd+rev)/2 - 50               — 편향을 걷어낸 탐지력
# 축을 합치지 않는다 (chance 도 최대값도 다르다). 두 줄로 쌓아 x 만 공유한다.
def _scoring_cells(report):
    R = json.load(open(report))
    out = {}
    for c in R["scoring"]["cells"]:
        out.setdefault((c["condition"], c["violation_type"]), []).append(c)
    return out


def fig_information_vs_scoring(R, out, report, width=8.0):
    SC = _scoring_cells(report)
    fig, axes = plt.subplots(2, 2, figsize=(width, 5.0), squeeze=False)
    xs = list(range(6))
    for ci, (t, tl, ch) in enumerate([x for x in TGT if x[0] != "env"]):
        # --- 위: probing
        ax = axes[0][ci]
        frame(ax, (0, 108), (0, 25, 50, 75, 100))
        for i, c in enumerate(CONDS):
            if "occlusion" in c:
                ax.axvspan(i - 0.42, i + 0.42, color=BAND, lw=0, zorder=0)
        y = [R.self_acc(t, "pred__f17to32", c)[0] for c in CONDS]
        ax.axhline(ch * 100, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        ax.plot(xs, y, "-^", color=ORANGE, lw=1.4, ms=5, zorder=3, mec="white", mew=0.8)
        for x, v in zip(xs, y):
            ax.annotate(f"{v:.0f}", (x, v - 3.5), ha="center", va="top", fontsize=6.8,
                        color=ORANGE, zorder=5)
        ax.annotate(f"chance {ch*100:.1f}%", (0.985, 0.03), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=6.6, color=MUTED, style="italic")
        ax.set_xticks(xs); ax.set_xticklabels([])
        ax.set_title(tl, fontsize=9.8, color=INK, pad=4.0)
        if ci == 0:
            ax.set_ylabel("probing:  p self (%)", fontsize=8.4, color=INK, labelpad=2.0)
        # --- 아래: 채점
        ax = axes[1][ci]
        frame(ax, (-3, 55), (0, 25, 50))
        for i, c in enumerate(CONDS):
            if "occlusion" in c:
                ax.axvspan(i - 0.42, i + 0.42, color=BAND, lw=0, zorder=0)
        ys, ns = [], []
        for c in CONDS:
            cells = SC.get((c, t), [])
            n = sum(x["n"] for x in cells)
            ys.append(sum(x["sensitivity"] * x["n"] for x in cells) / n if n else float("nan"))
            ns.append(n)
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        ax.plot(xs, ys, "-o", color=BLUE, lw=1.4, ms=5, zorder=3, mec="white", mew=0.8)
        for x, v in zip(xs, ys):
            ax.annotate(f"{v:.1f}", (x, v + 1.8), ha="center", va="bottom", fontsize=6.8,
                        color=BLUE, zorder=5)
        ax.annotate("chance = 0", (0.985, 0.90), xycoords="axes fraction", ha="right",
                    va="top", fontsize=6.6, color=MUTED, style="italic")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[c] for c in CONDS], rotation=40, ha="right", fontsize=7.2)
        if ci == 0:
            ax.set_ylabel("scoring:  sensitivity (pp)", fontsize=8.4, color=INK, labelpad=2.0)
    fig.subplots_adjust(top=0.90, bottom=0.20, wspace=0.22, hspace=0.10)
    panel_labels(fig, [axes[1][0], axes[1][1]], pad=0.028)
    fig.text(0.5, -0.005, "same clips, same tokens. shaded = occluded. "
             "identity stays readable in the predictor output while the scoring rule "
             "loses it.", ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_information_vs_scoring")
    print("    조건별  p self / 채점 sensitivity")
    for t, tl, ch in [x for x in TGT if x[0] != "env"]:
        for c in CONDS:
            cells = SC.get((c, t), []); n = sum(x["n"] for x in cells)
            sens = sum(x["sensitivity"] * x["n"] for x in cells) / n if n else float("nan")
            print(f"      {tl:7s} {c:22s} p {R.self_acc(t,'pred__f17to32',c)[0]:6.2f}   "
                  f"sens {sens:6.2f}  (n_pair {n})")


# ============================== 5. 팔 안 vis <-> occ 이식 (운동 통제)
# 팔 안에서는 궤적이 같다 (fixed_speed, v0 동일). 그래서 "attention query 가 위치
# 특이적이라 안 옮겨간다" 는 설명이 팔 사이에서만큼 강하게 통하지 않는다.
# h 는 **실제로 가려진 미래**를 본 인코더라 p 와 같은 가림을 겪는다.
#
# ⚠️ 재는 것은 **readout 의 이식 여부**뿐이다. "표현이 회전했다/이동했다" 같은
#    기하 주장은 여기서 나오지 않는다 — 각도도 부분공간 겹침도 중심 이동도 안 쟀다.
#    게다가 probe 가 선형이 아니라 attentive pooling + 선형이라, 실패가 클래스를
#    가르는 방향에서 온 것인지 **어느 토큰을 모으느냐**에서 온 것인지 구분되지 않는다.
#    쓸 수 있는 문장: "비가림으로 학습한 head 가 같은 팔의 가림 조건에서 못 읽는다.
#    그 실패는 z·h 보다 p 에서 훨씬 크다."
def _vo(R, t, fit, arm, direction):
    _, vis, occ, _c = arm
    tr, ev = (vis, occ) if direction == "vo" else (occ, vis)
    return R.acc(t, fit, tr, R.group == ev)


def fig_transfer_vis_occ(R, out, width=9.6):
    MK = {"z": "s", "h": "o", "p": "^"}
    COL = {"z": INK2, "h": BLUE, "p": ORANGE}
    xs, xlab, groups = [], [], []
    for gi, arm in enumerate(ARMS):
        b = gi * 5.6
        xs += [b, b + 2.3]; xlab += ["V$\\to$O", "O$\\to$V"]
        groups.append((arm[0], b + 1.15, b, b + 2.3))
    fig, axes = plt.subplots(1, 3, figsize=(width, 3.0))
    for pi, (ax, (t, tl, ch)) in enumerate(zip(axes, TGT)):
        frame(ax, (0, 108), (0, 25, 50, 75, 100))
        for _, _, x0, _ in groups:
            ax.axvspan(x0 - 0.72, x0 + 0.72, color=BAND, lw=0, zorder=0)
        ax.axhline(ch * 100, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        Y = {sh: [_vo(R, t, fit, arm, d) for arm in ARMS for d in ("vo", "ov")]
             for fit, sh, _ in PTS}
        # 같은 x 에서 값이 가까우면 라벨이 겹친다 — 낮은 값부터 아래로 더 내린다
        rank = [{sh: r for r, (sh, _) in
                 enumerate(sorted(((s2, Y[s2][i]) for s2 in Y), key=lambda z: -z[1]))}
                for i in range(len(xs))]
        for j, (fit, sh, _) in enumerate(PTS):
            dx = (j - 1) * 0.42
            y = Y[sh]
            for _, _, x0, x1 in groups:
                k = [i for i, xx in enumerate(xs) if xx in (x0, x1)]
                ax.plot([xs[i] + dx for i in k], [y[i] for i in k], "-",
                        color=COL[sh], lw=1.3, zorder=3, alpha=0.9)
            ax.plot([x + dx for x in xs], y, MK[sh], color=COL[sh], ms=4.5,
                    zorder=4 + (2 - j), mec="white", mew=0.8)
            for i, (xx, yy) in enumerate(zip(xs, y)):
                if yy > 99.5:                       # 천장은 안 적는다 (env 가 전부 100)
                    continue
                ax.annotate(f"{yy:.0f}", (xx + dx, yy - 3.0 - 5.4 * rank[i][sh]),
                            ha="center", va="top", fontsize=6.4, color=COL[sh], zorder=5)
        ax.set_xlim(-1.0, groups[-1][3] + 1.0)
        ax.set_xticks(xs); ax.set_xticklabels(xlab, fontsize=7.4)
        for gname, gx, _, _ in groups:
            ax.annotate(gname, (gx, -0.145), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7.6, color=INK)
        ax.annotate(f"chance {ch*100:.1f}%", (0.985, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=6.6, color=MUTED, style="italic")
        ax.set_title(tl, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("Transfer accuracy (%)", fontsize=8.8, color=INK, labelpad=2.0)
    h = [Line2D([], [], color=COL[s], marker=MK[s], ls="-", lw=1.3, ms=4.5, mec="white",
                mew=0.8, label=f"{s}  {d}") for _, s, d in PTS]
    fig.subplots_adjust(top=0.86, bottom=0.24, wspace=0.20)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
               frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=1.8,
               handletextpad=0.5)
    panel_labels(fig, axes)
    fig.text(0.5, -0.015, "within a motion arm the trajectory is identical, so only the "
             "occluder differs. shaded = train on visible, test on occluded. "
             "this measures readout transfer, not representation geometry.",
             ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_transfer_vis_occ")


def fig_transfer_vis_occ_k(R, out, width=9.0):
    # k=0 은 head 가 **학습한 조건 자신**(= self, 전부 100.0)이다. 이식 값이 아니라
    # 기준점이라 다른 k 그림과 같은 관례로 띄우고 점선으로 잇는다.
    # 이걸 빼면 "가림막이 등장하는 것"(100 -> 45)과 "가림이 길어지는 것"(45 -> 15)이
    # 분리되지 않는다 — 여기서는 뒤쪽도 실재하는 효과라 둘 다 보여야 한다.
    KX = ["0", "1", "2", "3", "4"]
    xs = [0.0, 1.6, 2.6, 3.6, 4.6]
    fig, axes = plt.subplots(3, 3, figsize=(width, 7.4), squeeze=False)
    for ri, (t, tl, ch) in enumerate(TGT):
        for ci, (fit, sh, dsc) in enumerate(PTS):
            ax = axes[ri][ci]
            frame(ax, (0, 108), (0, 25, 50, 75, 100))
            ax.axvspan(xs[1] - 0.55, xs[-1] + 0.55, color=BAND, lw=0, zorder=0)
            ax.axhline(ch * 100, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
            for ai, (aname, vis, occ, col) in enumerate(ARMS):
                dx = (ai - 1) * 0.09
                y = [R.acc(t, fit, vis, (R.group == vis) if k == "0"
                           else ((R.group == occ) & (R.k == k))) for k in KX]
                ax.plot([x + dx for x in xs[1:]], y[1:], "-o", color=col, lw=1.3,
                        ms=3.8, zorder=3, mec="white", mew=0.7, label=aname)
                ax.plot([xs[0] + dx, xs[1] + dx], y[:2], ":", color=col, lw=1.0,
                        alpha=0.5, zorder=2)
                ax.plot([xs[0] + dx], y[:1], "o", color=col, ms=4.2, zorder=3,
                        mec="white", mew=0.7)
            ax.set_xlim(-0.8, xs[-1] + 0.8); ax.set_xticks(xs)
            ax.set_xticklabels(KX, fontsize=7.6)
            ax.set_title(f"{tl}  —  {sh}  {dsc}", fontsize=8.8, color=INK, pad=3.0)
            if ci == 0:
                ax.set_ylabel("VIS$\\to$OCC acc. (%)", fontsize=8.2, color=INK, labelpad=2.0)
            if ri == 2:
                ax.set_xlabel("hidden frames per side  $k$", fontsize=8.2, color=INK,
                              labelpad=1.5)
            ax.annotate(f"chance {ch*100:.1f}%", (0.98, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=6.6, color=MUTED, style="italic")
    h = [Line2D([], [], color=c, marker="o", ls="-", lw=1.3, ms=3.8, mec="white", mew=0.7,
                label=a) for a, _, _, c in ARMS]
    fig.subplots_adjust(top=0.925, bottom=0.09, wspace=0.24, hspace=0.52)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3,
               frameon=False, fontsize=8.4, handlelength=1.8, columnspacing=2.0,
               handletextpad=0.5)
    panel_labels(fig, [a for r in axes for a in r])
    fig.text(0.5, 0.012, "head trained on that arm's VISIBLE condition. k=0 is that same "
             "(visible) condition, i.e. its own self accuracy; k$\\geq$1 is the occluded "
             "condition sliced by k. self accuracy is flat in k, so this is not information loss.",
             ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_transfer_vis_occ_k")


def main():
    ap = argparse.ArgumentParser()
    base = "z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith"
    ap.add_argument("--run", default=base)
    ap.add_argument("--index", default="data_csv/intphysgen_v11/index_probe.csv")
    ap.add_argument("--outdir", default="z_research/IntPhysGenV11/figures/probing")
    ap.add_argument("--report", default="z_research/IntPhysGenV11/exp_results/report.json")
    a = ap.parse_args()
    R = Runs(f"{a.run}/predictions.json", a.index, f"{a.run}/summary.json")
    if not R.verify():
        raise SystemExit("summary.json 과 안 맞는다 — 그림을 만들지 않는다")
    # k 를 쓰는 그림과 안 쓰는 그림을 폴더로 가른다
    cond, byk = Path(a.outdir) / "by_condition", Path(a.outdir) / "by_k"
    print("[by_condition]   핵심: 1 정보 vs 채점 -> 2 이식")
    fig_information_vs_scoring(R, cond, a.report)      # 핵심 1
    fig_transfer_vis_occ(R, cond)                      # 핵심 2
    fig_confusion_transfer(R, cond)                    # 핵심 2 의 기전
    fig_self_condition(R, cond)                        # 부록 — 정보 존재 전수
    fig_transfer_matrix(R, cond)                       # 부록 — 6x6 전수
    print("[by_k]   핵심: 3 k dose -> 4 붕괴")
    fig_transfer_vis_occ_k(R, byk)                     # 핵심 3
    fig_confusion_transfer_k(R, byk)                   # 핵심 4
    fig_self_k(R, byk)                                 # 부록 — self 는 k 에 평평
    fig_self_k_motion(R, byk)                          # 부록 — 운동 타입별 (Static 만 떨어진다)

    print("\n  k dose (세 팔 합산)")
    for t, tl, _ in TGT:
        for fit, sh, _ in PTS:
            y, n = _k_series(R, t, fit)
            print(f"    {tl:11s} {sh}  " + "  ".join(f"k{k}={v:6.2f}" for k, v in zip(KS, y))
                  + "   n=" + ",".join(str(x) for x in n))


if __name__ == "__main__":
    main()
