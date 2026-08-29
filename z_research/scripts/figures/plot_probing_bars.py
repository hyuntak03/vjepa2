#!/usr/bin/env python3
"""Figure — attentive probing self-accuracy. 세 지점(z / h / p) x 조건 4개.

세 지점은 전부 (n_tokens, 1280) 이고 같은 probe 사양으로 읽는다:
  z = context encoder(ctx_masked)   h = target encoder    p = predictor 출력

⚠️ 여기 쓰는 head 는 전부 `groups=None` (train 1510개 전체) 이다.
   조건별로 학습한 head 는 n_train 이 375~382 로 작아 수렴하지 않는다
   (v10 실측: color/static_occlusion train_acc 0.323, shape/static_occlusion 0.773).
   그 값들은 표현이 아니라 학습 실패를 재는 것이라 그림에 넣지 않는다.

⚠️ 이 실행의 protocol 은 `eval: [self]` 뿐이라 h->p 이식(정렬손실)이 없다.
   CLAUDE.md §5-4 의 분해를 다시 하려면 runs[].eval 에 이식 대상을 넣고 다시 돌려야 한다.

  python z_research/scripts/figures/plot_probing_bars.py \
      --summary z_research/IntPhysGenV10/exp_results/attn_probe__v10_vith/summary.json \
      --output z_research/IntPhysGenV10/figures/fig4_probing.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
# flat 팔을 함께 그릴 때의 순서. ramp/flat 짝을 붙여 둔다.
COND6 = ["static_visible", "moving_visible", "moving_visible_flat",
         "moving_occlusion", "moving_occlusion_flat", "static_occlusion"]
# 패널 하나가 4그룹에 ~2.1in 뿐이라 "no occluder" 는 안 들어간다
CLABEL = {"static_visible": "Static\nvisible", "moving_visible": "Moving\nvisible",
          "moving_occlusion": "Moving\noccluded", "static_occlusion": "Static\noccluded"}
# (fit 이름, 범례 라벨, 색)
SRC = [("contextF__f1to16", "$z$  context encoder", BLUE),
       ("targetF__f17to32", "$h$  target encoder", ORANGE),
       ("pred__f17to32", "$p$  predictor", AQUA)]
PANEL = ["(a)", "(b)", "(c)"]
INK_D = "#111111"
BAND = "#efefef"                      # 가림 구간 음영
# 같은 파랑의 명도 3단계 — 계열이 아니라 "같은 축 위의 진행"이라 단색조로 간다
TINTS = (0.52, 0.30, 0.00)
MARKS = ("s", "^", "o")
# x축 2단 라벨: 위 = 정지/이동, 아래 = 가림 유무 (참조 그림과 같은 배치)
XTOP = {"static_visible": "Static", "moving_visible": "Moving",
        "moving_occlusion": "Moving", "static_occlusion": "Static",
        "moving_visible_flat": "Moving", "moving_occlusion_flat": "Moving"}
# 6조건일 때는 패널 하나에 ~2.1in 뿐이라 "Moving" 이 안 들어간다
XTOP6 = {"static_visible": "Static", "moving_visible": "Mov",
         "moving_occlusion": "Mov", "static_occlusion": "Static",
         "moving_visible_flat": "Mov", "moving_occlusion_flat": "Mov"}
XSUB = {"moving_visible": "ramp", "moving_occlusion": "ramp",
        "moving_visible_flat": "flat", "moving_occlusion_flat": "flat"}
# 색 = 지면. flat 만 주황으로 뺀다 (마커 모양은 z/h/p 그대로).
def cond_base(c):
    return ORANGE if c.endswith("_flat") else BLUE
TGT_CAP = {"shape": "shape", "color": "colour", "env": "background (env)"}


def tint(hex_color: str, f: float) -> str:
    """흰색과 섞어 밝은 색조. alpha 를 쓰면 음영 띠가 비쳐 탁해진다."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def group_fit(P, targets, output: Path):
    """조건별로만 학습한 head 를 그 조건에서 평가. train_acc 를 나란히 둬서
    "표현이 나쁜 것"과 "학습이 안 된 것"을 눈으로 가르게 한다."""
    def rec(tgt, grp):
        for r in P:
            if r["fit"] == "pred__f17to32" and r["target"] == tgt \
                    and (r.get("groups") or None) == grp:
                return r
        raise SystemExit(f"없음: target={tgt} groups={grp}")

    fig, axes = plt.subplots(1, len(targets), figsize=(7.0, 2.85))
    axes = np.atleast_1d(axes)
    W = 0.30
    for pi, (ax, tgt) in enumerate(zip(axes, targets)):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8.5, colors=INK2, pad=2.0)
        pooled = rec(tgt, None)
        chance = 100 * pooled["chance"]
        for ci, c in enumerate(CONDS):
            r = rec(tgt, [c])
            vals = [100 * r["train_acc"],
                    100 * r["evals"]["pred__f17to32"]["per_group"][c]["acc"]]
            for bi, (y, col) in enumerate(zip(vals, (BLUE, ORANGE))):
                x = ci + (bi - 0.5) * W
                ax.bar(x, y, width=W * 0.88, color=col, edgecolor=col, lw=0.6,
                       alpha=0.88, zorder=3)
                ax.annotate(f"{y:.1f}", (x, y + 1.4), ha="center", va="bottom",
                            fontsize=6.0, color=INK, rotation=90, zorder=5)
            ref = 100 * pooled["evals"]["pred__f17to32"]["per_group"][c]["acc"]
            ax.plot([ci - 0.5 * W - W * 0.44, ci + 0.5 * W + W * 0.44], [ref, ref],
                    color=INK_D, lw=1.0, ls=(0, (1.6, 1.6)), zorder=6)
            # 막대가 이 높이까지 차 있으므로 흰 배경을 깔아야 읽힌다
            ax.annotate(f"n={r['n_train']}", (ci, 3.0), ha="center", va="bottom",
                        fontsize=5.8, color=INK2, zorder=7,
                        bbox=dict(fc="white", ec="none", pad=0.7))
        ax.axhline(chance, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=6.8, color=INK2,
                           linespacing=1.0)
        ax.set_xlim(-0.5 - W, len(CONDS) - 0.5 + W)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(tgt, fontsize=10, color=INK, pad=5)
        if pi:
            ax.set_yticklabels([])
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    axes[0].set_ylabel("accuracy (%)", fontsize=9.5)

    handles = [Patch(facecolor=BLUE, edgecolor=BLUE, alpha=0.88,
                     label="train acc. (fit on that condition only)"),
               Patch(facecolor=ORANGE, edgecolor=ORANGE, alpha=0.88,
                     label="val acc. on that condition"),
               Line2D([], [], color=INK_D, lw=1.0, ls=(0, (1.6, 1.6)),
                      label="head fit on all four (4x data)"),
               Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.2,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.115))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.845, bottom=0.20, wspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {output.with_suffix('.pdf')}  +  .png")

    for tgt in targets:
        print(f"\n{tgt}")
        print(f"  {'condition':<20}{'n_train':>8}{'train_acc':>11}"
              f"{'val(self cond)':>15}{'pooled head':>13}{'차이':>8}")
        for c in CONDS:
            r = rec(tgt, [c])
            v = 100 * r["evals"]["pred__f17to32"]["per_group"][c]["acc"]
            ref = 100 * rec(tgt, None)["evals"]["pred__f17to32"]["per_group"][c]["acc"]
            flag = "  <- train_acc 낮음" if r["train_acc"] < 0.95 else ""
            print(f"  {c:<20}{r['n_train']:>8}{100*r['train_acc']:>11.1f}"
                  f"{v:>15.2f}{ref:>13.2f}{v-ref:>8.2f}{flag}")


def self_line(P, targets, output: Path, per_cond: bool = True):
    """z / h / p 의 self(val) 정확도를 조건축 위에 점·선으로.
    마커는 세 지점 고정: 네모=z(context enc) 세모=h(target enc) 동그라미=p(predictor).

    per_cond=True (기본): **시나리오마다 head 를 따로 학습**해 그 시나리오에서 평가한다
        (fit_groups_sweep 의 조건별 head). 조건 간 타협이 없는 값이라 성능 보고는 이쪽이 맞다.
        ⚠️ n_train 이 375~382 로 얇아 수렴하지 않는 칸이 생긴다. train_acc < 0.95 인 칸은
           그림에는 표시하지 않는다. stdout 표의 (tr ...) 와 '!' 로만 남기므로
           **수치를 인용할 때 train_acc 를 반드시 함께 볼 것** (v10: p/static_occlusion
           shape 0.77, color 0.32 — 학습 데이터조차 못 맞춘 값이다).
    per_cond=False: 네 조건 전부로 학습한 head 하나(n_train 1510)를 조건별로 쪼개 평가.
    """
    def rec(fit, tgt, grp=None):
        # P 는 여러 summary.json 의 probing 을 이어 붙인 것 (ramp 팔 + flat 팔)
        for r in P:
            if r["fit"] == fit and r["target"] == tgt \
                    and (r.get("groups") or None) == grp:
                return r
        raise SystemExit(f"없음: fit={fit} target={tgt} groups={grp}")

    conds = [c for c in COND6
             if any((r.get("groups") or [None])[0] == c for r in P)] or CONDS

    LAB = ["$z$  context enc.", "$h$  target enc. (GT frames)",
           "$p$  predictor (predicted frames)"]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.0, 2.95))
    axes = np.atleast_1d(axes)
    DX = 0.23
    for pi, (ax, tgt) in enumerate(zip(axes, targets)):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_axisbelow(True)
        occ = [i for i, c in enumerate(conds) if "occlusion" in c]
        ax.axvspan(min(occ) - 0.5, max(occ) + 0.5 + DX + 0.25,
                   color=BAND, lw=0, zorder=0)
        ax.grid(axis="y", color="white", lw=0.9, zorder=1)
        ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.4, zorder=1)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8.5, colors=INK2, pad=2.0)

        chance = 100 * rec(SRC[0][0], tgt)["chance"]
        for ci, c in enumerate(conds):
            rs = [rec(f, tgt, [c] if per_cond else None) for f, _l, _c in SRC]
            ys = [100 * r["evals"][r["fit"]]["per_group"][c]["acc"] for r in rs]
            xs = [ci + (k - 1) * DX for k in range(3)]
            base = cond_base(c)          # 색 = 지면 (flat 만 주황)
            ax.plot(xs, ys, color=tint(base, 0.34), lw=1.1, zorder=3,
                    solid_capstyle="round")
            for x, y, t, mk in zip(xs, ys, TINTS, MARKS):
                ax.plot(x, y, mk, ms=7.0, color=tint(base, t), mec="white", mew=0.8,
                        zorder=4, clip_on=False)
        ax.axhline(chance, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
        ax.annotate(f"chance {chance:.1f}%", (-0.5 - DX, chance + 2.5),
                    ha="left", va="bottom", fontsize=6.4, color=MUTED, zorder=5)

        ax.set_xticks(range(len(conds)))
        n6 = len(conds) > 4
        ax.set_xticklabels([(XTOP6 if n6 else XTOP)[c]
                            + (f"\n{XSUB[c]}" if n6 and c in XSUB else "")
                            for c in conds], fontsize=7.0 if n6 else 8.6, color=INK,
                           linespacing=1.0)
        ax.set_xlim(-0.5 - DX, len(conds) - 0.5 + DX)
        ax.tick_params(axis="x", length=0, pad=3)
        vis = [i for i, c in enumerate(conds) if "occlusion" not in c]
        for xs_, lab in ((np.mean(vis), "no occluder"), (np.mean(occ), "occluded")):
            ax.annotate(lab, (xs_, -0.215 if n6 else -0.145),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8.0, color=INK2, style="italic")
        if pi:
            ax.set_yticklabels([])
        ax.text(0.5, -0.30, f"{PANEL[pi]} {TGT_CAP.get(tgt, tgt)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    axes[0].set_ylabel("probe acc. (%)", fontsize=9.5)

    handles = [Line2D([], [], ls="none", marker=MARKS[i], ms=7.0,
                      color=tint(BLUE, TINTS[i]), mec="white", mew=0.8, label=LAB[i])
               for i in range(3)]
    if len(conds) > 4:
        handles += [Line2D([], [], color=tint(BLUE, 0.10), lw=3.0,
                           label="ramp / static ground"),
                    Line2D([], [], color=tint(ORANGE, 0.10), lw=3.0,
                           label="flat ground")]
    handles.append(Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance"))
    ncol, yanc, bot = (4, 1.005, 0.235) if len(conds) <= 4 else (5, 1.055, 0.285)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, fontsize=8.3,
               handlelength=1.2, handletextpad=0.6, columnspacing=1.5,
               bbox_to_anchor=(0.5, yanc))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.935, bottom=bot, wspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {output.with_suffix('.pdf')}  +  .png")

    src = "시나리오별 head" if per_cond else "전체 학습 head 하나"
    for tgt in targets:
        print(f"\n{tgt}   ({src})")
        print(f"  {'':<14}" + "".join(f"{c.replace('_','+'):>22}" for c in conds))
        for f, lab, _c in SRC:
            cells = []
            for c in conds:
                r = rec(f, tgt, [c] if per_cond else None)
                a = 100 * r["evals"][f]["per_group"][c]["acc"]
                cells.append(f"{a:7.2f} (tr {r['train_acc']:.2f})"
                             + ("!" if r["train_acc"] < 0.95 else " "))
            print(f"  {lab.split('  ')[-1]:<14}" + "".join(f"{x:>22}" for x in cells))


def group_fit_line(P, targets, output: Path):
    """group_fit 을 점·선으로. 조건마다 세 값을 선으로 이어 낙차를 바로 읽게 한다."""
    def rec(tgt, grp):
        for r in P:
            if r["fit"] == "pred__f17to32" and r["target"] == tgt \
                    and (r.get("groups") or None) == grp:
                return r
        raise SystemExit(f"없음: target={tgt} groups={grp}")

    SER = [("train acc. (fit on that condition only)", 0),
           ("val acc. on that condition", 1),
           ("val acc., head fit on all four (4x data)", 2)]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.0, 2.95))
    axes = np.atleast_1d(axes)
    DX = 0.23
    for pi, (ax, tgt) in enumerate(zip(axes, targets)):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_axisbelow(True)
        # 가림 구간 음영 — 격자보다 아래
        ax.axvspan(1.5, 3.5 + DX + 0.25, color=BAND, lw=0, zorder=0)
        ax.grid(axis="y", color="white", lw=0.9, zorder=1)
        ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.4, zorder=1)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8.5, colors=INK2, pad=2.0)

        pooled = rec(tgt, None)
        chance = 100 * pooled["chance"]
        for ci, c in enumerate(CONDS):
            r = rec(tgt, [c])
            ys = [100 * r["train_acc"],
                  100 * r["evals"]["pred__f17to32"]["per_group"][c]["acc"],
                  100 * pooled["evals"]["pred__f17to32"]["per_group"][c]["acc"]]
            xs = [ci + (k - 1) * DX for k in range(3)]
            ax.plot(xs, ys, color=tint(BLUE, 0.34), lw=1.1, zorder=3,
                    solid_capstyle="round")
            for x, y, t, mk in zip(xs, ys, TINTS, MARKS):
                ax.plot(x, y, mk, ms=7.0, color=tint(BLUE, t), mec="white", mew=0.8,
                        zorder=4, clip_on=False)
        ax.axhline(chance, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)

        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([XTOP[c] for c in CONDS], fontsize=8.6, color=INK)
        ax.set_xlim(-0.5 - DX, len(CONDS) - 0.5 + DX)
        ax.tick_params(axis="x", length=0, pad=3)
        # 아래 단: 가림 유무를 두 그룹에 걸쳐 하나씩
        for xc, lab in ((0.5, "no occluder"), (2.5, "occluded")):
            ax.annotate(lab, (xc, -0.145), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8.0, color=INK2, style="italic")
        if pi:
            ax.set_yticklabels([])
        ax.text(0.5, -0.30, f"{PANEL[pi]} {TGT_CAP.get(tgt, tgt)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    axes[0].set_ylabel("probe acc. (%)", fontsize=9.5)

    handles = [Line2D([], [], ls="none", marker=MARKS[i], ms=7.0,
                      color=tint(BLUE, TINTS[i]), mec="white", mew=0.8, label=lab)
               for lab, i in SER]
    handles.append(Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.3,
               handlelength=1.2, handletextpad=0.6, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.020))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.895, bottom=0.235, wspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {output.with_suffix('.pdf')}  +  .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True, action="append",
                    help="여러 번 주면 이어 붙인다 (ramp 팔 + flat 팔)")
    ap.add_argument("--targets", nargs="*", default=["shape", "color", "env"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--head", choices=["per-condition", "pooled"], default="per-condition",
                    help="per-condition = 시나리오마다 head 를 따로 학습(성능 보고용, 기본). "
                         "pooled = 네 조건 전부로 학습한 head 하나")
    ap.add_argument("--line", action="store_true",
                    help="막대 대신 점·선 (fig_attn_probe_selfonly 스타일)")
    ap.add_argument("--group-fit", action="store_true",
                    help="조건별로만 학습한 head 를 train_acc 와 나란히 그린다")
    a = ap.parse_args()

    P = [r for f in a.summary for r in json.loads(f.read_text())["probing"]]
    if a.group_fit:
        (group_fit_line if a.line else group_fit)(P, a.targets, a.output)
        return
    if a.line:
        self_line(P, a.targets, a.output, per_cond=(a.head == "per-condition"))
        return

    def rec(fit, tgt):
        for r in P:
            if r["fit"] == fit and r["target"] == tgt and not r.get("groups"):
                return r
        raise SystemExit(f"없음: fit={fit} target={tgt} groups=None")

    fig, axes = plt.subplots(1, len(a.targets), figsize=(7.0, 2.75))
    axes = np.atleast_1d(axes)
    W = 0.26
    for pi, (ax, tgt) in enumerate(zip(axes, a.targets)):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8.5, colors=INK2, pad=2.0)

        chance = None
        for si, (fit, _lab, col) in enumerate(SRC):
            r = rec(fit, tgt)
            chance = 100 * r["chance"]
            pg = r["evals"][fit]["per_group"]
            for ci, c in enumerate(CONDS):
                x = ci + (si - 1) * W
                y = 100 * pg[c]["acc"]
                ax.bar(x, y, width=W * 0.88, color=col, edgecolor=col, lw=0.6,
                       alpha=0.88, zorder=3)
                # aqua 는 밝은 배경에서 대비가 낮다 -> 모든 막대에 값을 직접 단다
                ax.annotate(f"{y:.1f}", (x, y + 1.4), ha="center", va="bottom",
                            fontsize=5.9, color=INK, rotation=90, zorder=5)
        ax.axhline(chance, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.annotate(f"chance {chance:.1f}%", (-0.5 - W, chance + 3.0),
                    ha="left", va="bottom", fontsize=6.2, color=MUTED, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=0.8))

        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=6.8, color=INK2,
                           linespacing=1.0)
        ax.set_xlim(-0.5 - W, len(CONDS) - 0.5 + W)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(tgt, fontsize=10, color=INK, pad=5)
        if pi:
            ax.set_yticklabels([])
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    axes[0].set_ylabel("probe acc. (%)", fontsize=9.5)

    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.88, label=l) for _f, l, c in SRC]
    handles.append(Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.6,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.045))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.845, bottom=0.20, wspace=0.09)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png")

    for tgt in a.targets:
        print(f"\n{tgt}  (chance {100*rec(SRC[0][0], tgt)['chance']:.1f}%)")
        print(f"  {'':<22}{'overall':>9}" + "".join(f"{c.replace('_','+'):>17}" for c in CONDS))
        for fit, lab, _c in SRC:
            r = rec(fit, tgt); pg = r["evals"][fit]["per_group"]
            nm = lab.split("  ")[-1]
            print(f"  {nm:<22}{100*r['evals'][fit]['overall']:>9.2f}"
                  + "".join(f"{100*pg[c]['acc']:>17.2f}" for c in CONDS)
                  + f"   train_acc={r['train_acc']:.3f}")


if __name__ == "__main__":
    main()
