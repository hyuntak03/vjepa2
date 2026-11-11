#!/usr/bin/env python3
"""VOBench v3 가림 시점별 사라짐 · 나타남 짝 정답률 (pairwise accuracy) — 논문 그림 (2026-11-09, 사용자 "cvpr paper 에 넣을 그림", "vanish appear 나눠서, 움직임 / 정지는 평균").

값은 채점 산출물에서 다시 셈한다 (손으로 옮기지 않는다): 판 가림막 (visible 은 가림막 없음) · 짝 A 사라짐 (pos_a vs imp_ab) · 짝 B 나타남 (pos_b vs imp_ba),
s(불가능) > s(가능) 정답, 같으면 0.5. 움직임 · 정지는 쌍을 합쳐 센다 (조건마다 쌍 수가 같아 두 칸 평균과 같다). 오차 막대 = Wilson 95 %.
판정은 op_cells_vob3_vem.py 와 같다.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_v3_timing.py [--model vith] [--out z_analysis/figures/vobench_v3_timing_release]
  --layout split    : (a) 움직이는 물체 · (b) 정지 물체 두 패널, 패널마다 사라짐 · 나타남 (2026-11-09 "moving / static 나눠야")
  --layout grouped4 : 한 패널, 조건마다 막대 4 개 (색 = 사라짐 · 나타남, 빗금 = 정지)
  --layout pooled   : 움직임 · 정지 합침 (이전 판). --motion moving 이면 움직이는 물체만
  2026-11-10 사용자 "split 에서 (b) 는 빼고": --layout pooled --motion moving --out z_analysis/figures/vobench_v3_timing_release_moving
    (split 의 (a) 패널과 같은 값 · 한 패널이라 패널 라벨 대신 x 축 이름)
"""
import argparse, csv, json, math, os
from collections import defaultdict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = f"{REPO}/cvpr/results/eval/testbed"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5})
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#222222", "#5c5c5c", "#9a9a9a"
# 2026-11-09 사용자 "색 별로, 파스텔톤으로": 채움 = 파스텔 (validate_palette: CVD ΔE 12.7 · 일반 ΔE 17.6 통과, 밝기 · 대비는 파스텔이라 미달 →
#   같은 색상의 진한 테두리 + 값 직접 라벨로 보완)
PASTEL = {"A": ("#8fb8e8", "#3f78b5"), "B": ("#f6a98f", "#c4603e")}     # (채움, 테두리)
NAMES = {"A": "Vanish", "B": "Appear"}


def outcomes(index, per_block, occ, pair="A", motion="all"):
    pv = json.load(open(per_block))["per_video_surprise"]
    by = defaultdict(dict)
    for r in csv.DictReader(open(index)):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    out = []
    for (b, pid), v in by.items():
        p, i = v["pos"], v["imp"]
        is_static = p["motion"] == "flat_static"
        if pid != pair or (motion == "moving" and is_static) or (motion == "static" and not is_static) or not p["occlusion"].startswith(occ):
            continue
        if p["occluder_kind"] != ("none" if occ == "visible" else "panel"):
            continue
        a, c = pv[p["video_id"]], pv[i["video_id"]]
        out.append(1.0 if c > a else (0.5 if c == a else 0.0))
    return out


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


CONDS = ("Visible", "Early", "Mid", "Late")


def cells(model, motion_filter):
    """{(조건, 짝, 운동): (정답률, (CI 아래, 위), n)}. 운동 = moving | static | all."""
    vem = (f"{REPO}/data_csv/vobench_v3_vem/index.csv", f"{RES}/vobench_v3_vem_{model}/per_block.json")
    late = (f"{REPO}/data_csv/vobench_v3/index.csv", f"{RES}/vobench_v3_{model}/per_block.json")
    src = {"Visible": (*vem, "visible"), "Early": (*vem, "early"), "Mid": (*vem, "mid"), "Late": (*late, "late")}
    out = {}
    for c in CONDS:
        ix, pb, occ = src[c]
        for pair in ("A", "B"):
            for mo in motion_filter:
                o = outcomes(ix, pb, occ, pair, mo)
                k, n = sum(o), len(o)
                out[(c, pair, mo)] = (100 * k / n, wilson(k, n), n)
                print(f"{NAMES[pair]:7s} {mo:7s} {c:8s} {100 * k / n:5.1f}  n={n}  95% CI [{out[(c, pair, mo)][1][0]:.1f}, {out[(c, pair, mo)][1][1]:.1f}]")
    return out


def style(ax, ylabel=True):
    ax.axhline(50, color=GRID, lw=0.7, ls=(0, (3, 2)), zorder=1)      # 우연 수준 (캡션에서 설명)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    if ylabel:
        ax.set_ylabel("Pairwise accuracy (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2.5)


def bar(ax, x, v, ci, pair, w, hatch=None):
    fill, edge = PASTEL[pair]
    ax.bar([x], [v], width=w, color=fill, edgecolor=edge, linewidth=0.6, hatch=hatch, zorder=2)
    ax.errorbar([x], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK2, elinewidth=0.55, capsize=1.3, capthick=0.55, zorder=3)
    if v < 99.95:                                                      # 100 은 축으로 읽힌다 — 100 이 아닌 칸만 직접 라벨
        nudge = -2.0 if pair == "A" else 2.0                           # 같은 조건의 이웃 막대 · 라벨과 안 겹치게 사라짐은 왼쪽, 나타남은 오른쪽으로
        ax.annotate(f"{v:.1f}", xy=(x, ci[1]), xytext=(nudge, 1.2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.8, color=INK, zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vith")
    ap.add_argument("--layout", default="split", choices=["split", "grouped4", "pooled"])
    ap.add_argument("--motion", default="all", choices=["all", "moving"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mpl.rcParams["hatch.linewidth"] = 0.5
    if a.layout == "split":
        C = cells(a.model, ("moving", "static"))
        fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.0), sharey=True)
        w, gap = 0.38, 0.03
        for ax, mo, lab in zip(axes, ("moving", "static"), ("(a) Moving objects", "(b) Static objects")):
            for i, c in enumerate(CONDS):
                for j, pair in enumerate(("A", "B")):
                    v, ci, n = C[(c, pair, mo)]
                    bar(ax, i + (j - 0.5) * (w + gap), v, ci, pair, w)
            ax.set_xticks(range(4), CONDS, fontsize=7)
            ax.set_xlim(-0.6, 3.6)
            style(ax, ylabel=ax is axes[0])
            ax.text(0.5, -0.2, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        handles = [mpl.patches.Patch(facecolor=PASTEL[p][0], edgecolor=PASTEL[p][1], linewidth=0.6, label=NAMES[p]) for p in ("A", "B")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7.5, handlelength=1.0, handletextpad=0.4,
                   columnspacing=1.2, bbox_to_anchor=(0.54, 1.0))
        fig.tight_layout(pad=0.3, rect=(0, 0.02, 1, 0.93), w_pad=0.6)
    elif a.layout == "grouped4":
        C = cells(a.model, ("moving", "static"))
        fig, ax = plt.subplots(figsize=(3.3, 2.05))
        w, gap = 0.2, 0.015
        order = [("A", "moving", None), ("A", "static", "/////"), ("B", "moving", None), ("B", "static", "/////")]
        for i, c in enumerate(CONDS):
            for j, (pair, mo, h) in enumerate(order):
                v, ci, n = C[(c, pair, mo)]
                bar(ax, i + (j - 1.5) * (w + gap), v, ci, pair, w, hatch=h)
        ax.set_xticks(range(4), CONDS)
        ax.set_xlim(-0.55, 3.55)
        ax.set_xlabel("Occlusion timing")
        style(ax)
        handles = [mpl.patches.Patch(facecolor=PASTEL[p][0], edgecolor=PASTEL[p][1], linewidth=0.6, hatch=h, label=f"{NAMES[p]} · {mo}")
                   for p, mo, h in order]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=6.6,
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
        fig.tight_layout(pad=0.3)
    else:
        C = cells(a.model, (a.motion,))
        fig, ax = plt.subplots(figsize=(3.25, 2.0))
        w, gap = 0.36, 0.04
        for i, c in enumerate(CONDS):
            for j, pair in enumerate(("A", "B")):
                v, ci, n = C[(c, pair, a.motion)]
                bar(ax, i + (j - 0.5) * (w + gap), v, ci, pair, w)
        ax.set_xticks(range(4), CONDS)
        ax.set_xlim(-0.55, 3.55)
        ax.set_xlabel("Occlusion timing")
        style(ax)
        handles = [mpl.patches.Patch(facecolor=PASTEL[p][0], edgecolor=PASTEL[p][1], linewidth=0.6, label=NAMES[p]) for p in ("A", "B")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5,
                  handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
        fig.tight_layout(pad=0.3)
    out = a.out or f"z_analysis/figures/vobench_v3_timing_release_{a.layout}"
    out = out if os.path.isabs(out) else os.path.join(REPO, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print("저장", out + ".pdf · .png")


if __name__ == "__main__":
    main()
