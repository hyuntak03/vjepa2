#!/usr/bin/env python3
"""VOBench_v1 튜블릿 정체 probe 그림 (2026-10-10). 입력 = vobench_tubelet_probe.py 의 results/<run>/{primitive,realistic}.json.

  fig_probe_transfer   패널 = objset. x = 평가 칸 (encoder 보임 · encoder 가림 · predictor 보임 · predictor late 가림 · late 재등장),
                       쌓은 막대 = 정답 정체 / 다른 물체 / none 으로 답한 비율 (run 평균), 오차막대 = 정답의 run 간 최소–최대
  fig_probe_visible    같은 꼴, visible (가림막 없음) 클립만
  fig_probe_late       패널 = objset. late (문맥 끝 가림) 클립만 시간축 t0–t15: encoder h · predictor p 의 정답 % · none %, 가린 튜블릿 = 빈 표식
  fig_probe_by_t       패널 = objset. x = 미래 튜블릿 t8–t15, 선 = predictor 보임의 정답 % · none %, predictor late (재등장) 의 정답 % · none %,
                       회색 띠 = encoder 보임 천장 (run 간 범위). 띠 = run 간 최소–최대

run = 유효 4 개 (e35_s0_user · e35_s1 · e10_s0 · e10_s1). e35_s0_realistic_only 는 e35_s0_user 와 수치가 같아 빼고,
_invalid_run1_realistic_collapsed 는 무효 (README). 스타일 = plot_v11_surprise.py.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_vobench_probe.py              # head = h 학습 (이식) → z_research/VOBench_v1/vjepa2/probing/figures/
  $P z_research/scripts/figures/plot_vobench_probe.py --set p_only  # head = p 학습 (p 상한) → …/figures/p_only/   (encoder 칸은 p head 를 h 에 건 값)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_v11_surprise import BLUE, ORANGE, GREEN, INK, INK2, MUTED, BAND, PANEL, frame, tint  # noqa: E402  (rcParams 도 같이)

REPO = HERE.parents[2]
BASE = REPO / "z_research/VOBench_v1/vjepa2/probing"
SETS = {   # head 학습 표현 → (run 들, 그림 하위 폴더)
    "h": (["e35_s0_user", "e35_s1", "e10_s0", "e10_s1"], ""),                       # encoder h 보임으로 학습 → h · p 에 (이식)
    "p_only": (["p_only_e35_s0", "p_only_e35_s1", "p_only_e10_s0", "p_only_e10_s1"], "p_only"),   # predictor p 보임 미래로 학습 (p 상한)
}
RUNS, SUB = SETS["h"]
import csv as _csv
OCC_OF = {r["video_id"]: r["occlusion"] for r in _csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv"))}   # 클립 → 가림 시점
OBJ = ["primitive", "realistic"]
CATS = [("① h_vis · object", "encoder\nvisible"),
        ("② h_occ · object", "encoder\noccluded"),
        ("③ p_vis · object", "predictor\nvisible"),
        ("④ p_late · object, 정답 가림", "predictor late\n(still hidden)"),
        ("④ p_late · object, 정답 재등장", "predictor late\n(reappeared)")]
TS = list(range(8, 16))


def load():
    R = {o: [json.load(open(BASE / "results" / r / f"{o}.json")) for r in RUNS] for o in OBJ}
    for o in OBJ:
        for r in R[o]:
            assert r["objset"] == o
    return R


def save(fig, name):
    out = BASE / "figures" / SUB; out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((out / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out / name}.pdf  + .png")


def fig_transfer(R):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    js = {}
    for pi, (ax, o) in enumerate(zip(axes, OBJ)):
        frame(ax, (0, 112), (0, 25, 50, 75, 100))
        js[o] = {}
        for xi, (key, _) in enumerate(CATS):
            acc = np.array([r["cats"][key]["acc"] for r in R[o]])
            oth = np.array([r["cats"][key]["other"] for r in R[o]])
            non = np.array([r["cats"][key]["none"] for r in R[o]])
            m = dict(acc=acc.mean(), other=oth.mean(), none=non.mean())
            js[o][key] = dict(acc=acc.tolist(), other=oth.tolist(), none=non.tolist(), n=R[o][0]["cats"][key]["n"])
            bot = 0
            for v, col in ((m["acc"], BLUE), (m["other"], GREEN), (m["none"], ORANGE)):
                ax.bar(xi, v, bottom=bot, width=0.62, color=tint(col, 0.15), edgecolor="white", lw=0.6, zorder=3)
                bot += v
            ax.errorbar(xi, m["acc"], yerr=[[m["acc"] - acc.min()], [acc.max() - m["acc"]]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=5)
            ax.annotate(f"{m['acc']:.0f}", (xi, 101.5), ha="center", va="bottom", fontsize=7.0, color=BLUE, fontweight="bold", zorder=6)   # 정답 % (막대 위 고정 높이)
        ax.axvspan(1.5, 4.5, color=BAND, lw=0, zorder=0)
        ax.set_xticks(range(len(CATS))); ax.set_xticklabels([l for _, l in CATS], fontsize=6.6, linespacing=0.95)
        ax.set_xlim(-0.55, len(CATS) - 0.45); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(f"{o.capitalize()} objects", fontsize=9.0, color=INK, pad=4)
        if pi == 0:
            ax.set_ylabel("share of test tubelets (%)", fontsize=8.6)
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Patch(facecolor=tint(c, 0.15), label=l) for c, l in
         ((BLUE, "correct identity (shape × colour)"), (GREEN, "another object"), (ORANGE, "none"))]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.8, handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.25, wspace=0.06)
    save(fig, "fig_probe_transfer")
    return js


def fig_by_t(R):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    LINES = [("③ p_vis · object", "acc", BLUE, "-", "o", "predictor visible: correct"),
             ("③ p_vis · object", "none", ORANGE, "-", "o", "predictor visible: none"),
             ("④ p_late · object, 정답 재등장", "acc", BLUE, (0, (2.5, 1.6)), "^", "predictor late (reappeared): correct"),
             ("④ p_late · object, 정답 재등장", "none", ORANGE, (0, (2.5, 1.6)), "^", "predictor late (reappeared): none")]
    js = {}
    for pi, (ax, o) in enumerate(zip(axes, OBJ)):
        frame(ax, (-3, 108), (0, 25, 50, 75, 100))
        ceil = np.array([r["cats"]["① h_vis · object"]["acc"] for r in R[o]])
        ax.axhspan(ceil.min(), ceil.max(), color=MUTED, alpha=0.25, lw=0, zorder=1)
        ax.annotate(f"encoder visible {ceil.mean():.0f}", (8.25, ceil.max() + 1.5), ha="left", va="bottom", fontsize=6.4, color=INK2, style="italic")
        js[o] = {"ceiling": ceil.tolist()}
        for key, met, col, ls, mk, lab in LINES:
            ts = [t for t in TS if str(t) in R[o][0]["cats"][key]["by_t"]]
            Y = np.array([[r["cats"][key]["by_t"][str(t)][met] for t in ts] for r in R[o]])
            js[o][f"{key}|{met}"] = dict(t=ts, y=Y.tolist())
            ax.fill_between(ts, Y.min(0), Y.max(0), color=col, alpha=0.12, lw=0, zorder=2)
            ax.plot(ts, Y.mean(0), ls=ls, marker=mk, color=col, ms=3.6, lw=1.3, mec="white", mew=0.6, zorder=4)
        ax.set_xticks(TS); ax.set_xticklabels([f"t{t}" for t in TS], fontsize=7.2)
        ax.set_xlim(7.6, 15.4); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlabel("future tubelet (t8 = first after the context)", fontsize=7.8, color=INK)
        ax.set_title(f"{o.capitalize()} objects", fontsize=9.0, color=INK, pad=4)
        if pi == 0:
            ax.set_ylabel("share of test tubelets (%)", fontsize=8.6)
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.6, lw=1.3, mec="white", mew=0.6, label=l) for _, _, c, ls, mk, l in LINES]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=7.4, handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, 1.07))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.83, bottom=0.25, wspace=0.06)
    save(fig, "fig_probe_by_t")
    return js


def _timeline(r, src, cond):
    """run 하나에서 cond (가림 시점) 의 물체 클립 샘플 {(clip, t): (state, pred, gt)} — 칸들의 samples 를 모아 중복 제거."""
    out = {}
    for name, c in r["cats"].items():
        if "object" not in name or not name.split(" ")[1].startswith(src + "_"):
            continue
        for clip, t, state, k, pr, gt in c["samples"]:
            if OCC_OF[clip] == cond:
                out[(clip, t)] = (state, pr, gt)
    return out


def fig_timeline(R, cond, name, title):
    """cond 클립만, 시간축 t0–t15. h (t0–15) · p (t8–15) 의 정답 % · none %. 가린 튜블릿 = 빈 표식."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    js = {}
    for pi, (ax, o) in enumerate(zip(axes, OBJ)):
        frame(ax, (-3, 108), (0, 25, 50, 75, 100))
        if cond == "late":
            ax.axvspan(6.5, 8.5, color=BAND, lw=0, zorder=0)
        ax.axvline(7.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.annotate("context | future", (7.5, 104), ha="center", va="bottom", fontsize=6.2, color=INK2, style="italic")
        js[o] = {}
        for src, col, ts in (("h", GREEN, range(16)), ("p", ORANGE, range(8, 16))):
            per_run = []
            for r in R[o]:
                none = r["classes"].index("empty")
                S = _timeline(r, src, cond)
                by, occ = {}, {}
                for (clip, t), (state, pr, gt) in S.items():
                    by.setdefault(t, []).append((pr == gt, pr == none)); occ.setdefault(t, set()).add(state)
                per_run.append((by, occ))
            ts = [t for t in ts if t in per_run[0][0]]
            for mi, (met, ls) in enumerate((("acc", "-"), ("none", (0, (1.2, 1.4))))):
                Y = np.array([[100 * np.mean([x[mi] for x in by[t]]) for t in ts] for by, _ in per_run]); m = Y.mean(0)
                js[o][f"{src}|{met}"] = dict(t=ts, y=Y.tolist(), n=[len(per_run[0][0][t]) for t in ts])
                ax.fill_between(ts, Y.min(0), Y.max(0), color=col, alpha=0.12, lw=0, zorder=2)
                ax.plot(ts, m, ls=ls, color=col, lw=1.3 if met == "acc" else 1.1, zorder=3)
                for t, y in zip(ts, m):
                    hid = per_run[0][1][t] == {"occ"}
                    ax.plot(t, y, marker="o" if met == "acc" else ".", ms=4.2 if met == "acc" else 3.0, ls="none", zorder=4,
                            color="white" if hid else col, mec=col, mew=1.0)
        ax.set_xticks(range(16)); ax.set_xticklabels([f"t{t}" if t % 2 == 0 else "" for t in range(16)], fontsize=7.2)
        ax.set_xlim(-0.5, 15.5); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_xlabel("tubelet (t0–t7 context, t8–t15 future)", fontsize=7.8, color=INK)
        ax.set_title(f"{o.capitalize()} objects, {title}", fontsize=9.0, color=INK, pad=11)
        if pi == 0:
            ax.set_ylabel("share of test tubelets (%)", fontsize=8.6)
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=GREEN, lw=1.3, marker="o", ms=4, label="encoder h: correct identity"),
         Line2D([], [], color=GREEN, lw=1.1, ls=(0, (1.2, 1.4)), label="encoder h: none chosen"),
         Line2D([], [], color=ORANGE, lw=1.3, marker="o", ms=4, label="predictor p: correct identity"),
         Line2D([], [], color=ORANGE, lw=1.1, ls=(0, (1.2, 1.4)), label="predictor p: none chosen")]
    if cond == "late":
        h += [Line2D([], [], color=INK2, marker="o", ms=4, ls="none", mfc="white", label="occluded tubelet"),
              Patch(facecolor=BAND, label="occluder covers the object (late, k = 2, 3)")]
    fig.legend(handles=h, loc="upper center", ncol=3 if cond == "late" else 2, frameon=False, fontsize=7.2, handletextpad=0.5,
               columnspacing=1.4, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.80, bottom=0.25, wspace=0.06)
    save(fig, name)
    return js


def main():
    global RUNS, SUB
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="h", choices=list(SETS)); a = ap.parse_args()
    RUNS, SUB = SETS[a.set]
    R = load()
    js = {"runs": RUNS, "transfer": fig_transfer(R), "by_t": fig_by_t(R), "late": fig_timeline(R, "late", "fig_probe_late", "late occlusion"),
          "visible": fig_timeline(R, "visible", "fig_probe_visible", "no occluder (visible)")}
    json.dump(js, open(BASE / "figures" / SUB / "figure_values.json", "w"), indent=1, ensure_ascii=False)
    for o in OBJ:
        print(o, {k[:16]: f"{np.mean(v['acc']):.1f} [{min(v['acc']):.1f}, {max(v['acc']):.1f}]" for k, v in js["transfer"][o].items()})


if __name__ == "__main__":
    main()
