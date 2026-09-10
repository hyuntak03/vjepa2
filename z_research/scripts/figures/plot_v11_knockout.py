#!/usr/bin/env python3
"""v11_full attention knockout 그림 — 어느 간선을 어느 층에서 끊으면 채점이 무너지는가.

입력  exp_results/knockout__v11_full_vith/results.json  (merge.py 산출물, `breakdown` 포함)

  python z_research/scripts/figures/plot_v11_knockout.py \
    --results z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/results.json \
    --outdir  z_research/IntPhysGenV11_occlusion_timing_ablation/figures

── 읽는 법 ────────────────────────────────────────────────────────────────────
x 축은 **창의 중심 층**이다 (`--width` 1 = 층 하나씩, 3 = ±1, 7 = ±3; 양 끝은 잘린다).
맨 오른쪽 `all` 은 12층 전부 끊은 것. y 는 **절대 정확도(%)** 이고 회색 점선이 같은 부분집합의
기준선(`clean_null`), 검은 점선이 chance 50 이다.

⚠️ Δ 가 아니라 절대값을 그리는 이유: late 처럼 기준선이 이미 chance 근처(55%)인 칸은
   떨어질 여지가 5pp 뿐이라 Δ 만 보면 "영향이 없다" 로 잘못 읽힌다. 기준선을 같이 그린다.
⚠️ 간선 이름은 `<query>-><key>` 다. 사용자 표기 "context → mask (mask query 가 context key 를
   못 읽음)" 은 하네스의 `mask->ctx` 이고, 그림에는 "mask reads ctx" 로 적는다.

── 색 규칙 ─────────────────────────────────────────────────────────────────────
한 그림에 범주형 색상 척도는 하나 — **간선 3종** (mask reads ctx 파랑 / mask reads mask 초록 /
ctx reads ctx 주황). violation·motion·timing·k 는 전부 **패널**로 뺀다.

  --width 1|3|7      results.json 에 그 폭의 명세가 있어야 한다 (each = 1)
  --edges ctx|hidden ctx = mask->ctx / mask->mask / ctx->ctx,  hidden = mask->hidden / mask->hidden_ctrl
  --figbase DIR      폴더 이름 (기본 06_knockout). --sub 로 하위 (예: w3)

폴더:  06_knockout/                 전체
       06_knockout/by_violation/    패널 = violation
       06_knockout/by_motion/       패널 = motion
       06_knockout/by_timing/       패널 = timing (visible/early/mid/late)  (+ violation 별 3장)
       06_knockout/by_k/            패널 = k (가림 조건만; visible 의 k 는 명목값이라 뺀다)  (+ violation 별 3장)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
PANEL = [f"({c})" for c in "abcdefghi"]
LGF = 8.5

PRESET = {
    "ctx": (["mask->ctx", "mask->mask", "ctx->ctx"],
            {"mask->ctx": "mask reads ctx", "mask->mask": "mask reads mask", "ctx->ctx": "ctx reads ctx"},
            {"mask->ctx": BLUE, "mask->mask": GREEN, "ctx->ctx": ORANGE}, None),
    "hidden": (["mask->hidden", "mask->hidden_ctrl"],
               {"mask->hidden": "mask reads occluded ctx frames", "mask->hidden_ctrl": "control: same # of neighbouring frames"},
               {"mask->hidden": BLUE, "mask->hidden_ctrl": ORANGE}, "mask->ctx@all"),
}
EDGES, ELAB, ECOL, REF = PRESET["ctx"]          # main() 이 --edges 로 바꾼다
WIDTH = 7
VIOL = ["vanish", "shape", "color"]
VLAB = {"vanish": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
MOT = ["static", "moving_flat", "moving"]
MLAB = {"static": "Static", "moving_flat": "Moving (flat)", "moving": "Moving (ramp)"}
TIM = ["visible", "early", "mid", "late"]
TLAB = {"visible": "No occluder", "early": "Occluded early", "mid": "Occluded mid", "late": "Occluded late (v11)"}
KS = ["1", "2", "3", "4"]

FIGDIR = {}


def set_figdir(base: str, sub: str = ""):
    root = f"{base}/{sub}".rstrip("/")
    FIGDIR.clear()
    FIGDIR.update({
        "fig_ko_layers": root, "fig_ko_drift": root,
        "fig_ko_violation": f"{root}/by_violation", "fig_ko_motion": f"{root}/by_motion",
        "fig_ko_timing": f"{root}/by_timing", "fig_ko_k": f"{root}/by_k"})
    for v in VIOL:
        FIGDIR[f"fig_ko_timing_{v}"] = f"{root}/by_timing"
        FIGDIR[f"fig_ko_k_{v}"] = f"{root}/by_k"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42,
    "axes.titlesize": 9.2, "axes.labelsize": 8.4, "figure.dpi": 120,
})


# ═══ 데이터 ════════════════════════════════════════════════════════════════════

def load(path: Path):
    R = json.loads(path.read_text())
    v = R["verify"]["resume_vs_forward_max_abs_diff"]
    if v != 0.0:
        raise SystemExit(f"verify.resume_vs_forward_max_abs_diff = {v}; 0 이어야 유효한 실행이다")
    if "breakdown" not in R:
        raise SystemExit("results.json 에 breakdown 이 없다 — merge.py 산출물이어야 한다")
    L = R["meta"]["n_layers"]
    names = {s["name"] for s in R["specs"]}
    need = [n for e in EDGES for n in window_names(e, L)] + [f"{e}@all" for e in EDGES]
    miss = [n for n in need if n not in names]
    if miss:
        raise SystemExit(f"results.json 에 폭 {WIDTH} 명세가 없다 (예 {miss[:3]}). layer_mode={R['meta'].get('layer_mode')} window={R['meta'].get('window')}")
    return R, L


def window_names(edge: str, L: int, r: int | None = None) -> list[str]:
    """중심 층 i=0..L-1 순서의 명세 이름. specs.sweep('window'/'each') 와 같은 규칙."""
    r = (WIDTH - 1) // 2 if r is None else r
    out = []
    for i in range(L):
        a, b = max(0, i - r), min(L - 1, i + r)
        out.append(f"{edge}@L{a}" if a == b else f"{edge}@L{a}-{b}")
    return out


def pool(R, axis: str, keep) -> tuple[dict, int, int]:
    """breakdown[axis] 의 level 중 keep(level_key) 가 참인 것을 **block 수 가중**으로 합친다.
    block 평균의 가중 평균 = 합친 집합의 block 평균이라 score_blocks 의 정의와 같다."""
    tab = R["breakdown"][axis]
    rows = [(v["n_block"], v["n_pair"], v["acc"]) for k, v in tab.items() if keep(k)]
    if not rows:
        return None, 0, 0            # 표본이 작은 스모크에서만 생긴다. 패널을 비워 둔다
    nb = sum(r[0] for r in rows); npair = sum(r[1] for r in rows)
    names = rows[0][2].keys()
    acc = {n: sum(r[0] * r[2][n] for r in rows) / nb for n in names}
    return acc, nb, npair


def check_consistency(R):
    """breakdown 을 전부 합치면 specs/clean_null 의 전체 값과 일치해야 한다."""
    acc, nb, npair = pool(R, "condition", lambda k: True)
    ref = R["clean_null"]["surprise_acc"]
    assert nb == ref["n_block"] and npair == ref["n_pair"], (nb, npair, ref)
    assert abs(acc["clean_null"] - ref["acc"]) < 1e-2, (acc["clean_null"], ref["acc"])
    for s in R["specs"]:
        a = s["metrics"]["surprise_acc"]["acc"]
        assert abs(acc[s["name"]] - a) < 1e-2, (s["name"], acc[s["name"]], a)
    print(f"  [check] breakdown 합 = 전체 (block {nb}, pair {npair}, clean_null {ref['acc']:.2f}%)")


# ═══ 그리기 ════════════════════════════════════════════════════════════════════

def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.axhline(50, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.set_ylim(0, 112); ax.set_yticks((0, 25, 50, 75, 100)); ax.tick_params(labelsize=7.4, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def save(fig, out: Path, name):
    d = out / FIGDIR.get(name, "")
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((d / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {FIGDIR.get(name,'')}/{name}")
    plt.close(fig)


def panel_labels(fig, axes, pad=0.018):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(np.ravel(axes), PANEL):
        bb = ax.get_position(); bot = bb.y0
        parts = list(ax.get_xticklabels()) + ([ax.xaxis.label] if ax.get_xlabel() else [])
        for t in parts:
            try:
                bot = min(bot, t.get_window_extent(rend).transformed(inv).y0)
            except Exception:
                pass
        fig.text(bb.x0 + bb.width / 2, bot - pad, lab, ha="center", va="top", fontsize=9.0, color=INK)


def _label_end(ax, x, vals, colors, fs=6.4):
    """`all` 자리의 값 라벨 — 낮은 것부터 위로 쌓아 겹치지 않게."""
    g = 6.0; prev = -1e9
    for i in np.argsort(vals):
        yy = max(vals[i] + 2.5, prev + g)
        ax.text(x + 0.35, yy, f"{vals[i]:.0f}", ha="left", va="center", fontsize=fs, color=colors[i], zorder=6)
        prev = yy


def layer_panel(ax, R, L, acc: dict, base: float, show_legend: bool):
    """한 패널: x = 창 중심 층 0..L-1 + `all`, 선 = 간선 3종, 점선 = 기준선."""
    xs = list(range(L))
    ax.axhline(base, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=3)
    ends = []
    for e in EDGES:
        ys = [acc[n] for n in window_names(e, L)]
        ax.plot(xs, ys, marker="o", ms=3.4, lw=1.7, color=ECOL[e], mec="white", mew=0.6,
                label=ELAB[e] if show_legend else None, zorder=4)
        ya = acc[f"{e}@all"]
        ax.plot([L + 0.6], [ya], marker="s", ms=4.2, color=ECOL[e], mec="white", mew=0.6, zorder=5)
        ends.append(ya)
    _label_end(ax, L + 0.6, np.array(ends), [ECOL[e] for e in EDGES])
    if REF and REF in acc:                       # 참조: 문맥 전부를 못 읽을 때
        ax.plot([L + 0.6], [acc[REF]], marker="D", ms=4.0, color=INK2, mec="white", mew=0.6, zorder=5)
        ax.text(L + 0.95, acc[REF], f"{acc[REF]:.0f}", ha="left", va="center", fontsize=6.4, color=INK2)
    ax.axvline(L - 0.2, color=MUTED, lw=0.5, alpha=0.6)
    ax.set_xlim(-0.6, L + 1.9)
    ax.set_xticks(xs + [L + 0.6]); ax.set_xticklabels([str(i) for i in xs] + ["all"], fontsize=6.8)
    ax.set_xlabel("knockout layer" if WIDTH == 1 else f"knockout window centre (layer, ±{(WIDTH-1)//2})",
                  fontsize=8.0, color=INK)


def fig_panels(R, L, out, name, levels, getacc, titles, ncol=None, width=9.6):
    got = [(lv, getacc(lv)) for lv in levels]
    got = [(lv, g) for lv, g in got if g[0] is not None]      # 부분집합에 없는 level 은 뺀다
    if not got:
        print(f"  [skip] {name}: 데이터 없음"); return
    levels = [lv for lv, _ in got]
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(width * n / 4 if n > 1 else 3.6, 3.25), sharey=True, squeeze=False)
    axes = axes[0]
    for i, (ax, (lv, (acc, nb, npair))) in enumerate(zip(axes, got)):
        frame(ax)
        layer_panel(ax, R, L, acc, acc["clean_null"], show_legend=(i == 0))
        ax.set_title(f"{titles[lv]}\nbaseline {acc['clean_null']:.1f} · n_pair {npair}",
                     fontsize=8.0, color=INK, pad=4, linespacing=1.15)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=8.4, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    if REF:
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], marker="D", ls="", color=INK2, ms=4)); l.append("mask reads no ctx at all")
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.075 if n > 1 else 0.16, right=0.995, top=0.76, bottom=0.25, wspace=0.10)
    panel_labels(fig, axes)
    save(fig, out, name)


def fig_drift(R, L, out):
    """예측 자체가 얼마나 움직였나 — 정확도가 안 변해도 여기서는 보인다 (README §6)."""
    d = {s["name"]: s["metrics"]["pred_drift"]["rel_l1"] for s in R["specs"] if "pred_drift" in s["metrics"]}
    if not d:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    xs = list(range(L)); ends = []
    for e in EDGES:
        ax.plot(xs, [d[n] for n in window_names(e, L)], marker="o", ms=3.4, lw=1.7, color=ECOL[e],
                mec="white", mew=0.6, label=ELAB[e], zorder=4)
        ax.plot([L + 0.6], [d[f"{e}@all"]], marker="s", ms=4.2, color=ECOL[e], mec="white", mew=0.6, zorder=5)
        ends.append(d[f"{e}@all"])
    for v, e in zip(ends, EDGES):
        ax.text(L + 1.0, v, f"{v:.2f}", fontsize=6.4, color=ECOL[e], va="center")
    ax.axvline(L - 0.2, color=MUTED, lw=0.5, alpha=0.6)
    ax.set_xlim(-0.6, L + 2.4); ax.set_xticks(xs + [L + 0.6]); ax.set_xticklabels([str(i) for i in xs] + ["all"], fontsize=6.8)
    ax.tick_params(labelsize=7.4, colors=INK2)
    ax.set_xlabel("knockout layer" if WIDTH == 1 else f"knockout window centre (layer, ±{(WIDTH-1)//2})",
                  fontsize=8.0, color=INK)
    ax.set_ylabel("|p − p_clean| / |p_clean|", fontsize=8.4, color=INK)
    ax.legend(frameon=False, fontsize=LGF, loc="upper left", handlelength=1.4)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.25)
    panel_labels(fig, [ax])
    save(fig, out, "fig_ko_drift")


def fig_hidden_cells(R, out):
    """hidden 명세의 핵심 판 — 층별 곡선은 평평하고 효과는 전 층(`all`)에서만 나므로,
    `all` 값을 (운동 = 패널) × (타이밍 = x) 칸으로 편다. 점 = 기준선 / hidden / 대조군."""
    TT = ["early", "mid", "late"]
    series = [("clean_null", "baseline (no knockout)", INK2, "o"),
              ("mask->hidden@all", ELAB["mask->hidden"], ECOL["mask->hidden"], "s"),
              ("mask->hidden_ctrl@all", ELAB["mask->hidden_ctrl"], ECOL["mask->hidden_ctrl"], "s")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ax, m in zip(axes, MOT):
        frame(ax); ax.set_ylim(40, 100); ax.set_yticks((50, 75, 100))
        cells = [pool(R, "timing|motion", lambda k, t=t: k == f"{t}|{m}") for t in TT]
        dx = {"clean_null": -0.22, "mask->hidden@all": 0.0, "mask->hidden_ctrl@all": 0.22}
        for name, lab, col, mk in series:
            ys = [c[0][name] for c in cells]
            xs = [i + dx[name] for i in range(len(TT))]
            ax.plot(xs, ys, ls="", marker=mk, ms=5.5, color=col, mec="white", mew=0.6,
                    label=lab if m == MOT[0] else None, zorder=5)
            for x, y in zip(xs, ys):
                ax.text(x, y + 2.2, f"{y:.0f}", ha="center", va="bottom", fontsize=6.4, color=col, zorder=6)
        for i in range(len(TT)):
            ax.axvline(i + 0.5, color=MUTED, lw=0.4, alpha=0.5) if i < len(TT) - 1 else None
        ax.set_xlim(-0.6, len(TT) - 0.4); ax.set_xticks(range(len(TT)))
        ax.set_xticklabels([TLAB[t].replace("Occluded ", "").replace(" (v11)", "") for t in TT], fontsize=7.4)
        ax.set_xlabel("occlusion timing", fontsize=8.0, color=INK)
        ax.set_title(f"{MLAB[m]}\nn_pair {cells[0][2]} per cell", fontsize=8.0, color=INK, pad=4, linespacing=1.15)
    axes[0].set_ylabel("Accuracy (%)  —  all 12 layers cut", fontsize=8.4, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.78, bottom=0.25, wspace=0.10)
    panel_labels(fig, axes)
    FIGDIR["fig_ko_hidden_cells"] = FIGDIR["fig_ko_layers"]
    save(fig, out, "fig_ko_hidden_cells")


# ═══ main ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1, help="1 (층 하나씩) | 3 (±1) | 7 (±3)")
    ap.add_argument("--edges", default="ctx", choices=list(PRESET))
    ap.add_argument("--figbase", default="06_knockout")
    ap.add_argument("--sub", default="")
    a = ap.parse_args()
    global EDGES, ELAB, ECOL, REF, WIDTH
    EDGES, ELAB, ECOL, REF = PRESET[a.edges]
    WIDTH = a.width
    set_figdir(a.figbase, a.sub)
    R, L = load(a.results)
    check_consistency(R)
    out = a.outdir
    m = R["meta"]
    print(f"  {m['dataset']}/{m['model']}  block {m['n_block']} / pair {R['clean_null']['surprise_acc']['n_pair']}"
          f"  명세 {len(R['specs'])}  기준선 clean_null {R['clean_null']['surprise_acc']['acc']:.2f}%")

    # 전체
    ncond = len(R["meta"]["blocks_by_cond"])
    fig_panels(R, L, out, "fig_ko_layers", ["all"], lambda _: pool(R, "condition", lambda k: True),
               {"all": f"All {ncond} conditions"})
    if a.edges == "hidden":
        fig_hidden_cells(R, out)
    fig_drift(R, L, out)
    # 독립 축
    fig_panels(R, L, out, "fig_ko_violation", VIOL, lambda v: pool(R, "violation", lambda k: k == v), VLAB)
    fig_panels(R, L, out, "fig_ko_motion", MOT, lambda mo: pool(R, "motion", lambda k: k == mo), MLAB)
    fig_panels(R, L, out, "fig_ko_timing", TIM, lambda t: pool(R, "timing", lambda k: k == t), TLAB)
    fig_panels(R, L, out, "fig_ko_k", KS,
               lambda kk: pool(R, "k|timing", lambda k: k.split("|")[0] == kk and k.split("|")[1] != "visible"),
               {k: f"k = {k}  (occluded only)" for k in KS})
    # timing x violation, k x violation
    for v in VIOL:
        fig_panels(R, L, out, f"fig_ko_timing_{v}", TIM,
                   lambda t, v=v: pool(R, "timing|violation", lambda k: k == f"{t}|{v}"),
                   {t: f"{VLAB[v]}\n{TLAB[t]}" for t in TIM})
        fig_panels(R, L, out, f"fig_ko_k_{v}", KS,
                   lambda kk, v=v: pool(R, "k|timing|violation",
                                        lambda k: k.split("|")[0] == kk and k.split("|")[1] != "visible"
                                        and k.split("|")[2] == v),
                   {k: f"{VLAB[v]}\nk = {k}  (occluded only)" for k in KS})


if __name__ == "__main__":
    main()
