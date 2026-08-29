#!/usr/bin/env python3
"""Figure 3 — 방향 편향. 세 위반 계열 전부에 대해 정방향/역방향을 나란히 놓는다.

묻는 것: "A -> B 를 잡아내면 B -> A 도 잡아내는가?"
  - 두 방향 모두 높다        -> 위반 자체를 본다
  - 한쪽 ~100 / 반대 ~0      -> 표현 선호(편향)만 있고 위반은 못 본다
  - 두 방향의 평균이 곧 지표다: 평균 50% == 순전히 편향으로 갈린 것.
    (한 쌍의 두 방향은 같은 두 후보 미래를 역할만 바꿔 쓰기 때문이다.)

막대마다 "-> 무엇이 되는가" 를 직접 달아서 색에 의미를 싣지 않는다.

--by-condition 를 주면 조건 4개를 2x2 로 펼친다 (패널 (a)-(d)).
이때 방향의 색은 **전 조건을 합쳤을 때 더 높은 쪽**을 파랑으로 고정한다.
그래야 같은 전이가 네 패널에서 같은 색을 유지해 패널 간 비교가 성립한다.
⚠️ 이 순서는 데이터에서 정한 기술적(descriptive) 규칙이지 사전 가설이 아니다.

  python z_research/scripts/plot_direction_bias.py \
      --result-dir z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith \
      --index data_csv/intphysgen_v10/index_probe.csv \
      --output z_research/IntPhysGenV10/figures/fig3_direction_bias.pdf
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
FAMS = ["vanish", "shape", "color"]
FLABEL = {"vanish": "Object permanence", "shape": "Shape consistency",
          "color": "Color consistency"}
SHORT = {"vanish": "permanence", "shape": "shape", "color": "color"}
PANEL = ["(a)", "(b)", "(c)", "(d)"]
CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
CLABEL = {"static_visible": "Static, no occluder", "moving_visible": "Moving, no occluder",
          "moving_occlusion": "Moving, occluded", "static_occlusion": "Static, occluded"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(result_dir: Path, index_csv: Path):
    S = json.loads((result_dir / "per_block.json").read_text())["per_video_surprise"]
    rows = list(csv.DictReader(index_csv.open()))
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["block_id"], r["pair_id"])].append(r)

    out = []
    for key, v in grp.items():
        pos = [x for x in v if x["plausible"] == "1"]
        imp = [x for x in v if x["plausible"] != "1"]
        if len(pos) != 1 or len(imp) != 1:
            raise ValueError(f"bad matched pair {key}: {len(pos)} pos / {len(imp)} imp")
        p, i = pos[0], imp[0]
        sp, si = S[p["video_id"]], S[i["video_id"]]
        ok = 1.0 if si > sp else (0.5 if si == sp else 0.0)
        fam = i["violation_type"]
        if fam == "vanish":
            # imp_vanish = 물체가 사라진다(미래가 빈다) / imp_appear = 빈 데서 나타난다
            a, b = ("object", "empty") if i["role"] == "imp_vanish" else ("empty", "object")
        elif fam == "shape":
            a, b = p["shape_pre"], i["shape_post"]
        elif fam == "color":
            a, b = p["color_pre"], i["color_post"]
        else:
            continue
        out.append(dict(cond=p["condition"], fam=fam, src=a, dst=b, ok=ok))

    got = float(np.mean([x["ok"] for x in out]))
    want = json.loads((result_dir / "summary.json").read_text())["surprise"]["overall"]["block_pairwise"]
    if abs(got - want) > 1e-9:
        raise ValueError(f"재계산 {got:.6f} != summary.json {want:.6f}")
    print(f"  검증 OK — 재계산 overall {got*100:.4f}% == summary.json {want*100:.4f}%  (n={len(out)})")
    return out


def transitions(pairs, fam, conds=None):
    """무순서 쌍 {A,B} -> [(A->B acc, n), (B->A acc, n)]. 정렬은 A 알파벳순."""
    sel = [x for x in pairs if x["fam"] == fam and (conds is None or x["cond"] in conds)]
    d = collections.defaultdict(list)
    for x in sel:
        d[(x["src"], x["dst"])].append(x["ok"])
    seen, out = set(), []
    for (a, b) in sorted(d):
        k = frozenset((a, b))
        if k in seen:
            continue
        seen.add(k)
        fwd, rev = d.get((a, b), []), d.get((b, a), [])
        if not fwd or not rev:
            continue
        out.append(dict(a=a, b=b,
                        fwd=100 * float(np.mean(fwd)), n_fwd=len(fwd),
                        rev=100 * float(np.mean(rev)), n_rev=len(rev)))
    return out


def abbr(s: str) -> str:
    """축이 빽빽해서 줄인다. 8개 shape / 8개 color 안에서 3글자면 충돌이 없다."""
    return {"object": "obj", "empty": "emp"}.get(s, s[:3])


def panel(ax, rows, order, show_y, title):
    """한 조건 패널. rows 는 transitions() 결과, order 는 전역 방향 고정 규칙."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8, colors=INK2, pad=2.0)
    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)

    W = 0.34
    for gi, r in enumerate(rows):
        hi_dst = order[frozenset((r["a"], r["b"]))]      # 파랑이 될 방향의 도착 상태
        pairs = [(r["fwd"], r["b"]), (r["rev"], r["a"])]
        if pairs[0][1] != hi_dst:
            pairs = pairs[::-1]
        m = 0.5 * (r["fwd"] + r["rev"])
        vals = [v for v, _ in pairs]
        for bi, (val, _dst) in enumerate(pairs):
            x = gi + (bi - 0.5) * W
            ax.bar(x, val, width=W * 0.86, color=(BLUE, ORANGE)[bi],
                   edgecolor=(BLUE, ORANGE)[bi], lw=0.6, alpha=0.85, zorder=3)
            # 라벨이 평균선을 뚫거나 옆 막대 라벨과 붙는 걸 피한다
            y = val + 1.2
            if abs(val - m) < 5.0:
                y = max(val, m) + 5.0
            if bi == 1 and abs(vals[0] - vals[1]) < 4.0:
                y += 6.0
            ax.annotate(f"{val:.0f}", (x, y), ha="center", va="bottom",
                        fontsize=5.2, color=INK, zorder=7)
        ax.plot([gi - 0.5 * W - W * 0.43, gi + 0.5 * W + W * 0.43], [m, m],
                color=INK, lw=0.9, ls=(0, (1.5, 1.5)), zorder=5)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{abbr(r['a'])}\n$\\leftrightarrow${abbr(r['b'])}" for r in rows],
                       fontsize=5.9, color=INK2, linespacing=0.95)
    ax.set_xlim(-0.5 - 0.5 * W, len(rows) - 1 + 0.5 + 0.5 * W)
    ax.tick_params(axis="x", length=0, pad=2)
    if not show_y:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=9.0, color=INK, pad=13)


def family_bands(ax, counts):
    """패널 위쪽에 계열 구분선과 이름. counts 는 [n_vanish, n_shape, n_color]."""
    start = 0
    for (f, n) in zip(FAMS, counts):
        if n == 0:
            continue
        if start:
            ax.axvline(start - 0.5, color=MUTED, lw=0.5, alpha=0.40, zorder=1)
        ax.annotate(SHORT[f], (start + (n - 1) / 2, 116), ha="center", va="bottom",
                    fontsize=6.4, color=INK2, zorder=6)
        start += n


def by_condition(pairs, output: Path):
    # 방향 색은 전 조건을 합친 결과로 고정한다 (패널 간 비교가 성립하도록)
    pooled = {f: transitions(pairs, f) for f in FAMS}
    order = {}
    for f in FAMS:
        for r in pooled[f]:
            order[frozenset((r["a"], r["b"]))] = r["b"] if r["fwd"] >= r["rev"] else r["a"]

    conds = [c for c in CONDS if any(x["cond"] == c for x in pairs)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.75))
    counts = [len(pooled[f]) for f in FAMS]
    for i, (ax, c) in enumerate(zip(axes.ravel(), conds)):
        rows = []
        for f in FAMS:
            rows += transitions(pairs, f, [c])
        panel(ax, rows, order, show_y=(i % 2 == 0), title=CLABEL[c])
        family_bands(ax, counts)
        ax.text(0.5, -0.255, PANEL[i], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("pairwise acc. (%)", fontsize=9)

    handles = [Patch(facecolor=BLUE, edgecolor=BLUE, alpha=0.85,
                     label="direction favored overall"),
               Patch(facecolor=ORANGE, edgecolor=ORANGE, alpha=0.85, label="its reverse"),
               Line2D([], [], color=INK, lw=0.9, ls=(0, (1.5, 1.5)),
                      label="mean of the two (50% = pure direction bias)"),
               Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance (50%)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.2,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.035))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.885, bottom=0.115,
                        hspace=0.62, wspace=0.075)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {output.with_suffix('.pdf')}  +  .png")

    for c in conds:
        print(f"\n{CLABEL[c]}")
        print(f"  {'pair':<24}{'favored':>9}{'reverse':>9}{'mean':>8}{'|gap|':>8}")
        for f in FAMS:
            for r in transitions(pairs, f, [c]):
                hi = order[frozenset((r["a"], r["b"]))]
                fav, rev = ((r["fwd"], r["rev"]) if r["b"] == hi else (r["rev"], r["fwd"]))
                m = 0.5 * (r["fwd"] + r["rev"])
                print(f"  {r['a']+' <-> '+r['b']:<24}{fav:>9.1f}{rev:>9.1f}"
                      f"{m:>8.1f}{abs(fav-rev):>8.1f}")
        for f in FAMS:
            rs = transitions(pairs, f, [c])
            mm = float(np.mean([0.5 * (r["fwd"] + r["rev"]) for r in rs]))
            gg = float(np.mean([abs(r["fwd"] - r["rev"]) for r in rs]))
            print(f"    {SHORT[f]:<12} mean {mm:6.1f}   |gap| {gg:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conds", nargs="*", default=None,
                    help="특정 condition 만 (기본: 전부 합침)")
    ap.add_argument("--by-condition", action="store_true",
                    help="조건 4개를 2x2 패널 (a)-(d) 로 펼친다")
    a = ap.parse_args()

    pairs = load(a.result_dir, a.index)
    if a.by_condition:
        by_condition(pairs, a.output)
        return
    data = {f: transitions(pairs, f, a.conds) for f in FAMS}
    ncol = [len(data[f]) for f in FAMS]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.95),
                             gridspec_kw={"width_ratios": ncol, "wspace": 0.10})
    W = 0.30
    for ax, f in zip(axes, FAMS):
        rows = data[f]
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=9, colors=INK2, pad=2.0)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)

        for gi, r in enumerate(rows):
            for bi, (val, dst, col) in enumerate(
                    [(r["fwd"], r["b"], BLUE), (r["rev"], r["a"], ORANGE)]):
                x = gi + (bi - 0.5) * W
                ax.bar(x, val, width=W * 0.88, color=col, edgecolor=col,
                       lw=0.7, alpha=0.85, zorder=3)
                ax.annotate(f"{val:.0f}", (x, val + 1.4), ha="center", va="bottom",
                            fontsize=7.4, color=INK, zorder=5)
                ax.annotate(f"$\\rightarrow${dst}", (x, 2.5), ha="center", va="bottom",
                            fontsize=6.1, color="white" if val > 18 else INK2,
                            rotation=90, zorder=6)
            m = 0.5 * (r["fwd"] + r["rev"])
            ax.plot([gi - 0.5 * W - W * 0.44, gi + 0.5 * W + W * 0.44], [m, m],
                    color=INK, lw=1.0, ls=(0, (1.6, 1.6)), zorder=6)
            ax.annotate(f"{m:.0f}", (gi + 0.5 * W + W * 0.50, m), ha="left", va="center",
                        fontsize=6.6, color=INK2, zorder=6)

        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{r['a']}\n$\\leftrightarrow$ {r['b']}" for r in rows],
                           fontsize=7.4)
        ax.set_xlim(-0.5 - 0.5 * W, len(rows) - 1 + 0.5 + 0.5 * W)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(FLABEL[f], fontsize=10, color=INK, pad=6)
        if f != FAMS[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("pairwise acc. (%)", fontsize=10)

    handles = [Patch(facecolor=BLUE, edgecolor=BLUE, alpha=0.85, label="one direction"),
               Patch(facecolor=ORANGE, edgecolor=ORANGE, alpha=0.85, label="its reverse"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (1.6, 1.6)),
                      label="mean of the two (50% = pure direction bias)"),
               Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance (50%)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.4,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.055))

    fig.subplots_adjust(left=0.070, right=0.995, top=0.795, bottom=0.135)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png")

    for f in FAMS:
        print(f"\n{FLABEL[f]}")
        print(f"  {'pair':<24}{'fwd':>8}{'rev':>8}{'mean':>8}{'|gap|':>8}   (n/dir)")
        for r in data[f]:
            m = 0.5 * (r["fwd"] + r["rev"])
            print(f"  {r['a']+' -> '+r['b']:<24}{r['fwd']:>8.1f}{r['rev']:>8.1f}"
                  f"{m:>8.1f}{abs(r['fwd']-r['rev']):>8.1f}   {r['n_fwd']}/{r['n_rev']}")
        # 가림 유무로 쪼갠 평균 — 편향이 어디서 커지는지
        for tag, cs in [("no occluder", ["static_visible", "moving_visible"]),
                        ("occluded", ["moving_occlusion", "static_occlusion"])]:
            rs = transitions(pairs, f, cs)
            mm = float(np.mean([0.5 * (r["fwd"] + r["rev"]) for r in rs]))
            gg = float(np.mean([abs(r["fwd"] - r["rev"]) for r in rs]))
            print(f"    {tag:<14} mean {mm:6.1f}   |gap| {gg:6.1f}")


if __name__ == "__main__":
    main()
