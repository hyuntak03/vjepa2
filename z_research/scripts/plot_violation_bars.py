#!/usr/bin/env python3
"""Figure 1 — violation type x (motion x occlusion) pairwise accuracy.

summary.json 의 by_block_type 은 condition 4행만 준다. violation type 까지 쪼개려면
per_block.json 의 per_video_surprise 를 index 와 조인해 쌍 단위로 다시 채점해야 한다.
이 스크립트가 그걸 하고, 재계산한 overall 이 summary.json 과 맞는지 스스로 검증한다.

  python z_research/scripts/plot_violation_bars.py \
      --result-dir z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith \
      --index data_csv/intphysgen_v10/index.csv \
      --output z_research/IntPhysGenV10/figures/fig1_violation_bars.pdf
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

# 데이터에 실제로 있는 조건만 이 순서대로 쓴다 (v10 은 4개, flat 팔이 붙으면 6개).
# ramp/flat 짝을 나란히 둬서 그 대비가 바로 읽히게 한다.
COND_ORDER = ["static_visible",
              "moving_visible", "moving_visible_flat",
              "moving_occlusion", "moving_occlusion_flat",
              "static_occlusion"]
TITLE = {"static_visible": "Static, no occluder", "moving_visible": "Moving, no occluder",
         "moving_occlusion": "Moving, occluded", "static_occlusion": "Static, occluded",
         "moving_visible_flat": "Moving (flat), no occluder",
         "moving_occlusion_flat": "Moving (flat), occluded"}
# index 의 violation_type 값 -> 논문 표기. 순서가 곧 그림의 그룹 순서다.
VIOL = ["vanish", "shape", "color"]
VLABEL = {"vanish": "Object permanence",
          "shape": "Shape consistency",
          "color": "Color consistency"}

# 검증된 팔레트는 slot 1-3 뿐인데 조건은 4개다. 그래서 4색을 임의로 쓰지 않고
# 2x2 설계를 그대로 인코딩한다 — 색 = 가림 유무, 명암 = 정지/이동.
# 모든 막대에 값을 직접 달아 명암만으로 구분할 필요가 없게 한다.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"


def tint(hex_color: str, f: float) -> str:
    """흰색과 섞어 밝은 색조를 만든다 (alpha 를 쓰면 격자선이 비쳐 탁해진다)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


# 파스텔 톤. 면은 밝게 깔되 같은 색상의 진한 테두리를 둘러 흰 배경에서 형태가 죽지 않게 한다.
# 모든 막대에 값을 직접 달기 때문에 면 대비가 낮아도 읽는 데 지장이 없다.
# 기존 4조건은 손대지 않는다 (색 = 가림, 명암 = 정지/이동). fig2·fig3 과 뜻이 같아야 한다.
# flat 팔만 **세 번째 색(초록)** 으로 따로 뺀다 — 빗금은 너무 헷갈렸다.
# 초록 안에서는 이동밖에 없으므로 명암이 가림 유무를 맡는다 (연함 = 안 가림).
# ⚠️ 초록은 밝은 배경에서 대비가 3:1 미만이라 **모든 막대에 값을 직접 단다** (§8-3).
FILL_LIGHT, FILL_DARK, EDGE_D = 0.46, 0.16, 0.28
BAR = {"static_visible":        (BLUE,   FILL_DARK,  "Static + visible"),
       "moving_visible":        (BLUE,   FILL_LIGHT, "Moving + visible (ramp)"),
       "moving_visible_flat":   (GREEN,  FILL_LIGHT, "Moving + visible (flat)"),
       "moving_occlusion":      (ORANGE, FILL_LIGHT, "Moving + occluded (ramp)"),
       "moving_occlusion_flat": (GREEN,  FILL_DARK,  "Moving + occluded (flat)"),
       "static_occlusion":      (ORANGE, FILL_DARK,  "Static + occluded")}

# --flat-style hatch 로 쓸 대안: 색·명암은 4조건 규칙 그대로 두고 지면만 빗금.
# 축 3개를 색 2개로 다 담아 처음엔 이걸 썼는데 읽기 어렵다는 평가가 있었다.
BAR_HATCH = {"static_visible":        (BLUE,   FILL_DARK,  "",   "Static + visible"),
             "moving_visible":        (BLUE,   FILL_LIGHT, "//", "Moving + visible (ramp)"),
             "moving_visible_flat":   (BLUE,   FILL_LIGHT, "",   "Moving + visible (flat)"),
             "moving_occlusion":      (ORANGE, FILL_LIGHT, "//", "Moving + occluded (ramp)"),
             "moving_occlusion_flat": (ORANGE, FILL_LIGHT, "",   "Moving + occluded (flat)"),
             "static_occlusion":      (ORANGE, FILL_DARK,  "",   "Static + occluded")}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def score(result_dir: Path, index_csv: Path):
    """쌍 단위로 다시 채점한다. matched pair = (block_id, pair_id) 안의 가능/불가능 2개."""
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
        sp, si = S[pos[0]["video_id"]], S[imp[0]["video_id"]]
        out.append((pos[0]["condition"], imp[0]["violation_type"],
                    1.0 if si > sp else (0.5 if si == sp else 0.0)))

    # 자체 검증: 재계산 overall 이 summary.json 과 같아야 한다
    got = float(np.mean([a for _, _, a in out]))
    want = json.loads((result_dir / "summary.json").read_text())["surprise"]["overall"]["block_pairwise"]
    if abs(got - want) > 1e-9:
        raise ValueError(f"재계산 {got:.6f} != summary.json {want:.6f}")
    print(f"  검증 OK — {result_dir.name}: 재계산 {got*100:.4f}% == summary.json "
          f"{want*100:.4f}%  (n={len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    # ramp 팔(v10)과 flat 팔(v10_flat)은 **서로 다른 데이터셋**이라 summary.json 이
    # 따로 나온다. 한 그림에 6조건을 그리려면 여러 결과를 이어 붙여야 한다.
    # --result-dir / --index 를 같은 횟수만큼 반복해서 준다 (짝이 맞아야 한다).
    ap.add_argument("--result-dir", type=Path, required=True, action="append")
    ap.add_argument("--index", type=Path, required=True, action="append")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--flat-style", choices=["green", "hatch"], default="green",
                    help="flat 팔을 초록으로 뺄지(기본), 빗금으로 얹을지")
    ap.add_argument("--title", default=None, help="논문용은 캡션으로 빼는 게 관례라 기본은 없음")
    ap.add_argument("--subtitle", default=None)
    a = ap.parse_args()

    if len(a.result_dir) != len(a.index):
        raise SystemExit(f"--result-dir {len(a.result_dir)}개 vs --index {len(a.index)}개 — 짝이 안 맞는다")
    global BAR
    if a.flat_style == "hatch":
        BAR = BAR_HATCH
    pairs = []
    for rd, ix in zip(a.result_dir, a.index):
        pairs += score(rd, ix)          # 각 결과를 자기 summary.json 과 대조 검증한다
    present = {c for c, _v, _x in pairs}
    unknown = present - set(COND_ORDER)
    if unknown:
        raise SystemExit(f"BAR 에 스타일이 없는 condition: {sorted(unknown)}")
    CONDS = [c for c in COND_ORDER if c in present]
    acc, n = {}, {}
    for c in CONDS:
        for v in VIOL:
            s = [x for cc, vv, x in pairs if cc == c and vv == v]
            acc[(c, v)], n[(c, v)] = 100 * float(np.mean(s)), len(s)
    overall = 100 * float(np.mean([x for _, _, x in pairs]))

    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=9, colors=INK2, pad=2.0)
    ax.set_ylabel("pairwise acc. (\\%)".replace("\\", ""), fontsize=10)

    # 막대 4개가 한 그룹. 그룹 반폭 = 1.5W + 막대 반폭(0.45W) = 1.95W.
    # xlim 을 여기에 맞춰야 y축과 첫 막대 사이에 빈 띠가 안 생긴다.
    nb = len(CONDS)
    W, PAD = 0.84 / nb, 0.055                 # 그룹 폭을 조건 수로 나눈다
    HALF = (nb / 2 - 0.5 + 0.45) * W
    for gi, v in enumerate(VIOL):
        for bi, c in enumerate(CONDS):
            base, f, *rest = BAR[c]
            hatch = rest[0] if len(rest) == 2 else ""
            x = gi + (bi - (nb - 1) / 2) * W
            y = acc[(c, v)]
            ax.bar(x, y, width=W * 0.90, color=tint(base, f),
                   edgecolor=tint(base, max(0.0, f - EDGE_D)), lw=0.7, zorder=3,
                   hatch=hatch)
            ax.annotate(f"{y:.1f}", (x, y + 1.6), ha="center", va="bottom",
                        fontsize=7.8 if nb <= 4 else 6.4, color=INK, zorder=5,
                        rotation=0 if nb <= 4 else 90)
        if gi:                                   # 그룹 사이 옅은 구분선
            ax.axvline(gi - 0.5, color=MUTED, lw=0.5, alpha=0.35, zorder=1)

    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)

    ax.set_xticks(range(len(VIOL)))
    ax.set_xticklabels([VLABEL[v] for v in VIOL], fontsize=10.5)
    ax.set_xlim(-HALF - PAD, len(VIOL) - 1 + HALF + PAD)
    ax.tick_params(axis="x", length=0, pad=4)

    # chance 는 범례로 넣는다 — 축 안에 글씨를 두면 그만큼 오른쪽 여백을 비워야 한다
    handles = [Patch(facecolor=tint(BAR[c][0], BAR[c][1]),
                     edgecolor=tint(BAR[c][0], max(0.0, BAR[c][1] - EDGE_D)), lw=0.7,
                     hatch=(BAR[c][2] if len(BAR[c]) == 4 else ""),
                     label=BAR[c][-1]) for c in CONDS]
    handles.append(Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)),
                          label="chance (50%)"))
    ncol, yanc = (5, 1.015) if nb <= 4 else (4, 1.075)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               fontsize=9 if nb <= 4 else 8.0,
               handlelength=1.5, handletextpad=0.5, columnspacing=1.35,
               bbox_to_anchor=(0.5, yanc))

    top = 0.885
    if a.title:
        fig.suptitle(a.title, fontsize=11.5, y=1.10, x=0.02, ha="left", color=INK)
    fig.subplots_adjust(left=0.068, right=0.998, top=top, bottom=0.115)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png")

    print(f"\n{'condition':<24}" + "".join(f"{VLABEL[v]:>10}" for v in VIOL) + f"{'all':>10}")
    for c in CONDS:
        s = [x for cc, _, x in pairs if cc == c]
        print(f"{c:<24}" + "".join(f"{acc[(c,v)]:>10.2f}" for v in VIOL)
              + f"{100*float(np.mean(s)):>10.2f}")
    print(f"{'all':<24}" + "".join(
        f"{100*float(np.mean([x for _,vv,x in pairs if vv==v])):>10.2f}" for v in VIOL)
        + f"{overall:>10.2f}")
    print(f"\nn per cell: {n[(CONDS[0], VIOL[0])]} (vanish) / {n[(CONDS[0], VIOL[1])]} (others)"
          f" | total {len(pairs)} matched pairs")


if __name__ == "__main__":
    main()
