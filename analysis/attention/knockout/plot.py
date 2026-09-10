#!/usr/bin/env python3
"""knockout 그림 — results.json 하나에서 그린다.

  layers   층별 Δ정확도, 간선 종류별 선 (--layers each 로 돌린 결과)
  edges    간선별 Δ정확도를 조건으로 쪼갠 막대 (--layers all 로 돌린 결과)
  drift    같은 축의 예측 이동량 (pred_drift 를 켰을 때만)

  python -m analysis.attention.knockout.plot \
      --results z_research/IntPhysGenV11/exp_results/knockout__v11_vith/results.json \
      --outdir  z_research/IntPhysGenV11/figures/knockout

⚠️ Δ 는 `meta.delta_ref` 기준선 대비다 (기본 `clean_null` = 전부 True 인 마스크로 돈 판).
   전수 채점 73.37% 가 아니라 **그 표본의 기준선** 대비로 읽을 것.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
PANEL = [f"({c})" for c in "abcdefgh"]
FIGDIR = {"fig_knockout_acc": "", "fig_knockout_layers": "", "fig_knockout_edges": "",
          "fig_knockout_drift": ""}
STYLE = [(BLUE, "-", "o"), (ORANGE, "-", "s"), (GREEN, "-", "^"),
         (BLUE, "--", "v"), (ORANGE, "--", "D"), (GREEN, "--", "P")]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.tick_params(labelsize=8.0, colors=INK2, pad=2.0)


def save(fig, out: Path, name):
    d = out / FIGDIR.get(name, "")
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((d / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {d / name}.pdf + .png")


def edge_key(s):
    return f"{s['q']}->{s['k']}" + (f":{s['rel']}" if s["rel"] != "any" else "")


def style_map(R):
    """간선 -> (색, 선종류, 마커). **결과 전체에서 한 번만 정한다.**

    그림마다 따로 정하면 같은 간선이 그림마다 다른 색이 된다 (선 그림 파랑 = 막대 그림
    초록이었다). 두 그림을 나란히 읽으려면 배정이 하나여야 한다.
    """
    keys = sorted({edge_key(s) for s in R["specs"]})
    return {k: STYLE[i % len(STYLE)] for i, k in enumerate(keys)}


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def fig_layers(R, out, width=7.0):
    """층별 Δ정확도 — 어느 층에서 그 경로를 끊는 것이 아픈가."""
    single = [s for s in R["specs"] if len(s["layers"]) == 1]
    if not single:
        print("  [skip] fig_knockout_layers — 단일 층 명세가 없다 (--layers each 로 돌릴 것)")
        return
    by = {}
    for s in single:
        by.setdefault(edge_key(s), {})[s["layers"][0]] = _get(s, "delta", "surprise_acc", "acc")
    L = R["meta"]["n_layers"]
    ST = style_map(R)
    fig, ax = plt.subplots(figsize=(width, width * 0.44))
    frame(ax)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    for i, (k, d) in enumerate(sorted(by.items())):
        c, ls, mk = ST[k]
        x = sorted(d)
        ax.plot(x, [d[j] for j in x], ls, marker=mk, ms=3.8, lw=1.3, color=c,
                mec="white", mew=0.7, zorder=3 + i, label=k)
    ax.set_xticks(range(L))
    ax.set_xlabel("predictor layer where the edge is cut", fontsize=8.5, color=INK2)
    ax.set_ylabel("Δ pairwise accuracy (pp)", fontsize=8.5, color=INK2)
    base = _get(R, "clean_null", "surprise_acc", "acc") or _get(R, "clean", "surprise_acc", "acc")
    ax.set_title(f"baseline {base:.1f}%  ·  n_pair {_get(R,'clean','surprise_acc','n_pair')}"
                 f"  ·  chance 50", fontsize=8.5, color=INK, pad=4)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    save(fig, out, "fig_knockout_layers")


def fig_acc(R, out, width=7.4):
    """information flow — y = 채점 정확도, x = 간선을 끊은 층.

    Δ 판(`fig_knockout_layers`)과 달리 **절대 정확도**라 chance 50 과 기준선이 같이 보인다.
    오른쪽 칸은 그 간선을 **전 층 동시에** 끊은 값이다 (`@all`).
    """
    ST = style_map(R)
    L = R["meta"]["n_layers"]
    nom = R["meta"].get("window_nominal") or {}
    # 연속 구간이면 전부 그린다. x = 구간의 중심, 계열 = (간선, 명목 창 폭).
    # 창은 양 끝에서 잘리므로 실제 층수 대신 meta 의 명목 폭으로 묶어야 선이 안 끊긴다.
    by, allv = {}, {}
    for s in R["specs"]:
        lay = s["layers"]
        if len(lay) >= L or lay != list(range(lay[0], lay[-1] + 1)):
            continue
        for w in nom.get(s["name"], [len(lay)]):
            by.setdefault((edge_key(s), w), {})[(lay[0] + lay[-1]) / 2] = \
                _get(s, "metrics", "surprise_acc", "acc")
    if not by:
        print("  [skip] fig_knockout_acc — 연속 구간 명세가 없다 (--layers each|window 로 돌릴 것)")
        return
    for s in R["specs"]:
        if len(s["layers"]) == L:
            allv[edge_key(s)] = _get(s, "metrics", "surprise_acc", "acc")
    ws = sorted({w for _, w in by})
    WLS = ["-", "--", ":", "-."]
    base = _get(R, "clean_null", "surprise_acc", "acc") or _get(R, "clean", "surprise_acc", "acc")

    fig, ax = plt.subplots(figsize=(width, width * 0.46))
    frame(ax)
    ax.axhline(base, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.axhline(50, color=INK2, lw=0.9, ls=":", zorder=2)
    # mask->ctx 를 전 층에서 끊으면 mask 토큰에 영상 정보가 한 번도 안 들어온다.
    # 그러면 예측이 클립과 무관한 **상수**가 된다 (실측: 서로 다른 클립의 p 가 비트 단위 동일).
    # 그래서 이 값은 눈금이 아니라 **바닥선**이다 — 기준선으로 그려야 나머지가 읽힌다.
    floor = allv.get("mask->ctx")
    if floor is not None:
        ax.axhline(floor, color=ORANGE, lw=0.9, ls="-.", zorder=2)
    xa = L + 0.9                                   # @all 을 놓을 자리
    ax.axvline(L - 0.45, color=MUTED, lw=0.6, zorder=1)
    seen_all = set()
    for (k, w) in sorted(by):
        c, ls, mk = ST[k]
        if len(ws) > 1:                      # 폭이 여러 개면 선종류로 폭을 가른다
            ls = WLS[ws.index(w) % len(WLS)]
        lab = k if len(ws) == 1 else f"{k}  w={w}"
        d = by[(k, w)]
        x = sorted(d)
        ax.plot(x, [d[j] for j in x], ls, marker=mk, ms=4.0, lw=1.4, color=c,
                mec="white", mew=0.7, zorder=4, label=lab)
        if k in allv and k not in seen_all:
            seen_all.add(k)
            ax.plot([xa], [allv[k]], marker=mk, ms=6.0, color=c, mec="white", mew=0.9,
                    zorder=5, ls="none")
    ax.set_xticks(list(range(L)) + [xa])
    ax.set_xticklabels([str(i) for i in range(L)] + ["all"], fontsize=8)
    ax.set_xlim(-0.7, xa + 0.7)
    ax.set_xlabel("predictor layer (window centre) where the edge is cut",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel("pairwise accuracy (%)", fontsize=8.5, color=INK2)
    ax.text(-0.55, base + 0.9, f"no knockout  {base:.1f}", fontsize=7.2, color=MUTED, va="bottom")
    ax.text(-0.55, 50 + 0.9, "chance 50", fontsize=7.2, color=INK2, va="bottom")
    if floor is not None:
        ax.text(-0.55, floor + 0.9, f"context blocked \u2192 constant prediction  {floor:.1f}",
                fontsize=7.2, color=ORANGE, va="bottom")
    ax.set_title(f"{R['meta']['dataset']} · {R['meta']['arch']} · "
                 f"{R['meta']['n_block']} blocks / {_get(R,'clean','surprise_acc','n_pair')} pairs",
                 fontsize=8.5, color=INK, pad=4)
    # 바닥선과 chance 사이 빈 띠에 범례를 놓는다 (lower left 는 바닥선 라벨과 겹친다)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="center left",
              bbox_to_anchor=(0.01, 0.24))
    save(fig, out, "fig_knockout_acc")


def fig_edges(R, out, width=7.6):
    """전 층 동시에 끊었을 때 — 간선별 Δ 를 조건으로 쪼갠다."""
    L = R["meta"]["n_layers"]
    full = [s for s in R["specs"] if len(s["layers"]) == L]
    if not full:
        print("  [skip] fig_knockout_edges — 전 층 명세가 없다 (--layers all 로 돌릴 것)")
        return
    conds = sorted(_get(R, "clean", "surprise_acc", "by_block_type", default={}))
    if not conds:
        print("  [skip] fig_knockout_edges — by_block_type 이 없다 (scoring.breakdown 확인)")
        return
    ST = style_map(R)
    full = sorted(full, key=edge_key)          # 선 그림과 같은 순서로
    fig, ax = plt.subplots(figsize=(width, width * 0.36))
    frame(ax)
    ax.axhline(0, color=INK2, lw=0.8, zorder=4)
    W = 0.8 / len(full)
    x = np.arange(len(conds))
    for i, s in enumerate(full):
        v = [_get(s, "delta", "surprise_acc", "by_block_type", c, default=np.nan) for c in conds]
        k = edge_key(s)
        c, ls, _ = ST[k]
        f = 0.45 if ls == "--" else 0.0        # 선 그림의 파선 = 막대의 옅은 채움
        ax.bar(x + (i - (len(full) - 1) / 2) * W, v, width=W * 0.9,
               color=c if f == 0 else _tint(c, f), edgecolor=c, lw=0.6, zorder=3, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=7)
    ax.set_ylabel("Δ pairwise accuracy (pp)", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=min(3, len(full)), loc="lower left")
    save(fig, out, "fig_knockout_edges")


def _tint(h, f):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


def fig_drift(R, out, width=7.0):
    """예측이 얼마나 움직였나 — 정확도가 안 변해도 여기서는 보인다."""
    single = [s for s in R["specs"] if len(s["layers"]) == 1
              and _get(s, "metrics", "pred_drift") is not None]
    if not single:
        print("  [skip] fig_knockout_drift — pred_drift 가 없다 (--metrics 에 넣을 것)")
        return
    by = {}
    for s in single:
        by.setdefault(edge_key(s), {})[s["layers"][0]] = (
            _get(s, "metrics", "pred_drift", "rel_l1"),
            _get(s, "metrics", "pred_drift", "cosine"))
    L = R["meta"]["n_layers"]
    ST = style_map(R)
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.36))
    for pi, (lab, j) in enumerate([("relative L1 drift  |p - p_clean| / |p_clean|", 0),
                                   ("cosine(p, p_clean)", 1)]):
        ax = axes[pi]
        frame(ax)
        for i, (k, d) in enumerate(sorted(by.items())):
            c, ls, mk = ST[k]
            x = sorted(d)
            ax.plot(x, [d[t][j] for t in x], ls, marker=mk, ms=3.4, lw=1.2, color=c,
                    mec="white", mew=0.6, zorder=3 + i, label=k)
        ax.set_xticks(range(0, L, 2))
        ax.set_xlabel("layer where the edge is cut", fontsize=8.5, color=INK2)
        ax.set_title(lab, fontsize=8.5, color=INK, pad=4)
        ax.text(0.5, -0.32, PANEL[pi], transform=ax.transAxes, ha="center",
                fontsize=9, color=INK)
    h = [Line2D([], [], color=ST[k][0], ls=ST[k][1], marker=ST[k][2], ms=3.4, lw=1.2,
                mec="white", label=k) for k in sorted(by)]
    fig.legend(handles=h, loc="lower center", ncol=min(3, len(h)), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.26))
    save(fig, out, "fig_knockout_drift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--which", nargs="*", default=["acc", "layers", "edges", "drift"])
    a = ap.parse_args()
    R = json.loads(a.results.read_text())
    v = R.get("verify", {})
    assert v.get("resume_vs_forward_max_abs_diff", 1) == 0.0, \
        "resume 가 실제 forward 를 재현하지 못한 결과다 — 그리지 않는다"
    print(f"  검증 OK — resume vs forward {v['resume_vs_forward_max_abs_diff']:.1e} | "
          f"null-mask {v['null_mask_vs_forward_max_abs_diff']:.2e} | "
          f"Δ 기준선 {R['meta'].get('delta_ref')} | 명세 {len(R['specs'])}개")
    for w in a.which:
        {"acc": fig_acc, "layers": fig_layers, "edges": fig_edges,
         "drift": fig_drift}[w](R, a.outdir)


if __name__ == "__main__":
    main()
