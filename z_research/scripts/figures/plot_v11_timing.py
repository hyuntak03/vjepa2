#!/usr/bin/env python3
"""v11_full (12조건) 그림 — 채점 · probing · 기전.

입력 세 개를 모두 `exp_results/` 에서 읽는다:
  report.json       채점 cell (condition x violation x k x occ_timing) + probing cell
  attn_probe__v11_full_vith/summary.json      108 head (self 는 per_group[fit] 이다!)
  attn_probe__v11_full_vith/predictions.json  clip 별 예측 -> k 로 사후 분해
  surprise_c16t32__v11_full_vith/per_block.json  surprise 원값 -> margin 분해

⚠️ **self 정확도는 `evals[fit]["per_group"][fit_group]`** 이다.
   `evals[fit]["overall"]` 은 12조건 평균이라 이식 실패가 섞인다.

── 색 규칙 (2026-09-02) ────────────────────────────────────────────────────────
**한 그림에 범주형 색상 척도는 하나만 쓴다.** 전에는 z/h/p 를 파랑·초록·주황으로,
같은 그림 범례에서 early/mid/late 도 파랑·초록·주황으로 써서 못 읽었다.

  범주형 (파랑 #2a78d6 / 주황 #eb6834 / 초록 #1baf7a)
      violation / point(z,h,p) / motion  중 **한 그림에 하나만** 선으로 쓴다
  순서형 (회색 명도 램프)
      occlusion timing (early < mid < late) — 순서가 있으므로 명도로 준다
  x 축
      k, timing 처럼 순서가 있는 축은 가능하면 축으로 뺀다

  python z_research/scripts/figures/plot_v11_timing.py \
    --exp z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results \
    --index data_csv/intphysgen_v11_full/index_probe.csv \
    --outdir z_research/IntPhysGenV11_occlusion_timing_ablation/figures
"""
from __future__ import annotations
import argparse, collections, colorsys, csv, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, BAND = "#000000", "#3b3b3b", "#9a9a9a", "#f2f2f2"
PANEL = [f"({c})" for c in "abcdefghi"]

VIOL = ["vanish", "shape", "color"]
VLAB = {"vanish": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
VCOL = {"vanish": BLUE, "shape": ORANGE, "color": GREEN}
MOT = ["static", "flat", "ramp"]
MLAB = {"static": "Static", "flat": "Moving (flat)", "ramp": "Moving (ramp)"}
MCOL = {"static": BLUE, "flat": GREEN, "ramp": ORANGE}
PTS = ["z", "h", "p"]
PLAB = {"z": "z  context encoder", "h": "h  target encoder", "p": "p  predictor"}
PCOL = {"z": BLUE, "h": GREEN, "p": ORANGE}
TIM = ["vis", "early", "mid", "late"]
TLAB = {"vis": "no\noccluder", "early": "early", "mid": "mid", "late": "late\n(v11)"}
# timing 도 **서로 다른 색**을 쓴다. 명도만 다른 램프는 범례에서 구분이 안 됐다.
# 대신 timing 이 선인 그림에서는 **다른 어떤 것도 범주형 팔레트를 쓰지 않게** 한다
# (패널 제목·축 라벨을 중립색으로 돌린다). 그래야 한 색이 한 가지만 뜻한다.
TRAMP = {"early": BLUE, "mid": GREEN, "late": ORANGE}
TGT = {"shape": "Shape", "color": "Colour", "env": "Background"}
PT = {"contextF__f1to16": "z", "pred__f17to32": "p", "targetF__f17to32": "h"}
ARM = {"static": ["static_visible", "static_occlusion_early", "static_occlusion_mid",
                  "static_occlusion"],
       "flat":   ["moving_visible_flat", "moving_occlusion_flat_early",
                  "moving_occlusion_flat_mid", "moving_occlusion_flat"],
       "ramp":   ["moving_visible", "moving_occlusion_early", "moving_occlusion_mid",
                  "moving_occlusion"]}
KS = ["1", "2", "3", "4"]

FIGDIR = {
    # ── 규칙 ──────────────────────────────────────────────────────────────────
    # k 를 축으로 쪼갠 그림은 **반드시 `by_k/`** 에 둔다. 그 폴더의 그림은 전부
    #   x = k (0 = 가림막 없음),  선 = 타이밍,  나머지는 전부 고정
    # 이라 한 폴더 안에서 읽는 법이 같다.
    "fig_timing_step":     "01_scoring",
    "fig_timing_k":        "01_scoring/by_k",
    # 02_probing 은 **어디서 어디로 readout 을 옮기는가**로 셋으로 나눈다.
    "fig_self_shape":      "02_probing/01_self",
    "fig_self_color":      "02_probing/01_self",
    "fig_self_k_shape":    "02_probing/01_self/by_k",
    "fig_self_k_color":    "02_probing/01_self/by_k",
    # 비가림 -> 가림
    "fig_vis_occ_shape":   "02_probing/02_vis_occ",
    "fig_vis_occ_color":   "02_probing/02_vis_occ",
    "fig_vis_occ_k_shape": "02_probing/02_vis_occ/by_k",
    "fig_vis_occ_k_color": "02_probing/02_vis_occ/by_k",
    # GT 미래(h) <-> 예측 미래(p) 그림은 2026-09-03 에 뺐다. h→h / p→p 는 전부 100 이라
    # 정보가 없고, h→p 의 late 붕괴는 vis_occ 와 겹친다. 건질 것은 "h→p 가 비가림에서
    # 이미 57~95" 하나뿐이라 03_mechanism 설명에 숫자로 넣었다. 수치는 README 표 8.
    "fig_margin": "03_mechanism", "fig_margin_vs_acc": "03_mechanism",
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
LGF = 8.5          # 범례 글자 (전에 7.0 이라 안 읽혔다)


def _pastel(h, lift=0.26, sat=1.06):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    return "#%02x%02x%02x" % tuple(
        round(255 * c) for c in colorsys.hls_to_rgb(hh, ll + (1 - ll) * lift, min(1.0, ss * sat)))


PASTEL = {c: _pastel(c) for c in (BLUE, ORANGE, GREEN)}


def frame(ax, ylim=(40, 108), yt=(50, 75, 100), chance=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    if chance is not None:
        ax.axhline(chance, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.set_ylim(*ylim); ax.set_yticks(yt); ax.tick_params(labelsize=7.4, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_zorder(30)


def save(fig, out: Path, name):
    d = out / FIGDIR.get(name, "")
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((d / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {FIGDIR.get(name,'')}/{name}")
    plt.close(fig)


def panel_labels(fig, axes, pad=0.018):
    """패널 라벨은 **눈금 라벨과 x 축 제목 아래**에 둔다.

    전에는 축 바닥에서 고정 오프셋만큼 내렸는데, x 축 제목이 있는 그림에서
    라벨이 제목 위에 겹쳐 찍혔다 ("k (0 = n(a)ccluder)"). 여기서는 그 패널의
    눈금 라벨·x 제목의 실제 bbox 를 재서 그 아래에 놓는다.
    """
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(np.ravel(axes), PANEL):
        bb = ax.get_position(); bot = bb.y0
        parts = list(ax.get_xticklabels())
        if ax.get_xlabel():
            parts.append(ax.xaxis.label)
        for t in parts:
            try:
                bot = min(bot, t.get_window_extent(rend).transformed(inv).y0)
            except Exception:
                pass
        fig.text(bb.x0 + bb.width / 2, bot - pad, lab, ha="center", va="top",
                 fontsize=9.0, color=INK)


def lines(ax, ys, colors, labels=None, fmt="%.0f", gap=None, fs=6.4):
    """여러 계열을 그리고 **같은 x 의 값 라벨을 세로로 쌓는다**.

    전에는 계열마다 x 를 조금씩 밀어 라벨을 떨어뜨렸는데, 값이 거의 같으면
    (예: 100/100/100, 0.08/0.08) 그래도 겹쳤다. 여기서는 x 는 그대로 두고
    **낮은 값부터 위로 최소 gap 씩 밀어** 올린다 -> 절대 안 겹친다.
    """
    for y, c, lb in zip(ys, colors, labels or [None] * len(ys)):
        ax.plot(range(len(y)), y, marker="o", ms=5.2, lw=1.9, color=_pastel(c),
                mec="white", mew=0.8, label=lb, zorder=4)
    lo, hi = ax.get_ylim()
    g = gap if gap is not None else (hi - lo) * 0.042
    for xi in range(len(ys[0])):
        col = [y[xi] for y in ys]
        _stack(ax, xi, col, colors, g, fmt, fs)


def _stack(ax, x, col, colors, g, fmt, fs):
    """같은 x 의 라벨을 아래에서 위로 쌓되, **어떤 계열의 마커 위에도 안 얹히게** 민다.

    쌓기만 하면 낮은 값의 라벨이 바로 위 계열의 마커에 닿는다 (0.27 라벨이 0.54 마커에
    걸렸다). 글자는 anchor 에서 위로 자라므로 그 높이까지 감안해 비켜 준다.
    """
    pts = [v for v in col if np.isfinite(v)]
    prev = -1e18
    for i in np.argsort(col):
        if not np.isfinite(col[i]):
            continue
        yy = max(col[i] + g * 0.45, prev + g)
        for _ in range(40):                       # 다른 마커와 겹치면 위로 민다
            if any(-g * 0.30 < v - yy < g * 0.95 for v in pts if abs(v - col[i]) > 1e-12):
                yy += g * 0.28
            else:
                break
        ax.text(x, yy, fmt % col[i], ha="center", va="bottom", fontsize=fs,
                color=colors[i], zorder=6)
        prev = yy


# ═══ 데이터 ════════════════════════════════════════════════════════════════════
def mot_of(c): return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def _margins(S, index: Path):
    meta = {r["video_id"]: r for r in csv.DictReader(index.open())}
    blk = collections.defaultdict(dict)
    for v, s in S.items():
        blk[meta[v]["source_block"]][meta[v]["variant"]] = (s, meta[v])
    mg = collections.defaultdict(list)
    for d in blk.values():
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if pos in d and imp in d:
                sp, mp = d[pos]; si, _ = d[imp]
                mg[(mp.get("occ_timing") or "vis", mp["violation_type"])].append((sp, si))
    return mg


def find_run(exp: Path, prefix: str):
    """`<prefix>__<데이터셋>_<모델>` 디렉토리를 찾는다 — 데이터셋 이름을 박아 두지 않는다."""
    c = sorted(d for d in exp.glob(f"{prefix}__*") if d.is_dir())
    return c[0] if c else None


def load(exp: Path, index: Path):
    R = json.loads((exp / "report.json").read_text())
    cells = [dict(c, _m=mot_of(c["condition"]), _t=(c.get("occ_timing") or "vis"))
             for c in R["scoring"]["cells"]]
    ov = R["scoring"]["overall"][0]
    tot = sum(c["n"] for c in cells)
    wa = sum(c["acc"] * c["n"] for c in cells) / tot
    assert ov["verified"] and abs(wa - ov["overall"]) < 0.02, "report 검증 실패"
    print(f"  검증 OK — {ov['dataset']} {ov['overall']:.2f}%  n={ov['n_pair']} "
          f"(cells 재합산 {wa:.2f}%)")

    # ⚠️ probing 수치는 **전부 attn_probe_xfer 한 run**에서 읽는다.
    #    그 run 이 상위집합이다 (z/h/p self + h<->p 양방향). 두 run 을 섞으면 self 값이
    #    미묘하게 달라진다 — 실측 108칸 중 1칸이 6.1pt 벌어졌다 (미수렴 head).
    pb = find_run(exp, "attn_probe_xfer")
    if pb is None:                       # 채점만 돌린 데이터셋 (probing 미실행)
        S0 = json.loads((find_run(exp, "surprise_c16t32") / "per_block.json").read_text())
        return cells, None, None, None, _margins(S0["per_video_surprise"], index)
    P = json.loads((pb / "summary.json").read_text())["probing"]
    T, ch = {}, {}
    for r in P:
        ev = r["evals"][r["fit"]]; p = PT[r["fit"]]
        ch[r["target"]] = 100 * r["chance"]
        for g, c in ev["per_group"].items():
            T[(r["target"], p, r["groups"][0], g)] = 100 * c["acc"]

    M = json.loads((pb / "predictions.json").read_text())
    meta = {r["video_id"]: r for r in csv.DictReader(index.open())}
    vid = M["val_video_ids"]
    cond = np.array([meta[v]["condition"] for v in vid])
    kk = np.array([meta[v]["sym_k"] for v in vid])
    gold = {t: np.asarray(M["targets"][t]["gold"]) for t in M["targets"]}
    H = {(h["target"], PT[h["fit"]], h["groups"][0]): np.asarray(h["pred"])
         for h in M["heads"] if h["eval"] == h["fit"]}

    S = json.loads((find_run(exp, "surprise_c16t32") / "per_block.json").read_text())
    S = S["per_video_surprise"]
    blk = collections.defaultdict(dict)
    for v, s in S.items():
        m = meta[v]; blk[m["source_block"]][m["variant"]] = (s, m)
    mg = collections.defaultdict(list)
    for d in blk.values():
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if pos in d and imp in d:
                sp, mp = d[pos]; si, _ = d[imp]
                mg[(mp.get("occ_timing") or "vis", mp["violation_type"])].append((sp, si))
    return cells, T, ch, (H, gold, cond, kk), mg


def agg(cells, *keys):
    d = collections.defaultdict(lambda: [0.0, 0])
    for c in cells:
        k = tuple(c[x] for x in keys); d[k][0] += c["acc"] * c["n"]; d[k][1] += c["n"]
    return {k: v[0] / v[1] for k, v in d.items()}


def acc_of(pred, gold, m):
    return 100.0 * float((pred[m] == gold[m]).mean()) if m.sum() else float("nan")


# ═══ 01 채점 ═══════════════════════════════════════════════════════════════════
def fig_timing_step(cells, out):
    """주장 1 — 계단은 위반마다 다르다.  패널 = 위반, 선 = 운동, x = 타이밍."""
    A = agg(cells, "_m", "violation_type", "_t")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ci, (ax, v) in enumerate(zip(axes, VIOL)):
        ax.axvspan(2.5, 3.5, color=BAND, zorder=0)
        frame(ax, ylim=(40, 122), chance=50)
        lines(ax, [[A[(m, v, t)] for t in TIM] for m in MOT], [MCOL[m] for m in MOT],
              [MLAB[m] if ci == 0 else None for m in MOT])
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4))
        ax.set_xticklabels([TLAB[t] for t in TIM], fontsize=7.2, color=INK)
        ax.set_title(VLAB[v], fontsize=9.2, color=VCOL[v], pad=4)
        if ci == 0:
            ax.set_ylabel("pairwise accuracy (%)", fontsize=8.6, color=INK)
    axes[0].legend(frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.70, 1.13), handlelength=1.3, columnspacing=1.8)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.275, wspace=0.10)
    panel_labels(fig, axes)
    save(fig, out, "fig_timing_step")


def fig_timing_k(cells, out):
    """주장 2 — 가림 양(k)은 거의 무관하다.  선 = 타이밍(명도 램프)."""
    A = agg(cells, "sym_k", "violation_type", "_t"); V0 = agg(cells, "violation_type", "_t")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ci, (ax, v) in enumerate(zip(axes, VIOL)):
        y0 = V0[(v, "vis")]
        frame(ax, ylim=(40, 122), chance=50)
        TT = ["early", "mid", "late"]
        lines(ax, [[y0] + [A[(k, v, t)] for k in KS] for t in TT], [TRAMP[t] for t in TT],
              [t if ci == 0 else None for t in TT])
        ax.set_xlim(-0.45, 4.45); ax.set_xticks(range(5))
        ax.set_xticklabels(["0"] + KS, fontsize=7.2)
        ax.set_xlabel("k   (0 = no occluder)", fontsize=8.2, color=INK)
        ax.set_title(VLAB[v], fontsize=9.2, color=INK, pad=4)   # 선이 색을 쓰므로 제목은 중립
        if ci == 0:
            ax.set_ylabel("pairwise accuracy (%)", fontsize=8.6, color=INK)
    axes[0].legend(frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.70, 1.13), handlelength=1.3, columnspacing=1.8)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.285, wspace=0.10)
    panel_labels(fig, axes)
    save(fig, out, "fig_timing_k")


# ═══ 02 probing ════════════════════════════════════════════════════════════════
def _probe_fig(T, ch, out, tgt, mode):
    """mode 'self'  자기 조건 head        mode 'vis'  비가림 head 를 이식."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ci, (ax, m) in enumerate(zip(axes, MOT)):
        gs = ARM[m]; ax.axvspan(2.5, 3.5, color=BAND, zorder=0)
        frame(ax, ylim=(0, 128), yt=(0, 25, 50, 75, 100), chance=ch[tgt])
        Y = [[T[(tgt, p, g, g)] for g in gs] if mode == "self"
             else [T[(tgt, p, gs[0], g)] for g in gs] for p in PTS]
        lines(ax, Y, [PCOL[p] for p in PTS], [PLAB[p] if ci == 0 else None for p in PTS])
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4))
        ax.set_xticklabels([TLAB[t] for t in TIM], fontsize=7.2, color=INK)
        ax.set_title(MLAB[m], fontsize=9.2, color=INK, pad=4)
        if ci == 0:
            ax.set_ylabel(("self probe" if mode == "self" else "readout from  no occluder")
                          + "\naccuracy (%)", fontsize=8.4, color=INK)
    axes[0].legend(frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.70, 1.13), handlelength=1.3, columnspacing=1.6)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.845, bottom=0.275, wspace=0.10)
    panel_labels(fig, axes)
    save(fig, out, f"fig_self_{tgt}" if mode == "self" else f"fig_vis_occ_{tgt}")


def _by_k(getpred, ch, out, tgt, name, ylab):
    """`by_k/` 공용 판 — x = k(0 = 가림막 없음), 선 = 타이밍, 패널 = 운동.

    `getpred(arm)` 가 (예측배열, 그 팔의 조건 4개) 를 준다. 나머지는 전부 고정이라
    한 폴더 안의 그림들이 같은 방식으로 읽힌다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ci, (ax, m) in enumerate(zip(axes, MOT)):
        pr, g, gs, cond, kk = getpred(m)
        y0 = acc_of(pr, g, cond == gs[0])
        frame(ax, ylim=(0, 128), yt=(0, 25, 50, 75, 100), chance=ch[tgt])
        TT = ["early", "mid", "late"]
        lines(ax, [[y0] + [acc_of(pr, g, (cond == ec) & (kk == k)) for k in KS]
                   for ec in gs[1:]], [TRAMP[t] for t in TT],
              [t if ci == 0 else None for t in TT])
        ax.set_xlim(-0.45, 4.45); ax.set_xticks(range(5))
        ax.set_xticklabels(["0"] + KS, fontsize=7.2)
        ax.set_xlabel("k   (0 = no occluder)", fontsize=8.2, color=INK)
        ax.set_title(MLAB[m], fontsize=9.2, color=INK, pad=4)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=8.4, color=INK)
    axes[0].legend(frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.70, 1.13), handlelength=1.3, columnspacing=1.8)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.845, bottom=0.285, wspace=0.10)
    panel_labels(fig, axes)
    save(fig, out, name)


def fig_vis_occ_k(pred_pack, ch, out, tgt):
    """비가림 readout 을 k 별 가림 조건에 — **p 만** (색 충돌을 없앤다)."""
    H, gold, cond, kk = pred_pack
    _by_k(lambda m: (H[(tgt, "p", ARM[m][0])], gold[tgt], ARM[m], cond, kk), ch, out, tgt,
          f"fig_vis_occ_k_{tgt}", f"{TGT[tgt]}:  p  readout\nfrom no occluder (%)")
    return


def _margin_stats(mg):
    """-> {(timing, viol): (base, margin%, effect size d, acc)}"""
    S = {}
    for (t, v), a in mg.items():
        a = np.array(a); rel = (a[:, 1] - a[:, 0]) / a[:, 0] * 100
        S[(t, v)] = (a[:, 0].mean(), rel.mean(), abs(rel.mean()) / rel.std(),
                     100 * (rel > 0).mean())
    return S


def fig_margin(mg, out):
    """주장 3 — 정보가 사라진 게 아니라 **신호가 묽어진다**.

    채점은 |p-h(가능)| 와 |p-h(불가능)| 의 대소다. 그 차이(margin)는 base 의 0.08~3.7%
    밖에 안 된다. 가림이 붙으면 base 는 늘고 margin 은 줄어 **효과크기**가 준다.
    정확도는 margin 의 부호 비율이므로 효과크기가 곧 정확도다.
    """
    S = _margin_stats(mg)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))
    specs = [("base", "|p − h| of the possible future", (0.566, 0.594),
              (0.57, 0.58, 0.59), lambda s: s[0], "%.3f"),
             ("margin", "margin  (impossible − possible)\nas % of base", (-0.15, 4.9),
              (0, 1, 2, 3, 4), lambda s: s[1], "%.2f"),
             ("d", "effect size   |mean| / SD", (0, 1.72), (0, 0.5, 1.0, 1.5),
              lambda s: s[2], "%.2f")]
    for ci, (ax, (key, ylab, ylim, yt, get, fmt)) in enumerate(zip(axes, specs)):
        ax.axvspan(2.5, 3.5, color=BAND, zorder=0)
        # ⚠️ (a) 에서 shape 와 colour 의 base 는 소수 셋째 자리까지 같다 (둘 다 물체 하나를
        #    바꾼 것이라 장면 통계가 같다). 그대로 그리면 한 선이 다른 선에 가려진다 -> x 를 민다.
        # ⚠️ (a) 에서 shape 와 colour 의 base 는 소수 셋째 자리까지 같다 (둘 다 물체 하나를
        #    바꾼 것이라 장면 통계가 같다). 선이 겹치므로 x 를 조금 민다. 라벨은 스택이 처리한다.
        DX = {"vanish": -0.05, "shape": 0.0, "color": 0.05}
        ax.set_ylim(*ylim)
        for v in VIOL:
            y = [get(S[(t, v)]) for t in TIM]
            xs = np.arange(4) + (DX[v] if key == "base" else 0.0)
            ax.plot(xs, y, marker="o", ms=5.2, lw=1.9, color=_pastel(VCOL[v]),
                    mec="white", mew=0.8, label=VLAB[v] if ci == 0 else None, zorder=4)
        g = (ylim[1] - ylim[0]) * 0.055
        for xi in range(4):
            col = [get(S[(TIM[xi], v)]) for v in VIOL]
            _stack(ax, xi, col, [VCOL[v] for v in VIOL], g, fmt, 6.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
        ax.set_ylim(*ylim); ax.set_yticks(yt); ax.tick_params(labelsize=7.4, colors=INK2)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4))
        ax.set_xticklabels([TLAB[t] for t in TIM], fontsize=7.2, color=INK)
        ax.set_ylabel(ylab, fontsize=8.0, color=INK)
    axes[0].legend(frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                   bbox_to_anchor=(1.75, 1.13), handlelength=1.3, columnspacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.275, wspace=0.34)
    panel_labels(fig, axes)
    save(fig, out, "fig_margin")


def fig_margin_vs_acc(mg, out):
    """주장 3 의 확인 — 12 셀 전부에서 효과크기가 정확도를 결정한다."""
    S = _margin_stats(mg)
    MK = {"vis": "o", "early": "^", "mid": "s", "late": "D"}
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    xs, ys = [], []
    for v in VIOL:
        for t in TIM:
            d, a = S[(t, v)][2], S[(t, v)][3]
            xs.append(d); ys.append(a)
            ax.scatter(d, a, s=42, marker=MK[t], color=_pastel(VCOL[v]),
                       edgecolor="white", lw=0.8, zorder=4)
    r = np.corrcoef(xs, ys)[0, 1]
    ax.annotate(f"Spearman ρ = {_spearman(xs, ys):.3f}", (0.04, 0.95), xycoords="axes fraction",
                fontsize=7.4, color=INK2, va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(color=MUTED, alpha=0.30, lw=0.4)
    ax.axhline(50, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("effect size of the margin   |mean| / SD", fontsize=8.4, color=INK)
    ax.set_ylabel("pairwise accuracy (%)", fontsize=8.4, color=INK)
    ax.tick_params(labelsize=7.4, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    h1 = [plt.Line2D([], [], marker="o", ls="", color=_pastel(VCOL[v]), mec="white",
                     ms=7.0, label=VLAB[v].split()[0] if v != "vanish" else "Permanence")
          for v in VIOL]
    h2 = [plt.Line2D([], [], marker=MK[t], ls="", color=MUTED, mec="white", ms=7.0,
                     label=TLAB[t].replace("\n", " ")) for t in TIM]
    lg1 = ax.legend(handles=h1, frameon=False, fontsize=LGF, ncol=3, loc="lower center",
                    bbox_to_anchor=(0.5, 1.13), handlelength=0.9, columnspacing=1.0,
                    handletextpad=0.4)
    ax.add_artist(lg1)
    ax.legend(handles=h2, frameon=False, fontsize=LGF, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=0.9, columnspacing=0.9,
              handletextpad=0.4)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.815, bottom=0.155)
    save(fig, out, "fig_margin_vs_acc")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ---- 표현을 가로지르는 이식: GT 미래(h) <-> 예측 미래(p) ------------------------
# h 와 p 는 **같은 미래 자리의 같은 shape (2048,1280)** 이라 readout 을 서로 걸 수 있다.
#   h -> p   GT 미래에서 배운 자리로 예측 미래를 읽는다  (채점이 실제로 하는 일)
#   p -> h   그 반대
# 비대칭이면 한쪽이 더 좁은/다른 표현이라는 뜻이다.
# ⚠️ 이 수치는 `attn_probe_xfer` 프로토콜에서만 나온다 (부모 attn_probe 는 eval: [self] 뿐).
SRC = {"h": "targetF__f17to32", "p": "pred__f17to32"}


def load_xfer(exp: Path):
    r = find_run(exp, "attn_probe_xfer")
    f = (r / "summary.json") if r else Path("/nonexistent")      # load() 와 같은 run
    if not f.exists():
        return None
    X = {}
    for r in json.loads(f.read_text())["probing"]:
        fit = PT[r["fit"]]
        for ev, cell in r["evals"].items():
            to = PT.get(ev, ev)
            for g, c in cell["per_group"].items():
                X[(r["target"], fit, to, r["groups"][0], g)] = 100 * c["acc"]
    return X


def load_xfer_preds(exp: Path, index: Path):
    """attn_probe_xfer 의 clip 별 예측 -> (target, fit지점, eval지점, 조건) 로 색인."""
    r = find_run(exp, "attn_probe_xfer")
    f = (r / "predictions.json") if r else Path("/nonexistent")  # load() 와 같은 run
    if not f.exists():
        return None
    M = json.loads(f.read_text())
    meta = {r["video_id"]: r for r in csv.DictReader(index.open())}
    vid = M["val_video_ids"]
    cond = np.array([meta[v]["condition"] for v in vid])
    kk = np.array([meta[v]["sym_k"] for v in vid])
    gold = {t: np.asarray(M["targets"][t]["gold"]) for t in M["targets"]}
    H = {(h["target"], PT[h["fit"]], PT.get(h["eval"], h["eval"]), h["groups"][0]):
         np.asarray(h["pred"]) for h in M["heads"]}
    return H, gold, cond, kk


def fig_self_k(pred_pack, ch, out, tgt):
    """자기 조건 head 를 k 별로 — **p 만**. `by_k/fig_vis_occ_k` 의 대조군이다.
    self 가 k 에 평평하면 이식 붕괴는 **정보 손실이 아니다**."""
    H, gold, cond, kk = pred_pack

    def get(m):
        gs = ARM[m]
        # 각 타이밍은 자기 조건 head 를 쓴다 -> 예측 배열이 조건마다 다르다.
        # 하나의 배열로 못 합치므로 조건별 head 예측을 그 조건 행에만 채워 넣는다.
        pr = np.full(len(cond), -1, dtype=np.int64)
        for ec in gs:
            m_ = cond == ec
            pr[m_] = H[(tgt, "p", ec)][m_]
        return pr, gold[tgt], gs, cond, kk
    _by_k(get, ch, out, tgt, f"fig_self_k_{tgt}",
          f"{TGT[tgt]}:  p  self probe (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    cells, T, ch, pred_pack, mg = load(a.exp, a.index)
    has_probe = T is not None
    fig_timing_step(cells, a.outdir)
    fig_timing_k(cells, a.outdir)
    if not has_probe:
        print("  (probing 결과 없음 — 채점·기전 그림만 그린다)")
        fig_margin(mg, a.outdir); fig_margin_vs_acc(mg, a.outdir); return
    for t in ("shape", "color"):
        _probe_fig(T, ch, a.outdir, t, "self")
        _probe_fig(T, ch, a.outdir, t, "vis")
        fig_vis_occ_k(pred_pack, ch, a.outdir, t)
    for t in ("shape", "color"):
        fig_self_k(pred_pack, ch, a.outdir, t)
    fig_margin(mg, a.outdir)
    fig_margin_vs_acc(mg, a.outdir)


if __name__ == "__main__":
    main()
