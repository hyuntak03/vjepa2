#!/usr/bin/env python3
"""RollOut_v1 — predictor 가 만든 미래에서 위치가 읽히고, 그 기울기가 속도인가.

패널 하나에 주장 하나, 지표 하나:
  (a) 모든 시점에서 위치가 읽힌다   x = 튜블릿 0..7, y = 위치 오차   지표: 위치 R² (클립내부)
  (b) 그 기울기가 속도다            x = 진짜 속도,  y = 읽은 속도    지표: 속도 R²

⚠️ "읽은 위치 vs 진짜 위치" 산점도는 **그리지 말 것.** 진짜 위치 분산의 80.7% 가 anchor
   (클립 시작 위치 3단계)라, 점이 대각선에 붙는 것의 대부분이 "어디서 시작했는지 안다" 는
   뜻이 된다. 우리가 묻는 움직임 성분은 나머지 19.3% 뿐이다.

색 = 표현 종류.  h(파랑) = target encoder(실제 미래를 봄) = 도구 천장,
                p(주황) = predictor(미래를 못 봄) = 핵심.

⚠️ (b) 를 R² 로 그리지 말 것. 클립 평균을 빼면 t=3,4 의 정답 분산이 t=0 의 1/50 (0.0007)
   이라 R² 분모가 0 에 가까워져 가운데가 U 자로 꺼진다. 정확도가 아니라 정규화 함정이다.
   절대 오차(cm)로 그리면 깨끗하다.

폰트: Times New Roman → 없으면 Nimbus Roman (metric-compatible 클론, CLAUDE.md §8-4).
수치는 `position_regression.json` 과 대조 검증한다 (틀리면 죽는다).

  python z_research/scripts/figures/plot_rollout_position.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib as mpl
import numpy as np
mpl.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout_position as rp                                     # noqa: E402

FIGDIR = {"fig_position_readout": "position"}
OUTROOT = rp.ROOT / "z_research/RollOutV1/figures"
JSON = rp.ROOT / "z_research/RollOutV1/exp_results/position_regression.json"

INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SRC = {"h": (BLUE, "$h$  target encoder  (saw the future)"),
       "p": (ORANGE, "$p$  predictor  (did not)")}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
FS_LAB, FS_TICK, FS_NOTE, FS_LEG = 9.0, 8.0, 8.2, 9.0


def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color=MUTED, alpha=0.22, lw=0.4)
    ax.tick_params(labelsize=FS_TICK, colors=INK2, pad=2.0)


def note(ax, txt):
    """패널 지표 — 데이터와 안 겹치게 축 밖 위쪽에."""
    ax.text(0.0, 1.02, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_NOTE, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--outdir", type=Path, default=OUTROOT)
    a = ap.parse_args()

    ref = json.loads(JSON.read_text())["results"]
    idx = {r["video_id"]: r for r in csv.DictReader(rp.INDEX.open())}
    vids, F = rp.pool()
    rows = [idx[v] for v in vids]
    Y, v_true, anc_n, hw, fps = rp.labels_from_index(rows)
    cond = np.array([r["condition"] for r in rows])
    cm = hw / (6.0 / fps)

    rng = np.random.RandomState(0); tr, va = [], []
    for c in np.unique(cond):
        i = np.where(cond == c)[0]; rng.shuffle(i); h = len(i) // 2
        tr += list(i[:h]); va += list(i[h:])
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    flat = lambda s, ids: F[s][ids].reshape(-1, 1280).astype(np.float64)
    W = {s: rp.fit_lstsq(flat(s, tr), Y[tr].ravel()) for s in ("h", "p")}
    XH = {s: rp.apply_w(flat(s, va), W[s]).reshape(len(va), 8) for s in ("h", "p")}

    # ── json 대조 ────────────────────────────────────────────────────────────
    yc = Y[va] - Y[va].mean(1, keepdims=True)
    for s in ("h", "p"):
        xc = XH[s] - XH[s].mean(1, keepdims=True)
        r2p = 1 - ((yc - xc) ** 2).sum() / (yc ** 2).sum()
        b, _ = rp.slope(XH[s]); bl, _ = rp.slope(Y[va])
        A = np.vstack([bl, np.ones_like(bl)]).T
        be, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
        r2v = 1 - ((b - A @ [be, c0]) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        for got, want, nm in ((r2p, ref[f"{s}->{s}"]["r2_pos_within_clip"], "위치"),
                              (r2v, ref[f"{s}->{s}"]["r2_speed"], "속도")):
            if abs(got - want) > 2e-3:
                sys.exit(f"ERROR: {s} {nm} R² {got:.4f} != json {want:.4f}")
    print("  검증 OK — json 과 일치")

    E = {s: np.abs(XH[s] - Y[va]) * hw for s in ("h", "p")}
    vs = np.unique(v_true[va])

    fig, ax = plt.subplots(1, 2, figsize=(a.width * 0.72, 2.85))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.295, wspace=0.34)
    B_, C_ = ax

    # ── (a) 모든 시점에서 위치가 읽힌다 ─────────────────────────────────────
    frame(B_)
    tt = np.arange(8)
    half_obj = float(rows[0]["obj_size_cm"]) / 2
    B_.axhline(half_obj, ls=":", color=MUTED, lw=1.0, zorder=1)
    B_.text(7.3, half_obj, "half object width", fontsize=FS_TICK - 0.5, color=MUTED,
            ha="right", va="bottom")
    for s in ("h", "p"):
        m, se = E[s].mean(0), E[s].std(0, ddof=1) / np.sqrt(len(E[s]))
        B_.fill_between(tt, m - se, m + se, color=SRC[s][0], alpha=0.22, lw=0, zorder=2)
        B_.plot(tt, m, "-o", color=SRC[s][0], lw=1.3, ms=3.8, mec="white", mew=0.5, zorder=3)
    B_.set_xticks(tt); B_.set_xlim(-0.4, 7.4); B_.set_ylim(0, half_obj * 1.14)
    B_.set_xlabel("tubelet index  (future slot)", fontsize=FS_LAB)
    B_.set_ylabel("position error  (cm)", fontsize=FS_LAB)
    note(B_, f"position $R^2$ (within clip)   $h$ {ref['h->h']['r2_pos_within_clip']:.3f}"
             f"    $p$ {ref['p->p']['r2_pos_within_clip']:.3f}")

    # ── (b) 그 기울기가 속도다 ──────────────────────────────────────────────
    frame(C_)
    vl = 185
    C_.plot([-vl, vl], [-vl, vl], "--", color=MUTED, lw=0.9, zorder=1)
    for s in ("h", "p"):
        b, _ = rp.slope(XH[s]); bh = b * cm
        m = np.array([bh[v_true[va] == u].mean() for u in vs])
        sd = np.array([bh[v_true[va] == u].std(ddof=1) for u in vs])
        C_.errorbar(vs, m, yerr=sd, fmt="o", ms=4.0, color=SRC[s][0], mec="white",
                    mew=0.5, elinewidth=0.9, capsize=2.0, zorder=3)
    C_.set_xlim(-vl, vl); C_.set_ylim(-vl, vl)
    C_.set_xticks([-160, 0, 160]); C_.set_yticks([-160, 0, 160])
    C_.set_xlabel("true velocity  (cm/s)", fontsize=FS_LAB)
    C_.set_ylabel("read-out velocity  (cm/s)", fontsize=FS_LAB)
    note(C_, f"velocity $R^2$   $h$ {ref['h->h']['r2_speed']:.3f}"
             f"    $p$ {ref['p->p']['r2_speed']:.3f}")

    for k, x in enumerate(ax):
        x.text(0.5, -0.285, f"({'ab'[k]})", transform=x.transAxes, ha="center",
               va="top", fontsize=FS_LAB + 0.5)

    hs = [plt.Line2D([], [], color=SRC[s][0], lw=1.6, marker="o", ms=4.5,
                     mec="white", mew=0.5, label=SRC[s][1]) for s in ("h", "p")]
    hs.append(plt.Line2D([], [], color=MUTED, ls="--", lw=1.0, label="perfect read-out"))
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEG,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.6,
               bbox_to_anchor=(0.5, -0.015))

    d = a.outdir / FIGDIR["fig_position_readout"]
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_position_readout.{ext}", dpi=240)
    print(f"  -> {d}/fig_position_readout.{{pdf,png}}")


if __name__ == "__main__":
    main()
