#!/usr/bin/env python3
"""predictor attention 그림 — extract.py 가 낸 attn.npz 하나에서 전부 그린다.

  matrix    층별 (질의 튜블릿 x 키 튜블릿) attention 질량. 주력
  profile   미래 질의가 각 프레임에 준 질량, 층을 겹쳐서. "어떤 프레임을 보나"
  distance  층별 attention distance (공간 px / 시간 frame) + 엔트로피
  group     조건별 비교 (문맥 질량 · 공간 거리 · 엔트로피)

  python -m analysis.attention.predictor_attn.plot \
      --npz z_research/IntPhysGenV11/exp_results/predictor_attn__v11_vith/attn.npz \
      --outdir z_research/IntPhysGenV11/figures/attention

⚠️ 시간 단위는 **샘플 프레임**이다 (프로토콜이 읽는 32장). 튜블릿 1 = 샘플 2장 = 원본 6장.
⚠️ 질의 튜블릿 0..C-1 은 진짜 문맥 토큰, C..T-1 은 **mask token** 이다. 둘은 성격이
   다르므로 한 축에 섞어 읽지 말 것 (그림에서 경계선으로 갈라 뒀다).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
PANEL = [f"({c})" for c in "abcdefghijkl"]

# 여기 없는 이름은 outdir 최상위에 떨어진다 — 새 그림이 눈에 띄라고 일부러.
FIGDIR = {
    "fig_attn_matrix": "01_layerwise",
    "fig_frame_profile": "01_layerwise",
    # 2026-09-03: 층별 산점 판(fig_attn_distance)은 헤드 12개가 겹쳐 안 읽혀 _superseded 로.
    # 대신 Interpreting Physics in Video World Models Fig.3 형식 — y=헤드, x=층, 색=거리.
    "fig_attn_distance": "_superseded",
    "fig_head_sdist": "02_distance",
    "fig_head_tdist": "02_distance",
    "fig_attn_group": "03_condition",
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tint(h, f):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


def frame(ax, ylim=None, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=MUTED, alpha=0.30, lw=0.4)
    ax.tick_params(labelsize=8.0, colors=INK2, pad=2.0)
    if ylim:
        ax.set_ylim(*ylim)


def save(fig, out: Path, name):
    d = out / FIGDIR.get(name, "")
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((d / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {d / name}.pdf + .png")


class A:
    """attn.npz 를 읽고 축 라벨·검증을 한 자리에 둔다."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.z = z
        self.groups = [str(g) for g in z["groups"]]
        self.T, self.C = int(z["T"]), int(z["C"])
        self.L, self.H = int(z["n_layers"]), int(z["n_heads"])
        self.tub, self.S = int(z["tubelet"]), int(z["S"])
        self.mass, self.same = z["mass"], z["same"]                  # (G,L,H,T,T)
        self.sdist, self.tdist = z["sdist_px"], z["tdist_frames"]    # (G,L,H,T)
        self.sdist_patch = z["sdist_patch"]                            # (G,L,H,T) 패치 단위
        self.toff, self.ent = z["toff_frames"], z["entropy"]
        self.n_clips = z["n_clips"]
        rs = self.mass.sum(-1)
        err = float(np.abs(rs - 1).max())
        assert err < 1e-3, f"mass 행 합이 1 에서 {err:.2e} 벗어났다 — npz 가 깨졌다"
        print(f"  검증 OK — mass 행 합 오차 {err:.2e} | 층 {self.L} 헤드 {self.H} "
              f"튜블릿 {self.T}(문맥 {self.C}) | 그룹 {len(self.groups)} "
              f"클립 {list(self.n_clips)}")

    def ticks(self):
        return [f"{t*self.tub+1}" for t in range(self.T)]

    def fut(self):
        return slice(self.C, self.T)


# ------------------------------------------------------------------ 1. matrix
def fig_matrix(a: A, out: Path, gi=None, width=7.6):
    """층별 (질의 x 키) 튜블릿 attention. 헤드·그룹 평균."""
    M = a.mass[gi] if gi is not None else a.mass.mean(0)          # (L,H,T,T)
    M = M.mean(1)                                                 # (L,T,T)
    vmax = float(np.percentile(M, 99.5))
    cm = LinearSegmentedColormap.from_list("m", ["#ffffff", tint(BLUE, 0.55), BLUE, "#10305c"])
    nc = 4
    nr = int(np.ceil(a.L / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(width, width * nr / nc * 1.02))
    for li in range(nr * nc):
        ax = axes.flat[li]
        if li >= a.L:
            ax.axis("off")
            continue
        ax.imshow(M[li], cmap=cm, vmin=0, vmax=vmax, origin="upper", aspect="equal")
        for v in (a.C - 0.5,):
            ax.axhline(v, color=ORANGE, lw=0.8)
            ax.axvline(v, color=ORANGE, lw=0.8)
        ax.set_title(f"layer {li}", fontsize=8.5, color=INK, pad=3)
        ax.set_xticks([0, a.C, a.T - 1]); ax.set_yticks([0, a.C, a.T - 1])
        ax.set_xticklabels([a.ticks()[0], a.ticks()[a.C], a.ticks()[-1]], fontsize=6.5)
        ax.set_yticklabels([a.ticks()[0], a.ticks()[a.C], a.ticks()[-1]], fontsize=6.5)
        ax.tick_params(length=1.6, colors=INK2, pad=1.5)
        if li % nc == 0:
            ax.set_ylabel("query frame", fontsize=7.5, color=INK2)
        if li // nc == nr - 1:
            ax.set_xlabel("key frame", fontsize=7.5, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.015)
    cb.set_label("attention mass per key frame (row sums to 1)", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    save(fig, out, "fig_attn_matrix")


# ----------------------------------------------------------------- 2. profile
def fig_profile(a: A, out: Path, width=7.0):
    """질의가 각 키 프레임에 준 질량. 층을 색으로 겹친다."""
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.40))
    M = a.mass.mean((0, 2))                                       # (L,T,T) 그룹·헤드 평균
    parts = [("context queries (real tokens)", slice(0, a.C)),
             ("future queries (mask tokens)", a.fut())]
    x = np.arange(a.T)
    for pi, (lab, sl) in enumerate(parts):
        ax = axes[pi]
        frame(ax)
        for li in range(a.L):
            y = M[li, sl].mean(0)
            ax.plot(x, y, "-", lw=1.2, color=tint(BLUE, 0.78 - 0.78 * li / max(1, a.L - 1)),
                    zorder=3 + li)
        ax.axvline(a.C - 0.5, color=ORANGE, lw=0.9, ls="--", zorder=2)
        ax.axhline(1.0 / a.T, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_xticks(x[::2]); ax.set_xticklabels(a.ticks()[::2], fontsize=7)
        ax.set_xlabel("key frame", fontsize=8.5, color=INK2)
        ax.set_title(lab, fontsize=8.5, color=INK, pad=4)
        ax.text(0.5, -0.30, PANEL[pi], transform=ax.transAxes, ha="center",
                fontsize=9, color=INK)
        if pi == 0:
            ax.set_ylabel("attention mass", fontsize=8.5, color=INK2)
    h = [Line2D([], [], color=tint(BLUE, 0.78), lw=1.4, label="layer 0"),
         Line2D([], [], color=tint(BLUE, 0.0), lw=1.4, label=f"layer {a.L-1}"),
         Line2D([], [], color=ORANGE, lw=0.9, ls="--", label="context / future split"),
         Line2D([], [], color=MUTED, lw=0.8, ls=":", label="uniform (1/T)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.28))
    save(fig, out, "fig_frame_profile")


# ---------------------------------------------------------------- 3. distance
def fig_distance(a: A, out: Path, width=7.6):
    """층별 attention distance 와 엔트로피. 헤드는 산점, 평균은 선."""
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.30))
    series = [("context", slice(0, a.C), BLUE), ("future", a.fut(), ORANGE)]
    panels = [(a.sdist, "spatial attention distance (px)", None),
              (a.tdist, "temporal attention distance (frames)", None),
              (a.ent, "attention entropy (nats)", None)]
    x = np.arange(a.L)
    for pi, (arr, lab, yl) in enumerate(panels):
        ax = axes[pi]
        frame(ax, yl)
        for name, sl, col in series:
            v = arr[:, :, :, sl].mean((0, 3))                     # (L,H)
            for h in range(a.H):
                ax.plot(x, v[:, h], ".", ms=2.0, color=tint(col, 0.62), zorder=2)
            ax.plot(x, v.mean(1), "-o", ms=3.4, lw=1.3, color=col, zorder=4,
                    mec="white", mew=0.7)
        if pi == 2:
            ax.axhline(float(np.log(a.T * a.S)), color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_xticks(x[::2])
        ax.set_xlabel("predictor layer", fontsize=8.5, color=INK2)
        ax.set_title(lab, fontsize=8.5, color=INK, pad=4)
        ax.text(0.5, -0.34, PANEL[pi], transform=ax.transAxes, ha="center",
                fontsize=9, color=INK)
    h = [Line2D([], [], color=c, marker="o", ms=3.4, lw=1.3, mec="white",
                label=f"{n} queries") for n, _, c in series]
    h.append(Line2D([], [], color=MUTED, lw=0.8, ls=":", label="uniform entropy log(N)"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.30))
    save(fig, out, "fig_attn_distance")


# ------------------------------------------------------------ 3b. head x layer
def fig_head(a: A, out: Path, kind="sdist", queries="future", width=7.0):
    """헤드 × 층 히트맵 — Interpreting Physics in Video World Models Fig.3 형식.

    y = attention head, x = predictor layer, 색 = 거리. **거리가 작을수록 진한 파랑.**
    칸마다 값을 적는다. 그룹(조건)은 평균한다 — 여섯 조건이 이 집계에서 거의 겹친다
    (03_condition/ 참고). 질의는 기본 **future(mask token)** 만 쓴다 — 문맥 토큰과
    성격이 다르니 섞지 않는다 (README ⚠️).
    """
    arr = a.sdist_patch if kind == "sdist" else a.tdist
    sl = {"future": a.fut(), "context": slice(0, a.C), "all": slice(0, a.T)}[queries]
    M = arr[:, :, :, sl].mean((0, 3)).T                                # (H, L)
    unit = "patches" if kind == "sdist" else "frames"
    cm = plt.get_cmap("Blues_r")                                       # 작을수록 진하게
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    im = ax.imshow(M, cmap=cm, aspect="auto", origin="lower",
                   vmin=float(M.min()), vmax=float(M.max()))
    lo, hi = float(M.min()), float(M.max())
    for h in range(a.H):
        for l in range(a.L):
            v = M[h, l]
            ax.text(l, h, f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                    color="white" if (v - lo) / max(hi - lo, 1e-9) < 0.45 else INK)
    ax.set_xticks(range(a.L)); ax.set_yticks(range(a.H))
    ax.set_xticklabels([str(i) for i in range(a.L)], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(a.H)], fontsize=7)
    ax.tick_params(length=0, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)
    ax.set_xlabel("Layer", fontsize=9, color=INK)
    ax.set_ylabel("Attention Head", fontsize=9, color=INK)
    ax.set_title(f"V-JEPA 2 ViT-H predictor: {'spatial' if kind == 'sdist' else 'temporal'} "
                 f"attention distance per head   ({queries} queries)",
                 fontsize=9, color=INK, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"Distance ({unit})", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.92, bottom=0.11)
    save(fig, out, f"fig_head_{kind}")


# ------------------------------------------------------------------- 4. group
def fig_group(a: A, out: Path, width=7.6):
    """조건별 비교 — 미래 질의만. 그룹이 하나뿐이면 건너뛴다."""
    if len(a.groups) < 2:
        print("  [skip] fig_attn_group — 그룹이 1개다 (--group-by 를 준 적 없다)")
        return
    fut = a.fut()
    panels = [(a.mass[:, :, :, fut, :a.C].sum(-1).mean((2, 3)), "mass on context frames"),
              (a.sdist[:, :, :, fut].mean((2, 3)), "spatial attention distance (px)"),
              (a.ent[:, :, :, fut].mean((2, 3)), "attention entropy (nats)")]
    base = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.30))
    x = np.arange(a.L)
    for pi, (v, lab) in enumerate(panels):
        ax = axes[pi]
        frame(ax)
        for gi, g in enumerate(a.groups):
            col = base[(gi // 2) % 3] if len(a.groups) % 2 == 0 else base[gi % 3]
            ls = "-" if gi % 2 == 0 else "--"
            ax.plot(x, v[gi], ls, lw=1.3, color=col, zorder=3,
                    marker="o" if gi % 2 == 0 else "s", ms=3.0, mec="white", mew=0.6)
        ax.set_xticks(x[::2])
        ax.set_xlabel("predictor layer", fontsize=8.5, color=INK2)
        ax.set_title(lab, fontsize=8.5, color=INK, pad=4)
        ax.text(0.5, -0.34, PANEL[pi], transform=ax.transAxes, ha="center",
                fontsize=9, color=INK)
    h = [Line2D([], [], color=base[(gi // 2) % 3] if len(a.groups) % 2 == 0 else base[gi % 3],
                ls="-" if gi % 2 == 0 else "--", lw=1.3,
                marker="o" if gi % 2 == 0 else "s", ms=3.0, mec="white",
                label=g) for gi, g in enumerate(a.groups)]
    fig.legend(handles=h, loc="lower center", ncol=min(3, len(a.groups)), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.34))
    save(fig, out, "fig_attn_group")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--which", nargs="*",
                    default=["matrix", "profile", "distance", "group", "head"])
    ap.add_argument("--queries", default="future", choices=["future", "context", "all"],
                    help="head 히트맵의 질의 부분집합 (기본 future = mask token)")
    ap.add_argument("--group", type=int, default=None,
                    help="matrix 를 그룹 하나로만 (기본: 그룹 평균)")
    a_ = ap.parse_args()
    a = A(a_.npz)
    sm = a_.npz.parent / "summary.json"
    if sm.exists():
        v = json.loads(sm.read_text()).get("verify", {})
        print(f"  extract 검증 — 행합 {v.get('row_sum_max_err'):.2e} | "
              f"attn@v vs 실제 max|Δ| {v.get('max_abs_diff', float('nan')):.2e}")
    for w in a_.which:
        {"matrix": lambda: fig_matrix(a, a_.outdir, a_.group),
         "profile": lambda: fig_profile(a, a_.outdir),
         "distance": lambda: fig_distance(a, a_.outdir),
         "head": lambda: [fig_head(a, a_.outdir, k, a_.queries) for k in ("sdist", "tdist")],
         "group": lambda: fig_group(a, a_.outdir)}[w]()


if __name__ == "__main__":
    main()
