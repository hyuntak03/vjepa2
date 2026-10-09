#!/usr/bin/env python3
"""IntPhys 1 의 88.89 칸 (Garrido sliding, skip2_w32, Filtered) 에서 창마다 어떤 raw 프레임이 어느 encoder 로 가는가.

창은 채점기와 같은 생성기 (`evals.world_model_analysis.eval._intphys1_windows`) 로 만든다 — 손으로 다시 짜지 않는다.
  frame_budget official: 100 프레임 → skip 2 로 솎아 raw 0,2,…,96 (49 장). 97–99 는 버려진다.
  창 32 장 (= raw 64 프레임 폭), 시작점 stride 2 (솎은 격자 기준) → raw 시작 0,4,…,32 의 9 개.
  C ∈ {4,8,12,16,20} (context_mult × 32/16). 창마다: target encoder = 32 장 전부, context encoder = 앞 C 장,
  predictor 가 뒤 M = 32 − C 장을 예측하고 그 자리의 LN(target) 과 L1 로 채점. Filtered = 시작점마다 C 중 최소 surprise.

그림 (z_research/Benchmarks/figures/intphys1_windows/):
  window_map_skip2_w32.png         45 창 (9 시작 × 5 C) × raw 프레임 0–99 의 역할 지도
  frames_skip2_w32_<video>.png     한 영상의 9 창 × 창 안 32 장 썸네일. 행마다 Filtered 가 고른 C 로 경계를 긋는다
                                   (per_window.json 의 창 surprise 에서 min 을 다시 계산), pos/imp 픽셀이 처음 갈리는 프레임에 표식

  python z_research/scripts/figures/plot_intphys1_windows.py [--pos O1_01_2 --imp O1_01_1]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
PER_WIN = ROOT / "z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32/per_window.json"
OUT = ROOT / "z_research/Benchmarks/figures/intphys1_windows"
N_RAW, SKIP, WZ, STRIDE, TUB = 100, 2, 32, 2, 2
CTX = [m * WZ // 16 for m in (2, 4, 6, 8, 10)]
BLUE, ORANGE, SKIPPED, BG, INK = "#2a78d6", "#eb6834", "#d9d9d9", "#ffffff", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def windows():
    wins = _intphys1_windows(N_RAW, SKIP, WZ, CTX, STRIDE, TUB, "official")
    by = {}
    for C, cf, tf in wins:
        by.setdefault(cf[0], {})[C] = (cf, tf)
    return by                                   # raw 시작 → C → (context raw idx, future raw idx)


def frame_path(vid: str, raw: int) -> Path:
    b, q, r = vid.split("_")
    return FR / b / q / r / "scene" / f"scene_{raw + 1:03d}.png"       # frames_start 1


def window_map(by):
    starts = sorted(by)
    rows = [(s, C) for s in starts for C in CTX]
    gap = 0.6
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    y = 0.0; yt, yl = [], []
    for gi, s in enumerate(starts):
        for C in CTX:
            cf, tf = by[s][C]
            span = range(cf[0], tf[-1] + 1)
            for f in range(N_RAW):
                col = BLUE if f in cf else ORANGE if f in tf else SKIPPED if f in span else None
                if col:
                    ax.add_patch(Rectangle((f, y), 1, 0.9, color=col, lw=0))
            yt.append(y + 0.45); yl.append(f"C={C:>2}  M={WZ - C:>2}")
            y += 1
        ax.text(-9.5, y - 2.5, f"start\nraw {s}", ha="center", va="center", fontsize=7, color=INK)
        y += gap
    ax.axvline(97, color=INK, lw=0.6, ls=":")
    ax.text(97.6, y / 2, "raw 97–99 unused", fontsize=6.5, color=INK, va="center", rotation=90)
    ax.set_xlim(-14, N_RAW); ax.set_ylim(y, -1.6)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.5, family="DejaVu Sans Mono")
    ax.set_xticks(range(0, N_RAW + 1, 10)); ax.set_xlabel("raw frame index (0-based; file scene_{i+1:03d}.png)")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    for sp in ("right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BLUE, label="context (context enc. + target enc.)"),
                       Patch(color=ORANGE, label="future (predicted; target enc.)"),
                       Patch(color=SKIPPED, label="skipped by frame skip 2 (no input)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    ax.set_title("IntPhys 1 · skip2_w32 (88.89 cell): 9 starts × 5 context lengths; target encoder always gets all 32 frames",
                 fontsize=8, pad=24)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "window_map_skip2_w32.png", dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def load_thumb(vid, raw, px):
    return np.asarray(Image.open(frame_path(vid, raw)).convert("RGB").resize((px, px), Image.BILINEAR))


def first_divergence(pos, imp):
    for f in range(N_RAW):
        a = np.asarray(Image.open(frame_path(pos, f))); b = np.asarray(Image.open(frame_path(imp, f)))
        if not np.array_equal(a, b):
            return f
    return None


def chosen_C(vid):
    d = json.load(open(PER_WIN))["windows"][vid]
    best = {}
    for combo, s, C, sur in d:
        if combo == "skip2_w32" and (s not in best or sur < best[s][1]):
            best[s] = (C, sur)
    return {s: c for s, (c, _) in best.items()}


def frames_figure(by, vid, role, div):
    starts = sorted(by)
    Cs = chosen_C(vid)
    px, bd, tg, sg = 72, 3, 4, 12                  # 썸네일, 테두리, 튜블릿 사이 틈, 창 사이 틈
    cell = px + 2 * bd
    width = WZ * cell + (WZ // TUB - 1) * tg
    rowh = cell + 14
    H = len(starts) * (rowh + sg)
    canvas = np.full((H, width, 3), 255, np.uint8)
    labels, divmarks = [], []
    for ri, s in enumerate(starts):
        C = Cs[s]
        cf, tf = by[s][C]
        frames = cf + tf
        y0 = ri * (rowh + sg)
        for k, f in enumerate(frames):
            x0 = k * cell + (k // TUB) * tg
            col = np.array(matplotlib.colors.to_rgb(BLUE if k < C else ORANGE)) * 255
            canvas[y0:y0 + cell, x0:x0 + cell] = col.astype(np.uint8)
            canvas[y0 + bd:y0 + bd + px, x0 + bd:x0 + bd + px] = load_thumb(vid, f, px)
            labels.append((x0, y0, f))
        if div is not None:
            ks = [k for k, f in enumerate(frames) if f >= div]
            if ks:                                   # 이 창에서 갈린 뒤 첫 프레임
                divmarks.append((ks[0] * cell + (ks[0] // TUB) * tg, y0))
    fig_w = 7.0 * 1.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * H / width + 0.9))
    ax.imshow(canvas, interpolation="nearest")
    for x0, y0, f in labels:
        ax.text(x0 + cell / 2, y0 + cell + 1, str(f), ha="center", va="top", fontsize=4.2, color=INK)
    for x0, y0 in divmarks:
        ax.plot(x0 + cell / 2, y0 - 2.5, marker="v", ms=4, color=INK, clip_on=False)
    for ri, s in enumerate(starts):
        y0 = ri * (rowh + sg); C = Cs[s]
        xb = C * cell + (C // TUB) * tg - tg / 2 - 0.5
        ax.plot([xb, xb], [y0 - 2, y0 + cell + 1], color=INK, lw=1.4)
        for Co in CTX:
            if Co != C:
                xo = Co * cell + (Co // TUB) * tg - tg / 2 - 0.5
                ax.plot([xo, xo], [y0 + cell + 0.5, y0 + cell + 5], color=INK, lw=0.5, alpha=0.6)
        ax.text(-6, y0 + cell / 2, f"start raw {s}\nC*={C} (Filtered)", ha="right", va="center", fontsize=6, color=INK)
    ax.set_xlim(-2, width + 1); ax.set_ylim(H, -8); ax.axis("off")
    ttl = (f"{vid} ({role}) · IntPhys 1 skip2_w32: each row = one window fed to V-JEPA 2 ViT-H "
           f"(32 frames; number = raw index)")
    ax.set_title(ttl, fontsize=7.5, loc="left")
    handles = [Patch(color=BLUE, label="context (context encoder + target encoder)"),
               Patch(color=ORANGE, label="future (predicted; target encoder)"),
               plt.Line2D([], [], color=INK, lw=1.4, label="boundary at C* chosen by Filtered (min over C)"),
               plt.Line2D([], [], color=INK, lw=0.5, label="other C candidates")]
    if div is not None:
        handles.append(plt.Line2D([], [], color=INK, marker="v", lw=0, ms=4, label=f"first pos/imp pixel difference (raw {div})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=6.5)
    fig.savefig(OUT / f"frames_skip2_w32_{vid}.png", dpi=240, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return Cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="O1_01_2")
    ap.add_argument("--imp", default="O1_01_1")
    a = ap.parse_args()
    by = windows()
    assert sorted(by) == list(range(0, 33, 4)) and all(sorted(v) == CTX for v in by.values()), "창 구성이 기대와 다르다"
    for s in by:
        for C, (cf, tf) in by[s].items():
            assert len(cf) == C and len(cf) + len(tf) == WZ and cf + tf == list(range(s, s + 2 * WZ, 2))
    window_map(by)
    div = first_divergence(a.pos, a.imp)
    out = {"first_pixel_divergence_raw": div}
    for vid, role in ((a.pos, "possible"), (a.imp, "impossible")):
        out[vid] = frames_figure(by, vid, role, div)
    print(json.dumps(out, indent=1))
    print("→", OUT)


if __name__ == "__main__":
    main()
