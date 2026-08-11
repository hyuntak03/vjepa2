# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
# Layer-wise validation-accuracy plot (x = encoder layer, y = val acc),
# one line per probe. Enabled by config experiment.analysis.plot.
# -----------------------------------------------------------------------------

import logging
import math
from collections import defaultdict

logger = logging.getLogger()


def _elbow_x(xs, ys):
    """Elbow/knee = point of max perpendicular distance from the chord joining the
    first and last point. Parameter-free; for a rising-then-flat layer curve this is
    the 'saturation' layer (where accuracy stops improving meaningfully). Returns x."""
    if len(xs) < 3:
        return None
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy)
    if denom == 0:
        return None
    best_x, best_d = None, -1.0
    for x, y in zip(xs, ys):
        d = abs(dy * (x - x0) - dx * (y - y0)) / denom  # |cross| / |chord|
        if d > best_d:
            best_d, best_x = d, x
    return best_x


def plot_layer_val_acc(heads, val_acc, out_path, subtitle=None, num_classes=None,
                       metric="accuracy", target_label=None, pez=None, legend_title=None,
                       annotate=None):
    """Save a [layer x metric] line plot (one line per probe).

    heads:    list of dict with keys 'layer' and 'name' (name == "L<layer>_<probe>")
    val_acc:  dict head_name -> metric value (accuracy % | R^2)
    metric:   'accuracy' (y in %, chance line, 0-120) | 'r2' (y in R^2, baseline at 0, ..1.0)
    target_label: regressed-variable name (R^2 plots) shown in the title
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional; never break training
        logger.warning(f"plot skipped (matplotlib unavailable): {e}")
        return None

    is_r2 = metric == "r2"
    vfmt = (lambda v: f"R²={v:.3f}") if is_r2 else (lambda v: f"{v:.1f}%")

    # group (x, val) by probe label. Prefer an explicit `probe` key (set by the VLM
    # harness); fall back to parsing `name` for the clip harness. Splitting `name`
    # on the first '_' is WRONG for VLM stage tags (e.g. "block_5_linear-mean"), hence
    # the explicit key.
    # one line per `series`: the regressed VARIABLE for R^2 (so speed/direction/accel are separate
    # curves on one plot), else the probe type for classification. Fall back to parsing `name`.
    series = defaultdict(list)
    for h in heads:
        probe_label = (h.get("series") or h.get("probe")
                       or (h["name"].split("_", 1)[1] if "_" in h["name"] else h["name"]))
        series[probe_label].append((h["layer"], val_acc[h["name"]]))

    # optional per-x stage labels (VLM): use the stage name for ticks/annotations
    stage_by_x = {h["layer"]: h["stage"] for h in heads if h.get("stage") is not None}

    # per-curve peak stars / elbow diamonds / vertical guides are readable for a few
    # curves and pure noise for many (10 probed attributes = 30 annotations). Default:
    # annotate only when the plot has at most 4 curves. Peaks stay in the bar chart /
    # attribute_summary.csv either way.
    if annotate is None:
        annotate = len(series) <= 4

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Physics Emergence Zone: shade a layer-fraction band (paper marks it ~1/3 depth, band ≈0.2–0.4).
    # pez = [lo, hi] fractions in [0,1]; mapped onto the layer axis via the deepest layer index.
    if pez:
        depth = max((h["layer"] for h in heads), default=0) or 1
        ax.axvspan(pez[0] * depth, pez[1] * depth, color="gray", alpha=0.15, zorder=0)
        ax.text((pez[0] + pez[1]) / 2 * depth, 0.99, "PEZ", color="dimgray", fontsize=8,
                ha="center", va="top", transform=ax.get_xaxis_transform())

    all_y = []
    for probe_label in sorted(series):
        # DEDUP by layer: when multiple heads share the same (series, layer) -- e.g. an HP sweep
        # runs many probe configurations under a single `probe: linear-mean` label -- take the MAX
        # (paper's max-over-HP protocol). Without this the plotter connects all HPs sequentially at
        # each x, producing a sawtooth. Idempotent for the single-head case.
        layer_best = {}
        for x, y in series[probe_label]:
            if x not in layer_best or y > layer_best[x]:
                layer_best[x] = y
        pts = sorted(layer_best.items(), key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_y += ys
        (line,) = ax.plot(xs, ys, linewidth=2, label=probe_label)  # 선만 (마커 없음)
        color = line.get_color()
        if not annotate:
            continue      # many curves -> per-curve peak/elbow marks would bury the plot
        # PEAK = argmax metric for this probe (first stage reaching the global max if tied).
        # Mark it explicitly: star marker on the curve + vertical guide + bold label.
        bx, by = max(pts, key=lambda t: t[1])
        peak_label = stage_by_x.get(bx, f"L{bx}")
        ax.axvline(bx, color=color, linestyle="--", alpha=0.35, linewidth=1)
        ax.scatter([bx], [by], s=90, marker="*", color=color, zorder=5,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(f"peak: {peak_label} ({vfmt(by)})", (bx, by),
                    textcoords="offset points", xytext=(0, 9), fontsize=8,
                    fontweight="bold", ha="center", color=color)
        # ELBOW = saturation layer (max distance from the first->last chord). Hollow
        # diamond + dotted guide, distinct from the filled-star peak. Skipped if == peak.
        # Only drawn for the 'direction' series — it's the meaningful emergence marker
        # (speed/accel are high from the start, so their elbow isn't informative).
        ex = _elbow_x(xs, ys) if (annotate and probe_label == "direction") else None
        if ex is not None and ex != bx:
            ey = dict(zip(xs, ys))[ex]
            elbow_label = stage_by_x.get(ex, f"L{ex}")
            ax.axvline(ex, color=color, linestyle=":", alpha=0.4, linewidth=1)
            ax.scatter([ex], [ey], s=75, marker="D", facecolors="none",
                       edgecolors=color, linewidths=1.4, zorder=5)
            ax.annotate(f"elbow: {elbow_label} ({vfmt(ey)})", (ex, ey),
                        textcoords="offset points", xytext=(6, -16), fontsize=8,
                        fontweight="bold", ha="left", color=color)

    # x-axis as LAYER FRACTION (0..1), matching the paper. Data/PEZ/peak positions stay in
    # layer-index space (axis is linear), we just place ticks at fractions of the deepest layer.
    depth = max((h["layer"] for h in heads), default=0) or 1
    ax.set_xticks([depth * f for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlim(-0.02 * depth, 1.02 * depth)
    ax.set_xlabel("layer fraction")

    if is_r2:
        ax.set_ylabel("validation R²")
        lo = min(0.0, min(all_y) if all_y else 0.0)
        ax.set_ylim(lo - 0.03, 1.03)
        start = math.floor(lo / 0.2) * 0.2
        ax.set_yticks([round(start + 0.2 * k, 2) for k in range(int(round((1.0 - start) / 0.2)) + 1)])
        # R²=0 baseline = predicting the target mean (a probe below this is worse than the mean).
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
        ax.text(0.995, 0.02, "R²=0 (predict mean)", color="red", fontsize=8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
    else:
        ax.set_ylabel("val accuracy (%)")
        ax.set_ylim(0, 120)                 # fixed y-scale so plots are comparable across runs
        ax.set_yticks(range(0, 121, 20))    # 0,20,40,60,80,100,120
        if num_classes:                     # random-chance baseline (= 100 / #classes)
            chance = 100.0 / num_classes
            ax.axhline(chance, color="red", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
            ax.text(0.995, chance + 1.5, f"chance {chance:.1f}%", color="red", fontsize=8,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())

    head = "Layer-wise probing — " + (f"R²: {target_label}" if (is_r2 and target_label) else
                                      ("R²" if is_r2 else "val accuracy"))
    title = head + (f"\n({subtitle})" if subtitle else "")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # many curves (e.g. one per probed attribute) would bury the plot under the legend box
    # -> park it outside the axes instead of "lower right".
    lt = legend_title or ("variable" if is_r2 else "probe")
    if len(series) > 5:
        ax.legend(title=lt, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    else:
        ax.legend(title=lt, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"saved layer-vs-valacc plot -> {out_path}")
    return out_path


# ── per-attribute BEST accuracy bar chart ────────────────────────────────────────────
# The layer curve answers "where in the network"; this answers "which attributes are
# linearly decodable at all", which is the number that gets reported. One bar per
# attribute = best val accuracy over all layers, with that attribute's own random-chance
# level marked ON the bar (chance differs per attribute: 2-way 50% vs 8-way 12.5%, so a
# single reference line would be meaningless).
#
# Colors are the validated reference palette: categorical slot-1 blue for real concepts,
# muted gray for negative-control attributes. Identity never rests on color alone — the
# control bars are also labelled "(control)" and carry a legend entry.
_C_CONCEPT = "#2a78d6"   # categorical slot 1
_C_CONTROL = "#898781"   # muted ink
_C_INK = "#0b0b0b"
_C_INK2 = "#52514e"
_C_GRID = "#e1e0d9"
_C_AXIS = "#c3c2b7"

DEFAULT_CONTROL_ATTRS = ("random_label", "seed_parity")


def plot_attr_best_bar(rows, out_path, subtitle=None, controls=DEFAULT_CONTROL_ATTRS):
    """Horizontal bar chart of best-over-layers val accuracy, one bar per attribute.

    rows: list of dicts with keys
        attribute, num_classes, chance, best_val_acc, delta_over_chance, best_stage
    Sorted here by delta_over_chance (most-above-chance first).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"bar plot skipped (matplotlib unavailable): {e}")
        return None
    if not rows:
        return None

    controls = set(controls or ())
    rows = sorted(rows, key=lambda r: r["delta_over_chance"])   # barh draws bottom-up
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, 0.52 * n + 1.9))
    fig.patch.set_facecolor("white")

    ys = list(range(n))
    for y, r in zip(ys, rows):
        is_ctl = r["attribute"] in controls
        color = _C_CONTROL if is_ctl else _C_CONCEPT
        # bar height 0.62 leaves a surface gap between adjacent bars
        ax.barh(y, r["best_val_acc"], height=0.62, color=color, zorder=3)
        # this attribute's OWN chance level, marked on its own bar
        ax.plot([r["chance"], r["chance"]], [y - 0.36, y + 0.36],
                color=_C_INK, linewidth=1.6, zorder=5)
        ax.annotate(f"{r['best_val_acc']:.1f}%   chance {r['chance']:.1f}%  "
                    f"(+{r['delta_over_chance']:.1f}%p, {r['best_stage']})",
                    (r["best_val_acc"], y), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8.5, color=_C_INK2)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['attribute']}  ({r['num_classes']}-way)"
                        + ("  (control)" if r["attribute"] in controls else "")
                        for r in rows], fontsize=9.5, color=_C_INK)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("best validation accuracy over all layers (%)", color=_C_INK2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.grid(True, color=_C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_C_AXIS)
    ax.tick_params(colors=_C_INK2, length=0)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=_C_CONCEPT, label="concept attribute"),
               Line2D([0], [0], color=_C_INK, linewidth=1.6, label="random chance")]
    if any(r["attribute"] in controls for r in rows):
        handles.insert(1, Patch(facecolor=_C_CONTROL, label="negative control"))
    # below the axes: a legend inside the plot area would sit on top of the shortest
    # bars' end-labels (the control rows, which sort to the bottom).
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles), frameon=False, fontsize=8.5)

    title = "Linear probing — best val accuracy per attribute"
    ax.set_title(title + (f"\n({subtitle})" if subtitle else ""),
                 color=_C_INK, fontsize=11, loc="left")
    # bbox_inches='tight': the end-of-bar labels are drawn OUTSIDE the axes (past x=100),
    # which tight_layout does not account for — this expands the canvas to include them.
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved per-attribute bar chart -> {out_path}")
    return out_path
