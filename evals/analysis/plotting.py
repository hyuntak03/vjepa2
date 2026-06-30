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
                       metric="accuracy", target_label=None, pez=None):
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
        pts = sorted(series[probe_label], key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_y += ys
        (line,) = ax.plot(xs, ys, linewidth=2, label=probe_label)  # 선만 (마커 없음)
        color = line.get_color()
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
        ex = _elbow_x(xs, ys)
        if ex is not None and ex != bx:
            ey = dict(zip(xs, ys))[ex]
            elbow_label = stage_by_x.get(ex, f"L{ex}")
            ax.axvline(ex, color=color, linestyle=":", alpha=0.4, linewidth=1)
            ax.scatter([ex], [ey], s=75, marker="D", facecolors="none",
                       edgecolors=color, linewidths=1.4, zorder=5)
            ax.annotate(f"elbow: {elbow_label} ({vfmt(ey)})", (ex, ey),
                        textcoords="offset points", xytext=(6, -16), fontsize=8,
                        fontweight="bold", ha="left", color=color)

    if stage_by_x:
        xs_sorted = sorted(stage_by_x)
        ax.set_xticks(xs_sorted)
        ax.set_xticklabels([stage_by_x[x] for x in xs_sorted], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("stage")
    else:
        ax.set_xlabel("encoder layer (0-indexed block)")

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
    ax.legend(title=("variable" if is_r2 else "probe"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"saved layer-vs-valacc plot -> {out_path}")
    return out_path
