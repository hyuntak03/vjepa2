# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -----------------------------------------------------------------------------
# Layer-wise validation-accuracy plot (x = encoder layer, y = val acc),
# one line per probe. Enabled by config experiment.analysis.plot.
# -----------------------------------------------------------------------------

import logging
from collections import defaultdict

logger = logging.getLogger()


def plot_layer_val_acc(heads, val_acc, out_path, subtitle=None):
    """Save a [layer x val-acc] line plot (one line per probe).

    heads:    list of dict with keys 'layer' and 'name' (name == "L<layer>_<probe>")
    val_acc:  dict head_name -> val accuracy (%)
    out_path: where to save the .png
    subtitle: extra text appended to the title (e.g. "best over 20 epochs")
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional; never break training
        logger.warning(f"plot skipped (matplotlib unavailable): {e}")
        return None

    # group (x, acc) by probe label. Prefer an explicit `probe` key (set by the VLM
    # harness); fall back to parsing `name` for the clip harness. Splitting `name`
    # on the first '_' is WRONG for VLM stage tags (e.g. "block_5_linear-mean"), hence
    # the explicit key.
    series = defaultdict(list)
    for h in heads:
        probe_label = h.get("probe") or (h["name"].split("_", 1)[1] if "_" in h["name"] else h["name"])
        series[probe_label].append((h["layer"], val_acc[h["name"]]))

    # optional per-x stage labels (VLM): use the stage name for ticks/annotations
    stage_by_x = {h["layer"]: h["stage"] for h in heads if h.get("stage") is not None}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for probe_label in sorted(series):
        pts = sorted(series[probe_label], key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=2, label=probe_label)  # 선만 (마커 없음)
        # annotate best stage per probe
        bx, by = max(pts, key=lambda t: t[1])
        ax.annotate(f"{stage_by_x.get(bx, f'L{bx}')}:{by:.1f}", (bx, by),
                    textcoords="offset points", xytext=(0, 6), fontsize=8)

    if stage_by_x:
        xs_sorted = sorted(stage_by_x)
        ax.set_xticks(xs_sorted)
        ax.set_xticklabels([stage_by_x[x] for x in xs_sorted], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("stage")
    else:
        ax.set_xlabel("encoder layer (0-indexed block)")
    ax.set_ylabel("val accuracy (%)")
    title = "Layer-wise probing — val accuracy"
    if subtitle:
        title += f"\n({subtitle})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="probe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"saved layer-vs-valacc plot -> {out_path}")
    return out_path
