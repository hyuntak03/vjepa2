# -----------------------------------------------------------------------------
# Mode: attention_distance  (Appendix C.6 / Fig. 19 & Fig. 3)
#
# Per (layer, head) attention-weighted SPATIAL (patch) and TEMPORAL (tubelet)
# distance of the frozen V-JEPA2 RoPE encoder, captured via the additive
# attention_hooks SDPA patch (encoder output stays bit-identical — capture is a
# detached side computation). Writes attention_distance.json and a dual-axis plot
# of layer-mean distance (Dbar = mean over heads) and head specialization
# (S = std over heads) vs LAYER FRACTION, with PEZ shading.
#
# Paper signature to look for: Dbar DIPS and S SPIKES around one-third depth
# (the Physics Emergence Zone) as spatiotemporally-local heads emerge alongside
# the long-range heads.
# -----------------------------------------------------------------------------
import json
import logging
import os

import torch

from evals.analysis.attention_hooks import (
    AttentionDistanceCollector,
    _find_rope_attn,
    attention_hooks,
)

from . import register

logger = logging.getLogger(__name__)


@register("attention_distance")
def run(cfg, ctx):
    assert ctx.data_mode == "clip", (
        "attention_distance requires the V-JEPA clip encoder (data_mode='clip'); "
        f"got data_mode={ctx.data_mode!r}"
    )
    rope = _find_rope_attn(ctx.encoder)
    if not rope:
        raise RuntimeError("attention_distance: no RoPEAttention blocks found on the encoder")
    num_layers = len(rope)
    num_heads = int(rope[0].num_heads)
    max_batches = cfg.get("max_batches", 8)

    collector = AttentionDistanceCollector(
        num_layers=num_layers,
        num_heads=num_heads,
        query_chunk=int(cfg.get("query_chunk", 512)),
        max_batches=max_batches,
    )

    loader = ctx.make_val_clip_loader()
    n = 0
    with attention_hooks(ctx.encoder, collector=collector):
        for data in loader:
            ctx.encode_clip(data)  # encoder forward -> SDPA patch captures per-head distance
            n += 1
            if max_batches is not None and n >= max_batches:
                break

    out = collector.finalize()
    out["n_batches"] = n
    outdir = os.path.join(ctx.folder, "attention_distance")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "attention_distance.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        f"[attention_distance] captured {num_layers}x{num_heads} heads over {n} val batches -> {outdir}"
    )
    sub = f"vjepa | {n} val batches"
    _plot_heatmap(out, os.path.join(outdir, "attention_distance.png"), subtitle=sub,
                  annotate=cfg.get("annotate", True))
    _plot_layerwise(out, os.path.join(outdir, "attention_distance_layerwise.png"), ctx.plot_pez, subtitle=sub)


def _plot_heatmap(out, path, subtitle=None, annotate=True):
    """Paper Fig. 3 style: per-head attention-locality heatmap, x=Layer, y=Attention Head,
    colour = spatial distance (patches). Blue, LOW distance = DARK (cmap 'Blues_r'), so the
    unusually-local heads that emerge at the PEZ stand out as dark cells."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        logger.warning(f"[attention_distance] heatmap skipped (matplotlib/numpy unavailable): {e}")
        return
    Z = np.array(out["spatial_distance"], dtype=float).T   # (H, L): rows=head, cols=layer
    H, L = Z.shape
    vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    mid = 0.5 * (vmin + vmax)

    fig, ax = plt.subplots(figsize=(max(9, L * 0.44), max(4.5, H * 0.34)))
    im = ax.imshow(Z, aspect="auto", cmap="Blues_r", origin="lower",  # low dist = dark blue
                   vmin=vmin, vmax=vmax, extent=[-0.5, L - 0.5, -0.5, H - 0.5])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Head")
    ax.set_xticks(range(L)); ax.set_xticklabels(range(L), fontsize=6)
    ax.set_yticks(range(H)); ax.set_yticklabels(range(H), fontsize=6)
    if annotate:
        for h in range(H):
            for lyr in range(L):
                v = Z[h, lyr]
                ax.text(lyr, h, f"{v:.1f}", ha="center", va="center", fontsize=4.2,
                        color=("white" if v < mid else "#222"))  # dark cell -> white text
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Distance (patches)")
    ax.set_title("V-JEPA v2-L: Attention Distance Per Head" + (f"  ({subtitle})" if subtitle else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"[attention_distance] heatmap -> {path}")


def _plot_layerwise(out, path, pez, subtitle=None):
    """Companion Appendix Fig. 19: dual-axis line plot of layer-mean attention distance
    (Dbar = mean over heads) and head specialization (S = std over heads) vs layer fraction."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"[attention_distance] layerwise plot skipped: {e}")
        return
    import statistics

    sd = out["spatial_distance"]
    L = out["num_layers"]
    depth = (L - 1) or 1
    xs = [layer / depth for layer in range(L)]
    dbar = [sum(row) / len(row) for row in sd]
    spread = [statistics.pstdev(row) if len(row) > 1 else 0.0 for row in sd]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pez:
        ax.axvspan(pez[0], pez[1], color="gray", alpha=0.15, zorder=0)
        ax.text((pez[0] + pez[1]) / 2, 0.99, "PEZ", color="dimgray", fontsize=8,
                ha="center", va="top", transform=ax.get_xaxis_transform())
    c1, c2 = "tab:red", "tab:blue"
    ax.plot(xs, dbar, color=c1, lw=2, marker="o", ms=3, label="attention distance (mean over heads)")
    ax.set_xlabel("layer fraction")
    ax.set_ylabel("attention distance (patches)", color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, spread, color=c2, lw=2, ls="--", marker="s", ms=3,
             label="head specialization (std over heads)")
    ax2.set_ylabel("head specialization (patches)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper center", fontsize=8)
    ax.set_title("Attention distance & head specialization" + (f"\n({subtitle})" if subtitle else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info(f"[attention_distance] layerwise plot -> {path}")
