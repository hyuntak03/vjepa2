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
    _plot(out, os.path.join(outdir, "attention_distance.png"), ctx.plot_pez,
          subtitle=f"vjepa | {n} val batches")


def _plot(out, path, pez, subtitle=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional; never break the analysis
        logger.warning(f"[attention_distance] plot skipped (matplotlib unavailable): {e}")
        return
    import statistics

    sd = out["spatial_distance"]  # [L][H] attention-weighted spatial distance (patches)
    L = out["num_layers"]
    depth = (L - 1) or 1
    xs = [layer / depth for layer in range(L)]                 # layer fraction 0..1
    dbar = [sum(row) / len(row) for row in sd]                 # mean over heads
    spread = [statistics.pstdev(row) if len(row) > 1 else 0.0 for row in sd]  # std over heads

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
    logger.info(f"[attention_distance] plot -> {path}")
