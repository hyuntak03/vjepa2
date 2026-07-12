# -----------------------------------------------------------------------------
# Optional visualizations for the IntPhys2 surprise harness.
#
# Two figure families are supported (both default-off unless the YAML asks):
#
#   1) surprise-vs-time curves per scene (paper Fig. 9 in the debug set):
#      plots per-window surprise for each of the 4 videos in a scene, with
#      possible/impossible colour-coded, so you can eyeball where the model
#      diverges.
#
#   2) pairwise-accuracy bar chart / breakdown table:
#      simple bar plot per breakdown axis (condition / Difficulty / Camera).
#
# Kept isolated so import-time failure of matplotlib does not sink the harness.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # noqa
        logger.warning(f"matplotlib not available ({e}); plots will be skipped")
        return None


def plot_scene_surprise(
    scene_index: int,
    traces: Dict[str, Dict],  # {type_str -> {"window_starts": np.ndarray, "surprise": np.ndarray, ...}}
    out_path: str,
    context_length: int,
) -> None:
    """Fig 9 companion: per-window surprise for one scene's 4 videos."""
    plt = _try_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    for type_str, trace in sorted(traces.items()):
        starts = np.asarray(trace["window_starts"])
        s = np.asarray(trace["surprise"])
        color = "tab:red" if "Impossible" in type_str else "tab:blue"
        ls = "-" if type_str.startswith("1_") else "--"
        ax.plot(starts, s, color=color, linestyle=ls, marker="o", markersize=3, label=type_str)
    ax.set_xlabel("window start frame")
    ax.set_ylabel(f"surprise (C={context_length})")
    ax.set_title(f"IntPhys2 scene {scene_index} — per-window surprise")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_breakdown_bars(
    breakdown: Dict[str, Dict[str, float]],
    out_path: str,
    title: str = "IntPhys2 pairwise accuracy",
) -> None:
    """Bar plot of pairwise accuracy per breakdown key."""
    plt = _try_matplotlib()
    if plt is None or not breakdown:
        return
    keys = list(breakdown.keys())
    accs = [
        breakdown[k].get("pairwise_accuracy", float("nan")) * 100
        for k in keys
    ]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(keys)), 3.5), dpi=120)
    x = np.arange(len(keys))
    ax.bar(x, accs, color="tab:blue", edgecolor="k", linewidth=0.5)
    ax.axhline(50.0, color="k", linestyle="--", linewidth=0.7, label="chance")
    ax.set_ylim(0, 100)
    ax.set_ylabel("pairwise accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_context_sweep(
    per_context_length: Dict[int, Dict[str, object]],
    out_path: str,
) -> None:
    """One curve: pairwise accuracy vs context length (paper's D.3 hyperparam sweep)."""
    plt = _try_matplotlib()
    if plt is None or not per_context_length:
        return
    Cs = sorted(per_context_length.keys())
    accs = [per_context_length[c]["overall"]["pairwise_accuracy"] * 100 for c in Cs]
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.plot(Cs, accs, marker="o", color="tab:blue")
    ax.axhline(50.0, color="k", linestyle="--", linewidth=0.7, label="chance")
    ax.set_xlabel("context length C (frames)")
    ax.set_ylabel("pairwise accuracy (%)")
    ax.set_title("IntPhys2 context-length sweep")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
