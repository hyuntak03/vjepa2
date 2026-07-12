# -----------------------------------------------------------------------------
# IntPhys2 metrics: pairwise accuracy + single-video AUROC + breakdowns.
#
# The IntPhys2 paper defines TWO evaluation protocols (Appendix D, "Evaluating
# prediction-based models"):
#
#   1. PAIRWISE ACCURACY (main protocol; Table 2/3 uses this)
#      Within each SceneIndex there are two pairs -- (1_Possible, 1_Impossible)
#      and (2_Possible, 2_Impossible) -- matched so that the possible video of
#      one pair becomes the impossible video of the other.
#      For each of the 2 pairs, the model is "correct" iff
#          surprise(impossible) > surprise(possible).
#      Chance = 50%. This is what the paper reports as "Easy/Medium/Hard/Overall".
#
#   2. SINGLE-VIDEO AUROC (harder; Fig. 4 middle)
#      Score each video with its aggregated surprise and compute AUROC where
#      "impossible" is the positive class. No pairing.
#
# Both are computed here. We also emit breakdowns by whichever CSV columns the
# caller asks for (condition / Difficulty / Camera / env, ...).
#
# Aggregation over sliding windows (paper Eq. 2): mean or max. Configurable.
#
# Aggregation over CONTEXT-LENGTH SWEEP (paper D.3, Table 8):
#   For each context C in the swept set (V-JEPA 2 default: {4,6,8,10,12,14}),
#   compute the whole metric independently, then report the MAX accuracy across
#   the sweep. This is the protocol behind the numbers in Table 2. Individual
#   per-C numbers are also emitted for diagnosis.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------- data structures ---------------------------------


@dataclass
class VideoRecord:
    """One row of results for a single (video, context_length) evaluation."""

    row_index: int
    scene_index: int
    pair_id: int
    is_impossible: bool
    condition: str
    difficulty: str
    camera: str
    env: str
    context_length: int
    surprise_avg: float
    surprise_max: float
    n_windows: int


def to_dataframe(records: Sequence[VideoRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "row_index", "scene_index", "pair_id", "is_impossible",
            "condition", "difficulty", "camera", "env",
            "context_length", "surprise_avg", "surprise_max", "n_windows",
        ])
    return pd.DataFrame([r.__dict__ for r in records])


# --------------------------- pairwise accuracy -------------------------------


def pairwise_accuracy(df: pd.DataFrame, surprise_col: str) -> Dict[str, float]:
    """Fraction of (SceneIndex, pair_id) pairs where impossible > possible on `surprise_col`.

    A pair is counted only if BOTH the possible and the impossible video are present
    (and non-NaN). Ties count as 0.5 (Wilcoxon convention -- matches the AUROC math).
    """
    # Drop videos whose surprise is NaN (e.g. failed to load).
    good = df[df[surprise_col].notna()].copy()
    if good.empty:
        return {"pairwise_accuracy": float("nan"), "n_pairs": 0, "n_correct": 0, "n_tied": 0}

    n_correct = 0
    n_tied = 0
    n_pairs = 0
    for (scene, pair), rows in good.groupby(["scene_index", "pair_id"]):
        pos = rows[~rows["is_impossible"]]
        imp = rows[rows["is_impossible"]]
        if len(pos) != 1 or len(imp) != 1:
            # Missing video in this pair -- skip. (Log at DEBUG so runs w/ many drops aren't spammy.)
            logger.debug(f"scene {scene} pair {pair}: incomplete "
                         f"(pos={len(pos)}, imp={len(imp)}); skipping")
            continue
        s_pos = float(pos.iloc[0][surprise_col])
        s_imp = float(imp.iloc[0][surprise_col])
        if s_imp > s_pos:
            n_correct += 1
        elif s_imp == s_pos:
            n_tied += 1
        n_pairs += 1

    if n_pairs == 0:
        return {"pairwise_accuracy": float("nan"), "n_pairs": 0, "n_correct": 0, "n_tied": 0}
    acc = (n_correct + 0.5 * n_tied) / n_pairs
    return {
        "pairwise_accuracy": float(acc),
        "n_pairs": int(n_pairs),
        "n_correct": int(n_correct),
        "n_tied": int(n_tied),
    }


def pairwise_accuracy_breakdown(
    df: pd.DataFrame, surprise_col: str, by: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Pairwise accuracy computed independently per group of `by` values.

    Grouping happens BEFORE pairing -- both members of a pair must share the same
    group key (which is how the IntPhys2 CSV is laid out: pair members always share
    condition / Difficulty / Camera / env).
    """
    out: Dict[str, Dict[str, float]] = {}
    if not by:
        return out
    for key, sub in df.groupby(list(by)):
        if not isinstance(key, tuple):
            key = (key,)
        tag = "|".join(f"{c}={v}" for c, v in zip(by, key))
        out[tag] = pairwise_accuracy(sub, surprise_col)
    return out


# --------------------------- cross-pair (IntPhys1) ---------------------------


def cross_pair_accuracy(df: pd.DataFrame, surprise_col: str) -> Dict[str, float]:
    """IntPhys1-style pairwise accuracy: within each quadruplet (scene_index)
    enumerate ALL 2*2 (possible, impossible) cross-pairs and count how often
    surprise(imp) > surprise(pos).

    The paper's IntPhys1 protocol reports pairwise accuracy across every
    (possible, impossible) pair *inside* a quadruplet, not against a fixed pair-id
    matching (IntPhys1's 4 runs are not laid out as 2 canonical pairs the way
    IntPhys2's `type` column labels them). With 2 possible + 2 impossible per
    quadruplet, this is 4 cross-pairs per scene * N scenes total.
    """
    good = df[df[surprise_col].notna()].copy()
    if good.empty:
        return {"pairwise_accuracy": float("nan"), "n_pairs": 0, "n_correct": 0, "n_tied": 0}

    n_correct = n_tied = n_pairs = 0
    for _, sub in good.groupby("scene_index"):
        pos = sub[~sub["is_impossible"]][surprise_col].to_numpy()
        imp = sub[sub["is_impossible"]][surprise_col].to_numpy()
        if len(pos) == 0 or len(imp) == 0:
            continue
        for p in pos:
            for i in imp:
                if i > p:
                    n_correct += 1
                elif i == p:
                    n_tied += 1
                n_pairs += 1

    if n_pairs == 0:
        return {"pairwise_accuracy": float("nan"), "n_pairs": 0, "n_correct": 0, "n_tied": 0}
    acc = (n_correct + 0.5 * n_tied) / n_pairs
    return {
        "pairwise_accuracy": float(acc),
        "n_pairs": int(n_pairs),
        "n_correct": int(n_correct),
        "n_tied": int(n_tied),
    }


def cross_pair_accuracy_breakdown(
    df: pd.DataFrame, surprise_col: str, by: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not by:
        return out
    for key, sub in df.groupby(list(by)):
        if not isinstance(key, tuple):
            key = (key,)
        tag = "|".join(f"{c}={v}" for c, v in zip(by, key))
        out[tag] = cross_pair_accuracy(sub, surprise_col)
    return out


# --------------------------- single-video AUROC ------------------------------


def single_video_auroc(df: pd.DataFrame, surprise_col: str) -> Dict[str, float]:
    """AUROC treating `is_impossible` as positive, `surprise_col` as the score.

    Wilcoxon-Mann-Whitney statistic: for each (pos, neg) pair, count 1 for
    score_pos > score_neg, 0.5 for ties, 0 otherwise. AUC = mean over pairs.
    """
    good = df[df[surprise_col].notna()].copy()
    if good.empty:
        return {"auroc": float("nan"), "n_pos": 0, "n_neg": 0}
    pos = good[good["is_impossible"]][surprise_col].to_numpy()
    neg = good[~good["is_impossible"]][surprise_col].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return {"auroc": float("nan"), "n_pos": int(len(pos)), "n_neg": int(len(neg))}

    # Rank-based AUROC (no external sklearn dep).
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # average-rank ties
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1  # 1-indexed average rank
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    r_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    auroc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return {"auroc": float(auroc), "n_pos": int(n_pos), "n_neg": int(n_neg)}


# --------------------------- context-length sweep ----------------------------


def sweep_max_pairwise(
    df: pd.DataFrame, surprise_col: str, breakdown_by: Sequence[str],
    pairing: str = "type_matched",
) -> Dict[str, object]:
    """Paper's context-sweep protocol: MAX pairwise accuracy across context lengths.

    `df` must contain rows for MULTIPLE context_length values. We compute the whole
    pairwise-accuracy table for each C and pick the C that maximises overall pairwise
    accuracy; breakdowns and AUROC are reported for THAT SAME best C for consistency
    with the paper (Table 2: "best hyper-parameters found across models").

    `pairing`:
      * "type_matched" (default) -- IntPhys2 protocol: groupby (scene_index, pair_id)
        and require exactly 1 possible + 1 impossible per group.
      * "cross_pair"             -- IntPhys1 protocol: enumerate all 2*2 cross-pairs
        per quadruplet (scene_index).
    """
    pair_fn = pairwise_accuracy if pairing == "type_matched" else cross_pair_accuracy
    break_fn = (
        pairwise_accuracy_breakdown if pairing == "type_matched" else cross_pair_accuracy_breakdown
    )
    per_c: Dict[int, Dict[str, object]] = {}
    for C, sub in df.groupby("context_length"):
        overall = pair_fn(sub, surprise_col)
        break_by = break_fn(sub, surprise_col, breakdown_by)
        auc = single_video_auroc(sub, surprise_col)
        per_c[int(C)] = {"overall": overall, "breakdown": break_by, "auroc": auc}

    if not per_c:
        return {"best_context_length": None, "per_context_length": {}, "best": None}

    # Pick the context length that maximises overall pairwise accuracy.
    best_C = max(
        per_c.keys(),
        key=lambda c: (
            per_c[c]["overall"]["pairwise_accuracy"]
            if per_c[c]["overall"]["pairwise_accuracy"] == per_c[c]["overall"]["pairwise_accuracy"]  # not NaN
            else -1.0
        ),
    )
    return {
        "best_context_length": int(best_C),
        "per_context_length": per_c,
        "best": per_c[best_C],
    }


# --------------------------- summary printer ---------------------------------


def format_summary(name: str, block: Dict[str, object], indent: int = 0) -> str:
    """Turn a metrics dict into a readable multi-line string for logs / summary.txt."""
    pad = " " * indent
    lines = [f"{pad}[{name}]"]
    if "pairwise_accuracy" in block:
        acc = block["pairwise_accuracy"]
        lines.append(
            f"{pad}  pairwise_accuracy={acc*100:.2f}%  "
            f"(correct={block['n_correct']}/{block['n_pairs']}, ties={block['n_tied']})"
        )
    if "auroc" in block:
        lines.append(f"{pad}  auroc={block['auroc']:.4f}  (n_pos={block['n_pos']}, n_neg={block['n_neg']})")
    return "\n".join(lines)
