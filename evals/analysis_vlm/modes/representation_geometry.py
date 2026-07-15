# -----------------------------------------------------------------------------
# Mode: representation_geometry
#
# Per-layer analysis of the class-conditional geometry of frozen-encoder mean-
# pooled features, used to diagnose WHY linear probing accuracy differs between
# IntPhys 1 and IntPhys 2 (both use identical encoder + preprocessing + probe
# form, so any linear-probe delta is a delta in the underlying representation).
#
# Metrics per layer (train split, since val may be too small):
#   * Fisher Discriminant Ratio (FDR)   ||μ_p - μ_i||² / (tr Σ_p + tr Σ_i)
#     -- upper bound on linear separability; higher = easier for a linear probe.
#   * Centroid cosine distance          1 - cos(μ_p, μ_i)
#     -- angular gap between class means.
#   * Per-class feature norm            mean & std of ||h||₂ within each class.
#   * Participation ratio (PR)          (Σ σ)² / Σ σ²   of the (n, D) feature
#     covariance -- the effective dimensionality of the representation. Low PR =
#     one/two dominant directions carry the signal.
#   * Class-conditional PR              PR computed within each class separately.
#
# Assumes:
#   * ctx.tr_feats, ctx.tr_labels are populated (cache_features: true, one of
#     cache_pooling: mean|max|pooled -- rejected otherwise).
#   * ctx.stages / ctx.embed_dims defined per layer.
#
# Outputs to  <folder>/representation_geometry/
#   * representation_geometry.json     -- per-layer metrics (list, indexed by stage)
#   * representation_geometry.png      -- layer-wise line plot of FDR, cosine, PR
#   * class_norms.png                  -- per-layer mean±std norm by class
# -----------------------------------------------------------------------------
import json
import logging
import os

import numpy as np
import torch

from . import register

logger = logging.getLogger(__name__)


def _pool_to_D(feats_stage: torch.Tensor, embed_dim: int, cache_pooling: str) -> torch.Tensor:
    """Squash a cached stage tensor down to (N, D) regardless of cache layout.
    * mean cache -> (N, D)   :: pass-through
    * max cache  -> (N, D)   :: pass-through
    * pooled     -> (N, 2D)  :: take the mean half (paper-faithful)
    * tokens     -> (N, T, D):: mean over the token axis (spatiotemporal mean)
    """
    x = feats_stage
    if x.dim() == 2 and x.size(-1) == embed_dim:
        return x
    if x.dim() == 2 and x.size(-1) == 2 * embed_dim:
        return x[..., :embed_dim]
    if x.dim() == 3:
        return x.mean(dim=1)
    raise ValueError(
        f"representation_geometry: unsupported feature shape {tuple(x.shape)} "
        f"for embed_dim={embed_dim}, cache_pooling={cache_pooling!r}"
    )


def _participation_ratio(x: torch.Tensor) -> float:
    """(sum_i σ_i)² / sum_i σ_i²   where σ are singular values of the centered
    (N, D) matrix. Bounded above by min(N, D). Higher = flatter spectrum =
    features spread across more effective dimensions.

    Computed on fp64 for numerical safety on small counts."""
    x = x.double()
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on the centered data. For N < D we can do gesvd on x @ x.T (small);
    # for N >= D we still just call svd on x -- both are cheap at our sizes.
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        # fallback: eigendecompose the covariance
        cov = (x.T @ x) / max(1, x.size(0) - 1)
        w = torch.linalg.eigvalsh(cov).clamp(min=0)
        s = torch.sqrt(w)
    s = s.clamp(min=0)
    num = (s.sum() ** 2)
    den = (s ** 2).sum().clamp(min=1e-30)
    return float(num / den)


def _fisher_ratio(x: torch.Tensor, y: torch.Tensor, n_classes: int = 2) -> tuple[float, float]:
    """Return (FDR, centroid_cosine_distance).

    FDR = ||μ_0 - μ_1||² / (tr Σ_0 + tr Σ_1),  tr Σ_c = Σ_i Var(h_i | c)."""
    x = x.double()
    mus = []
    tr_sigma = 0.0
    for c in range(n_classes):
        m = y == c
        if not m.any():
            return float("nan"), float("nan")
        xc = x[m]
        mus.append(xc.mean(dim=0))
        tr_sigma += float((xc.var(dim=0, unbiased=True)).sum())
    diff = mus[0] - mus[1]
    fdr = float((diff * diff).sum() / max(tr_sigma, 1e-30))
    cos = float(torch.nn.functional.cosine_similarity(
        mus[0].unsqueeze(0), mus[1].unsqueeze(0)
    ).item())
    return fdr, 1.0 - cos


def _class_norm_stats(x: torch.Tensor, y: torch.Tensor, n_classes: int = 2):
    """Per-class mean and std of the L2 norm of features."""
    x = x.double()
    norms = x.norm(dim=1)
    out = []
    for c in range(n_classes):
        m = y == c
        if not m.any():
            out.append({"count": 0, "norm_mean": float("nan"), "norm_std": float("nan")})
            continue
        nc = norms[m]
        out.append({
            "count": int(m.sum()),
            "norm_mean": float(nc.mean()),
            "norm_std": float(nc.std(unbiased=True)) if nc.numel() > 1 else 0.0,
        })
    return out


@register("representation_geometry")
def run(cfg, ctx):
    if ctx.tr_feats is None:
        raise RuntimeError(
            "representation_geometry needs cached train features "
            "(set optimization.cache_features: true, cache_pooling: mean|max|pooled)"
        )
    stages = ctx.stages
    embed_dims = ctx.embed_dims
    cache_pool = ctx.cache_pooling or "mean"
    n_classes = int(ctx.num_classes or 2)

    layers = []
    for stage_pos, stage_name in enumerate(stages):
        # cache tensors are (N, ...) already reduced by reduce_feature. Unify to (N, D).
        x = _pool_to_D(ctx.tr_feats[stage_pos], embed_dim=embed_dims[stage_pos],
                       cache_pooling=cache_pool)
        y = ctx.tr_labels
        # move to cpu fp64 for stable stats
        x = x.detach().to("cpu")
        y = y.detach().to("cpu").long()

        fdr, cos_dist = _fisher_ratio(x, y, n_classes=n_classes)
        norms = _class_norm_stats(x, y, n_classes=n_classes)
        pr_all = _participation_ratio(x)
        pr_by_class = []
        for c in range(n_classes):
            m = y == c
            pr_by_class.append(_participation_ratio(x[m]) if m.any() else float("nan"))

        # layer index for the plotter: parse from stage string when possible ("Lxx" or int-like)
        try:
            layer_int = int(str(stage_name).lstrip("L"))
        except ValueError:
            layer_int = stage_pos

        layers.append({
            "stage": str(stage_name),
            "layer": layer_int,
            "n_train": int(x.size(0)),
            "embed_dim": int(x.size(1)),
            "fisher_ratio": fdr,
            "centroid_cosine_distance": cos_dist,
            "class_norms": norms,
            "participation_ratio": pr_all,
            "participation_ratio_by_class": pr_by_class,
        })
        _norm_str = ", ".join(f"{n['norm_mean']:.2f}" for n in norms)
        logger.info(
            f"[representation_geometry] {stage_name}: FDR={fdr:.4f}  "
            f"cos_dist={cos_dist:.4f}  PR={pr_all:.1f}  norms=[{_norm_str}]"
        )

    out = {
        "num_classes": n_classes,
        "cache_pooling": cache_pool,
        "layers": layers,
    }
    outdir = os.path.join(ctx.folder, "representation_geometry")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "representation_geometry.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        f"[representation_geometry] {len(layers)} layers analyzed on "
        f"n_train={layers[0]['n_train']} samples -> {outdir}"
    )

    _plot_layerwise(out, os.path.join(outdir, "representation_geometry.png"),
                    ctx.plot_pez, subtitle=f"vjepa | n_train={layers[0]['n_train']}")
    _plot_norms(out, os.path.join(outdir, "class_norms.png"),
                subtitle=f"vjepa | n_train={layers[0]['n_train']}")


def _plot_layerwise(out, path, pez, subtitle=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"[representation_geometry] plot skipped ({e})")
        return

    Ls = [d["layer"] for d in out["layers"]]
    fdr = [d["fisher_ratio"] for d in out["layers"]]
    cos = [d["centroid_cosine_distance"] for d in out["layers"]]
    pr = [d["participation_ratio"] for d in out["layers"]]
    depth = max(Ls) if Ls else 1
    frac = [l / depth for l in Ls] if depth else Ls

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, y, ylabel, color in zip(
        axes,
        [fdr, cos, pr],
        ["Fisher discriminant ratio\n(||μ₀-μ₁||² / (tr Σ₀ + tr Σ₁))",
         "1 - cos(μ_possible, μ_impossible)",
         "Participation ratio\n(effective # of feature dims)"],
        ["#d62728", "#1f77b4", "#2ca02c"],
    ):
        ax.plot(frac, y, marker="o", linewidth=2, color=color)
        if pez:
            ax.axvspan(pez[0], pez[1], color="gray", alpha=0.15)
        ax.set_xlabel("layer fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_title("Class linear-separability")
    axes[1].set_title("Class-mean angular gap")
    axes[2].set_title("Effective dimensionality")
    fig.suptitle("Representation geometry per layer" + (f" ({subtitle})" if subtitle else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info(f"[representation_geometry] layerwise plot -> {path}")


def _plot_norms(out, path, subtitle=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"[representation_geometry] norm plot skipped ({e})")
        return

    Ls = [d["layer"] for d in out["layers"]]
    depth = max(Ls) if Ls else 1
    frac = [l / depth for l in Ls] if depth else Ls
    class_names = ("possible (y=0)", "impossible (y=1)")
    colors = ("#1f77b4", "#d62728")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, (name, col) in enumerate(zip(class_names, colors)):
        means = [d["class_norms"][c]["norm_mean"] for d in out["layers"]]
        stds = [d["class_norms"][c]["norm_std"] for d in out["layers"]]
        m = np.array(means, dtype=float)
        s = np.array(stds, dtype=float)
        ax.plot(frac, m, marker="o", linewidth=2, color=col, label=name)
        ax.fill_between(frac, m - s, m + s, alpha=0.15, color=col)
    ax.set_xlabel("layer fraction")
    ax.set_ylabel("‖h‖₂ (mean ± std)")
    ax.set_title("Class-conditional feature norms per layer" + (f" ({subtitle})" if subtitle else ""))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info(f"[representation_geometry] norms plot -> {path}")
