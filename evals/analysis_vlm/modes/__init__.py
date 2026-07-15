# -----------------------------------------------------------------------------
# Additive post-hoc analysis modes for the analysis_vlm harness (paper repro:
# attention distance/ablation, orthogonal-probe steering, circular direction
# geometry). Everything here is reached ONLY when a config sets
# `experiment.analysis.modes`; a run without that key never imports or executes
# any of this, so existing behavior is byte-for-byte identical.
#
# A mode is a callable `fn(cfg: dict, ctx: AnalysisContext) -> None` registered by
# name via @register("<name>"). run_modes() dispatches the config's {name: cfg}
# map to them; each writes its outputs under `<folder>/<name>/`.
# -----------------------------------------------------------------------------
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_REGISTRY: "dict[str, Callable]" = {}


def register(name):
    """Decorator: register a mode implementation under `name`."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def _import_modes():
    """Import mode modules so they self-register. Kept lazy (called only from
    run_modes) so `from evals.analysis_vlm.modes import AnalysisContext` has no
    side effects and never fails on an unfinished mode file."""
    from . import attention_distance  # noqa: F401
    from . import representation_geometry  # noqa: F401
    # future phases self-register here as they are added:
    # from . import orthogonal_probe_sequence, steering, direction_tuning, attention_ablation


@dataclass
class AnalysisContext:
    """Read-only handle passed to every mode: the frozen encoder, the trained
    probe heads + their metadata, cached features (if any), the standardization
    stats needed to recover raw targets, and a couple of closures back into
    eval.py's local scope (encode a clip batch / build a val clip loader)."""
    encoder: Any
    device: Any
    folder: str
    rank: int = 0
    world_size: int = 1
    use_bfloat16: bool = False
    plot_pez: Any = None
    # task / probe state
    task: str = "regression"
    num_classes: int = 0
    heads: Any = None
    best_val: Any = None
    stages: Any = None
    embed_dims: Any = None
    reg_vars: Any = None           # [(var_name, [cols...]), ...]
    targets_t: Any = None          # standardized targets on device (indexed by CSV label)
    targets_npy: Optional[str] = None   # path to RAW (unstandardized) targets.npy
    col_mu: Any = None             # per-column nanmean used to standardize (recover raw)
    col_sd: Any = None             # per-column nanstd
    # cached features (None when cache_features=false)
    cache_pooling: Optional[str] = None
    data_mode: Optional[str] = None
    tr_feats: Any = None
    tr_labels: Any = None
    va_feats: Any = None
    va_labels: Any = None
    # closures into the eval.py scope
    encode_clip: Optional[Callable] = None          # fn(data_batch) -> encoder forward
    make_val_clip_loader: Optional[Callable] = None  # fn() -> val clip DataLoader


def run_modes(modes_cfg, ctx):
    """Dispatch `experiment.analysis.modes` (a {name: cfg} mapping) to registered
    modes. `cfg` may be True/{}/None (run with defaults), a dict (options), or
    False / {enabled: false} (skip). Unknown names raise (with the valid list)."""
    if not modes_cfg:
        return
    _import_modes()
    for name, cfg in modes_cfg.items():
        if cfg is False or (isinstance(cfg, dict) and cfg.get("enabled") is False):
            logger.info(f"[modes] '{name}' disabled -> skip")
            continue
        if name not in _REGISTRY:
            raise ValueError(f"unknown analysis mode {name!r}; valid: {sorted(_REGISTRY)}")
        cfg = {} if cfg in (True, None) else dict(cfg)
        cfg.pop("enabled", None)
        out = os.path.join(ctx.folder, name)
        os.makedirs(out, exist_ok=True)
        logger.info(f"[modes] running '{name}' -> {out}")
        _REGISTRY[name](cfg, ctx)
