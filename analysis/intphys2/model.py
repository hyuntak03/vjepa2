# -----------------------------------------------------------------------------
# V-JEPA(2) encoder + predictor loader for IntPhys2 surprise scoring.
#
# The V-JEPA 2 checkpoint (single .pth or .pt) stores three state_dicts:
#     state_dict["encoder"]         : online encoder (theta)
#     state_dict["target_encoder"]  : EMA teacher   (theta_bar)
#     state_dict["predictor"]       : latent-space predictor p
#
# During training the JEPA loss is (see app/vjepa/train.py:440-450):
#     h = target_encoder(clip)                          # full clip -> all tokens
#     h = LayerNorm(h)                                  # per-token, over feature dim
#     z = encoder(clip, masks_enc)                      # ONLY context tokens
#     z = predictor(z, masks_enc, masks_pred)           # predict target tokens
#     loss = mean(|z_pred - h_masked|^loss_exp) / loss_exp   # ~ L1 (loss_exp=1) or SmoothL1
#
# For zero-shot IntPhys eval we reuse exactly this recipe, treating the loss as the
# per-window "surprise" scalar. The only zero-shot convention is which state_dict key
# feeds each role -- both `encoder` and `target_encoder` can be legally used for either
# role; the standard analysis practice (matching `evals/analysis/**`) is EMA-only
# (target_encoder for both). We expose the choice via YAML.
#
# We build the ViT with `num_frames = window_size` so RoPE grid_depth matches the
# temporal length we actually feed at inference. Because V-JEPA 2 uses RoPE (no
# learned pos_embed), the pretrained state_dict is agnostic to `num_frames` -- you
# can change the window from the pretrained fpc without re-training.
#
# All model / predictor / arch knobs are YAML-driven; see build_from_config().
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import src.models.predictor as vit_predictor_mod
import src.models.vision_transformer as vit_mod

logger = logging.getLogger(__name__)


# Encoder factory names live in `src/models/vision_transformer.py`.
# Only V-JEPA 2 (RoPE) variants are used at eval time.
_ARCH_FACTORY = {
    "vit_large": "vit_large",
    "vit_huge": "vit_huge",
    "vit_giant": "vit_giant_xformers",
    "vit_giant_xformers": "vit_giant_xformers",
}


@dataclass
class VJEPA2Bundle:
    """A frozen V-JEPA 2 encoder + predictor ready for surprise scoring."""

    encoder: nn.Module
    predictor: nn.Module
    context_encoder: nn.Module  # alias into `encoder` unless dual-encoder mode is used
    target_encoder: nn.Module  # alias into `encoder` unless dual-encoder mode is used
    img_size: int
    patch_size: int
    tubelet_size: int
    num_frames: int  # == window_size (frames) at eval time
    embed_dim: int
    device: torch.device
    dtype: torch.dtype

    # Grid dimensions for the FULL M-frame window this bundle was built for.
    @property
    def num_temporal_tokens(self) -> int:
        return self.num_frames // self.tubelet_size

    @property
    def num_spatial_tokens(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def num_tokens(self) -> int:
        return self.num_temporal_tokens * self.num_spatial_tokens


def _clean_backbone_key(state_dict):
    """Strip DDP/backbone prefixes so state_dict keys match the freshly constructed module."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        cleaned[k] = v
    return cleaned


def _load_checkpoint(path: str) -> dict:
    logger.info(f"Loading V-JEPA 2 checkpoint from {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_encoder(
    *,
    arch_name: str,
    img_size: int,
    patch_size: int,
    tubelet_size: int,
    num_frames: int,
    use_rope: bool = True,
    uniform_power: bool = False,
    use_sdpa: bool = True,
    extra: Optional[dict] = None,
) -> nn.Module:
    factory_name = _ARCH_FACTORY.get(arch_name)
    if factory_name is None:
        raise ValueError(
            f"unknown encoder arch '{arch_name}'; valid: {sorted(_ARCH_FACTORY)}"
        )
    factory = getattr(vit_mod, factory_name)
    kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_rope=use_rope,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    if extra:
        kwargs.update(extra)
    return factory(**kwargs)


def _build_predictor(
    *,
    img_size: int,
    patch_size: int,
    tubelet_size: int,
    num_frames: int,
    encoder_embed_dim: int,
    predictor_embed_dim: int = 384,
    predictor_depth: int = 12,
    predictor_num_heads: int = 12,
    num_mask_tokens: int = 10,
    use_rope: bool = True,
    uniform_power: bool = False,
    use_sdpa: bool = True,
    extra: Optional[dict] = None,
) -> nn.Module:
    kwargs = dict(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_mask_tokens=True,
        embed_dim=encoder_embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        num_mask_tokens=num_mask_tokens,
        use_rope=use_rope,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=False,
        wide_silu=True,
    )
    if extra:
        kwargs.update(extra)
    return vit_predictor_mod.vit_predictor(**kwargs)


def _load_state_dict(module: nn.Module, sub_state: dict, tag: str, strict: bool = False):
    cleaned = _clean_backbone_key(sub_state)
    # Drop non-matching-shape keys quietly (e.g. pos_embed when the pretrained clip
    # was fpc64 and we're now building for fpc48; RoPE mode does not use these).
    model_state = module.state_dict()
    to_load = {}
    for k, v in cleaned.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            logger.warning(
                f"[{tag}] shape mismatch on key '{k}': "
                f"pretrained {tuple(v.shape)} vs model {tuple(model_state[k].shape)}; skipping"
            )
            continue
        to_load[k] = v
    missing = set(model_state) - set(to_load)
    unexpected = set(cleaned) - set(model_state)
    msg = module.load_state_dict(to_load, strict=strict)
    if missing:
        logger.info(f"[{tag}] {len(missing)} keys not in checkpoint (kept init): first 3 = {list(missing)[:3]}")
    if unexpected:
        logger.info(f"[{tag}] {len(unexpected)} unexpected keys in checkpoint (dropped): first 3 = {list(unexpected)[:3]}")
    logger.info(f"[{tag}] load_state_dict: {msg}")


def build_from_config(cfg: dict, device: torch.device) -> VJEPA2Bundle:
    """Build encoder + predictor from a YAML `model:` block.

    Expected schema (all fields optional except `checkpoint`; sane V-JEPA 2 defaults):

    model:
      checkpoint: /abs/path/to/model.pth                # required
      arch_name: vit_large                              # vit_large | vit_huge | vit_giant
      img_size: 256                                     # ViT-L pretrain: 256
      patch_size: 16
      tubelet_size: 2
      window_size: 48                                   # M -- must match surprise.window_size
      use_rope: true
      uniform_power: false                              # V-JEPA 2 uses false
      # Which state_dict key powers each role (both default to `target_encoder`).
      context_encoder_key: target_encoder               # target_encoder | encoder
      target_encoder_key:  target_encoder
      dual_encoder: false                               # if true, load TWO separate modules
      # Predictor knobs (V-JEPA 2 ViT-L defaults):
      predictor:
        embed_dim: 384
        depth: 12
        num_heads: 12
        num_mask_tokens: 10
      dtype: bfloat16                                   # cast the frozen graph
    """
    ckpt_path = cfg["checkpoint"]
    arch_name = cfg.get("arch_name", "vit_large")
    img_size = int(cfg.get("img_size", 256))
    patch_size = int(cfg.get("patch_size", 16))
    tubelet_size = int(cfg.get("tubelet_size", 2))
    num_frames = int(cfg.get("window_size", 48))
    use_rope = bool(cfg.get("use_rope", True))
    uniform_power = bool(cfg.get("uniform_power", False))

    context_key = cfg.get("context_encoder_key", "target_encoder")
    target_key = cfg.get("target_encoder_key", "target_encoder")
    dual_encoder = bool(cfg.get("dual_encoder", False))

    if not dual_encoder and context_key != target_key:
        # Same physical module can only serve both roles if the two YAML keys agree.
        # Force the user to opt into dual-encoder mode if they want different keys.
        raise ValueError(
            "context_encoder_key != target_encoder_key requires dual_encoder: true; "
            f"got context={context_key!r}, target={target_key!r}"
        )

    predictor_cfg = cfg.get("predictor", {}) or {}
    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name

    state = _load_checkpoint(ckpt_path)

    # ---- encoder(s) ---------------------------------------------------------
    encoder = _build_encoder(
        arch_name=arch_name, img_size=img_size, patch_size=patch_size,
        tubelet_size=tubelet_size, num_frames=num_frames,
        use_rope=use_rope, uniform_power=uniform_power,
    )
    _load_state_dict(encoder, state[target_key], tag=f"encoder<-{target_key}")
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    if dual_encoder:
        context_encoder = _build_encoder(
            arch_name=arch_name, img_size=img_size, patch_size=patch_size,
            tubelet_size=tubelet_size, num_frames=num_frames,
            use_rope=use_rope, uniform_power=uniform_power,
        )
        _load_state_dict(context_encoder, state[context_key], tag=f"context_encoder<-{context_key}")
        context_encoder = context_encoder.to(device=device, dtype=dtype).eval()
        for p in context_encoder.parameters():
            p.requires_grad_(False)
        target_encoder = encoder  # the one we already loaded from target_encoder_key
    else:
        context_encoder = encoder
        target_encoder = encoder

    # ---- predictor ----------------------------------------------------------
    predictor = _build_predictor(
        img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
        num_frames=num_frames,
        encoder_embed_dim=encoder.embed_dim,
        predictor_embed_dim=int(predictor_cfg.get("embed_dim", 384)),
        predictor_depth=int(predictor_cfg.get("depth", 12)),
        predictor_num_heads=int(predictor_cfg.get("num_heads", 12)),
        num_mask_tokens=int(predictor_cfg.get("num_mask_tokens", 10)),
        use_rope=use_rope, uniform_power=uniform_power,
    )
    _load_state_dict(predictor, state["predictor"], tag="predictor")
    predictor = predictor.to(device=device, dtype=dtype).eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    del state  # release the fp32 checkpoint copy on CPU

    return VJEPA2Bundle(
        encoder=encoder,
        predictor=predictor,
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        img_size=img_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        num_frames=num_frames,
        embed_dim=encoder.embed_dim,
        device=device,
        dtype=dtype,
    )
