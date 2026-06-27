"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

Analysis encoder wrapper: returns features from MULTIPLE encoder layers
SEPARATELY (one tensor per layer), so that an independent probe can be attached
to each layer for layer-wise analysis.

This differs from:
  - vit_encoder_multiclip.py            -> returns only the LAST layer
  - vit_encoder_multiclip_multilevel.py -> CONCATENATES several layers into one
                                           long token sequence (single probe)

forward(x, clip_indices) returns:
    List[Tensor],  len == len(out_layers),  each Tensor of shape (B, N, D)
    in the same order as `out_layers`.
"""

import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    # -- which encoder blocks to read features from (0-indexed, output is AFTER
    #    that block). e.g. for vit_large (depth 24): last layer == 23.
    out_layers = wrapper_kwargs.get("out_layers")
    if out_layers is None:
        raise ValueError("analysis encoder requires wrapper_kwargs.out_layers (list of int)")

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, out_layers=out_layers, **enc_kwargs
    )

    pretrained_dict = checkpoint[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")

    # wrapper_kwargs may carry out_layers; ClipAggregation consumes the rest.
    # `dtype` (optional) is a compute-dtype knob (e.g. float32 on CPU); pop it here
    # and cast the assembled module rather than passing it to the wrapper ctor.
    agg_kwargs = {k: v for k, v in wrapper_kwargs.items() if k not in ("out_layers", "dtype")}
    model = MultiLayerClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        out_layers=out_layers,
        **agg_kwargs,
    )
    dtype = wrapper_kwargs.get("dtype")
    if dtype is not None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = model.to(dtype=torch_dtype)
    del checkpoint
    return model


class MultiLayerClipAggregation(nn.Module):
    """
    Process each clip independently, then for EACH requested encoder layer
    aggregate the multi-clip / multi-view tokens into a single (B, N, D) tensor.
    Returns a list over layers.
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        out_layers=None,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.out_layers = out_layers

        # 1D-temporal pos-embedding (same option as the stock wrapper)
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along the batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        layer_outputs = self.model(x)  # list over layers, each (B', N, D)
        if not isinstance(layer_outputs, list):
            layer_outputs = [layer_outputs]

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outs in enumerate(all_outputs):
                outs = [o.reshape(B, T, S, D) for o in outs]
                outs = torch.cat(outs, dim=1).flatten(1, 2)  # concat along temporal dim
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)
                    pos_embed = torch.cat(pos_embed, dim=1)
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)
                    pos_embed = pos_embed.flatten(1, 2)
                    outs = outs + pos_embed
                all_outputs[i] = outs
            # concat the (default 1) spatial views along the token dim -> (B, N, D)
            return torch.cat(all_outputs, dim=1)

        return [multiviews_postprocess(lo) for lo in layer_outputs]
