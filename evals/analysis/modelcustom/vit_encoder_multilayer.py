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

    # Feature-extraction path selection (per-layer probing knob):
    #   apply_encoder_norm=False (DEFAULT): forward-hook path. Base ViT is built with
    #     out_layers=None so src/models/vision_transformer.py:205-206
    #     `outs.append(self.norm(x))` is NEVER executed. MultiLayerClipAggregation
    #     registers hooks on each requested block and returns the RAW residual x_N.
    #     Semantically: "what does layer N actually encode?"
    #
    #   apply_encoder_norm=True: stock VJEPA2 path. out_layers is passed to the ViT
    #     ctor, which triggers line 205-206 and returns self.norm(x_N) per intermediate.
    #     self.norm is the FINAL LayerNorm whose gain/bias were trained on the LAST
    #     block's distribution. Applying it to every intermediate is a CONCAT-wrapper
    #     convention (evals/video_classification_frozen/modelcustom/
    #     vit_encoder_multiclip_multilevel.py); reactivated here as an ABLATION knob
    #     to test the hypothesis that the paper (Joseph 2026 Fig 1) shape (dramatic
    #     Peak → L23 degradation) comes from this normalization artifact.
    apply_encoder_norm = bool(wrapper_kwargs.get("apply_encoder_norm", False))
    logger.info(f"apply_encoder_norm={apply_encoder_norm}  "
                f"({'stock self.norm-per-intermediate (Meta concat convention)' if apply_encoder_norm else 'raw x_N via forward hooks'})")

    _vit_out_layers = out_layers if apply_encoder_norm else None
    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip,
        out_layers=_vit_out_layers, **enc_kwargs,
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
        apply_encoder_norm=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.out_layers = list(out_layers) if out_layers is not None else []
        self.apply_encoder_norm = apply_encoder_norm

        # Forward hooks: only registered in the RAW x_N path (apply_encoder_norm=False).
        # When True, we consume the ViT's built-in out_layers return (already self.norm-ed)
        # and skip hooks entirely.
        self._captured = []
        self._hook_handles = []
        if not self.apply_encoder_norm:
            for idx in self.out_layers:
                h = model.blocks[idx].register_forward_hook(self._make_capture_hook())
                self._hook_handles.append(h)

        # 1D-temporal pos-embedding (same option as the stock wrapper)
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _make_capture_hook(self):
        def _hook(_module, _input, output):
            self._captured.append(output)
        return _hook

    def forward(self, x, clip_indices=None):
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along the batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        if self.apply_encoder_norm:
            # Stock ViT path: model was built with out_layers=<list>, so its forward
            # returns a LIST of self.norm(x_N) for each requested layer, in the SAME
            # order as out_layers (src/models/vision_transformer.py:198-209).
            layer_outputs = self.model(x)
            if not isinstance(layer_outputs, list):
                layer_outputs = [layer_outputs]
            assert len(layer_outputs) == len(self.out_layers), (
                f"stock-out_layers mismatch: got {len(layer_outputs)} outputs for "
                f"{len(self.out_layers)} requested layers"
            )
        else:
            # Raw x_N path: base ViT runs normally, hooks capture the residual stream
            # after each requested block. Hooks fire in block-execution order (0..depth-1)
            # regardless of out_layers ordering, so we reorder to match out_layers[i].
            self._captured = []
            _ = self.model(x)  # discard the final (self.norm-ed) return
            assert len(self._captured) == len(self.out_layers), (
                f"forward-hook capture mismatch: got {len(self._captured)} outputs for "
                f"{len(self.out_layers)} requested layers"
            )
            exec_order = sorted(range(len(self.out_layers)), key=lambda i: self.out_layers[i])
            pos_of = {oi: pi for pi, oi in enumerate(exec_order)}
            layer_outputs = [self._captured[pos_of[i]] for i in range(len(self.out_layers))]

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
