"""
LLaVA-Video-7B-Qwen2 vision encoder, loaded WITHOUT the 7B LLM.

Verified loading recipe (workflow vlm-encoder-verify):
  * vision tower weights double-nested under 'model.vision_tower.vision_tower.*';
    projector (mlp2x_gelu) under 'model.mm_projector.*'.
  * build SigLipVisionModel, drop the last encoder layer (-> 26), head=Identity,
    then load_state_dict. Feature tap = vision_model(...).hidden_states[-1].

Run in the `lmms_eval_llavavideo` conda env (LLaVA-NeXT repo on PYTHONPATH).

Selectable feature stages (config: analysis.stages, or "all"):
  layer_<i>                   (B, T*729, 1152)   per SigLIP layer i in 0..25 (hidden_states[i+1])
  after_vision_encoder        (B, T*729, 1152)   == layer_25 (final encoder output)
  after_projector             (B, T*729, 3584)   projector applied to the final encoder output
  after_vision_encoder_pool2  (B, T*196, 1152)   2x bilinear spatial pool
  after_projector_pool2       (B, T*196, 3584)
  "all" -> [layer_0 .. layer_25, after_projector]

API contract (consumed by evals.analysis_vlm.eval):
  .stages : list[str] (RESOLVED, "all" already expanded); .embed_dims : list[int]
  forward(frames_list) -> list[Tensor(B, N, D)] aligned with .stages
    frames_list: list of B tensors, each (T, H, W, C) uint8 (raw RGB frames)
"""

import glob
import logging
import math
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.analysis_vlm.loadutil import resolve_model_dir

logger = logging.getLogger()

_DEFAULT_REPO = "/data/hyuntak/project/2026/vlm_direction/LLaVA-NeXT"
_NAMED = {
    "after_vision_encoder": 1152,
    "after_projector": 3584,
    "after_vision_encoder_pool2": 1152,
    "after_projector_pool2": 3584,
}


def _stage_dim(name):
    if name in _NAMED:
        return _NAMED[name]
    if re.fullmatch(r"layer_\d+", name):
        return 1152
    raise ValueError(f"unknown LLaVA stage {name!r}")


def init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs):
    # config knobs (model_kwargs.wrapper_kwargs):
    #   pretrained / cache_dir : weight location (HF repo id + cache, or local dir)
    #   llava_repo             : path to the LLaVA-NeXT source repo (SigLip code)
    #   out_stages             : list of stages, or the string "all"
    #   spatial_pool_stride    : bilinear 2x-pool stride for *_pool2 stages
    #   dtype                  : encoder forward dtype (float16 on GPU, float32 for CPU)
    # lmms-eval args conv_template / video_decode_backend / mm_spatial_pool_mode (bilinear,
    #   used) / max_frames_num (= data.frames_per_clip) / force_sample (= uniform sampling,
    #   always on) do not affect vision-only feature extraction.
    wk = dict(wrapper_kwargs or {})
    repo = wk.get("llava_repo", _DEFAULT_REPO)
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    snap = resolve_model_dir(checkpoint, wk)
    stages_cfg = wk.get("out_stages") or ["after_vision_encoder", "after_projector"]
    dtype = getattr(torch, wk.get("dtype", "float16"))
    return LLaVAVideoEncoder(snap, stages_cfg, pool_stride=wk.get("spatial_pool_stride", 2),
                             num_temporal=int(frames_per_clip), dtype=dtype)


class LLaVAVideoEncoder(nn.Module):
    def __init__(self, snap, stages_cfg, pool_stride=2, num_temporal=8, dtype=torch.float16):
        super().__init__()
        from safetensors import safe_open

        from llava.model.multimodal_encoder.siglip_encoder import (
            SigLipImageProcessor,
            SigLipVisionConfig,
            SigLipVisionModel,
        )

        vis_pfx, proj_pfx = "model.vision_tower.vision_tower.", "model.mm_projector."
        vis_sd, proj_sd = {}, {}
        for shard in sorted(glob.glob(os.path.join(snap, "model-*-of-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(vis_pfx):
                        vis_sd[k[len(vis_pfx):]] = f.get_tensor(k).float()
                    elif k.startswith(proj_pfx):
                        proj_sd[k[len(proj_pfx):]] = f.get_tensor(k).float()
        if not vis_sd:
            raise RuntimeError(f"no vision-tower weights ('{vis_pfx}*') found in {snap}")

        vt = SigLipVisionModel(SigLipVisionConfig())
        del vt.vision_model.encoder.layers[-1:]
        vt.vision_model.head = nn.Identity()
        miss, unexp = vt.load_state_dict(vis_sd, strict=False)
        assert not miss and not unexp, f"vision tower key mismatch: missing={miss} unexpected={unexp}"
        self.vision_model = vt.vision_model
        self.num_layers = len(self.vision_model.encoder.layers)  # 26

        self.projector = nn.Sequential(nn.Linear(1152, 3584), nn.GELU(), nn.Linear(3584, 3584))
        self.projector.load_state_dict(proj_sd, strict=True)
        self.image_processor = SigLipImageProcessor()

        self.stages = self._resolve_stages(stages_cfg)
        self.pool_stride = pool_stride
        self.embed_dims = [_stage_dim(s) for s in self.stages]
        self.embed_dim = self.embed_dims[0]
        self.tubelet_size = 1
        self.num_temporal = int(num_temporal)   # per-frame encoder -> T temporal positions
        self._need_proj = any("projector" in s for s in self.stages)

        self.to(dtype)
        logger.info(f"LLaVAVideoEncoder ready | {len(self.stages)} stages "
                    f"(layers=0..{self.num_layers-1}) dims={self.embed_dims} dtype={dtype}")

    # Stage config forms (config: analysis.stages):
    #   structured dict (preferred):
    #       {vision_encoder: [5,11,25] | "all", after_projector: true,
    #        after_vision_encoder_pool2: true, after_projector_pool2: true}
    #     -> only `vision_encoder` carries a per-layer selection; the others are toggles.
    #   shorthand: "all" == {vision_encoder: all};  [int,...] == {vision_encoder: [...]}
    #   legacy: [concrete stage name strings] (e.g. ["after_vision_encoder","after_projector"])
    _TOGGLES = ["after_vision_encoder", "after_projector",
                "after_vision_encoder_pool2", "after_projector_pool2"]

    def _layer_stage(self, i):
        if not (0 <= int(i) < self.num_layers):
            raise ValueError(f"vision_encoder layer {i} out of range 0..{self.num_layers - 1}")
        return f"layer_{int(i)}"

    def _resolve_stages(self, spec):
        if isinstance(spec, str):
            spec = {"vision_encoder": "all"} if spec.lower() == "all" else {"vision_encoder": [spec]}
        if isinstance(spec, (list, tuple)):
            if all(isinstance(x, int) for x in spec):
                spec = {"vision_encoder": list(spec)}
            else:  # legacy: list of concrete stage-name strings
                out = [str(s) for s in spec]
                for s in out:
                    _stage_dim(s)
                    m = re.fullmatch(r"layer_(\d+)", s)
                    if m:
                        self._layer_stage(int(m.group(1)))
                return out
        assert isinstance(spec, dict), f"analysis.stages must be a dict/list/'all', got {type(spec)}"

        out = []
        ve = spec.get("vision_encoder")
        if ve not in (None, False):
            layers = range(self.num_layers) if (isinstance(ve, str) and ve.lower() == "all") else ve
            out += [self._layer_stage(i) for i in layers]
        for key in self._TOGGLES:
            if spec.get(key):
                out.append(key)
        if not out:
            raise ValueError("analysis.stages selected nothing (set vision_encoder and/or a toggle)")
        return out

    @property
    def _dev(self):
        return next(self.parameters()).device

    @property
    def _dt(self):
        return next(self.parameters()).dtype

    def _pool2d(self, x):
        nf, n, d = x.shape
        hw = int(round(math.sqrt(n)))
        out = math.ceil(hw / self.pool_stride)
        x = x.view(nf, hw, hw, d).permute(0, 3, 1, 2)
        x = F.interpolate(x.float(), size=[out, out], mode="bilinear", align_corners=False).to(x.dtype)
        return x.permute(0, 2, 3, 1).reshape(nf, out * out, d)

    @torch.no_grad()
    def forward(self, frames_list):
        dev, dt = self._dev, self._dt
        b = len(frames_list)
        t = int(frames_list[0].shape[0])

        pv = []
        for frames in frames_list:
            arr = frames.cpu().numpy() if torch.is_tensor(frames) else frames
            proc = self.image_processor.preprocess(
                [arr[i] for i in range(arr.shape[0])], return_tensors="pt"
            )["pixel_values"]
            pv.append(proc)
        pv = torch.cat(pv, dim=0).to(dev, dt)            # (B*T, 3, 384, 384)

        # one forward gives all layer hidden states: hs[0]=embeddings, hs[i+1]=layer i output
        hs = self.vision_model(pixel_values=pv, output_hidden_states=True).hidden_states
        proj_final = self.projector(hs[-1]) if self._need_proj else None

        def _stage(name):
            m = re.fullmatch(r"layer_(\d+)", name)
            if m:
                x = hs[int(m.group(1)) + 1]
            elif name == "after_vision_encoder":
                x = hs[-1]
            elif name == "after_projector":
                x = proj_final
            elif name == "after_vision_encoder_pool2":
                x = self._pool2d(hs[-1])
            elif name == "after_projector_pool2":
                x = self._pool2d(proj_final)
            else:
                raise ValueError(name)
            bt, ntok, dx = x.shape
            return x.reshape(b, t * ntok, dx)            # (B, T*Ntok, D)

        return [_stage(s) for s in self.stages]
