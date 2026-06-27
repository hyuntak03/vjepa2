"""
Qwen3-VL-4B-Instruct vision encoder, loaded WITHOUT the 4B LLM.

Verified loading recipe (workflow vlm-encoder-verify):
  * build ONLY Qwen3VLVisionModel(Qwen3VLConfig.from_pretrained(snap).vision_config).
  * load just 'model.visual.*' tensors. Real 4B vision config:
    hidden_size=1024, out_hidden_size=2560, depth=24, deepstack_visual_indexes=[5,11,17].

Run in the `lmms_eval_py311_2.7` conda env (transformers qwen3_vl).

Selectable feature stages (config: analysis.stages, or "all"):
  block_<i>       (B, grid_t*gh*gw,         1024)   per vision block i in 0..23 (raw, pre-merge; via hook)
  before_merger   (B, grid_t*gh*gw,         1024)   == block_23 (merger input)
  after_merger    (B, grid_t*(gh/2)*(gw/2), 2560)   image_embeds fed to the LLM
  deepstack_<i>   (B, grid_t*(gh/2)*(gw/2), 2560)   merged deepstack feature, i in [5,11,17]
  "all" -> [block_0 .. block_23, after_merger]

Resolution / token-count control (wrapper_kwargs):
  resize_mode=smart (default): processor smart_resize within [min_pixels, max_pixels]
     (lmms-eval default 8192 / 112896). Uniform-resolution videos -> uniform N -> batchable.
  resize_mode=fixed: pre-resize to (qwen_fixed_h, qwen_fixed_w) (multiples of 32), do_resize=False.

API contract (consumed by evals.analysis_vlm.eval):
  .stages : list[str] (RESOLVED, "all" expanded); .embed_dims : list[int]
  forward(frames_list) -> list[Tensor(B, N, D)] aligned with .stages
    frames_list: list of B tensors, each (T, H, W, C) uint8 (raw RGB frames)
"""

import glob
import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.analysis_vlm.loadutil import resolve_model_dir

logger = logging.getLogger()


def init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs):
    # config knobs (model_kwargs.wrapper_kwargs):
    #   pretrained / cache_dir   : weight location (HF repo id + cache, or local dir)
    #   out_stages               : list of stages, or the string "all"
    #   resize_mode              : "smart" (min/max_pixels) | "fixed" (qwen_fixed_h/w)
    #   min_pixels / max_pixels  : smart-resize budget (lmms-eval 8192 / 112896)
    #   qwen_fixed_h/w           : fixed resolution (multiples of 32) for resize_mode=fixed
    #   attn_implementation      : "sdpa" (default) | "eager" | "flash_attention_2"
    #   dtype                    : float16 on GPU, float32 for CPU
    wk = dict(wrapper_kwargs or {})
    snap = resolve_model_dir(checkpoint, wk)
    stages_cfg = wk.get("out_stages") or ["before_merger", "after_merger"]
    dtype = getattr(torch, wk.get("dtype", "float16"))
    return Qwen3VLEncoder(
        snap, stages_cfg,
        resize_mode=wk.get("resize_mode", "smart"),
        min_pixels=int(wk.get("min_pixels", 8192)),
        max_pixels=int(wk.get("max_pixels", 112896)),
        fixed_h=wk.get("qwen_fixed_h"), fixed_w=wk.get("qwen_fixed_w"),
        attn_impl=wk.get("attn_implementation", "sdpa"),
        frames_per_clip=int(frames_per_clip),
        dtype=dtype,
    )


class Qwen3VLEncoder(nn.Module):
    def __init__(self, snap, stages_cfg, resize_mode="smart", min_pixels=8192, max_pixels=112896,
                 fixed_h=None, fixed_w=None, attn_impl="sdpa", frames_per_clip=8, dtype=torch.float16):
        super().__init__()
        from safetensors import safe_open
        from transformers import AutoVideoProcessor, Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        cfg = Qwen3VLConfig.from_pretrained(snap)
        vcfg = cfg.vision_config
        vcfg._attn_implementation = attn_impl
        self.visual = Qwen3VLVisionModel(vcfg)

        pfx, state = "model.visual.", {}
        for sh in sorted(glob.glob(os.path.join(snap, "model-*.safetensors"))):
            with safe_open(sh, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(pfx):
                        state[k[len(pfx):]] = f.get_tensor(k).float()
        if not state:
            raise RuntimeError(f"no vision weights ('{pfx}*') found in {snap}")
        miss, unexp = self.visual.load_state_dict(state, strict=False)
        assert not miss and not unexp, f"visual key mismatch: missing={miss} unexpected={unexp}"

        self.merge = int(vcfg.spatial_merge_size)            # 2
        self.patch = int(vcfg.patch_size)                    # 16
        self.tpatch = int(getattr(vcfg, "temporal_patch_size", 2))
        self.depth = int(vcfg.depth)                         # 24
        self.deepstack_idx = list(getattr(vcfg, "deepstack_visual_indexes", []))
        self.hid, self.out = int(vcfg.hidden_size), int(vcfg.out_hidden_size)  # 1024, 2560

        self.stages = self._resolve_stages(stages_cfg)
        self.embed_dims = [self._stage_dim(s) for s in self.stages]
        self.embed_dim = self.embed_dims[0]
        self.tubelet_size = self.tpatch
        self.num_temporal = int(frames_per_clip) // self.tpatch   # grid_t temporal positions

        # which raw block outputs (1024-dim, pre-merge) we must capture via hooks
        self._need_blocks = set()
        for s in self.stages:
            m = re.fullmatch(r"block_(\d+)", s)
            if m:
                self._need_blocks.add(int(m.group(1)))
            elif s == "before_merger":
                self._need_blocks.add(self.depth - 1)
        self._captured = {}
        for i in self._need_blocks:
            self.visual.blocks[i].register_forward_hook(self._mk_hook(i))

        # resize / processor
        self.resize_mode = resize_mode
        if resize_mode == "fixed":
            assert fixed_h and fixed_w, "resize_mode=fixed needs qwen_fixed_h / qwen_fixed_w"
            step = self.patch * self.merge
            assert int(fixed_h) % step == 0 and int(fixed_w) % step == 0, \
                f"qwen_fixed_h/w must be multiples of patch*merge={step}"
            self.fixed_h, self.fixed_w = int(fixed_h), int(fixed_w)
            self.video_processor = AutoVideoProcessor.from_pretrained(snap)
        else:
            self.fixed_h = self.fixed_w = None
            self.video_processor = AutoVideoProcessor.from_pretrained(
                snap, min_pixels=min_pixels, max_pixels=max_pixels
            )
            for attr, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
                if hasattr(self.video_processor, attr):
                    setattr(self.video_processor, attr, val)

        self.to(dtype)
        logger.info(
            f"Qwen3VLEncoder ready | {len(self.stages)} stages (blocks=0..{self.depth-1}) "
            f"dims={self.embed_dims} resize={resize_mode} pixels=[{min_pixels},{max_pixels}] "
            f"fixed=({fixed_h}x{fixed_w}) attn={attn_impl} deepstack={self.deepstack_idx} dtype={dtype}"
        )

    def _stage_dim(self, name):
        if name in ("after_merger",) or name.startswith("deepstack_"):
            return self.out
        if name == "before_merger" or re.fullmatch(r"block_\d+", name):
            return self.hid
        raise ValueError(f"unknown Qwen stage {name!r}")

    # Stage config forms (config: analysis.stages):
    #   structured dict (preferred):
    #       {vision_encoder: [5,11,23] | "all", before_merger: true, after_merger: true,
    #        deepstack: [5,11,17] | "all" | true}
    #     -> only `vision_encoder` carries a per-block selection; others are toggles
    #        (deepstack additionally accepts a subset list of its merge indexes).
    #   shorthand: "all" == {vision_encoder: all};  [int,...] == {vision_encoder: [...]}
    #   legacy: [concrete stage name strings] (e.g. ["before_merger","after_merger","deepstack_5"])
    def _block_stage(self, i):
        if not (0 <= int(i) < self.depth):
            raise ValueError(f"vision_encoder block {i} out of range 0..{self.depth - 1}")
        return f"block_{int(i)}"

    def _resolve_stages(self, spec):
        if isinstance(spec, str):
            spec = {"vision_encoder": "all"} if spec.lower() == "all" else {"vision_encoder": [spec]}
        if isinstance(spec, (list, tuple)):
            if all(isinstance(x, int) for x in spec):
                spec = {"vision_encoder": list(spec)}
            else:  # legacy: concrete stage-name strings
                out = [str(s) for s in spec]
                for s in out:
                    self._stage_dim(s)
                    m = re.fullmatch(r"block_(\d+)", s)
                    if m:
                        self._block_stage(int(m.group(1)))
                    if s.startswith("deepstack_") and int(s.split("_")[1]) not in self.deepstack_idx:
                        raise ValueError(f"{s}: deepstack indexes are {self.deepstack_idx}")
                return out
        assert isinstance(spec, dict), f"analysis.stages must be a dict/list/'all', got {type(spec)}"

        out = []
        ve = spec.get("vision_encoder")
        if ve not in (None, False):
            blocks = range(self.depth) if (isinstance(ve, str) and ve.lower() == "all") else ve
            out += [self._block_stage(i) for i in blocks]
        if spec.get("before_merger"):
            out.append("before_merger")
        if spec.get("after_merger"):
            out.append("after_merger")
        ds = spec.get("deepstack")
        if ds not in (None, False):
            idxs = self.deepstack_idx if (ds is True or (isinstance(ds, str) and ds.lower() == "all")) else ds
            for i in idxs:
                if int(i) not in self.deepstack_idx:
                    raise ValueError(f"deepstack {i} not in {self.deepstack_idx}")
                out.append(f"deepstack_{int(i)}")
        if not out:
            raise ValueError("analysis.stages selected nothing (set vision_encoder and/or a toggle)")
        return out

    def _mk_hook(self, i):
        def hook(_m, _inp, out):
            self._captured[i] = out[0] if isinstance(out, tuple) else out
        return hook

    @property
    def _dev(self):
        return next(self.parameters()).device

    @property
    def _dt(self):
        return next(self.parameters()).dtype

    def _preprocess(self, frames):
        if self.resize_mode == "fixed":
            t = frames.permute(0, 3, 1, 2).float()
            t = F.interpolate(t, size=(self.fixed_h, self.fixed_w), mode="bilinear", align_corners=False)
            vid = t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            out = self.video_processor(videos=[vid], do_resize=False, return_tensors="pt")
        else:
            vid = frames.cpu().numpy()
            out = self.video_processor(videos=[vid], return_tensors="pt")
        return out["pixel_values_videos"], out["video_grid_thw"]

    @torch.no_grad()
    def forward(self, frames_list):
        dev, dt = self._dev, self._dt
        pvs, grids = [], []
        for frames in frames_list:
            frames = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
            pv, grid = self._preprocess(frames)
            pvs.append(pv)
            grids.append(grid)
        pixel_values = torch.cat(pvs, dim=0).to(dev, dt)
        grid_thw = torch.cat(grids, dim=0).to(dev)           # (B, 3)

        self._captured = {}
        image_embeds, deepstack = self.visual(pixel_values, grid_thw=grid_thw)

        merge2 = self.merge ** 2
        after_sizes = (grid_thw.prod(-1) // merge2).tolist()
        before_sizes = grid_thw.prod(-1).tolist()

        def _stack(parts, stage):
            try:
                return torch.stack(list(parts), dim=0)
            except RuntimeError as e:
                raise RuntimeError(
                    f"stage '{stage}': videos produced different token counts "
                    f"({[p.shape[0] for p in parts]}); happens under resize_mode=smart with "
                    f"differing input resolutions. Use resize_mode=fixed + qwen_fixed_h/w."
                ) from e

        outs = []
        for name in self.stages:
            if name == "after_merger":
                parts = torch.split(image_embeds, after_sizes, dim=0)
            elif name == "before_merger":
                parts = torch.split(self._captured[self.depth - 1], before_sizes, dim=0)
            elif re.fullmatch(r"block_\d+", name):
                parts = torch.split(self._captured[int(name.split("_")[1])], before_sizes, dim=0)
            elif name.startswith("deepstack_"):
                di = int(name.split("_")[1])
                parts = torch.split(deepstack[self.deepstack_idx.index(di)], after_sizes, dim=0)
            else:
                raise ValueError(name)
            outs.append(_stack(parts, name))
        return outs
