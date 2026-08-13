"""Blender occlusion dataset grouped by scene, for predictor-surprise scoring.

Layout (same as the probing eval):
    <root>/index.csv                 # one row per (scene, variant)
    <root>/videos/scene_XXXXXX_<variant>.mp4      # 40 frames @ 8fps, 256x256

Each scene holds 4 variants that SHARE the context frames (1..32) and differ only after
`divergence_frame` (33..36):

    possible     none              물체가 그대로 다시 나온다
    imp_shape    identity_shape    모양이 바뀌어 나온다
    imp_color    identity_colour   색이 바뀌어 나온다
    imp_vanish   permanence        아예 다시 나오지 않는다

Because the context is shared, the predictor sees exactly ONE input per scene -- so
`z_pred` is identical across variants (verified: max deviation 0.0 over the whole set).
Every surprise difference therefore comes purely from the target encode. That is a
cleaner setup than IntPhys, where the impossible clip has its own context.

`splice_context` re-attaches the context of `context_variant` onto every variant after
decoding. On this dataset the decoded contexts are already identical, so it is a no-op;
it is kept on as a guarantee against lossy-decode drift.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

import torch

try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def _decode(path: str, n_frames: int) -> torch.Tensor:
    """-> (C, T, H, W) float in [0, 1]."""
    vr = decord.VideoReader(path, num_threads=1)
    assert len(vr) >= n_frames, f"{path}: {len(vr)} frames < requested {n_frames}"
    arr = vr.get_batch(list(range(n_frames))).asnumpy()
    return torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0


class OcclusionScenes:
    """Indexable set of scenes; item i -> (scene_id, {variant: clip}, meta).

    Each clip is (C, T, H, W), ImageNet-normalized.
    """

    def __init__(
        self,
        root: str,
        index_csv: str = "index.csv",
        *,
        variants: Sequence[str] = ("possible", "imp_shape", "imp_color", "imp_vanish"),
        context_variant: str = "possible",
        n_frames: int = 40,
        splice_context: bool = True,
        context_length: int = 32,
        split: Optional[str] = None,          # None = 모든 scene (학습이 없으므로 기본값)
        limit_scenes: Optional[int] = None,
    ):
        assert decord is not None, "decord is required (pip install decord)"
        self.root = root
        self.variants = tuple(variants)
        self.context_variant = context_variant
        assert context_variant in self.variants, (
            f"context_variant {context_variant!r} not in variants {self.variants}")
        self.n_frames = int(n_frames)
        self.splice_context = bool(splice_context)
        self.context_length = int(context_length)

        rows = [r for r in csv.DictReader(open(os.path.join(root, index_csv)))
                if r["variant"] in self.variants]
        if split:
            rows = [r for r in rows if r["split"] == split]

        self.scenes: Dict[int, dict] = {}
        for r in rows:
            self.scenes.setdefault(int(r["scene_id"]), {})[r["variant"]] = r
        # keep only scenes that carry every requested variant
        complete = {k: v for k, v in self.scenes.items() if len(v) == len(self.variants)}
        self.dropped = len(self.scenes) - len(complete)
        self.scenes = complete
        self.ids: List[int] = sorted(self.scenes)
        if limit_scenes is not None:
            self.ids = self.ids[:limit_scenes]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        sid = self.ids[i]
        sc = self.scenes[sid]
        frames = {v: _decode(os.path.join(self.root, sc[v]["file"]), self.n_frames)
                  for v in self.variants}
        clips = {}
        if self.splice_context:
            ctx = frames[self.context_variant][:, : self.context_length]
            for v in self.variants:
                clip = torch.cat([ctx, frames[v][:, self.context_length:]], dim=1)
                clips[v] = (clip - IMAGENET_MEAN) / IMAGENET_STD
        else:
            for v in self.variants:
                clips[v] = (frames[v] - IMAGENET_MEAN) / IMAGENET_STD

        row = sc[self.context_variant]
        meta = {k: row.get(k, "") for k in
                ("scene_id", "split", "divergence_frame", "reappear_frame", "entry_frame",
                 "n_hidden_frames", "shape", "color", "obj_px")}
        meta["variant_violation"] = {v: sc[v].get("violation", "") for v in self.variants}
        return sid, clips, meta

    def summary(self) -> str:
        extra = f" (variant 누락으로 {self.dropped} scene 제외)" if self.dropped else ""
        return (f"{len(self)} scenes x {len(self.variants)} variants "
                f"= {len(self) * len(self.variants)} videos{extra} | "
                f"frames={self.n_frames} context={self.context_length} "
                f"| context_variant={self.context_variant} splice={self.splice_context}")
