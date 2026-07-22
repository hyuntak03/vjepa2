"""Load the parabolic possible/impossible dataset.

Layout (produced by data_gen/make_parabolic_dataset.py):
    <root>/metadata.csv                       # one row per (scene, variant)
    <root>/videos/scene_XXXXXX_{possible,higher,frozen}.mp4

Every variant of a scene shares identical context frames (0..ctx_end). Because mp4
is lossy, we do NOT rely on decoded byte-equality: instead we splice the DECODED
context of `context_variant` (default 'possible') onto every variant's future, so the
context fed to the predictor and to each target encoding is literally identical.
"""
from __future__ import annotations

import csv
import os

import torch

try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def _decode(path: str, n_frames: int) -> torch.Tensor:
    vr = decord.VideoReader(path, num_threads=1)
    idx = list(range(min(n_frames, len(vr))))
    arr = vr.get_batch(idx).asnumpy()                         # (T, H, W, C) uint8
    return torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0   # (C, T, H, W)


class ParabolicScenes:
    """Indexable set of scenes; item i -> (scene_id, {variant: clip}, meta_dict).

    Each clip is (C, T, H, W), ImageNet-normalized, with the shared context spliced in.
    """

    def __init__(self, root: str, variants=("possible", "higher", "frozen"),
                 context_variant: str = "possible"):
        assert decord is not None, "decord is required (pip install decord)"
        self.root = root
        self.variants = tuple(variants)
        self.context_variant = context_variant
        assert context_variant in self.variants

        rows = list(csv.DictReader(open(os.path.join(root, "metadata.csv"))))
        self.n_frames = int(rows[0]["n_frames"])
        self.ctx_frames = int(rows[0]["ctx_end"]) + 1          # frames 0..ctx_end
        self.scenes: dict = {}
        for r in rows:
            self.scenes.setdefault(int(r["scene_id"]), {})[r["variant"]] = r
        self.ids = sorted(self.scenes)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        sid = self.ids[i]
        sc = self.scenes[sid]
        frames = {v: _decode(os.path.join(self.root, sc[v]["file"]), self.n_frames)
                  for v in self.variants}
        pctx = frames[self.context_variant][:, : self.ctx_frames]     # shared context (C, ctx, H, W)
        clips = {}
        for v in self.variants:
            clip = torch.cat([pctx, frames[v][:, self.ctx_frames:]], dim=1)   # ctx + variant future
            clips[v] = (clip - IMAGENET_MEAN) / IMAGENET_STD
        meta = dict(sc[self.context_variant])
        return sid, clips, meta
