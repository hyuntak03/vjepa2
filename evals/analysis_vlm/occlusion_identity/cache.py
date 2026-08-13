"""On-disk token cache (one .npy memmap per base sequence).

Attentive probes need the token SET, so we cannot cache pooled vectors. Tokens are big,
so we (a) store fp16 and (b) store BASE sequences rather than per-source slices, letting
overlapping windows share storage -- e.g. `occ24__target`, `future__target` and
`full__target` are all views into the single `target` base.

Sizing, for ViT-L @ 256px / patch 16 / tubelet 2 (S=256, D=1024), 40-frame clips:
    target            (40/2)*256 = 5120 tokens ->  10.5 MB/video
    ctx_masked        (32/2)*256 = 4096        ->   8.4 MB/video
    predictor          (8/2)*256 = 1024        ->   2.1 MB/video
    isolated:<8f>      (8/2)*256 = 1024        ->   2.1 MB/video each
-> ~25 MB/video, ~29 GB for the full 1164-video set. Put `cache_dir` somewhere with room.

Extraction writes each batch straight into the memmaps, so peak RAM stays at one batch.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np


class TokenCache:
    def __init__(self, cache_dir: str, tag: str, video_ids: Sequence[str],
                 base_counts: Dict[str, int], embed_dim: int, dtype: str = "float16"):
        self.dir = os.path.join(cache_dir, tag)
        self.video_ids = list(video_ids)
        self.base_counts = dict(base_counts)
        self.embed_dim = int(embed_dim)
        self.dtype = dtype
        self.meta_path = os.path.join(self.dir, "meta.json")

    # ---- layout ------------------------------------------------------------
    def _path(self, base: str) -> str:
        return os.path.join(self.dir, base.replace(":", "_") + ".npy")

    def _meta(self) -> dict:
        return {"video_ids": self.video_ids, "base_counts": self.base_counts,
                "embed_dim": self.embed_dim, "dtype": self.dtype}

    def nbytes(self) -> int:
        per = sum(self.base_counts.values()) * self.embed_dim
        return per * len(self.video_ids) * np.dtype(self.dtype).itemsize

    # ---- read / write ------------------------------------------------------
    def matches(self) -> bool:
        """True iff a complete cache for exactly this video list + base set exists."""
        if not os.path.exists(self.meta_path):
            return False
        try:
            m = json.load(open(self.meta_path))
        except json.JSONDecodeError:
            return False
        if (m.get("video_ids") != self.video_ids
                or m.get("embed_dim") != self.embed_dim
                or m.get("dtype") != self.dtype):
            return False
        # a superset of bases is fine; every base we need must be present and sized right
        have = m.get("base_counts", {})
        return all(have.get(b) == n and os.path.exists(self._path(b))
                   for b, n in self.base_counts.items())

    def open_write(self) -> Dict[str, np.memmap]:
        os.makedirs(self.dir, exist_ok=True)
        mm = {
            b: np.lib.format.open_memmap(
                self._path(b), mode="w+", dtype=self.dtype,
                shape=(len(self.video_ids), n, self.embed_dim))
            for b, n in self.base_counts.items()
        }
        return mm

    def finalize(self, mm: Dict[str, np.memmap]) -> None:
        for v in mm.values():
            v.flush()
        json.dump(self._meta(), open(self.meta_path, "w"))

    def open_read(self) -> Dict[str, np.memmap]:
        return {b: np.load(self._path(b), mmap_mode="r") for b in self.base_counts}
