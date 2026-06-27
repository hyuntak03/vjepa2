# -----------------------------------------------------------------------------
# Raw-frame video dataloader for VLM-encoder probing.
#
# The VLM vision encoders (LLaVA-Video SigLIP, Qwen3-VL ViT) each require their
# OWN native preprocessing (SigLIP 384 resize / Qwen smart-resize + patch pack),
# so unlike the V-JEPA path we do NOT normalize/clip here. We just uniformly
# sample `frames_per_clip` raw RGB frames per video and hand them to the backend,
# which preprocesses natively.
#
# CSV format (identical to the V-JEPA path): "<abs_mp4_path> <int_label>",
# space-delimited, no header.
#
# Collate keeps frames as a LIST of (T,H,W,C) uint8 tensors (videos may differ
# in spatial resolution); the backend resizes each natively. Labels are stacked.
# -----------------------------------------------------------------------------

import logging

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger()


class RawVideoDataset(Dataset):
    """Uniformly sample `frames_per_clip` raw frames (uint8 THWC) from each video."""

    def __init__(self, csv_path, frames_per_clip, decord_threads=2):
        try:
            data = pd.read_csv(csv_path, header=None, delimiter=" ")
        except pd.errors.ParserError:
            data = pd.read_csv(csv_path, header=None, delimiter="::")
        self.paths = list(data.values[:, 0])
        self.labels = list(data.values[:, 1])
        self.T = int(frames_per_clip)
        self.decord_threads = decord_threads
        logger.info(f"RawVideoDataset: {len(self.paths)} videos from {csv_path} (T={self.T})")

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        vr = VideoReader(self.paths[i], num_threads=self.decord_threads, ctx=cpu(0))
        n = len(vr)
        # uniform sampling across the whole clip; repeats gracefully if n < T
        idx = np.linspace(0, max(n - 1, 0), self.T).round().astype(np.int64)
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        return torch.from_numpy(frames)

    def __getitem__(self, i):
        for _ in range(8):
            try:
                return self._load(i), int(self.labels[i])
            except Exception as e:  # corrupt/missing video -> try another sample
                logger.warning(f"failed to load {self.paths[i]} ({e}); retrying with a random sample")
                i = np.random.randint(len(self))
        raise RuntimeError("RawVideoDataset: could not load any valid video after retries")


def _collate(batch):
    frames = [b[0] for b in batch]  # list of (T,H,W,C) uint8 tensors (variable H,W)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return frames, labels


class _UnpaddedShardSampler(torch.utils.data.Sampler):
    """Contiguous-free strided shard (rank::world_size) with NO padding -> the union over
    ranks is EXACTLY the dataset, each sample once. Used for eval / cache pre-pass so that
    AllReduceSum'd accuracy is exact (DistributedSampler pads with duplicates, skewing it)."""

    def __init__(self, n, world_size, rank):
        self.indices = list(range(rank, n, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_raw_dataloader(
    csv_path,
    frames_per_clip,
    batch_size,
    world_size,
    rank,
    training,
    num_workers=6,
    decord_threads=2,
    persistent=None,
):
    # persistent_workers: keep workers alive across iterations. Good for per-epoch training
    # loaders, but for a one-shot cache pre-pass set persistent=False so workers are released
    # before the next split's loader spawns (avoids worker pile-up / spawn deadlock).
    if persistent is None:
        persistent = num_workers > 0
    ds = RawVideoDataset(csv_path, frames_per_clip, decord_threads=decord_threads)
    sampler = None
    if world_size > 1:
        if training:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
            )
        else:
            # eval / cache pre-pass: exact unpadded sharding (no duplicate-padding skew)
            sampler = _UnpaddedShardSampler(len(ds), world_size, rank)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(training and sampler is None),
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=False,
        drop_last=False,
        persistent_workers=(persistent and num_workers > 0),
        # prefetch more so decode (CPU) overlaps the now-fast SDPA encode in the cache pre-pass
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    return loader, sampler
