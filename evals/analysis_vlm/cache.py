# -----------------------------------------------------------------------------
# Frozen-encoder FEATURE CACHE for probing.
#
# The encoder is frozen and our preprocessing is deterministic (no augmentation),
# so each video's features are IDENTICAL every epoch -> re-encoding per epoch is
# pure waste. Instead we run the encoder ONCE (a single deterministic pre-pass),
# cache the per-stage features in RAM, then train the probes for many epochs over
# the cache (no decode, no encoder forward) -> epochs become ~seconds.
#
# Two cache granularities (config: optimization.cache_pooling):
#   "pooled"  -> store [mean ‖ max] over tokens, shape (n, 2D) per stage. TINY.
#                Works with LINEAR probes only (attentive needs the token set).
#                Ideal for the all-layer `vision_encoder: all` linear scan.
#   "tokens"  -> store the full (N, D) token tensor per stage. Works with ALL
#                probe types incl. attentive, but large (scales with N * #stages).
#
# DDP: each rank pre-passes its OWN fixed shard (shuffle=False DistributedSampler)
# and caches it; training then shuffles locally. Metrics all-reduce as usual.
# -----------------------------------------------------------------------------

import logging
import queue
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger()

_PREFETCH_END = object()


class _ThreadPrefetcher:
    """Background-THREAD prefetch over a num_workers=0 loader. A daemon thread pulls
    (decodes) the next batch into a small queue while the main thread runs the GPU
    encode; decord releases the GIL during decode, so decode overlaps encode. Unlike
    num_workers>0 there are NO subprocess workers -> none of the spawn / worker-respawn
    deadlocks that bite the cache pre-pass at the train->val loader transition."""

    def __init__(self, loader, depth=2):
        self._q = queue.Queue(maxsize=max(1, depth))
        self._err = None
        try:
            self._len = len(loader)
        except TypeError:  # IterableDataset / bare iterator -> tqdm falls back to total=None
            self._len = None
        self._thread = threading.Thread(
            target=self._run, args=(loader,), daemon=True, name="cache-prefetch"
        )
        self._thread.start()

    def _run(self, loader):
        try:
            for batch in loader:
                self._q.put(batch)
        except Exception as e:  # surface decode errors to the consuming thread
            self._err = e
        finally:
            self._q.put(_PREFETCH_END)

    def __len__(self):
        if self._len is None:
            raise TypeError("prefetched loader length unknown")
        return self._len

    def __iter__(self):
        while True:
            batch = self._q.get()
            if batch is _PREFETCH_END:
                if self._err is not None:
                    raise self._err
                return
            yield batch


def reduce_feature(feat, mode, num_temporal=None):
    """feat: (B, N, D).
      'tokens'    -> (B, N, D)                       full tokens (all probes; large)
      'pooled'    -> (B, 2D) = [mean ‖ max] over N   global pool (linear only; tiny)
      'framewise' -> (B, T, D) spatial-mean per frame (keeps temporal; needs num_temporal;
                     supports linear/framewise/attentive over the T per-frame vectors; small)
    """
    if mode == "tokens":
        return feat
    if mode == "pooled":
        return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=-1)
    if mode == "framewise":
        if num_temporal is None:
            raise ValueError("cache_pooling='framewise' needs num_temporal (VLM backends only)")
        b, n, d = feat.shape
        if n % num_temporal != 0:
            raise ValueError(f"framewise cache: token count {n} not divisible by num_temporal={num_temporal}")
        s = n // num_temporal
        return feat.view(b, num_temporal, s, d).mean(dim=2)   # (B, T, D)
    raise ValueError(f"unknown cache_pooling {mode!r} (expected 'tokens' | 'pooled' | 'framewise')")


@torch.no_grad()
def build_feature_cache(encode_fn, loader, cache_pooling, num_temporal=None, max_gb=None,
                        label="cache", rank=0):
    """One deterministic pre-pass: encode every sample, reduce, accumulate on CPU (fp16).

    encode_fn(data) -> (feats: list[(B,N,D)] per stage, labels: (B,), bsz)
    returns: (feats_cat: list[stage] of (n_local, ...) fp16 CPU, labels: (n_local,) long CPU)

    Aborts up front (after the first batch) if the estimated per-rank RAM exceeds max_gb,
    so a `cache_pooling='tokens'` selection with many stages/tokens can't silently OOM.
    """
    try:
        n_target = len(loader.sampler) if loader.sampler is not None else len(loader.dataset)
    except Exception:
        n_target = None

    # overlap CPU decode with the GPU encode WITHOUT subprocess workers (the pre-pass loader
    # is num_workers=0): a daemon thread decodes the next batch while we encode the current one.
    src = _ThreadPrefetcher(loader, depth=2)
    iterator = src
    if rank == 0:
        try:
            from tqdm import tqdm

            iterator = tqdm(src, desc=f"{label} (encode)", dynamic_ncols=True,
                            mininterval=2.0, leave=False)  # tqdm reads len(src) itself
        except Exception:
            pass

    per_stage, labels_acc, n = None, [], 0
    for data in iterator:
        feats, labels, bsz = encode_fn(data)
        reduced = [reduce_feature(f, cache_pooling, num_temporal).half().cpu() for f in feats]
        if per_stage is None:
            per_stage = [[] for _ in reduced]
            per_sample_mb = sum(r[:1].element_size() * r[:1].nelement() for r in reduced) / 1024.0**2
            if n_target:
                est_gb = per_sample_mb * n_target / 1024.0
                logger.info(f"[{label}] estimated cache ~{est_gb:.1f} GB "
                            f"({n_target} samples x {per_sample_mb:.2f} MB, pooling={cache_pooling})")
                if max_gb and est_gb > max_gb:
                    raise RuntimeError(
                        f"[{label}] estimated feature cache {est_gb:.0f} GB exceeds cache_max_gb={max_gb} "
                        f"(per-rank host RAM). Reduce stages/frames, shard across more GPUs, use "
                        f"cache_pooling='pooled' (linear probes only), or raise optimization.cache_max_gb."
                    )
        for si, r in enumerate(reduced):
            per_stage[si].append(r)
        labels_acc.append(labels.cpu())
        n += bsz
    if per_stage is None:
        raise RuntimeError("feature cache pre-pass saw 0 samples")
    try:
        feats_cat = [torch.cat(s, dim=0) for s in per_stage]
    except RuntimeError as e:
        raise RuntimeError(
            f"[{label}] failed to concatenate cached features — token count N differs across batches. "
            f"This happens with cache_pooling='tokens' when videos have different resolutions/lengths. "
            f"Use cache_pooling='pooled' (linear), or for Qwen set resize_mode='fixed' + qwen_fixed_h/w, "
            f"or ensure uniform video resolution."
        ) from e
    labels = torch.cat(labels_acc, dim=0)
    mb = sum(f.element_size() * f.nelement() for f in feats_cat) / 1024.0**2
    logger.info(f"[{label}] feature cache built: {n} samples x {len(feats_cat)} stages "
                f"({cache_pooling}) -> {mb:.0f} MB RAM (this rank)")
    return feats_cat, labels


class CachedTensorDataset(Dataset):
    def __init__(self, feats_cat, labels):
        self.feats = feats_cat            # list[stage] of (n, ...)
        self.labels = labels              # (n,)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, i):
        return [f[i] for f in self.feats], int(self.labels[i])


def _cached_collate(batch):
    nstage = len(batch[0][0])
    feats = [torch.stack([b[0][s] for b in batch], dim=0) for s in range(nstage)]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return feats, labels


def make_cached_loader(feats_cat, labels, batch_size, training):
    # cache is already this rank's shard -> plain DataLoader, just shuffle locally for train.
    ds = CachedTensorDataset(feats_cat, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=0,
                      collate_fn=_cached_collate, drop_last=False)


class PooledLinearProbe(nn.Module):
    """Linear probe over a PRE-POOLED cached vector x=(B, 2D)=[mean ‖ max]."""

    def __init__(self, embed_dim, num_classes, pooling="mean", pre_norm=True):
        super().__init__()
        self.D = embed_dim
        self.pooling = pooling
        in_dim = embed_dim * (2 if pooling == "meanmax" else 1)
        self.norm = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):  # x: (B, 2D)
        d = self.D
        if self.pooling == "mean":
            z = x[..., :d]
        elif self.pooling == "max":
            z = x[..., d:2 * d]
        elif self.pooling == "meanmax":
            z = x[..., :2 * d]
        else:
            raise ValueError(f"unknown pooling {self.pooling!r}")
        return self.linear(self.norm(z))
