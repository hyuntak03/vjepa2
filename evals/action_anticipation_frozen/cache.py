"""EK100 encoder 특징 캐시 위에서 probe 를 학습·채점한다 (2026-10-16). 캐시는 cvpr/eval/ek100/cache/extract.py 가 만든다.

config `experiment.data.cache_dir: <…/ek100_feats/<이름>>` 가 있을 때만 eval.py 가 이 경로로 간다 (없으면 공식 경로 그대로).
  - 데이터: <cache_dir>/{train,val}/joint.npy (+ blocks.npy) · meta.npz. udata 형식은 엔진과 같다 (feats, verb, noun, at).
    train 은 epoch e 에 view (e % views) 를 쓴다. 증강은 추출 때 view 마다 한 번 뽑혀 고정돼 있다 (엔진은 epoch 마다 새로 뽑는다 — 차이).
  - 모델: wrapper 의 encoder 를 캐시 값을 돌려주는 자리로 바꾸고 (encoder 는 GPU 에서 내린다), AR kind 의 encode_blocks 도 캐시의 blocks 를 돌려준다.
    predictor · probe 계산은 wrapper.forward 를 그대로 탄다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger()


class CacheSet(torch.utils.data.Dataset):
    def __init__(self, d: Path, blocks: bool):
        self.d, self.blocks = Path(d), blocks
        m = np.load(self.d / "meta.npz")
        self.info = __import__("json").load(open(self.d / "info.json"))
        self.n_clip = int(self.info["n_clip"]); self.views = int(self.info["views"])
        ok = m["valid"].astype(bool)
        self.row, self.view = m["row"][ok], m["view"][ok]
        self.verb, self.noun, self.at = m["verb"][ok], m["noun"][ok], m["at"][ok].astype(np.float64)
        self.cur_view = 0
        self.idx = {v: np.nonzero(self.view == v)[0] for v in range(self.views)}
        self._mm = None

    def set_view(self, v):
        self.cur_view = int(v) % self.views

    def n(self):
        return len(self.idx[self.cur_view])

    def _open(self):
        if self._mm is None:
            ks = ("joint", "blocks") if self.blocks else ("joint",)
            self._mm = {k: np.load(self.d / f"{k}.npy", mmap_mode="r") for k in ks}
            self._sc = {k: np.load(self.d / f"{k}_scale.npy", mmap_mode="r") for k in ks if (self.d / f"{k}_scale.npy").exists()}   # int8 캐시 (extract.py --dtype q8)
        return self._mm

    def _row(self, k, r):
        x = np.ascontiguousarray(self._mm[k][r])
        if k in self._sc:
            return torch.from_numpy(x.astype(np.float32) * self._sc[k][r][None]).half()
        return torch.from_numpy(x)

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        mm = self._open(); r = int(self.view[i]) * self.n_clip + int(self.row[i])
        x = self._row("joint", r)
        if self.blocks:
            x = torch.cat([x, self._row("blocks", r)], 0)
        return x, int(self.verb[i]), int(self.noun[i]), float(self.at[i])


class _Sampler(torch.utils.data.Sampler):
    """train: epoch 마다 그 view 의 행을 섞어 rank 로 나눈다 (모자라면 앞에서 채워 rank 끼리 길이를 맞춘다).
    val: 순서대로 rank::ws, 채우지 않는다 (exact val 이 clip 을 정확히 한 번 센다)."""

    def __init__(self, ds: CacheSet, training, ws, rank):
        self.ds, self.training, self.ws, self.rank, self.epoch = ds, training, ws, rank, 0

    def set_epoch(self, e):
        self.epoch = e; self.ds.set_view(e)

    def _ids(self):
        ids = self.ds.idx[self.ds.cur_view]
        if not self.training:
            return ids[self.rank::self.ws]
        g = np.random.default_rng(self.epoch); ids = ids[g.permutation(len(ids))]
        per = -(-len(ids) // self.ws); ids = np.concatenate([ids, ids[: per * self.ws - len(ids)]])
        return ids[self.rank::self.ws]

    def __iter__(self):
        return iter(self._ids().tolist())

    def __len__(self):
        return len(self._ids())


class _Info:
    def __init__(self, sampler):
        self.sampler = sampler

    def set_epoch(self, e):
        self.sampler.set_epoch(e)


def init_cached_data(cache_dir, training, batch_size, world_size, rank, num_workers, blocks):
    split = "train" if training else "val"
    ds = CacheSet(Path(cache_dir) / split, blocks)
    sp = _Sampler(ds, training, world_size, rank)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sp, num_workers=num_workers, pin_memory=True,
                                     drop_last=False, persistent_workers=False, prefetch_factor=4 if num_workers else None)
    n = ds.n()
    dl.num_batches = n // (world_size * batch_size)          # 엔진과 같은 정의 (num_clips // (ws*bs))
    dl.num_samples = n
    logger.info(f"[cache] {split} {Path(cache_dir) / split}: rows {n} (view {ds.cur_view}/{ds.views}) blocks={blocks} · num_batches {dl.num_batches}")
    return ds, dl, _Info(sp)


class _Stash(torch.nn.Module):
    """encoder 자리. 캐시 값을 돌려준다."""

    def __init__(self, embed_dim, patch_size, tubelet_size):
        super().__init__()
        self.embed_dim, self.patch_size, self.tubelet_size = embed_dim, patch_size, tubelet_size
        self.cur = None

    def forward(self, x):
        return self.cur


class CachedModel(torch.nn.Module):
    """model(feats, at): feats = (B, N[+N], D) 캐시 → wrapper.forward. N = 4096 (joint), 뒤 N 은 blocks (AR kind 만)."""

    def __init__(self, wrapper, n_tok):
        super().__init__()
        enc = wrapper.encoder
        wrapper.encoder = _Stash(enc.embed_dim, enc.patch_size, enc.tubelet_size)
        del enc; torch.cuda.empty_cache()
        self.w, self.n_tok, self.embed_dim = wrapper, n_tok, wrapper.embed_dim
        self.is_ar = getattr(wrapper.predictor, "ek_kind", "default") in ("ar", "ar_state")
        self._blocks = None
        if self.is_ar:
            import app.vjepa_frozen.ar as AR
            orig = AR.encode_blocks

            def _cached_blocks(enc, clips, tub, _o=orig):
                return self._blocks if isinstance(enc, _Stash) else _o(enc, clips, tub)
            AR.encode_blocks = _cached_blocks

    def forward(self, x, anticipation_times):
        x = x.float()
        self.w.encoder.cur = x[:, : self.n_tok]
        if self.is_ar:
            assert x.size(1) == 2 * self.n_tok, "AR kind 는 캐시에 blocks 가 있어야 한다 (data.cache_blocks: true)"
            self._blocks = x[:, self.n_tok:]
        return self.w(x[:, :1], anticipation_times)
