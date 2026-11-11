"""블록 단위 attention 끊기 — predictor 호출 동안만 SDPA 를 감싸 softmax 전에 −inf 를 더한다 (2026-11-06, cvpr_auto 요청 T1 · T2).

모든 predictor 경로에서 query 는 key 시퀀스의 **끝쪽** 에 붙는다:
  릴리즈 (정렬된 전체 토큰, Nq == Nk) · prefix / 블록 인과 one-shot (PrefixSpec 의 블록별 SDPA: q 조각 = k 조각의 마지막) ·
  AR KV rollout (새 블록 query, key = 과거 캐시 + 새 블록).
그래서 query i 의 절대 토큰 인덱스 = Nk − Nq + i, key j 의 인덱스 = j, 블록 = 인덱스 // S.

mode
  past : 블록 last 이후 (≥ last) 의 모든 query 가 문맥 블록 0..last−1 을 못 본다 (T1, 과거 끊기)
  last : 모든 query 가 블록 last 를 못 본다 (T2, 마지막 문맥 블록 끊기)
last = 마지막 문맥 블록 (0 부터, 문맥 8 블록이면 7). 가중치 · encoder · 정답은 그대로.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

_ORIG = F.scaled_dot_product_attention


@contextlib.contextmanager
def block_knockout(mode: str | None, tokens_per_block: int, last: int):
    if not mode:
        yield
        return
    if mode not in ("past", "last"):
        raise ValueError(f"attn_knockout={mode!r}: past | last")
    S = int(tokens_per_block)
    cache = {}

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if attn_mask is not None or is_causal:
            raise RuntimeError("block_knockout: 마스크 있는 SDPA 는 다루지 않는다 (PrefixSpec · KV 경로는 마스크 없이 부른다)")
        Nq, Nk = q.size(-2), k.size(-2)
        key = (Nq, Nk, q.dtype, q.device)
        if key not in cache:
            qb = (Nk - Nq + torch.arange(Nq, device=q.device)) // S
            kb = torch.arange(Nk, device=q.device) // S
            if mode == "past":
                cut = (qb[:, None] >= last) & (kb[None, :] < last)
            else:
                cut = (kb[None, :] == last).expand(Nq, Nk)
            if bool(cut.all(dim=1).any()):
                raise RuntimeError(f"block_knockout {mode}: 모든 key 가 끊긴 query 가 있다 (Nq {Nq}, Nk {Nk})")
            cache[key] = None if not bool(cut.any()) else torch.zeros(Nq, Nk, dtype=q.dtype, device=q.device).masked_fill(cut, float("-inf"))
        m = cache[key]
        if m is None:
            return _ORIG(q, k, v, dropout_p=dropout_p, scale=scale, **kw)
        return _ORIG(q, k, v, attn_mask=m, dropout_p=dropout_p, scale=scale, **kw)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG
