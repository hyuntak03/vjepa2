#!/usr/bin/env python3
"""predictor attention hook — v3 파이프라인 전용 사본 (2026-09-25).

`z_research/scripts/analysis/rollout2_predictor_locality.py` 의 `Hook` · `patch_attention` 을 **동작 그대로** 옮겼다.
v3 (RollOut_v3) 스크립트가 RollOutV2 스크립트 (import 시 RollOutV2 config·캐시 경로를 읽는다) 에 기대지 않게 하려는 것이다.
("RollOutV2가 아니라 V3로" 사용자 지시, VERIFY_0925 §M4 12).

  Hook.rec_rows : (B, R) long, 기록할 query 행 (없으면 기록 안 함)
  Hook.kmask    : (N, N) bool, True = 허용. sdpa attn_mask 로 전 층에 걸린다 (없으면 원래 full attention)
  Hook.store    : 층마다 (B, R, N) head 평균 softmax 를 append
  Hook.per_head : 추가 옵션 (기본 False = 원본과 같음). True 면 store 에 head 평균 대신 (B, H, R, N) 을 넣는다.

RoPEAttention.forward (src/models/utils/modules.py) 와 같은 계산: qkv → d/h/w RoPE 회전 → sdpa (is_causal False, dropout 0).
검사: m4_ring_null_cpu.py 가 clip 하나에서 hook(kmask=None) · hook(kmask=전부 True) 의 p 를 원래 predictor 의 p 와 비교해 남긴다.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.utils.modules import rotate_queries_or_keys  # noqa: E402


class Hook:
    """RoPEAttention.forward 대체. rec_rows 가 있으면 그 query 행의 softmax 를 층마다 모은다; kmask 가 있으면 sdpa attn_mask."""
    def __init__(self, per_head: bool = False):
        self.rec_rows = None; self.kmask = None; self.store = []; self.per_head = per_head


def patch_attention(pred, hook):
    for blk in pred.predictor_blocks:
        att = blk.attn

        def fwd(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, _a=att):
            B, N, C = x.size()
            qkv = _a.qkv(x).unflatten(-1, (3, _a.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            m = mask.unsqueeze(1).repeat(1, _a.num_heads, 1)
            d, h, w = _a.separate_positions(m, H_patches, W_patches)
            s = 0; qs, ks = [], []
            for pos, dim in ((d, _a.d_dim), (h, _a.h_dim), (w, _a.w_dim)):
                qs.append(rotate_queries_or_keys(q[..., s:s + dim], pos=pos)); ks.append(rotate_queries_or_keys(k[..., s:s + dim], pos=pos)); s += dim
            if s < _a.head_dim:
                qs.append(q[..., s:]); ks.append(k[..., s:])
            q = torch.cat(qs, -1); k = torch.cat(ks, -1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=hook.kmask)
            if hook.rec_rows is not None:                                  # (B, R) query 행
                qi = torch.gather(q, 2, hook.rec_rows[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[-1]))
                logit = (qi.float() @ k.float().transpose(-2, -1)) * _a.scale
                if hook.kmask is not None:
                    km = hook.kmask[hook.rec_rows[0]]                     # 모든 clip 같은 반경일 때만
                    logit = logit.masked_fill(~km[None, None], float("-inf"))
                sm = logit.softmax(-1)
                hook.store.append(sm if hook.per_head else sm.mean(1))    # 기본: (B, R, N) head 평균
            x = out.transpose(1, 2).reshape(B, N, C)
            return _a.proj_drop(_a.proj(x))
        att.forward = fwd
