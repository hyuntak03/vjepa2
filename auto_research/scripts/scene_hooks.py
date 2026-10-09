#!/usr/bin/env python3
"""인과 개입용 predictor attention hook — release (RoPEAttention, Block) 와 Ariel (ACRoPEAttention, ACBlock, PrefixSpec) 공용 (2026-09-25).

predictor_hooks.py (M4) 는 RoPEAttention 시그니처만 알고 attn_mask 를 hook.kmask 로 덮어써서 Ariel prefix 규칙을 지웠다.
이 hook 은 원래 attn_mask 를 dense bool 로 바꾼 뒤 개입을 **그 위에 얹는다**.

  hook.rule      : None (원래 규칙) | "full" | "prefix" (문맥→문맥, 미래 b → 문맥 + 미래 ≤ b) | "isolated" (문맥→문맥, 미래 b → 문맥 + b)
                   — 원래 규칙을 **대체**한다 (I3: attention 규칙).
  hook.pos_scale : None | float s — 공간 RoPE 위치 (h, w) 를 s 배. hook.pos_rows = "all" (전 토큰) | "future" (미래 질의 행만: 그 행의 q 와 모든 k 를
                   축척판으로 회전; 문맥 행은 원래) (I2: 거리 척도).
  hook.temp      : None | τ — 미래 질의 행의 logit 을 τ 배 (정답 위치를 주지 않는 날카롭히기).
  hook.bias      : None | (N, N) float — logit 에 더한다 (I1: 조회 보조). 허용 안 된 칸은 −inf 유지.
  hook.rec       : None | dict(q_rows (R,), key_src (R,), ring (R, 256) bool, key_copy (R,), l_off) — 미래 질의 행의 attention 확률에서
                   출처 key · 같은 거리 ring 평균 · 복사 key (마지막 문맥 튜블릿 안) 질량을 층마다 hook.rec_out 에 (L, 3, R) 로 쌓는다.
  hook.qshift    : None | (N, 2) float — 질의 행의 공간 RoPE 위치에 더할 (dh, dw) 칸. key 는 그대로 → "이 자리의 질의를 출처 자리에서 하게" (정답 없는 판은 문맥 flow 외삽).
  hook.layers    : None (전 층) | set(int).
위치 규약: 토큰은 절대 인덱스 순으로 정렬돼 들어온다 (release predictor.py:223 · rollout_predictor 정렬). block = 위치 // 256.
temp · pos_rows=future · rec 는 attention 을 직접 계산한다 (N=4096 에서 logit 약 400 MB/층).
검사: 개입 없음이면 원래 출력과 같아야 한다 (e_scene_reach 가 첫 clip 에서 기록: release 0.0, Ariel 4.9e-4).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.utils.modules import rotate_queries_or_keys, PrefixSpec, ACRoPEAttention  # noqa: E402

S_BLOCK = 256


class SceneHook:
    def __init__(self):
        self.n_ctx_blocks = None; self.reset()

    def reset(self):
        self.rule = None; self.pos_scale = None; self.pos_rows = "all"; self.temp = None; self.bias = None; self.layers = None
        self.rec = None; self.rec_out = []
        self.qshift = None      # (N, 2) float: 질의 행의 RoPE (h, w) 위치에 더할 오프셋 (칸). key 위치는 그대로 (REVIEW_v2 2 순위: 질의 위치 평행이동)


def rule_mask(block, n_ctx, rule):
    bq, bk = block[:, None], block[None, :]
    ctx_q, ctx_k = bq < n_ctx, bk < n_ctx
    if rule == "full":
        return torch.ones(len(block), len(block), dtype=torch.bool, device=block.device)
    if rule == "prefix":
        return torch.where(ctx_q, ctx_k, bk <= bq)
    if rule == "isolated":
        return torch.where(ctx_q, ctx_k, ctx_k | (bk == bq))
    raise ValueError(rule)


def _dense_from(attn_mask, block):
    if attn_mask is None:
        return None
    if isinstance(attn_mask, PrefixSpec):
        return rule_mask(block, attn_mask.n_ctx, "prefix")
    if torch.is_tensor(attn_mask) and attn_mask.dtype == torch.bool:
        return attn_mask
    if torch.is_tensor(attn_mask):                       # float 가산 마스크 (mret predictor: −inf 허용 + bias). 그대로 쓴다
        return attn_mask
    raise TypeError(f"지원 안 하는 attn_mask: {type(attn_mask)}")


def _rotate(_a, q, k, d, h, w, hq=None, wq=None):
    s = 0; qs, ks = [], []
    for pos, posq, dim in ((d, d, _a.d_dim), (h, h if hq is None else hq, _a.h_dim), (w, w if wq is None else wq, _a.w_dim)):
        qs.append(rotate_queries_or_keys(q[..., s:s + dim], pos=posq)); ks.append(rotate_queries_or_keys(k[..., s:s + dim], pos=pos)); s += dim
    if s < _a.head_dim:
        qs.append(q[..., s:]); ks.append(k[..., s:])
    return torch.cat(qs, -1), torch.cat(ks, -1)


def patch(pred, hook: SceneHook):
    for li, blk in enumerate(pred.predictor_blocks):
        att = blk.attn
        is_ac = isinstance(att, ACRoPEAttention)

        def fwd(x, mask=None, attn_mask=None, T=None, H=None, W=None, H_patches=None, W_patches=None, action_tokens=0, _a=att, _ac=is_ac, _li=li):
            assert action_tokens == 0
            Hp = H if H is not None else H_patches; Wp = W if W is not None else W_patches
            B, N, C = x.size()
            qkv = _a.qkv(x).unflatten(-1, (3, _a.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q0, k0, v = qkv[0], qkv[1], qkv[2]
            ids = mask
            m = ids.unsqueeze(1).repeat(1, _a.num_heads, 1)
            d, h, w = _a.separate_positions(m, Hp, Wp)
            if _ac:
                h = h * (_a.grid_size / Hp); w = w * (_a.grid_size / Wp)
            on = hook.layers is None or _li in hook.layers
            block = ids[0] // S_BLOCK; n_ctx = hook.n_ctx_blocks; fut = block >= n_ctx
            ps = hook.pos_scale if on else None
            if ps is not None and hook.pos_rows == "all":
                q, k = _rotate(_a, q0, k0, d, h * ps, w * ps)
            elif on and hook.qshift is not None:
                sh = hook.qshift.to(h.dtype)                                          # (N, 2) = (dh, dw)
                q, k = _rotate(_a, q0, k0, d, h, w, hq=h + sh[None, None, :, 0], wq=w + sh[None, None, :, 1])
            else:
                q, k = _rotate(_a, q0, k0, d, h, w)
            am = rule_mask(block, n_ctx, hook.rule) if (on and hook.rule is not None) else _dense_from(attn_mask, block)
            fm = None
            if am is not None and am.dtype != torch.bool:      # 이미 float 가산 마스크 (mret). 규칙 개입은 그 위에 못 얹는다
                fm, am = am.to(q.dtype), None
                if on and hook.bias is not None:
                    fm = fm + hook.bias.to(q.dtype)
            elif on and hook.bias is not None:
                fm = hook.bias.to(q.dtype)
                if am is not None:
                    fm = fm.masked_fill(~am, float("-inf"))
            manual = on and (hook.temp is not None or (ps is not None and hook.pos_rows == "future") or hook.rec is not None)
            if not manual:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=fm if fm is not None else am)
            else:
                logit = (q.float() @ k.float().transpose(-2, -1)) * _a.scale                     # (B, H, N, N)
                if ps is not None and hook.pos_rows == "future":
                    qf, kf = _rotate(_a, q0, k0, d, h * ps, w * ps)
                    logit[:, :, fut] = (qf[:, :, fut].float() @ kf.float().transpose(-2, -1)) * _a.scale
                if hook.temp is not None:
                    logit[:, :, fut] = logit[:, :, fut] * hook.temp
                if fm is not None:
                    logit = logit + fm.float()
                elif am is not None:
                    logit = logit.masked_fill(~am, float("-inf"))
                logit = torch.nan_to_num(logit, nan=float("-inf"))
                pr = logit.softmax(-1)
                if hook.rec is not None:
                    r = hook.rec; P = pr[0, :, r["q_rows"]].mean(0)                            # (R, N) head 평균
                    PL = P[:, r["l_off"]:r["l_off"] + S_BLOCK]                                    # 마지막 문맥 튜블릿 key
                    src = r["key_src"].clamp(min=0)
                    p_src = torch.gather(PL, 1, src[:, None])[:, 0].masked_fill(r["key_src"] < 0, float("nan"))
                    p_ring = (PL * r["ring"]).sum(1) / r["ring"].sum(1).clamp(min=1)
                    p_copy = torch.gather(PL, 1, r["key_copy"][:, None])[:, 0]
                    hook.rec_out.append(torch.stack([p_src, p_ring, p_copy]).half())
                out = (pr.to(v.dtype) @ v)
            x = out.transpose(1, 2).reshape(B, N, C)
            return _a.proj_drop(_a.proj(x))
        att.forward = fwd
    return hook
