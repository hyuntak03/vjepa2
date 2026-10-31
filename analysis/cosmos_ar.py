"""Cosmos-1.0-Autoregressive-4B — 독립 구현 (megatron / transformer_engine 없이).

무엇을 하나
    이산 토큰 자기회귀 비디오 모델 Cosmos-1.0-Autoregressive-4B 로 영상의 **미래 토큰 음의
    로그우도 (NLL)** 를 teacher forcing 으로 잰다. 직관물리 채점의 surprise 로 쓴다:
        surprise = mean_{미래 토큰} -log p(x_i | x_<i)

공식 코드에서 옮긴 것 (cosmos-predict1/cosmos_predict1/autoregressive/, import 하지 않는다)
    networks/transformer.py   Transformer / TransformerBlock (pre-norm, residual)
    modules/attention.py      GQA (32 q head, 8 kv head, head_dim 128), QK RMSNorm (rope 전),
                              SDPA scale 1/sqrt(128), 인과 마스크
    modules/embedding.py      RotaryPositionEmbeddingPytorchV2 + RotaryPositionEmbedding.get_inv_freq
                              / compute_freqs 의 "3D" 분기 (t 44 · h 42 · w 42 채널, theta 500000,
                              cos/sin 은 float32 로 만든 뒤 q dtype(bf16) 으로 내림)
    modules/normalization.py  RMSNorm (float32 로 정규화 → 입력 dtype → * weight)
    modules/mlp.py            SwiGLU  w2(silu(w1 x) * w3 x)
    configs/base/model_config.py create_video2world_model_config (4b, base):
        rope_dim 3D, add_special_tokens False (BOV/EOV 없음), use_qk_normalization True,
        rope_theta 500000 (ModelConfig 기본, nemo model.yaml rotary_base 500000 과 일치),
        YaRN 없음 (12b 만), pad_to_multiple_of 64 (위치 표 뒤에 0 을 붙일 뿐 앞 위치에 무영향)
    inference/world_generation_pipeline.py (base): 33 프레임, 640x1024, 잠재 격자 [5, 40, 64],
        문맥 = 잠재 1 칸 (1 프레임) 또는 2 칸 (9 프레임)
    utils/inference.py: 픽셀 [-1, 1], 종횡비 유지 resize + center crop
    tokenizer/discrete_video.py + quantizers.py: 인코더 jit 은 bf16, 토큰 = FSQ codes_to_indices
        (levels [8,8,8,5,5,5], basis cumprod) — 인코더가 돌려주는 indices 와 같은지 검사한다

토큰 순서: (t, h, w) 래스터. 시작 토큰 없음 → 첫 토큰은 예측되지 않는다.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

COSMOS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint", "cosmos")
AR_CKPT = os.path.join(COSMOS_ROOT, "Cosmos-1.0-Autoregressive-4B", "model.pt")
TOK_DIR = os.path.join(COSMOS_ROOT, "Cosmos-1.0-Tokenizer-DV8x16x16")

CFG_4B = dict(dim=4096, n_layers=16, n_heads=32, n_kv_heads=8, ffn_hidden_size=14336,
              norm_eps=1e-5, vocab_size=64000, rope_theta=500000.0)
OFFICIAL_LATENT_SHAPE = (5, 40, 64)       # 33 프레임 x 640 x 1024
FSQ_LEVELS = (8, 8, 8, 5, 5, 5)


# ----------------------------------------------------------------------------- 모듈
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


def _rotate_half_te(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_3d_cos_sin(latent_shape: Sequence[int], head_dim: int = 128, theta: float = 500000.0,
                    device=None):
    """공식 RotaryPositionEmbedding(rope_dim='3D') + PytorchV2 의 cos/sin. 반환 (L, head_dim) float32."""
    T, H, W = latent_shape
    dim_h = head_dim // 6 * 2                       # 42
    dim_t = head_dim - 2 * dim_h                    # 44
    sp = torch.arange(0, dim_h, 2)[: dim_h // 2].float().to(device) / dim_h
    tp = torch.arange(0, dim_t, 2)[: dim_t // 2].float().to(device) / dim_t
    spatial_inv = 1.0 / (theta ** sp)
    temporal_inv = 1.0 / (theta ** tp)
    seq = torch.arange(max(latent_shape), dtype=torch.float, device=device)
    et = torch.outer(seq[:T], temporal_inv)         # (T, 22)
    eh = torch.outer(seq[:H], spatial_inv)          # (H, 21)
    ew = torch.outer(seq[:W], spatial_inv)          # (W, 21)
    e = torch.cat([
        et[:, None, None, :].expand(T, H, W, -1),
        eh[None, :, None, :].expand(T, H, W, -1),
        ew[None, None, :, :].expand(T, H, W, -1),
    ] * 2, dim=-1).reshape(T * H * W, head_dim).float()
    return torch.cos(e), torch.sin(e)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, eps):
        super().__init__()
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.hd = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, dim, bias=False)
        self.q_norm = RMSNorm(self.hd, eps)
        self.k_norm = RMSNorm(self.hd, eps)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, L, self.n_heads, self.hd))
        k = self.k_norm(self.wk(x).view(B, L, self.n_kv, self.hd))
        v = self.wv(x).view(B, L, self.n_kv, self.hd)
        c, s = cos[None, :, None, :].to(q.dtype), sin[None, :, None, :].to(q.dtype)
        q = q * c + _rotate_half_te(q) * s
        k = k * c + _rotate_half_te(k) * s
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(self.hd))
        return self.wo(o.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = Attention(c["dim"], c["n_heads"], c["n_kv_heads"], c["norm_eps"])
        self.feed_forward = MLP(c["dim"], c["ffn_hidden_size"])
        self.attention_norm = RMSNorm(c["dim"], c["norm_eps"])
        self.ffn_norm = RMSNorm(c["dim"], c["norm_eps"])

    def forward(self, x, cos, sin):
        h = x + self.attention(self.attention_norm(x), cos, sin)
        return h + self.feed_forward(self.ffn_norm(h))


class CosmosAR(nn.Module):
    def __init__(self, c=CFG_4B):
        super().__init__()
        self.c = dict(c)
        self.tok_embeddings = nn.Embedding(c["vocab_size"], c["dim"])
        self.layers = nn.ModuleList([Block(c) for _ in range(c["n_layers"])])
        self.norm = RMSNorm(c["dim"], c["norm_eps"])
        self.output = nn.Linear(c["dim"], c["vocab_size"], bias=False)
        self._rope = {}

    def rope(self, latent_shape, device):
        key = (tuple(latent_shape), str(device))
        if key not in self._rope:
            self._rope[key] = rope_3d_cos_sin(latent_shape, self.c["dim"] // self.c["n_heads"],
                                              self.c["rope_theta"], device)
        return self._rope[key]

    def hidden(self, tokens, latent_shape):
        """tokens (B, L) — L 은 (t,h,w) 래스터 앞부분. 최종 norm 까지 한 hidden (B, L, dim)."""
        L = tokens.shape[1]
        cos, sin = self.rope(latent_shape, tokens.device)
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, cos[:L], sin[:L])
        return self.norm(h)

    def forward(self, tokens, latent_shape):
        return self.output(self.hidden(tokens, latent_shape))


def load_ar(ckpt=AR_CKPT, device="cuda", dtype=torch.bfloat16):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    with torch.device("meta"):
        m = CosmosAR()
    res = m.load_state_dict(sd, strict=True, assign=True)
    m = m.to(device=device, dtype=dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, res


# ----------------------------------------------------------------------------- 토크나이저
def fsq_codes_to_indices(codes, levels=FSQ_LEVELS):
    """공식 FSQuantizer.codes_to_indices. codes (..., 6) in [-1,1] → int index."""
    lv = torch.tensor(levels, device=codes.device)
    basis = torch.cumprod(torch.tensor((1,) + tuple(levels[:-1]), device=codes.device), 0)
    half = lv // 2
    z = (codes.float() * half + half)
    return (z * basis).sum(-1).round().to(torch.int64)


class CosmosTokenizer:
    def __init__(self, tok_dir=TOK_DIR, device="cuda"):
        torch._C._jit_set_texpr_fuser_enabled(False)
        self.enc = torch.jit.load(os.path.join(tok_dir, "encoder.jit"), map_location=device).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.to(torch.bfloat16)
        self.device = device

    @torch.no_grad()
    def encode(self, video):
        """video (B, 3, T, H, W) float in [-1, 1], T = 1 + 8k. 반환 (indices (B, t, h, w) int64, 일치율)."""
        idx, q, _ = self.enc(video.to(self.device, torch.bfloat16))
        idx2 = fsq_codes_to_indices(q.permute(0, 2, 3, 4, 1))
        agree = float((idx.long() == idx2).float().mean())
        return idx2, agree


# ----------------------------------------------------------------------------- 입력 규약
def frames_to_input(frames_uint8, mode="pad", out_hw=(640, 1024)):
    """frames (T, H, W, 3) uint8 numpy/tensor → (1, 3, T, 640, 1024) in [-1,1], 유효 열 마스크.

    mode="pad" : 종횡비 유지로 640x1024 안에 맞춰 넣고 (bilinear, antialias) 나머지를 -1 (검정) 로
                 채운다 (letterbox). 콘텐츠 크기는 16 의 배수로 내림해 잠재 칸 경계에 맞춘다.
                 정사각 512 입력 → 640x640 이 가운데, 좌우 192 px (잠재 12 열씩) 패딩.
    mode="crop": 공식 resize_input (종횡비 유지 확대 + center crop). 정사각이면 위아래 37.5% 손실.
    반환 valid: 잠재 격자 (H/16, W/16) 중 콘텐츠가 있는 칸 (bool).
    """
    x = torch.as_tensor(frames_uint8)
    x = x.permute(0, 3, 1, 2).float() / 255.0 * 2 - 1           # (T, 3, H, W)
    T, _, H, W = x.shape
    oh, ow = out_hw
    valid = torch.zeros(oh // 16, ow // 16, dtype=torch.bool)
    if mode == "crop":
        r = max(ow / W, oh / H)
        nh, nw = int(math.ceil(r * H)), int(math.ceil(r * W))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        t0, l0 = (nh - oh) // 2, (nw - ow) // 2
        x = x[:, :, t0:t0 + oh, l0:l0 + ow]
        valid[:] = True
    else:
        r = min(oh / H, ow / W)
        nh, nw = int(round(H * r)), int(round(W * r))
        nh, nw = min(oh, nh - nh % 16), min(ow, nw - nw % 16)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        t0 = (oh - nh) // 2
        t0 -= t0 % 16
        l0 = (ow - nw) // 2
        l0 -= l0 % 16
        out = torch.full((T, 3, oh, ow), -1.0)
        out[:, :, t0:t0 + nh, l0:l0 + nw] = x
        x = out
        valid[t0 // 16:(t0 + nh) // 16, l0 // 16:(l0 + nw) // 16] = True
    return x.permute(1, 0, 2, 3).unsqueeze(0).clamp(-1, 1), valid


# ----------------------------------------------------------------------------- 채점
@torch.no_grad()
def token_nll(model, tokens, latent_shape, n_ctx_latent, chunk=2048):
    """tokens (B, t*h*w). 미래 잠재 칸 (n_ctx_latent 부터) 토큰마다 -log p. 반환 (B, t-n_ctx, h, w) float32."""
    T, H, W = latent_shape
    S = H * W
    start = n_ctx_latent * S
    hid = model.hidden(tokens, latent_shape)                      # (B, L, D)
    hsel = hid[:, start - 1:-1]                                   # 위치 p 가 토큰 p+1 을 예측
    tgt = tokens[:, start:]
    out = []
    for i in range(0, hsel.shape[1], chunk):
        lg = model.output(hsel[:, i:i + chunk]).float()
        lp = torch.log_softmax(lg, dim=-1)
        out.append(-lp.gather(-1, tgt[:, i:i + chunk, None])[..., 0])
    nll = torch.cat(out, 1)
    return nll.view(tokens.shape[0], T - n_ctx_latent, H, W)


@torch.no_grad()
def nll_future(model, tokenizer, frames_uint8, n_ctx_latent=3, mode="pad"):
    """frames (33, H, W, 3) uint8 → dict(nll_mean (유효 열만), nll_mean_all, per_latent, agree, tokens)."""
    vid, valid = frames_to_input(frames_uint8, mode=mode)
    idx, agree = tokenizer.encode(vid)                            # (1, 5, 40, 64)
    shape = tuple(idx.shape[1:])
    tok = idx.reshape(1, -1).to(next(model.parameters()).device)
    nll = token_nll(model, tok, shape, n_ctx_latent)[0]           # (t_f, 40, 64)
    cv = valid.to(nll.device)
    return {
        "nll_mean": float(nll[:, cv].mean()),
        "nll_mean_all": float(nll.mean()),
        "per_latent": [float(nll[i][cv].mean()) for i in range(nll.shape[0])],
        "agree": agree,
        "tokens": idx.cpu(),
        "nll_map": nll.cpu(),
    }
