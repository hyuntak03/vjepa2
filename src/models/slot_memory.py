"""블록 AR 위 slot 메모리 (2026-11-07, 사용자: "Pretrained AR 위에 Slot 메모리 post-training", "아이디어 최대한 잘 살려서").

왜 [측정, 2026-11-06]: AR 내부 상태에는 가려진 물체의 기억이 있는데 (real 장면 81–92 % 판별) 예측에 반영되는 몫은 1–4 %,
  예측 블록 query 는 61–66 % 를 자기 블록에 쓰고 과거 블록의 물체에는 3 % 만 쓴다. 그리고 문맥을 블록 순서대로 encoding 하면
  학습 없이 영속이 풀린다 (V-JEPA 2 VOB late A 4 → 76). → 블록 순서대로 갱신되는 **물체 단위 상태** 를 predictor 안에 두고
  패치 토큰이 그것을 직접 읽게 한다.

구성 (SAVi 식 + gate + Flamingo 식 0 시작 cross-attention):
  SlotMemory   slot K 개 (predictor 폭 D). 블록마다
               ① 전이 (slot 끼리 self-attention + MLP, 첫 블록은 학습된 초기값 그대로)
               ② 교정 (slot attention: 토큰마다 slot 들에 softmax → slot 이 토큰을 두고 경쟁, iters 회 GRU 갱신)
               ③ gate g = σ(W [LN(전이), LN(교정), log 몫]) — 몫 = 그 slot 이 가져간 토큰 비율 × K (근거 세기).
                  slot = 전이 + g · (교정 − 전이). 물체가 가려져 근거가 없으면 g 를 닫아 기억을 지킨다 (학습으로).
               입력 = encoder 공간 블록 토큰 (LN, 1280): 관측 블록은 LN(h_b), rollout 블록은 AR 이 그린 예측 p_b (같은 공간).
               입력은 detach — slot 은 trunk 의 입력 흐름을 옆에서 읽는다 (보조 손실이 trunk embedding 을 흔들지 않게).
  SlotXAttn    패치 → slot cross-attention. 블록 b 의 패치는 블록 b 까지 갱신된 slot 만 본다 (블록 인과 유지).
               출력 = tanh(gain · a) · o(attn), a = 0 시작 → 0 step 에서 기존 AR 과 출력이 정확히 같다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos2d(grid: int, dim: int) -> torch.Tensor:
    """(1, grid*grid, dim) 2D sin-cos 위치 (행 · 열 각 dim/2)."""
    if dim % 4:
        raise ValueError(f"sincos2d: dim {dim} 은 4 의 배수여야 한다")
    d = dim // 4
    om = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    ey, ex = ys.reshape(-1, 1) * om[None], xs.reshape(-1, 1) * om[None]
    return torch.cat([ey.sin(), ey.cos(), ex.sin(), ex.cos()], 1)[None]


class SlotMemory(nn.Module):
    def __init__(self, in_dim: int, dim: int, n_slots: int = 8, iters: int = 2, grid: int = 16, tr_heads: int = 4,
                 gate_bias: float = 2.0):
        super().__init__()
        self.K, self.D, self.iters, self.gate_bias = int(n_slots), int(dim), int(iters), float(gate_bias)
        self.slot_init = nn.Parameter(torch.zeros(1, self.K, dim))
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self._grid = grid
        # 입력: encoder 공간 토큰 → slot 폭
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, dim)
        self.in_mlp_norm = nn.LayerNorm(dim)
        self.in_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.kv_norm = nn.LayerNorm(dim)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        # slot attention
        self.q_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # 전이 (slot 끼리)
        self.tr_norm1 = nn.LayerNorm(dim)
        self.tr_attn = nn.MultiheadAttention(dim, tr_heads, batch_first=True)
        self.tr_norm2 = nn.LayerNorm(dim)
        self.tr_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # gate
        self.gate = nn.Linear(2 * dim + 1, 1)

    def reset_special(self):
        """predictor 의 _init_weights (Linear trunc_normal 0.02) 뒤에 부른다."""
        with torch.no_grad():
            nn.init.normal_(self.slot_init, std=0.5)
            self.pos.copy_(_sincos2d(self._grid, self.D) * 0.5)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, self.gate_bias)
            # 전이는 처음에 항등 (잔차 0) — 학습 초기 slot 이 흔들리지 않게
            nn.init.zeros_(self.tr_attn.out_proj.weight); nn.init.zeros_(self.tr_attn.out_proj.bias)
            nn.init.zeros_(self.tr_mlp[2].weight); nn.init.zeros_(self.tr_mlp[2].bias)

    def init(self, B: int) -> torch.Tensor:
        return self.slot_init.expand(B, -1, -1)

    def transition(self, s: torch.Tensor) -> torch.Tensor:
        y = self.tr_norm1(s)
        s = s + self.tr_attn(y, y, y, need_weights=False)[0]
        return s + self.tr_mlp(self.tr_norm2(s))

    def encode(self, u: torch.Tensor):
        """u (B, S, in_dim) encoder 공간 블록 토큰 → (k, v) (B, S, D)."""
        x = self.in_proj(self.in_norm(u)) + self.pos.to(u.dtype)
        x = x + self.in_mlp(self.in_mlp_norm(x))
        x = self.kv_norm(x)
        return self.k(x), self.v(x)

    def step(self, s: torch.Tensor, u: torch.Tensor, first: bool):
        """s (B, K, D) 이전 블록 뒤 slot, u (B, S, in_dim) 이번 블록 토큰.
        return (slot (B, K, D), attention (B, K, S) — 토큰마다 slot 에 softmax 한 값 (마지막 회), gate (B, K, 1))."""
        B, K, D = s.shape
        prior = s if first else self.transition(s)
        k, v = self.encode(u.detach())
        c = prior
        attn = None
        for _ in range(self.iters):
            q = self.q(self.q_norm(c))
            logits = torch.einsum("bkd,bnd->bkn", q.float(), k.float()) / math.sqrt(D)
            attn = logits.softmax(dim=1)                                       # slot 끼리 경쟁 (토큰마다)
            w = (attn + 1e-8) / (attn + 1e-8).sum(-1, keepdim=True)            # 토큰 가중 평균
            upd = torch.einsum("bkn,bnd->bkd", w, v.float()).to(c.dtype)
            c = self.gru(upd.reshape(-1, D), c.reshape(-1, D)).view(B, K, D)
            c = c + self.mlp(self.mlp_norm(c))
        mass = attn.mean(-1, keepdim=True) * K                                 # 1 = 고르게 나눈 몫
        gin = torch.cat([F.layer_norm(prior.float(), (D,)), F.layer_norm(c.float(), (D,)), torch.log(mass + 1e-4)], -1)
        g = torch.sigmoid(self.gate(gin.to(self.gate.weight.dtype)).float())
        out = prior + g.to(c.dtype) * (c - prior)
        return out, attn, g


class SlotXAttn(nn.Module):
    """패치 (Bn, S, D) → slot (Bn, K, D) cross-attention. tanh(gain·a) 문 (a = 0 시작)."""

    def __init__(self, dim: int, heads: int = 6, gain: float = 10.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"SlotXAttn: dim {dim} 이 heads {heads} 로 안 나뉜다")
        self.h, self.gain = int(heads), float(gain)
        self.nq, self.nk = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(1))

    def gate_value(self) -> float:
        return float(torch.tanh(self.gain * self.a.detach()).item())

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        Bn, N, D = h.shape
        K, H = s.size(1), self.h
        q = self.q(self.nq(h)).view(Bn, N, H, D // H).transpose(1, 2)
        kv = self.kv(self.nk(s.to(h.dtype))).view(Bn, K, 2, H, D // H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, kv[0], kv[1]).transpose(1, 2).reshape(Bn, N, D)
        return torch.tanh(self.gain * self.a).to(h.dtype) * self.o(y)
