"""Pixel decoder (2026-10-20, 사용자 지시): 블록 latent (256 토큰 × 1280, LN(target_encoder)) → 그 블록 2 장의 픽셀.

ViT-S 크기 (폭 384 · 12 층 · 6 head). 입력 Linear(1280 → 384) + 학습 위치 임베딩 (256) → Transformer → LN →
Linear(384 → tubelet 2 × 16 × 16 × 3) → (B, 3, 2, 256, 256). 출력은 정규화된 픽셀 공간 (ImageNet mean/std).
입력이 AR predictor 의 정답 · 예측과 같은 공간이라 predictor 미래를 그대로 그림으로 볼 수 있다."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PixelDecoder(nn.Module):
    def __init__(self, in_dim=1280, dim=384, depth=12, heads=6, grid=16, patch=16, tubelet=2):
        super().__init__()
        self.grid, self.patch, self.tubelet = grid, patch, tubelet
        self.inp = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, tubelet * patch * patch * 3)

    def forward(self, z):
        """z (B, 256, in_dim) → (B, 3, tubelet, H, W)"""
        x = self.inp(z) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.out(self.norm(x))                                   # (B, 256, t·p·p·3)
        B, g, p, t = x.size(0), self.grid, self.patch, self.tubelet
        x = x.view(B, g, g, t, p, p, 3).permute(0, 6, 3, 1, 4, 2, 5)  # (B, 3, t, g, p, g, p)
        return x.reshape(B, 3, t, g * p, g * p)
