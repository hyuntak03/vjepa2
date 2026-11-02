"""자리 계산기 (scaffold) — 학습 때만 쓰는 "어디" 비계 (2026-11-01, 사용자 설계 · 문서 predictor_lab/scaffold/README.md).

문맥 블록마다 물체 자리 하나를 읽고 (①), 보이는 블록에 직선 · 2 차식을 맞춰 (②) 미래로 굴리고 (③),
그 자리에 "여기에 물체를 그려라" 표시 Δ 를 만들어 (④) predictor 의 미래 mask token 에 더한다.
학습하는 것: ① 의 물체다움 벡터 w (0 단계, 라벨로) · ④ 의 e · γ (1 단계, predictor 와 같이). ② · ③ 은 파라미터 없는 계산.
최종 (3 단계) 에는 이 모듈 · Δ · hint head 를 모두 떼어 predictor 는 릴리즈와 같은 구조가 된다.

모양 약속: z (B, T, S, D) — 블록 T 개 · 토큰 S = 16×16 · 차원 D. 좌표는 토큰 단위 (열, 행) ∈ [0, 15].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 16


def token_coords(device=None) -> torch.Tensor:
    """(S, 2) 토큰 중심 좌표 (열, 행)."""
    r, c = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
    return torch.stack([c.flatten(), r.flatten()], -1).float()


class ObjectScore(nn.Module):
    """① 물체다움 점수 a(r) = w · LN(z(r)) + b. 0 단계에서 라벨 (보이는 target 토큰) 로 학습한다."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 1)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (..., S, D) → (..., S)
        return self.lin(F.layer_norm(z.float(), z.shape[-1:])).squeeze(-1)


def readout(a: torch.Tensor, temp: float = 1.0, topk: int = 0):
    """① 점수 (B, T, S) → 자리 x (B, T, 2) = softmax 무게중심 · 보이는 정도 m (B, T) = 최고 점수.
    topk > 0 이면 점수 상위 k 토큰만으로 무게중심 (배경 끌림 대응, README 깨질 수 있는 지점 2)."""
    xy = token_coords(a.device)
    s = a / temp
    if topk:
        kth = s.topk(topk, -1).values[..., -1:]
        s = s.masked_fill(s < kth, float("-inf"))
    p = s.softmax(-1)
    return p @ xy, a.max(-1).values


def fit_track(x: torch.Tensor, m: torch.Tensor, tau: float, mode: str = "visible", order: int = 1, kappa: float = 1.0):
    """② 보이는 블록에 x = x0 + u t (+ ½ a t²) 를 최소제곱으로 맞춘다 (파라미터 없음).
    x (B, T, 2) · m (B, T). mode: "visible" = m > τ 인 블록만 (무게 1/0) · "weighted" = 무게 σ((m − τ)/κ).
    점이 모자라면 (보이는 블록 < order + 1) 차수를 낮춘다 (1 점이면 그 자리에 멈춤 = u 0).
    반환: 계수 (B, 3, 2) — x(t) = c0 + c1 t + c2 t² (t = 블록 번호), 그리고 쓴 무게 (B, T)."""
    B, T, _ = x.shape
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    w = (m > tau).to(x.dtype) if mode == "visible" else torch.sigmoid((m - tau) / kappa)
    coef = torch.zeros(B, 3, 2, device=x.device, dtype=x.dtype)
    n_vis = (w > 0.5).sum(-1)
    for k in range(B):
        ordk = int(min(order, max(int(n_vis[k]) - 1, 0)))
        if float(w[k].sum()) < 1e-6:                               # 한 번도 안 보임 → 마지막 블록 자리에 둔다
            coef[k, 0] = x[k, -1]; continue
        A = torch.stack([t ** j for j in range(ordk + 1)], -1)     # (T, ordk+1)
        W = w[k].clamp_min(0)[:, None]
        sol = torch.linalg.lstsq(A * W.sqrt(), x[k] * W.sqrt()).solution    # (ordk+1, 2)
        coef[k, :ordk + 1] = sol
    return coef, w


def rollout(coef: torch.Tensor, t_future: torch.Tensor) -> torch.Tensor:
    """③ 계수 (B, 3, 2) · 미래 블록 번호 (F,) → 미래 자리 (B, F, 2). 화면 밖도 그대로 낸다 (자르지 않음)."""
    tt = t_future.to(coef.dtype)
    A = torch.stack([torch.ones_like(tt), tt, tt ** 2], -1)       # (F, 3)
    return torch.einsum("fj,bjc->bfc", A, coef)


class PlaceMarker(nn.Module):
    """④ 자리 표시 Δ_t = γ · g_t ⊗ e — 계산한 미래 자리 중심 Gaussian 지도 g (S,) 에 "물체 있음" 벡터 e (d_p) 를 곱한다.
    Δ 에는 모양 · 색이 없다 (where 만). γ 는 0 에서 시작 → 첫 step 의 predictor 는 릴리즈와 같다. e · γ 는 1 단계에서 학습."""

    def __init__(self, d_p: int, sigma: float = 1.5):
        super().__init__()
        self.e = nn.Parameter(torch.randn(d_p) / math.sqrt(d_p))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.sigma = sigma

    def gauss(self, x_fut: torch.Tensor) -> torch.Tensor:         # (B, F, 2) → (B, F, S)
        xy = token_coords(x_fut.device)
        d2 = ((xy[None, None] - x_fut[:, :, None]) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * self.sigma ** 2))

    def forward(self, x_fut: torch.Tensor) -> torch.Tensor:       # → Δ (B, F, S, d_p)
        return self.gamma * self.gauss(x_fut)[..., None] * self.e


def calc(a_ctx: torch.Tensor, tau: float, t_future: torch.Tensor, mode="visible", order=1, temp=1.0, topk=0, kappa=1.0):
    """①–③ 한 번에: 문맥 점수 (B, Tc, S) → 미래 자리 (B, F, 2) · 문맥 자리 (B, Tc, 2) · m (B, Tc) · 무게."""
    x, m = readout(a_ctx, temp, topk)
    coef, w = fit_track(x, m, tau, mode, order, kappa)
    return rollout(coef, t_future), x, m, w


# ============================================================================= 2 판: (T, 256) 물체 지도 (2026-11-01 사용자 설계)
class MapScaffold(nn.Module):
    """물체 지도 scaffold — 문맥 encoder 토큰 (B, Tc·S, D) → 시간 칸마다 물체 확률 logit (B, Tc + Tf, S).

    점 · 직선 · 문턱 없이 지도로만 간다 (물체 여럿이면 덩어리가 여럿일 뿐). 0 단계에서 encoder 고정 · 이 모듈만 학습 (라벨 = 가려진 부분 포함 물체 점유).
    문맥 토큰: LN (affine 없음) → 선형 D → d. 미래 칸 토큰: 학습되는 query 하나를 Tf·S 자리에 복사 (predictor mask token 꼴).
    RoPE (시간 · 행 · 열 상대 위치, predictor 와 같은 Block) — 절대 위치 임베딩은 없다: 자리는 상대 위치와 문맥 내용으로만 계산한다
    (학습셋 끝 자리를 외울 여지를 줄인다)."""

    def __init__(self, dim_in: int = 1280, d: int = 384, depth: int = 4, heads: int = 12, n_ctx: int = 8, n_fut: int = 8):
        super().__init__()
        from src.models.utils.modules import Block
        self.n_ctx, self.n_fut = n_ctx, n_fut
        self.inp = nn.Linear(dim_in, d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio=4.0, qkv_bias=True, use_rope=True, grid_size=G) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.constant_(self.out.bias, -3.0)                    # 물체 칸 약 5 % 사전

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # z (B, Tc·S, D) → logit (B, Tc + Tf, S)
        B = z.size(0)
        x = self.inp(F.layer_norm(z.float(), z.shape[-1:]))
        x = torch.cat([x, self.query.expand(B, self.n_fut * G * G, -1)], 1)
        for blk in self.blocks:                                   # mask 없음 → 위치 = arange → (칸, 행, 열) = 문맥 0..Tc-1, 미래 Tc..
            x = blk(x)
        return self.out(self.norm(x)).squeeze(-1).view(B, self.n_ctx + self.n_fut, G * G)
