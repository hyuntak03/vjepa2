"""움직임 정렬 조회 (motion-aligned retrieval) predictor — DESIGN_CHOICES DC1(b) (2026-09-25).

무엇이 다른가 (prefix 판 `VisionTransformerPredictorPrefix` 에 **조회 bias 하나** 를 더한다):
  미래 질의 (블록 b ≥ C, 칸 s) 가 마지막 문맥 블록 (L = C−1) 의 key (칸 k) 를 볼 때 attention logit 에
      bias_l(b, s; k) = β_l · exp(−‖s − (k + v_k · Δt)‖² / 2σ²),   Δt = b − L (블록), v_k = head(z_L[k]) (칸/블록)
  를 더한다. v_k 는 **문맥 encoder 토큰에서 학습된 head** 가 낸 "이 key 의 내용이 블록당 몇 칸 움직이나" 다.
  즉 "k 의 내용이 Δt 뒤에 s 에 올 것" 이면 (b, s) → (L, k) 조회를 키운다. 나머지 (다른 문맥 블록 · 미래 key) 는 bias 0.

왜 (auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md):
  · 병목 = 미래 질의가 **먼 출처** 문맥 내용을 가져오는 attention 강도. 진짜 출처에 logit +β 를 주면 reach 가 세기에 비례해 는다
    (인공 팬 · SSv2 · EK100 · v3 모두). 규칙 · encoder · 일반 날카롭히기는 아니다.
  · 실영상에서는 출처 **방향** 도 약하고, 학습 없는 등속 외삽 출처는 무효 (45 % 일치) → 출처 추정은 학습된 모듈이어야 한다.
  그래서 bias 의 **모양** (출처 정렬 Gaussian) 은 고정하고, **v_k (어디로) 와 β_l (얼마나)** 만 학습한다. JEPA 손실로 end-to-end.
  (선택) `aux_flow`: v_k 를 문맥 flow 로 보조 감독 — 학습 루프가 `flow_target` 을 주면 쓴다. 기본 없음.

호출 규약은 릴리즈 · prefix 와 같다: forward(x, masks_x, masks_y, mask_index). x 는 문맥 토큰 (B, Kx, D_enc), 블록-major 정렬.
prefix_impl 은 dense 로 강제한다 (float mask 가 필요; PrefixSpec split 커널은 bias 를 못 받는다).
state_dict: prefix 판 + `mret_head.*`, `mret_beta`, (`mret_logsigma`). 릴리즈 · Ariel 체크포인트는 strict=False 로 읽고 새 키만 초기화한다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rollout_predictor import VisionTransformerPredictorPrefix, build_prefix_mask


class VisionTransformerPredictorMRet(VisionTransformerPredictorPrefix):
    def __init__(self, *args, mret_hidden=256, mret_beta_init=0.5, mret_sigma=1.0, mret_learn_sigma=False,
                 mret_per_layer=True, mret_vmax=4.0, **kwargs):
        kwargs["prefix_impl"] = "dense"
        super().__init__(*args, **kwargs)
        D = self.predictor_embed.in_features
        self.mret_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, mret_hidden), nn.GELU(), nn.Linear(mret_hidden, 2))
        # 시작: v ≈ 0 (bias 가 자기 자리 근처) + β 작게. v 를 정확히 0 으로 두면 β·core 의 v-기울기가 대칭으로 상쇄돼 안 움직인다 → 작은 난수.
        nn.init.normal_(self.mret_head[-1].weight, std=0.02); nn.init.zeros_(self.mret_head[-1].bias)
        n_l = len(self.predictor_blocks) if mret_per_layer else 1
        self.mret_beta = nn.Parameter(torch.full((n_l,), float(mret_beta_init)))
        ls = torch.tensor(math.log(float(mret_sigma)))
        self.mret_logsigma = nn.Parameter(ls) if mret_learn_sigma else None
        self.register_buffer("_mret_logsigma_fixed", ls, persistent=False)
        self.mret_vmax = float(mret_vmax)
        self.attn_regime = "prefix_causal+mret"
        self._last_v = None

    # ------------------------------------------------------------------ bias
    def _sigma(self):
        return (self.mret_logsigma if self.mret_logsigma is not None else self._mret_logsigma_fixed).exp()

    def velocity(self, x_ctx_last: torch.Tensor) -> torch.Tensor:
        """(B, S, D_enc) 마지막 문맥 블록 encoder 토큰 → (B, S, 2) 칸/블록 (x, y). tanh 로 ±vmax 안."""
        return torch.tanh(self.mret_head(x_ctx_last.float())) * self.mret_vmax

    def _bias_core(self, v: torch.Tensor, n_ctx: int, n_pred: int) -> torch.Tensor:
        """(B, S, 2) → (B, n_pred·S, S) = exp(−‖s − (k + v_k Δt)‖²/2σ²)  [질의 (b, s) × key k in L]."""
        B, S, _ = v.shape; G = self.grid_width
        ks = torch.arange(S, device=v.device); kx, ky = (ks % G).float(), (ks // G).float()
        sx, sy = kx, ky                                                                      # 질의 칸 좌표 (같은 격자)
        dts = torch.arange(1, n_pred + 1, device=v.device, dtype=v.dtype)                      # Δt = 1..K (블록)
        # 예측 착지: (B, K, S_k, 2)
        land_x = kx[None, None, :] + v[:, None, :, 0] * dts[None, :, None]
        land_y = ky[None, None, :] + v[:, None, :, 1] * dts[None, :, None]
        d2 = (sx[None, None, :, None] - land_x[:, :, None, :]) ** 2 + (sy[None, None, :, None] - land_y[:, :, None, :]) ** 2   # (B, K, S_q, S_k)
        return torch.exp(-d2 / (2 * self._sigma() ** 2)).reshape(B, n_pred * S, S)

    def forward(self, x, masks_x, masks_y, mask_index=0, has_cls=False):
        if isinstance(masks_x, list): masks_x = masks_x[0]
        if isinstance(masks_y, list): masks_y = masks_y[0]
        B = x.size(0); S = self.tokens_per_block
        n_ctx = int(masks_x.size(1) // S); n_pred = int(masks_y.size(1) // S)
        assert n_ctx * S == masks_x.size(1) and n_pred * S == masks_y.size(1), "mret: 블록 정렬 마스크만 지원"
        v = self.velocity(x[:, (n_ctx - 1) * S: n_ctx * S])                                   # (B, S, 2)
        self._last_v = v.detach()
        core = self._bias_core(v, n_ctx, n_pred)                                              # (B, K·S, S)

        xe = self.predictor_embed(x)
        mt = self.mask_tokens[mask_index % self.num_mask_tokens]
        y = mt.expand(B, masks_y.size(1), -1)
        idx = torch.cat([masks_x, masks_y], dim=1); seq = torch.cat([xe, y], dim=1)
        order = torch.argsort(idx, dim=1); idx = torch.gather(idx, 1, order)
        seq = torch.gather(seq, 1, order.unsqueeze(-1).expand(-1, -1, seq.size(-1)))
        N = idx.size(1); block_of = idx[0] // S
        assert bool((idx[0] == torch.arange(N, device=idx.device)).all()), "mret: 문맥 · 미래가 블록 0 부터 연속이어야 한다"
        allow = build_prefix_mask(block_of, n_ctx).to(seq.device)                              # (N, N) bool
        neg = torch.zeros(N, N, device=seq.device, dtype=torch.float32).masked_fill(~allow, float("-inf"))
        # bias 자리: 질의 행 [n_ctx·S, N), key 열 [(n_ctx−1)·S, n_ctx·S)
        q0, k0 = n_ctx * S, (n_ctx - 1) * S
        h = seq
        for li, blk in enumerate(self.predictor_blocks):
            beta = self.mret_beta[li if self.mret_beta.numel() > 1 else 0]
            am = neg.unsqueeze(0).expand(B, -1, -1).clone()
            am[:, q0:, k0:k0 + S] = am[:, q0:, k0:k0 + S] + beta * core
            am = am.unsqueeze(1).to(h.dtype)                                                  # (B, 1, N, N) — head 공유
            h = blk(h, mask=idx, attn_mask=am, T=None, H=self.grid_height, W=self.grid_width, action_tokens=0)
        h = self.predictor_proj(self.predictor_norm(h))
        inv = torch.argsort(order, dim=1)
        h = torch.gather(h, 1, inv.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return h[:, masks_x.size(1):, :]


def vit_mret_predictor(**kwargs):
    kwargs.pop("mask_mode", None)
    return VisionTransformerPredictorMRet(mask_mode="prefix", **kwargs)
