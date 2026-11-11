import math
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path


try:                                                   # torch >= 2.5
    from torch.nn.attention.flex_attention import BlockMask as _FlexBlockMask, flex_attention as _flex_attention_eager
    _flex_attention = torch.compile(_flex_attention_eager, dynamic=False)
except Exception:                                      # noqa: BLE001
    _FlexBlockMask, _flex_attention = None, None


class PrefixSpec:
    """(문맥 블록 수, 미래 블록 수, 블록당 토큰). ACRoPEAttention 이 마스크 없이 블록별 dense SDPA 로 푼다.
    토큰이 인덱스 순으로 정렬돼 있고 블록 경계가 맞아야 한다 (prefix_window / temporal_prefix)."""
    __slots__ = ("n_ctx", "n_pred", "tokens_per_block", "future_causal")
    def __init__(self, n_ctx, n_pred, tokens_per_block, future_causal=True):
        self.n_ctx, self.n_pred, self.tokens_per_block = int(n_ctx), int(n_pred), int(tokens_per_block)
        self.future_causal = bool(future_causal)   # False = 미래 블록끼리 인과 없이 서로 다 본다 (prefix_full, 2026-09-24)


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    N_T = add_tokens + (H * W)
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by commenting
    # -- out the two lines below, and uncommenting the following two lines.
    # -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    # emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    # emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.register_buffer("t_scale", None, persistent=False)   # 2026-11-06 헤드별 시간 척도 (set_time_scale). None = 예전과 같다
        self.vo_on = False   # 2026-11-06 시간을 value · 출력에 (set_vo_time)
        self.register_buffer("t_pair_keep", None, persistent=False)   # 2026-11-07 Harmonic 시간 RoPE (set_time_pairs). None = 예전과 같다

    def set_time_pairs(self, spec):
        """2026-11-07 (사용자: "딱 mash-vlm 처럼", MASH-VLM Harmonic-RoPE 를 시간 축에): 시간 RoPE 의 회전 짝 (차원 2p, 2p+1) 중
        spec 에 든 짝만 시간 회전 (고유 번호 = 순서·간격), 나머지 짝은 회전 안 함 (균형 번호 = 모든 블록 같은 시간 → 최근 편향 없는 내용 매칭).
        spec: None | 'all' | '0-2-4' 같은 짝 번호 목록 (d_dim/2 = 5 짝). 원 코드의 짝 버그 (주파수가 짝을 가로질러 반복) 는 그대로 — 회전 짝 단위로 고른다."""
        if spec in (None, "", "none", "all"):
            self.t_pair_keep = None
            return
        P = self.d_dim // 2
        keep = sorted({int(t) for t in str(spec).replace(",", "-").split("-") if str(t).strip() != ""})
        if not keep or min(keep) < 0 or max(keep) >= P:
            raise ValueError(f"time_pairs: 0..{P - 1} 의 짝 번호 목록: {spec!r}")
        m = torch.zeros(self.d_dim, dtype=torch.bool)
        for p_ in keep:
            m[2 * p_: 2 * p_ + 2] = True
        self.register_buffer("t_pair_keep", m, persistent=False)

    def set_past_gate(self, on: bool):
        """2026-11-07 now/past gate (사용자 "게이팅 해봐 포스트 트레이닝으로"): query 마다 · head 마다 학습되는 δ 를 **이전 블록 key 의 logit 에만** 더한다.
        = 자기 블록 attention O_now 와 과거 블록 attention O_past 를 g = σ(LSE_past − LSE_now + δ) 로 섞는 것과 수식상 같다
        (각 갈래는 따로 정규화 → 자기 블록이 과거 몫을 뺏지 않는다). δ = past_gate(x) (Linear(dim → heads), 0 시작 → 기존과 같다).
        구현: q·k 에 차원 하나를 덧붙인다 (q 쪽 δ·√d, k 쪽 과거 1 · 자기 블록 0) → 마스크 없이 SDPA (flash) 그대로. 8 의 배수로 0 패딩."""
        if not on:
            return
        self.past_gate = nn.Linear(self.num_heads * self.head_dim, self.num_heads, bias=True)
        nn.init.zeros_(self.past_gate.weight); nn.init.zeros_(self.past_gate.bias)

    def _pg_sdpa(self, q, k, v, delta, n_own):
        """q (B,H,Nq,D), k/v (B,H,Nk,D), delta (B,H,Nq): 마지막 n_own 개 key = 자기 블록 (가산점 없음), 그 앞 = 과거 (logit + delta)."""
        B, H, Nq, D = q.shape
        Nk = k.size(2)
        pad = (-(D + 1)) % 8
        qe = torch.cat([q, (delta * math.sqrt(D)).unsqueeze(-1).to(q.dtype), q.new_zeros(B, H, Nq, pad)], -1)
        flag = torch.cat([k.new_ones(Nk - n_own), k.new_zeros(n_own)]).view(1, 1, Nk, 1).expand(B, H, Nk, 1)
        ke = torch.cat([k, flag, k.new_zeros(B, H, Nk, pad)], -1)
        ve = torch.cat([v, v.new_zeros(B, H, Nk, pad + 1)], -1)
        out = F.scaled_dot_product_attention(qe, ke, ve, dropout_p=self.proj_drop_prob, scale=1.0 / math.sqrt(D))
        return out[..., :D]

    # ------------------------------------------------------------------ 2026-11-08 시간 누적 하이브리드
    # (사용자 "Gated Attention 해보자 … 시간축 … 구조가 시간 누적이면", Qwen3-Next 의 Gated DeltaNet 3 : Gated Attention 1 을 블록 단위로)
    def set_out_gate(self, on: bool):
        """Qwen Gated Attention (G1, Qiu et al. NeurIPS 2025): attention 출력 (head 합친 뒤, proj 전) 에
        query 토큰별 · 채널별 σ(W x) 를 곱한다. x = attention 입력 (norm 뒤)."""
        if on:
            C = self.num_heads * self.head_dim
            self.out_gate = nn.Linear(C, C, bias=True)

    def set_block_mem(self, on: bool):
        """누적 층: 자기 블록 안 = softmax attention (공간 전체), 과거 블록 = **블록 기억 S 에서만** 읽는다.
        S (head 마다 D×D, key → value 사전) 는 블록 하나가 끝날 때마다 갱신된다 (블록 단위 delta 규칙, _mem_prep · _mem_step):
            S_t = α · S ;  S ← S_t + (Kᵀ diag(β) K + I)⁻¹ Kᵀ diag(β) (V − K S_t)
          = 옛 기억 쪽으로 규제한 가중 최소제곱 (블록 안 토큰은 순서 없이 동시에 쓴다). 기억 연산은 autocast 를 끄고 float32.
        α (잊기, head · 블록마다, 그 블록 입력 평균으로) · β (쓰기 세기, 토큰 · head 마다).
        기억 query/key = 시간 회전 없음 (행 · 열 RoPE 만) + L2 정규화 → 먼 과거는 내용 · 자리로 찾는다.
        출력 = σ(g_own) ⊙ O_own + σ(g_mem) ⊙ O_mem (시간축 게이트: 자기 블록 읽기와 기억 읽기가 softmax 로 경쟁하지 않는다)."""
        if not on:
            return
        C = self.num_heads * self.head_dim
        self.mem_gate = nn.Linear(C, 2 * C, bias=True)
        self.mem_beta = nn.Linear(C, self.num_heads, bias=True)
        self.mem_alpha = nn.Linear(C, self.num_heads, bias=True)

    def reset_block_mem(self):
        """predictor 의 _init_weights 뒤에 부른다: α 편향 3 (σ = 0.95, 처음엔 거의 안 잊는다) · β · 게이트 편향 0 (0.5)."""
        nn.init.constant_(self.mem_alpha.bias, 3.0)
        nn.init.zeros_(self.mem_beta.bias)
        nn.init.zeros_(self.mem_gate.bias)

    def _mem_qk(self, q_raw, k_raw, q, k):
        """기억용 query/key (float32): 앞 d_dim (시간) 차원은 회전 전 값, 행 · 열 · 나머지는 회전된 값. L2 정규화."""
        d = self.d_dim
        qm = torch.cat([q_raw[..., :d], q[..., d:]], -1).float()
        km = torch.cat([k_raw[..., :d], k[..., d:]], -1).float()
        return F.normalize(qm, dim=-1), F.normalize(km, dim=-1)

    def _mem_ab(self, x, nblk, S):
        """x (B, nblk*S, C) attention 입력 → α (B, H, nblk), β (B, H, nblk, S) (float32)."""
        B = x.size(0)
        alpha = torch.sigmoid(self.mem_alpha(x.view(B, nblk, S, -1).mean(2)).float()).permute(0, 2, 1)
        beta = torch.sigmoid(self.mem_beta(x).float()).view(B, nblk, S, -1).permute(0, 3, 1, 2)
        return alpha, beta

    @staticmethod
    def _mem_prep(km, v, beta, lam=1.0):
        """블록마다 병렬로: G = Kᵀ diag(β) K, U = Kᵀ diag(β) V, M = (G + λI)⁻¹. km (..., S, D) 정규화, v (..., S, Dv), beta (..., S).
        (2026-11-09 속도: 무거운 연산 (토큰 합 · 역행렬) 을 블록 축으로 한 번에 — 순차 루프에는 _mem_step 의 32×32 matmul 하나만 남는다.)"""
        kb = km * beta.unsqueeze(-1)
        G = kb.transpose(-1, -2) @ km
        U = kb.transpose(-1, -2) @ v
        eye = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)
        return torch.linalg.inv(G + lam * eye), U

    @staticmethod
    def _mem_step(Sm, M, U, alpha, lam=1.0):
        """블록 단위 delta 규칙 = 옛 기억 (α·S) 쪽으로 규제한 가중 최소제곱:
            S ← α·S + (G + λI)⁻¹ (U − G α·S)  =  M (λ α·S + U)        (I − M G = λ M 이라 두 식이 같다)
        Sm (B,H,D,Dv) 또는 None (빈 기억), M (B,H,D,D), U (B,H,D,Dv), alpha (B,H) → 새 Sm (float32).
        (2026-11-08 정정: 경사 한 걸음 (‖Kᵀ diag(β) K‖_F 로 나눔) 은 블록 안 소수 토큰 방향을 거의 못 쓴다 — 장난감 key 에서
         물체 토큰 10 / 256 의 값 복원 12–19 % vs 이 규칙 51–59 %. 고윳값이 [0, 1) 이라 항상 안정.)"""
        if Sm is None:
            return M @ U
        return M @ (lam * alpha[..., None, None] * Sm + U)

    def _mem_mix(self, x_in, o_own, o_mem):
        """x_in (B,N,C), o_own · o_mem (B,H,N,D) → proj 뒤 (B,N,C) = proj(σ(g_own) ⊙ O_own + σ(g_mem) ⊙ O_mem)."""
        B, N, C = x_in.shape
        g = torch.sigmoid(self.mem_gate(x_in))
        o1 = o_own.transpose(1, 2).reshape(B, N, C)
        o2 = o_mem.transpose(1, 2).reshape(B, N, C).to(o1.dtype)
        return self.proj_drop(self.proj(g[..., :C].to(o1.dtype) * o1 + g[..., C:].to(o1.dtype) * o2))

    def _time_pairs(self, rot, orig):
        """Harmonic 시간 RoPE: 고른 짝만 회전 결과, 나머지는 원래 값 (시간 0)."""
        if self.t_pair_keep is None:
            return rot
        return torch.where(self.t_pair_keep.to(rot.device), rot, orig)

    def set_time_scale(self, mode):
        """2026-11-06 (what–when 분리): 시간 RoPE 위치에 헤드별 척도를 곱한다. 행 · 열 위치는 그대로.
        mode: None | 'split:K' (앞 K 헤드 시간 없음, 나머지 그대로 — 고정) | 'zero' (전부 시간 없음 — 고정) | 'learned' (헤드별 학습, 1 로 시작)."""
        if mode in (None, "", "none"):
            self.t_scale = None
            return
        H = self.num_heads
        if mode == "learned":
            # 척도 = 1 + gain · u, u 는 0 에서 시작 (원래 모델과 같다). gain 100 = u 의 학습률을 100 배로 쓰는 재매개화:
            #   Adam 은 step 마다 대략 lr 만큼 움직이므로 lr 5e-5 로는 2,000 step 에 척도가 0.1 밖에 못 간다 → 100 배면 0 까지 갈 수 있다.
            self.t_learn = nn.Parameter(torch.zeros(H))
            self.t_gain = 100.0
            self.register_buffer("t_scale", torch.ones(H), persistent=False)   # 자리 표시 (separate_positions 가 t_learn 을 쓴다)
        elif mode == "zero":
            self.register_buffer("t_scale", torch.zeros(H), persistent=False)
        elif str(mode).startswith("scale:"):     # 모든 헤드 같은 척도 δ (시간 RoPE 주기 × 1/δ). 2026-11-06 사용자 "주기를 늘리자"
            self.register_buffer("t_scale", torch.full((H,), float(str(mode).split(":", 1)[1])), persistent=False)
        elif str(mode).startswith("split:"):
            k = int(str(mode).split(":", 1)[1])
            if not 0 < k < H:
                raise ValueError(f"time_scale split:K 는 0 < K < {H}: {mode!r}")
            self.register_buffer("t_scale", torch.cat([torch.zeros(k), torch.ones(H - k)]), persistent=False)
        else:
            raise ValueError(f"time_scale: none | split:K | zero | scale:δ | learned — {mode!r}")

    def set_vo_time(self, mode):
        """2026-11-06 (사용자: "RoPE 구현해볼래"): 시간을 가져오기 (value · 출력) 에 싣는다 (VO 방식, Su Jianlin VO-RoPE 와 같은 꼴).
        value 의 앞 vo_dim 차원을 그 key 의 블록 시간만큼 회전하고, attention 합 뒤 출력을 query 의 블록 시간만큼 되돌린다
        → 가져온 내용이 (key 시간 − query 시간) 만큼 회전된 채로 합쳐진다 (상대 시간). 고르기 쪽 시간 제거는 set_time_scale('zero') 로 따로.
        세기 = gain · u (헤드별, u 는 0 에서 시작 → 처음엔 회전 없음). 정확한 짝 회전 (인접 두 차원이 같은 각도), 주파수 = 2π / (16 · 2^k)."""
        if mode in (None, "", "none"):
            self.vo_on = False
            return
        if mode != "learned":
            raise ValueError(f"vo_time: none | learned — {mode!r}")
        self.vo_on = True
        self.vo_learn = nn.Parameter(torch.zeros(self.num_heads))
        self.vo_gain = 100.0
        self.vo_dim = 10
        self.register_buffer("vo_freq", 2 * math.pi / (16.0 * 2.0 ** torch.arange(self.vo_dim // 2, dtype=torch.float32)), persistent=False)

    def _vo(self, x, t, sign):
        """x (B, heads, N, D), t (B, heads, N) 블록 시간. 앞 vo_dim 차원을 sign · 세기 · t · 주파수 만큼 정확히 회전."""
        sc = (self.vo_gain * self.vo_learn).view(1, -1, 1, 1)
        ang = sign * sc * t.float().unsqueeze(-1) * self.vo_freq                  # (B, heads, N, vo_dim/2)
        c, sn = ang.cos(), ang.sin()
        xr = x[..., : self.vo_dim].float()
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        rot = torch.stack([x1 * c - x2 * sn, x1 * sn + x2 * c], dim=-1).flatten(-2)
        return torch.cat([rot.to(x.dtype), x[..., self.vo_dim:]], dim=-1)

    def _get_frame_pos(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        frame_ids = 1.0 * frame_ids
        if self.t_scale is not None:          # 2026-11-06 헤드별 시간 척도 (행 · 열은 위에서 원래 frame_ids 로 이미 뺐다)
            t = (1.0 + self.t_gain * self.t_learn) if hasattr(self, "t_learn") else self.t_scale
            t = t.to(frame_ids.dtype)
            if frame_ids.dim() >= 2 and frame_ids.size(-2) == self.num_heads:
                frame_ids = frame_ids * t.view(-1, 1)
            else:
                frame_ids = frame_ids.unsqueeze(-2) * t.view(-1, 1)
        return frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward_kv(self, x, mask, H, W, past=None):
        """KV 캐시 경로 (2026-09-30, ctx_state rollout). x (B, N, C) **새 토큰만**, mask (B, N) 그 절대 인덱스,
        past = 이전 토큰들의 (k, v) (RoPE 적용 뒤, (B, heads, M, D)) 또는 None.
        새 query 는 [past ; 새 토큰] key 를 **전부** 본다 — prefix(문맥 양방향 + 미래 block-causal) 에서
        새 블록 하나를 붙일 때와 정확히 같다 (문맥 prefill 은 past=None 으로 문맥끼리만).
        RoPE 위치 계산은 forward 와 같다. 마스크가 없어 SDPA 가 flash 를 탄다. return (출력 (B, N, C), (k_all, v_all))."""
        B, N, C = x.size()
        m = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        d_mask, h_mask, w_mask = self.separate_positions(m, H, W)
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = 0
        qd = self._time_pairs(rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask), q[..., s : s + self.d_dim])
        kd = self._time_pairs(rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask), k[..., s : s + self.d_dim])
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        if self.vo_on:                                       # 새 토큰 value 를 자기 블록 시간만큼 회전 (과거 캐시는 이미 회전됨)
            t_raw = (m // int(H * W)).float()
            v = self._vo(v, t_raw, +1.0)
        if getattr(self, "mem_gate", None) is not None:      # 2026-11-08 누적 층: past = 블록 기억 S (B,H,D,D) 또는 None. 새 토큰 = 블록 하나
            if N != int(H * W) or self.vo_on:
                raise ValueError(f"block_mem forward_kv: 블록 하나씩만 (N={N}, H·W={int(H * W)}), vo_time 과 같이 못 쓴다")
            o_own = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
            alpha, beta = self._mem_ab(x, 1, N)
            with torch.autocast(device_type=x.device.type, enabled=False):   # 기억은 float32
                qm, km = self._mem_qk(qkv[0], qkv[1], q, k)
                vf = v.float()
                o_mem = torch.zeros_like(vf) if past is None else qm @ past
                M, U = self._mem_prep(km, vf, beta[:, :, 0])
                Sm = self._mem_step(past, M, U, alpha[:, :, 0])
            return self._mem_mix(x, o_own, o_mem), Sm
        x_in = x
        n_new = k.size(2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        if getattr(self, "past_gate", None) is not None and past is not None:   # 2026-11-07 now/past gate: 캐시 = 과거, 새 토큰 = 자기 블록
            delta = self.past_gate(x).transpose(1, 2).float()                    # (B, H, N)
            x = self._pg_sdpa(q, k, v, delta, n_new)
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
        if self.vo_on:                                       # 출력을 query 블록 시간만큼 되돌림 → 상대 시간
            x = self._vo(x, t_raw, -1.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        if getattr(self, "out_gate", None) is not None:      # 2026-11-08 Qwen 출력 게이트
            x = x * torch.sigmoid(self.out_gate(x_in)).to(x.dtype)
        x = self.proj_drop(self.proj(x))
        return x, (k, v)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)
        else:
            mask = torch.arange(int(T * H * W), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)

        # -- snap spatial positions to grid size
        h_mask *= self.grid_size / H
        w_mask *= self.grid_size / W

        # -- split out action tokens from sequence
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # [B, T, 1+H*W, D]

            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                a = x[:, :, i : i + 1, :].flatten(1, 2)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
                # --
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                qr = q[..., self.d_dim :]
                kr = k[..., self.d_dim :]
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        s = 0
        # Rotate depth
        qd = self._time_pairs(rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask), q[..., s : s + self.d_dim])
        kd = self._time_pairs(rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask), k[..., s : s + self.d_dim])
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, T, A, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)

            q = merge_(q, action_q)
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if self.vo_on:                                       # 2026-11-06 시간을 value 에 (set_vo_time)
            if action_tokens > 0:
                raise ValueError("vo_time 은 action_tokens 와 같이 못 쓴다")
            _ids = mask if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1)
            t_raw = (_ids // int(H * W)).float()
            v = self._vo(v, t_raw, +1.0)
        if getattr(self, "mem_gate", None) is not None:
            # 2026-11-08 누적 층 (teacher forcing, 블록 인과 = PrefixSpec(1, nb-1)): 자기 블록 softmax 는 (head, 블록) 을 한 묶음으로 SDPA 한 번,
            #   기억은 블록 순서대로 쓴다 (블록 b 는 블록 < b 까지 쓴 기억을 읽는다 — forward_kv 와 같은 값)
            if isinstance(attn_mask, PrefixSpec):
                if attn_mask.n_ctx != 1 or not attn_mask.future_causal:
                    raise ValueError("block_mem 은 블록 인과 (PrefixSpec(1, nb-1), kind=ar) 전용")
                S = attn_mask.tokens_per_block
            elif attn_mask is None and N == int(H * W):     # 블록 하나뿐 (forward_seq nb = 1)
                S = N
            else:
                raise ValueError("block_mem 은 블록 인과 (PrefixSpec(1, nb-1), kind=ar) 전용")
            if action_tokens or self.vo_on or N % S:
                raise ValueError("block_mem: action_tokens · vo_time 없음, 토큰 수는 블록 정렬")
            nb, Dh = N // S, q.size(-1)
            o_own = F.scaled_dot_product_attention(q.reshape(B, -1, S, Dh), k.reshape(B, -1, S, Dh), v.reshape(B, -1, S, Dh),
                                                   dropout_p=self.proj_drop_prob).reshape(B, self.num_heads, N, Dh)
            alpha, beta = self._mem_ab(x, nb, S)
            with torch.autocast(device_type=x.device.type, enabled=False):   # 기억은 float32 (bf16 행렬곱 누적 오차를 피한다)
                qm, km = self._mem_qk(qkv[0], qkv[1], q, k)
                H_, D_ = qm.size(1), qm.size(-1)
                vf = v.float()
                Dv = vf.size(-1)
                states = [qm.new_zeros(B, H_, D_, Dv)]                     # 블록 0 이 읽는 기억 = 빈 기억
                if nb > 1:                                                  # 블록 0..nb−2 가 쓴다 (마지막 블록의 쓰기는 아무도 안 읽는다)
                    M, U = self._mem_prep(km.reshape(B, H_, nb, S, D_)[:, :, :-1], vf.reshape(B, H_, nb, S, Dv)[:, :, :-1],
                                          beta[:, :, :-1])
                    Sm = None
                    for b in range(nb - 1):
                        Sm = self._mem_step(Sm, M[:, :, b], U[:, :, b], alpha[:, :, b])
                        states.append(Sm)
                o_mem = (qm.reshape(B, H_, nb, S, D_) @ torch.stack(states, 2)).reshape(B, H_, N, Dv)   # 블록 b 는 블록 < b 까지 쓴 기억을 읽는다
            return self._mem_mix(x, o_own, o_mem)
        x_in = x
        # 2026-09-22: (N,N) dense bool 마스크는 flash 를 못 타 math/mem-eff 경로로 떨어진다
        # (실측 prefix 3.96 s vs full 1.85 s, B=28 N=6144). BlockMask 를 주면 flex_attention 의
        # **블록 희소 fused 커널**로 돈다 (rollout_predictor.build_prefix_block_mask).
        if isinstance(attn_mask, PrefixSpec):
            # ★ prefix 마스크를 **마스크 없이** 계산한다 (2026-09-22). 문맥 query 는 문맥 key 만,
            #   미래 블록 t 의 query 는 [문맥 ; 미래 <= t] key 를 본다 — 각각이 dense 라 flash 를 탄다.
            #   (N,N) bool 마스크는 mem-eff/math 경로(2 배 느림·점수 행렬 실체화)였고, flex_attention 은
            #   DDP 아래서 컴파일에 실패해 eager(점수 행렬 32 GB)로 떨어져 OOM 을 냈다. 이 경로는 컴파일 의존이 없다.
            Cq, S = attn_mask.n_ctx * attn_mask.tokens_per_block, attn_mask.tokens_per_block
            pg = getattr(self, "past_gate", None)
            if pg is not None and (attn_mask.n_ctx != 1 or not attn_mask.future_causal):
                raise ValueError("past_gate 는 블록 인과 (PrefixSpec(1, nb-1), kind=ar) 전용")
            delta = pg(x).transpose(1, 2).float() if pg is not None else None   # (B, H, N) — x = attention 입력 (norm 뒤)
            outs = [F.scaled_dot_product_attention(q[:, :, :Cq], k[:, :, :Cq], v[:, :, :Cq], dropout_p=self.proj_drop_prob)]
            if attn_mask.future_causal:
                for t in range(attn_mask.n_pred):
                    qs, ke = slice(Cq + t * S, Cq + (t + 1) * S), Cq + (t + 1) * S
                    if delta is not None:   # 2026-11-07 now/past gate: 자기 블록 = 마지막 S key
                        outs.append(self._pg_sdpa(q[:, :, qs], k[:, :, :ke], v[:, :, :ke], delta[:, :, qs], S))
                    else:
                        outs.append(F.scaled_dot_product_attention(q[:, :, qs], k[:, :, :ke], v[:, :, :ke], dropout_p=self.proj_drop_prob))
            else:   # prefix_full: 미래 query 전체가 [문맥 ; 미래 전부] key 를 본다 — SDPA 한 번
                outs.append(F.scaled_dot_product_attention(q[:, :, Cq:], k, v, dropout_p=self.proj_drop_prob))
            x = torch.cat(outs, dim=2)
            attn = None
        elif getattr(self, "past_gate", None) is not None:
            raise ValueError("past_gate 는 PrefixSpec (블록 인과) 경로 전용")
        elif _FlexBlockMask is not None and isinstance(attn_mask, _FlexBlockMask):
            x = _flex_attention(q, k, v, block_mask=attn_mask)
            attn = None
        elif attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.vo_on:
            x = self._vo(x, t_raw, -1.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        if getattr(self, "out_gate", None) is not None:      # 2026-11-08 Qwen 출력 게이트
            x = x * torch.sigmoid(self.out_gate(x_in)).to(x.dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_kv(self, x, mask, H, W, past=None):
        """KV 캐시 경로 (ACRoPEAttention.forward_kv). return (x, (k, v))."""
        y, kv = self.attn.forward_kv(self.norm1(x), mask, H, W, past)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, kv

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
