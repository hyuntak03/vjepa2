"""Prefix-마스크 predictor — 릴리즈 predictor 에서 **attention 마스크 하나만** 바꾼 판.

왜 (`z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §0-5, §6 — H1):

    릴리즈 predictor 의 미래 슬롯은 **전부 같은 mask token** 이라 첫 층 query 가 **위치로만**
    정해지고 자기 자리 주변을 본다. 물체의 과거 자리가 도달 범위(약 2~3 칸) 밖이면 못 찾는다.

사다리로 한 번에 한 변수만 바꾼다:

    A  릴리즈          full attention          + mask token      (기준선, 기존 코드)
    B  prefix          prefix 마스크            + mask token      ← 이 파일의 기본
    C  rollout         prefix 마스크            + 내용 되먹임      (rollout(), 나중)

**prefix 마스크** (B):
        문맥 j<C     미래 j>=C
  i<C :   ■            □        문맥은 서로 전부 본다 (이미 관측된 것이다). 미래는 못 본다
  i>=C:   ■            j<=i     미래는 문맥 전부 + 자기 이전 미래까지

  · 문맥을 인과적으로 가리지 않는 것이 요점이다 — 관측된 구간을 스스로 가릴 이유가 없다.
  · 미래끼리 순서를 지키게 하면, 위치 C+k 의 표현이 **깊이를 따라** C..C+k-1 을 거쳐 만들어진다.
    (layer 0 에서는 mask token 이 전부 같지만, layer 1 이후에는 앞 미래 슬롯이 이미 문맥을
     흡수한 상태라 정보가 앞으로 전파된다. "mask token 은 정보가 0" 은 layer 0 에서만 맞다.)

호출 규약은 **릴리즈와 동일**하다 — `forward(x, masks_x, masks_y, mask_index)`. 그래서
학습 루프(`app/vjepa_frozen/train.py`)도 채점기(`analysis/intphys2/model.py`)도 안 고친다.
파라미터 이름·모양도 릴리즈와 같아(`ACBlock` == `Block`, 2026-09-22 실측) `load_predictor: true`
strict 로드가 된다 → post-FT 팔과 scratch 팔(= **H0 의 첫 검정**)이 플래그 하나 차이다.

**C·K 분리**: `masks_y` 를 원하는 미래 블록만 담아 주면 된다. RoPE 가 절대 위치라 창을 자르지
않아도 "문맥 C 블록 → 그 뒤 K 블록" 이 그대로 성립한다 (`MaskSampler.prefix_window`).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import ACBlock, PrefixSpec
from src.utils.tensors import trunc_normal_

logger = logging.getLogger(__name__)

MASK_MODES = ("prefix", "prefix_full", "causal", "full", "ar", "ctx_ar", "ctx_state")
# ctx_ar (2026-09-27, 사용자 요청): 문맥은 **창 단위 z** (context encoder 가 C 블록을 한 번에, prefix 팔과 같다),
#   미래 자리는 블록별 LN(h) (TF) 또는 자기 예측 (rollout). 문맥끼리 양방향, 미래는 block-causal (= prefix 마스크).
#   위치 C-1 (문맥 마지막 블록) 의 출력이 블록 C 를, 미래 위치 t 의 출력이 블록 t+1 을 예측한다. 문맥 안쪽 위치에는
#   손실을 두지 않는다 (문맥이 양방향이라 다음 문맥 블록을 이미 본다). 문맥/미래 토큰은 type embedding 으로 구분.
# ar (2026-09-26): 팔 C. mask token 없이 **모든 블록의 LN(encoder) 토큰**을 넣고 블록 인과 attention 으로
#   위치 b 의 출력 = 블록 b+1 예측 (V-JEPA 2-AC 와 같은 LM 꼴). forward_seq / rollout 를 쓴다.
#   release 규약 forward(x, masks_x, masks_y) 는 이 모드에서 막는다 (입력 공간이 다르다).
# prefix_full (2026-09-24): prefix 와 같이 문맥은 문맥만 보지만, 미래 블록끼리는 인과 마스크 없이 서로 다 본다
#   (미래 query = 문맥 전부 + 미래 전부). "미래 쪽 block-causal 만 뺀" 대조 팔.


def build_prefix_mask(block_of: torch.Tensor, n_ctx_blocks: int, future_causal: bool = True) -> torch.Tensor:
    """토큰별 블록 번호 -> (N, N) bool attention 마스크. True = attend 허용.

    block_of : (N,) 각 토큰이 속한 블록(tubelet) 번호, **정렬된 순서**
    future_causal : True = 미래 query 는 문맥 + 자기 이전(포함) 미래 (prefix).
                    False = 미래 query 는 문맥 + 미래 전부 (prefix_full).
    """
    bi = block_of.view(-1, 1)          # query 쪽
    bj = block_of.view(1, -1)          # key 쪽
    is_ctx_q, is_ctx_k = bi < n_ctx_blocks, bj < n_ctx_blocks
    fut_k = (is_ctx_k | (bj <= bi)) if future_causal else torch.ones_like(is_ctx_k)
    # 문맥 query: 문맥만.  미래 query: 문맥 전부 + (자기 이전 미래 | 미래 전부)
    return torch.where(is_ctx_q, is_ctx_k, fut_k)


def build_prefix_block_mask(n_ctx_blocks: int, n_blocks: int, tokens_per_block: int, device):
    """prefix 마스크를 flex_attention BlockMask 로 (블록 = tubelet = 256 토큰, 커널 블록과 정렬).

    문맥 query 는 문맥만, 미래 query 는 문맥 전부 + 자기 이전(포함) 미래. build_prefix_mask 와 같은 의미.
    """
    from torch.nn.attention.flex_attention import create_block_mask
    S, C = tokens_per_block, n_ctx_blocks
    def mask_mod(b, h, q, kv):
        bq, bk = q // S, kv // S
        return torch.where(bq < C, bk < C, (bk < C) | (bk <= bq))
    N = n_blocks * S
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device, BLOCK_SIZE=S)


class HintCrossAttn(nn.Module):
    """힌트 cross-attention (2026-10-11, 사용자 결정) — 병목형, self-attn 뒤·MLP 앞에 residual 로 붙는다.

    query = 블록 b 토큰 (LN → down `dim→inner`), key/value = 블록 b+1 의 힌트 토큰 (LN → `inner`).
    블록 b 는 **블록 b+1 의 힌트만** 본다 (블록별로 묶어 SDPA, 마스크 없음). 3D RoPE 를 q·k 에 건다 (절대 위치).
    up `inner→dim` 은 weight·bias 모두 0 초기화 — 초기에는 branch 가 정확히 0 이라 힌트 없는 모델과 같다. gate 없음.
    """

    def __init__(self, dim, inner=96, num_heads=4, grid_size=16, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        if inner % num_heads:
            raise ValueError(f"hint inner {inner} 는 heads {num_heads} 의 배수여야 한다")
        self.num_heads, self.head_dim = num_heads, inner // num_heads
        self.rd = int(2 * ((self.head_dim // 3) // 2))        # ACRoPEAttention 과 같은 d/h/w 분할
        self.grid_size = grid_size
        self.norm_q, self.norm_kv = norm_layer(dim), norm_layer(dim)
        self.q = nn.Linear(dim, inner)
        self.kv = nn.Linear(dim, 2 * inner)
        self.up = nn.Linear(inner, dim)

    def zero_init(self):
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _rope(self, x, ids, H, W):
        from src.models.utils.modules import rotate_queries_or_keys
        T = H * W
        d = (ids // T).float()
        hh = ((ids % T) // W).float() * (self.grid_size / H)
        ww = ((ids % T) % W).float() * (self.grid_size / W)
        d, hh, ww = (t.unsqueeze(1).expand(-1, self.num_heads, -1) for t in (d, hh, ww))
        r = self.rd
        parts = [rotate_queries_or_keys(x[..., :r], pos=d), rotate_queries_or_keys(x[..., r:2 * r], pos=hh),
                 rotate_queries_or_keys(x[..., 2 * r:3 * r], pos=ww)]
        if 3 * r < self.head_dim:
            parts.append(x[..., 3 * r:])
        return torch.cat(parts, dim=-1)

    def forward(self, x, idx, hint, hint_idx, S, H, W):
        """x (B, nb*S, dim), idx (B, nb*S) 절대 인덱스; hint (B, nb*Nh, dim) = 블록 1..nb 의 힌트 (predictor_embed 뒤),
        hint_idx (B, nb*Nh) 그 절대 인덱스. 블록 b query ↔ hint 묶음 b. return (B, nb*S, dim) branch 출력."""
        B, N, _ = x.shape
        nb = N // S
        Nh = hint.size(1) // nb
        hd, nh = self.head_dim, self.num_heads
        q = self.q(self.norm_q(x)).view(B, N, nh, hd).transpose(1, 2)              # (B, nh, N, hd)
        k, v = self.kv(self.norm_kv(hint)).view(B, nb * Nh, 2, nh, hd).permute(2, 0, 3, 1, 4)
        q, k = self._rope(q, idx, H, W), self._rope(k, hint_idx, H, W)
        q = q.view(B, nh, nb, S, hd).transpose(1, 2).reshape(B * nb, nh, S, hd)
        k = k.view(B, nh, nb, Nh, hd).transpose(1, 2).reshape(B * nb, nh, Nh, hd)
        v = v.reshape(B, nh, nb, Nh, hd).transpose(1, 2).reshape(B * nb, nh, Nh, hd)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)            # (B*nb, nh, S, hd)
        o = o.view(B, nb, nh, S, hd).permute(0, 1, 3, 2, 4).reshape(B, N, nh * hd)
        return self.up(o)


class VisionTransformerPredictorPrefix(nn.Module):
    """릴리즈 `VisionTransformerPredictor` 와 같은 구조·같은 호출 규약, attention 마스크만 다르다."""

    def __init__(
        self,
        img_size=(256, 256), patch_size=16, num_frames=32, tubelet_size=2,
        embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), init_std=0.02,
        use_rope=True, use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=False,
        num_mask_tokens=10, zero_init_mask_tokens=True,
        mask_mode="prefix", n_registers=0, out_embed_dim=None, **kwargs,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode 는 {MASK_MODES}: {mask_mode!r}")
        self.mask_mode = mask_mode
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.tokens_per_block = self.grid_height * self.grid_width
        self.tubelet_size = tubelet_size
        self.max_blocks = num_frames // tubelet_size
        self.n_registers = int(n_registers)
        self.use_activation_checkpointing = use_activation_checkpointing
        self.init_std = init_std

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)])
        if self.n_registers:
            self.registers = nn.Parameter(torch.zeros(1, 1, self.n_registers, predictor_embed_dim))
        if mask_mode == "ctx_ar":
            # [0] = 문맥 z 토큰, [1] = 미래 토큰 (LN(h) 또는 자기 예측). 다른 모드는 state_dict 를 그대로 두기 위해 안 만든다
            self.type_embed = nn.Parameter(torch.zeros(2, 1, 1, predictor_embed_dim))
        elif mask_mode == "ctx_state":
            # [0] = 문맥 z 토큰, [1] = 관측 토큰 (TF: 블록별 LN(h) 를 predictor_embed), [2] = 상태 토큰 (rollout: 자기 은닉)
            self.type_embed = nn.Parameter(torch.zeros(3, 1, 1, predictor_embed_dim))
        # ar 되먹임 (2026-10-12): output (기본, AC 와 같다 — 출력 LN 을 다시 predictor_embed) | state (predictor 상태 384 를 그대로)
        self.ar_feedback = str(kwargs.get("ar_feedback", "output") or "output")
        if self.ar_feedback not in ("output", "state"):
            raise ValueError(f"ar_feedback 는 output | state: {self.ar_feedback!r}")
        if self.ar_feedback == "state":
            if mask_mode != "ar":
                raise ValueError("ar_feedback=state 는 mask_mode='ar' 전용")
            # [0] = 관측 토큰 (블록별 LN(h) 를 predictor_embed), [1] = 상태 토큰 (rollout: 이전 걸음의 predictor_norm 출력)
            self.type_embed = nn.Parameter(torch.zeros(2, 1, 1, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            ACBlock(use_rope=use_rope, grid_size=self.grid_height, grid_depth=self.max_blocks,
                    dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    use_sdpa=use_sdpa)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim or embed_dim, bias=True)
        # 힌트 cross-attn (2026-10-11, mask_mode=ar 전용). 0 이면 만들지 않는다 (state_dict 그대로)
        self.hint_cross_dim = int(kwargs.get("hint_cross_dim", 0) or 0)
        if self.hint_cross_dim:
            if mask_mode != "ar":
                raise ValueError("hint_cross_dim 은 mask_mode='ar' 전용")
            self.hint_attn = nn.ModuleList([
                HintCrossAttn(predictor_embed_dim, self.hint_cross_dim, int(kwargs.get("hint_cross_heads", 4)),
                              grid_size=self.grid_height, norm_layer=norm_layer)
                for _ in range(depth)])

        # prefix 구현: split(기본, 마스크 없는 블록별 dense SDPA) | flex(DDP 에서 컴파일 실패 사례 있음) | dense
        self.prefix_impl = str(kwargs.get("prefix_impl", "split"))
        self.use_flex = self.prefix_impl == "flex"
        self._bm_cache = {}
        self.apply(self._init_weights)
        self._rescale_blocks()
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        if self.n_registers:
            trunc_normal_(self.registers, std=init_std)
        if mask_mode in ("ctx_ar", "ctx_state") or self.ar_feedback == "state":
            trunc_normal_(self.type_embed, std=init_std)
        if self.hint_cross_dim:
            for m in self.hint_attn:
                m.zero_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for i, layer in enumerate(self.predictor_blocks, start=1):
            layer.attn.proj.weight.data.div_((2.0 * i) ** 0.5)
            layer.mlp.fc2.weight.data.div_((2.0 * i) ** 0.5)

    # ---------------------------------------------------------------- forward
    def forward(self, x, masks_x=None, masks_y=None, mask_index=0, has_cls=False, ar=None):
        """릴리즈와 같은 규약 (+ `ar=` 는 팔 C 학습용 분기, 아래).

        x       : (B, |masks_x|, embed_dim)  문맥 토큰 (context encoder 출력)
        masks_x : (B, Kx) 문맥 토큰의 flat 인덱스
        masks_y : (B, Ky) 예측할 토큰의 flat 인덱스  ← 여기 몇 블록을 담느냐가 K 다
        return  : (B, Ky, out_embed_dim)

        ar={"idx": (B,N), "n_ctx_blocks": C, "rollout_steps": R}  (mask_mode == "ar" 전용)
            x 는 **모든 블록**의 LN(encoder) 토큰 (B, N, embed_dim). (p_tf, p_ar) 를 돌려준다 —
            p_tf (B, N-S, out): 위치 0..nb-2 의 다음 블록 예측 (teacher forcing, 블록 1..nb-1 과 비교)
            p_ar (B, R*S, out): 문맥 C 블록에서 R 블록 자기회귀 (첫 스텝은 p_tf 의 위치 C-1 재사용 = AC 와 같다)
            DDP 가 forward 만 감싸므로 두 손실을 **한 forward 안**에서 낸다 (grad 동기화 때문).
        """
        if ar is not None and self.mask_mode == "ctx_state":
            # ar={"h_obs": (B, K*S, D) 블록별 LN(h) (TF 관측 입력), "idx": (B, (C+K)*S), "n_ctx_blocks": C, "rollout_steps": R}
            S = self.tokens_per_block
            idx, C, R = ar["idx"], int(ar["n_ctx_blocks"]), int(ar["rollout_steps"])
            # h_obs 가 None 이면 TF 를 건너뛴다 (loss.tf_weight = 0). rollout 은 KV 캐시판 (결과는 rollout_state 와 같다)
            p_tf = None
            if ar.get("h_obs") is not None:
                h_tf = self.forward_state(x, self.embed_obs(ar["h_obs"][:, :-S]), idx[:, : idx.size(1) - S], C)
                p_tf = self.render(h_tf)                                               # 블록 C..C+K-1 예측 (K 블록)
            p_ar = self.render(self.rollout_state_kv(x, idx[:, : C * S], R))
            return p_tf, p_ar
        if ar is not None and self.mask_mode == "ctx_ar":
            # ar={"h_fut": (B, K*S, D) 블록별 LN(h) 미래 GT, "idx": (B, (C+K)*S), "n_ctx_blocks": C, "rollout_steps": R}
            S = self.tokens_per_block
            idx, C, R = ar["idx"], int(ar["n_ctx_blocks"]), int(ar["rollout_steps"])
            h_fut = ar["h_fut"]
            p_tf = self.forward_mixed(x, h_fut[:, :-S], idx[:, : idx.size(1) - S], C)      # 블록 C..C+K-1 예측 (K 블록)
            p_ar = self.rollout_ctx(x, idx[:, : C * S], R, first=p_tf[:, :S])
            return p_tf, p_ar
        if ar is not None:
            if self.mask_mode != "ar":
                raise ValueError(f"ar= 분기는 mask_mode='ar' 전용 (지금 {self.mask_mode})")
            S = self.tokens_per_block
            idx, C, R = ar["idx"], int(ar["n_ctx_blocks"]), int(ar["rollout_steps"])
            p_tf = self.forward_seq(x[:, :-S], idx[:, :-S], hint=ar.get("hint"))   # 힌트는 TF 에만. rollout 은 힌트 없음
            if getattr(self, "ar_kv_rollout", True):          # 2026-10-11: KV 캐시판 (값·grad 동일, 긴 K 에서 빠르다)
                p_ar = self.rollout_kv(x[:, : C * S], idx[:, : C * S], R)
            else:
                first = p_tf[:, (C - 1) * S: C * S]
                p_ar = self.rollout(x[:, : C * S], idx[:, : C * S], R, first=first)
            return p_tf, p_ar
        if self.mask_mode in ("ar", "ctx_ar", "ctx_state"):
            raise RuntimeError(f"mask_mode='{self.mask_mode}' predictor 는 mask token 규약 forward(x, masks_x, masks_y) 를 지원하지 않는다 — "
                               "forward_seq(LN(h) 전 블록) / rollout(LN(h) 문맥, R) 을 쓸 것 (app/vjepa_frozen/ar.py)")
        if isinstance(masks_x, list):
            masks_x = masks_x[0]
        if isinstance(masks_y, list):
            masks_y = masks_y[0]
        B = x.size(0)
        x = self.predictor_embed(x)                                   # (B, Kx, D)
        mt = self.mask_tokens[mask_index % self.num_mask_tokens]
        y = mt.expand(B, masks_y.size(1), -1)                         # (B, Ky, D)

        idx = torch.cat([masks_x, masks_y], dim=1)                    # (B, N)
        seq = torch.cat([x, y], dim=1)                                # (B, N, D)
        order = torch.argsort(idx, dim=1)
        idx = torch.gather(idx, 1, order)
        seq = torch.gather(seq, 1, order.unsqueeze(-1).expand(-1, -1, seq.size(-1)))

        S, N = self.tokens_per_block, idx.size(1)
        block_of = idx[0] // S                                        # 배치 안에서 같은 배치(마스크 공유)
        n_ctx_blocks = int(masks_x.size(1) // S)
        if self.mask_mode == "full":
            attn_mask = None
        elif self.mask_mode in ("prefix", "prefix_full"):
            # ★ 블록 희소 커널. 문맥·미래가 인덱스 순으로 정렬돼 있고 블록 경계(256 토큰)가 정확히
            #   맞아야 한다 — prefix_window / temporal_prefix 가 그렇게 만든다. 안 맞으면 dense 로 폴백.
            fut_causal = self.mask_mode == "prefix"
            n_blocks = N // S
            contiguous = bool((idx[0] == torch.arange(N, device=idx.device)).all())
            key = (n_ctx_blocks, n_blocks, str(seq.device))
            if contiguous and self.prefix_impl == "split":
                attn_mask = PrefixSpec(n_ctx_blocks, n_blocks - n_ctx_blocks, S, future_causal=fut_causal)  # 마스크 없음, flash
            elif contiguous and self.prefix_impl == "flex" and fut_causal:
                attn_mask = self._bm_cache.get(key)
                if attn_mask is None:
                    attn_mask = build_prefix_block_mask(n_ctx_blocks, n_blocks, S, seq.device)
                    self._bm_cache[key] = attn_mask
            else:
                attn_mask = build_prefix_mask(block_of, n_ctx_blocks, future_causal=fut_causal).to(seq.device)
        else:                                                          # causal: 문맥끼리도 인과
            attn_mask = (block_of.view(1, -1) <= block_of.view(-1, 1)).to(seq.device)

        # ⚠️ RoPE 는 `mask` 인자로 **절대 토큰 인덱스**를 받는다. 그래야 창을 안 잘라도
        #    "블록 C+3" 이 제 위치를 갖는다 (C·K 분리의 근거).
        h = seq
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, idx, attn_mask, None, self.grid_height, self.grid_width, 0, use_reentrant=False)
            else:
                h = blk(h, mask=idx, attn_mask=attn_mask, T=None,
                        H=self.grid_height, W=self.grid_width, action_tokens=0)
        h = self.predictor_proj(self.predictor_norm(h))               # (B, N, out)

        # 정렬을 되돌려 masks_y 순서로 뽑는다
        inv = torch.argsort(order, dim=1)
        h = torch.gather(h, 1, inv.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return h[:, masks_x.size(1):, :]

    # ---------------------------------------------------------------- 팔 C (ar, 2026-09-26)
    def _block_hint(self, blk, xa, h, idx, attn_mask, hx, hidx, gate):
        Hg, Wg = self.grid_height, self.grid_width
        h = h + blk.drop_path(blk.attn(blk.norm1(h), mask=idx, attn_mask=attn_mask, T=None, H=Hg, W=Wg, action_tokens=0))
        y = xa(h, idx, hx, hidx, self.tokens_per_block, Hg, Wg)
        h = h + (y if gate == 1.0 else gate * y)
        return h + blk.drop_path(blk.mlp(blk.norm2(h)))

    def _run_blocks_hint(self, h, idx, attn_mask, hx, hidx, gate):
        for blk, xa in zip(self.predictor_blocks, self.hint_attn):
            if self.use_activation_checkpointing and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(self._block_hint, blk, xa, h, idx, attn_mask, hx, hidx, gate,
                                                      use_reentrant=False)
            else:
                h = self._block_hint(blk, xa, h, idx, attn_mask, hx, hidx, gate)
        return h

    def _run_blocks(self, h, idx, attn_mask):
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, idx, attn_mask, None, self.grid_height, self.grid_width, 0, use_reentrant=False)
            else:
                h = blk(h, mask=idx, attn_mask=attn_mask, T=None, H=self.grid_height, W=self.grid_width, action_tokens=0)
        return h

    def forward_seq(self, x, idx, hint=None):
        """팔 C. x (B, N, embed_dim) = 블록 0..nb-1 의 **LN 된** encoder 토큰 (블록 정렬·연속), idx (B, N) 절대 인덱스.
        블록 인과 attention (블록 b 의 query 는 블록 <= b 의 key). return (B, N, out): 위치 b 의 출력 = 블록 b+1 예측,
        출력에도 affine-free LN (AC `normalize_reps`) — 되먹임 입력과 타깃이 같은 공간에 있게 한다."""
        S = self.tokens_per_block
        nb = x.size(1) // S
        if x.size(1) != nb * S:
            raise ValueError(f"forward_seq: 토큰 수 {x.size(1)} 가 블록({S}) 정렬이 아니다")
        h = self._ar_embed_obs(x)
        # 블록 0 을 '문맥 1 블록' 으로, 나머지를 인과 미래로 두면 PrefixSpec 이 그대로 블록 인과다 (flash 경로, 마스크 없음)
        spec = PrefixSpec(1, nb - 1, S, future_causal=True) if nb > 1 else None
        if hint is not None and hint["x"].size(1) > 0:
            # hint={"x": (B, nb*Nh, embed_dim) LN 된 힌트 encoder 토큰 (블록 1..nb), "idx": (B, nb*Nh) 절대 인덱스, "gate": float}
            if not self.hint_cross_dim:
                raise ValueError("hint 를 받았는데 hint_cross_dim = 0")
            hx = self.predictor_embed(hint["x"])                    # 공유 투영 (type embedding 없음)
            h = self._run_blocks_hint(h, idx, spec, hx, hint["idx"], float(hint.get("gate", 1.0)))
        else:
            h = self._run_blocks(h, idx, spec)                      # 힌트 0 개 → cross-attn 을 부르지 않는다
        h = self.predictor_proj(self.predictor_norm(h))
        return torch.nn.functional.layer_norm(h, (h.size(-1),))

    def rollout(self, x_ctx, idx_ctx, n_steps, first=None):
        """팔 C. 문맥 C 블록의 LN(h) 에서 `n_steps` 블록을 자기회귀로 채운다 (KV 캐시 없음, 매 스텝 전체 재계산).
        first : 첫 스텝 예측을 밖에서 주면 (TF pass 의 위치 C-1 출력) 재사용한다 — 수학적으로 같은 값이다 (인과).
        return (B, n_steps*S, out). 되먹임은 grad 를 끊지 않는다 (AC 와 같다)."""
        if self.ar_feedback == "state":
            raise NotImplementedError("ar_feedback=state 는 KV 판 rollout_kv 만 지원한다 (ar_kv_rollout=True)")
        S = self.tokens_per_block
        seq, idx, preds = x_ctx, idx_ctx, []
        for r in range(int(n_steps)):
            p = first if (r == 0 and first is not None) else self.forward_seq(seq, idx)[:, -S:]
            preds.append(p)
            if r + 1 < n_steps:
                seq = torch.cat([seq, p], dim=1)
                idx = torch.cat([idx, idx[:, -S:] + S], dim=1)
        return torch.cat(preds, dim=1)


    # ---------------------------------------------------------------- ctx_ar (2026-09-27)
    def forward_mixed(self, z_ctx, h_fut, idx, n_ctx_blocks):
        """ctx_ar. z_ctx (B, C*S, embed_dim) 창 단위 문맥 z (LN 안 함, prefix 팔과 같다),
        h_fut (B, F*S, embed_dim) 미래 자리 입력 (블록별 LN(h) 또는 자기 예측, F >= 0), idx (B, (C+F)*S) 절대 인덱스.
        attention = prefix 마스크 (문맥 양방향, 미래 block-causal). return (B, (F+1)*S, out), LN 적용:
        [위치 C-1 출력 = 블록 C 예측 ; 미래 위치 C..C+F-1 출력 = 블록 C+1..C+F 예측]."""
        S, C = self.tokens_per_block, int(n_ctx_blocks)
        F_ = h_fut.size(1) // S
        x = self.predictor_embed(z_ctx) + self.type_embed[0]
        if F_ > 0:
            y = self.predictor_embed(h_fut) + self.type_embed[1]
            x = torch.cat([x, y], dim=1)
        spec = PrefixSpec(C, F_, S, future_causal=True)
        h = self._run_blocks(x, idx, spec)
        h = self.predictor_proj(self.predictor_norm(h))[:, (C - 1) * S:]
        return torch.nn.functional.layer_norm(h, (h.size(-1),))

    def rollout_ctx(self, z_ctx, idx_ctx, n_steps, first=None):
        """ctx_ar. 문맥 z 뒤에 자기 예측을 붙여 n_steps 블록을 굴린다. first = TF 의 블록 C 예측 재사용 (같은 값)."""
        S = self.tokens_per_block
        C = idx_ctx.size(1) // S
        fut = z_ctx.new_zeros(z_ctx.size(0), 0, z_ctx.size(-1))
        idx = idx_ctx
        preds = []
        for r in range(int(n_steps)):
            if r == 0 and first is not None:
                p = first
            else:
                p = self.forward_mixed(z_ctx, fut, idx, C)[:, -S:]
            preds.append(p)
            if r + 1 < n_steps:
                fut = torch.cat([fut, p], dim=1)
                idx = torch.cat([idx, idx[:, -S:] + S], dim=1)
        return torch.cat(preds, dim=1)


    # ---------------------------------------------------------------- ctx_state (2026-09-30)
    # ctx_ar 와 **되먹임 공간 하나만** 다르다 (사용자 결정): rollout 이 그린 출력(LN(proj(.)), 1280) 대신
    # **그리기 전 predictor 상태** (predictor_norm 출력, 384) 를 다음 걸음 입력으로 되먹인다.
    #   문맥   : z (context encoder, 문맥 창 한 번, 양방향) + type[0]. 걸음 내내 고정
    #   TF     : 미래 자리 = predictor_embed(블록별 LN(h)) + type[1] (관측 토큰, 한 forward 로 병렬)
    #   rollout: 미래 자리 = 이전 걸음의 상태 + type[2] (상태 토큰). 문맥 z 만 주고 시작
    #   attention = prefix (문맥 양방향, 미래 block-causal). 손실은 render(상태) = LN(proj(norm(.))) 에 건다
    def embed_obs(self, h_obs):
        """블록별 LN(h) (B, F*S, embed_dim) -> 관측 토큰 (B, F*S, predictor_dim)."""
        return self.predictor_embed(h_obs) + self.type_embed[1]

    def render(self, s):
        """상태 (B, n, predictor_dim, predictor_norm 뒤) -> encoder 공간 LN (B, n, out). 손실·채점은 여기서."""
        return torch.nn.functional.layer_norm(self.predictor_proj(s), (self.predictor_proj.out_features,))

    def forward_state(self, z_ctx, fut_tok, idx, n_ctx_blocks):
        """z_ctx (B, C*S, embed_dim), fut_tok (B, F*S, predictor_dim) 이미 임베딩된 미래 토큰 (관측 또는 상태, F >= 0),
        idx (B, (C+F)*S). return 상태 (B, (F+1)*S, predictor_dim) = predictor_norm 출력:
        [위치 C-1 = 블록 C 의 상태 ; 미래 위치 C..C+F-1 = 블록 C+1..C+F 의 상태]."""
        S, C = self.tokens_per_block, int(n_ctx_blocks)
        F_ = fut_tok.size(1) // S
        x = self.predictor_embed(z_ctx) + self.type_embed[0]
        if F_ > 0:
            x = torch.cat([x, fut_tok], dim=1)
        spec = PrefixSpec(C, F_, S, future_causal=True)
        h = self._run_blocks(x, idx, spec)
        return self.predictor_norm(h[:, (C - 1) * S:])

    def rollout_state(self, z_ctx, idx_ctx, n_steps, first=None):
        """문맥 z 뒤에 **자기 상태**를 붙여 n_steps 블록을 굴린다 (grad 유지, KV 캐시 없음).
        first = TF 의 위치 C-1 상태 재사용 (문맥만 보므로 같은 값). return 상태 (B, n_steps*S, predictor_dim)."""
        S = self.tokens_per_block
        C = idx_ctx.size(1) // S
        fut = z_ctx.new_zeros(z_ctx.size(0), 0, self.predictor_embed.out_features)
        idx = idx_ctx
        states = []
        for r in range(int(n_steps)):
            if r == 0 and first is not None:
                s = first
            else:
                s = self.forward_state(z_ctx, fut, idx, C)[:, -S:]
            states.append(s)
            if r + 1 < n_steps:
                fut = torch.cat([fut, s + self.type_embed[2]], dim=1)
                idx = torch.cat([idx, idx[:, -S:] + S], dim=1)
        return torch.cat(states, dim=1)


    def rollout_state_kv(self, z_ctx, idx_ctx, n_steps):
        """rollout_state 의 **KV 캐시판** (2026-09-30). 결과·gradient 가 같다 (tests: z_training/harness 의 검증 참고):
        문맥은 미래를 안 보고 (문맥끼리 양방향), 미래는 block-causal 이라 이미 계산한 토큰의 층별 (k, v) 가
        뒤에 블록을 붙여도 안 변한다. 그래서 문맥을 한 번 prefill 하고, 걸음마다 **새 블록 S 토큰만** 계산한다.
        (지금 rollout_state 는 걸음마다 [문맥 ; 미래 전부] 를 다시 돈다 — C=8,K=8 에서 92 블록 vs 16 블록)
        ⚠️ activation checkpointing 은 이 경로에서 안 쓴다 (새 블록 활성값만 쌓여 메모리가 작다).
        return 상태 (B, n_steps*S, predictor_dim)."""
        S, Hg, Wg = self.tokens_per_block, self.grid_height, self.grid_width
        h = self.predictor_embed(z_ctx) + self.type_embed[0]
        caches = []
        for blk in self.predictor_blocks:                                       # prefill: 문맥끼리만
            h, kv = blk.forward_kv(h, idx_ctx, Hg, Wg, None)
            caches.append(kv)
        s = self.predictor_norm(h[:, -S:])                                      # 위치 C-1 -> 블록 C 의 상태
        states = [s]
        pos = idx_ctx[:, -S:]
        for _ in range(1, int(n_steps)):
            pos = pos + S                                                       # 상태 s_{C+r-1} 은 블록 C+r-1 자리에 놓인다
            h = s + self.type_embed[2]
            for i, blk in enumerate(self.predictor_blocks):
                h, caches[i] = blk.forward_kv(h, pos, Hg, Wg, caches[i])
            s = self.predictor_norm(h)                                          # -> 블록 C+r 의 상태
            states.append(s)
        return torch.cat(states, dim=1)


    # ---------------------------------------------------------------- ar KV 캐시 (2026-10-11)
    def _ar_embed_obs(self, x):
        """ar 입력 임베딩: 블록별 LN(h) -> predictor 공간. state 되먹임이면 관측 표시 type[0] 를 더한다."""
        h = self.predictor_embed(x)
        return h + self.type_embed[0] if self.ar_feedback == "state" else h

    def _ar_step_kv(self, x_blk, idx_blk, caches, embedded=False):
        """ar: 블록 하나 (B, S, embed_dim, LN 된 encoder 공간) 를 넣고 다음 블록 예측 (LN) 을 낸다. caches 는 층별 (k, v) 또는 None.
        embedded=True 면 x_blk 가 이미 predictor 공간 토큰이다 (state 되먹임). return (예측 LN, caches, 상태 = predictor_norm 출력)."""
        Hg, Wg = self.grid_height, self.grid_width
        h = x_blk if embedded else self._ar_embed_obs(x_blk)
        new = []
        for i, blk in enumerate(self.predictor_blocks):
            h, kv = blk.forward_kv(h, idx_blk, Hg, Wg, None if caches is None else caches[i])
            new.append(kv)
        st = self.predictor_norm(h)
        h = self.predictor_proj(st)
        return torch.nn.functional.layer_norm(h, (h.size(-1),)), new, st

    def rollout_kv(self, x_ctx, idx_ctx, n_steps):
        """`rollout` (ar) 의 **KV 캐시판**. ar 은 문맥까지 block-causal 이라 이미 계산한 블록의 층별 (k, v) 가
        뒤에 블록을 붙여도 안 변한다 → 문맥을 블록 하나씩 채우고 (C 번), 걸음마다 새 블록 S 토큰만 계산한다.
        값·grad 는 `rollout` 과 같다 (검증: z_training/RESULTS 참고). return (B, n_steps*S, out) LN."""
        S = self.tokens_per_block
        C = x_ctx.size(1) // S
        caches, p, st = None, None, None
        for b in range(C):                                   # 문맥: 블록 b 는 블록 <= b 만 본다
            p, caches, st = self._ar_step_kv(x_ctx[:, b * S:(b + 1) * S], idx_ctx[:, b * S:(b + 1) * S], caches)
        preds = [p]                                          # 위치 C-1 출력 = 블록 C 예측
        pos = idx_ctx[:, -S:]
        state = self.ar_feedback == "state"
        for _ in range(1, int(n_steps)):
            pos = pos + S
            if state:   # 2026-10-12: 그리기 전 상태 (predictor_norm 출력, 384) + type[1] 을 블록 자리에 그대로 되먹인다 (grad 유지)
                p, caches, st = self._ar_step_kv(st + self.type_embed[1], pos, caches, embedded=True)
            else:       # 자기 예측 (LN) 을 다음 입력으로 (AC 와 같다)
                p, caches, st = self._ar_step_kv(p, pos, caches)
            preds.append(p)
        return torch.cat(preds, dim=1)


def vit_prefix_predictor(**kwargs) -> VisionTransformerPredictorPrefix:
    return VisionTransformerPredictorPrefix(mlp_ratio=4, qkv_bias=True, **kwargs)
