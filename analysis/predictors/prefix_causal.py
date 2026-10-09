"""block-causal / future-only predictor (채점 전용, `kind: prefix`).

릴리즈 V-JEPA 2 predictor 는 정렬된 [문맥 토큰 ; 미래 mask 토큰] 시퀀스에 **full
self-attention** 을 건다. 이 변종은 같은 파라미터 위에 attention mask 하나만 얹는다:

    * 문맥(context) 끼리 — **양방향 전부** (prefix-LM 의 prefix)
    * 문맥 -> 미래       — **차단**. 문맥 표현은 미래 mask 토큰을 보지 못한다
    * 미래 -> 문맥       — 허용 (문맥은 언제나 과거다)
    * 미래 -> 미래       — **block-causal**. 같은 시간 블록(tubelet slot) 안은 양방향,
                           다음 블록은 못 본다

"block" 은 **시간 tubelet slot** 이다. 토큰 index 는 PatchEmbed3D 관례로
`idx = t * S + s` (S = (img/patch)^2 = slot 당 공간 토큰 수) 이므로 `slot = idx // S`.

한 줄 식 (query i, key j):

    allowed[i, j] = is_ctx[j]  OR  (is_tgt[i] AND slot[j] <= slot[i])

state_dict 키는 릴리즈 predictor 와 **완전히 같다** (`VisionTransformerPredictor` 상속,
파라미터를 하나도 더하거나 빼지 않는다). 그래서 같은 latest.pt 를 두 kind 로 다 읽을 수
있고, 둘의 차이는 **오직 attention mask** 다. 그 차이가 점수에 얼마나 미치는지는
`z_research/scripts/analysis/check_prefix_predictor.py` 가 재서 보여 준다.

⚠️ 미검증 — 이 파일이 재현하는 것은 "block_causal_future_only" 라는 이름과
   체크포인트 `arch.kind: prefix` / `mask.type: prefix_window` 로부터 읽은 규칙이다.
   학습 코드(Ariel 쪽 fork)를 대조한 것이 아니다. 남은 자유도:
     (1) 미래가 **자기 블록**을 양방향으로 보는가 (여기서는 본다)
     (2) causal 단위가 블록인가 토큰인가 (여기서는 블록; `causal_unit` 로 바꿀 수 있다)
     (3) 학습 때 문맥 창이 항상 clip 의 맨 앞이었는가 (RoPE 위치가 걸린다)
"""
from __future__ import annotations

import torch

from src.masks.utils import apply_masks
from src.models.predictor import VisionTransformerPredictor
from src.utils.tensors import repeat_interleave_batch


def prefix_causal_attn_mask(
    token_idx: torch.Tensor,        # (B, N) 정렬된 시퀀스 각 자리의 **전역 토큰 index**
    is_ctx: torch.Tensor,           # (B, N) 그 자리가 문맥 토큰인가
    spatial_tokens: int,            # slot 당 공간 토큰 수 S
    causal_unit: str = "block",     # "block" (시간 slot) | "token" (전역 index)
) -> torch.Tensor:
    """-> (B, 1, N, N) bool. True = attend 가능 (SDPA 관례)."""
    if causal_unit == "block":
        order = token_idx // spatial_tokens          # 시간 slot
    elif causal_unit == "token":
        order = token_idx
    else:
        raise ValueError(f"causal_unit must be 'block' or 'token'; got {causal_unit!r}")

    key_is_ctx = is_ctx.unsqueeze(1)                 # (B, 1, N) -> key j
    query_is_tgt = (~is_ctx).unsqueeze(2)            # (B, N, 1) -> query i
    not_after = order.unsqueeze(1) <= order.unsqueeze(2)   # slot[j] <= slot[i]
    allowed = key_is_ctx | (query_is_tgt & not_after)
    return allowed.unsqueeze(1)                      # (B, 1, N, N), head 축은 broadcast


class PrefixCausalPredictor(VisionTransformerPredictor):
    """`VisionTransformerPredictor` + block-causal / future-only attention mask.

    forward 서명은 부모와 같다 — 채점기(`analysis/intphys2/surprise.py`,
    `evals/analysis_vlm/occlusion_identity/forward.py`)가 그대로 부른다.
    """

    def __init__(self, *args, causal_unit: str = "block", **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_unit = causal_unit
        self.attn_regime = f"prefix_causal[{causal_unit}]"

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        """부모 forward 와 같되 블록마다 attn_mask 를 먹인다.

        부모 코드를 베껴 온 이유: 마스크는 **정렬이 끝난 뒤**의 자리 배치를 알아야
        만들 수 있는데 부모는 그 중간 상태를 밖으로 내주지 않는다. `src/` 를 고치지
        않기로 했으므로(§eval 전용 분기) 여기서 한 번만 복제한다.
        """
        assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
        assert not has_cls, "PrefixCausalPredictor: has_cls 는 지원하지 않는다 (predictor 에 cls 토큰이 없다)"
        assert self.chop_last_n_tokens == 0, "PrefixCausalPredictor: chop_last_n_tokens 는 지원하지 않는다"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)
        _, N_ctxt, D = x.shape

        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens += pos_embs

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        masks_x_c = torch.cat(masks_x, dim=0)
        masks_y_c = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x_c, masks_y_c], dim=1)

        # ── 여기부터가 부모와 다르다 ──────────────────────────────────────────
        # 정렬 **전에** 문맥/미래 표시를 만들어 같은 argsort 를 태운다. 그래야
        # "문맥은 항상 시퀀스 앞" 같은 가정 없이 어떤 mask 조합에서도 맞는다.
        is_ctx = torch.zeros(masks.shape, dtype=torch.bool, device=masks.device)
        is_ctx[:, : masks_x_c.size(1)] = True

        argsort = torch.argsort(masks, dim=1)  # [B, N]
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        is_ctx = torch.stack([is_ctx[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        spatial = self.grid_height * self.grid_width
        # 채점은 배치 전체가 같은 (문맥, 미래) 배치를 쓴다 -> (1,1,N,N) 으로 줄여
        # (B,1,N,N) bool 텐서를 안 만든다 (N=4096, B=16 이면 268MB).
        uniform = bool((masks == masks[:1]).all()) and bool((is_ctx == is_ctx[:1]).all())
        attn_mask = prefix_causal_attn_mask(
            masks[:1] if uniform else masks,
            is_ctx[:1] if uniform else is_ctx,
            spatial_tokens=spatial,
            causal_unit=self.causal_unit,
        )
        # ─────────────────────────────────────────────────────────────────────

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, attn_mask, use_reentrant=False)
            else:
                x = blk(x, mask=masks, attn_mask=attn_mask)
        x = self.predictor_norm(x)

        if not self.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        return self.predictor_proj(x)
