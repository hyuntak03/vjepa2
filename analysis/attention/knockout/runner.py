"""predictor 를 prefix 캐시로 이어 돌린다 — knockout 스윕을 싸게 만든다.

왜 KV cache 가 아닌가
---------------------
predictor 는 causal decoder 가 아니라 4096 토큰을 한 번에 도는 **bidirectional**
transformer다. 층 ℓ 에서 간선을 끊으면 ℓ 의 출력이 바뀌고, 따라서 **ℓ 위의 모든 층의
k·v 가 전부 바뀐다.** 재사용할 수 있는 건 ℓ 아래뿐인데, 그 층들은 k·v 를 캐싱할 게 아니라
**아예 다시 돌 필요가 없다.** 그래서 여기서 캐싱하는 것은 KV 가 아니라 **층별 residual
stream** `x_ℓ` 이고, 이것이 KV 캐시를 엄격히 지배한다.

  clean forward 한 번 -> x_0 .. x_{L-1} 저장
  명세 s 는 min(s.layers) 부터 이어 돈다  ->  L 층 스윕이 L x L 대신 L(L+1)/2 층 계산

그런데 **진짜 큰 절약은 이 위에 있다** — 인코더다. 문맥 인코더 z 와 target h 는
knockout 과 무관하므로 **배치당 한 번만** 계산해 모든 명세가 공유한다.
실측 FLOP 비: 인코더 2 개 ≈ 6.7 TFLOP vs predictor 1 회 ≈ 0.27 TFLOP (25배).
즉 명세를 25개까지 늘려도 전체 비용은 채점 한 번의 2배 안쪽이다.

정확성
------
`resume(x_0, l0=0, attn_masks=None)` 은 실제 `predictor.forward` 와 **비트 단위로 같아야
한다.** run.py 가 첫 배치에서 그걸 대조해 결과 json 의 verify 에 남긴다.
"""
from __future__ import annotations

import contextlib

import torch


def assert_layout(predictor, grid):
    """이 러너가 기대하는 predictor 구성. 하나라도 다르면 tail 재현이 틀어진다."""
    if predictor.return_all_tokens:
        raise ValueError("predictor.return_all_tokens=True 는 지원하지 않는다")
    if getattr(predictor, "chop_last_n_tokens", 0):
        raise ValueError("predictor.chop_last_n_tokens != 0 은 지원하지 않는다")
    if predictor.num_patches != grid.N:
        raise ValueError(f"predictor.num_patches={predictor.num_patches} != 격자 N={grid.N}")


class _Capture:
    """블록마다 입력 residual stream 과 RoPE 위치 mask 를 받아 둔다."""

    def __init__(self, predictor):
        self.blocks = list(predictor.predictor_blocks)
        self.x: list[torch.Tensor] = [None] * len(self.blocks)
        self.masks = None
        self._h = []

    def _pre(self, li):
        def hook(mod, args, kwargs):
            self.x[li] = args[0].detach()
            if li == 0:
                self.masks = kwargs.get("mask")
            return None
        return hook

    def __enter__(self):
        for li, b in enumerate(self.blocks):
            self._h.append(b.register_forward_pre_hook(self._pre(li), with_kwargs=True))
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()
        self._h.clear()
        return False


@torch.no_grad()
def clean_forward(predictor, z, ctx_idx, tgt_idx, mask_index, keep_prefix=True):
    """실제 predictor.forward 를 그대로 돌리면서 층별 입력을 받아 둔다.

    Returns (p, prefix, masks, n_ctx)
      prefix[ℓ] = 블록 ℓ 의 **입력**. keep_prefix=False 면 prefix[0] 만 채운다.
    """
    with _Capture(predictor) as cap:
        p = predictor(z, ctx_idx, tgt_idx, mask_index=mask_index)
    masks = cap.masks
    n_ctx = int(ctx_idx.size(1))
    # 토큰 정렬이 항등인지 확인한다. 우리 프로토콜은 ctx=[0,C·S) / tgt=[C·S,N) 라
    # predictor 의 argsort 가 항등이고, 그래서 tail 이 x[:, n_ctx:] 한 줄로 끝난다.
    if masks is not None:
        ar = torch.arange(masks.size(1), device=masks.device)
        if not torch.equal(masks[0], ar):
            raise ValueError("predictor 내부 토큰 순서가 항등이 아니다 — resume tail 을 못 쓴다")
    prefix = cap.x if keep_prefix else [cap.x[0]] + [None] * (len(cap.x) - 1)
    return p, prefix, masks, n_ctx


@torch.no_grad()
def resume(predictor, x, l0, masks, n_ctx, attn_masks=None):
    """블록 l0 부터 끝까지 돌리고 predictor 의 tail 을 재현한다.

    attn_masks[ℓ] 가 (N,N) bool 이면 그 층에서 **True 인 자리만 attend** 한다
    (SDPA 규약). None 이면 그 층은 온전한 attention 이다.
    """
    blocks = predictor.predictor_blocks
    for li in range(l0, len(blocks)):
        am = None if attn_masks is None else attn_masks[li]
        x = blocks[li](x, mask=masks, attn_mask=am)
    x = predictor.predictor_norm(x)
    x = x[:, n_ctx:]                       # return_all_tokens=False, 역정렬은 항등
    return predictor.predictor_proj(x)


def autocast_ctx(cfg_model):
    """eval.run_surprise 와 같은 규약으로 autocast 를 만든다 (수치 경로를 맞춘다)."""
    types = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "none": None, "": None}
    name = str(cfg_model.get("autocast", "none")).lower()
    if name not in types:
        raise ValueError(f"model.autocast={name!r}; {sorted(types)} 중 하나여야 한다")
    d = types[name]
    return (torch.autocast("cuda", dtype=d) if d else contextlib.nullcontext()), name
