"""predictor 구조 변종 레지스트리 (채점 전용).

릴리즈 V-JEPA 2 predictor 는 **문맥·미래 토큰이 전부 서로를 본다** (full self-attention).
다르게 학습된 predictor 를 채점하려면 학습 때의 attention 규칙을 그대로 재현해야 한다.

| kind | 구현 | 무엇 |
|---|---|---|
| `default` / `oneshot` | `src/models/predictor.py` (릴리즈, 무수정) | full self-attention |
| **`prefix`** | **`src/models/rollout_predictor.py` (Ariel 학습 코드 그대로)** | 문맥 양방향 · 문맥→미래 차단 · 미래→미래 block-causal |
| `causal` | 〃 (`mask_mode="causal"`) | 문맥끼리도 인과 |
| `prefix_full` | 〃 (`mask_mode="prefix_full"`, 2026-09-27 이식) | prefix 와 같되 미래 블록끼리 인과 없이 서로 다 본다 (대조 팔) |
| **`ar`** | 〃 (`mask_mode="ar"`, 2026-09-27 이식, Ariel 팔 C) | **자기회귀**: mask token 없이 LN(블록별 target encoder) 토큰을 넣고 블록 인과로 다음 블록을 낸다. 채점은 `analysis/predictors/ar_scoring.py` (release 규약 forward 는 막혀 있다) |
| `prefix_xcheck` | `analysis/predictors/prefix_causal.py` (독립 복원) | `prefix` 와 **대조용**. 보고에는 쓰지 않는다 |

⚠️ **`prefix` 는 Ariel 의 학습 코드를 그대로 쓴다** (2026-09-23 에 `z_ariel/vjepa2_train_code_20260923`
에서 `src/models/rollout_predictor.py` 와 `src/models/utils/modules.py` 의 `PrefixSpec` 분기를
가져왔다). 우리가 먼저 만든 독립 복원(`prefix_xcheck`)과 **상대 3.4e-07** 로 일치해서
복원이 맞았음이 확인됐지만, 정본은 학습 코드 쪽으로 둔다.

쓰는 곳은 `analysis/intphys2/model.py::_build_predictor` 한 군데다.
체크포인트가 `arch.kind` 를 들고 있으면 그것이 기본값이 되고, config 가 다른 kind 를
말하면 **죽는다** (조용히 틀린 mask 로 채점하는 것이 제일 나쁜 실패라서).
"""
from __future__ import annotations

from functools import partial

import torch.nn as nn

import src.models.predictor as _release
from src.models.rollout_predictor import vit_prefix_predictor as _ariel


def _default(**kw) -> nn.Module:
    return _release.vit_predictor(**kw)


def _prefix(**kw) -> nn.Module:
    return _ariel(mask_mode="prefix", **kw)


def _causal(**kw) -> nn.Module:
    return _ariel(mask_mode="causal", **kw)


def _prefix_full(**kw) -> nn.Module:
    return _ariel(mask_mode="prefix_full", **kw)


def _ctx_ar(**kw) -> nn.Module:
    """2026-09-29 Ariel 팔 C' (ctx_ar): 문맥은 창 단위 z (online encoder, prefix 팔과 같다) + 미래는 블록별 LN(h) 되먹임, type embedding 2 개.
    state_dict 에 `type_embed` 가 더 있다 (다른 kind 와 strict 호환 안 됨). 채점 · 판독은 forward_mixed / rollout_ctx."""
    return _ariel(mask_mode="ctx_ar", **kw)


def _ar(**kw) -> nn.Module:
    """Ariel 팔 C (2026-09-26 학습 코드, 2026-09-27 이식). forward(x, masks_x, masks_y) 는 RuntimeError —
    forward_seq / rollout 만 쓴다. 채점 경로는 ar_scoring.py."""
    return _ariel(mask_mode="ar", **kw)


def _mret(**kw) -> nn.Module:
    """DC1(b) 움직임 정렬 조회 bias (src/models/mret_predictor.py). state_dict = prefix + mret_* (새 키)."""
    from src.models.mret_predictor import vit_mret_predictor
    kw.pop("prefix_impl", None)
    return vit_mret_predictor(**kw)


def _prefix_xcheck(**kw) -> nn.Module:
    from analysis.predictors.prefix_causal import PrefixCausalPredictor
    kw.pop("prefix_impl", None)
    kw.pop("n_registers", None)
    return PrefixCausalPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kw)


# kind -> factory. 값은 모두 릴리즈 predictor 와 **state_dict 키가 같다**
# (변종은 attention mask 만 바꾼다). 그래서 어느 kind 로 지어도 같은 파일을 읽는다.
PREDICTOR_KINDS = {
    "default": _default,
    "oneshot": _default,          # Ariel 학습 config 의 이름
    "prefix": _prefix,
    "causal": _causal,
    "prefix_xcheck": _prefix_xcheck,
    "mret": _mret,                # 2026-09-25 DC1(b)
    "prefix_full": _prefix_full,  # 2026-09-27 Ariel 0927 이식 (미래 블록끼리 양방향)
    "ar": _ar,
    "ctx_ar": _ctx_ar,            # 2026-09-29 Ariel 팔 C' (문맥 z + 미래 되먹임)                    # 2026-09-27 Ariel 팔 C (자기회귀). 채점은 ar_scoring.py
}

# 이 kind 들만 아래 키를 받는다 (릴리즈 predictor 는 모른다)
KIND_ONLY_KEYS = ("prefix_impl", "n_registers", "mret_hidden", "mret_beta_init", "mret_sigma", "mret_learn_sigma", "mret_per_layer", "mret_vmax")


def build(kind: str = "default", **kwargs) -> nn.Module:
    if kind not in PREDICTOR_KINDS:
        raise ValueError(f"unknown predictor kind {kind!r}; valid: {sorted(PREDICTOR_KINDS)}")
    if PREDICTOR_KINDS[kind] is _default:
        for k in KIND_ONLY_KEYS:
            kwargs.pop(k, None)
    return PREDICTOR_KINDS[kind](**kwargs)


def is_ar(predictor) -> bool:
    """자기회귀 (kind=ar) predictor 인가. 이 predictor 는 mask token 규약 forward 를 못 쓰고,
    문맥 · 타깃이 **블록별** LN(target_encoder) 공간이다 — 채점은 analysis/predictors/ar_scoring.py."""
    m = getattr(predictor, "module", predictor)
    return getattr(m, "mask_mode", None) == "ar"


def is_ctx_ar(predictor) -> bool:
    """kind=ctx_ar 인가 (문맥은 z, 미래만 되먹임). ar_scoring 의 블록별 문맥 경로는 못 쓴다."""
    m = getattr(predictor, "module", predictor)
    return getattr(m, "mask_mode", None) == "ctx_ar"
