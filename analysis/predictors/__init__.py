"""predictor 구조 변종 레지스트리 (채점 전용).

| kind | 구현 | 채점 경로 |
|---|---|---|
| `default` / `oneshot` | `src/models/predictor.py` (릴리즈) — full self-attention | mask token |
| `prefix` · `causal` · `prefix_full` | `src/models/rollout_predictor.py` (Ariel 학습 코드) — attention 마스크만 다르다 | mask token |
| `ar` | 〃 — 블록별 LN(target_encoder) 입력, 블록 인과, rollout 은 **render (1280) 되먹임** | `ar_scoring.py` |
| `ar_state` | 〃 — `ar` 와 같고 rollout 만 **상태 (predictor_norm, 384) 되먹임** (+ `type_embed`) | `ar_scoring.py` |
| `ctx_ar` | 〃 — 문맥 = online encoder 창 z, 미래만 블록별 되먹임 (+ `type_embed`) | `ar_scoring.py` (ctx 경로) |
| `mret` | `src/models/mret_predictor.py` | mask token |
| `prefix_xcheck` | `analysis/predictors/prefix_causal.py` — `prefix` 독립 복원, 대조 전용 | mask token |

kind 는 체크포인트 `arch.kind` 가 선언하고, config 와 다르면 `analysis/intphys2/model.py` 가 죽는다.
`KIND_ONLY_KEYS` 는 config `model.predictor.<키>` 가 이기고, 없으면 체크포인트 `arch` 값을 쓴다.
"""
from __future__ import annotations

from functools import partial

import torch.nn as nn

import src.models.predictor as _release
from src.models.rollout_predictor import vit_prefix_predictor as _ariel

ROLLOUT_KINDS = ("prefix", "causal", "prefix_full", "ar", "ar_state", "ctx_ar")
AR_KINDS = ("ar", "ar_state")          # 블록별 LN(h) 문맥 + rollout (ar_scoring.rollout_from_blocks)
CTX_AR_KINDS = ("ctx_ar",)             # 문맥 z + rollout (ar_scoring.ctx_ar_from_blocks)

KIND_ONLY_KEYS = ("prefix_impl", "n_registers", "kv_rollout",
                  "mret_hidden", "mret_beta_init", "mret_sigma", "mret_learn_sigma", "mret_per_layer", "mret_vmax")


def _default(**kw) -> nn.Module:
    return _release.vit_predictor(**kw)


def _mret(**kw) -> nn.Module:
    from src.models.mret_predictor import vit_mret_predictor
    kw.pop("prefix_impl", None)
    kw.pop("kv_rollout", None)
    return vit_mret_predictor(**kw)


def _prefix_xcheck(**kw) -> nn.Module:
    from analysis.predictors.prefix_causal import PrefixCausalPredictor
    for k in ("prefix_impl", "n_registers", "kv_rollout"):
        kw.pop(k, None)
    return PrefixCausalPredictor(mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kw)


PREDICTOR_KINDS = {
    "default": _default,
    "oneshot": _default,
    "mret": _mret,
    "prefix_xcheck": _prefix_xcheck,
    **{k: partial(_ariel, mask_mode=k) for k in ROLLOUT_KINDS},
}


def build(kind: str = "default", **kwargs) -> nn.Module:
    if kind not in PREDICTOR_KINDS:
        raise ValueError(f"unknown predictor kind {kind!r}; valid: {sorted(PREDICTOR_KINDS)}")
    if PREDICTOR_KINDS[kind] is _default:
        for k in KIND_ONLY_KEYS:
            kwargs.pop(k, None)
    return PREDICTOR_KINDS[kind](**kwargs)


def kind_of(predictor) -> str:
    return getattr(getattr(predictor, "module", predictor), "mask_mode", None) or "default"


def is_ar(predictor) -> bool:
    """mask token 규약 대신 블록별 LN(target_encoder) 문맥에서 rollout 하는 predictor (ar · ar_state)."""
    return kind_of(predictor) in AR_KINDS


def is_ctx_ar(predictor) -> bool:
    return kind_of(predictor) in CTX_AR_KINDS


def is_rollout(predictor) -> bool:
    """ar_scoring 경로를 타는가 (AR_KINDS ∪ CTX_AR_KINDS)."""
    return is_ar(predictor) or is_ctx_ar(predictor)
