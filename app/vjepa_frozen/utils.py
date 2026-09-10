"""frozen-encoder predictor 학습의 모델/옵티마이저 조립.

모델은 **채점기와 같은 함수**로 짓는다 (analysis/intphys2/model.py 의 _build_encoder /
_build_predictor / _load_state_dict). 그래야 여기서 학습한 predictor 가 surprise_c16t32 /
intphys1_sliding 에 그대로 들어간다 (predictor_checkpoint 키, 같은 파일 참고).
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from analysis.intphys2.model import (
    _build_encoder,
    _build_predictor,
    _clean_backbone_key,
    _load_checkpoint,
    _load_state_dict,
)
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16, "none": None, "": None}


def parse_dtype(name) -> Optional[torch.dtype]:
    key = str(name).lower()
    if key not in _DTYPES:
        raise ValueError(f"dtype={name!r}; {sorted(_DTYPES)} 중 하나")
    return _DTYPES[key]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def build_frozen_encoders(m: dict, state: dict, device: torch.device, dtype: torch.dtype,
                          num_frames: int) -> Tuple[nn.Module, nn.Module]:
    """(context_encoder, target_encoder). 둘 다 frozen/eval. 키가 같으면 한 모듈을 공유한다.

    기본 키는 surprise_c16t32 와 같다: context <- `encoder`(online), target <- `target_encoder`(EMA).
    """
    kw = dict(arch_name=m.get("arch_name", "vit_huge"), img_size=int(m.get("img_size", 256)),
              patch_size=int(m.get("patch_size", 16)), tubelet_size=int(m.get("tubelet_size", 2)),
              num_frames=int(num_frames), use_rope=bool(m.get("use_rope", True)),
              uniform_power=bool(m.get("uniform_power", False)), use_sdpa=bool(m.get("use_sdpa", True)))
    ctx_key = m.get("context_encoder_key", "encoder")
    tgt_key = m.get("target_encoder_key", "target_encoder")
    for k in (ctx_key, tgt_key):
        if k not in state:
            raise KeyError(f"체크포인트에 {k!r} 가 없다. 있는 키: {list(state)}")

    def _one(key, tag):
        enc = _build_encoder(**kw)
        _load_state_dict(enc, state[key], tag=tag)
        _assert_fully_loaded(enc, state[key], tag)
        enc = enc.to(device=device, dtype=dtype).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc

    tgt = _one(tgt_key, f"target_encoder<-{tgt_key}")
    ctx = tgt if ctx_key == tgt_key else _one(ctx_key, f"context_encoder<-{ctx_key}")
    logger.info(f"encoders frozen: context<-{ctx_key} target<-{tgt_key} "
                f"({'shared' if ctx is tgt else 'dual'}), dtype={dtype}, params={count_parameters(tgt, False)/1e6:.1f}M")
    return ctx, tgt


def _assert_fully_loaded(module: nn.Module, sub_state: dict, tag: str):
    """_load_state_dict 는 strict=False 라 빠진 키를 조용히 init 으로 둔다. 여기서는 그걸 허용하지 않는다."""
    have = set(_clean_backbone_key(sub_state))
    need = set(module.state_dict())
    missing = sorted(need - have)
    if missing:
        raise RuntimeError(f"[{tag}] 체크포인트에 없는 파라미터 {len(missing)}개: {missing[:5]} ...")


def build_predictor(m: dict, encoder_embed_dim: int, state: Optional[dict], device: torch.device,
                    num_frames: int) -> nn.Module:
    """predictor (fp32 마스터 가중치). state 가 있으면 릴리즈 predictor 를 통째로 로드한다 (post-FT)."""
    pc = dict(m.get("predictor") or {})
    extra = {"use_activation_checkpointing": bool(pc.get("use_activation_checkpointing", False)),
             "zero_init_mask_tokens": bool(pc.get("zero_init_mask_tokens", True))}
    pred = _build_predictor(
        img_size=int(m.get("img_size", 256)), patch_size=int(m.get("patch_size", 16)),
        tubelet_size=int(m.get("tubelet_size", 2)), num_frames=int(num_frames),
        encoder_embed_dim=int(encoder_embed_dim),
        predictor_embed_dim=int(pc.get("embed_dim", 384)), predictor_depth=int(pc.get("depth", 12)),
        predictor_num_heads=int(pc.get("num_heads", 12)), num_mask_tokens=int(pc.get("num_mask_tokens", 10)),
        use_rope=bool(m.get("use_rope", True)), uniform_power=bool(m.get("uniform_power", False)),
        use_sdpa=bool(m.get("use_sdpa", True)), extra=extra)
    if state is not None:
        sd = _clean_backbone_key(state["predictor"])
        msg = pred.load_state_dict(sd, strict=True)            # post-FT 는 전부 맞아야 한다
        logger.info(f"predictor <- checkpoint['predictor'] (strict): {msg}")
    else:
        logger.info("predictor: 새로 초기화 (scratch)")
    pred = pred.to(device)
    logger.info(f"predictor params: {count_parameters(pred)/1e6:.2f}M "
                f"(embed {pc.get('embed_dim', 384)} / depth {pc.get('depth', 12)} / heads {pc.get('num_heads', 12)} "
                f"/ mask tokens {pc.get('num_mask_tokens', 10)})")
    return pred


def load_base_checkpoint(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model.checkpoint 가 없다: {path}")
    return _load_checkpoint(path)


def init_opt(predictor: nn.Module, iterations_per_epoch: int, start_lr: float, ref_lr: float,
             final_lr: float, warmup_epochs: float, num_epochs: int, wd: float, final_wd: float,
             ipe_scale: float = 1.0, betas=(0.9, 0.999), eps: float = 1e-8, use_scaler: bool = False):
    """app/vjepa/utils.init_opt 의 predictor-only 판. bias / 1-D 파라미터는 weight decay 제외."""
    decay = [p for n, p in predictor.named_parameters() if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)]
    no_decay = [p for n, p in predictor.named_parameters() if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))]
    param_groups = [{"params": decay},
                    {"params": no_decay, "WD_exclude": True, "weight_decay": 0}]
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(betas), eps=eps)
    T_max = int(ipe_scale * num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(warmup_epochs * iterations_per_epoch),
                                     start_lr=start_lr, ref_lr=ref_lr, final_lr=final_lr, T_max=T_max)
    wd_scheduler = CosineWDSchedule(optimizer, ref_wd=wd, final_wd=final_wd, T_max=T_max)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    return optimizer, scaler, scheduler, wd_scheduler


# ----------------------------------------------------------------------------- checkpoint
def save_checkpoint(path: str, predictor: nn.Module, optimizer, scaler, epoch: int, step: int,
                    loss: float, cfg: dict, arch: dict):
    """predictor 만 담는다 (encoder 는 base checkpoint 를 가리킨다).

    키 `predictor` 는 채점기(analysis/intphys2/model.py: model.predictor_checkpoint) 와
    z_training/harness/export_ckpt.py 가 읽는다. DDP 로 감쌌으면 .module 을 풀어 저장한다.
    """
    mod = predictor.module if hasattr(predictor, "module") else predictor
    save = {"predictor": mod.state_dict(), "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch, "step": step, "loss": loss,
            "base_checkpoint": cfg["model"]["checkpoint"],
            "context_encoder_key": cfg["model"].get("context_encoder_key", "encoder"),
            "target_encoder_key": cfg["model"].get("target_encoder_key", "target_encoder"),
            "arch": arch, "config": cfg, "format": "vjepa_frozen/predictor-only/v1"}
    tmp = path + ".tmp"
    torch.save(save, tmp)
    os.replace(tmp, path)                                     # 저장 중 죽어도 latest.pt 가 반쪽이 안 된다


def load_resume(path: str, predictor: nn.Module, optimizer, scaler):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mod = predictor.module if hasattr(predictor, "module") else predictor
    msg = mod.load_state_dict(ck["predictor"], strict=True)
    optimizer.load_state_dict(ck["opt"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    logger.info(f"resume <- {path}: epoch {ck['epoch']} step {ck.get('step')} loss {ck.get('loss')} ({msg})")
    return int(ck["epoch"]), int(ck.get("step", 0))
