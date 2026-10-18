"""encoder LoRA (2026-10-18, 사용자 지시: "encoder 엔 LoRA 달고 encoder 까지 열어서 release predictor 학습").

ViT 블록의 nn.Linear (기본 attn.qkv · attn.proj · mlp.fc1 · mlp.fc2) 를 LoRALinear 로 감싼다:
  y = W x + b + (alpha / r) · B (A x),   A ~ kaiming, B = 0  → 붙인 직후 출력은 원래 encoder 와 비트 단위로 같다.
원 가중치 W · b 는 얼린 채 (requires_grad False) 두고 A · B 만 학습한다. A · B 는 fp32 로 둔다 (encoder 본체는 bf16 일 수 있다).
학습 (app/vjepa_frozen/train.py, model.encoder_lora) 은 **context encoder (online `encoder`) 에만** 붙인다 — target encoder (EMA) 는 얼린 채라 타깃이 움직이지 않는다.
체크포인트에는 `encoder_lora` (A · B state_dict) 와 `encoder_lora_cfg` 가 같이 저장되고, 채점기 (analysis/intphys2/model.py) 가 그것을 보고 같은 자리에 다시 붙인다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = int(rank), float(alpha) / float(rank)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):          # 다른 코드가 .weight 를 읽는 경우 (init rescale 등) 원 가중치를 준다
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        d = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return y + (self.scale * d).to(y.dtype)


def apply_lora(encoder: nn.Module, rank: int = 8, alpha: float = 16.0, targets=DEFAULT_TARGETS) -> list:
    """encoder.blocks[*] 안의 이름이 targets 인 nn.Linear 를 LoRALinear 로 바꾼다. 학습할 파라미터 목록을 돌려준다."""
    targets = tuple(targets)
    n = 0
    for blk in encoder.blocks:
        for parent in [m for m in blk.modules()]:
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, rank, alpha)); n += 1
    params = [p for n_, p in encoder.named_parameters() if "lora_" in n_]
    for p in params:
        p.requires_grad_(True)
    if n == 0:
        raise RuntimeError(f"LoRA 를 붙일 Linear 가 없다 (targets={targets})")
    encoder.lora_info = dict(rank=int(rank), alpha=float(alpha), targets=list(targets), n_layers=n,
                             n_params=int(sum(p.numel() for p in params)))
    return params


def lora_state_dict(encoder: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in encoder.state_dict().items() if "lora_" in k}


def load_lora(encoder: nn.Module, sd: dict):
    own = {k for k in encoder.state_dict() if "lora_" in k}
    if set(sd) != own:
        raise RuntimeError(f"LoRA 키 불일치: 체크포인트 {len(sd)} vs 모델 {len(own)} (예: {sorted(set(sd) ^ own)[:3]})")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
