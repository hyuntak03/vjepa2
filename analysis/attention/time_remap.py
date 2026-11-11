"""추론 시점 시간 위치 바꾸기 (2026-11-06, 사용자: "추론할 때 개입해서 real bench 나 다른 bench 점수 오르는지 봐").

블록 AR predictor (ACRoPEAttention) 의 시간 RoPE 위치 = 블록 번호를, 클립마다 정한 '시계' 표 (B, 블록 수) 로 바꾼다.
행 · 열 위치는 그대로. predictor 호출 동안만 separate_positions 를 감싼다 (가중치 · encoder · 정답 그대로).
  scale:δ   시계 = 블록 번호 × δ (시간 간격 축소, 라벨 없음)
  novelty   관측 기반 시계 (라벨 없음): 문맥 블록 t 의 진행 = min(1, 변화_t / 클립 중앙값),
            변화_t = 평균 |LN z_t − LN z_{t−1}| (문맥 encoder 블록 토큰). 미래 블록은 블록당 1 씩.
  oracle    시계 멈춤 (정답 라벨 = index 의 samples 열, 판별 상한): 가려진 샘플 (H) 이 든 문맥 블록은 시계를 안 올린다.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from src.models.utils.modules import ACRoPEAttention

_ORIG = ACRoPEAttention.separate_positions
_TABLE = {"t": None}


def _sep(self, ids, H_patches, W_patches):
    d, h, w = _ORIG(self, ids, H_patches, W_patches)
    T = _TABLE["t"]
    if T is None:
        return d, h, w
    if d.dim() != 3 or d.size(0) != T.size(0):
        raise RuntimeError(f"time_remap: 위치 모양 {tuple(d.shape)} 과 시계 표 {tuple(T.shape)} 이 안 맞는다")
    idx = d.long().clamp_(0, T.size(1) - 1)
    d2 = torch.gather(T.to(d.dtype)[:, None, :].expand(-1, d.size(1), -1), 2, idx)
    return d2, h, w


def clock_table(mode: str, n: int, n_blocks: int, n_ctx: int, z=None, spatial: int = 256, samples=None, device=None):
    """(n, n_blocks) 시계 표."""
    ar = torch.arange(n_blocks, device=device, dtype=torch.float32)
    if mode.startswith("scale:"):
        return (ar * float(mode.split(":", 1)[1]))[None].expand(n, -1).contiguous()
    if mode == "novelty":
        zz = F.layer_norm(z.float(), (z.size(-1),)).view(n, n_ctx, spatial, -1)
        ch = (zz[:, 1:] - zz[:, :-1]).abs().mean((2, 3))                     # (n, n_ctx-1)
        adv = (ch / ch.median(1, keepdim=True).values.clamp_min(1e-6)).clamp(max=1.0)
        c = torch.cat([torch.zeros(n, 1, device=z.device), adv.cumsum(1)], 1)  # 문맥 시계
        fut = c[:, -1:] + torch.arange(1, n_blocks - n_ctx + 1, device=z.device)[None]
        return torch.cat([c, fut], 1)
    if mode == "oracle":
        rows = []
        for s in samples:
            ctx = s.split("|")[0] if s else ""
            c, out = 0.0, []
            for b in range(n_ctx):
                hid = "H" in ctx[2 * b:2 * b + 2]
                if b > 0 and not hid:
                    c += 1.0
                out.append(c)
            out += [c + j for j in range(1, n_blocks - n_ctx + 1)]
            rows.append(out)
        return torch.tensor(rows, device=device, dtype=torch.float32)
    raise ValueError(f"time_remap 모드 {mode!r}: scale:<δ> | novelty | oracle")


@contextlib.contextmanager
def time_remap(table):
    if table is None:
        yield
        return
    _TABLE["t"] = table
    ACRoPEAttention.separate_positions = _sep
    try:
        yield
    finally:
        ACRoPEAttention.separate_positions = _ORIG
        _TABLE["t"] = None
