"""predictor 토큰 격자 ↔ (tubelet, height, width) 매핑과 거리 행렬.

predictor 입력 시퀀스는 **정렬된 토큰 인덱스 그대로**다. predictor.forward 가
masks_x(문맥) 와 masks_y(미래) 를 이어 붙인 뒤 argsort 로 정렬하는데
(src/models/predictor.py:218-225), 우리 프로토콜에서는
    masks_x = [0 .. C*S)      masks_y = [C*S .. T*S)
라 argsort 가 항등이다. 그래서 시퀀스 위치 = 토큰 인덱스이고

    idx -> t = idx // S,  h = (idx % S) // G,  w = idx % G        (S = G*G)

RoPE 도 정확히 같은 분해를 쓴다 (modules.py RoPEAttention.separate_positions,
H_patches/W_patches 가 None 이면 grid_size 로 나눈다). 그래서 이 파일의 좌표계는
모델이 실제로 쓰는 좌표계와 같다.

시간 단위 주의 — 세 가지가 다르다:
    tubelet   1  =  샘플 프레임 2장  =  원본(raw) 프레임 6장 (frames_stride=3)
표에 쓰는 값은 전부 **샘플 프레임**이다 (프로토콜이 읽는 32장 기준).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    """한 predictor forward 의 토큰 격자."""

    n_frames: int          # 클립 길이 (샘플 프레임)  = 32
    ctx_frames: int        # 문맥 길이 (샘플 프레임)  = 16
    tubelet: int           # 2
    patch: int             # 16
    img: int               # 256
    stride: int = 1        # 원본 프레임 stride (표시용)

    @property
    def G(self) -> int:            # 한 변의 패치 수
        return self.img // self.patch

    @property
    def S(self) -> int:            # 튜블릿당 공간 토큰 수
        return self.G * self.G

    @property
    def T(self) -> int:            # 전체 튜블릿 수
        return self.n_frames // self.tubelet

    @property
    def C(self) -> int:            # 문맥 튜블릿 수 (미래는 C..T-1)
        return self.ctx_frames // self.tubelet

    @property
    def N(self) -> int:            # predictor 시퀀스 길이
        return self.T * self.S

    def frame_span(self, t: int) -> tuple[int, int]:
        """튜블릿 t 가 덮는 샘플 프레임 (1-idx, 양끝 포함) — 사람이 읽는 표기."""
        return t * self.tubelet + 1, (t + 1) * self.tubelet

    def tick_labels(self) -> list[str]:
        """플롯 축 라벨. 튜블릿 -> 'f1-2' 처럼."""
        return [f"{a}-{b}" for a, b in (self.frame_span(t) for t in range(self.T))]

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        d, m = cfg["data"], cfg["model"]
        feat = cfg.get("features") or cfg.get("surprise") or {}
        ctx = int(feat.get("context_length", 32))
        g = cls(n_frames=int(d["n_frames"]), ctx_frames=ctx,
                tubelet=int(m.get("tubelet_size", 2)), patch=int(m.get("patch_size", 16)),
                img=int(m.get("img_size", 256)), stride=int(d.get("frames_stride", 1)))
        if g.ctx_frames % g.tubelet or g.n_frames % g.tubelet:
            raise ValueError(f"tubelet({g.tubelet}) 정렬 아님: n_frames={g.n_frames} ctx={g.ctx_frames}")
        if int(m.get("window_size", g.n_frames)) != g.n_frames:
            raise ValueError(
                f"model.window_size({m.get('window_size')}) != data.n_frames({g.n_frames}). "
                "predictor 의 RoPE grid_depth 가 달라져 토큰 격자가 어긋난다")
        return g

    def describe(self) -> str:
        return (f"{self.n_frames}f (ctx {self.ctx_frames} + fut {self.n_frames - self.ctx_frames}) "
                f"| tubelet {self.tubelet} -> T={self.T} (문맥 0..{self.C-1}, 미래 {self.C}..{self.T-1}) "
                f"| {self.img}/{self.patch} -> {self.G}x{self.G}=S {self.S} | N={self.N}")


def spatial_distance(grid: Grid, device) -> torch.Tensor:
    """(S, S) 유클리드 거리. **패치 단위** — 픽셀로 보려면 grid.patch 를 곱한다."""
    ii = torch.arange(grid.S, device=device)
    h, w = (ii // grid.G).float(), (ii % grid.G).float()
    return torch.sqrt((h[:, None] - h[None, :]) ** 2 + (w[:, None] - w[None, :]) ** 2)


def query_coords(grid: Grid, device) -> tuple[torch.Tensor, torch.Tensor]:
    """시퀀스 위치 -> (튜블릿 인덱스, 공간 인덱스). 둘 다 (N,) long."""
    ii = torch.arange(grid.N, device=device)
    return ii // grid.S, ii % grid.S
