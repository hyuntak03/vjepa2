"""predictor 의 attention 을 층·헤드별로 집계한다. 모델 forward 는 건드리지 않는다.

왜 재계산하나 — flash 는 원리적으로 못 빼낸다
---------------------------------------------
predictor 의 각 Block 은 `F.scaled_dot_product_attention` 을 쓴다
(modules.py RoPEAttention.forward). SDPA/flash 는 softmax 를 타일 안에서 융합해
**attention 행렬을 메모리에 만들지 않는다** — 그게 속도·메모리 이득의 원천이고,
그래서 hook 으로 꺼낼 attention weight 자체가 존재하지 않는다.
(`sdpa_kernel(MATH)` 로 backend 를 바꿔도 내부 텐서를 노출하지 않는다.)

그래서 여기서는 **실제 forward 를 flash 그대로 두고**(모델 출력 불변),
forward_pre_hook 으로 층 입력 x 와 RoPE 위치 mask 만 받아 q·k 를 다시 만들고
query chunk 단위로 attention 을 재계산해 그 자리에서 집계로 접는다.
전체 행렬은 어디에도 저장하지 않는다 (한 청크 = B x H x q_chunk x N).

  비용: predictor 는 384-dim / 12층 / 12헤드라 재계산이 싸다. 비싼 ViT-H 인코더는
        건드리지 않는다. qkv 투영이 한 번 중복되지만 attention 행렬 비용에 묻힌다.

RoPE 는 q·k 에 **적용된 뒤** 잡아야 한다. 아래 `_rope` 는 RoPEAttention.forward 의
회전 구간을 그대로 옮긴 것이고, 입력(mask 텐서)도 모듈이 받은 바로 그 텐서다.

검증
----
첫 배치 첫 층의 첫 청크에서 우리가 만든 attn@v 를 proj 까지 통과시켜 **모듈의 실제
출력**과 대조한다. `max|Δ|` 와 출력 스케일을 summary.json.verify 에 남긴다.
(fp16 autocast 에서는 SDPA 내부 누산 순서가 달라 정확히 0 이 되지는 않는다.)

집계 (전부 층 x 헤드 x 질의 튜블릿 단위, 클립·공간 위치에 대해 평균)
    mass [Tq, Tk]  질의 튜블릿이 각 키 튜블릿에 준 attention 질량. 행 합 = 1
    same [Tq, Tk]  그 중 **같은 공간 패치**에만 준 질량 (균등 기대값 = mass/S)
    sdist[Tq]      공간 attention distance (패치 단위. 픽셀은 x patch_size)
    tdist[Tq]      시간 attention distance |Δt| (튜블릿 단위)
    toff [Tq]      부호 있는 Δt — 음수면 과거(문맥) 쪽
    ent  [Tq]      attention 엔트로피 (nats). 균등 상한 = log(N)
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.utils.modules import RoPEAttention, rotate_queries_or_keys

from .geometry import Grid, query_coords, spatial_distance


def _rope(m: RoPEAttention, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor):
    """RoPEAttention.forward 의 회전 구간을 그대로 재현한다 (modules.py:337-368).

    mask 는 모듈이 실제로 받은 토큰 인덱스 텐서 (B, N) 를 그대로 넘긴다 —
    dtype·모양을 우리가 만들지 않으므로 원본과 어긋날 여지가 없다.
    """
    mask = mask.unsqueeze(1).repeat(1, m.num_heads, 1)
    d_mask, h_mask, w_mask = m.separate_positions(mask, None, None)
    s = 0
    qd = rotate_queries_or_keys(q[..., s:s + m.d_dim], pos=d_mask)
    kd = rotate_queries_or_keys(k[..., s:s + m.d_dim], pos=d_mask)
    s += m.d_dim
    qh = rotate_queries_or_keys(q[..., s:s + m.h_dim], pos=h_mask)
    kh = rotate_queries_or_keys(k[..., s:s + m.h_dim], pos=h_mask)
    s += m.h_dim
    qw = rotate_queries_or_keys(q[..., s:s + m.w_dim], pos=w_mask)
    kw = rotate_queries_or_keys(k[..., s:s + m.w_dim], pos=w_mask)
    s += m.w_dim
    if s < m.head_dim:
        return (torch.cat([qd, qh, qw, q[..., s:]], dim=-1),
                torch.cat([kd, kh, kw, k[..., s:]], dim=-1))
    return torch.cat([qd, qh, qw], dim=-1), torch.cat([kd, kh, kw], dim=-1)


class AttnRecorder:
    """predictor 에 붙여 층·헤드별 attention 집계를 모은다.

        rec = AttnRecorder(bundle.predictor, grid, n_groups=6, device=dev)
        with rec:
            for ...:
                rec.set_group(gi)
                predictor(z, ctx_idx, tgt_idx, mask_index=0)
        out = rec.numpy()

    ⚠️ q_chunk 는 grid.S 의 배수여야 한다 (청크가 튜블릿 경계에 맞아야 집계가 단순해진다).
    """

    def __init__(self, predictor, grid: Grid, n_groups: int, device,
                 q_chunk: int = 512, verify: bool = True):
        self.blocks = list(predictor.predictor_blocks)
        self.mods = [b.attn for b in self.blocks]
        bad = [i for i, m in enumerate(self.mods) if not isinstance(m, RoPEAttention)]
        if bad:
            raise TypeError(f"predictor block {bad} 이 RoPEAttention 이 아니다 "
                            "(model.use_rope: true 여야 한다)")
        self.grid, self.device = grid, device
        self.L, self.H = len(self.mods), self.mods[0].num_heads
        if q_chunk % grid.S:
            raise ValueError(f"q_chunk({q_chunk}) 는 S({grid.S}) 의 배수여야 한다")
        self.q_chunk = q_chunk
        self.verify, self._ver_done, self._ver_buf = verify, False, None
        self.verify_stat: dict | None = None

        G, L, H, T = n_groups, self.L, self.H, grid.T
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
        self.mass, self.same = z(G, L, H, T, T), z(G, L, H, T, T)
        self.sdist, self.tdist = z(G, L, H, T), z(G, L, H, T)
        self.toff, self.ent = z(G, L, H, T), z(G, L, H, T)
        self.count = z(G, T)                       # 질의 개수 (클립 x 공간위치)
        self.n_clips = np.zeros(G, dtype=np.int64)

        self._tq, self._sq = query_coords(grid, device)
        self._D = spatial_distance(grid, device)   # (S, S) 패치 단위
        self._tarange = torch.arange(grid.T, device=device)
        self._g, self._handles = 0, []

    # ---- 붙이고 떼기 ------------------------------------------------------------
    def __enter__(self):
        for li, m in enumerate(self.mods):
            self._handles.append(
                m.register_forward_pre_hook(self._make_pre(li), with_kwargs=True))
        if self.verify:
            self._handles.append(self.mods[0].register_forward_hook(self._verify_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def set_group(self, gi: int, n_clips: int = 0):
        self._g = gi
        self.n_clips[gi] += n_clips

    # ---- hook 본체 --------------------------------------------------------------
    def _make_pre(self, li):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs["x"]
            self._accumulate(li, mod, x, kwargs.get("mask"))
            return None                            # 실제 forward 는 그대로 진행된다
        return hook

    def _verify_hook(self, mod, args, output):
        if self._ver_buf is None:
            return None
        a, b, ours = self._ver_buf
        self._ver_buf = None
        real = output[:, a:b].float()
        self.verify_stat = {
            "layer": 0, "queries": [int(a), int(b)],
            "max_abs_diff": float((ours - real).abs().max()),
            "mean_abs_out": float(real.abs().mean()),
            "max_abs_out": float(real.abs().max()),
        }
        return None

    @torch.no_grad()                 # inference_mode 가 아니다 — 누산기에 in-place 로 더한다
    def _accumulate(self, li, m: RoPEAttention, x: torch.Tensor, mask):
        B, N, C = x.shape
        if N != self.grid.N:
            raise ValueError(f"시퀀스 길이 {N} != 격자 N {self.grid.N}. "
                             "data.n_frames / model.window_size 를 확인할 것")
        g, T, S, Hh = self._g, self.grid.T, self.grid.S, self.H

        qkv = m.qkv(x).unflatten(-1, (3, Hh, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, N, D)
        if mask is None:                                       # 모듈의 기본 경로와 같게
            mask = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        q, k = _rope(m, q, k, mask)
        # softmax 는 fp32 로. autocast fp16 에서 로짓을 그대로 쓰면 꼬리가 뭉개진다.
        kT = k.float().transpose(-1, -2).contiguous()

        for a in range(0, N, self.q_chunk):
            b = min(a + self.q_chunk, N)
            Q, nt, t0 = b - a, (b - a) // S, a // S
            p = (torch.matmul(q[:, :, a:b].float(), kT) * m.scale).softmax(-1)   # (B,H,Q,N)

            ent = -(p * p.clamp_min(1e-12).log()).sum(-1)                        # (B,H,Q)
            pv = p.view(B, Hh, Q, T, S)
            pt, ps = pv.sum(-1), pv.sum(-2)                                      # (B,H,Q,T) (B,H,Q,S)
            sq, tq = self._sq[a:b], self._tq[a:b]
            same = torch.gather(pv, -1, sq.view(1, 1, Q, 1, 1).expand(B, Hh, Q, T, 1)).squeeze(-1)
            sdist = (ps * self._D[sq]).sum(-1)                                   # (B,H,Q) 패치 단위
            dt = (self._tarange.view(1, -1) - tq.view(-1, 1)).float()            # (Q,T) 튜블릿 단위
            tdist, toff = (pt * dt.abs()).sum(-1), (pt * dt).sum(-1)

            # 청크가 튜블릿 경계에 맞으므로 (Q -> nt x S) 로 접어 공간·배치를 한 번에 지운다
            r4 = lambda z: z.view(B, Hh, nt, S, T).sum((0, 3)).double()          # noqa: E731
            r3 = lambda z: z.view(B, Hh, nt, S).sum((0, 3)).double()             # noqa: E731
            sl = slice(t0, t0 + nt)
            self.mass[g, li, :, sl] += r4(pt)
            self.same[g, li, :, sl] += r4(same)
            self.sdist[g, li, :, sl] += r3(sdist)
            self.tdist[g, li, :, sl] += r3(tdist)
            self.toff[g, li, :, sl] += r3(toff)
            self.ent[g, li, :, sl] += r3(ent)
            if li == 0:
                self.count[g, sl] += float(B * S)

            if self.verify and not self._ver_done and li == 0 and a == 0:
                o = torch.matmul(p, v.float()).transpose(1, 2).reshape(B, Q, C)
                self._ver_buf = (a, b, m.proj(o.to(x.dtype)).float())
                self._ver_done = True

    # ---- 결과 -------------------------------------------------------------------
    def numpy(self) -> dict:
        """정규화된 집계. mass/same 의 행 합은 1 (수치오차 범위)."""
        c = self.count.clamp_min(1.0)
        n4 = c[:, None, None, :, None]
        n3 = c[:, None, None, :]
        out = {
            "mass": (self.mass / n4).cpu().numpy().astype(np.float32),
            "same": (self.same / n4).cpu().numpy().astype(np.float32),
            "sdist_patch": (self.sdist / n3).cpu().numpy().astype(np.float32),
            "tdist_tub": (self.tdist / n3).cpu().numpy().astype(np.float32),
            "toff_tub": (self.toff / n3).cpu().numpy().astype(np.float32),
            "entropy": (self.ent / n3).cpu().numpy().astype(np.float32),
            "n_query": self.count.cpu().numpy().astype(np.float64),
            "n_clips": self.n_clips.copy(),
        }
        # 사람이 읽는 단위로도 같이 낸다 (패치 -> 픽셀, 튜블릿 -> 샘플 프레임)
        out["sdist_px"] = out["sdist_patch"] * self.grid.patch
        out["tdist_frames"] = out["tdist_tub"] * self.grid.tubelet
        out["toff_frames"] = out["toff_tub"] * self.grid.tubelet
        return out
