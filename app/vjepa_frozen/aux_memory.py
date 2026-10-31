"""보조 기억 모듈 — 고정 릴리즈 predictor 를 돕는 B 안 (2026-10-30, 사용자 "ㅇㅇ 하고 학습까지 바로 해봐").

구조
  aux   : 문맥 토큰 z (C 블록 × 256, 1280) → 미래 칸마다 기억 벡터 m (1280). 릴리즈 predictor 와 **같은 클래스**를 작게 (폭 256 · 4 층 · 4 head,
          마스크 토큰 1 개, RoPE 시간·세로·가로) 지어 그대로 쓴다 — 문맥 토큰을 칸째로 옮기지 않고 내용을 새로 만든다
          (latent 에서 토큰 이동은 무효로 쟀다: CLAUDE.md predictor 관문 1 · z_analysis/results/transport_upper_bound).
  cross : 고정 릴리즈 predictor 의 지정 층 (기본 4 · 8 · 12 번째 블록 뒤) 에서 **미래 (마스크) 토큰만** m 을 cross-attention 으로 읽는다.
          Q = 그 층의 은닉 (384), K · V = m 을 384 로 투영, 같은 3 축 RoPE (위치 = 미래 토큰 번호). 출력에 층별 gate (0 에서 시작) 를 곱해 더한다.
          → 학습 전에는 릴리즈와 비트 단위로 같다. 문맥 토큰은 이 경로를 쓰지 않는다.
학습: 릴리즈 predictor 는 고정 (requires_grad False), aux · cross · gate 만. 손실은 그대로 (미래 토큰 latent L1). 라벨 없음.
  ⚠️ 2026-10-31 측정: 라벨 없이는 영속을 안 배운다 — 손실 감소가 칸 종류와 무관하게 고르다 (재등장 칸 +2.75 % < 배경 +3.33 %,
  z_analysis/results/aux_perm_indomain). → sup (아래) 로 기억 m 에 물체 칸 감독을 건다.
sup (2026-10-31, 사용자 "모듈한테 context 보고 물체 놓아져야될 칸 supervision"): 기억 m 위에 선형 head (1280 → 3) — 미래 칸마다
  배경 / 보이는 물체 / 가려진 물체 (문맥에서 본 물체만). forward(return_occ=True) 가 logits 를 함께 돌려주고 학습 루프가 CE 를 건다
  (라벨: z_training/data/build_perm_occupancy.py). 채점에는 안 쓴다.
  ⚠️ 2026-10-31 측정: 칸 분류만으로는 phys3d 안에서만 따라가고 VOBench 에서는 "문맥 끝 그대로" 수준 (z_analysis/results/aux_occ_generalization).
sup.disp (2026-10-31, 상대 표현): 기억 m 위 선형 head (1280 → 2) — 미래 블록 t 의 칸 (r, c) 에서, 문맥에서 그 칸에 마지막으로 보였던 물체가
  블록 t 까지 움직인 양 (dx, dy, 칸 단위). 칸에 묶인 상대량이라 장면 배치로 외울 수 없다. 근거: 문맥 토큰의 속도 선형 probe 가
  phys3d → VOBench 로 방향 cos 0.96 (z_analysis/results/velocity_probe_cross_generator).
  2026-10-31 측정: 이동량은 VOBench 로 넘어간다 (EPE 2.75 vs 0 이동 3.99, cos 0.85) 그런데 VOBench 채점은 그대로 — 고정 predictor 가 못 그린다.
train_pred (2026-10-31, 사용자 "당연히 predictor 는 열어야지"): 릴리즈 predictor 도 학습 (학습률은 train.py 가 pred_lr_mult 로 낮춘다).
sup.p_weight: predictor **출력** p 위 칸 분류 head (1280 → 3) — 미래 칸마다 보이는 물체 / 가려진 물체를 p 에서 읽을 수 있어야 한다
  (재등장 칸에 물체를 실제로 그리라는 압력. 기억 m 의 head 와 별개).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.utils import apply_masks
from src.models.utils.modules import rotate_queries_or_keys as _rot


class AuxCross(nn.Module):
    """미래 토큰 은닉 (B, N, dim) 이 같은 미래 토큰 자리의 기억 m (B, N, kv_dim) 을 읽는다. 출력 = gate × proj(attn)."""

    def __init__(self, dim: int, kv_dim: int, heads: int, grid: int):
        super().__init__()
        self.heads, self.hd, self.grid = heads, dim // heads, grid
        self.rd = 2 * ((self.hd // 3) // 2)                     # 축마다 회전 차원 (릴리즈 RoPEAttention 과 같은 식)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        for m in (self.q, self.kv, self.proj):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def _rope(self, t, pos):
        tpf = self.grid * self.grid
        d = pos // tpf; hh = (pos - d * tpf) // self.grid; ww = pos - d * tpf - hh * self.grid
        s, parts = 0, []
        for p in (d, hh, ww):
            parts.append(_rot(t[..., s:s + self.rd], pos=p.unsqueeze(1).expand(-1, self.heads, -1))); s += self.rd
        parts.append(t[..., s:])
        return torch.cat(parts, -1)

    def forward(self, x, m, pos):
        B, N, D = x.shape
        q = self.q(self.norm(x)).view(B, N, self.heads, self.hd).transpose(1, 2)
        k, v = self.kv(m).view(B, N, 2, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self._rope(q, pos), self._rope(k, pos)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.gate * self.proj(o.transpose(1, 2).reshape(B, N, D))


class AuxRelease(nn.Module):
    """고정 릴리즈 predictor + 보조 기억 모듈. forward 는 릴리즈 VisionTransformerPredictor.forward 와 같은 계산에 cross 만 끼운다."""

    def __init__(self, pred: nn.Module, cfg: dict, embed_dim: int, img_size: int, patch: int, tubelet: int, num_frames: int):
        super().__init__()
        from analysis.intphys2.model import _build_predictor
        self.pred = pred
        self.train_pred = bool(cfg.get("train_pred", False))
        for p in self.pred.parameters():
            p.requires_grad_(self.train_pred)
        self.aux = _build_predictor(img_size=img_size, patch_size=patch, tubelet_size=tubelet, num_frames=num_frames,
                                    encoder_embed_dim=embed_dim, predictor_embed_dim=int(cfg.get("dim", 256)),
                                    predictor_depth=int(cfg.get("depth", 4)), predictor_num_heads=int(cfg.get("heads", 4)),
                                    num_mask_tokens=1, use_rope=True, uniform_power=False, use_sdpa=True,
                                    extra={"use_activation_checkpointing": False, "zero_init_mask_tokens": False})
        pd = self.pred.predictor_embed.out_features
        self.layers = [int(i) for i in cfg.get("layers", [3, 7, 11])]   # 0 부터 센 블록 번호 (그 블록 뒤에 끼운다)
        self.cross = nn.ModuleDict({str(i): AuxCross(pd, embed_dim, int(cfg.get("cross_heads", 6)), img_size // patch) for i in self.layers})
        sup = cfg.get("sup") or {}
        self.occ_head = nn.Linear(embed_dim, 3) if sup else None   # 미래 칸: 배경 / 보이는 물체 / 가려진 물체
        self.disp_head = nn.Linear(embed_dim, 2) if sup.get("disp_labels") else None   # 미래 칸: 그 칸의 문맥 물체 이동량 (dx, dy)
        self.p_occ_head = nn.Linear(embed_dim, 3) if sup.get("p_weight") else None   # predictor 출력 p 의 미래 칸 분류
        self.last_occ = self.last_disp = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.pred.eval()                                         # 고정 predictor 는 늘 eval (dropout · drop_path 없음과 같게)
        return self

    def gates(self):
        return {i: float(self.cross[str(i)].gate) for i in self.layers}

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False, return_occ=False):
        assert not has_cls
        P = self.pred
        mx = masks_x[0] if isinstance(masks_x, list) else masks_x
        my = masks_y[0] if isinstance(masks_y, list) else masks_y
        m = self.aux(x, mx, my, mask_index=0)                    # (B, Nt, embed_dim), masks_y 순서
        self.last_occ = self.occ_head(m) if self.occ_head is not None else None   # (B, Nt, 3), masks_y 순서
        self.last_disp = self.disp_head(m) if self.disp_head is not None else None  # (B, Nt, 2), 칸 단위
        B = x.size(0)
        h = P.predictor_embed(x); n_ctx = h.size(1)
        pt = P.mask_tokens[mask_index % P.num_mask_tokens].repeat(B, P.num_patches, 1)
        pt = apply_masks(pt, [my])
        h = torch.cat([h, pt], dim=1)
        masks = torch.cat([mx, my], dim=1)
        order = torch.argsort(masks, dim=1)
        masks = torch.gather(masks, 1, order)
        g = lambda t, idx: torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))
        h = g(h, order)
        idx_t = torch.nonzero(order >= n_ctx)[:, 1].view(B, -1)  # 정렬된 열에서 미래 토큰 자리 (행마다 오름차순)
        m_s = g(m, torch.gather(order, 1, idx_t) - n_ctx)       # 그 자리의 기억
        pos_t = torch.gather(masks, 1, idx_t)
        for i, blk in enumerate(P.predictor_blocks):
            h = blk(h, mask=masks, attn_mask=None)
            if str(i) in self.cross:
                ht = g(h, idx_t)
                h = h.scatter(1, idx_t.unsqueeze(-1).expand(-1, -1, h.size(-1)), ht + self.cross[str(i)](ht, m_s, pos_t))
        h = P.predictor_norm(h)
        h = g(h, torch.argsort(order, dim=1))[:, n_ctx:]
        out = P.predictor_proj(h)
        # return_occ: 학습 루프 (aux.sup) 가 칸 logits 를 **forward 출력으로** 받는다 — 부수 속성으로만 내보내면 DDP (find_unused_parameters) 가
        #   occ_head 를 안 쓴 파라미터로 보고 먼저 ready 처리한 뒤 CE backward 에서 한 번 더 ready → 오류 (2026-10-31 실측)
        if return_occ:
            return out, {"occ": self.last_occ, "disp": self.last_disp, "p_occ": self.p_occ_head(out) if self.p_occ_head is not None else None}
        return out
