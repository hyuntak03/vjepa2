"""잠재 행동 (latent action) 블록 AR — 2026-10-24, 사용자 결정 ("learning latent action in the wild 구현해서 학습").
원 논문: Garrido et al. 2026, "Learning Latent Action World Models In The Wild" (arXiv 2601.05230) — 고정 V-JEPA 2 encoder ·
4 fps · 2 장 = latent 하나 · L1 · teacher forcing 위에 역동역학 모듈 z_t = g(s_t, s_{t+1}) + 행동 조건 world model p(s_{0:t}, z_t).

여기 판 (우리 블록 AR 에 맞춤):
  역동역학 g : 블록 t · t+1 정답 토큰 (LN(target_encoder)) → 칸마다 [s_t, s_{t+1} − s_t] → 투영 · transformer 4 층 → query 1 개 모으기 → (μ, log σ²) 128 차원
               잡음 방식 (논문 "noisy", VAE 식): 학습 z = μ + σ·ε, 손실 β·KL(q(z|s_t,s_{t+1}) ‖ N(0, I))  (논문 β ∈ 1e-6 … 5e-3)
  조건      : z → SiLU → Linear (bias 없음, 0 초기화) → 층마다 (이동 · 배율 · 게이트) × (attention · MLP) = AdaLN-zero 식.
               블록 t 의 토큰은 z_t (전이 t→t+1) 로 변조된다 (논문의 frame-wise 조건). 잔차는 x + (1 + gate)·f(…) — 0 초기화면 원래 블록 그대로.
               ★ bias 가 없어서 **z = 0 이면 변조가 정확히 0** → 저장한 predictor (키 그대로) 를 기존 채점기로 돌리면 "추론 때 z = 0" 이 된다.
  학습      : TF (위치 0..nb−2, z_0..z_{nb−2}) + rollout (문맥 블록 b 는 z_b, 걸음 r 은 z_{C−1+r}) 모두 정답 쌍의 z (teacher forcing).
  추론      : 기본 z = 0 (미래 정보 없음). 다른 규칙 (문맥 전이만 μ, 마지막 문맥 z 유지) 은 분석 스크립트에서.
변조는 predictor 모듈을 갈아끼우지 않고 hook 으로 건다 (state_dict 키가 그대로라 채점기 strict 로드가 된다). holder 가 비면 hook 은 아무것도 안 한다.
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class IDM(nn.Module):
    """역동역학: (s_t, s_{t+1}) 각 (M, S, D) → (μ, log σ²) 각 (M, z_dim).
    2026-10-24 (첫 실행 OOM, 장당 95 GB): 칸을 pool×pool 평균으로 줄이고 (16×16 → 8×8) 층마다 activation checkpointing."""

    def __init__(self, in_dim: int = 1280, dim: int = 384, depth: int = 4, heads: int = 6, z_dim: int = 128, n_tok: int = 256,
                 pool: int = 2, ckpt: bool = True):
        super().__init__()
        self.g = int(round(n_tok ** 0.5)); self.pool = int(pool); self.ckpt = bool(ckpt)
        n = (self.g // self.pool) ** 2
        self.inp = nn.Sequential(nn.Linear(2 * in_dim, dim), nn.LayerNorm(dim))
        self.pos = nn.Parameter(torch.zeros(1, n, dim)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, "gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * z_dim))

    def _pool(self, s):
        if self.pool == 1:
            return s
        M, S, D = s.shape
        x = s.view(M, self.g, self.g, D).permute(0, 3, 1, 2)
        return F.avg_pool2d(x, self.pool).flatten(2).transpose(1, 2)

    def forward(self, sa: torch.Tensor, sb: torch.Tensor):
        sa, sb = self._pool(sa), self._pool(sb)
        u = self.inp(torch.cat([sa, sb - sa], -1)) + self.pos
        for lyr in self.blocks.layers:
            u = torch.utils.checkpoint.checkpoint(lyr, u, use_reentrant=False) if (self.ckpt and self.training) else lyr(u)
        o, _ = self.att(self.q.expand(u.size(0), -1, -1), u, u, need_weights=False)
        mu, logvar = self.out(o[:, 0]).float().chunk(2, -1)
        return mu, logvar.clamp(-12.0, 8.0)


class LatentActionAR(nn.Module):
    """predictor (kind=ar, ar_feedback=state) 를 감싼다. forward(h, ar=…) → (p_tf, p_ar, kl, 로그). DDP 는 이 모듈을 감싼다."""

    def __init__(self, pred: nn.Module, cfg: dict, embed_dim: int):
        super().__init__()
        if getattr(pred, "gating_residual", False) or getattr(pred, "use_activation_checkpointing", False):
            raise ValueError("latent action 은 gating_residual · activation checkpointing 과 같이 쓰지 않는다 (hook 변조를 다시 계산하면 어긋난다)")
        if getattr(pred, "ar_feedback", "output") != "state":
            raise ValueError("latent action 은 ar_feedback: state 판에서만 검증했다")
        self.pred = pred
        self.S = pred.tokens_per_block
        self.L = len(pred.predictor_blocks)
        self.pd = pred.predictor_embed.out_features
        self.z_dim = int(cfg.get("z_dim", 128))
        self.idm = IDM(embed_dim, int(cfg.get("idm_dim", 384)), int(cfg.get("idm_depth", 4)), int(cfg.get("idm_heads", 6)), self.z_dim, self.S,
                       int(cfg.get("idm_pool", 2)), bool(cfg.get("idm_checkpoint", True)))
        self.zmod = nn.Sequential(nn.SiLU(), nn.Linear(self.z_dim, self.L * 6 * self.pd, bias=False))
        nn.init.zeros_(self.zmod[1].weight)
        self._m = None                                   # (B, nblk, L, 6, pd) — hook 이 읽는다. None 이면 변조 없음
        for i, blk in enumerate(pred.predictor_blocks):
            blk.norm1.register_forward_hook(self._norm_hook(i, 0))
            blk.norm2.register_forward_hook(self._norm_hook(i, 3))
            blk.attn.register_forward_hook(self._gate_hook(i, 2))
            blk.mlp.register_forward_hook(self._gate_hook(i, 5))
            fkv = blk.attn.forward_kv                    # KV 경로는 __call__ 을 안 거쳐서 hook 이 안 걸린다 → 인스턴스 함수로 감싼다
            def _fkv(*a, _f=fkv, _i=i, **k):
                y, kv = _f(*a, **k)
                return self._gate(y, _i, 2), kv
            blk.attn.forward_kv = _fkv

    # ---------------------------------------------------------------- 변조
    def _split(self, x, i, j):
        """x (B, N, pd), N = nblk·S → 변조 성분 j 를 블록별로 맞춘 (B, nblk, 1, pd) 와 x 의 블록 모양."""
        B, N, D = x.shape
        m = self._m
        nblk = N // self.S
        if m.size(1) != nblk or nblk * self.S != N:
            raise RuntimeError(f"latent action 변조 블록 수 {m.size(1)} 와 입력 토큰 {N} (= {N / self.S:.2f} 블록) 이 안 맞는다")
        return m[:, :, i, j].unsqueeze(2).to(x.dtype), x.view(B, nblk, self.S, D)

    def _norm_hook(self, i, j):
        def f(mod, inp, out):
            if self._m is None:
                return out
            shift, xb = self._split(out, i, j)
            scale, _ = self._split(out, i, j + 1)
            return (xb * (1 + scale) + shift).view_as(out)
        return f

    def _gate(self, y, i, j):
        if self._m is None:
            return y
        g, yb = self._split(y, i, j)
        return (yb * (1 + g)).view_as(y)

    def _gate_hook(self, i, j):
        def f(mod, inp, out):
            if isinstance(out, tuple):
                return (self._gate(out[0], i, j),) + tuple(out[1:])
            return self._gate(out, i, j)
        return f

    def mods(self, z):
        """z (B, n, z_dim) → (B, n, L, 6, pd)."""
        B, n, _ = z.shape
        return self.zmod(z.to(self.zmod[1].weight.dtype)).view(B, n, self.L, 6, self.pd)

    # ---------------------------------------------------------------- 역동역학
    def infer(self, h, sample: bool):
        """h (B, nb·S, D) → z (B, nb−1, z_dim) (전이 0..nb−2), μ, log σ²."""
        B, N, D = h.shape
        nb = N // self.S
        hb = h.view(B, nb, self.S, D)
        mu, lv = self.idm(hb[:, :-1].reshape(-1, self.S, D), hb[:, 1:].reshape(-1, self.S, D))
        mu, lv = mu.view(B, nb - 1, -1), lv.view(B, nb - 1, -1)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return z, mu, lv

    # ---------------------------------------------------------------- 학습 forward
    def forward(self, h, ar):
        S, P = self.S, self.pred
        idx, C, R = ar["idx"], int(ar["n_ctx_blocks"]), int(ar["rollout_steps"])
        nb = h.size(1) // S
        z, mu, lv = self.infer(h.detach(), sample=self.training)
        kl = (0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv)).sum(-1).mean()
        m = self.mods(z)                                                     # (B, nb−1, L, 6, pd)
        try:
            self._m = m                                                      # TF: 위치 0..nb−2 = 블록 0..nb−2 입력, z_0..z_{nb−2}
            p_tf = P.forward_seq(h[:, :-S], idx[:, :-S])
            caches = None
            for b in range(C):                                               # rollout: 문맥 블록 b 는 z_b
                self._m = m[:, b: b + 1]
                p, caches, st = P._ar_step_kv(h[:, b * S:(b + 1) * S], idx[:, b * S:(b + 1) * S], caches)
            preds, pos = [p], idx[:, (C - 1) * S: C * S]
            for r in range(1, R):                                            # 걸음 r: 위치 C−1+r, z_{C−1+r}
                pos = pos + S
                self._m = m[:, C - 1 + r: C + r]
                p, caches, st = P._ar_step_kv(st + P.type_embed[1], pos, caches, embedded=True)
                preds.append(p)
        finally:
            self._m = None
        log = {"kl": float(kl.detach()), "zstd": float(z.detach().float().std()), "mu2": float(mu.detach().pow(2).mean()),
               "sig": float((0.5 * lv.detach()).exp().mean())}
        return p_tf, torch.cat(preds, dim=1), kl, log


def lam_losses(lam, h, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, beta: float):
    """ar_losses 의 잠재 행동 판. lam = LatentActionAR (DDP 래퍼여도 된다). return (loss, jloss, sloss, R, 로그)."""
    from app.vjepa_frozen.ar import block_indices, l1
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K or nb < 2:
        raise ValueError(f"lam_losses: nb={nb} 가 C+K={C + K} 와 다르거나 2 미만")
    R = max(1, min(int(K), int(rollout_steps)))
    p_tf, p_ar, kl, log = lam(h, ar={"idx": block_indices(B, nb, S, h.device), "n_ctx_blocks": C, "rollout_steps": R})
    jloss = l1(p_tf, h[:, S:], loss_exp)
    sloss = l1(p_ar, h[:, C * S: (C + R) * S], loss_exp)
    return jloss + sloss + float(beta) * kl, jloss, sloss, R, log
