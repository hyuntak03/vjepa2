"""motion (2026-10-24, 사용자 결정) — MotionJEPA (Karmann et al. 2026, arXiv 2609.23881) 의 DISReg 를 frozen encoder 블록 AR 에 옮긴다.

원본: 차분 encoder  d_t = DiffEnc(o_{t+1} − o_t)  (픽셀 차분 영상 → CLS → projector)
      차분 예측기  d̂_t = DiffPred(z_t, z_{t+1})  (3 층 MLP)
      L = MSE(z_{t+1}, ẑ_{t+1}) + 0.25·SIGReg(z) + 2·SIGReg(d) + 0.5·MSE(d̂, d)       (encoder 까지 학습)

여기 (encoder 고정 → z 는 바뀌지 않으므로 SIGReg(z) 는 없다):
  ① 맞추기 (정답 경로) — d = DiffEnc(블록 b+1 프레임 − 블록 b 프레임),  d̂ = DiffPred(h_b, h_{b+1})  (h = 정답 토큰 LN(target_encoder(블록)))
       w_sigreg · SIGReg(d) + w_fit · MSE(d̂, d)  →  DiffEnc · DiffPred 만 학습
  ② 예측 경로 — 같은 DiffPred (가중치 고정) 를 predictor 출력 쌍에:  TF (h_b, p̂_{b+1}) · rollout (p̂_{b}, p̂_{b+1}) (첫 쌍은 마지막 문맥 h)
       w_pred · MSE(DiffPred_frozen(·, p̂), d.detach())  →  predictor 만 학습
     예측이 정답 토큰과 같은 방식으로 "블록 사이에 바뀐 것" 을 담아야 내려간다 (멈춘 세계 = 바뀐 것 없음 은 손실이 크다).
     DiffPred 를 예측 쪽에서 같이 학습하면 head 와 predictor 가 따로 적힌 코드로 우회한다 (2026-10-23 state head 측정) — 그래서 끊는다.
  시험 · 채점에는 이 모듈을 쓰지 않는다 (predictor 체크포인트만). 미래 정보는 학습 손실의 타깃으로만 들어간다.

SIGReg 는 원본 코드 (motion-jepa/src/models/lewm/module.py, 단일 GPU 판) 그대로 — 단 통계량의 N 배율을 고정값 ref_n (원본 batch 128) 로 둔다.
  우리 batch 는 (C, K) 에 따라 N = B·(nb−1) 이 237~475 로 변해서, N 을 곱하면 같은 가중치가 step 마다 다른 세기가 된다.
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class SIGReg(nn.Module):
    """Epps–Pulley 를 무작위 방향으로 평균 (LeJEPA). 원본: t ∈ [0, 3] 17 knot, 사다리꼴 가중 × exp(−t²/2), 방향 1024 개를 step 마다 새로."""

    def __init__(self, knots: int = 17, num_proj: int = 1024, ref_n: float = 128.0):
        super().__init__()
        self.num_proj, self.ref_n = int(num_proj), float(ref_n)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[[0, -1]] = dt
        win = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", win)
        self.register_buffer("weights", w * win)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, D) — 표본 N 개. return 스칼라. autocast 밖 fp32 (cos · sin 평균이 bf16 에서 뭉개진다)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            A = torch.randn(x.size(-1), self.num_proj, device=x.device)
            A = A / A.norm(p=2, dim=0)
            xt = (x @ A).unsqueeze(-1) * self.t                                     # (N, P, knots)
            err = (xt.cos().mean(0) - self.phi).square() + xt.sin().mean(0).square()   # (P, knots)
            return (err @ self.weights).mean() * self.ref_n


class DiffEncoder(nn.Module):
    """차분 영상 encoder — ViT-Tiny (원본과 같은 크기: 192 차원 · 12 층 · 3 head), CLS → projector (Linear · BatchNorm · GELU · Linear).
    입력 = 블록 둘의 프레임별 차분을 채널로 쌓은 것 (tubelet 2 → 6 채널), patch 16 → 칸 256 개 (V-JEPA 토큰 격자와 같다)."""

    def __init__(self, img_size: int = 256, patch: int = 16, in_ch: int = 6, dim: int = 192, depth: int = 12, heads: int = 3,
                 out_dim: int = 192, proj_hidden: int = 2048, ckpt: bool = True):
        super().__init__()
        self.ckpt = bool(ckpt)     # activation checkpointing (층마다). 최대 창에서 차분 영상 ~430 장이라 activation 이 수 GB
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.GELU(), nn.Linear(proj_hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (M, in_ch, H, W) → (M, out_dim)"""
        t = self.patch_embed(x).flatten(2).transpose(1, 2)                      # (M, n, dim)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1) + self.pos
        for lyr in self.blocks.layers:
            t = torch.utils.checkpoint.checkpoint(lyr, t, use_reentrant=False) if (self.ckpt and self.training) else lyr(t)
        t = self.norm(t)
        return self.proj(t[:, 0].float())


class DiffPredictor(nn.Module):
    """차분 예측기 — 칸마다 [z_b, z_{b+1} − z_b] (2·D) → MLP (LN · GELU) → 칸 평균 → MLP → d̂.
    원본 (CLS 벡터 둘 → 3 층 MLP, hidden 512, 층마다 LN · GELU) 을 토큰 격자에 맞춘 판."""

    def __init__(self, in_dim: int = 1280, hidden: int = 512, out_dim: int = 192):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        """za, zb (M, S, D) — 앞 · 뒤 블록 토큰. return (M, out_dim)"""
        u = self.tok(torch.cat([za, zb - za], -1))
        return self.head(u.mean(1)).float()


class AttnDiffPredictor(nn.Module):
    """2026-10-24 진단 C 판: 칸마다 [z_b, z_{b+1} − z_b] → MLP → transformer 1 층 (칸 위치 임베딩) → query q 개 attention 모으기 → MLP → d̂.
    칸 평균은 "어디가 바뀌었나" 를 지운다 — 모으기를 학습한다 (motion.pred_type: attn)."""

    def __init__(self, in_dim: int = 1280, hidden: int = 512, out_dim: int = 192, n_tok: int = 256, q: int = 8, heads: int = 8):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, n_tok, hidden)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.mix = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, 0.0, "gelu", batch_first=True, norm_first=True)
        self.q = nn.Parameter(torch.randn(1, q, hidden) * 0.02)
        self.att = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(q * hidden), nn.Linear(q * hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        u = self.mix(self.tok(torch.cat([za, zb - za], -1)) + self.pos)
        o, _ = self.att(self.q.expand(u.size(0), -1, -1), u, u, need_weights=False)
        return self.head(o.flatten(1)).float()


class Motion(nn.Module):
    def __init__(self, cfg: dict, embed_dim: int, img_size: int, patch: int, tubelet: int):
        super().__init__()
        dim = int(cfg.get("dim", 192))
        self.tubelet = int(tubelet)
        self.diff = str(cfg.get("diff", "block"))           # block (블록 b+1 프레임 i − 블록 b 프레임 i, 0.5 초) | boundary (경계 프레임 2b+2 − 2b+1, 0.25 초)
        if self.diff not in ("block", "boundary"):
            raise ValueError(f"motion.diff 는 block | boundary: {self.diff!r}")
        in_ch = 3 * self.tubelet if self.diff == "block" else 3
        self.enc = DiffEncoder(img_size, patch, in_ch, int(cfg.get("enc_dim", 192)), int(cfg.get("enc_depth", 12)),
                               int(cfg.get("enc_heads", 3)), dim, int(cfg.get("proj_hidden", 2048)), bool(cfg.get("enc_checkpoint", True)))
        pt = str(cfg.get("pred_type", "mean"))             # mean (칸 평균, 처음 판) | attn (attention 모으기)
        if pt == "mean":
            self.pred = DiffPredictor(embed_dim, int(cfg.get("pred_hidden", 512)), dim)
        elif pt == "attn":
            self.pred = AttnDiffPredictor(embed_dim, int(cfg.get("pred_hidden", 512)), dim, (img_size // patch) ** 2)
        else:
            raise ValueError(f"motion.pred_type 은 mean | attn: {pt!r}")
        sg = cfg.get("sigreg") or {}
        self.sigreg = SIGReg(int(sg.get("knots", 17)), int(sg.get("num_proj", 1024)), float(sg.get("ref_n", 128)))


def diff_images(clips: torch.Tensor, tubelet: int, kind: str = "block") -> torch.Tensor:
    """clips (B, 3, nb·tubelet, H, W) 정규화된 프레임 → 전이 b→b+1 마다 차분 영상.
    block    (B·(nb−1), 3·tubelet, H, W) = 블록 b+1 의 프레임 i − 블록 b 의 프레임 i (간격 = 블록 하나)
    boundary (B·(nb−1), 3, H, W)         = 블록 b+1 첫 프레임 − 블록 b 마지막 프레임 (연속 두 프레임, 원본 MotionJEPA 와 같은 간격)"""
    B, C3, T, H, W = clips.shape
    nb = T // tubelet
    x = clips[:, :, : nb * tubelet].reshape(B, C3, nb, tubelet, H, W)
    if kind == "boundary":
        d = x[:, :, 1:, 0] - x[:, :, :-1, -1]                                   # (B, 3, nb−1, H, W)
        return d.permute(0, 2, 1, 3, 4).reshape(B * (nb - 1), C3, H, W)
    d = x[:, :, 1:] - x[:, :, :-1]                                              # (B, 3, nb−1, tubelet, H, W)
    return d.permute(0, 2, 1, 3, 4, 5).reshape(B * (nb - 1), C3 * tubelet, H, W)


def _pairs(z: torch.Tensor, S: int):
    """z (B, n·S, D) → 이웃 쌍 (B·(n−1), S, D) 둘."""
    B, N, D = z.shape
    n = N // S
    z = z.reshape(B, n, S, D)
    return z[:, :-1].reshape(-1, S, D), z[:, 1:].reshape(-1, S, D)


def motion_losses(mod: Motion, clips: torch.Tensor, h: torch.Tensor, p_tf: torch.Tensor, p_ar: torch.Tensor,
                  S: int, C: int, R: int, cfg: dict):
    """h (B, nb·S, D) 정답 토큰 · p_tf (B, (nb−1)·S, D) = 블록 1..nb−1 TF 예측 · p_ar (B, R·S, D) = 블록 C..C+R−1 rollout.
    return (loss, 로그). loss = ① (DiffEnc · DiffPred 만 grad) + ② (predictor 만 grad)."""
    B, N, D = h.shape
    nb = N // S
    d = mod.enc(diff_images(clips, mod.tubelet, mod.diff)).view(B, nb - 1, -1)          # (B, nb−1, dd) 전이 0..nb−2
    hd = h.detach()
    za, zb = _pairs(hd, S)
    d_fit = mod.pred(za, zb).view(B, nb - 1, -1)
    l_sig = mod.sigreg(d.reshape(-1, d.size(-1)))
    l_fit = F.mse_loss(d_fit, d)
    # ② 예측 경로: DiffPred 가중치 고정 (functional_call), 타깃 d 끊음
    frozen = {n_: q.detach() for n_, q in mod.pred.named_parameters()}
    fp = lambda a, b: torch.func.functional_call(mod.pred, frozen, (a, b))
    dt = d.detach()
    ptf = p_tf.reshape(B, nb - 1, S, D)
    d_tf = fp(hd.reshape(B, nb, S, D)[:, :-1].reshape(-1, S, D), ptf.reshape(-1, S, D)).view(B, nb - 1, -1)   # 전이 b→b+1 = (h_b, p̂_{b+1})
    l_tf = F.mse_loss(d_tf, dt)
    prev = torch.cat([hd[:, (C - 1) * S: C * S], p_ar[:, : (R - 1) * S]], 1)    # rollout 쌍: (h_{C−1}, p̂_C), (p̂_C, p̂_{C+1}), …
    d_ar = fp(prev.reshape(-1, S, D), p_ar.reshape(-1, S, D)).view(B, R, -1)  # 전이 C−1 .. C+R−2
    l_ar = F.mse_loss(d_ar, dt[:, C - 1: C - 1 + R])
    sp = cfg.get("split") or {}
    loss = (float(cfg.get("w_sigreg", 2.0)) * l_sig + float(cfg.get("w_fit", 0.5)) * l_fit
            + float(cfg.get("w_pred", 0.5)) * (float(sp.get("tf", 1.0)) * l_tf + float(sp.get("ar", 1.0)) * l_ar))
    with torch.no_grad():   # 기준: "바뀐 것 없음" (뒤 블록 = 앞 블록) 을 넣었을 때의 오차. 정답 쌍 < 예측 쌍 < 멈춤 이면 예측이 움직임을 담는다
        d_copy = fp(hd.reshape(B, nb, S, D)[:, :-1].reshape(-1, S, D), hd.reshape(B, nb, S, D)[:, :-1].reshape(-1, S, D)).view(B, nb - 1, -1)
        l_copy = F.mse_loss(d_copy, dt)
        dvar = float(d.detach().float().var(0).mean())
    log = {"sig": float(l_sig.detach()), "fit": float(l_fit.detach()), "tf": float(l_tf.detach()), "ar": float(l_ar.detach()),
           "copy": float(l_copy), "dvar": dvar}
    return loss, log
