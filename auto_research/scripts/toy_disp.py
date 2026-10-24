#!/usr/bin/env python3
"""[P] H1 (공간) 장난감 — toy_factor.py ([T]) 를 복사해 출력 머리만 바꿨다 (2026-10-21, 원본은 건드리지 않는다).
  --head disp : predictor 가 칸마다 2 차원 변위 u (칸 단위) 와 작은 잔차 r 를 낸다. 출력 토큰 = 마지막 입력 장을 u 로 옮겨 쓴 것 (토러스, 쌍선형)
                + res_scale · tanh(r). 변위는 "내용이 어디서 오나" (역방향): 출력(c) = x_last(c + u(c)).
  --head token : 원본과 같다 (비교용).
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python auto_research/scripts/toy_disp.py --name d_b_j06 --head disp --jit 0.6 --p_stop 0.4 \
      --vmin 1 --vmax 2 --rope 1 --obj_fixed 1.0 --steps 6000 --branch_eval 1 --out auto_research/results/toy_disp
이하 원본 설명:
[T] 요인 장난감: 잡음 비 (jit) × 운동 불확실성 (sig_a) × 정답 잡음 (jit_tgt) — "잡음 → 평균 → 운반 학습 지연/부재" 를 직접 본다
(auto_research, 2026-10-21. `toy_occupancy.py` 를 복사해 고쳤다. 원본은 진행 중 run 이 쓰므로 건드리지 않는다).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name f_j06 --jit 0.6

원본과 다른 점
  * 기본 크기를 줄였다: G 8 (64 칸), dim 128, depth 4, C 4 · K 4 → 한 step 이 원본의 약 1/30.
  * 입력 잡음 jit 과 **정답 잡음 jit_tgt** 를 나눴다. 정답 = 깨끗한 렌더 + (jit_tgt / jit) × (입력과 같은 잡음 실현).
    jit_tgt = jit 이면 원본과 같다 (정답 = 관측). jit_tgt = 0 이면 "정답만 잡음을 지운" 개입 (실제 latent 의 randomized smoothing 대응).
  * 학습 없는 기준 예측기 (같은 데이터 위) 의 손실: copy · mean_C · stay_dn (배경 = 물체 없던 문맥 장 평균, 물체 = 문맥 궤적 평균, **옛 자리**) ·
    oracle (같은 추정을 **참 새 자리**에) → "평균 보상" = L(copy) − L(stay_dn), "운반 보상" = L(stay_dn) − L(oracle). 실제 latent 의 12.6 % / 0.7 % 와 대조.
  * 지표 추가: bg_ratio = 먼 배경 칸에서 |p − 깨끗한 배경| / |x_t − 깨끗한 배경| (1 = 복사, 1/√C 근처 = 평균), pres_old (물체가 떠난 칸에 남은 존재감).
세계 · 모델 · 운반 이득 정의는 원본과 같다 (16×16 → G×G 토러스, 2×2 물체, 속도 0.5–1.5 칸/장, block-causal transformer, TF 전 위치 + rollout R 걸음)."""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/toy_disp"))
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--D", type=int, default=32)
ap.add_argument("--w", type=float, default=2.0)
ap.add_argument("--contrast", type=float, default=1.0)
ap.add_argument("--jit", type=float, default=0.1, help="입력 (관측) 잡음")
ap.add_argument("--jit_tgt", type=float, default=None, help="정답 잡음 (기본 = jit). 같은 잡음 실현을 jit_tgt/jit 배로")
ap.add_argument("--sig_a", type=float, default=0.0)
ap.add_argument("--p0", type=float, default=0.0)
ap.add_argument("--p_stop", type=float, default=0.0, help="갈래 세계: 첫 미래 장부터 확률 p_stop 으로 멈춘다 (문맥에서는 알 수 없음) → 미래 = 간다 (1−p_stop) / 멈춘다 (p_stop) 두 갈래")
ap.add_argument("--branch_eval", type=int, default=0, help="1 = 학습 끝에 갈래 평가 (간다 자리 · 멈춘 자리 진하기, 분포 예측이면 표본마다)")
ap.add_argument("--vmin", type=float, default=0.5)
ap.add_argument("--vmax", type=float, default=1.5)
ap.add_argument("--C", type=int, default=4)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--R", type=int, default=2)
ap.add_argument("--nh", type=int, default=3, help="wta: 가설 수")
ap.add_argument("--ns", type=int, default=2, help="crps: 학습 표본 수 M (공정 CRPS = 평균 |X_i − y| − Σ_{i≠j} |X_i − X_j| / (2M(M−1)))")
ap.add_argument("--wta_eps", type=float, default=0.05, help="wta: 진 가설에도 주는 몫 (Rupprecht 등 2017 의 완화)")
ap.add_argument("--loss", default="l1", choices=["l1", "l2", "crps", "wta", "es"], help="crps = 잡음 입력 predictor + 표본 2 개 공정 CRPS (= L1 + 퍼짐 항, 엄밀 적정 점수; 결정적이면 L1 과 같다)")
ap.add_argument("--nz", type=int, default=16, help="crps: 잡음 입력 차원 (영상 하나에 하나, 모든 칸 · 장에 공유)")
ap.add_argument("--xi_fix", type=float, default=0.0, help=">0 이면 ξ 척도 s 를 이 값으로 고정 (학습 안 함)")
ap.add_argument("--xi", default="none", choices=["none", "gauss", "laplace"], help="crps: 표본 = g(H, z) + s ⊙ ξ, ξ = 좌표마다 독립 잡음 (블록 고유 성분 담당), s = 차원별 학습 척도. 평가는 g 에")
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", default="tf", choices=["tf", "cnn"], help="tf = 블록 AR 형태 transformer, cnn = 원형 패딩 합성곱 (최근 C 장을 채널로 쌓음)")
ap.add_argument("--width", type=int, default=128, help="cnn 채널 수")
ap.add_argument("--obj_fixed", type=float, default=0.0, help=">0 이면 물체 토큰 = 고정 방향 (모든 영상 공통, 크기 obj_fixed·√D 쯤) + 영상마다 작은 정체 (표준편차 0.3). 물체를 배경과 구별할 수 있게 (실제 latent 처럼)")
ap.add_argument("--head", default="token", choices=["token", "disp"])
ap.add_argument("--res_scale", type=float, default=0.1, help="disp: 잔차 크기 상한 (토큰 값 표준편차 ≈ 1 대비). 0 이면 잔차 없음")
ap.add_argument("--umax", type=float, default=3.0, help="disp: 변위 크기 상한 (칸), tanh 로 묶는다")
ap.add_argument("--disp_loss", type=int, default=0, help="1 = 변위 공간 손실: 예측 변위와 정답 변위 (물체 칸 = −(물체 이동), 나머지 0) 의 L1 (TF 전 위치). 토큰 손실은 안 쓴다")
ap.add_argument("--rope", type=int, default=0, help="1 = 공간 2D 회전 위치 부호화 (토러스 주기 조화파, 실제 predictor 처럼 상대 위치로 attention)")
a = ap.parse_args()
if a.jit_tgt is None:
    a.jit_tgt = a.jit
dev = torch.device("cuda")
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(8)
G, D, T = a.G, a.D, a.C + a.K
S = G * G
cell = torch.arange(G, device=dev).float()


def ov1(x, w):
    x = x[..., None] % G
    o = torch.zeros(*x.shape[:-1], G, device=dev)
    for sh in (-G, 0, G):
        o += (torch.minimum(cell + 1, x + sh + w) - torch.maximum(cell, x + sh)).clamp_min(0)
    return o.clamp(max=1)


def alpha(pos):
    return (ov1(pos[..., 1], a.w)[..., :, None] * ov1(pos[..., 0], a.w)[..., None, :]).flatten(-2)


def sample_motion(B, T_, x0=None, v0=None, static=None):
    if x0 is None:
        x0 = torch.rand(B, 2, device=dev) * G
        ang = torch.rand(B, device=dev) * 2 * math.pi
        sp = a.vmin + (a.vmax - a.vmin) * torch.rand(B, device=dev)
        v0 = torch.stack([ang.cos(), ang.sin()], -1) * sp[:, None]
        static = torch.rand(B, device=dev) < a.p0
    v0 = torch.where(static[:, None], torch.zeros_like(v0), v0)
    xs, x, v = [x0], x0, v0
    stop = (torch.rand(B, device=dev) < a.p_stop) if (a.p_stop > 0 and x0 is not None and T_ == T) or (a.p_stop > 0 and T_ == T) else None
    for i in range(T_ - 1):
        v = v + a.sig_a * torch.randn_like(v) * (~static[:, None]).float()
        if stop is not None and i + 1 >= a.C:                              # 장 C 부터 멈춤 (장 C−1 → C 이동 없음)
            v = torch.where(stop[:, None], torch.zeros_like(v), v)
        x = x + v
        xs.append(x)
    return torch.stack(xs, 1), static


def render_clean(pos, bg, o):
    al = alpha(pos)[..., None]
    return (1 - al) * bg[:, None] + al * (a.contrast * o)[:, None, None]


_g = torch.Generator(device="cpu"); _g.manual_seed(777)
O_FIX = torch.randn(D, generator=_g).to(dev)


def batch(B):
    pos, static = sample_motion(B, T)
    bg = torch.randn(B, S, D, device=dev)
    o = torch.randn(B, D, device=dev)
    if a.obj_fixed > 0:
        o = a.obj_fixed * O_FIX[None] + 0.3 * o
    clean = render_clean(pos, bg, o)
    eps = torch.randn_like(clean)
    x_in = clean + a.jit * eps
    x_tg = clean + a.jit_tgt * eps
    return x_in, x_tg, clean, pos, bg, o, static


NH = 4
_rope_cache = {}


def rope_cs(n):
    """칸 위치 (x, y) 의 회전 각: 축마다 토러스 조화파 k = 1..G/2 (주기 G 칸) → (N, 2·G/2) cos · sin."""
    if n not in _rope_cache:
        yy, xx = torch.div(torch.arange(S, device=dev), G, rounding_mode="floor"), torch.arange(S, device=dev) % G
        kk = torch.arange(1, G // 2 + 1, device=dev).float() * 2 * math.pi / G
        ang = torch.cat([xx[:, None] * kk, yy[:, None] * kk], 1).repeat(n, 1)          # (n·S, G)
        _rope_cache[n] = (ang.cos(), ang.sin())
    return _rope_cache[n]


def apply_rope(q, cs):
    c, sn = cs                                        # (N, P)
    P = c.shape[1]
    q1, q2, rest = q[..., :P], q[..., P:2 * P], q[..., 2 * P:]
    return torch.cat([q1 * c - q2 * sn, q1 * sn + q2 * c, rest], -1)


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj, s.h = nn.Linear(d, 3 * d), nn.Linear(d, d), h
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        B, N, d = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        if a.rope:
            cs = rope_cs(N // S)
            q, k = apply_rope(q, cs), apply_rope(k, cs)
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, N, d))
        return x + s.mlp(s.n2(x))


_gy = torch.div(torch.arange(S, device=dev), G, rounding_mode="floor").float()
_gx = (torch.arange(S, device=dev) % G).float()


def warp_torus(x, u):
    """x (B,n,S,D) 입력 장들, u (B,n,S,2) 변위 (dy,dx). 위치 i 의 출력 = x[:, i] 를 칸 c + u(c) 에서 쌍선형으로 읽은 것 (토러스)."""
    B, n = x.shape[:2]
    py = _gy[None, None] + u[..., 0]; px = _gx[None, None] + u[..., 1]
    y0 = torch.floor(py); x0 = torch.floor(px); wy = py - y0; wx = px - x0
    out = 0
    for dy, wyy in ((0, 1 - wy), (1, wy)):
        for dx, wxx in ((0, 1 - wx), (1, wx)):
            j = (((y0 + dy) % G) * G + ((x0 + dx) % G)).long()                     # (B,n,S)
            v = torch.gather(x, 2, j[..., None].expand(-1, -1, -1, x.shape[-1]))
            out = out + (wyy * wxx)[..., None] * v
    return out


class Pred(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Linear(D, a.dim)
        s.pos_t = nn.Parameter(torch.zeros(T, 1, a.dim)); s.pos_s = nn.Parameter(torch.zeros(1, S, a.dim))
        nn.init.normal_(s.pos_t, std=0.02); nn.init.normal_(s.pos_s, std=0.02)
        s.blocks = nn.ModuleList([Block(a.dim, 4) for _ in range(a.depth)])
        s.out = nn.Sequential(nn.LayerNorm(a.dim), nn.Linear(a.dim, D * (a.nh if a.loss == "wta" else 1)))
        if a.head == "disp":
            s.out = nn.Sequential(nn.LayerNorm(a.dim), nn.Linear(a.dim, 2 + D))
            nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)        # 시작 = 변위 0, 잔차 0 → 복사
        s.hyp = 0
        if a.loss in ("crps", "es"):
            s.noise = nn.Linear(a.nz, a.dim)
            if a.xi != "none":
                s.log_s = nn.Parameter(torch.full((D,), math.log(a.xi_fix if a.xi_fix > 0 else 0.05)), requires_grad=a.xi_fix <= 0)

    def forward(s, x, zn=None):
        B, n = x.shape[:2]
        z = (s.inp(x) + s.pos_t[:n] + s.pos_s).view(B, n * S, -1)
        if a.loss in ("crps", "es"):
            if zn is None:
                zn = torch.randn(B, a.nz, device=dev)
            z = z + s.noise(zn)[:, None, :]
        fi = torch.arange(n, device=dev).repeat_interleave(S)
        mask = fi[:, None] >= fi[None, :]
        for b in s.blocks:
            z = b(z, mask)
        o = s.out(z)
        if a.head == "disp":
            o = o.view(B, n, S, 2 + D)
            u = a.umax * torch.tanh(o[..., :2])                                     # (B,n,S,2) = (dy, dx) 칸
            r = a.res_scale * torch.tanh(o[..., 2:]) if a.res_scale > 0 else 0.0
            s._last_u = u
            return warp_torus(x, u) + r
        if a.loss == "wta":
            o = o.view(B, n, S, a.nh, D)
            return o if s.hyp < 0 else o[..., s.hyp, :]
        return o.view(B, n, S, D)

    def rollout(s, ctx, k, zn=None):
        if a.loss in ("crps", "es") and zn is None:
            zn = torch.randn(ctx.shape[0], a.nz, device=dev)
        seq, outs = ctx, []
        for _ in range(k):
            p = s(seq, zn)[:, -1]
            outs.append(p); seq = torch.cat([seq, p[:, None]], 1)
        return torch.stack(outs, 1)


class CNNPred(nn.Module):
    """최근 C 장 (B,C,S,D) → 다음 장 (B,S,D). 3×3 원형 패딩 합성곱 depth 층 (수용 영역 2·depth+1 칸). 잔차로 마지막 장을 더하지 않는다."""
    def __init__(s):
        super().__init__()
        L = [nn.Conv2d(a.C * D, a.width, 3, padding=1, padding_mode="circular"), nn.GELU()]
        for _ in range(a.depth - 2):
            L += [nn.Conv2d(a.width, a.width, 3, padding=1, padding_mode="circular"), nn.GELU()]
        L += [nn.Conv2d(a.width, D, 3, padding=1, padding_mode="circular")]
        s.net = nn.Sequential(*L)
        s.skip = nn.Conv2d(a.C * D, D, 1)      # 입력 C 장 → 출력 1×1 선형 경로 (복사 · 평균을 똑같이 쉽게; 마지막 장 잔차는 아님)

    def tf_next(s, seq):
        B = seq.shape[0]
        x = seq[:, -a.C:]
        if x.shape[1] < a.C:
            x = torch.cat([x[:, :1].expand(-1, a.C - x.shape[1], -1, -1), x], 1)
        x = x.view(B, a.C, G, G, D).permute(0, 1, 4, 2, 3).reshape(B, a.C * D, G, G)
        return (s.net(x) + s.skip(x)).permute(0, 2, 3, 1).reshape(B, S, D)

    def rollout(s, ctx, k):
        seq, outs = ctx, []
        for _ in range(k):
            p = s.tf_next(seq)
            outs.append(p); seq = torch.cat([seq, p[:, None]], 1)
        return torch.stack(outs, 1)


def lossfn(p, y):
    return ((p - y) ** 2).mean() if a.loss == "l2" else (p - y).abs().mean()


def es_m(ps, y):                     # 블록 노름 energy score: 장 하나 (칸 × 차원) 전체의 L2 노름. ps: M 개 (B, n, S, D)
    nrm = lambda t: t.pow(2).sum((2, 3)).clamp_min(1e-12).sqrt() / math.sqrt(S * D)      # (B, n)
    M = len(ps)
    fit = sum(nrm(p_ - y).mean() for p_ in ps) / M
    spr = sum(nrm(ps[i] - ps[j]).mean() for i in range(M) for j in range(M) if i != j) / (2 * M * (M - 1))
    return fit - spr


def crps_m(ps, y):                   # 표본 M 개 공정 CRPS
    if a.xi != "none":
        sc = model.log_s.exp()
        draw = (lambda t: torch.randn_like(t)) if a.xi == "gauss" else (lambda t: torch.distributions.Laplace(0.0, 1.0).sample(t.shape).to(t.device) / math.sqrt(2))
        ps = [p_ + sc * draw(p_) for p_ in ps]
    M = len(ps)
    fit = sum((p_ - y).abs().mean() for p_ in ps) / M
    spr = sum((ps[i] - ps[j]).abs().mean() for i in range(M) for j in range(M) if i != j) / (2 * M * (M - 1))
    return fit - spr


def crps(p1, p2, y):                 # 표본 2 개 공정 CRPS (좌표별 합 → 평균). p1 = p2 이면 L1 과 같다
    if a.loss == "crps" and a.xi != "none":                           # 표본 = g + s ⊙ ξ (ξ 를 표본마다 새로)
        sc = model.log_s.exp()
        draw = (lambda t: torch.randn_like(t)) if a.xi == "gauss" else (lambda t: torch.distributions.Laplace(0.0, 1.0).sample(t.shape).to(t.device) / math.sqrt(2))
        p1, p2 = p1 + sc * draw(p1), p2 + sc * draw(p2)
    return 0.5 * ((p1 - y).abs().mean() + (p2 - y).abs().mean()) - 0.5 * (p1 - p2).abs().mean()


model = (Pred() if a.arch == "tf" else CNNPred()).to(dev)
if a.arch == "tf":
    model.tf_next = lambda seq: model(seq)[:, -1]
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def readout(tok, o):
    cs = F.cosine_similarity(tok, (a.contrast * o)[:, None], dim=-1)
    j = cs.argmax(1); jy, jx = j // G, j % G
    offs = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], device=dev)
    yy, xx = (jy[:, None] + offs[:, 0]) % G, (jx[:, None] + offs[:, 1]) % G
    w = cs.gather(1, yy * G + xx).clamp_min(0) + 1e-6
    cy = jy.float() + (offs[:, 0].float() * w).sum(1) / w.sum(1)
    cx = jx.float() + (offs[:, 1].float() * w).sum(1) / w.sum(1)
    return torch.stack([cx, cy], -1) + 0.5, cs.amax(1)


center = lambda pos: pos + a.w / 2
wrapd = lambda d: (d + G / 2) % G - G / 2


@torch.no_grad()
def presence_mask(tok, o, m):
    cs = F.cosine_similarity(tok, (a.contrast * o)[:, None], dim=-1)
    m = m.float()
    return (cs * m).sum(1) / m.sum(1).clamp_min(1), m.sum(1) > 0


@torch.no_grad()
def baselines(x_in, x_tg, pos, static):
    """학습 없는 예측기의 TF 한 걸음 손실 (장 C-1 → C, 화면 평균, 정답 = x_tg). 움직인 clip 만."""
    C = a.C
    ctx = x_in[:, :C]; y = x_tg[:, C]
    al = alpha(pos)                                                     # (B,T,S)
    free = (al[:, :C] < 1e-3).float()[..., None]                        # (B,C,S,1)
    bg_est = (ctx * free).sum(1) / free.sum(1).clamp_min(1)
    bg_est = torch.where(free.sum(1) > 0, bg_est, ctx.mean(1))
    full = (al[:, :C] > 0.999).float()[..., None]
    o_est = (ctx * full).sum((1, 2)) / full.sum((1, 2)).clamp_min(1)    # (B,D)
    al_old, al_new = alpha(pos[:, C - 1])[..., None], alpha(pos[:, C])[..., None]
    stay = (1 - al_old) * bg_est + al_old * o_est[:, None]
    orac = (1 - al_new) * bg_est + al_new * o_est[:, None]
    c = center(pos)
    mov = (~static) & (wrapd(c[:, C] - c[:, C - 1]).norm(dim=-1) > 0.3)
    f = lambda p: (p - y).abs().mean((1, 2))[mov] if a.loss == "l1" else ((p - y) ** 2).mean((1, 2))[mov]
    return {"copy": f(x_in[:, C - 1]), "mean_C": f(ctx.mean(1)), "stay_dn": f(stay), "oracle": f(orac)}


@torch.no_grad()
def evaluate(nb=4):
    model.eval()
    torch.manual_seed(1234)
    acc = {k: [] for k in ("peak_tf", "g_tf", "g_ro", "pres_tf", "pres_old_tf", "pres_ro", "pres_gt", "bg_ratio", "bg_ratio_mean", "loss_tf",
                           "b_copy", "b_mean_C", "b_stay_dn", "b_oracle", "m_tf")}
    for _ in range(nb):
        x_in, x_tg, clean, pos, bg, o, static = batch(128)
        C = a.C
        if a.arch == "tf":
            p_tf = model(x_in[:, :-1])
            acc["loss_tf"].append(lossfn(p_tf, x_tg[:, 1:]).item())
        else:
            p_tf = torch.zeros_like(x_in[:, 1:])
            p_tf[:, C - 1] = model.tf_next(x_in[:, :C])
        ro = model.rollout(x_in[:, :a.C], a.K)
        c = center(pos)
        mov = (~static) & (wrapd(c[:, C] - c[:, C - 1]).norm(dim=-1) > 0.3)
        q, mx = readout(p_tf[:, C - 1], o)
        acc["peak_tf"].append(mx[mov])                                   # 예측 안 물체다움의 최대 (또렷함)
        dtr = wrapd(c[:, C] - c[:, C - 1]); dp = wrapd(q - c[:, C - 1])
        acc["g_tf"].append(((dp * dtr).sum(-1) / dtr.pow(2).sum(-1).clamp_min(1e-6))[mov])
        al_o, al_n = alpha(pos[:, C - 1]), alpha(pos[:, C])
        pn, okn = presence_mask(p_tf[:, C - 1], o, al_n > 0.5)
        po, oko = presence_mask(p_tf[:, C - 1], o, (al_o > 0.5) & (al_n < 0.05))
        acc["pres_tf"].append(pn[mov & okn]); acc["pres_old_tf"].append(po[mov & oko])
        # 먼 배경: 문맥 · 다음 장 모두 물체와 안 겹치는 칸
        farm = (alpha(pos[:, :C + 1]).amax(1) < 1e-3)                    # (B,S)
        e_m = ((p_tf[:, C - 1] - clean[:, C]).abs().mean(-1) * farm).sum(1)
        e_c = ((x_in[:, C - 1] - clean[:, C]).abs().mean(-1) * farm).sum(1)
        e_a = ((x_in[:, :C].mean(1) - clean[:, C]).abs().mean(-1) * farm).sum(1)
        okf = (farm.sum(1) > 0) & (e_c > 0)
        if a.jit > 0:
            acc["bg_ratio"].append((e_m / e_c.clamp_min(1e-9))[okf]); acc["bg_ratio_mean"].append((e_a / e_c.clamp_min(1e-9))[okf])
        gr, pr = [], []
        for k in range(a.K):
            q, _ = readout(ro[:, k], o)
            dtr = wrapd(c[:, C + k] - c[:, C - 1]); dp = wrapd(q - c[:, C - 1])
            gr.append((dp * dtr).sum(-1) / dtr.pow(2).sum(-1).clamp_min(1e-6))
            pr.append(presence_mask(ro[:, k], o, alpha(pos[:, C + k]) > 0.5)[0])
        acc["g_ro"].append(torch.stack(gr, 1)[mov]); acc["pres_ro"].append(torch.stack(pr, 1)[mov])
        acc["pres_gt"].append(torch.stack([presence_mask(x_in[:, C + k], o, alpha(pos[:, C + k]) > 0.5)[0] for k in range(a.K)], 1)[mov])
        bl = baselines(x_in, x_tg, pos, static)
        for k_, v in bl.items():
            acc["b_" + k_].append(v)
        y = x_tg[:, C]
        acc["m_tf"].append(((p_tf[:, C - 1] - y).abs().mean((1, 2)) if a.loss == "l1" else ((p_tf[:, C - 1] - y) ** 2).mean((1, 2)))[mov])
    model.train()
    r = {}
    for k, v in acc.items():
        if not v:
            continue
        if isinstance(v[0], float):
            r[k] = float(np.mean(v)); continue
        t = torch.cat(v).float()
        if k.startswith(("b_", "m_")):
            r[k] = float(t.mean()); continue
        r[k] = (t.median(0).values.tolist() if t.dim() > 1 else float(t.median()))
    return r


hist = []
t0 = time.time()
r = evaluate(); r["step"] = 0; hist.append(r)
print(f"[{a.name}] 기준 (TF 한 걸음 손실, 움직인 clip): copy {r['b_copy']:.4f} mean_C {r['b_mean_C']:.4f} stay_dn {r['b_stay_dn']:.4f} oracle {r['b_oracle']:.4f}"
      f" → 평균 보상 {(r['b_copy'] - r['b_stay_dn']) / r['b_copy'] * 100:.1f} % · 운반 보상 {(r['b_stay_dn'] - r['b_oracle']) / r['b_copy'] * 100:.1f} % (copy 대비)", flush=True)
for it in range(1, a.steps + 1):
    x_in, x_tg, clean, pos, bg, o, static = batch(a.bs)
    if a.loss == "es":
        loss = es_m([model(x_in[:, :-1]) for _ in range(a.ns)], x_tg[:, 1:])
        if a.R > 0:
            loss = loss + es_m([model.rollout(x_in[:, :a.C], a.R, torch.randn(a.bs, a.nz, device=dev)) for _ in range(a.ns)], x_tg[:, a.C:a.C + a.R])
    elif a.loss == "crps" and a.ns > 2:
        loss = crps_m([model(x_in[:, :-1]) for _ in range(a.ns)], x_tg[:, 1:])
        if a.R > 0:
            loss = loss + crps_m([model.rollout(x_in[:, :a.C], a.R, torch.randn(a.bs, a.nz, device=dev)) for _ in range(a.ns)], x_tg[:, a.C:a.C + a.R])
    elif a.loss == "crps":
        loss = crps(model(x_in[:, :-1]), model(x_in[:, :-1]), x_tg[:, 1:])
        if a.R > 0:
            z1, z2 = torch.randn(a.bs, a.nz, device=dev), torch.randn(a.bs, a.nz, device=dev)
            r1, r2 = model.rollout(x_in[:, :a.C], a.R, z1), model.rollout(x_in[:, :a.C], a.R, z2)
            loss = loss + crps(r1, r2, x_tg[:, a.C:a.C + a.R])
    elif a.loss == "wta":
        model.hyp = -1
        P = model(x_in[:, :-1])                                          # (B, n, S, nh, D)
        e = (P - x_tg[:, 1:, :, None]).abs().mean((2, 4))                # (B, n, nh) 블록 전체 L1
        m = e.min(-1)
        loss = ((1 - a.wta_eps) * m.values + a.wta_eps * e.mean(-1)).mean()
        model.hyp = 0
    elif a.arch == "tf" and a.head == "disp" and a.disp_loss:
        model(x_in[:, :-1]); u = model._last_u                              # (B, T-1, S, 2) (dy, dx)
        al = alpha(pos)                                                       # (B, T, S)
        v = wrapd(pos[:, 1:] - pos[:, :-1])                                   # (B, T-1, 2) (x, y)
        ut = torch.stack([-v[..., 1], -v[..., 0]], -1)[:, :, None, :] * (al[:, 1:, :, None] > 0.5).float()
        if a.disp_loss == 1:
            loss = (u - ut).abs().mean()                                         # L1 (칸별 중앙값)
        elif a.disp_loss == 2:
            loss = (u - ut).pow(2).mean()                                        # L2 (칸별 평균)
        else:
            loss = F.smooth_l1_loss(u, ut, beta=0.1)                             # 3 = Huber: 0 근처만 L2 (기울기 막힘 방지), 크면 L1 (중앙값)
    elif a.arch == "tf":
        p = model(x_in[:, :-1])
        loss = lossfn(p, x_tg[:, 1:])
    else:
        ks = range(a.C, T)
        loss = sum(lossfn(model.tf_next(x_in[:, k - a.C:k]), x_tg[:, k]) for k in ks) / len(ks)
    if a.R > 0 and a.loss not in ("crps", "wta", "es") and not a.disp_loss:
        seq = x_in[:, :a.C]
        for rr in range(a.R):
            pr = model.tf_next(seq)[:, None]
            loss = loss + lossfn(pr, x_tg[:, a.C + rr:a.C + rr + 1]) / a.R
            seq = torch.cat([seq, pr], 1)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if it % a.eval_every == 0 or it == a.steps:
        r = evaluate(); r["step"] = it; r["loss"] = loss.item(); hist.append(r)
        print(f"[{a.name}] {it} loss {loss.item():.4f} tf1 {r['m_tf']:.4f} (stay_dn {r['b_stay_dn']:.4f} oracle {r['b_oracle']:.4f}) g_tf {r['g_tf']:.2f} "
              f"peak {r['peak_tf']:.2f} pres_new {r['pres_tf']:.2f} pres_old {r.get('pres_old_tf', float('nan')):.2f} bg_ratio {r.get('bg_ratio', float('nan')):.2f} "
              f"(mean_C {r.get('bg_ratio_mean', float('nan')):.2f}) g_ro {[round(v, 2) for v in r['g_ro']]} pres_ro {[round(v, 2) for v in r['pres_ro']]} "
              f"({time.time() - t0:.0f}s)", flush=True)
branch = {}
if a.branch_eval:
    model.eval()
    torch.manual_seed(4321)
    with torch.no_grad():
        PG, PS, NS = [], [], (16 if a.loss in ("crps", "es") else (a.nh if a.loss == "wta" else 1))
        for _ in range(4):
            x_in, x_tg, clean, pos, bg, o, static = batch(128)
            C = a.C
            vlast = pos[:, C - 1] - pos[:, C - 2]
            go_pos, st_pos = pos[:, C - 1] + vlast, pos[:, C - 1]
            ag, ast = alpha(go_pos), alpha(st_pos)
            mg, ms = (ag > 0.5) & (ast < 0.05), (ast > 0.5) & (ag < 0.05)          # 서로 안 겹치는 칸만
            ok = (~static) & (mg.sum(1) > 0) & (ms.sum(1) > 0)
            pg_s, ps_s = [], []
            for k in range(NS):
                if a.loss == "wta":
                    model.hyp = k
                p1 = model.tf_next(x_in[:, :C])
                pg_s.append(presence_mask(p1, o, mg)[0]); ps_s.append(presence_mask(p1, o, ms)[0])
            PG.append(torch.stack(pg_s, 1)[ok]); PS.append(torch.stack(ps_s, 1)[ok])
            model.hyp = 0
        PG, PS = torch.cat(PG).cpu().numpy(), torch.cat(PS).cpu().numpy()          # (n, NS)
    hg = np.histogram(PG.ravel(), bins=10, range=(-0.2, 1.0))[0].tolist()
    go = (PG > 0.6) & (PS < 0.3); st = (PS > 0.6) & (PG < 0.3); both = (PG > 0.6) & (PS > 0.6); none = (PG < 0.3) & (PS < 0.3)
    branch = dict(n_ctx=int(PG.shape[0]), n_samples=int(PG.shape[1]), pres_go_mean=float(PG.mean()), pres_stay_mean=float(PS.mean()),
                  pres_go_of_sample_mean=float(PG.mean(1).mean()), frac_go=float(go.mean()), frac_stay=float(st.mean()), frac_both=float(both.mean()),
                  frac_none=float(none.mean()), frac_mixed=float(1 - go.mean() - st.mean() - both.mean() - none.mean()), hist_pres_go=hg,
                  corr_go_stay=float(np.corrcoef(PG.ravel(), PS.ravel())[0, 1]),
                  frac_ctx_both_branches=float((go.any(1) & st.any(1)).mean()), frac_ctx_go_any=float(go.any(1).mean()), frac_ctx_stay_any=float(st.any(1).mean()))
    print(f"[{a.name}] 갈래 평가 (p_stop {a.p_stop}, 문맥 {branch['n_ctx']} × 표본 {NS}): 간다 자리 진하기 평균 {branch['pres_go_mean']:.2f} · 멈춘 자리 {branch['pres_stay_mean']:.2f} | "
          f"표본별: 간다만 {branch['frac_go']:.2f} · 멈춤만 {branch['frac_stay']:.2f} · 둘 다 {branch['frac_both']:.2f} · 둘 다 없음 {branch['frac_none']:.2f} · 섞임 {branch['frac_mixed']:.2f} · "
          f"상관 {branch['corr_go_stay']:.2f} · 문맥 중 두 갈래 모두 진하게 그린 표본 있음 {branch['frac_ctx_both_branches']:.2f} · 간다 자리 진하기 분포 {hg}", flush=True)
if a.loss == "crps" and a.xi != "none":
    sv = model.log_s.exp().detach().cpu().numpy()
    branch["xi_scale_mean"] = float(sv.mean()); branch["xi_scale_std"] = float(sv.std())
    print(f"[{a.name}] 학습된 ξ 척도 s: 평균 {sv.mean():.3f} (표준편차 {sv.std():.3f}) — 참 토큰 잡음 표준편차 {a.jit_tgt}", flush=True)
json.dump({"args": vars(a), "hist": hist, "branch": branch}, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
print(f"[{a.name}] done", flush=True)
