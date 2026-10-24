#!/usr/bin/env python3
"""장난감: 토큰 격자 위 움직이는 물체 + 정지 배경. 블록 AR 과 같은 형태의 predictor 를 L1 등으로 학습하고
이론 (분리 손실 → 칸별 중앙값 → 점유 확률 문턱) 의 예측과 비교한다 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python auto_research/scripts/toy_occupancy.py --name det_l1 --sig_a 0 --loss l1

세계 (G×G 토러스, 토큰 D 차원):
  배경 b(c) ~ N(0, I) — clip 마다 새로, 시간에 따라 고정 (= 느린 특징). 장마다 잡음 jit · N(0, I).
  물체: 한 변 w 칸 정사각형, 정체 o ~ N(0, I) (clip 마다). 칸 c 의 토큰 = (1-α_c) b(c) + α_c (contrast · o),  α_c = 칸과 정사각형의 겹친 넓이.
  운동: v_{t+1} = v_t + sig_a · N(0, I) (무작위 가속 = 진짜 불확실성), x_{t+1} = x_t + v_{t+1}. 확률 p0 로 clip 내내 정지.
predictor: 장 단위 block-causal transformer (블록 AR 과 같은 형태). TF 전 위치 + rollout R 걸음 손실.
평가: 운반 이득 (TF 한 걸음 · rollout), 물체 존재감 (참 위치 칸에서 예측 토큰과 o 의 cos), 그리고 **같은 생성 과정에서 표본 256 개로 계산한
Bayes 최적 예측** (L1 → 칸 · 차원별 중앙값, L2 → 평균) 과의 거리."""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/toy"))
ap.add_argument("--G", type=int, default=16)
ap.add_argument("--D", type=int, default=32)
ap.add_argument("--w", type=float, default=2.0)
ap.add_argument("--contrast", type=float, default=1.0)
ap.add_argument("--jit", type=float, default=0.1)
ap.add_argument("--sig_a", type=float, default=0.0)
ap.add_argument("--p0", type=float, default=0.0)
ap.add_argument("--vmin", type=float, default=0.5)
ap.add_argument("--vmax", type=float, default=1.5)
ap.add_argument("--C", type=int, default=4)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--R", type=int, default=2, help="rollout 손실 걸음")
ap.add_argument("--loss", default="l1", choices=["l1", "l2", "wl1", "mh_l1"])
ap.add_argument("--lam", type=float, default=4.0, help="wl1: 정답 변화 칸 가중 배수")
ap.add_argument("--nh", type=int, default=4, help="mh_l1: 가설 수 (승자 독식)")
ap.add_argument("--dim", type=int, default=192)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
torch.manual_seed(a.seed); np.random.seed(a.seed)
G, D, T = a.G, a.D, a.C + a.K
S = G * G
cell = torch.arange(G, device=dev).float()


def ov1(x, w):                     # (...,) 왼쪽 끝 x, 길이 w → (..., G) 각 칸 [c, c+1) 과 겹친 길이 (토러스)
    x = x[..., None] % G
    o = torch.zeros(*x.shape[:-1], G, device=dev)
    for sh in (-G, 0, G):
        o += (torch.minimum(cell + 1, x + sh + w) - torch.maximum(cell, x + sh)).clamp_min(0)
    return o.clamp(max=1)


def alpha(pos):                    # pos (..., 2) → (..., S)
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
    for _ in range(T_ - 1):
        v = v + a.sig_a * torch.randn_like(v) * (~static[:, None]).float()
        x = x + v
        xs.append(x)
    return torch.stack(xs, 1), static                     # (B, T_, 2)


def render(pos, bg, o, jit=True):  # pos (B,T,2), bg (B,S,D), o (B,D) → (B,T,S,D)
    al = alpha(pos)[..., None]                            # (B,T,S,1)
    tok = (1 - al) * bg[:, None] + al * (a.contrast * o)[:, None, None]
    if jit and a.jit > 0:
        tok = tok + a.jit * torch.randn_like(tok)
    return tok


def batch(B):
    pos, static = sample_motion(B, T)
    bg = torch.randn(B, S, D, device=dev)
    o = torch.randn(B, D, device=dev)
    return render(pos, bg, o), pos, bg, o, static


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj, s.h = nn.Linear(d, 3 * d), nn.Linear(d, d), h
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        B, N, d = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, N, d))
        return x + s.mlp(s.n2(x))


class Pred(nn.Module):
    """입력 장 0..n-1 → 장 1..n 예측. 장 단위 block-causal. mh_l1 이면 출력 nh 개."""
    def __init__(s):
        super().__init__()
        nh = a.nh if a.loss == "mh_l1" else 1
        s.inp = nn.Linear(D, a.dim)
        s.pos_t = nn.Parameter(torch.zeros(T, 1, a.dim)); s.pos_s = nn.Parameter(torch.zeros(1, S, a.dim))
        nn.init.normal_(s.pos_t, std=0.02); nn.init.normal_(s.pos_s, std=0.02)
        s.blocks = nn.ModuleList([Block(a.dim, 6) for _ in range(a.depth)])
        s.out = nn.Sequential(nn.LayerNorm(a.dim), nn.Linear(a.dim, D * nh)); s.nh = nh

    def forward(s, x):              # x (B,n,S,D) → (B,n,S,nh,D)
        B, n = x.shape[:2]
        z = (s.inp(x) + s.pos_t[:n] + s.pos_s).view(B, n * S, -1)
        fi = torch.arange(n, device=dev).repeat_interleave(S)
        mask = fi[:, None] >= fi[None, :]
        for b in s.blocks:
            z = b(z, mask)
        return s.out(z).view(B, n, S, s.nh, D)

    def rollout(s, ctx, k, hyp=0):  # 가설 hyp 를 되먹임
        seq, outs = ctx, []
        for _ in range(k):
            p = s(seq)[:, -1, :, hyp]
            outs.append(p); seq = torch.cat([seq, p[:, None]], 1)
        return torch.stack(outs, 1)


def lossfn(p, y, prev):            # p (B,n,S,nh,D), y (B,n,S,D), prev (B,n,S,D) = 직전 장 정답 (가중용)
    if a.loss == "l1":
        return (p[..., 0, :] - y).abs().mean()
    if a.loss == "l2":
        return ((p[..., 0, :] - y) ** 2).mean()
    if a.loss == "wl1":            # 정답이 크게 바뀐 칸 (≈ 픽셀 변화) 에 λ 배 — 정답 의존 가중
        chg = (y - prev).abs().mean(-1, keepdim=True)
        w = 1 + (a.lam - 1) * (chg > 0.5 * chg.amax(dim=2, keepdim=True)).float()
        return ((p[..., 0, :] - y).abs() * w).sum() / (w.sum() * D)
    e = (p - y[..., None, :]).abs().mean(-1)            # mh: 칸마다가 아니라 장 전체로 승자 (B,n,nh)
    e = e.mean(2)
    return e.min(-1).values.mean()


model = Pred().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def readout(tok, o):                # tok (B,S,D) → 위치 (B,2) (3×3 cos 가중 중심, 토러스 무시: 경계 근처는 평가에서 뺌), 최대 cos
    cs = F.cosine_similarity(tok, (a.contrast * o)[:, None], dim=-1)          # (B,S)
    j = cs.argmax(1); jy, jx = j // G, j % G
    offs = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], device=dev)
    yy, xx = (jy[:, None] + offs[:, 0]) % G, (jx[:, None] + offs[:, 1]) % G
    w = cs.gather(1, yy * G + xx).clamp_min(0) + 1e-6
    cy = jy.float() + (offs[:, 0].float() * w).sum(1) / w.sum(1)
    cx = jx.float() + (offs[:, 1].float() * w).sum(1) / w.sum(1)
    return torch.stack([cx, cy], -1) + 0.5, cs.amax(1)


def center(pos):
    return pos + a.w / 2


def wrapd(d):
    return (d + G / 2) % G - G / 2


@torch.no_grad()
def presence(tok, pos, o):          # 참 위치 발자국 칸 (α>0.5) 에서 예측 토큰과 o 의 평균 cos
    al = alpha(pos)
    cs = F.cosine_similarity(tok, (a.contrast * o)[:, None], dim=-1)
    m = (al > 0.5).float()
    return (cs * m).sum(1) / m.sum(1).clamp_min(1)


@torch.no_grad()
def bayes(pos, bg, o, static, kind, n=256):
    """문맥 끝 (장 C-1) 상태에서 같은 과정으로 미래를 n 번 뽑아 렌더 → 칸 · 차원별 중앙값 (l1) / 평균 (l2). (B,K,S,D)"""
    B = pos.shape[0]
    x0 = pos[:, a.C - 1]; v0 = pos[:, a.C - 1] - pos[:, a.C - 2]
    outs = []
    for i in range(B):
        P_, _ = sample_motion(n, a.K + 1, x0[i:i + 1].expand(n, 2), v0[i:i + 1].expand(n, 2), static[i:i + 1].expand(n))
        tok = render(P_[:, 1:], bg[i:i + 1].expand(n, S, D), o[i:i + 1].expand(n, D), jit=False)   # (n,K,S,D)
        outs.append(tok.median(0).values if kind == "l1" else tok.mean(0))
    return torch.stack(outs)


@torch.no_grad()
def evaluate(nb=8, with_bayes=True):
    model.eval()
    g = torch.Generator(device=dev); torch.manual_seed(1234)
    acc = {k: [] for k in ("g_tf", "g_ro", "pres_tf", "pres_ro", "pres_gt", "pres_bayes", "g_bayes", "maxcos_ro", "dist_bayes", "dist_copy", "loss_tf")}
    for _ in range(nb):
        x, pos, bg, o, static = batch(64)
        p_tf = model(x[:, :-1])[..., 0, :]                      # (B,T-1,S,D) → 장 1..T-1
        ro = model.rollout(x[:, :a.C], a.K)                     # (B,K,S,D) → 장 C..T-1
        acc["loss_tf"].append((p_tf - x[:, 1:]).abs().mean().item())
        c = center(pos)
        mov = (~static) & (wrapd(c[:, a.C] - c[:, a.C - 1]).norm(dim=-1) > 0.3)
        # TF 한 걸음: 장 C-1 → C
        q, _ = readout(p_tf[:, a.C - 1], o)
        dtr = wrapd(c[:, a.C] - c[:, a.C - 1]); dp = wrapd(q - c[:, a.C - 1])
        acc["g_tf"].append(((dp * dtr).sum(-1) / dtr.pow(2).sum(-1).clamp_min(1e-6))[mov])
        acc["pres_tf"].append(presence(p_tf[:, a.C - 1], pos[:, a.C], o)[mov])
        gr, pr, mc = [], [], []
        for k in range(a.K):
            q, m_ = readout(ro[:, k], o)
            dtr = wrapd(c[:, a.C + k] - c[:, a.C - 1]); dp = wrapd(q - c[:, a.C - 1])
            gr.append((dp * dtr).sum(-1) / dtr.pow(2).sum(-1).clamp_min(1e-6)); pr.append(presence(ro[:, k], pos[:, a.C + k], o)); mc.append(m_)
        acc["g_ro"].append(torch.stack(gr, 1)[mov]); acc["pres_ro"].append(torch.stack(pr, 1)[mov]); acc["maxcos_ro"].append(torch.stack(mc, 1)[mov])
        acc["pres_gt"].append(torch.stack([presence(x[:, a.C + k], pos[:, a.C + k], o) for k in range(a.K)], 1)[mov])
        if with_bayes:
            kind = "l2" if a.loss == "l2" else "l1"
            bz = bayes(pos[:16], bg[:16], o[:16], static[:16], kind)
            m16 = mov[:16]
            acc["pres_bayes"].append(torch.stack([presence(bz[:, k], pos[:16, a.C + k], o[:16]) for k in range(a.K)], 1)[m16])
            gb = []
            for k in range(a.K):
                q, _ = readout(bz[:, k], o[:16]); dtr = wrapd(c[:16, a.C + k] - c[:16, a.C - 1]); dp = wrapd(q - c[:16, a.C - 1])
                gb.append((dp * dtr).sum(-1) / dtr.pow(2).sum(-1).clamp_min(1e-6))
            acc["g_bayes"].append(torch.stack(gb, 1)[m16])
            acc["dist_bayes"].append((ro[:16] - bz).abs().mean((2, 3))[m16])
            acc["dist_copy"].append((x[:16, a.C - 1:a.C] - bz).abs().mean((2, 3))[m16])
    model.train()
    r = {}
    for k, v in acc.items():
        if not v:
            continue
        if isinstance(v[0], float):
            r[k] = float(np.mean(v)); continue
        t = torch.cat(v).float()
        r[k] = (t.median(0).values.tolist() if t.dim() > 1 else float(t.median()))
    return r


hist = []
t0 = time.time()
for it in range(1, a.steps + 1):
    x, pos, bg, o, static = batch(a.bs)
    p = model(x[:, :-1])
    loss = lossfn(p, x[:, 1:], x[:, :-1])
    if a.R > 0:                                         # rollout 손실 (블록 AR 과 같은 형태, 첫 가설 되먹임)
        seq = x[:, :a.C]
        for r in range(a.R):
            pr = model(seq)[:, -1:]
            loss = loss + lossfn(pr, x[:, a.C + r:a.C + r + 1], x[:, a.C + r - 1:a.C + r]) / a.R
            seq = torch.cat([seq, pr[:, :, :, 0]], 1)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if it % a.eval_every == 0 or it == a.steps:
        r = evaluate(nb=4, with_bayes=(it == a.steps))
        r["step"] = it; r["loss"] = loss.item(); hist.append(r)
        print(f"[{a.name}] {it} loss {loss.item():.4f} g_tf {r['g_tf']:.2f} g_ro {[round(v, 2) for v in r['g_ro']]} "
              f"pres_ro {[round(v, 2) for v in r['pres_ro']]} pres_gt {[round(v, 2) for v in r['pres_gt']]} ({time.time() - t0:.0f}s)", flush=True)
fin = hist[-1]
print(f"[{a.name}] FINAL bayes: pres {[round(v, 2) for v in fin.get('pres_bayes', [])]} g {[round(v, 2) for v in fin.get('g_bayes', [])]} "
      f"| dist(model, bayes) {[round(v, 3) for v in fin.get('dist_bayes', [])]} dist(copy, bayes) {[round(v, 3) for v in fin.get('dist_copy', [])]}")
json.dump({"args": vars(a), "hist": hist}, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
