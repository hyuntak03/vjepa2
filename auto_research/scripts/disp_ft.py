#!/usr/bin/env python3
"""[T] 방법 B — 변위 공간 손실 (실제 latent): 기준 블록 AR 에 변위 머리 u · 도착 마스크 머리 m 을 붙여 이어 학습 (auto_research, 2026-10-21).
z_training · app · src 는 수정하지 않는다 (import · 래퍼만).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/disp_ft.py --name disp_sg --steps 2000 [--lr 1e-4] [--m_sg 0]

예측 (TF 위치 b → 블록 b+1, 칸 c):
  u(c) ∈ R² (칸 단위) = "이 칸 내용이 블록 b 의 어디서 왔나" (역방향, 블록 b 의 위치 = c + u)   — 마지막 층 은닉 (384) → 선형 2
  m(c) ∈ [0, 1]       = "움직이는 내용이 이 칸에 들어오나"                                         — 마지막 층 은닉 → 선형 1 → 시그모이드
  최종 = m ⊙ warp(h_b, u) + (1 − m) ⊙ p     (warp = 블록 b 토큰 격자를 c + u 에서 쌍선형 표본, p = 원래 토큰 예측)
손실 = 토큰 L1 (최종 vs 정답) + λ_u · Huber_δ(u − u_flow) [앞-뒤 일관 칸만] + λ_m · BCE(m, m_flow).
  u_flow = 블록 b+1 → b 역방향 RAFT 흐름의 칸 평균. 앞-뒤 일관 = |U(c) + V(c + U(c))| < 0.25 칸. m_flow = 1[|U(c)| ≥ 0.5 칸].
  --m_sg 1: 토큰 L1 의 기울기를 m 에서 끊는다 → "얼마나 진하게" 를 토큰 손실이 정하지 못하고 BCE (흐름) 만 정한다.
  ⚠️ (다) 를 온전히 지키려면 u 도 끊어야 한다 (--u_sg 1) — 아니면 토큰 L1 이 옮겨 붙이기를 거쳐 u 를 당긴다. 주 팔 = --comp 0 (학습 손실에서 옮겨 붙이기 없음).
데이터: 자연 영상 학습 캐시 (/workspace/auto_research_cache/natural_train, 블록 8 개 창) + 칸 흐름 (_Uflow 역방향 · _Vflow 순방향).
저장: /workspace/auto_research_cache/ar_ft/<name>.pt (predictor · u_head · m_head)."""
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--lam_u", type=float, default=1.0)
ap.add_argument("--lam_m", type=float, default=0.1)
ap.add_argument("--delta", type=float, default=0.25)
ap.add_argument("--m_sg", type=int, default=1)
ap.add_argument("--u_sg", type=int, default=0, help="1 = 옮겨 붙이기에서 u 도 끊는다 (토큰 L1 이 u 를 못 당김)")
ap.add_argument("--comp", type=int, default=1, help="0 = (다) 주 팔: 학습 손실 안에서 옮겨 붙이지 않는다 — 토큰 L1 은 p 에만, u · m 은 자기 공간 손실만")
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--out", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--seed", type=int, default=0)
a, _ = ap.parse_known_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
G, S = 16, 256


class DispWrap(nn.Module):
    def __init__(s, pred):
        super().__init__()
        s.pred = pred
        s.u_head = nn.Linear(384, 2); s.m_head = nn.Linear(384, 1)
        nn.init.zeros_(s.u_head.weight); nn.init.zeros_(s.u_head.bias)
        nn.init.zeros_(s.m_head.weight); nn.init.constant_(s.m_head.bias, -2.0)      # 처음엔 m ≈ 0.12 (거의 원래 예측)
        s.nz, s.codes = 32, None

    def heads(s, x):                                                       # x (B, nb*S, 1280) → p (B,nb,S,D), u (B,nb,S,2), m (B,nb,S,1)
        B = x.size(0); nb = x.size(1) // S
        idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, st = s.pred.forward_seq(x, idx, return_states=True)
        with torch.autocast("cuda", enabled=False):
            u = s.u_head(st.float()).view(B, nb, S, 2); m = torch.sigmoid(s.m_head(st.float())).view(B, nb, S, 1)
        return p.float().view(B, nb, S, -1), u, m

    @staticmethod
    def warp(hb, u):                                                       # hb (B,nb,S,D) 블록 b 토큰, u (B,nb,S,2) → c + u 에서 쌍선형 표본
        B, nb, _, D = hb.shape
        g = hb.reshape(B * nb, G, G, D).permute(0, 3, 1, 2)
        yy, xx = torch.meshgrid(torch.arange(G, device=hb.device), torch.arange(G, device=hb.device), indexing="ij")
        cx = xx.reshape(-1).float() + u[..., 0].reshape(B * nb, S); cy = yy.reshape(-1).float() + u[..., 1].reshape(B * nb, S)
        grid = torch.stack([(cx + 0.5) / G * 2 - 1, (cy + 0.5) / G * 2 - 1], -1).view(B * nb, G, G, 2)
        out = F.grid_sample(g.float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=False)
        return out.permute(0, 2, 3, 1).reshape(B, nb, S, D)

    def final(s, x, m_sg=False, u_sg=False):
        B = x.size(0); nb = x.size(1) // S
        p, u, m = s.heads(x)
        with torch.autocast("cuda", enabled=False):
            w = s.warp(x.float().view(B, nb, S, -1), u.detach() if u_sg else u)
            mm = m.detach() if m_sg else m
            f = mm * w + (1 - mm) * p
        return f, p, u, m

    def tf(s, x, z=None, code=None):                                       # 판정 스크립트용: 최종 예측 (B, nb*S, D)
        f, _, _, _ = s.final(x)
        return f.reshape(x.size(0), -1, f.size(-1))


if __name__ == "__main__":
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    W = DispWrap(FT.load_pred("natural_ar_vw_state", "latest.pt")).to(dev)
    W.pred.train()
    hfiles = sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))
    srcs = [np.load(f, mmap_mode="r") for f in hfiles]
    Uf = [np.load(f.replace("_h.npy", "_Uflow.npy"), mmap_mode="r") for f in hfiles]
    Vf = [np.load(f.replace("_h.npy", "_Vflow.npy"), mmap_mode="r") for f in hfiles]
    sizes = [x.shape[0] for x in srcs]
    opt = torch.optim.AdamW([{"params": W.pred.parameters(), "lr": a.lr},
                             {"params": list(W.u_head.parameters()) + list(W.m_head.parameters()), "lr": a.lr_head}], weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    t0 = time.time()
    for it in range(1, a.steps + 1):
        si = rng.integers(len(srcs), size=a.bs); wi = [rng.integers(sizes[s]) for s in si]
        x = torch.from_numpy(np.stack([srcs[s][w] for s, w in zip(si, wi)])).to(dev).float()            # (B,8,S,D)
        U = torch.from_numpy(np.stack([np.asarray(Uf[s][w], np.float32) for s, w in zip(si, wi)])).to(dev).view(a.bs, 8, S, 2)
        V = torch.from_numpy(np.stack([np.asarray(Vf[s][w], np.float32) for s, w in zip(si, wi)])).to(dev).view(a.bs, 8, S, 2)
        u_t = U[:, 1:]                                                     # 위치 b (0..6) 의 표적 = 블록 b+1 → b 역방향 흐름
        # 앞-뒤 일관: 블록 b 의 c + u 자리 순방향 흐름 V[b] 를 가장 가까운 칸에서
        cxs = (torch.from_numpy(xx.reshape(-1)).to(dev)[None, None] + u_t[..., 0]).round().clamp(0, G - 1).long()
        cys = (torch.from_numpy(yy.reshape(-1)).to(dev)[None, None] + u_t[..., 1]).round().clamp(0, G - 1).long()
        vj = torch.gather(V[:, :7], 2, (cys * G + cxs)[..., None].expand(-1, -1, -1, 2))
        cons = ((u_t + vj).norm(dim=-1) < 0.25).float()                   # (B,7,S)
        mflow = (u_t.norm(dim=-1) >= 0.5).float()
        xin = x.reshape(a.bs, -1, x.size(-1))
        f, p, u, m = W.final(xin, m_sg=bool(a.m_sg), u_sg=bool(a.u_sg))
        f, u, m = f[:, :7], u[:, :7], m[:, :7, :, 0]
        y = x[:, 1:]
        l_tok = ((f if a.comp else p[:, :7]) - y).abs().mean()
        hub = F.huber_loss(u, u_t, reduction="none", delta=a.delta).sum(-1)
        l_u = (hub * cons).sum() / cons.sum().clamp_min(1)
        l_m = F.binary_cross_entropy(m.float().clamp(1e-5, 1 - 1e-5), mflow)
        loss = l_tok + a.lam_u * l_u + a.lam_m * l_m
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == 1:
            uerr = float(((u - u_t).norm(dim=-1) * cons).sum() / cons.sum().clamp_min(1))
            print(f"[{a.name}] {it} 토큰 L1 {float(l_tok):.4f} · u 오차 (일관 칸, 칸) {uerr:.3f} · m 평균 움직임/정지 칸 "
                  f"{float((m * mflow).sum() / mflow.sum().clamp_min(1)):.2f}/{float((m * (1 - mflow)).sum() / (1 - mflow).sum().clamp_min(1)):.2f} "
                  f"· BCE {float(l_m):.3f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"predictor": W.pred.state_dict(), "u_head": W.u_head.state_dict(), "m_head": W.m_head.state_dict(), "args": vars(a), "kind": "disp"},
               os.path.join(a.out, f"{a.name}.pt"))
    print("saved", flush=True)
