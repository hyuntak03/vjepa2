#!/usr/bin/env python3
"""[T] 방법 개입 (실제 latent, 짧게): 기준 블록 AR 을 자연 영상 블록 토큰 위에서 이어 학습한다 — 공정 CRPS + 블록 공유 저차원 잡음 입력 vs 대조군 L1 (auto_research, 2026-10-21).
z_training · app · src 는 수정하지 않는다 (import 만). 잡음 입력은 이 파일 안의 래퍼로 붙인다.

  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_ft.py --loss crps --name crps --steps 3000
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_ft.py --loss l1   --name l1   --steps 3000

데이터: /workspace/auto_research_cache/natural_train/{ssv2,k710_320,ht100m_1pct}_h.npy (N, 8, 256, 1280) fp16 — 블록별 LN(target_encoder) bf16, 4 fps, 16 장 창.
손실: TF 전 위치 (블록 0..b 로 블록 b+1 예측, b = 0..6). L1 = 칸 · 차원 평균 |p − y| (기준 학습과 같은 꼴).
      공정 CRPS = ½(|p1 − y| + |p2 − y|) − ½|p1 − p2| (p1, p2 = 서로 다른 잡음 z 의 표본). p1 = p2 이면 L1 과 같다.
잡음 입력: z ~ N(0, I_nz) 창마다 하나 → 선형 (nz → 384) → 관측 임베딩 뒤 **모든 토큰에 같은 벡터** 를 더함 (블록 · 칸 공유).
저장: /workspace/auto_research_cache/ar_ft/<name>.pt (predictor + 잡음 선형)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn as nn, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="natural_ar_vw_state")
ap.add_argument("--ckpt", default="latest.pt")
ap.add_argument("--loss", choices=["l1", "crps", "wta", "wl1", "mov"], required=True)
ap.add_argument("--lam", type=float, default=6.0, help="wl1: 문맥 흐름 ≥ 0.5 칸 (1 칸 팽창) 칸의 가중 (나머지 1, 위치마다 평균 1 로 정규화). mov = 그 칸만 손실")
ap.add_argument("--nh", type=int, default=3, help="wta: 가설 수 (가설마다 학습되는 입력 부호 벡터 하나)")
ap.add_argument("--wta_eps", type=float, default=0.05)
ap.add_argument("--pos_ctrl", type=int, default=0, help="양성 대조: 창마다 확률 ½ 로 정답 미래를 '멈춘 세계' (마지막 문맥 블록 복사) 로 바꾼다 → 같은 문맥에 두 미래")
ap.add_argument("--xi_fix", type=int, default=0, help="1 = ξ 척도를 초기값에 고정")
ap.add_argument("--s_file", default="", help="ξ 척도 (차원별, 1280) 파일. [E] 의 정지 칸 블록 간 차 /√2 의 차원별 표준편차 (nugget) — 자연 3 소스 평균")
ap.add_argument("--name", required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--lr_noise", type=float, default=1e-3)
ap.add_argument("--nz", type=int, default=32)
ap.add_argument("--noise_init", type=float, default=0.05)
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--out", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--xi", default="laplace", choices=["none", "gauss", "laplace"], help="crps: 표본 = g(H, z) + s ⊙ ξ, ξ 좌표마다 독립 (블록 고유 성분 담당), s 차원별 학습")
ap.add_argument("--s_init", type=float, default=-1.0, help="ξ 척도 초기값. 음수면 캐시에서 차원별 std(h_{b+1} − h_b)/√2 (블록 고유 성분의 상한)")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                          # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                     # noqa: E402
dev = torch.device("cuda")
torch.set_num_threads(8)
torch.manual_seed(a.seed); np.random.seed(a.seed)
S = 256


def load_pred(run, ck):
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k_) is not None:
            kw[k_] = pc[k_]
    m = vit_prefix_predictor(**kw)
    w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
         if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m


class NoiseWrap(nn.Module):
    """predictor 의 관측 임베딩 뒤에 창마다 하나의 잡음 벡터를 모든 토큰에 더한다 (원본 메서드를 감싸기만)."""
    def __init__(s, pred, nz, init):
        super().__init__()
        s.pred, s.nz = pred, nz
        s.proj = nn.Linear(nz, 384)
        nn.init.normal_(s.proj.weight, std=init / np.sqrt(nz)); nn.init.zeros_(s.proj.bias)
        s._zv = None
        s.log_s = nn.Parameter(torch.full((1280,), float(np.log(abs(a.s_init)))))
        s.codes = nn.Parameter(torch.randn(a.nh, 384) * init) if a.loss == "wta" else None
        orig = pred._ar_embed_obs

        def patched(x):
            h = orig(x)
            return h if s._zv is None else h + s._zv[:, None, :].to(h.dtype)
        pred._ar_embed_obs = patched

    def tf(s, x, z=None, code=None):           # x (B, nb*S, 1280) → (B, nb*S, 1280), 위치 b = 블록 b+1 예측
        s._zv = s.codes[code][None].expand(x.size(0), -1) if code is not None else (None if z is None else s.proj(z))
        idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        out = s.pred.forward_seq(x, idx)
        s._zv = None
        return out


if __name__ == "__main__":
    W = NoiseWrap(load_pred(a.run, a.ckpt), a.nz, a.noise_init).to(dev)
    W.pred.train()
    hfiles = sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))
    srcs = [np.load(f, mmap_mode="r") for f in hfiles]
    flows = [np.load(f.replace("_h.npy", "_Uflow.npy"), mmap_mode="r") for f in hfiles] if a.loss in ("wl1", "mov") else None
    sizes = [x.shape[0] for x in srcs]
    if a.loss == "crps" and a.xi != "none" and a.s_file:
        s0 = np.load(a.s_file)
        with torch.no_grad():
            W.log_s.copy_(torch.from_numpy(np.log(s0)).to(dev))
        print(f"ξ 척도 (파일 {a.s_file}): 평균 {s0.mean():.3f} · 범위 {s0.min():.3f}–{s0.max():.3f} · 고정 {bool(a.xi_fix)}", flush=True)
    elif a.loss == "crps" and a.xi != "none" and a.s_init < 0:
        dd = np.concatenate([np.asarray(x[:100, 1:], np.float32) - np.asarray(x[:100, :-1], np.float32) for x in srcs]).reshape(-1, 1280)
        s0 = dd.std(0) / np.sqrt(2)
        with torch.no_grad():
            W.log_s.copy_(torch.from_numpy(np.log(s0)).to(dev))
        print(f"ξ 척도 초기값 (차원별 std(h_(b+1) − h_b)/√2): 평균 {s0.mean():.3f} · 범위 {s0.min():.3f}–{s0.max():.3f}", flush=True)
    print(f"캐시 {[os.path.basename(f) for f in sorted(glob.glob(os.path.join(a.cache, '*_h.npy')))]} 창 {sizes}", flush=True)
    params = [{"params": W.pred.parameters(), "lr": a.lr}]
    if a.loss == "crps":
        params.append({"params": list(W.proj.parameters()) + ([] if a.xi_fix else [W.log_s]), "lr": a.lr_noise})
    if a.loss == "wta":
        params.append({"params": [W.codes], "lr": a.lr_noise})
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    t0 = time.time(); log = []
    for it in range(1, a.steps + 1):
        si = rng.integers(len(srcs), size=a.bs)
        wi = [rng.integers(sizes[s]) for s in si]
        xb = np.stack([srcs[s][w] for s, w in zip(si, wi)])                      # (B, 8, 256, 1280)
        if flows is not None:                                                    # 문맥만으로 정한 가중: 위치 b 의 문맥 흐름 U[b] (블록 b → b−1)
            ub = torch.from_numpy(np.stack([np.asarray(flows[s][w], np.float32) for s, w in zip(si, wi)])).to(dev)   # (B,8,16,16,2)
            mm = (ub.norm(dim=-1) >= 0.5).float()[:, :7]                         # 위치 0..6 → 블록 0..6 의 흐름 (블록 0 은 0 → 가중 1)
            mm = torch.nn.functional.max_pool2d(mm, 3, 1, 1).reshape(a.bs, 7, S, 1)            # 1 칸 팽창
            wgt = (1 + (a.lam - 1) * mm) if a.loss == "wl1" else mm
            wgt = wgt / wgt.mean(2, keepdim=True).clamp_min(1e-6)               # 위치마다 평균 1
        x = torch.from_numpy(xb).to(dev).float().reshape(a.bs, -1, 1280)
        y = x[:, S:]
        if a.pos_ctrl:                                                      # 위치 b 의 정답: 실제 h_{b+1} 또는 멈춘 세계 h_b (창마다 동전)
            coin = torch.from_numpy(rng.random(a.bs) < 0.5).to(dev)[:, None, None]
            y = torch.where(coin, x[:, S:], x[:, :-S])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.loss == "l1":
                p = W.tf(x)[:, :-S].float()
                loss = (p - y).abs().mean(); spread = 0.0
            elif a.loss in ("wl1", "mov"):
                p = W.tf(x)[:, :-S].float()
                loss = ((p - y).abs().view(a.bs, 7, S, -1) * wgt).mean(); spread = float(mm.mean())   # 움직임 칸 비율 기록
            elif a.loss == "wta":
                E = torch.stack([(W.tf(x, code=k)[:, :-S].float() - y).abs().view(a.bs, 7, S, -1).mean((2, 3)) for k in range(a.nh)], -1)   # (B, 7, nh)
                Ew = E.mean(1)                                            # (B, nh): 창 전체 (TF 7 걸음) 로 승자를 고른다 — 가설이 걸음마다 갈래를 바꾸지 않게
                loss = ((1 - a.wta_eps) * Ew.min(-1).values + a.wta_eps * Ew.mean(-1)).mean()
                win = torch.bincount(Ew.argmin(-1), minlength=a.nh).float()
                spread = float(win.max() / win.sum())                    # 가장 많이 이긴 가설의 몫 (1 = 붕괴)
            else:
                z1, z2 = torch.randn(a.bs, a.nz, device=dev), torch.randn(a.bs, a.nz, device=dev)
                g1 = W.tf(x, z1)[:, :-S].float(); g2 = W.tf(x, z2)[:, :-S].float()
                spread_g = float((g1 - g2).abs().mean())
                if a.xi != "none":
                    sc = W.log_s.exp()
                    if a.xi == "gauss":
                        p1, p2 = g1 + sc * torch.randn_like(g1), g2 + sc * torch.randn_like(g2)
                    else:
                        lap = torch.distributions.Laplace(0.0, 1.0)
                        p1 = g1 + sc * lap.sample(g1.shape).to(dev) / np.sqrt(2); p2 = g2 + sc * lap.sample(g2.shape).to(dev) / np.sqrt(2)
                else:
                    p1, p2 = g1, g2
                e1, e2, sp = (p1 - y).abs().mean(), (p2 - y).abs().mean(), (p1 - p2).abs().mean()
                loss = 0.5 * (e1 + e2) - 0.5 * sp; spread = spread_g
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == 1:
            sm = float(W.log_s.exp().mean()) if a.loss == "crps" else float("nan")
            rec = dict(step=it, loss=float(loss), spread_g=spread, s_mean=sm, sec=time.time() - t0)
            log.append(rec)
            print(f"[{a.name}] {it} loss {float(loss):.4f} g 표본 퍼짐 |g1−g2| {spread:.4f} · ξ 척도 평균 {sm:.3f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"predictor": W.pred.state_dict(), "noise_proj": W.proj.state_dict(), "log_s": W.log_s.detach().cpu(),
                "codes": None if W.codes is None else W.codes.detach().cpu(), "args": vars(a), "log": log}, os.path.join(a.out, f"{a.name}.pt"))
    print("saved", os.path.join(a.out, f"{a.name}.pt"), flush=True)
