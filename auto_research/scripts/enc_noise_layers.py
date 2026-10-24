#!/usr/bin/env python3
"""고리 1 [E] — 과민성은 encoder 의 어디서 생기나: 층별 섭동 증폭 · 고노름 토큰 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_noise_layers.py --n 48

블록 (2 장) = enc_noise_collect.py 가 저장한 픽셀 (k710 · vobench · phys3d) 의 블록 0. fp32 (TF32 끔).
  층 l (0 = patch 임베딩, 1..32 = transformer 블록 출력, 33 = 마지막 norm 뒤, 34 = 블록 AR 정답 공간 = affine 없는 LN 한 번 더)
  - 잡음 증폭: 토큰별 ‖h_l(x+n) − h_l(x)‖ / ‖h_l(x)‖  (n = ±1/255 정수 잡음)
  - 장면 척도: 토큰별 ‖h_l(x_i) − h_l(x_j)‖ / ‖h_l(x_i)‖  (다른 창 같은 칸)
  - 비율 = 잡음 / 장면 — 층을 지나며 커지면 "깊이가 미세 섭동을 증폭" (무작위 초기화 encoder 와 대조)
  고노름 토큰 (Darcet et al. 2024): 마지막 transformer 블록 출력 (norm 전) 의 토큰 노름 분포 · 그 칸의 잡음 민감도 · 칸 무늬 (픽셀 표준편차) 와의 상관."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--n", type=int, default=48, help="소스마다 블록 수")
ap.add_argument("--sources", nargs="+", default=["k710", "vobench", "phys3d"])
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/enc_noise_layers.json"))
a = ap.parse_args()
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(8)
dev = torch.device("cuda"); G = 16
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")


def build(name, SD):
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if name != "random":
        _load_state_dict(e, SD[name], tag="t")
    return e.to(dev).eval()


@torch.no_grad()
def layers(e, px):                    # px (B,2,256,256,3) uint8 → list of (B,256,D) per layer
    x = (px.float().div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    H = []
    hooks = [e.patch_embed.register_forward_hook(lambda m, i, o: H.append(o.float()))]
    for blk in e.blocks:
        hooks.append(blk.register_forward_hook(lambda m, i, o: H.append(o.float())))
    out = e(x).float()
    for h in hooks:
        h.remove()
    H.append(out); H.append(F.layer_norm(out, (out.size(-1),)))
    return H


def blocks_of(src):
    fs = sorted(glob.glob(os.path.join(a.dir, f"{src}_target_encoder_s*_pix.npy")))
    P = np.concatenate([np.asarray(np.load(f, mmap_mode="r")[:, :2]) for f in fs])     # 블록 0
    idx = np.random.default_rng(0).choice(len(P), min(a.n, len(P)), replace=False)
    return P[idx]


SD = _load_checkpoint(CK)
res = {}
for name in ("target_encoder", "encoder", "random"):
    e = build(name, SD)
    res[name] = {}
    for src in a.sources:
        P = blocks_of(src)
        g = torch.Generator(device=dev).manual_seed(0)
        noise, scene, norms, sens, tex = [], [], [], [], []
        prev = None
        for i in range(0, len(P), 4):
            px = torch.from_numpy(P[i:i + 4]).to(dev)
            n = torch.randint(-1, 2, px.shape, device=dev, generator=g)
            H0 = layers(e, px); H1 = layers(e, (px.int() + n).clamp(0, 255).to(torch.uint8))
            noise.append(torch.stack([((b - c).norm(dim=-1) / c.norm(dim=-1)).mean(-1) for b, c in zip(H1, H0)], 1).cpu())   # (B,L)
            # 장면 척도: 배치 안에서 한 칸씩 돌려 다른 창
            Hr = [h.roll(1, 0) for h in H0]
            scene.append(torch.stack([((b - c).norm(dim=-1) / c.norm(dim=-1)).mean(-1) for b, c in zip(Hr, H0)], 1).cpu())
            last = H0[32]                                                     # 마지막 transformer 블록 출력 (norm 전)
            norms.append(last.norm(dim=-1).cpu())                             # (B,256)
            sens.append((H1[34] - H0[34]).abs().mean(-1).cpu())               # 정답 공간 칸별 잡음 |Δh|
            g_ = px.float().mean(-1)[:, 0].view(-1, G, 16, G, 16)
            tex.append(g_.std(dim=(2, 4)).reshape(-1, G * G).cpu())
        noise = torch.cat(noise).mean(0).numpy(); scene = torch.cat(scene).mean(0).numpy()
        norms = torch.cat(norms).numpy(); sens = torch.cat(sens).numpy(); tex = torch.cat(tex).numpy()
        nz = norms.reshape(-1); med = np.median(nz)
        hi = norms > 3 * med
        r = {"noise_rel": noise.tolist(), "scene_rel": scene.tolist(), "ratio": (noise / scene).tolist(),
             "norm_median": float(med), "norm_p99": float(np.percentile(nz, 99)), "norm_max_over_median": float(nz.max() / med),
             "frac_high_norm_3x": float(hi.mean()),
             "sens_high_norm": float(sens[hi].mean()) if hi.any() else None, "sens_rest": float(sens[~hi].mean()),
             "corr_norm_sens": float(np.corrcoef(nz, sens.reshape(-1))[0, 1]),
             "corr_tex_sens": float(np.corrcoef(tex.reshape(-1), sens.reshape(-1))[0, 1]),
             "corr_tex_norm": float(np.corrcoef(tex.reshape(-1), nz)[0, 1]),
             "sens_flat": float(sens[tex < 2].mean()) if (tex < 2).any() else None, "sens_textured": float(sens[tex >= 8].mean()) if (tex >= 8).any() else None,
             "frac_flat": float((tex < 2).mean())}
        res[name][src] = r
        L = len(noise)
        pick = [0, 1, 4, 8, 16, 24, 31, 32, 33, 34]
        print(f"== {name} / {src}: 잡음/장면 비율 층 " + " ".join(f"{l}:{noise[l] / scene[l]:.3f}" for l in pick if l < L))
        print(f"     잡음 상대 크기 " + " ".join(f"{l}:{noise[l]:.4f}" for l in pick if l < L))
        print(f"     노름 중앙 {med:.1f} · p99 {r['norm_p99']:.1f} · 최대/중앙 {r['norm_max_over_median']:.1f} · 3배 넘는 토큰 {r['frac_high_norm_3x']:.2%} | "
              f"잡음 민감도 고노름 {r['sens_high_norm']} vs 나머지 {r['sens_rest']:.3f} | 상관(노름,민감) {r['corr_norm_sens']:.2f} (무늬,민감) {r['corr_tex_sens']:.2f} (무늬,노름) {r['corr_tex_norm']:.2f} | "
              f"민무늬 칸 {r['frac_flat']:.0%} 민감 {r['sens_flat']} vs 무늬 {r['sens_textured']}", flush=True)
    del e; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
