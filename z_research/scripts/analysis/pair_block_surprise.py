#!/usr/bin/env python3
"""한 가능/불가능 쌍에서 미래 블록마다 놀람을 나눠 본다 — 블록 AR (2026-10-21, 사용자 질문: "그림에선 떨어지는데 왜 떠 있는 쪽을 고르나").

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/pair_block_surprise.py \
      --pos realistic/ledge/z300_v320/v11_apple/pos --imp realistic/ledge/z300_v320/v11_apple/imp_float \
      --run natural_ar_vw_state latest.pt --run natural_ar_vw_state_phys3d_event e54.pt

채점과 같은 입력 (원본 0, 3, …, 93 장 → 블록 16, 문맥 블록 8 → 미래 블록 8 rollout). 문맥 16 장은 두 영상이 같아 예측 p 는 하나다.
미래 블록 b 마다:
  s_pos / s_imp   = mean |p − h|  (h = 그 영상의 블록별 LN target_encoder, 256 칸 · 1280 평균) — 채점 점수는 이 8 개의 평균
  diff            = s_pos − s_imp (양수면 그 블록이 imp 를 고르게 민다)
  갈린 칸 / 나머지 = 두 실제 미래가 가장 다른 칸 (|h_pos − h_imp| 상위 10 %) 과 나머지 칸으로 나눈 diff
  |h_pos − h_imp|  = 두 실제 미래가 그 블록에서 얼마나 다른가"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/benchmarks/VOBench_v1/gravity")
ap.add_argument("--pos", required=True)
ap.add_argument("--imp", required=True)
ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--top", type=float, default=0.10)
ap.add_argument("--decoder", default=None, help="pixel decoder 체크포인트 — 주면 예측을 픽셀로 그려 pos · imp 원본과 픽셀 L1 도 비교")
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
T, C, S = 32, 16, 256
IDX = [3 * k for k in range(T)]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev).eval()


RAW = {}


def blocks(rel):
    d = os.path.join(a.root, rel)
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    RAW[rel] = x[:, C:].permute(1, 0, 2, 3)                                                # 미래 16 장 (16, 3, H, W) [0, 1]
    x = (x - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(x.view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))                     # (16, 256, D)
    return LN(h.float())


hp, hi = blocks(a.pos), blocks(a.imp)
dec = None
if a.decoder:
    sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
    from model import PixelDecoder                                                      # noqa: E402
    _st = torch.load(a.decoder, map_location="cpu", weights_only=False); _mc = _st["cfg"]["model"]
    dec = PixelDecoder(in_dim=1280, dim=_mc["dim"], depth=_mc["depth"], heads=_mc["heads"]).to(dev).eval(); dec.load_state_dict(_st["decoder"])


def decode(tok):                                                                         # (8, 256, D) → (16, 3, H, W) [0, 1]
    with torch.no_grad():
        y = dec(tok.float())
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256)
ctx_same = float((hp[: C // 2] - hi[: C // 2]).abs().max())
fp, fi = hp[C // 2:], hi[C // 2:]                                                        # 미래 블록 8
gap = (fp - fi).abs().mean(-1)                                                           # (8, 256) 두 실제 미래의 차이
k = int(round(a.top * S))
print(f"pos {a.pos}\nimp {a.imp}\n문맥 블록 최대 차이 {ctx_same:.2e} (0 이면 문맥이 같다)")
print("두 실제 미래의 블록별 차이 |h_pos − h_imp| (256 칸 평균): " + " ".join(f"{v:.3f}" for v in gap.mean(-1).tolist()))
for run, ck in a.run:
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    pr = vit_prefix_predictor(**kw)
    sd = {k_: v for k_, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
          if not k_.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    pr.load_state_dict(sd, strict=True); pr = pr.to(dev).eval()
    idx = torch.arange(C // 2 * S, device=dev).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = pr.rollout_kv(hp[: C // 2].reshape(1, -1, hp.size(-1)), idx, (T - C) // 2).float()[0].view(8, S, -1)
    ep, ei = (p - fp).abs().mean(-1), (p - fi).abs().mean(-1)                             # (8, 256)
    print(f"\n[{run}:{ck}]  전체 s_pos {ep.mean():.4f} · s_imp {ei.mean():.4f} → {'pos ✓' if ep.mean() < ei.mean() else 'imp ✗'}")
    print(f"  {'블록':4s} {'s_pos':>7s} {'s_imp':>7s} {'diff':>8s} | {'갈린 칸 diff':>11s} {'나머지 diff':>11s} | {'갈린 칸 s_pos/s_imp':>18s}")
    for b in range(8):
        top = torch.zeros(S, dtype=torch.bool, device=dev); top[gap[b].argsort(descending=True)[:k]] = True
        d = ep[b] - ei[b]
        print(f"  {b + 1:4d} {ep[b].mean():7.4f} {ei[b].mean():7.4f} {d.mean():+8.4f} | {d[top].mean():+11.4f} {d[~top].mean():+11.4f} |"
              f"   {ep[b][top].mean():.3f} / {ei[b][top].mean():.3f}")
    dd = (ep - ei)
    print(f"  합 diff {dd.mean():+.4f} = 갈린 칸 몫 {(dd * 0 + dd).flatten()[torch.cat([gap[b].argsort(descending=True)[:k] + b * S for b in range(8)])].sum() / dd.numel():+.4f}"
          f" + 나머지 몫 {(dd.sum() - dd.flatten()[torch.cat([gap[b].argsort(descending=True)[:k] + b * S for b in range(8)])].sum()) / dd.numel():+.4f}")
    if dec is not None:
        rp, ri = RAW[a.pos], RAW[a.imp]
        yp = decode(p)
        gm = ((rp - ri).abs().mean(1) > 0.08)                                            # 두 원본이 다른 픽셀 (물체 · 그림자)
        print(f"  픽셀 (decoder 로 그린 예측 vs 원본, 미래 16 장)   두 원본이 다른 픽셀 비율 {gm.float().mean():.3f}")
        print(f"  {'블록':4s} {'L1 pos':>7s} {'L1 imp':>7s} {'diff':>8s} | {'다른 픽셀만 pos/imp':>18s}")
        for b in range(8):
            sl = slice(2 * b, 2 * b + 2)
            lp, li = (yp[sl] - rp[sl]).abs().mean(), (yp[sl] - ri[sl]).abs().mean()
            m = gm[sl].unsqueeze(1).expand(-1, 3, -1, -1)
            mp = (yp[sl] - rp[sl]).abs()[m].mean() if m.any() else torch.tensor(float("nan"))
            mi = (yp[sl] - ri[sl]).abs()[m].mean() if m.any() else torch.tensor(float("nan"))
            print(f"  {b + 1:4d} {lp:7.4f} {li:7.4f} {lp - li:+8.4f} |   {mp:.3f} / {mi:.3f}")
        LP, LI = (yp - rp).abs().mean(), (yp - ri).abs().mean()
        m = gm.unsqueeze(1).expand(-1, 3, -1, -1)
        print(f"  전체 픽셀 L1 pos {LP:.4f} · imp {LI:.4f} → {'pos ✓' if LP < LI else 'imp ✗'}   |  다른 픽셀만 pos {(yp - rp).abs()[m].mean():.3f} · imp {(yp - ri).abs()[m].mean():.3f} → {'pos ✓' if (yp - rp).abs()[m].mean() < (yp - ri).abs()[m].mean() else 'imp ✗'}")
