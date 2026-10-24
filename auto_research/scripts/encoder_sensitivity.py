#!/usr/bin/env python3
"""encoder_locality.py 의 "±1/255 잡음 → 토큰 64 % 변화" 를 반증 시도 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore auto_research/scripts/encoder_sensitivity.py --n 32

블록 = 캐시 phys3d val 의 장 (0, 1). 변형마다 원본 블록 대비 칸 평균 |Δh| (블록별 LN 공간) / 기준 (다른 clip 같은 칸).
  ind1      두 장에 서로 다른 ±1/255 균등 정수 잡음                (= 8 bit 한 단계 깜빡임)
  same1     두 장에 같은 ±1/255 잡음                               (공간 잡음, 시간 변화 없음)
  ind_q     두 장에 서로 다른 N(0, (0.25/255)²) 잡음                (한 단계보다 작은 깜빡임)
  dup0      (장0, 장0) — 완전 정지 블록 vs 원본 (장0, 장1)
  flick     장1 만 전체 밝기 +1/255                                 (매끄러운 시간 변화)
  bright    두 장 모두 밝기 +1/255
  fp32      ind1 을 fp32 로 (bf16 수치 문제인지)
encoder: target · random. 각 변형은 같은 블록의 원본 인코딩 (같은 dtype) 과 비교."""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/encoder_sensitivity.json"))
a = ap.parse_args()
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")


def build(name):
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if name == "target":
        _load_state_dict(e, _load_checkpoint(CK)["target_encoder"], tag="t")
    return e.to(dev).eval()


def enc(e, px, fp32=False):         # px (B,2,256,256,3) 0..1
    x = ((px.permute(0, 4, 1, 2, 3) - MEAN) / STD)
    with torch.no_grad():
        if fp32:
            h = e.float()(x.float()).float()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = e(x).float()
    return F.layer_norm(h, (h.size(-1),))


P = np.load(os.path.join(a.cache, "pix_val_s1.npy"), mmap_mode="r")
sel = list(range(0, P.shape[0], max(1, P.shape[0] // a.n)))[: a.n]
g = torch.Generator(device=dev).manual_seed(0)
out = {}
for name in ("target", "random"):
    e = build(name)
    R = {k: [] for k in ("ind1", "same1", "ind_q", "dup0", "flick", "bright", "fp32", "fp32_vs_bf16_clean")}
    prev, scale = None, []
    for w in sel:
        px = torch.from_numpy(P[w][:2].astype(np.float32) / 255).to(dev)[None]
        n1 = torch.randint(-1, 2, px.shape, device=dev, generator=g).float() / 255
        ns = torch.randint(-1, 2, (1, 1) + px.shape[2:], device=dev, generator=g).float().expand_as(px) / 255
        nq = torch.randn(px.shape, device=dev, generator=g) * 0.25 / 255
        fl = px.clone(); fl[:, 1] += 1 / 255
        V = {"ind1": px + n1, "same1": px + ns, "ind_q": px + nq, "dup0": px[:, :1].expand(-1, 2, -1, -1, -1), "flick": fl, "bright": px + 1 / 255}
        h0 = enc(e, px)
        hv = enc(e, torch.cat([v.clamp(0, 1) for v in V.values()]))
        for i, k in enumerate(V):
            R[k].append(float((hv[i] - h0[0]).abs().mean()))
        h0f = enc(e, px, fp32=True); h1f = enc(e, (px + n1).clamp(0, 1), fp32=True)
        R["fp32"].append(float((h1f - h0f).abs().mean())); R["fp32_vs_bf16_clean"].append(float((h0f - h0).abs().mean()))
        if prev is not None:
            scale.append(float((h0 - prev).abs().mean()))
        prev = h0
    sc = float(np.median(scale))
    out[name] = {"scale_other_clip": sc} | {k: float(np.median(v)) for k, v in R.items()}
    print(f"== {name}  기준 (다른 clip) {sc:.3f}")
    for k in R:
        print(f"  {k:20s} |Δh| {out[name][k]:.4f}  = {out[name][k] / sc:6.1%} of 기준")
    del e; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
