#!/usr/bin/env python3
"""H-E2 개입 실험: 블록 (2 장) 의 한 칸만 바꾸면 latent 가 어디까지 바뀌나 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore auto_research/scripts/encoder_locality.py --n 48

캐시된 phys3d val 픽셀에서 블록 하나를 고르고 (블록 0, 두 장),
  [noise]  모든 픽셀에 ±1/255 균등 잡음 (8 bit 양자화 한 단계, 눈에 안 보임) → 칸별 |Δh|
  [patch]  물체에서 먼 칸 c0 하나의 16×16 픽셀을 같은 영상의 다른 칸 내용으로 바꿈 → c0 로부터 체비셰프 거리 r 별 |Δh|
  [shift]  물체 bbox 영역을 오른쪽으로 16 px 옮겨 붙임 (= 물체 한 칸 이동, 배경은 그대로) → 물체 칸 / 고리 / 먼 칸 |Δh| 몫
encoder: target (V-JEPA 2 ViT-H target_encoder) · random (같은 구조 무작위 초기화). 출력은 블록별 LN (예측 정답과 같은 공간).
기준 척도: 서로 다른 clip 같은 칸 토큰 사이 |Δh| (전혀 다른 내용) — 비율로 읽는다."""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/encoder_locality.json"))
a = ap.parse_args()
dev = torch.device("cuda"); G = 16
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
encs = {}
for name in ("target", "random"):
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if name == "target":
        _load_state_dict(e, _load_checkpoint(CK)["target_encoder"], tag="t")
    encs[name] = e.to(dev, torch.bfloat16).eval()


def enc(e, px):                     # px (B,2,256,256,3) float 0..1 → (B,256,1280) LN
    x = px.permute(0, 4, 1, 2, 3)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = e(((x - MEAN) / STD).to(torch.bfloat16)).float()
    return F.layer_norm(h, (h.size(-1),))


P = np.load(os.path.join(a.cache, "pix_val_s0.npy"), mmap_mode="r")
L = np.load(os.path.join(a.cache, "lab_val_s0.npz"))
rng = np.random.RandomState(0)
sel = [w for w in range(P.shape[0]) if L["vis_frac"][w, 0] > 0.7 and L["vis_frac"][w, 1] > 0.7 and np.isfinite(L["bbox"][w, :2]).all()][: a.n]
cheb = lambda c0: np.maximum(np.abs(np.arange(G)[:, None] - c0[0]), np.abs(np.arange(G)[None, :] - c0[1])).reshape(-1)
R = {k: {"noise": [], "patch": {r: [] for r in range(16)}, "shift": [], "other_clip": []} for k in encs}
for i, w in enumerate(sel):
    px = torch.from_numpy(P[w][:2].astype(np.float32) / 255).to(dev)[None]          # (1,2,256,256,3)
    bb = L["bbox"][w, :2].mean(0)
    x0, y0, x1, y1 = [int(round(v)) for v in bb]
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, 256), min(y1, 256)
    ocell = np.zeros((G, G), bool); ocell[y0 // 16:(y1 - 1) // 16 + 1, x0 // 16:(x1 - 1) // 16 + 1] = True
    # 물체에서 가장 먼 칸 c0, 바꿀 내용은 c0 에서 4 칸 떨어진 칸 (물체 아닌 곳)
    yy, xx = np.mgrid[:G, :G]
    dist = np.where(ocell, 0, 99)
    oy, ox = np.argwhere(ocell).mean(0)
    far = np.hypot(yy - oy, xx - ox); c0 = np.unravel_index(np.argmax(far), far.shape)
    src = ((c0[0] + 4) % G, (c0[1] + 4) % G)
    noise = px + (torch.randint(-1, 2, px.shape, device=dev).float() / 255)
    patch = px.clone(); patch[:, :, c0[0] * 16:(c0[0] + 1) * 16, c0[1] * 16:(c0[1] + 1) * 16] = px[:, :, src[0] * 16:(src[0] + 1) * 16, src[1] * 16:(src[1] + 1) * 16]
    shift = px.clone()
    if x1 + 16 <= 256 and x1 > x0 and y1 > y0:
        bgfill = px[:, :, y0:y1, x0:x1].clone()
        # 물체 자리를 같은 행의 물체 없는 배경으로 메우기 어렵다 → 물체를 그대로 두고 16 px 옆에 한 번 더 붙이지 않고, 영역 전체를 옮긴다
        shift[:, :, y0:y1, x0 + 16:x1 + 16] = bgfill
        shift[:, :, y0:y1, x0:x0 + 16] = px[:, :, y0:y1, max(x0 - 16, 0):max(x0 - 16, 0) + 16] if x0 >= 16 else shift[:, :, y0:y1, x0:x0 + 16]
    for k, e in encs.items():
        h = enc(e, torch.cat([px, noise.clamp(0, 1), patch, shift]))
        d = (h[1:] - h[:1]).abs().mean(-1).cpu().numpy()                              # (3,256)
        R[k]["noise"].append(float(d[0].mean()))
        ch = cheb(c0)
        for r in range(16):
            if (ch == r).any():
                R[k]["patch"][r].append(float(d[1][ch == r].mean()))
        oc = np.zeros((G, G), bool); oc[y0 // 16:(y1 - 1) // 16 + 1, x0 // 16:min((x1 + 15) // 16 + 1, G)] = True
        ring = np.zeros_like(oc)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ring |= np.roll(np.roll(oc, dy, 0), dx, 1)
        ring &= ~oc; farm = ~(oc | ring)
        tot = d[2].sum()
        R[k]["shift"].append(dict(obj=float(d[2].reshape(G, G)[oc].sum() / tot), ring=float(d[2].reshape(G, G)[ring].sum() / tot),
                                  far=float(d[2].reshape(G, G)[farm].sum() / tot), area_obj=float(oc.mean()),
                                  obj_mean=float(d[2].reshape(G, G)[oc].mean()), far_mean=float(d[2].reshape(G, G)[farm].mean())))
        if i > 0:
            R[k]["other_clip"].append(float((h[0] - prev[k]).abs().mean().item()))
    prev = {k: enc(e, px)[0] for k, e in encs.items()}
out = {}
for k in encs:
    sc = np.median(R[k]["other_clip"])
    pr = {r: float(np.median(v)) for r, v in R[k]["patch"].items() if v}
    sh = {kk: float(np.median([x[kk] for x in R[k]["shift"]])) for kk in R[k]["shift"][0]}
    out[k] = dict(scale_other_clip=float(sc), noise=float(np.median(R[k]["noise"])), patch_by_r=pr, shift=sh)
    print(f"== {k}  (기준: 다른 clip 같은 칸 |Δh| {sc:.3f})")
    print(f"  ±1/255 잡음 → 칸 평균 |Δh| {out[k]['noise']:.4f} ({out[k]['noise'] / sc:.1%} of 기준)")
    print("  한 칸 바꿈 → 거리 r 별 |Δh| / 기준:", " ".join(f"r{r}:{v / sc:.3f}" for r, v in pr.items() if r <= 8))
    print(f"  물체 한 칸 이동 → L1 질량 몫 물체 {sh['obj']:.3f} (넓이 {sh['area_obj']:.3f}) 고리 {sh['ring']:.3f} 먼 칸 {sh['far']:.3f} | 칸당 물체 {sh['obj_mean']/sc:.3f} 먼 칸 {sh['far_mean']/sc:.3f} (기준 비)")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
