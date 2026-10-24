#!/usr/bin/env python3
"""고리 1 → 2 개입 [E]: 정지 칸 픽셀을 창 내내 얼려 encoder 입력의 미세 변화를 없애면, 손실이 주는 "평균 보상" 이 사라지나 (auto_research, 2026-10-21).

  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_noise_freeze_window.py --shard $k; done
  /venv/vjepa/bin/python auto_research/scripts/noise_averaging.py --cache /workspace/auto_research_cache/phys3d_freeze --enc frozen

phys3d 캐시 창 (32 장, 블록 16) 마다:
  frozen    칸마다 기준 픽셀을 들고 가며, 블록 b 의 그 칸 (두 장) 이 기준과 칸 평균 |Δ| < thr (0–255) 이면 기준 픽셀을 쓰고, 아니면 새 픽셀로 바꾸고 기준을 갱신.
            → 물체가 지나가는 칸은 그대로, 나머지 칸은 픽셀이 블록 사이에 정확히 같다 (encoder 입력의 미세 변화 제거)
  변형 4 개: orig bf16 (= 기존 캐시와 같은 처리) · orig fp32 · frozen bf16 · frozen fp32  → h_val_<tag>_s<k>.npy (N,16,256,1280) fp16
라벨 · 메타는 원래 캐시로 심볼릭 링크."""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--out", default="/workspace/auto_research_cache/phys3d_freeze")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--thr", type=float, default=2.0)
ap.add_argument("--variants", nargs="+", default=["orig32", "frozen", "frozen32"])
a = ap.parse_args()
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(8)
dev = torch.device("cuda"); G, R = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
SD = _load_checkpoint(CK)["target_encoder"]


def build():
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(e, SD, tag="t")
    return e.eval()


E32 = build().to(dev); E16 = build().to(dev, torch.bfloat16)
del SD


@torch.no_grad()
def encode(px, mode):                 # px (nb,2,R,R,3) uint8
    x = (px.float().div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    out = []
    for i in range(0, x.size(0), 8):
        xi = x[i:i + 8]
        if mode == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = E16(xi.to(torch.bfloat16))
        else:
            h = E32(xi.float())
        out.append(F.layer_norm(h.float(), (h.size(-1),)))
    return torch.cat(out)


def freeze(fr):                        # fr (32,R,R,3) uint8 → 얼린 것 · 얼린 칸 비율
    bl = fr.view(-1, 2, R, R, 3).float()
    ref = bl[0].clone(); out = [bl[0]]; frac = []
    for b in range(1, bl.size(0)):
        d = (bl[b] - ref).abs().mean(-1).mean(0)                         # (R,R)
        dc = d.view(G, 16, G, 16).mean(dim=(1, 3))                       # (G,G)
        keep = (dc < a.thr).view(1, G, 1, G, 1).expand(1, G, 16, G, 16).reshape(1, R, R, 1)
        cur = torch.where(keep, ref, bl[b])
        ref = cur
        out.append(cur); frac.append(float((dc < a.thr).float().mean()))
    return torch.stack(out).to(torch.uint8), float(np.mean(frac))


os.makedirs(a.out, exist_ok=True)
for f in ("lab_val", "meta_val"):
    src = os.path.join(a.cache, f"{f}_s{a.shard}." + ("npz" if f == "lab_val" else "json"))
    dst = os.path.join(a.out, os.path.basename(src))
    if not os.path.exists(dst):
        os.symlink(src, dst)
P = np.load(os.path.join(a.cache, f"pix_val_s{a.shard}.npy"), mmap_mode="r")
OUT = {v: [] for v in a.variants}; FR = []
for w in range(P.shape[0]):
    fr = torch.from_numpy(np.ascontiguousarray(P[w])).to(dev)
    fz, frac = freeze(fr); FR.append(frac)
    for v in a.variants:
        x = (fz if v.startswith("frozen") else fr.view(-1, 2, R, R, 3))
        OUT[v].append(encode(x, "fp32" if v.endswith("32") else "bf16").half().cpu().numpy())
    if w % 20 == 0:
        print(f"[s{a.shard}] {w}/{P.shape[0]} 얼린 칸 {frac:.2f}", flush=True)
for v, L in OUT.items():
    np.save(os.path.join(a.out, f"h_val_{v}_s{a.shard}.npy"), np.stack(L))
np.save(os.path.join(a.out, f"frozen_frac_s{a.shard}.npy"), np.array(FR))
print(f"[s{a.shard}] done, 얼린 칸 평균 {np.mean(FR):.3f}", flush=True)
