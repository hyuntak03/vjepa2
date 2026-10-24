#!/usr/bin/env python3
"""encoder 과민성이 입력 장 수에 따라 달라지나 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore auto_research/scripts/encoder_sensitivity_frames.py --n 24

캐시 phys3d val 의 32 장 (간격 3) 에서 앞 T 장을 한 번에 인코딩 (T = 2, 4, 8, 16, 32). 토큰 = (T/2 tubelet × 256 칸).
  ind1  : 모든 장에 서로 다른 ±1/255 잡음 → 같은 자리 토큰 |Δh| / 기준 (다른 clip 같은 자리)
  bf16  : 같은 입력의 bf16 vs fp32 인코딩 차이 / 기준
  temporal : 원본 T 장 인코딩에서 이웃 tubelet 사이 |Δh| (시간 변화) / 기준 — 물체가 한 번도 안 지나간 칸만
LN 은 토큰별 affine-free (블록 AR 의 정답과 같은 처리). fp32 로 계산 (bf16 행 제외)."""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--Ts", type=int, nargs="+", default=[2, 4, 8, 16, 32])
ap.add_argument("--enc", default="target_encoder", help="target_encoder | encoder")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/encoder_sensitivity_frames.json"))
a = ap.parse_args()
dev = torch.device("cuda"); G = 16
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=64, use_rope=True, uniform_power=False)
_load_state_dict(e, _load_checkpoint(CK)[a.enc], tag="t"); e = e.to(dev).eval()


def enc(px, bf16=False):            # px (B,T,256,256,3)
    x = ((px.permute(0, 4, 1, 2, 3) - MEAN) / STD)
    with torch.no_grad():
        if bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = e(x).float()
        else:
            h = e(x).float()
    return F.layer_norm(h, (h.size(-1),))


P = np.load(os.path.join(a.cache, "pix_val_s2.npy"), mmap_mode="r")
L = np.load(os.path.join(a.cache, "lab_val_s2.npz"))
sel = list(range(0, P.shape[0], max(1, P.shape[0] // a.n)))[: a.n]
g = torch.Generator(device=dev).manual_seed(0)
out = {}
for T in a.Ts:
    R = {"ind1": [], "bf16": [], "temporal": []}; scale = []; prev = None
    for w in sel:
        px = torch.from_numpy(P[w][:T].astype(np.float32) / 255).to(dev)[None]
        n1 = torch.randint(-1, 2, px.shape, device=dev, generator=g).float() / 255
        h0 = enc(px); h1 = enc((px + n1).clamp(0, 1)); hb = enc(px, bf16=True)
        R["ind1"].append(float((h1 - h0).abs().mean())); R["bf16"].append(float((hb - h0).abs().mean()))
        if T >= 4:
            ht = h0[0].view(T // 2, G * G, -1)
            bb = L["bbox"][w][:T]
            touched = np.zeros(G * G, bool)
            for f in range(T):
                if np.isfinite(bb[f]).all():
                    x0, y0, x1, y1 = bb[f]
                    for y in range(max(0, int(y0 // 16) - 1), min(G, int(y1 // 16) + 2)):
                        for x in range(max(0, int(x0 // 16) - 1), min(G, int(x1 // 16) + 2)):
                            touched[y * G + x] = True
            m = torch.from_numpy(~touched).to(dev)
            if m.any():
                R["temporal"].append(float((ht[1:] - ht[:-1]).abs().mean(-1)[:, m].mean()))
        if prev is not None and prev.shape == h0.shape:
            scale.append(float((h0 - prev).abs().mean()))
        prev = h0
    sc = float(np.median(scale))
    out[T] = {"scale": sc} | {k: (float(np.median(v)) if v else None) for k, v in R.items()}
    print(f"T={T:2d}  기준 {sc:.3f} | ±1/255 잡음 {out[T]['ind1'] / sc:6.1%} | bf16 vs fp32 {out[T]['bf16'] / sc:6.1%} | "
          f"이웃 tubelet 시간 변화 (물체 없는 칸) {('%.1f%%' % (100 * out[T]['temporal'] / sc)) if out[T]['temporal'] else '—'}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
