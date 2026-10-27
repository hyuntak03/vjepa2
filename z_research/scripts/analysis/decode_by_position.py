"""pixel decoder 가 물체 위치에 따라 못 그리나 — 정답 토큰 · 굴린 예측 토큰을 그려 물체 상자 PSNR 을 위치 구역별로 (2026-10-26, 학습 없음).
사용자: "decoder 가 학습 때 못 본 위치가 주어지면 못 그리는 거 아닌가?"

decoder: z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both (VOBench · gravity · IntPhys1 · IntPhys2 로 학습, phys3d 는 안 봄; 블록 · 창 latent 둘 다).
val 350 편 (ar_layers 캐시), 블록 8–15 중 물체가 보이는 블록. 블록 토큰 (256, 1280) → decoder → 2 장 → 원본 두 장 (12 fps 원본 6b · 6b+3) 과 비교.
  gt      : encoder 정답 블록 토큰
  rollout : 굴린 예측 (문맥 블록 0–7 → 걸음 b−7)
지표: 물체 상자 안 PSNR · 화면 전체 PSNR. 물체 중심 위치를 4 × 4 구역 (64 px) 으로 묶어 구역별 평균.

  CUDA_VISIBLE_DEVICES=0 MEM_GB=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/decode_by_position.py
출력: z_analysis/results/decode_by_position.json
"""
import json, os, sys
import numpy as np, torch

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
from model import PixelDecoder  # noqa: E402
import decord
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 3)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
C = "/workspace/auto_research_cache/ar_layers"; IDX = [3 * k for k in range(32)]
st = torch.load(f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt", map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"]); del st
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
meta = json.load(open(f"{C}/val_meta.json"))
psnr = lambda a, b: float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))
rows = []
for m in meta:
    z = np.load(f"{C}/val/{m['i']:05d}.npz")
    fr = decord.VideoReader(m["clip"], num_threads=2).get_batch(IDX).asnumpy()
    real = torch.from_numpy(fr).to(dev).float().permute(3, 0, 1, 2) / 255.0                  # (3, 32, H, W)
    bl = [b for b in range(8, 16) if m["blocks"][b]["ok"] and m["blocks"][b]["cells"]]
    if not bl:
        continue
    X = torch.from_numpy(np.stack([z["enc"][b] for b in bl] + [z["ro_out"][b - 8] for b in bl])).to(dev).float()
    with torch.no_grad():
        y = (dec(X) * STD + MEAN).clamp(0, 1)                                                     # (2n, 3, 2, H, W)
    n = len(bl)
    for i, b in enumerate(bl):
        B = m["blocks"][b]; x0, y0, x1, y1 = [int(np.clip(v, 0, 255)) for v in B["box"]]
        if x1 <= x0 or y1 <= y0:
            continue
        rr = real[:, 2 * b: 2 * b + 2]
        reg = lambda t: t[..., y0:y1 + 1, x0:x1 + 1]
        cx, cy = B["ctr"]
        rows.append({"bin": int(np.clip(cy // 64, 0, 3) * 4 + np.clip(cx // 64, 0, 3)), "step": b - 7,
                     "gt_obj": psnr(reg(y[i]), reg(rr)), "gt_full": psnr(y[i], rr), "ro_obj": psnr(reg(y[n + i]), reg(rr)), "ro_full": psnr(y[n + i], rr)})
R = {"overall": {k: round(float(np.mean([r[k] for r in rows])), 2) for k in ("gt_obj", "gt_full", "ro_obj", "ro_full")}, "n": len(rows), "by_bin": {}, "by_step": {}}
for g in range(16):
    G = [r for r in rows if r["bin"] == g]
    if G:
        R["by_bin"][g] = {"n": len(G), **{k: round(float(np.mean([r[k] for r in G])), 2) for k in ("gt_obj", "ro_obj")}}
for s in range(1, 9):
    G = [r for r in rows if r["step"] == s]
    R["by_step"][s] = {"n": len(G), **{k: round(float(np.mean([r[k] for r in G])), 2) for k in ("gt_obj", "gt_full", "ro_obj", "ro_full")}}
json.dump(R, open(f"{REPO}/z_analysis/results/decode_by_position.json", "w"), indent=1)
print("overall", R["overall"], "n", R["n"])
print("구역별 (행 = 위→아래, 열 = 왼→오) 물체 상자 PSNR  정답 / 굴린 예측 (블록 수)")
for r_ in range(4):
    print("  ", " | ".join(f"{R['by_bin'][r_ * 4 + c]['gt_obj']:5.1f}/{R['by_bin'][r_ * 4 + c]['ro_obj']:5.1f} ({R['by_bin'][r_ * 4 + c]['n']:3d})" if (r_ * 4 + c) in R["by_bin"] else "      -      " for c in range(4)))
print("걸음별", {s: (v["gt_obj"], v["ro_obj"]) for s, v in R["by_step"].items()})
