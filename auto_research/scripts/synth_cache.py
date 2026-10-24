#!/usr/bin/env python3
"""[T] 2 순위 데이터: phys3d 학습 분할 → 블록 토큰 + 라벨, 절반은 "무름" 증강 (블록마다 물체 색 변환) (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/synth_cache.py --n 800

창 = 16 장 (간격 3, 시작 0) = 블록 8 개. 물체 픽셀 = 그 장의 bbox 안에서 클립 시간 중앙값 (전체 장) 과 채널 합 |Δ| > 30 인 픽셀.
무름 (창의 절반): 블록 k 마다 새 무작위 색 변환 T_k (채널별 이득 0.5–1.5 · 오프셋 −60..60) 를 물체 픽셀에 0.7 섞기 (두 장 같은 변환). 단단함: 그대로.
라벨 (블록마다 둘째 장): 물체 중심 위치 (px) · bbox · 보임 · 안정도 (단단함 1 / 무름 0).
저장: <out>/synth_h.npy (N,8,256,1280) fp16 · synth_lab.npz (pos (N,8,2) · bbox (N,8,4) · vis (N,8) · stable (N,))."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/train/clips")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--alpha", type=float, default=0.7)
ap.add_argument("--out", default="/workspace/auto_research_cache/synth_aux")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
rng = np.random.default_rng(a.seed)
paths = sorted(glob.glob(os.path.join(a.root, "*", "*.mp4")))
rng.shuffle(paths)
idx = [3 * k for k in range(16)]
H, POS, BB, VIS, STB = [], [], [], [], []
for p in paths:
    if len(H) >= a.n:
        break
    st = np.load(p[:-4] + "_states.npz")
    if st["pos"].shape[0] < idx[-1] + 1:
        continue
    vr = decord.VideoReader(p, num_threads=1)
    allf = vr.get_batch(list(range(0, len(vr), 3))).asnumpy().astype(np.float32)          # 중앙값 배경용
    fr = vr.get_batch(idx).asnumpy().astype(np.float32)                                      # (16,H,W,3)
    if fr.shape[1] != 256:
        t_ = torch.from_numpy(fr).permute(0, 3, 1, 2); fr = F.interpolate(t_, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
        t_ = torch.from_numpy(allf).permute(0, 3, 1, 2); allf = F.interpolate(t_, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).numpy()
    med = np.median(allf, 0)
    soft = bool(rng.random() < 0.5)
    sc = 256.0 / max(vr[0].shape[0], 1)
    out = fr.copy()
    for k in range(8):
        if soft:
            gain = rng.uniform(0.5, 1.5, 3); off = rng.uniform(-60, 60, 3)
        for f in (2 * k, 2 * k + 1):
            bb = st["bbox"][idx[f], 0] * sc
            if not np.isfinite(bb).all():
                continue
            x0, y0, x1, y1 = [int(np.clip(round(v), 0, 256)) for v in bb]
            if x1 <= x0 or y1 <= y0:
                continue
            reg = fr[f, y0:y1, x0:x1]
            m = (np.abs(reg - med[y0:y1, x0:x1]).sum(-1) > 30)[..., None]
            if soft:
                tr = np.clip(reg * gain + off, 0, 255)
                out[f, y0:y1, x0:x1] = np.where(m, (1 - a.alpha) * reg + a.alpha * tr, reg)
    x = torch.from_numpy(np.round(out)).to(dev).permute(3, 0, 1, 2)[None] / 255
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(enc, ((x - MEAN) / STD).to(torch.bfloat16), 2)[0].view(8, 256, -1)
    H.append(h.half().cpu().numpy())
    sec = [idx[2 * k + 1] for k in range(8)]
    POS.append(st["pos"][sec, 0] * sc); BB.append(st["bbox"][sec, 0] * sc)
    VIS.append((st["vis_frac"][sec, 0] > 0.5) & (st["inframe_frac"][sec, 0] > 0.9)); STB.append(0 if soft else 1)
    if len(H) % 100 == 0:
        print(f"{len(H)}/{a.n}", flush=True)
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, "synth_h.npy"), np.stack(H))
np.savez(os.path.join(a.out, "synth_lab.npz"), pos=np.stack(POS), bbox=np.stack(BB), vis=np.stack(VIS), stable=np.array(STB))
print("done", len(H), "단단함", int(np.sum(STB)), flush=True)
