#!/usr/bin/env python3
"""phys3d_single_v1 clip 을 채점과 같은 방식으로 인코딩해 캐시한다 (auto_research, 2026-10-21).

  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_phys3d.py \
      --split val --shard $k --nshard 4 & done; wait

창: raw 0 부터 간격 3 으로 32 장 (= 블록 16 개, 블록 = 연속 2 장). 블록별 LN(target_encoder) — 블록 AR 의 문맥 · 정답과 같은 공간.
--encoder random 이면 같은 구조의 **무작위 초기화** ViT-H (학습 전 구조가 담는 것, MotionJEPA 의 Random 기준선과 같은 논리).
저장 (shard 마다): h (N,16,256,1280) fp16 · pix (N,32,256,256,3) uint8 (--pix) · lab.npz (장별 라벨 32 장)."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1")
ap.add_argument("--out", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--encoder", default="target", choices=["target", "random"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--pix", type=int, default=1)
ap.add_argument("--r0", type=int, nargs="+", default=[0])
a = ap.parse_args()
dev = torch.device("cuda")
NF, ST = 32, 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
torch.manual_seed(0)
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
if a.encoder == "target":
    _load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t")
enc = enc.to(dev, torch.bfloat16).eval()

paths = sorted(glob.glob(os.path.join(a.root, a.split, "clips", "*", "*.mp4")))
paths = [p for p in paths if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= max(a.r0) + ST * (NF - 1) + 1]
if a.limit:
    paths = paths[: a.limit]
paths = paths[a.shard:: a.nshard]
KEYS = ("pos", "vel", "acc", "bbox", "vis_frac", "present", "hidden", "inframe_frac", "impact")
H, PX, LAB, META = [], [], {k: [] for k in KEYS}, []
for i, p in enumerate(paths):
    st = np.load(p[:-4] + "_states.npz")
    vr = decord.VideoReader(p, num_threads=2)
    for r0 in a.r0:
        idx = [r0 + ST * k for k in range(NF)]
        fr = vr.get_batch(idx).asnumpy()                                              # (T,H,W,3)
        x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, ((x - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NF // 2, 256, -1)
        H.append(h.half().cpu().numpy())
        if a.pix:
            PX.append((x[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().byte().cpu().numpy())
        for k in KEYS:
            LAB[k].append(st[k][idx, 0])
        META.append(dict(path=os.path.relpath(p, a.root), r0=r0, scen=p.split("/")[-2], shape=int(st["shape_id"][0]), color=int(st["color_id"][0]),
                         size=float(st["size"][0]), area=float(st["area"][0])))
    if i % 50 == 0:
        print(f"[{a.shard}] {i}/{len(paths)}", flush=True)
os.makedirs(a.out, exist_ok=True)
tag = f"{a.split}_{a.encoder}_s{a.shard}"
np.save(os.path.join(a.out, f"h_{tag}.npy"), np.stack(H))
if a.pix:
    np.save(os.path.join(a.out, f"pix_{a.split}_s{a.shard}.npy"), np.stack(PX))
np.savez(os.path.join(a.out, f"lab_{a.split}_s{a.shard}.npz"), **{k: np.stack(v) for k, v in LAB.items()})
json.dump(META, open(os.path.join(a.out, f"meta_{a.split}_s{a.shard}.json"), "w"))
print(f"[{a.shard}] done {len(H)} windows", flush=True)
