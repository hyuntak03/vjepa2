#!/usr/bin/env python3
"""[T] 반사실 쌍 렌더 → 목표 encoder 블록 토큰 + 블록 라벨 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/cf_encode.py --split test

입력: /workspace/dataset/training_data/phys3d_cf_v1/<split>/scene_*/<variant>.{mp4,json,_states.npz} ([물리] gen_phys3d_cf_blender.py 산출).
PNG 장 (_frames, 무손실) 이 있으면 그것을, 없으면 mp4 를 읽는다. 장 0,3,…,93 (12 fps → 4 fps, 32 장) = 블록 16 (블록 b = 장 6b · 6b+3). 읽기 · 전처리는 state_obs_check.py 와 같다
  (256 bilinear → uint8 반올림 → ImageNet 정규화). 블록마다 목표 encoder (ViT-H, 2 장, bf16) → affine 없는 LN → fp16.
블록 라벨 = 원 장 6b+3 (블록 마지막 장, 사용자 window_labels reduce=last 와 같은 규약): pos (px, 256 화면) · vis_frac · hidden · inframe_frac · seen.
출력 /workspace/auto_research_cache/phys3d_cf_v1/<split>_h.npy (N, 16, 256, 1280) fp16 · <split>_lab.npz · <split>_meta.json (행 순서 = 장면 · 변형).
검산: 행마다 블록 0 토큰 표준편차 (LN 이면 ≈ 1), 같은 장면 변형끼리 충돌 전 블록 토큰 차 (짝 안 문맥이 같은지)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_cf_v1")
ap.add_argument("--out", default="/workspace/auto_research_cache/phys3d_cf_v1")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
import decord                                                               # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, NB = 256, 16
IDX = list(range(0, 96, 3))
LAB_T = [6 * b + 3 for b in range(NB)]
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CKE)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()

items = []
for sd in sorted(glob.glob(os.path.join(a.root, a.split, "scene_*"))):
    for js in sorted(glob.glob(os.path.join(sd, "*.json"))):
        stem = js[:-5]
        if (os.path.exists(stem + ".mp4") or os.path.isdir(stem + "_frames")) and os.path.exists(stem + "_states.npz"):
            items.append(stem)
print(f"[cf_encode] {a.split}: clip {len(items)}", flush=True)
os.makedirs(a.out, exist_ok=True)
hp = os.path.join(a.out, f"{a.split}_h.npy")
H = np.lib.format.open_memmap(hp, mode="w+", dtype=np.float16, shape=(len(items), NB, S, 1280))
LK = ("pos", "vis_frac", "hidden", "inframe_frac", "seen")
LAB = {k: [] for k in LK}; META = []
t0 = time.time()
for i, stem in enumerate(items):
    fdir = stem + "_frames"
    if os.path.isdir(fdir):                                                  # PNG (무손실) 우선 — mp4 압축은 짝 안 같은 문맥을 0.5–0.8 계조 흔든다
        fs = sorted(glob.glob(os.path.join(fdir, "f*.png")))
        fr = torch.from_numpy(np.stack([np.asarray(Image.open(fs[t]).convert("RGB")) for t in IDX])).permute(0, 3, 1, 2).float()
        src = "png"
    else:
        vr = decord.VideoReader(stem + ".mp4", num_threads=1)
        fr = torch.from_numpy(vr.get_batch(IDX).asnumpy()).permute(0, 3, 1, 2).float()
        src = "mp4"
    x = F.interpolate(fr, size=(256, 256), mode="bilinear", align_corners=False).round().clamp(0, 255) / 255.0
    x = ((x - MEAN) / STD).view(NB, 2, 3, 256, 256).permute(0, 2, 1, 3, 4).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(x[j:j + 8]) for j in range(0, NB, 8)], 0).float()
    H[i] = F.layer_norm(h, (1280,)).half().cpu().numpy()
    z = np.load(stem + "_states.npz")
    for k in LK:
        LAB[k].append(np.asarray(z[k])[LAB_T])
    m = json.load(open(stem + ".json"))
    META.append({k: m.get(k) for k in ("scene", "variant", "pair", "possible", "cf_axis", "obstacle_visible", "obstacle_collides", "t_hit", "t_diverge", "t_event", "ctx_copied_until",
                                       "identity", "shape", "color", "seed")} | {"obstacle": m["obstacle"], "screen": m["screen"], "camera": m["env"]["camera"]["mode"],
                                                                                  "path": stem + ".mp4", "row": i, "frames_src": src})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
H.flush()
np.savez(os.path.join(a.out, f"{a.split}_lab.npz"), **{k: np.stack(v) for k, v in LAB.items()}, lab_frames=np.array(LAB_T))
json.dump(META, open(os.path.join(a.out, f"{a.split}_meta.json"), "w"), indent=1)
# 검산
Hm = np.load(hp, mmap_mode="r")
sd0 = float(np.mean([np.asarray(Hm[i, 0], np.float32).std() for i in range(min(len(items), 20))]))
by = {}
for r in META:
    by.setdefault(r["scene"], {})[r["variant"]] = r["row"]
d_same, d_pair = [], []
for sc_, v in by.items():
    r0 = META[next(iter(v.values()))]; te = r0.get("t_event") or r0["t_hit"]
    tb = max(1, sum(1 for b in range(NB) if 6 * b + 3 < te))                  # 블록 마지막 장 6b+3 이 사건 장보다 앞인 블록
    if "W_pos" in v and "W_imp" in v:
        d_same.append(float(np.abs(np.asarray(Hm[v["W_pos"], :tb], np.float32) - np.asarray(Hm[v["W_imp"], :tb], np.float32)).mean()))
    if "W_pos" in v and "N_pos" in v:
        d_pair.append(float(np.abs(np.asarray(Hm[v["W_pos"], :tb], np.float32) - np.asarray(Hm[v["N_pos"], :tb], np.float32)).mean()))
print(f"[cf_encode] {a.split}: 블록 0 토큰 표준편차 {sd0:.3f} · 충돌 전 토큰 차: 같은 짝 (W_pos vs W_imp) {np.mean(d_same) if d_same else float('nan'):.4f} "
      f"(장면 {len(d_same)}) · 짝 사이 (W_pos vs N_pos) {np.mean(d_pair) if d_pair else float('nan'):.4f} (장면 {len(d_pair)}) · {time.time() - t0:.0f}s", flush=True)
