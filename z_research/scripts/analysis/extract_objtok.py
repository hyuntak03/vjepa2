"""3D 합성 학습 쪽 영상에서 물체 칸 정답 토큰을 대량으로 뽑는다 (2026-10-25, 학습 없음) — object_subspace_dom.py 의 train 쪽 표본 늘리기.
사용자: "학습 데이터로 해봐 그건 샘플 많잖아" (400 편 → 정체 하나당 약 8 편이라 평균 차가 재현되지 않았다).

phys3d_single_v1 train, 시나리오마다 고르게 N 편. 원본 0, 3, … 장 (4 fps), 블록 = 2 장.
  블록 공간 (AR): 블록마다 따로 LN(target_encoder(2 장)). 영상 길이에 맞춰 블록 min(16, n // 6) 개.
  창 공간 (릴리즈): 96 장 영상만, 32 장 한 번에 LN(target_encoder) → 블록 자리별.
  보이는 블록 (vis ≥ 0.5 · 화면 안 ≥ 0.9) 의 물체 칸 토큰 (fp16) + 라벨 (편 번호 · 모양 · 색 · 시나리오 · 블록 · 블록 안 변위 dx dy · 중심 x y).

  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/extract_objtok.py --shard $k --nshard 8 --n 4000 & done; wait
출력: /workspace/auto_research_cache/objtok_train/shard{k}.npz
"""
import argparse, glob, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--n", type=int, default=4000)
a = ap.parse_args()
OUT = "/workspace/auto_research_cache/objtok_train"; os.makedirs(OUT, exist_ok=True)
P3 = "/workspace/dataset/training_data/phys3d_single_v1/train/clips"
IDX = [3 * k for k in range(32)]; G = 16; S = 256; D = 1280
fs = sorted(glob.glob(f"{P3}/*/*.mp4")); by = {}
for f in fs:
    by.setdefault(f.split("/")[-2], []).append(f)
rng = random.Random(7)
for k in sorted(by):
    rng.shuffle(by[k])
pick = []
while len(pick) < a.n and any(by.values()):
    for k in sorted(by):
        if by[k] and len(pick) < a.n:
            pick.append(by[k].pop())
mine = pick[a.shard:: a.nshard]

import decord
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _load_checkpoint(CK)["target_encoder"]
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, tgt, tag="t"); enc2 = enc2.to(dev).eval()
encw = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
_load_state_dict(encw, tgt, tag="t"); encw = encw.to(dev).eval(); del tgt
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
cc = np.arange(G) * 16 + 8.0
tok = {"ar": [], "rel": []}; lab = {"ar": [], "rel": []}
t0 = time.time()
for i, f in enumerate(mine):
    st = np.load(f[:-4] + "_states.npz"); pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    vr = decord.VideoReader(f, num_threads=4); n = len(vr); nb = min(16, n // 6)
    fr = vr.get_batch(IDX[: 2 * nb]).asnumpy()
    xn = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = LN(enc2(xn.view(3, nb, 2, 256, 256).permute(1, 0, 2, 3, 4)).float())                  # (nb, S, D)
        hw = LN(encw(xn.unsqueeze(0)).float())[0].view(16, S, D) if nb == 16 else None
    gid = a.shard * 100000 + i
    for b in range(nb):
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        if not (min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9):
            continue
        box = [min(bb[f0, 0], bb[f1, 0]), min(bb[f0, 1], bb[f1, 1]), max(bb[f0, 2], bb[f1, 2]), max(bb[f0, 3], bb[f1, 3])]
        if not np.isfinite(box).all():
            continue
        cols = np.nonzero((cc >= box[0]) & (cc <= box[2]))[0]; rows = np.nonzero((cc >= box[1]) & (cc <= box[3]))[0]
        cells = [int(r * G + c) for r in rows for c in cols]
        if not cells:
            continue
        d = pos[f1] - pos[f0]; ctr = 0.5 * (pos[f0] + pos[f1])
        row = [gid, int(st["shape_id"][0]), int(st["color_id"][0]), sorted(by.keys()).index(f.split("/")[-2]) if False else 0, b, float(d[0]), float(d[1]), float(ctr[0]), float(ctr[1])]
        for sp, H in (("ar", hb), ("rel", hw)):
            if H is None:
                continue
            tok[sp].append(H[b, cells].half().cpu().numpy())
            lab[sp] += [row + [c] for c in cells]
    if i % 50 == 0:
        print(time.strftime("%H:%M:%S"), f"shard {a.shard}: {i}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
scen_names = np.array([f.split("/")[-2] for f in mine])
np.savez(f"{OUT}/shard{a.shard}.npz", tok_ar=np.concatenate(tok["ar"]), lab_ar=np.array(lab["ar"], np.float64),
         tok_rel=np.concatenate(tok["rel"]) if tok["rel"] else np.zeros((0, D), np.float16), lab_rel=np.array(lab["rel"], np.float64).reshape(-1, 10),
         clips=np.array(mine), scen=scen_names)
print("saved", a.shard, len(lab["ar"]), len(lab["rel"]), flush=True)
