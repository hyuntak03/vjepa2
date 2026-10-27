"""개념 벡터를 encoder · predictor 각각에서 구하기 위한 물체 칸 토큰 추출 (2026-10-26, 학습 없음).
사용자: "concept vector (encoder), predictor 이렇게 각각 구해야지. 그리고 그 2 개 비교해야될 듯".

phys3d_single_v1 train 96 장 영상, 시나리오마다 고르게 N 편 (기본 3,000 = 정체당 약 60 편). 원본 0, 3, …, 93 장 (4 fps) → 블록 16.
블록 AR (natural_ar_vw_state latest, 블록 공간):
  enc   블록 b 정답 LN(target_encoder(2 장))                         b = 0..15
  tf    teacher forcing 한 걸음 예측 (블록 0..b−1 정답 → 블록 b)       b = 1..15
  roll  굴린 예측 (문맥 블록 0..7 → 블록 8..15, 걸음 = b − 7)
물체가 보이는 블록의 물체 칸 토큰 (fp16) + 라벨 (편 · 모양 · 색 · 시나리오 번호 · 블록 · 블록 안 변위 dx dy) 을 출처별로 저장.
GPU 메모리 상한 MEM_GB (기본 6) — 다른 세션 학습과 같은 장을 나눠 쓴다.

  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((k+1)) MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/concept_vectors_extract.py --shard $k --nshard 4 & done; wait
출력: /workspace/auto_research_cache/concept_ar/shard{k}.npz
"""
import argparse, glob, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
OUT = "/workspace/auto_research_cache/concept_ar"; os.makedirs(OUT, exist_ok=True)
P3 = "/workspace/dataset/training_data/phys3d_single_v1/train/clips"
IDX = [3 * k for k in range(32)]; G = 16; S = 256; D = 1280; NBC = 8
SCEN = sorted(os.listdir(P3))
by = {}
for f in sorted(glob.glob(f"{P3}/*/*.mp4")):
    by.setdefault(f.split("/")[-2], []).append(f)
rng = random.Random(11)
for k in sorted(by):
    rng.shuffle(by[k])
pick = []
while len(pick) < a.n and any(by.values()):
    for k in sorted(by):
        while by[k]:
            f = by[k].pop()
            if np.load(f[:-4] + "_states.npz")["pos"].shape[0] >= 96:
                pick.append(f); break
        if len(pick) >= a.n:
            break
mine = pick[a.shard:: a.nshard]

import decord
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FTm
sys.argv = _argv
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
ar = FTm.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
cc = np.arange(G) * 16 + 8.0
idx = torch.arange(16 * S, device=dev).unsqueeze(0)
tok = {k: [] for k in ("enc", "tf", "roll")}; lab = {k: [] for k in ("enc", "tf", "roll")}
t0 = time.time()
for i, f in enumerate(mine):
    st = np.load(f[:-4] + "_states.npz"); pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
    xn = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = torch.cat([LN(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4)).float()) for q in range(8)])   # 블록 2 개씩 (메모리)
        tf = ar.forward_seq(hb[:15].reshape(1, -1, D), idx[:, : 15 * S]).float()[0].view(15, S, D)
        ro = ar.rollout_kv(hb[:NBC].reshape(1, -1, D), idx[:, : NBC * S], 16 - NBC).float()[0].view(16 - NBC, S, D)
    gid = a.shard * 100000 + i; sc = SCEN.index(f.split("/")[-2])
    for b in range(16):
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
        d = pos[f1] - pos[f0]
        row = [gid, int(st["shape_id"][0]), int(st["color_id"][0]), sc, b, float(d[0]), float(d[1])]
        for nm, H, ok in (("enc", hb[b] if True else None, True), ("tf", tf[b - 1] if b >= 1 else None, b >= 1), ("roll", ro[b - NBC] if b >= NBC else None, b >= NBC)):
            if ok:
                tok[nm].append(H[cells].half().cpu().numpy()); lab[nm] += [row] * len(cells)
    if i % 100 == 0:
        print(time.strftime("%H:%M:%S"), f"shard {a.shard}: {i}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
np.savez(f"{OUT}/shard{a.shard}.npz", **{f"tok_{k}": np.concatenate(v) for k, v in tok.items()}, **{f"lab_{k}": np.array(lab[k], np.float64) for k in lab}, scen_names=np.array(SCEN))
print("saved", a.shard, {k: len(v) for k, v in lab.items()}, flush=True)
