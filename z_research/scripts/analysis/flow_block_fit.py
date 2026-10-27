#!/usr/bin/env python3
"""fps 10 (블록 0.2 초) 에서 인접 두 블록 토큰의 차이를 광학 흐름으로 맞출 수 있나 (2026-10-27, 사용자 질문).

  # GPU 8 장 (rank 마다 clip 을 나눠), 끝나면 --aggregate
  for r in 0..7: CUDA_VISIBLE_DEVICES=$r python flow_block_fit.py --rank $r --world 8 &
  python flow_block_fit.py --aggregate

데이터 (블록 = 연속 2 장, 0.1 초 간격, 블록 사이 0.2 초):
  실제 영상 — data_csv/{k710_320, ssv2, ht100m_1pct}/fps10t_table.tsv 에서 각 --n clip, 16 장 (8 블록), 시간 기준 (학습 로더와 같은 규칙)
  결정적 3D 합성 — phys3d_single_v1 val, 30 fps 원본 3 장 간격 16 장
화면: 짧은 변 256 으로 줄이고 가운데 256×256. 토큰 h = LN(target_encoder(블록)), 16×16 칸 × 1280.
흐름: RAFT large (torchvision C_T_SKHT_V2, 12 반복). 블록 b → b+1 = 장 2b → 2b+2 (0.2 초). 뒤로 가는 흐름 (2b+2 → 2b) 으로 옮긴다.
   칸 흐름 = 16×16 픽셀 평균 (칸 단위 = px / 16).
측정 (블록 쌍마다 칸별, 토큰 L1 평균):
  ① 복사      |h_{b+1} − h_b|
  ② 흐름 옮김 |h_{b+1} − warp(h_b, 흐름)|      (칸 격자 bilinear)
  ③ 위약      |h_{b+1} − warp(h_b, 흐름 90° 회전)|   → 순 이득 = (③ − ②) / ① (관문 8: 위약을 뺀 값만 판정에 쓴다)
  ④ 선형 설명 Δh = h_{b+1} − h_b 를 그 칸 3×3 이웃 흐름 (18 차원) 으로 ridge (clip 반 맞춤 / 반 확인) 한 R²
칸을 흐름 크기로 나눈다: < 0.25 칸 · 0.25–1 칸 · ≥ 1 칸."""
import argparse, glob, json, math, os, random, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "z_research/results/flow_block_fit")
ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--n", type=int, default=200, help="데이터셋당 clip 수")
ap.add_argument("--aggregate", action="store_true")
a = ap.parse_args()
G, NB = 16, 8
BINS = ((0, 0.25), (0.25, 1.0), (1.0, 99.0))

if a.aggregate:
    parts = sorted(glob.glob(os.path.join(OUT, "part_*.npz")))
    D = {}
    for p in parts:
        z = np.load(p, allow_pickle=True)
        for src in z.files:
            D.setdefault(src, []).append(z[src])
    print(f"조각 {len(parts)} 개\n")
    print(f"{'데이터':10s} {'칸 흐름':>11s} {'칸 수':>8s} {'① 복사':>8s} {'② 흐름':>8s} {'③ 위약':>8s} {'②의 이득':>9s} {'위약 이득':>9s} {'순 이득':>8s}")
    summary = {}
    for src, arrs in D.items():
        X = np.concatenate(arrs)            # 열: mag, copy, warp, plac, dh(1280 은 따로) → 여기선 4 열
        for lo, hi in BINS:
            m = (X[:, 0] >= lo) & (X[:, 0] < hi)
            if m.sum() < 50: continue
            c, w, pl = X[m, 1].mean(), X[m, 2].mean(), X[m, 3].mean()
            g, gp = (c - w) / c, (c - pl) / c
            print(f"{src:10s} {f'{lo}–{hi if hi < 99 else chr(8734)}':>11s} {m.sum():8d} {c:8.4f} {w:8.4f} {pl:8.4f} {100 * g:8.1f}% {100 * gp:8.1f}% {100 * (g - gp):7.1f}%")
            summary[f"{src}|{lo}"] = dict(n=int(m.sum()), copy=float(c), warp=float(w), placebo=float(pl), net=float(g - gp))
    print("\n④ 선형 설명 (칸 3×3 흐름 → Δh, ridge, 확인 반의 R²)")
    for p in sorted(glob.glob(os.path.join(OUT, "ridge_*.json"))):
        print("  " + open(p).read().strip())
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    sys.exit()

import decord
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
from app.vjepa_frozen import ar as AR
from app.vjepa_frozen.var_window import _Table
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()

# ---- clip 목록 (rank 공통, 고정 시드) ------------------------------------------------------------
jobs = []
for name, path in (("k710", "data_csv/k710_320/fps10t_table.tsv"), ("ssv2", "data_csv/ssv2/fps10t_table.tsv"), ("how", "data_csv/ht100m_1pct/fps10t_table.tsv")):
    t = _Table(name, os.path.join(REPO, path), 1.0); lo = t.first_ok(2 * NB); rng = random.Random(f"fbf-{name}")
    for _ in range(a.n):
        j = rng.randrange(lo, len(t.paths))
        jobs.append((name, t.paths[j], t.fps[j], t.nframes[j], rng.random()))
ph = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4"))
rng = random.Random("fbf-phys3d")
for p in rng.sample(ph, min(a.n, len(ph))):
    jobs.append(("phys3d", p, 30, None, None))
jobs = jobs[a.rank::a.world]


def frames(job):
    name, p, f, n, u = job
    vr = decord.VideoReader(p, num_threads=2)
    n = len(vr)
    if name == "phys3d":
        idx = [3 * k for k in range(2 * NB)]
        if idx[-1] >= n: return None
    else:
        dur = (2 * NB - 1) / 10; tmax = (n - 1) / f - dur
        if tmax < 0: return None
        t0 = u * tmax
        idx = [min(n - 1, int(round((t0 + k / 10) * f))) for k in range(2 * NB)]
    x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(0, 3, 1, 2).float()      # (T, 3, H, W) 0..255
    H, W = x.shape[-2:]; s = 256 / min(H, W)
    x = F.interpolate(x, size=(max(256, round(H * s)), max(256, round(W * s))), mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]; y0, x0 = (H - 256) // 2, (W - 256) // 2
    return x[:, :, y0:y0 + 256, x0:x0 + 256]


def warp(hg, fl):
    """hg (D, 16, 16) 토큰 격자, fl (2, 16, 16) 칸 단위 (dx, dy) — 칸 (i,j) 에 hg 의 (i+dy, j+dx) 를 bilinear 로."""
    yy, xx = torch.meshgrid(torch.arange(G, device=dev).float(), torch.arange(G, device=dev).float(), indexing="ij")
    gx = (xx + fl[0]) / (G - 1) * 2 - 1; gy = (yy + fl[1]) / (G - 1) * 2 - 1
    return F.grid_sample(hg[None], torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="border", align_corners=True)[0]


RES = {}; RID = {}
for ci, job in enumerate(jobs):
    try:
        x = frames(job)
    except Exception as e:
        print("skip", job[1], e, flush=True); continue
    if x is None: continue
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(tgt, ((x.permute(1, 0, 2, 3)[None] / 255 - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(NB, G, G, -1)
        a1 = x[0:2 * NB - 2:2] / 127.5 - 1; a2 = x[2:2 * NB:2] / 127.5 - 1                      # 장 2b, 2b+2
        with torch.autocast("cuda", dtype=torch.float16):
            fb = raft(a2.contiguous(), a1.contiguous(), num_flow_updates=12)[-1].float()       # 2b+2 → 2b (뒤로), (NB-1, 2, 256, 256) px
        fc = F.avg_pool2d(fb, 16) / 16                                                          # 칸 단위 (NB-1, 2, 16, 16)
        rot = torch.stack([-fc[:, 1], fc[:, 0]], 1)                                             # 90° 회전 (위약)
        for b in range(NB - 1):
            hb = h[b].permute(2, 0, 1); hn = h[b + 1].permute(2, 0, 1)
            c = (hn - hb).abs().mean(0); w = (hn - warp(hb, fc[b])).abs().mean(0); pl = (hn - warp(hb, rot[b])).abs().mean(0)
            mag = fc[b].norm(dim=0)
            RES.setdefault(job[0], []).append(torch.stack([mag, c, w, pl], -1).reshape(-1, 4).cpu().numpy())
            # ④ 용: 칸 3×3 흐름 특징 (18) 과 Δh 의 저차원 투영 (Δh 1280 → 고정 무작위 64 축, 메모리)
            nb9 = F.unfold(F.pad(fc[b][None], (1, 1, 1, 1), mode="replicate"), 3)[0].T                # (256, 18)
            RID.setdefault(job[0], []).append((nb9.cpu().numpy(), (hn - hb).reshape(1280, -1).T.cpu().numpy()))
    if ci % 20 == 0:
        print(f"rank {a.rank}: {ci}/{len(jobs)}", flush=True)
np.savez(os.path.join(OUT, f"part_{a.rank}.npz"), **{k: np.concatenate(v) for k, v in RES.items()})
# ④ ridge: rank 안에서 clip 반 맞춤 / 반 확인 (Δh 전체 1280 차원, 다출력)
for src, L in RID.items():
    half = len(L) // 2
    if half < 7: continue
    Xa = np.concatenate([l[0] for l in L[:half]]); Ya = np.concatenate([l[1] for l in L[:half]])
    Xb = np.concatenate([l[0] for l in L[half:]]); Yb = np.concatenate([l[1] for l in L[half:]])
    Xa1 = np.c_[Xa, np.ones(len(Xa))]; Xb1 = np.c_[Xb, np.ones(len(Xb))]
    Wt = np.linalg.solve(Xa1.T @ Xa1 + 1e-2 * np.eye(Xa1.shape[1]), Xa1.T @ Ya)
    r2 = 1 - ((Yb - Xb1 @ Wt) ** 2).sum() / ((Yb - Yb.mean(0)) ** 2).sum()
    mv = np.linalg.norm(Xb[:, [4, 13]], axis=1) >= 0.25                                       # 가운데 칸 흐름 (unfold: dx 4 · dy 13) ≥ 0.25 칸
    r2m = 1 - ((Yb[mv] - Xb1[mv] @ Wt) ** 2).sum() / ((Yb[mv] - Yb[mv].mean(0)) ** 2).sum() if mv.sum() > 100 else float("nan")
    json.dump({"rank": a.rank, "src": src, "r2_all": float(r2), "r2_moving": float(r2m), "n_fit": int(len(Xa)), "n_test": int(len(Xb))},
              open(os.path.join(OUT, f"ridge_{src}_{a.rank}.json"), "w"))
print(f"rank {a.rank} done", flush=True)
