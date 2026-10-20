#!/usr/bin/env python3
"""2D 렌더 잡음 · latent 잡음 바닥 (2026-10-19, 사용자 질문 "2d render noise 측정해봄?").

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/latent_noise_floor.py [--n 96]

질문: 합성 step 의 predictor grad norm 이 자연 step 의 ~5 배다 — 2D 렌더 화면에서 정답 토큰 (블록별 LN(target_encoder)) 자체가
장마다 크게 출렁여 latent 손실 · gradient 를 키우는가?
재는 것 (학습과 같은 블록 인코딩 · 같은 프레임 간격):
  pix      연속 프레임 픽셀 |Δ| 평균 (0–255)
  copy     연속 블록 정답 토큰 L1 (LN, 토큰 평균) = "직전 블록 복사" 의 latent 손실
  합성 (phys2d_single_v1 val, stride 1 = 학습 합성 창과 같다):
     static  물체 속력 < 0.3 px/장 인 8 블록 창 · 가림막도 안 움직이는 것 (= 화면이 거의 그대로)
     all     무작위 창
  자연 (학습과 같은 fps 4 간격, 표의 stride): k710 · ssv2 무작위 창
static 합성의 copy 가 자연 all 보다 크면 → 화면이 그대로인데도 정답 토큰이 출렁인다 (encoder 잡음 · 렌더 떨림)."""
import argparse, csv, glob, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=96, help="세트마다 창 수")
ap.add_argument("--nb", type=int, default=8, help="창 블록 수")
ap.add_argument("--single", default="/workspace/dataset/training_data/phys2d_single_v1/val")
a = ap.parse_args()
dev, TUB = "cuda", 2
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
T = a.nb * TUB
rng = random.Random(0)


def read(path, idx):
    vr = decord.VideoReader(path, num_threads=1)
    if idx[-1] >= len(vr):
        return None
    fr = torch.from_numpy(vr.get_batch(idx).asnumpy())                 # (T, H, W, 3)
    H, W = fr.shape[1:3]
    s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2                  # 가운데 정사각 (자연 영상) — 합성은 이미 256²
    x = fr[:, y0:y0 + s, x0:x0 + s].permute(3, 0, 1, 2).float()          # (3, T, s, s)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return x


@torch.no_grad()
def measure(clips):
    pix, cp = [], []
    for i in range(0, len(clips), 4):
        x = torch.stack(clips[i:i + 4]).to(dev)                           # (B, 3, T, 256, 256) 0–255
        pix.append((x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=(1, 3, 4)).mean(1).cpu())
        xn = (x / 255.0 - MEAN) / STD
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, xn, TUB).float()                    # (B, nb*256, D)
        hb = h.reshape(h.size(0), a.nb, 256, -1)
        cp.append((hb[:, 1:] - hb[:, :-1]).abs().mean(dim=(-1, -2)).mean(1).cpu())
    pix, cp = torch.cat(pix), torch.cat(cp)
    return {"n": len(clips), "pix": float(pix.mean()), "copy": float(cp.mean()), "copy_med": float(cp.median())}


# ---- 합성: phys2d_single val
paths = sorted(glob.glob(f"{a.single}/clips/*/*.mp4")); rng.shuffle(paths)
static, alls = [], []
for p in paths:
    if len(static) >= a.n and len(alls) >= a.n:
        break
    d = np.load(p.replace(".mp4", "_states.npz"))
    meta = json.load(open(p.replace(".mp4", ".json")))
    n = d["pos"].shape[0]
    if n < T + 1:
        continue
    sp = np.linalg.norm(np.nan_to_num(d["vel"][:, 0]), axis=-1)
    occ = d["occluder_rect"]                                              # (n, k, 4)
    occ_mv = np.abs(np.diff(np.nan_to_num(occ), axis=0)).max(axis=(1, 2)) if occ.shape[1] else np.zeros(n - 1)
    if len(alls) < a.n:
        s0 = rng.randrange(0, n - T + 1); x = read(p, list(range(s0, s0 + T)))
        if x is not None:
            alls.append(x)
    if len(static) < a.n:
        ok = [s0 for s0 in range(0, n - T + 1) if sp[s0:s0 + T].max() < 0.3 and occ_mv[s0:s0 + T - 1].max(initial=0) < 0.3]
        if ok:
            x = read(p, list(range(ok[0], ok[0] + T)))
            if x is not None:
                static.append(x)
res = {"합성 static": measure(static), "합성 all": measure(alls)}

# ---- 자연: 학습과 같은 fps 4 간격 (표의 stride)
for name, tab in (("자연 k710", "k710_320"), ("자연 ssv2", "ssv2")):
    rows = list(csv.reader(open(os.path.join(REPO, f"data_csv/{tab}/fps4_table.tsv")), delimiter="\t"))
    rng.shuffle(rows); clips = []
    for r in rows:
        if len(clips) >= a.n:
            break
        path, nfr, sd = r[0], int(r[2]), int(r[3])
        if nfr < sd * (T - 1) + 1:
            continue
        s0 = rng.randrange(0, nfr - sd * (T - 1))
        try:
            x = read(path, [s0 + sd * k for k in range(T)])
        except Exception:
            x = None
        if x is not None:
            clips.append(x)
    res[name] = measure(clips)

print(f"\n연속 블록 정답 토큰 L1 (블록별 LN target_encoder, 토큰 평균 = 직전 블록 복사의 latent 손실) · 창 {a.nb} 블록")
print(f"  {'세트':12s} {'창':>4s} {'픽셀 |Δ|':>9s} {'copy L1':>9s} {'(중앙)':>8s}")
for k, v in res.items():
    print(f"  {k:12s} {v['n']:4d} {v['pix']:9.2f} {v['copy']:9.4f} {v['copy_med']:8.4f}")
out = os.path.join(REPO, "z_training/runs/_analysis"); os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "latent_noise_floor_2026-10-19.json"), "w"), ensure_ascii=False, indent=1)
