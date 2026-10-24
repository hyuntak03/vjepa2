#!/usr/bin/env python3
"""[T] 방법 C 데이터: 자연 영상 창에 "움직이는 조각" 을 붙여 블록 토큰으로 인코딩 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/patch_cache.py --src k710_320 --n 300 --seed 11
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/patch_cache.py --holdout 1 --n 100 --seed 12

창 = 16 장 (블록 8 개, 4 fps), 학습 로더와 같은 읽기 (가운데 정사각 → 256).
조각: 같은 영상 첫 장의 임의 자리에서 자른 정사각 (한 변 32–64 px). 문맥 + 미래 8 블록 내내 등속 (블록당 16–32 px = 1–2 칸, 임의 방향), 화면 안에 머문다.
  종류 (반반): 단단함 = 외형 그대로 / 무름 = 블록마다 외형 = 0.3·지금 + 0.7·새 조각 (새 조각도 첫 장의 다른 자리).
  붙이기 (덮어쓰기 · 단단한 경계) · 첫 외형 분포 · 크기 · 속도 분포는 두 종류가 같다 → 두 종류를 가르는 단서는 문맥 블록 비교뿐.
블록 토큰 = 붙인 픽셀을 블록별 LN(target_encoder) bf16 (학습 정답과 같은 처리).
저장: <out>/{tag}_h.npy (N,8,256,1280) fp16 · {tag}_patch.json (종류 · 크기 · 장마다 왼쪽 위 좌표)."""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="k710_320")
ap.add_argument("--holdout", type=int, default=0, help="1 = K710 보류 (val_min224) 영상에서")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--soft_c", type=float, default=0.3)
ap.add_argument("--train_cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--out", default="/workspace/auto_research_cache/natural_patch")
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

if a.holdout:
    rows = [l.split()[0] for l in open(os.path.join(REPO, "data_csv/k710_320/val_min224.csv")) if l.strip()]
    rng.shuffle(rows)
    items = ((p, None, None) for p in rows)
    tag = "k710_holdout_patch"
else:
    meta = json.load(open(os.path.join(a.train_cache, f"{a.src}_meta.json")))
    order = rng.permutation(len(meta))
    items = ((meta[i]["path"], meta[i]["start"], meta[i]["fstp"]) for i in order)
    tag = f"{a.src}_patch"


def read16(path, start, fstp):
    vr = decord.VideoReader(path, num_threads=1)
    if fstp is None:
        fstp = max(1, int(round(vr.get_avg_fps() / (30 / 7))))
        span = 15 * fstp + 1
        if len(vr) < span:
            return None
        start = int(rng.integers(0, len(vr) - span + 1))
    fr = vr.get_batch(list(range(start, start + 15 * fstp + 1, fstp))).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()     # (16,3,256,256) cpu


H, META = [], []
for path, start, fstp in items:
    if len(H) >= a.n:
        break
    try:
        x = read16(path, start, fstp)
    except Exception:
        continue
    if x is None:
        continue
    s = int(rng.integers(32, 65))
    for _ in range(50):                                                    # 화면 안에 머무는 궤적
        th = rng.uniform(0, 2 * math.pi); vb = rng.uniform(16, 32)       # 블록당 px
        vx, vy = vb / 2 * math.cos(th), vb / 2 * math.sin(th)            # 장당
        lo_x, hi_x = max(0, -15 * vx), min(256 - s, 256 - s - 15 * vx); lo_y, hi_y = max(0, -15 * vy), min(256 - s, 256 - s - 15 * vy)
        if lo_x <= hi_x and lo_y <= hi_y:
            break
    else:
        continue
    px0, py0 = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    soft = bool(rng.random() < 0.5)
    pick = lambda: (int(rng.integers(0, 256 - s + 1)), int(rng.integers(0, 256 - s + 1)))
    sx, sy = pick()
    A = x[0, :, sy:sy + s, sx:sx + s].clone()
    apps = [A]
    for k in range(1, 8):
        if soft:
            nx, ny = pick()
            A = a.soft_c * A + (1 - a.soft_c) * x[0, :, ny:ny + s, nx:nx + s]
        apps.append(A.clone())
    xs = x.clone(); tl = []
    for f in range(16):
        cxp, cyp = int(round(px0 + vx * f)), int(round(py0 + vy * f))
        xs[f, :, cyp:cyp + s, cxp:cxp + s] = apps[f // 2].round()
        tl.append((cxp, cyp))
    xx = (xs.to(dev) / 255).permute(1, 0, 2, 3)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(8, 256, -1)
    H.append(h.half().cpu().numpy())
    META.append(dict(path=path, soft=soft, size=s, tl=tl, src=(sx, sy)))
    if len(H) % 50 == 0:
        print(f"{tag}: {len(H)}/{a.n}", flush=True)
os.makedirs(a.out, exist_ok=True)
np.save(os.path.join(a.out, f"{tag}_h.npy"), np.stack(H)); json.dump(META, open(os.path.join(a.out, f"{tag}_patch.json"), "w"))
print("done", len(H), "단단함", sum(not m["soft"] for m in META), flush=True)
