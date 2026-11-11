#!/usr/bin/env python3
"""negimp.region=target 검사 (2026-11-08, GPU 1 장 · 학습 없음): target 영역 마스크가 맞는 자리를 잡는지, 그리고 가능 미래 vs 불가능 미래의
차이가 그 영역에 몰려 있는지 — 화소 (16×16 칸 · 2 장 블록) 와 latent (블록별 target encoder LN 토큰, 학습 입력 · 정답과 같은 공간) 둘 다.

  /venv/vjepa/bin/python z_research/scripts/analysis/negimp_region_check.py <resolved config.yaml> [--n 48]
출력: 종류 (vanish · appear · swap) 별 영역 크기 (토큰 비율), 차이 질량 중 영역 안 비율, 영역 안 / 밖 토큰 평균 차이.
"""
import argparse, collections, os, random, sys
import numpy as np, torch, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
import decord
from app.vjepa_frozen import utils as U
from app.vjepa_frozen.ar import encode_blocks
from app.vjepa_frozen.data import Mp4ListDataset, normalize_clips

ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("--n", type=int, default=48); a = ap.parse_args()
cfg = yaml.safe_load(open(a.cfg)); MD, NEG = cfg["model"], cfg["negimp"]
spec = dict(cfg["data"]["datasets"][0]); spec["negimp"] = {k: NEG[k] for k in ("families", "p", "p_object_ctx", "swap_frac", "context_frames", "region") if k in NEG}
class _T:   # 반전 · 자르기 없음 (영상은 256 정사각형이라 center square crop 은 항등) — uint8 (3,T,H,W)
    last_flip = False
    def __call__(self, x): return x.permute(3, 0, 1, 2).contiguous()
ds = Mp4ListDataset(spec, 32, _T())
dev = torch.device("cuda")
state = U.load_base_checkpoint(MD["checkpoint"])
_, tgt = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, 32)
rng = random.Random(0); random.seed(0)
names = {1: "vanish", 2: "appear", 3: "swap"}
acc = collections.defaultdict(lambda: collections.defaultdict(list))
got = collections.Counter(); tries = 0
while sum(got.values()) < a.n and tries < 20000:
    tries += 1
    i = rng.randrange(len(ds.paths)); code, fut = ds.neg.draw(i)
    if not code or got[code] >= a.n // 3: continue
    got[code] += 1
    idx = np.arange(rng.choice(ds.starts), 64, 2)[:32]
    pos = torch.from_numpy(decord.VideoReader(ds.paths[i], num_threads=2).get_batch(list(idx)).asnumpy()).permute(3, 0, 1, 2)
    fimp = torch.from_numpy(decord.VideoReader(ds.paths[fut], num_threads=2).get_batch(list(idx[16:])).asnumpy()).permute(3, 0, 1, 2)
    imp = torch.cat([pos[:, :16], fimp], 1)
    reg = ds._target_region([i, fut], idx)[8:].float().reshape(-1)              # 미래 8 블록 × 256
    # 화소: 16×16 칸 · 2 장 블록 평균 |차|
    dpx = (pos[:, 16:].float() - imp[:, 16:].float()).abs().mean(0)               # (16, 256, 256)
    dpx = dpx.reshape(8, 2, 16, 16, 16, 16).mean(dim=(1, 3, 5)).reshape(-1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = encode_blocks(tgt, normalize_clips(torch.stack([pos[:, 16:], imp[:, 16:]]).to(dev)).to(torch.bfloat16), 2).float()
    dlt = (h[0] - h[1]).abs().mean(-1).cpu()                                       # (8*256,)
    for nm, d in (("화소", dpx), ("latent", dlt)):
        inside = (d * reg).sum() / d.sum().clamp(min=1e-9)
        acc[names[code]][nm + " 차이 질량 중 영역 안"].append(float(inside))
        if reg.sum() > 0 and (1 - reg).sum() > 0:
            acc[names[code]][nm + " 영역 안 / 밖 평균 차이"].append(float((d * reg).sum() / reg.sum() / ((d * (1 - reg)).sum() / (1 - reg).sum()).clamp(min=1e-9)))
    acc[names[code]]["영역 크기 (미래 토큰 비율)"].append(float(reg.mean()))
    # 2026-11-09 시간 (블록) 단위: 미래 블록 중 target 이 (가능 ∪ 불가능 미래에서) 보이는 블록 vs 안 보이는 블록의 latent 차이
    vb = reg.reshape(8, 256).amax(1) > 0                                           # (8,) target 보이는 블록
    db = dlt.reshape(8, 256).mean(1)
    acc[names[code]]["보이는 블록 비율"].append(float(vb.float().mean()))
    if vb.any() and (~vb).any():
        acc[names[code]]["latent 블록 차이: 보이는 / 안 보이는"].append(float(db[vb].mean() / db[~vb].mean()))
    acc[names[code]]["latent 차이 질량 중 보이는 블록"].append(float(db[vb].sum() / db.sum()))
    if (~vb).any():
        acc[names[code]]["latent 토큰 |차| 절대값: 안 보이는 블록"].append(float(db[~vb].mean()))
    if vb.any():
        acc[names[code]]["latent 토큰 |차| 절대값: 보이는 블록"].append(float(db[vb].mean()))
for k, d in acc.items():
    print(f"== {k} (n {got[[c for c, v in names.items() if v == k][0]]})")
    for m, v in d.items():
        v = np.array(v); print(f"   {m}: 중앙 {np.median(v):.3f} · 10–90 % {np.percentile(v, 10):.3f}–{np.percentile(v, 90):.3f}")
