#!/usr/bin/env python3
"""[T] 도메인 문 진단용: 반사실 test 의 외형만 자연 영상 통계로 바꾼 판 → 블록 토큰 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/cf_style.py [--limit N]

장면마다 자연 영상 한 장 (K710 보류 영상, seed 고정) 을 고르고, 그 장의 채널별 평균 · 표준편차로 장면의 모든 장 (네 변형, 갈림 전 · 뒤) 을
같은 선형 변환으로 바꾼다: x' = (x − μ_장면) / σ_장면 · σ_자연 + μ_자연 (μ, σ 는 장면 W_pos 첫 장 기준). 장면 안에서는 한 변환이라
짝 안 갈림 전 문맥이 비트 단위로 같게 남는다. 질감 · 모양은 그대로다 (색 통계만 바꿈 — 단서).
인코딩은 cf_encode.py 와 같다 (PNG → 장 0,3,…,93 → 블록 16 → 목표 encoder bf16 → LN → fp16).
출력: /workspace/auto_research_cache/phys3d_cf_v1/test_style_{h.npy,meta.json} (행 순서 = test_meta 와 같음)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cf", default="/workspace/auto_research_cache/phys3d_cf_v1")
ap.add_argument("--ref_meta", default="/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
import decord                                                               # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, NB = 256, 16
IDX = list(range(0, 96, 3))
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CKE)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
M = json.load(open(os.path.join(a.cf, "test_meta.json")))
REF = json.load(open(a.ref_meta))
rng = np.random.default_rng(0)


def ref_stats(k):
    m = REF[int(rng.integers(len(REF)))]
    vr = decord.VideoReader(m["path"], num_threads=1)
    fr = vr[m["start"]].asnumpy().astype(np.float32)
    Hh, Ww = fr.shape[:2]; s = min(Hh, Ww); y0, x0 = (Hh - s) // 2, (Ww - s) // 2
    fr = fr[y0:y0 + s, x0:x0 + s].reshape(-1, 3)
    return fr.mean(0), fr.std(0) + 1e-3


def frames(stem):
    fs = sorted(glob.glob(stem + "_frames/f*.png"))
    return np.stack([np.asarray(Image.open(fs[t]).convert("RGB"), np.float32) for t in IDX])


by = {}
for m in M:
    by.setdefault(m["scene"], {})[m["variant"]] = m
rows = M[: (4 * a.limit) or None]
H = np.lib.format.open_memmap(os.path.join(a.cf, "test_style_h.npy"), mode="w+", dtype=np.float16, shape=(len(rows), NB, S, 1280))
tfm = {}
t0 = time.time()
for i, m in enumerate(rows):
    sc = m["scene"]
    if sc not in tfm:
        src = frames(by[sc]["W_pos"]["path"][:-4])[0].reshape(-1, 3)
        mu_r, sd_r = ref_stats(sc)
        tfm[sc] = (src.mean(0), src.std(0) + 1e-3, mu_r, sd_r)
    mu_s, sd_s, mu_r, sd_r = tfm[sc]
    x = (frames(m["path"][:-4]) - mu_s) / sd_s * sd_r + mu_r
    x = torch.from_numpy(np.clip(np.round(x), 0, 255)).permute(0, 3, 1, 2).float() / 255.0
    x = ((x - MEAN) / STD).view(NB, 2, 3, 256, 256).permute(0, 2, 1, 3, 4).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(x[j:j + 8]) for j in range(0, NB, 8)], 0).float()
    H[i] = F.layer_norm(h, (1280,)).half().cpu().numpy()
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
H.flush()
json.dump([dict(r, row=j) for j, r in enumerate(rows)], open(os.path.join(a.cf, "test_style_meta.json"), "w"), indent=1)
print(f"[cf_style] {len(rows)} clip · 장면 {len(tfm)} · {time.time() - t0:.0f}s", flush=True)
