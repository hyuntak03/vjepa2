#!/usr/bin/env python3
"""[T] 가까운 음성 대조 이어 학습 준비 (학습 없음): 자연 영상 학습 캐시의 "±1/255 로 다시 인코딩한 판" (auto_research, 2026-10-22 KST 18:3x).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/nce_reencode.py [--draws 2] [--limit N]

입력: /workspace/auto_research_cache/natural_train/{k710_320, ht100m_1pct, ssv2}_{h.npy, meta.json} (기존 이어 학습 팔이 쓴 캐시, 창 300 개 × 블록 8).
읽기 · 크기 · 반올림 · 인코딩은 cache_natural.py 와 같다 (meta 의 경로 · 시작 장 · 간격, 가운데 정사각 → 256 bilinear → uint8 반올림,
목표 encoder ViT-H bf16, AR.encode_blocks). 다시 인코딩 = 장마다 독립 ±1/255 픽셀 잡음 ([비판] crit_pmi_fresh.py 와 같은 식) 을 더해 인코딩.
판마다 다른 잡음 (draw d = 0..D−1, seed 고정). 출력: /workspace/auto_research_cache/natural_train_reenc/{src}_hn{d}.npy (fp16, 같은 모양 · 같은 행 순서).
검산: 창마다 깨끗한 재인코딩 vs 캐시 상대 차 (앞 10 창), 잡음 판 vs 캐시 상대 변화 (전부)."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--out", default="/workspace/auto_research_cache/natural_train_reenc")
ap.add_argument("--srcs", default="k710_320,ht100m_1pct,ssv2")
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--nb", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
NF = 2 * a.nb
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def encode(x):
    xx = (x / 255).permute(1, 0, 2, 3)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(a.nb, 256, -1).float()


summary = {}
for src in a.srcs.split(","):
    H = np.load(os.path.join(a.cache, f"{src}_h.npy"), mmap_mode="r"); meta = json.load(open(os.path.join(a.cache, f"{src}_meta.json")))
    N = len(meta) if not a.limit else min(a.limit, len(meta))
    outs = [np.lib.format.open_memmap(os.path.join(a.out, f"{src}_hn{d}.npy"), mode="w+", dtype=np.float16, shape=(N, a.nb, 256, 1280)) for d in range(a.draws)]
    gens = [torch.Generator(device=dev).manual_seed(1000 + 17 * d) for d in range(a.draws)]
    recon, chg = [], []
    t0 = time.time()
    for i in range(N):
        m = meta[i]
        vr = decord.VideoReader(m["path"], num_threads=1)
        fr = vr.get_batch(list(range(m["start"], m["start"] + (NF - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
        Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
        x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        h = torch.from_numpy(np.asarray(H[i], np.float32)).to(dev)
        if i < 10:
            hc = encode(x); recon.append(float((hc - h).abs().mean() / h.abs().mean()))
        for d in range(a.draws):
            xn = (x + torch.randint(0, 2, x.shape, device=dev, generator=gens[d]).float() * 2 - 1).clamp(0, 255)
            hn = encode(xn)
            outs[d][i] = hn.half().cpu().numpy()
            chg.append(float((hn - h).abs().mean() / h.abs().mean()))
        if (i + 1) % 50 == 0:
            print(f"  {src} {i + 1}/{N} ({time.time() - t0:.0f}s) 깨끗한 재인코딩 상대 차 {np.mean(recon):.4f} · 잡음 판 상대 변화 {np.mean(chg):.3f}", flush=True)
    for o in outs:
        o.flush()
    summary[src] = dict(n=N, recon_rel=float(np.mean(recon)), noise_rel=float(np.mean(chg)))
    print(f"[{src}] 창 {N} · 깨끗한 재인코딩 vs 캐시 상대 차 {np.mean(recon):.4f} · ±1/255 판 상대 변화 {np.mean(chg):.3f} ({time.time() - t0:.0f}s)", flush=True)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print("done", flush=True)
