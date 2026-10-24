#!/usr/bin/env python3
"""[T] 자연 영상 블록 토큰 + 칸 단위 광학 흐름 캐시 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_natural.py --src ssv2 --n 200 --nb 10

학습 로더와 같은 읽기: 4 fps 간격 (표의 간격 열), 높이 전체 가운데 정사각 → 256 bilinear → uint8 반올림.
블록 토큰 = 블록별 LN(target_encoder), bf16 (학습 정답과 같다). 흐름 = RAFT-large, 블록 대표 장 (둘째 장) 사이 역방향, 16 px 칸 평균 (칸 단위).
저장: /workspace/auto_research_cache/natural/{src}_h.npy (N, nb, 256, 1280) fp16 · {src}_U.npy (N, nb, 16, 16, 2) fp16 (U[:,0] = 0) · {src}_meta.json"""
import argparse, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, choices=["k710_320", "ssv2", "ht100m_1pct"])
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--nb", type=int, default=16, help="창의 블록 수 (장 수 = 2·nb)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="/workspace/auto_research_cache/natural")
ap.add_argument("--flow", type=int, default=1, help="0 = 흐름 계산 안 함 (학습용 캐시)")
ap.add_argument("--table", default="", help="다른 영상 목록 (csv: 경로 · 라벨, 예: K710 보류 val_min224.csv). 4 fps 간격 = round(fps / (30/7))")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights               # noqa: E402
dev = torch.device("cuda")
torch.set_num_threads(8)
NF = 2 * a.nb
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval() if a.flow else None
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
if a.table:
    rows = [[l.split()[0], "0", "0", "0", "999"] for l in open(a.table) if l.strip()]
else:
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv", a.src, "fps4_table.tsv"))]
    rows = [r for r in rows if int(r[4]) >= NF]
rng = random.Random(a.seed); rng.shuffle(rows)
print(f"{a.src}: 적격 영상 {len(rows)} (4 fps 장 ≥ {NF})", flush=True)
H, UU, META = [], [], []
for r in rows:
    if len(H) >= a.n:
        break
    try:
        vr = decord.VideoReader(r[0], num_threads=1)   # webm (SSv2) 은 다중 스레드 디코더가 실패한다
        fstp = int(r[3]) if not a.table else max(1, int(round(vr.get_avg_fps() / (30 / 7))))
        span = (NF - 1) * fstp + 1
        if len(vr) < span:
            continue
        s = rng.randint(0, len(vr) - span)
        fr = vr.get_batch(list(range(s, s + span, fstp))).asnumpy()
    except Exception:
        continue
    Hh, Ww = fr.shape[1:3]
    side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()       # (NF,3,256,256)
    with torch.no_grad():
        xx = (x / 255).permute(1, 0, 2, 3)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(a.nb, 256, -1)
        U = torch.zeros(a.nb, 16, 16, 2, device=dev)
        if a.flow:
            rep = x[1::2] / 127.5 - 1
            f = raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]
            U[1:] = (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1)
    H.append(h.half().cpu().numpy()); UU.append(U.half().cpu().numpy()); META.append(dict(path=r[0], start=s, fstp=fstp))
    if len(H) % 25 == 0:
        print(f"{len(H)}/{a.n}", flush=True)
os.makedirs(a.out, exist_ok=True)
tag = a.src + ("_holdout" if a.table else "")
np.save(os.path.join(a.out, f"{tag}_h.npy"), np.stack(H)); np.save(os.path.join(a.out, f"{tag}_U.npy"), np.stack(UU))
json.dump(META, open(os.path.join(a.out, f"{tag}_meta.json"), "w"))
print("done", len(H), flush=True)
