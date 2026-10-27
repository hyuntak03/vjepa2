"""릴리즈 predictor 예측 · 정답 (32 장 창 토큰) 을 블록마다 전체 평균 (1, 1280) 으로 (2026-10-26, 학습 없음).
사용자: "(AR 의 정체 개념 cos · 크기 비 그림) release 버전도 하나". AR 캐시 (ar_layers) 와 같은 클립 목록 · 라벨을 쓴다.

클립: /workspace/auto_research_cache/ar_layers/{train,val}_meta.json 의 클립 (3D 합성 96 장, 원본 0, 3, …, 93 장 = 32 장).
  pred  (8, 1280)   릴리즈 predictor: context_encoder (문맥 16 장) → predictor (미래 8 블록 한 번에). 블록마다 칸 256 평균
  gt    (16, 1280)  정답 = LN(target_encoder(32 장 한 번에)) 의 블록별 칸 평균 (릴리즈 채점 surprise_c16t32 의 타깃 공간)
fp16 autocast. GPU 메모리 상한 MEM_GB (기본 9).

  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((k+1)) MEM_GB=9 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_pooled_cache.py --shard $k --nshard 4 & done; wait
  /venv/vjepa/bin/python z_research/scripts/analysis/release_pooled_cache.py --merge --nshard 4
출력: /workspace/auto_research_cache/release_pooled/{train,val}.npz (pred · gt · i)
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/release_pooled"; os.makedirs(OUT, exist_ok=True)
IDX = [3 * k for k in range(32)]; T, C, S, D = 32, 16, 256, 1280
if a.merge:
    for split in ("val", "train"):
        parts = [np.load(f"{OUT}/{split}_shard{k}.npz") for k in range(a.nshard) if os.path.exists(f"{OUT}/{split}_shard{k}.npz")]
        i = np.concatenate([p["i"] for p in parts]); o = np.argsort(i)
        np.savez(f"{OUT}/{split}.npz", pred=np.concatenate([p["pred"] for p in parts])[o], gt=np.concatenate([p["gt"] for p in parts])[o], i=i[o])
        print(split, len(i))
    sys.exit(0)

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 9)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
import decord
from analysis.intphys2.model import build_from_config
from evals.world_model_analysis.eval import _context_target_indices
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
        "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
        "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
rel = build_from_config(MCFG, dev)
rel.context_encoder.half(); rel.target_encoder.half()                     # 메모리 상한 안에 넣으려고 (autocast fp16 과 같은 정밀도로 돈다)
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
for split in ("val", "train"):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))
    mine = [m for n, m in enumerate(meta) if n % a.nshard == a.shard]
    P, G, I = [], [], []; t0 = time.time()
    for n, m in enumerate(mine):
        fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
        xn = ((torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = rel.context_encoder(xn.half(), masks=[ci]); p = rel.predictor(z, ci, ti, mask_index=0); h = rel.target_encoder(xn.half())
        P.append(p.float()[0].view(8, S, D).mean(1).half().cpu().numpy())
        G.append(F.layer_norm(h.float(), (D,))[0].view(16, S, D).mean(1).half().cpu().numpy()); I.append(m["i"])
        if n % 200 == 0:
            print(time.strftime("%H:%M:%S"), split, f"shard {a.shard}: {n}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/{split}_shard{a.shard}.npz", pred=np.stack(P), gt=np.stack(G), i=np.array(I))
print("done shard", a.shard, flush=True)
