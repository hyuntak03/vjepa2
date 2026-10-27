"""물체 칸 평균 캐시 — encoder (정답) 와 예측 (AR 굴린 출력 / 릴리즈 predictor) 의 블록 8–15 토큰을 물체 칸에서 평균 (2026-10-26).
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘. predictor 는 그게 안 될 것 같은데 일단 해봐".
클립: ar_layers/{train,val}_meta.json (train 8,000 · val 350, 96 장 3D 합성). 물체 칸 = meta 의 블록별 cells (16×16 칸 중 물체 상자 안에 중심이 든 칸; 비면 물체 중심 칸 하나).
블록 b = 8 … 15 (예측 걸음 k = b − 7) 마다 (1280) 벡터 셋, 없으면 NaN:
  true  = 블록 b 의 진짜 물체 칸 평균 (블록 b 에 물체 보일 때)
  last  = 블록 7 (문맥 마지막) 의 물체 칸 평균 (블록 7 에 물체 보일 때) — 예측이 물체를 마지막에 본 자리에 두는지
  whole = 화면 전체 평균
  --model ar      : CPU. 전체 토큰 캐시 (ar_layers) 의 enc (블록 단위 LN(target_encoder)) · ro_out (굴린 예측 출력)
  --model release : GPU. 릴리즈 predictor (문맥 16 장 → 미래 8 블록 한 번에) 를 다시 돌린다. 정답 = 32 장 한 번에 LN(target_encoder). fp16, MEM_GB 상한
출력: /workspace/auto_research_cache/objcell_concept/{model}_{split}[_shard{k}].npz — tgt_true · tgt_last · tgt_whole · pred_true · pred_last · pred_whole (N, 8, 1280) fp16 · i
  /venv/vjepa/bin/python z_research/scripts/analysis/objcell_concept_cache.py --model ar
  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((k+1)) MEM_GB=10 nohup /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/objcell_concept_cache.py --model release --shard $k --nshard 4 & done
  /venv/vjepa/bin/python z_research/scripts/analysis/objcell_concept_cache.py --model release --merge --nshard 4
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["ar", "release"], required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/objcell_concept"; os.makedirs(OUT, exist_ok=True)
KEYS = ("tgt_true", "tgt_last", "tgt_whole", "pred_true", "pred_last", "pred_whole")


def cells_of(bl):
    return bl["cells"] or [int(min(15, max(0, bl["ctr"][1] // 16)) * 16 + min(15, max(0, bl["ctr"][0] // 16)))]


def reduce(T, P, m):
    """T: 정답 블록 토큰 (16, 256, 1280) · P: 예측 (8, 256, 1280, 블록 8–15) → 블록 8–15 마다 true · last · whole."""
    out = {k: np.full((8, 1280), np.nan, np.float32) for k in KEYS}
    last = cells_of(m["blocks"][7]) if m["blocks"][7]["ok"] else None
    for j in range(8):
        b = 8 + j; bl = m["blocks"][b]
        out["tgt_whole"][j] = T[b].mean(0); out["pred_whole"][j] = P[j].mean(0)
        if bl["ok"]:
            c = cells_of(bl); out["tgt_true"][j] = T[b, c].mean(0); out["pred_true"][j] = P[j, c].mean(0)
        if last is not None:
            out["tgt_last"][j] = T[b, last].mean(0); out["pred_last"][j] = P[j, last].mean(0)
    return out


def save(R, I, path):
    np.savez(path, **{k: np.stack([r[k] for r in R]).astype(np.float16) for k in KEYS}, i=np.array(I))


if a.merge:
    for split in ("val", "train"):
        parts = [np.load(f"{OUT}/{a.model}_{split}_shard{k}.npz") for k in range(a.nshard)]
        i = np.concatenate([p["i"] for p in parts]); o = np.argsort(i)
        np.savez(f"{OUT}/{a.model}_{split}.npz", **{k: np.concatenate([p[k] for p in parts])[o] for k in KEYS}, i=i[o]); print(split, len(i))
    sys.exit(0)

if a.model == "ar":
    from multiprocessing import Pool

    def one(arg):
        split, m = arg
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); T = z["enc"].astype(np.float32); P = z["ro_out"].astype(np.float32); z.close()
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return reduce(T, P, m), m["i"]

    for split in ("val", "train"):
        meta = json.load(open(f"{SRC}/{split}_meta.json")); t0 = time.time()
        with Pool(6) as pool:
            rs = list(pool.imap(one, [(split, m) for m in meta], chunksize=4))
        save([r[0] for r in rs], [r[1] for r in rs], f"{OUT}/ar_{split}.npz"); print(split, len(rs), f"{time.time() - t0:.0f}s", flush=True)
    sys.exit(0)

import torch, torch.nn.functional as F, decord
sys.path.insert(0, REPO)
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 10)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from analysis.intphys2.model import build_from_config
from evals.world_model_analysis.eval import _context_target_indices
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False, "dual_encoder": True,
        "context_encoder_key": "encoder", "target_encoder_key": "target_encoder", "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10},
        "dtype": "float32", "autocast": "float16", "window_size": 32}
rel = build_from_config(MCFG, dev); rel.context_encoder.half(); rel.target_encoder.half()
ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=256, batch_size=1, device=dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
IDX = [3 * k for k in range(32)]
for split in ("val", "train"):
    meta = json.load(open(f"{SRC}/{split}_meta.json")); mine = [m for n, m in enumerate(meta) if n % a.nshard == a.shard]; R, I = [], []; t0 = time.time()
    for n, m in enumerate(mine):
        fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
        xn = ((torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = rel.context_encoder(xn.half(), masks=[ci]); p = rel.predictor(z, ci, ti, mask_index=0); h = rel.target_encoder(xn.half())
        P = p.float()[0].view(8, 256, 1280).cpu().numpy(); T = F.layer_norm(h.float(), (1280,))[0].view(16, 256, 1280).cpu().numpy()
        R.append(reduce(T, P, m)); I.append(m["i"])
        if n % 200 == 0:
            print(time.strftime("%H:%M:%S"), split, f"shard {a.shard}: {n}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    save(R, I, f"{OUT}/release_{split}_shard{a.shard}.npz")
print("done shard", a.shard, flush=True)
