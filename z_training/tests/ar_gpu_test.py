"""실제 ViT-H 로 AR 학습 한 step 의 VRAM·시간 (GPU 1장). 
  python z_training/harness/resolve_train.py configs/training/natural_ar.yaml -o /tmp/ar_cfg.yaml --folder /tmp/ar_run
  CUDA_VISIBLE_DEVICES=0 python tmp/.../tests/ar_gpu_test.py /tmp/ar_cfg.yaml [compile]
2026-09-26 실측 (A6000 48G, B=28, checkpointing 켬): C=16 K=8 R=8 → 31 G, 20 s (enc 4.6 + pred/bwd 16)."""
import sys, time, yaml, torch
sys.path.insert(0, '.')
from app.vjepa_frozen import utils as U
from app.vjepa_frozen import ar as AR
cfg = yaml.safe_load(open(sys.argv[1]))
MD = cfg["model"]; MD.setdefault("use_sdpa", True)
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
state = U.load_base_checkpoint(MD["checkpoint"])
ctx, tgt = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, 48)
print("shared encoder:", ctx is tgt)
compile_ = len(sys.argv) > 2 and sys.argv[2] == "compile"
if compile_:
    tgt = torch.compile(tgt, mode="default", dynamic=True)
P = U.build_predictor(MD, 1280 if compile_ else tgt.embed_dim, None, dev, 48)
del state
opt = torch.optim.AdamW(P.parameters(), lr=1e-4)
B = int(cfg["data"]["batch_size"]); S = 256
def step(C, K, R):
    clips = torch.randn(B, 3, (C+K)*2, 256, 256, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, clips.to(torch.bfloat16), 2)
    torch.cuda.synchronize(); t1 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, jl, sl, R_ = AR.ar_losses(P, h, S, C, K, 1.0, R)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t2 = time.time()
    print(f"C={C:2d} K={K} R={R_} B={B} | enc {t1-t0:.2f}s pred+bwd {t2-t1:.2f}s total {t2-t0:.2f}s | "
          f"loss {loss.item():.4f} (tf {jl.item():.4f} ar {sl.item():.4f}) | max alloc {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    torch.cuda.reset_peak_memory_stats()
for (C, K) in [(16, 8), (16, 8), (8, 8), (2, 8), (16, 1), (4, 4)]:
    step(C, K, int(cfg["loss"]["rollout_steps"]))
