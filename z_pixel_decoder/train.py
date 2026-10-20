#!/usr/bin/env python3
"""Pixel decoder 학습 (2026-10-20). VOBench_v1 프레임 → 고정 ViT-H target_encoder (블록 = 2 장, LN) → decoder → 같은 2 장.

  torchrun --nproc-per-node=8 z_pixel_decoder/train.py --config z_pixel_decoder/configs/vobench_vits.yaml

블록 = 채점과 같은 간격: clip 의 프레임 (s, s+stride) — s 는 0..95−stride 무작위. 256² 로 bilinear 리사이즈, ImageNet 정규화.
split: source_block 단위 해시 (val_frac) — 같은 block 의 4 변이가 train / val 에 갈리지 않는다.
산출: runs/<name>/ — latest.pt (decoder 만) · log.csv · samples/step_N.png (val: 원본 위 · 복원 아래)."""
import argparse, csv, hashlib, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F, torch.distributed as dist, yaml
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from model import PixelDecoder                                                          # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
D, M, O = cfg["data"], cfg["model"], cfg["optim"]
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, ws = 0, 1
dev = torch.device("cuda")
out = os.path.join(HERE, "runs", cfg["name"]); os.makedirs(os.path.join(out, "samples"), exist_ok=True)
torch.manual_seed(cfg.get("seed", 0) + rank); random.seed(cfg.get("seed", 0) + rank)

# ---- 데이터
rows = list(csv.DictReader(open(os.path.join(REPO, D["index"]))))
isval = lambda r: int(hashlib.md5(r["source_block"].encode()).hexdigest(), 16) % 1000 < 1000 * D["val_frac"]
train_dirs = [os.path.join(D["root"], r["file_name"]) for r in rows if not isval(r)]
val_dirs = sorted(os.path.join(D["root"], r["file_name"]) for r in rows if isval(r))
S, STRIDE, NF = 256, int(D["stride"]), int(D.get("n_frames", 96))


def read_block(d, s0):
    """(3, 2, S, S) uint8. 깨진 PNG 가 있다 (VOBench realistic/permanence/early_k2/assets_flat_acc_tennis_ball/pos_a/000008.png, 10-20 확인) → None"""
    try:
        fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB").resize((S, S), Image.BILINEAR)) for i in (s0, s0 + STRIDE)]
    except Exception:
        return None
    return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2)


def sample_block(dirs, rng):
    while True:
        x = read_block(rng.choice(dirs), rng.randrange(0, NF - STRIDE))
        if x is not None:
            return x


class Blocks(torch.utils.data.IterableDataset):
    def __init__(self, dirs, seed):
        self.dirs, self.seed = dirs, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            yield sample_block(self.dirs, rng)                                  # (3, 2, S, S) uint8


def loader(dirs, seed, bs):
    return torch.utils.data.DataLoader(Blocks(dirs, seed), batch_size=bs, num_workers=int(D.get("workers", 8)), pin_memory=True,
                                       persistent_workers=True, prefetch_factor=4)


MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
norm = lambda x: (x.to(dev, non_blocking=True).float() / 255.0 - MEAN) / STD
denorm = lambda x: ((x * STD + MEAN).clamp(0, 1))

# ---- 모델
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(os.path.join(REPO, M["checkpoint"]))["target_encoder"], tag="t")
enc = enc.to(dev).eval().requires_grad_(False)
dec = PixelDecoder(in_dim=1280, dim=M["dim"], depth=M["depth"], heads=M["heads"]).to(dev)
n_par = sum(p.numel() for p in dec.parameters())
decw = torch.nn.parallel.DistributedDataParallel(dec, device_ids=[dev.index]) if ddp else dec
opt = torch.optim.AdamW(dec.parameters(), lr=O["lr"], weight_decay=O["weight_decay"], betas=(0.9, 0.95))
steps, warm = int(O["steps"]), int(O["warmup"])
lr_at = lambda s: O["lr"] * min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, steps - warm)))) * 0.9 + 0.1)
start = 0
ck = os.path.join(out, "latest.pt")
if os.path.isfile(ck):
    st = torch.load(ck, map_location="cpu", weights_only=False)
    dec.load_state_dict(st["decoder"]); opt.load_state_dict(st["opt"]); start = st["step"] + 1


@torch.no_grad()
def latent(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = enc(x).float()
    return F.layer_norm(z, (z.size(-1),))                                     # 학습 AR 정답과 같은 LN (affine 없음)


LPW = float(O.get("lpips", 0.0))
if LPW:
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
to01 = lambda x: x * STD + MEAN                                               # 정규화 → [0, 1] (clamp 없음: gradient 유지)


def loss_fn(y, x):
    """y, x 정규화 공간 (B, 3, 2, S, S). l1 = [0, 1] 픽셀 L1 (+ lpips × LPIPS(VGG), 블록의 2 장 각각) · mse = 정규화 공간 MSE"""
    if O.get("loss", "mse") == "mse":
        return F.mse_loss(y, x), {}
    y1, x1 = to01(y), to01(x)
    l1 = F.l1_loss(y1, x1)
    if not LPW:
        return l1, {"l1": l1.item()}
    fl = lambda t: t.permute(0, 2, 1, 3, 4).reshape(-1, 3, t.size(-2), t.size(-1)) * 2 - 1     # (B·2, 3, S, S), [-1, 1]
    lpv = lp(fl(y1), fl(x1)).mean()
    return l1 + LPW * lpv, {"l1": l1.item(), "lpips": lpv.item()}


@torch.no_grad()
def evaluate(step):
    dec.eval(); rng = random.Random(123); mse, psnr = [], []
    for i in range(0, int(cfg.get("n_val", 128)), 16):
        xs = [sample_block(val_dirs, rng) for _ in range(16)]
        x = norm(torch.stack(xs)); y = dec(latent(x))
        mse.append(F.mse_loss(y, x).item())
        e = ((denorm(y) - denorm(x)) ** 2).mean(dim=(1, 2, 3, 4)); psnr += (-10 * torch.log10(e)).tolist()
        if i == 0 and rank == 0:     # 그림: 위 = 원본 (각 블록 첫 장), 아래 = 복원
            top = torch.cat(list(denorm(x)[:8, :, 0]), -1); bot = torch.cat(list(denorm(y)[:8, :, 0]), -1)
            g = (torch.cat([top, bot], -2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(g).save(os.path.join(out, "samples", f"step_{step:06d}.png"))
    dec.train()
    return float(np.mean(mse)), float(np.mean(psnr))


if rank == 0:
    print(f"pixel decoder {n_par / 1e6:.1f}M 파라미터 · train clip {len(train_dirs)} · val clip {len(val_dirs)} · "
          f"{steps} step × {O['batch_size']}/GPU × {ws} GPU · lr {O['lr']} · 손실 {O.get('loss', 'mse')} + LPIPS ×{LPW:g} · 시작 step {start}", flush=True)
    logf = open(os.path.join(out, "log.csv"), "a"); logf.write("step,loss,lr,val_mse,val_psnr,sec\n") if start == 0 else None
it = iter(loader(train_dirs, cfg.get("seed", 0) * 100 + rank, int(O["batch_size"])))
t0 = time.time()
for step in range(start, steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = norm(next(it))
    z = latent(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = decw(z)
    loss, parts = loss_fn(y.float(), x)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), O.get("clip", 1.0)); opt.step()
    if rank == 0 and (step % 50 == 0):
        print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f" lr {lr_at(step):.2e} ({time.time() - t0:.0f}s)", flush=True)
    if (step + 1) % int(cfg.get("eval_every", 1000)) == 0 or step + 1 == steps:
        vm, vp = evaluate(step + 1) if rank == 0 else (0, 0)
        if rank == 0:
            print(f"[val step {step + 1}] mse {vm:.4f} · PSNR {vp:.2f} dB", flush=True)
            logf.write(f"{step + 1},{loss.item():.5f},{lr_at(step):.3e},{vm:.5f},{vp:.3f},{time.time() - t0:.0f}\n"); logf.flush()
            torch.save({"decoder": dec.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, ck)
        if ddp:
            dist.barrier()
if ddp:
    dist.destroy_process_group()
