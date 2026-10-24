#!/usr/bin/env python3
"""Pixel decoder 학습 (2026-10-20). VOBench 프레임 → 고정 ViT-H target_encoder (LN) → decoder → 블록 (2 장) 픽셀.

  torchrun --nproc-per-node=8 z_pixel_decoder/train.py --config z_pixel_decoder/configs/vobench_vits.yaml

clip = 채점과 같은 간격의 clip_frames 장 (o, o+stride, …), o 무작위 → 블록 clip_frames/2 개. 256² bilinear, ImageNet 정규화.
latent: block (블록 따로 인코딩) | window (clip 한 번에) | both (둘 다 한 batch 에). batch_size = GPU 당 clip 수.
split: source_block 단위 해시 (val_frac) — 같은 block 의 4 변이가 train / val 에 갈리지 않는다.
산출: runs/<name>/ — latest.pt (decoder 만) · log.csv · samples/step_N.png (val: 원본 위 · 복원 아래)."""
import argparse, csv, hashlib, math, os, random, re, sys, time
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

# ---- 데이터 (2026-10-20: sources 여러 개 · latent = block | window | both)
#   block  : 블록 (2 장) 을 따로 인코딩 — AR predictor 공간
#   window : clip_frames 장을 한 번에 인코딩해 블록 자리별로 자른다 — 릴리즈 predictor · 채점 (surprise_c16t32) 공간
#   both   : 같은 clip 을 두 방식으로 다 만들어 한 batch 에 (decoder 하나로 두 공간을 다 그린다)
SRC = D.get("sources") or [{"index": D["index"], "root": D["root"]}]
isval = lambda r: int(hashlib.md5(r["source_block"].encode()).hexdigest(), 16) % 1000 < 1000 * D["val_frac"]
# clip 항목 = (프레임 경로 틀 (frame 자리 비움), 첫 장 번호, 장 수). 소스마다 pattern · start · n_frames · split_key 를 줄 수 있다
#   기본은 VOBench 형식 ({file_name}/{frame:06d}.png, 0 부터, 96 장, source_block 으로 split)
isval = lambda key: int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < 1000 * D["val_frac"]
train_dirs, val_dirs = [], []
for src in SRC:
    pat = src.get("pattern", "{file_name}/{frame:06d}.png")
    if src.get("video"):        # 2026-10-22: mp4 소스 (IntPhys2 Main) — 항목 = ("mp4", 경로, 소스 stride, 장 수). index 는 절대경로 csv 도 된다
        for r in csv.DictReader(open(os.path.join(REPO, src["index"]))):
            item = ("mp4", os.path.join(src["root"], pat.format(**r)), int(src.get("stride", D["stride"])), int(src["n_frames"]))
            (val_dirs if isval(src["index"] + "/" + r[src.get("split_key", "source_block")]) else train_dirs).append(item)
        continue
    for r in csv.DictReader(open(os.path.join(REPO, src["index"]))):
        m_ = re.search(r"\{frame(?::([^}]*))?\}", pat)                       # {frame:06d} 자리만 비우고 나머지는 행 값으로
        tmpl = (os.path.join(src["root"], pat[:m_.start()].format(**r)), m_.group(1) or "", pat[m_.end():].format(**r))
        item = (tmpl, int(src.get("start", 0)), int(src.get("n_frames", D.get("n_frames", 96))))
        (val_dirs if isval(src.get("index") + "/" + r[src.get("split_key", "source_block")]) else train_dirs).append(item)
val_dirs.sort(key=str)          # mp4 항목과 PNG 항목이 섞여도 정렬되게 (2026-10-22)
S, STRIDE, NF = 256, int(D["stride"]), int(D.get("n_frames", 96))
CF = int(D.get("clip_frames", 32))                     # window 인코딩 길이 (채점 c16t32 와 같은 32 장)
LAT = str(D.get("latent", "block"))
assert LAT in ("block", "window", "both"), LAT


def read_clip(item, o):
    """(3, CF, S, S) uint8 — 그 clip 의 o, o+stride, … 번째 장 (CF 장). 깨진 PNG (VOBench realistic/permanence/early_k2/
    assets_flat_acc_tennis_ball/pos_a/000008.png, 10-20 확인) 가 걸리면 None"""
    if item[0] == "mp4":
        try:
            import decord
            fr = decord.VideoReader(item[1], num_threads=1).get_batch([o + item[2] * k for k in range(CF)]).asnumpy()
            fr = [np.asarray(Image.fromarray(f).resize((S, S), Image.BILINEAR)) for f in fr]
        except Exception:
            return None
        return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2)
    tmpl, st0, _ = item
    try:
        path = lambda n: tmpl[0] + format(n, tmpl[1]) + tmpl[2]
        fr = [np.asarray(Image.open(path(st0 + o + STRIDE * k)).convert("RGB").resize((S, S), Image.BILINEAR)) for k in range(CF)]
    except Exception:
        return None
    return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2)


def sample_clip(items, rng):
    while True:
        it_ = rng.choice(items)
        st_ = it_[2] if it_[0] == "mp4" else STRIDE
        x = read_clip(it_, rng.randrange(0, (it_[3] if it_[0] == "mp4" else it_[2]) - st_ * (CF - 1)))
        if x is not None:
            return x


class Clips(torch.utils.data.IterableDataset):
    def __init__(self, dirs, seed):
        self.dirs, self.seed = dirs, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            yield sample_clip(self.dirs, rng)                                   # (3, CF, S, S) uint8


def loader(dirs, seed, bs):
    return torch.utils.data.DataLoader(Clips(dirs, seed), batch_size=bs, num_workers=int(D.get("workers", 8)), pin_memory=True,
                                       persistent_workers=True, prefetch_factor=4)


MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
norm = lambda x: (x.to(dev, non_blocking=True).float() / 255.0 - MEAN) / STD
denorm = lambda x: ((x * STD + MEAN).clamp(0, 1))

# ---- 모델
_tgt = _load_checkpoint(os.path.join(REPO, M["checkpoint"]))["target_encoder"]
encs = {}
for nf in (2, CF):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, _tgt, tag="t"); encs[nf] = e.to(dev).eval().requires_grad_(False)
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
def latents(x, modes):
    """x (B, 3, CF, S, S) 정규화 → {mode: (B·nb, 256, D) LN 토큰}, 블록 정답 (B·nb, 3, 2, S, S). nb = CF / 2.
    LN 은 affine 없음 (AR 정답 · 채점 타깃과 같다)."""
    B, nb = x.size(0), CF // 2
    xb = x.view(B, 3, nb, 2, S, S).permute(0, 2, 1, 3, 4, 5).reshape(B * nb, 3, 2, S, S)
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if "block" in modes:
            z = encs[2](xb).float(); out["block"] = F.layer_norm(z, (z.size(-1),))
        if "window" in modes:
            z = encs[CF](x).float(); z = F.layer_norm(z, (z.size(-1),))
            out["window"] = z.view(B, nb, 256, -1).reshape(B * nb, 256, -1)
    return out, xb


MODES = ["block", "window"] if LAT == "both" else [LAT]


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
    """val clip n_val 개 → 모드별 블록 복원 PSNR ([0, 1]). 그림: 첫 clip 의 블록 8 개 첫 장 — 위 원본, 그 아래 모드별 복원"""
    dec.eval(); rng = random.Random(123); res = {m: [] for m in MODES}; mse = []
    for i in range(0, int(cfg.get("n_val", 32)), 4):
        x = norm(torch.stack([sample_clip(val_dirs, rng) for _ in range(4)]))
        zs, xb = latents(x, MODES)
        ys = {}
        for m in MODES:
            y = dec(zs[m]); ys[m] = y
            mse.append(F.mse_loss(y, xb).item())
            e = ((denorm(y) - denorm(xb)) ** 2).mean(dim=(1, 2, 3, 4)); res[m] += (-10 * torch.log10(e)).tolist()
        if i == 0 and rank == 0:
            pick = list(range(0, CF // 2, 2))[:8]
            rows_ = [torch.cat([denorm(xb)[j, :, 0] for j in pick], -1)] + [torch.cat([denorm(ys[m])[j, :, 0] for j in pick], -1) for m in MODES]
            g = (torch.cat(rows_, -2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(g).save(os.path.join(out, "samples", f"step_{step:06d}.png"))
    dec.train()
    return float(np.mean(mse)), {m: float(np.mean(v)) for m, v in res.items()}


if rank == 0:
    print(f"pixel decoder {n_par / 1e6:.1f}M 파라미터 · train clip {len(train_dirs)} · val clip {len(val_dirs)} · "
          f"{steps} step × {int(O['batch_size']) * CF // 2 * len(MODES)}/GPU × {ws} GPU · lr {O['lr']} · 손실 {O.get('loss', 'mse')} + LPIPS ×{LPW:g} · 시작 step {start}"
          f"   [clip {O['batch_size']}/GPU × 블록 {CF // 2} × latent {'+'.join(MODES)} · sources {[s_['index'] for s_ in SRC]}]", flush=True)
    logf = open(os.path.join(out, "log.csv"), "a"); logf.write("step,loss,lr,val_mse,val_psnr,sec\n") if start == 0 else None
it = iter(loader(train_dirs, cfg.get("seed", 0) * 100 + rank, int(O["batch_size"])))
t0 = time.time()
for step in range(start, steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = norm(next(it))                                                          # (B, 3, CF, S, S)
    zs, xb = latents(x, MODES)
    z = torch.cat([zs[m] for m in MODES]); tgt = xb.repeat(len(MODES), 1, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = decw(z)
    loss, parts = loss_fn(y.float(), tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), O.get("clip", 1.0)); opt.step()
    if rank == 0 and (step % 50 == 0):
        print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f" lr {lr_at(step):.2e} ({time.time() - t0:.0f}s)", flush=True)
    if (step + 1) % int(cfg.get("eval_every", 1000)) == 0 or step + 1 == steps:
        vm, vps = evaluate(step + 1) if rank == 0 else (0, {})
        if rank == 0:
            vp = float(np.mean(list(vps.values())))
            print(f"[val step {step + 1}] mse {vm:.4f} · PSNR {vp:.2f} dB   " + " · ".join(f"{m} {v:.2f}" for m, v in vps.items()), flush=True)
            logf.write(f"{step + 1},{loss.item():.5f},{lr_at(step):.3e},{vm:.5f},{vp:.3f},{time.time() - t0:.0f}\n"); logf.flush()
            torch.save({"decoder": dec.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, ck)
        if ddp:
            dist.barrier()
if ddp:
    dist.destroy_process_group()
