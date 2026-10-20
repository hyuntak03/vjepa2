#!/usr/bin/env python3
"""릴리즈 V-JEPA 2 predictor 가 만든 VOBench 미래를 pixel decoder 로 그림 (2026-10-20, 사용자 지시).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_predictor.py --set late --n 12
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_predictor.py --set gravity --n 12

채점 (surprise_c16t32) 과 같은 입력: 32 장 (원본 0, 3, …, 93 장), 문맥 16 장 (블록 8) → 미래 16 장 (블록 8), fp16 autocast.
decoder 는 **블록별** LN(target_encoder) 로 학습했다. 릴리즈 predictor 는 **32 장 한 번에** 넣은 target_encoder 의 미래 토큰을
예측하므로 공간이 다를 수 있다 → 세 가지를 같이 그려 PSNR 로 비교한다 (미래 16 장, [0, 1]):
  blk   블록별 정답 토큰 → decoder   (decoder 학습 분포, 상한)
  win   32 장 정답 토큰 (채점 타깃) → decoder   (공간 차이 확인)
  pred  릴리즈 predictor 예측 → decoder
GIF (32 장, 4 fps 아님 — 8 fps): 위 = 원본, 가운데 = 앞 16 장 원본 + 뒤 16 장 pred, 아래 = 앞 16 장 원본 + 뒤 16 장 win.
산출: runs/<decoder run>/renders/<set>/<clip>.gif · summary.json"""
import argparse, csv, json, os, sys, random
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import build_from_config                  # noqa: E402
from evals.world_model_analysis.eval import _context_target_indices    # noqa: E402
from model import PixelDecoder                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["late", "visible", "early", "mid", "gravity"], default="late")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--decoder", default=os.path.join(HERE, "runs/vobench_vits/latest.pt"))
ap.add_argument("--root", default="/workspace/dataset/benchmarks/VOBench_v1")
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
        "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
        "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32",
        "autocast": "float16", "window_size": 32}
bundle = build_from_config(MCFG, dev)
st = torch.load(a.decoder, map_location="cpu", weights_only=False)
mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
dec.load_state_dict(st["decoder"])
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()

T, C, S, G = 32, 16, 256, 16
IDX = [3 * k for k in range(T)]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
LN = lambda t: F.layer_norm(t, (t.size(-1),))


def load(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0           # (3, T, H, W)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    return x                                                                                  # [0, 1]


@torch.no_grad()
def decode(tok):
    """tok (nb·256, D) LN 토큰 → (nb·2, 3, 256, 256) [0, 1]"""
    nb = tok.size(0) // S
    y = dec(tok.view(nb, S, -1).float())                                                       # (nb, 3, 2, H, W)
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))
if a.set != "gravity":     # VOBench 가림 조건 (late · visible · early · mid), 가능 변이 pos_a
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == a.set and r["variant"] == "pos_a"]
else:
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_gravity/index.csv"))) if r["plausible"] == "1"]
ROOT = a.root if a.set != "gravity" else os.path.join(a.root, "gravity")      # gravity 세트는 VOBench_v1/gravity 아래
random.Random(0).shuffle(rows)
by = {}
for r in rows:                     # 위반 · objset 마다 고르게
    k = (r.get("objset"), r.get("violation") or r.get("law"))
    by.setdefault(k, []).append(r)
pick = []
while len(pick) < a.n and any(by.values()):
    for k in sorted(by):
        if by[k] and len(pick) < a.n:
            pick.append(by[k].pop())
out = os.path.join(os.path.dirname(a.decoder), "renders", a.set); os.makedirs(out, exist_ok=True)
summ = []
for r in pick:
    x01 = load(os.path.join(ROOT, r["file_name"]))                                          # (3, T, H, W)
    xn = ((x01 - MEAN) / STD).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        z = bundle.context_encoder(xn, masks=[ci])
        p = bundle.predictor(z, ci, ti, mask_index=0)
        h = bundle.target_encoder(xn)
        blk = enc2(xn[0].view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))                    # 블록별 (16, 256, D)
    p = p.float()[0]; hwin = LN(h.float())[0][C // 2 * S:]; hblk = LN(blk.float()).reshape(-1, blk.size(-1))[C // 2 * S:]
    real = x01[:, C:].permute(1, 0, 2, 3)                                                      # (16, 3, H, W)
    yp, yw, yb = decode(p), decode(hwin), decode(hblk)
    rec = {"clip": r["file_name"], "psnr_blk": psnr(yb, real), "psnr_win": psnr(yw, real), "psnr_pred": psnr(yp, real),
           "lat_l1_pred_vs_win": float((p - hwin).abs().mean()), "lat_l1_blk_vs_win": float((hblk - hwin).abs().mean())}
    summ.append(rec)
    ctx = x01[:, :C].permute(1, 0, 2, 3)
    rows3 = [x01.permute(1, 0, 2, 3), torch.cat([ctx, yp]), torch.cat([ctx, yw])]
    frames = []
    for t in range(T):
        im = torch.cat([rr[t] for rr in rows3], -1)                                           # 가로로 3 칸
        im = Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        tag = "context" if t < C else "future"
        for j, lab in enumerate(["original", f"release pred ({tag})", f"GT window latent ({tag})"]):
            dr.rectangle([j * 256, 0, j * 256 + 180, 14], fill=(0, 0, 0)); dr.text((j * 256 + 3, 1), lab, fill=(255, 255, 0) if t >= C and j else (255, 255, 255))
        dr.text((3, 242), f"frame {IDX[t]}", fill=(255, 255, 255))
        frames.append(im)
    name = r["file_name"].replace("/", "__")
    frames[0].save(os.path.join(out, f"{name}.gif"), save_all=True, append_images=frames[1:], duration=125, loop=0)
    print(f"{r['file_name']}: PSNR blk {rec['psnr_blk']:.1f} · win {rec['psnr_win']:.1f} · pred {rec['psnr_pred']:.1f} dB"
          f" | latent L1 pred↔win {rec['lat_l1_pred_vs_win']:.3f} · blk↔win {rec['lat_l1_blk_vs_win']:.3f}", flush=True)
m = lambda k: float(np.mean([s_[k] for s_ in summ]))
print(f"\n평균 ({len(summ)} clip) PSNR blk {m('psnr_blk'):.2f} · win {m('psnr_win'):.2f} · pred {m('psnr_pred'):.2f} dB")
json.dump({"set": a.set, "decoder": a.decoder, "clips": summ}, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
print("GIF:", out)
