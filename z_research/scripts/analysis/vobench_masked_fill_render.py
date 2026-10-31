"""VOBench 정지 · 가림 없음에서 미래 물체 구멍을 릴리즈 predictor 가 채우나 — pixel decoder 로 그림 (2026-10-29).

32 장 (원본 0, 3, …, 93) 을 context encoder 에 넣되, 미래 8 튜블릿에서 물체 중심 4x4 토큰만 뺀다 (문맥 8 튜블릿은 전부 보임).
릴리즈 predictor 가 빠진 토큰 (8 x 16 = 128) 을 mask token 으로 채운다 (mask_index 0).
그림 = 정답 latent LN(target_encoder(32 장)) 의 미래에서 구멍만 바꿔 decoder 로:
  GT latent · predictor 로 채움 · 복사 (문맥 마지막 튜블릿의 같은 자리 토큰) — 정지 물체라 복사가 쉬운 기준선이다.
출력 z_analysis/masked_fill_static/<clip>.gif · summary.json

  CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/vobench_masked_fill_render.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_pixel_decoder"))
from analysis.intphys2.model import build_from_config  # noqa: E402
from model import PixelDecoder  # noqa: E402

FR = Path("/local_datasets/world/testbed/VOBench_v1")
CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
DEC = REPO / "z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt"
OUT = REPO / "z_analysis/masked_fill_static"; OUT.mkdir(parents=True, exist_ok=True)
T, C, S, G, HOLE = 32, 16, 256, 16, 4
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))

m = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False,
     "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
     "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
B = build_from_config(m, dev)
st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"])


@torch.no_grad()
def decode(tok):                                                         # (nb·256, D) → (nb·2, 3, 256, 256)
    nb = tok.size(0) // S
    y = (dec(tok.view(nb, S, -1).float()) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


to8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
rows = [r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")) if r["motion"] == "flat_static" and r["occlusion"] == "visible" and r["variant"] == "pos_a"]
summ = []
for r in rows:
    d = FR / r["file_name"]; lab = json.load(open(d.parent / "label.json"))["labels"]["A"]
    x = torch.from_numpy(np.stack([np.asarray(Image.open(d / f"{3 * k:06d}.png").convert("RGB")) for k in range(T)])).permute(3, 0, 1, 2).float().to(dev) / 255
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    xn = ((x - MEAN) / STD).unsqueeze(0)
    hole, boxes = [], []                                                 # 미래 튜블릿마다 물체 중심 4x4 (튜블릿 첫 장의 label 위치)
    for t in range(C // 2, T // 2):
        L = lab[3 * 2 * t]; gi, gj = L["v"] / 512 * G, L["u"] / 512 * G
        i0, j0 = int(np.clip(round(gi - HOLE / 2), 0, G - HOLE)), int(np.clip(round(gj - HOLE / 2), 0, G - HOLE))
        boxes.append((i0, j0)); hole += [t * S + (i0 + a) * G + j0 + b for a in range(HOLE) for b in range(HOLE)]
    hs = set(hole); ci = torch.tensor([k for k in range(T // 2 * S) if k not in hs], device=dev).unsqueeze(0)
    ti = torch.tensor(hole, device=dev).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        p = B.predictor(B.context_encoder(xn, masks=[ci]), ci, ti, mask_index=0).float()[0]       # (128, D)
        h = LN(B.target_encoder(xn).float())[0]                                                  # (16·256, D)
    hp, hc = h.clone(), h.clone()
    hp[hole] = p
    hc[hole] = h[[(C // 2 - 1) * S + (k % S) for k in hole]]                                     # 복사: 문맥 마지막 튜블릿 같은 자리
    fut = slice(C // 2 * S, None)
    yg, yp, yc = decode(h[fut]), decode(hp[fut]), decode(hc[fut])
    real = x[:, C:].permute(1, 0, 2, 3)
    pm = torch.zeros_like(real[:, :1])                                   # 구멍 픽셀 마스크
    for t, (i0, j0) in enumerate(boxes):
        pm[2 * t:2 * t + 2, :, i0 * 16:(i0 + HOLE) * 16, j0 * 16:(j0 + HOLE) * 16] = 1
    hp_ = lambda y: float(-10 * torch.log10((((y - real) ** 2) * pm).sum() / (pm.sum() * 3)))
    rec = {"clip": r["video_id"], "obj": r["obj_a"],
           "latent_L1_hole": {"pred": float((p - h[hole]).abs().mean()), "copy": float((hc[hole] - h[hole]).abs().mean())},
           "psnr_hole": {"gt_latent": hp_(yg), "pred": hp_(yp), "copy": hp_(yc)}}
    summ.append(rec); print(rec, flush=True)
    ctx = to8(x[:, :C].permute(1, 0, 2, 3)); R = to8(x.permute(1, 0, 2, 3))
    cols = [("video (hole box)", R), ("GT latent", torch.cat([ctx, to8(yg)])), ("predictor fill", torch.cat([ctx, to8(yp)])), ("copy last ctx", torch.cat([ctx, to8(yc)]))]
    frames = []
    for f in range(T):
        im = Image.fromarray(torch.cat([c[f] for _, c in cols], -1).permute(1, 2, 0).numpy()); dr = ImageDraw.Draw(im)
        for j, (name, _) in enumerate(cols):
            dr.rectangle([j * 256, 0, j * 256 + 150, 13], fill=(0, 0, 0)); dr.text((j * 256 + 3, 1), name, fill=(255, 255, 0) if f >= C else (255, 255, 255))
            if f >= C:
                i0, j0 = boxes[(f - C) // 2]; dr.rectangle([j * 256 + j0 * 16, i0 * 16, j * 256 + (j0 + HOLE) * 16 - 1, (i0 + HOLE) * 16 - 1], outline=(255, 0, 0))
        dr.text((3, 242), f"frame {3 * f}" + (" (masked future)" if f >= C else " (context)"), fill=(255, 255, 255))
        frames.append(im)
    frames[0].save(OUT / f"{r['video_id']}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
for k in ("pred", "copy"):
    print(k, "latent L1 hole", np.mean([s["latent_L1_hole"][k] for s in summ]).round(3), "PSNR hole", np.mean([s["psnr_hole"][k] for s in summ]).round(2))
print("gt_latent PSNR hole", np.mean([s["psnr_hole"]["gt_latent"] for s in summ]).round(2))
