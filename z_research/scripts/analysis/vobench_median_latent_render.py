"""L1 최적해 = 차원별 중앙값 — VOBench 첫 블록 latent 의 토큰 자리별 · 차원별 중앙값을 pixel decoder 로 그린다 (2026-10-27).

블록 latent = 원본 0 · 3 번째 장 (채점 간격) 을 target_encoder 에 넣고 affine 없는 LN, (256, 1280).
대상 = 서로 다른 문맥의 첫 블록: block 마다 pos_a · pos_b (imp_ab · imp_ba 는 문맥이 같다) → 1,440 개.
그리는 것 (각 2 장):
  실제 영상 · 그 latent 를 decoder 로
  전체 중앙값 / 전체 평균           — 1,440 개에 걸쳐 토큰 자리 · 차원마다
  조건부 중앙값 / 조건부 평균      — 같은 objset · 배경 · 운동, 물체 (모양 · 색) 만 다른 영상들
  각 중앙값은 그대로 (L1 최적해) 와 LN(중앙값) (decoder 학습 분포) 둘 다
출력: z_analysis/median_latent/ — grid.png · grid.gif (2 장 번갈아) · stats.json

  CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/vobench_median_latent_render.py
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_pixel_decoder"))
from analysis.intphys2.model import build_from_config           # noqa: E402
from app.vjepa_frozen.ar import encode_blocks                   # noqa: E402
from model import PixelDecoder                                  # noqa: E402  (z_pixel_decoder/model.py)

FR = Path("/local_datasets/world/testbed/VOBench_v1")
OUT = REPO / "z_analysis/median_latent"
DEC = REPO / "z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt"
CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
S = 256
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))


def load2(d):
    fr = [np.asarray(Image.open(d / f"{i:06d}.png").convert("RGB")) for i in (0, 3)]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0            # (3, 2, H, W)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    return x                                                                                    # [0, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")) if r["variant"] in ("pos_a", "pos_b")]
    m = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
         "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
         "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16",
         "window_size": 32}
    enc = build_from_config(m, dev).target_encoder
    st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
    dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
    dec.load_state_dict(st["decoder"])

    Z, PIX = [], []
    with torch.no_grad():
        for k in range(0, len(rows), 32):
            xs = torch.stack([load2(FR / r["file_name"]) for r in rows[k:k + 32]])               # (B, 3, 2, 256, 256) [0, 1]
            xn = (xs - MEAN) / STD
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = encode_blocks(enc, xn, 2)                                                  # (B, 256, 1280) LN
            Z.append(z.float().cpu().half()); PIX.append((xs * 255).round().byte().cpu())
            if k % 320 == 0:
                print(f"encode {k + len(xs)}/{len(rows)}", flush=True)
    Z = torch.cat(Z).float(); PIX = torch.cat(PIX)                                             # (n, 256, 1280) · (n, 3, 2, 256, 256)

    def render(tok):                                                                           # (256, 1280) → (2, 3, 256, 256) uint8
        with torch.no_grad():
            y = dec(tok[None].to(dev).float())
            y = (y * STD + MEAN).clamp(0, 1)[0].permute(1, 0, 2, 3)
        return (y * 255).round().byte().cpu()

    # 조건부 그룹: objset · background · motion 이 같고 물체만 다른 영상
    key = lambda r: (r["objset"], r["background"], r["motion"])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[key(r)].append(i)
    ex = next(i for i, r in enumerate(rows) if r["objset"] == "primitive" and r["background"] == "v11" and r["motion"] == "flat_vel"
              and r["occlusion"] == "visible")
    g = groups[key(rows[ex])]
    med_all, mean_all = Z.median(0).values, Z.mean(0)
    med_g, mean_g = Z[g].median(0).values, Z[g].mean(0)
    panels = [("real frames", PIX[ex][:, :, :, :].permute(1, 0, 2, 3)),
              ("real latent -> decoder", render(Z[ex])),
              (f"conditional median (n={len(g)})", render(med_g)),
              ("LN(conditional median)", render(LN(med_g))),
              ("conditional mean", render(mean_g)),
              (f"global median (n={len(Z)})", render(med_all)),
              ("LN(global median)", render(LN(med_all))),
              ("global mean", render(mean_all))]
    stats = {"n_all": len(Z), "n_group": len(g), "group": list(key(rows[ex])), "example": rows[ex]["video_id"],
             "token_norm": {"real (LN)": float(Z[ex].norm(dim=-1).mean()), "cond_median": float(med_g.norm(dim=-1).mean()),
                            "cond_mean": float(mean_g.norm(dim=-1).mean()), "global_median": float(med_all.norm(dim=-1).mean()),
                            "global_mean": float(mean_all.norm(dim=-1).mean())},
             "l1_to_real": {"cond_median": float((med_g - Z[ex]).abs().mean()), "cond_mean": float((mean_g - Z[ex]).abs().mean()),
                            "global_median": float((med_all - Z[ex]).abs().mean()), "copy_other_clip_in_group": float((Z[g[1] if g[0] == ex else g[0]] - Z[ex]).abs().mean())},
             "mean_l1_over_group": {"to_cond_median": float((Z[g] - med_g).abs().mean()), "to_cond_mean": float((Z[g] - mean_g).abs().mean())}}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)

    # 그림: 위 = 장 0, 아래 = 장 1 (원본 0 · 3 번째 장)
    W, H, pad, top = 256, 256, 6, 34
    grid = Image.new("RGB", (len(panels) * (W + pad) + pad, top + 2 * (H + pad) + pad), "white"); dr = ImageDraw.Draw(grid)
    frames_gif = [Image.new("RGB", (len(panels) * (W + pad) + pad, top + H + pad), "white") for _ in range(2)]
    for j, (lab, im) in enumerate(panels):
        x0 = pad + j * (W + pad)
        dr.text((x0 + 2, 6), lab, fill="black"); dr.text((x0 + 2, 18), "frame 0 / frame 3 (raw)", fill="gray")
        for f in range(2):
            pil = Image.fromarray(im[f].permute(1, 2, 0).numpy())
            grid.paste(pil, (x0, top + f * (H + pad)))
            frames_gif[f].paste(pil, (x0, top)); ImageDraw.Draw(frames_gif[f]).text((x0 + 2, 6), f"{lab}  [frame {f * 3}]", fill="black")
    grid.save(OUT / "grid.png")
    frames_gif[0].save(OUT / "grid.gif", save_all=True, append_images=[frames_gif[1]], duration=400, loop=0)
    print(json.dumps(stats, indent=1)); print(OUT)


if __name__ == "__main__":
    main()
