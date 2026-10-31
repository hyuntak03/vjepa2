"""raw | AR | 문맥 중앙값 — VOBench 클립의 미래를 나란히 GIF 로 (2026-10-27).

문맥 = 원본 0, 3, …, 45 (16 장 = 8 블록), 미래 = 48, …, 93 (8 블록). 블록 latent = 블록별 LN(target_encoder) (AR 의 문맥 encoder 와 같다).
  AR      = vast ar_state ep70 rollout 8 블록 → pixel decoder
  median  = 문맥 8 블록의 토큰 자리별 · 차원별 중앙값 (256, 1280) → LN → 미래 8 블록 전부에 → pixel decoder (L1 표에는 LN 전 · 후 둘 다)
            (L1 의 최적 상수 = 중앙값. "문맥을 시간축으로 요약한 정지 예측")
GIF: 문맥 구간은 세 칸 모두 원본, 미래 구간부터 갈라진다. 수치: 걸음마다 참 미래 latent 와의 L1 (AR · median · 마지막 문맥 블록 복사).
출력 z_analysis/median_latent/ar_vs_median/<clip>.gif · summary.json

  CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/vobench_ar_vs_median_gif.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_pixel_decoder"))
from analysis.intphys2.model import build_from_config           # noqa: E402
from app.vjepa_frozen.ar import block_indices, encode_blocks    # noqa: E402
from model import PixelDecoder                                  # noqa: E402

FR = Path("/local_datasets/world/testbed/VOBench_v1")
OUT = REPO / "z_analysis/median_latent/ar_vs_median"
DEC = REPO / "z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt"
CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
AR_CKPT = REPO / "z_training/vast/ar_state/latest.pt"
S, T, C = 256, 8, 8
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
# (objset, background, motion, occlusion) — 운동 넷 + 정지 + 문맥 끝 가림
PICK = [("primitive", "v11", "flat_vel", "visible"), ("primitive", "v11", "flat_acc", "visible"),
        ("primitive", "v11", "flat_dec", "visible"), ("primitive", "v11", "ramp_down", "visible"),
        ("primitive", "v11", "ramp_up", "visible"), ("primitive", "v11", "flat_static", "visible"),
        ("primitive", "v11", "flat_vel", "late"), ("realistic", "assets", "flat_acc", "visible")]


def load(d):
    fr = [np.asarray(Image.open(d / f"{3 * k:06d}.png").convert("RGB")) for k in range(2 * C + 2 * T)]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0
    return F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")))
    clips = []
    for o, b, m, occ in PICK:
        r = next(r for r in rows if r["objset"] == o and r["background"] == b and r["motion"] == m and r["occlusion"].startswith(occ)
                 and r["variant"] == "pos_a" and r["obj_a"] != "empty")
        clips.append(r)
    mcfg = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
            "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16",
            "window_size": 32, "predictor_checkpoint": str(AR_CKPT)}
    bundle = build_from_config(mcfg, dev)
    st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
    dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
    dec.load_state_dict(st["decoder"])

    def render(tok):                                                   # (nb, 256, 1280) → (nb·2, 3, 256, 256) uint8
        y = dec(tok.float())
        y = (y * STD + MEAN).clamp(0, 1)
        return (y.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256) * 255).round().byte().cpu()

    summ = {}
    for r in clips:
        name = f"{r['objset']}_{r['background']}_{r['motion']}_{r['occlusion']}_{r['obj_a'].replace('/', '-')}"
        with torch.no_grad():
            x = load(FR / r["file_name"])                                                  # (3, 32, 256, 256) [0, 1]
            xn = ((x[None] - MEAN) / STD)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = encode_blocks(bundle.target_encoder, xn, 2)                          # (1, 16·256, 1280)
                p = bundle.predictor.rollout(hb[:, :C * S], block_indices(1, 2 * T, S, dev)[:, :C * S], T)
            h = hb.float().view(2 * T, S, -1); p = p.float().view(T, S, -1)
            med = h[:C].median(0).values                                                   # (256, 1280)
            fut = h[C:]
            l1 = lambda a: [float((a[t] - fut[t]).abs().mean()) for t in range(T)]
            lnm = F.layer_norm(med, (med.size(-1),))
            summ[name] = {"video_id": r["video_id"], "l1_ar": l1(p), "l1_median": l1(med[None].expand(T, -1, -1)),
                          "l1_ln_median": l1(lnm[None].expand(T, -1, -1)),
                          "norm": {"true": float(fut.norm(dim=-1).mean()), "ar": float(p.norm(dim=-1).mean()), "median": float(med.norm(dim=-1).mean())},
                          "l1_copy_last": l1(h[C - 1][None].expand(T, -1, -1))}
            raw = (x.permute(1, 0, 2, 3) * 255).round().byte().cpu()                       # (32, 3, 256, 256)
            ar_f = render(p.view(T, S, -1)); md_f = render(lnm[None].expand(T, -1, -1).contiguous())   # 그림은 LN(중앙값) — decoder · AR 출력과 같은 LN 공간
            gt_f = render(fut.contiguous())                                                # 참 미래 latent → decoder (decoder 한계 참고용)
        W, H, pad, top = 256, 256, 6, 30
        cols = [("raw", None), ("true latent -> decoder", gt_f), ("AR (vast ar_state ep70)", ar_f), ("LN(median of context blocks)", md_f)]
        frames = []
        for k in range(2 * C + 2 * T):
            im = Image.new("RGB", (len(cols) * (W + pad) + pad, top + H + pad), "white"); dr = ImageDraw.Draw(im)
            fut_k = k >= 2 * C
            for j, (lab, arr) in enumerate(cols):
                src = raw[k] if (arr is None or not fut_k) else arr[k - 2 * C]
                x0 = pad + j * (W + pad)
                im.paste(Image.fromarray(src.permute(1, 2, 0).numpy()), (x0, top))
                dr.text((x0 + 2, 4), lab, fill="black")
                dr.text((x0 + 2, 16), f"{'FUTURE' if fut_k else 'context (raw)'}  raw frame {3 * k}", fill="red" if fut_k else "gray")
            frames.append(im)
        strip = Image.new("RGB", (len(cols) * (W + pad) + pad, 3 * (top + H + pad) + pad), "white")  # 미래 장 0 · 7 · 15 (raw 48 · 69 · 93), 칸 이름 포함
        for ri, kk in enumerate((2 * C, 2 * C + 7, 2 * C + 15)):
            strip.paste(frames[kk], (0, pad + ri * (top + H + pad)))
        strip.save(OUT / f"{name}_strip.png")
        frames[0].save(OUT / f"{name}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
        s = summ[name]
        print(f"{name:55s} L1 to true future (t0→t7 mean) AR {np.mean(s['l1_ar']):.3f} · median {np.mean(s['l1_median']):.3f} · LN(median) {np.mean(s['l1_ln_median']):.3f} · |true| {s['norm']['true']:.1f} |AR| {s['norm']['ar']:.1f} |med| {s['norm']['median']:.1f} · copy {np.mean(s['l1_copy_last']):.3f}", flush=True)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(OUT)


if __name__ == "__main__":
    main()
