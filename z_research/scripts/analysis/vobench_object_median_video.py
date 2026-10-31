"""raw | AR | 블록별 **물체 토큰만** LN(차원별 중앙값) 으로 바꾼 영상 — VOBench (정지 · 움직임) × (보임 · 문맥 끝 가림) (2026-10-27).

vobench_block_median_video.py (모든 토큰을 공간 중앙값으로 → 화면이 균일한 질감으로 무너짐) 의 후속.
물체 토큰 = 라벨 물체 상자 (블록 두 장의 합집합, ext_u · ext_v) 안에 칸 중심이 드는 토큰 (없으면 가장 가까운 칸 하나).
물체가 가려진 블록 (두 장 중 하나라도 hidden ≥ 0.5) 은 바꾸지 않는다 — 그 자리 토큰은 가림막이다.
나머지 토큰은 참 latent 그대로.

블록 latent = 원본 0, 3, …, 93 (32 장 = 16 블록) 을 블록별 LN(target_encoder), 블록마다 (256, 1280).
  median  = 블록마다 256 토큰에 걸친 차원별 중앙값 (1, 1280) → LN → 256 자리 전부에 깔아 decoder → 그 블록의 2 장.
            16 블록 (문맥 + 미래) 을 시간순으로 이어 붙인다. "모든 토큰이 그 블록의 중앙값 토큰 하나로 무너지면" 의 그림.
  AR      = vast ar_state ep70: 문맥 8 블록 → rollout 8 블록 → decoder (문맥 구간은 원본)
  raw     = 원본 32 장
출력 z_analysis/median_latent/object_median/<조건>_<클립>.gif · all.gif (조건 × 클립 행) · summary.json

  CUDA_VISIBLE_DEVICES=1 $PY z_research/scripts/analysis/vobench_object_median_video.py
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
OUT = REPO / "z_analysis/median_latent/object_median"
DEC = REPO / "z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt"
CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
AR_CKPT = REPO / "z_training/vast/ar_state/latest.pt"
S, T, C, NPER = 256, 8, 8, 2
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
CONDS = [("static_visible", "flat_static", "visible"), ("static_occluded", "flat_static", "late"),
         ("moving_visible", "flat_vel", "visible"), ("moving_occluded", "flat_vel", "late")]


def load(d):
    fr = [np.asarray(Image.open(d / f"{3 * k:06d}.png").convert("RGB")) for k in range(32)]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0
    return F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")))
    mcfg = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
            "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16",
            "window_size": 32, "predictor_checkpoint": str(AR_CKPT)}
    bundle = build_from_config(mcfg, dev)
    st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
    dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
    dec.load_state_dict(st["decoder"])

    def render(tok):                                                   # (nb, 256, 1280) → (nb·2, 3, 256, 256) uint8
        y = (dec(tok.float()) * STD + MEAN).clamp(0, 1)
        return (y.permute(0, 2, 1, 3, 4).reshape(-1, 3, 256, 256) * 255).round().byte().cpu()

    W, H, pad, top, lab_w = 256, 256, 6, 30, 150
    cols = ["raw", "AR (vast ar_state ep70)", "object tokens -> LN(median)"]
    summ, rows_all = {}, []
    for cname, mot, occ in CONDS:
        picked = [r for r in rows if r["objset"] == "primitive" and r["background"] == "v11" and r["motion"] == mot
                  and r["occlusion"].startswith(occ) and r["variant"] == "pos_a" and r["obj_a"] != "empty"][:NPER]
        for r in picked:
            name = f"{cname}_{r['occlusion']}_{r['obj_a'].replace('/', '-')}"
            with torch.no_grad():
                x = load(FR / r["file_name"]); xn = (x[None] - MEAN) / STD
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hb = encode_blocks(bundle.target_encoder, xn, 2)
                    p = bundle.predictor.rollout(hb[:, :C * S], block_indices(1, 2 * T, S, dev)[:, :C * S], T)
                h = hb.float().view(2 * T, S, -1); p = p.float().view(T, S, -1)
                L = json.load(open(FR / Path(r["file_name"]).parent / "label.json"))["labels"]["A"]
                med_tok = h.clone(); n_obj, spread = [], []
                cy, cx = torch.meshgrid(torch.arange(16, device=dev) + 0.5, torch.arange(16, device=dev) + 0.5, indexing="ij")
                for b in range(2 * T):
                    fr = [L[3 * (2 * b)], L[3 * (2 * b + 1)]]
                    if max(f["hidden"] for f in fr) >= 0.5:
                        n_obj.append(0); spread.append(float("nan")); continue
                    m = torch.zeros(16, 16, dtype=torch.bool, device=dev)
                    for f in fr:                                                   # 512 px 좌표 → 16 칸 (1 칸 = 32 px)
                        u, v, eu, ev = f["u"] / 32, f["v"] / 32, f["ext_u"] / 32, f["ext_v"] / 32
                        m |= ((cx - u).abs() <= max(eu, 0.5)) & ((cy - v).abs() <= max(ev, 0.5))
                        m[min(15, int(v)), min(15, int(u))] = True
                    idx = m.flatten().nonzero()[:, 0]
                    tok = h[b, idx]                                                # (k, 1280)
                    med_tok[b, idx] = LN(tok.median(0).values)[None]
                    n_obj.append(int(len(idx))); spread.append(float((tok - tok.median(0).values).abs().mean()))
                raw = (x.permute(1, 0, 2, 3) * 255).round().byte().cpu()
                ar_f = render(p); md_f = render(med_tok)
                fut = h[C:]
                summ[name] = {"video_id": r["video_id"],
                              "l1_future_ar": float((p - fut).abs().mean()),
                              "l1_future_object_median": float((med_tok[C:] - fut).abs().mean()),
                              "n_object_tokens_per_block": n_obj, "object_token_spread_l1": spread}
            seq = []
            for k in range(32):
                fut_k = k >= 2 * C
                seq.append([raw[k], raw[k] if not fut_k else ar_f[k - 2 * C], md_f[k]])
            rows_all.append((name, seq))
            # 클립별 GIF
            frames = []
            for k, panel in enumerate(seq):
                im = Image.new("RGB", (len(cols) * (W + pad) + pad, top + H + pad), "white"); dr = ImageDraw.Draw(im)
                for j, src in enumerate(panel):
                    x0 = pad + j * (W + pad)
                    im.paste(Image.fromarray(src.permute(1, 2, 0).numpy()), (x0, top))
                    dr.text((x0 + 2, 4), cols[j], fill="black")
                    dr.text((x0 + 2, 16), f"{'FUTURE' if k >= 2 * C else 'context'}  raw frame {3 * k}", fill="red" if k >= 2 * C else "gray")
                frames.append(im)
            frames[0].save(OUT / f"{name}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
            print(f"{name:45s} 물체 토큰 수 {np.mean([x for x in n_obj if x]) if any(n_obj) else 0:.1f} · 물체 토큰끼리 L1 {np.nanmean(spread) if any(n_obj) else float('nan'):.3f} · 미래 L1 AR {summ[name]['l1_future_ar']:.3f} / 물체중앙값 {summ[name]['l1_future_object_median']:.3f}", flush=True)
    # 전체 한 장 GIF: 행 = 조건 × 클립
    frames = []
    for k in range(32):
        im = Image.new("RGB", (lab_w + len(cols) * (W + pad) + pad, top + len(rows_all) * (H + pad) + pad), "white"); dr = ImageDraw.Draw(im)
        for j, c in enumerate(cols):
            dr.text((lab_w + pad + j * (W + pad) + 2, 4), c, fill="black")
        dr.text((lab_w + pad + 2, 16), f"{'FUTURE' if k >= 2 * C else 'context'}  raw frame {3 * k}", fill="red" if k >= 2 * C else "gray")
        for i, (name, seq) in enumerate(rows_all):
            y0 = top + i * (H + pad)
            dr.text((4, y0 + H // 2 - 12), name.split("_")[0] + " " + name.split("_")[1], fill="black")
            dr.text((4, y0 + H // 2 + 2), "_".join(name.split("_")[2:]), fill="gray")
            for j, src in enumerate(seq[k]):
                im.paste(Image.fromarray(src.permute(1, 2, 0).numpy()), (lab_w + pad + j * (W + pad), y0))
        frames.append(im.resize((im.width // 2, im.height // 2)))
    frames[0].save(OUT / "all.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(OUT)


if __name__ == "__main__":
    main()
