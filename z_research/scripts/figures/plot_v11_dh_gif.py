#!/usr/bin/env python3
"""IntPhysGen v11 — **D_h(t) = |h(물체 있음) − h(빈 장면)|** 를 토큰 (16×16) 마다 그린 GIF.

v11 vanish block 의 가능 변이 두 개는 **물체 하나만 다르다** — `pos_a` (물체가 처음부터 끝까지) 와
`pos_b` (처음부터 끝까지 빈 장면). 배경 · 가림막 · k 는 같다. 그래서 두 clip 의 target encoder 토큰 차이는
**실제 프레임에서 물체가 만든 흔적**이다. 가림막 · 배경은 뺄셈에서 지워진다.

  h        = LN(target_encoder(32 샘플)) (affine 없는 layer norm, 채점과 같다) → (16 튜블릿, 256 토큰, 1280)
  D_h(t,i) = mean_d |h_a(t,i,d) − h_b(t,i,d)|                                   → (16, 16×16)
  색 척도  = 모든 칸 · 모든 튜블릿 공통 (vmax = 99.5 퍼센타일). 배경은 빈 장면 프레임 (물체가 안 보이게).
  흰 고리  = 진실 위치 (metadata `object_px_{x,y}_by_sample`), 점선 고리 = 그 샘플에서 물체가 가려짐

행 = static · flat · ramp, 열 묶음 = visible (k=0) · 문맥 끝 가림 k=2 · 문맥 중간 가림 k=2, 묶음마다 [물체 clip 프레임 | D_h].
각 칸은 env `concrete_brick` 의 첫 vanish block 이다 (이동 조건은 l2r). 모델 = ViT-H 릴리즈 (`vith`), dtype bf16.
검사: (1) 짝의 픽셀 차이가 물체 주변에만 있는가 (2) D_h argmax 가 진실 칸 ±1 안에 드는 튜블릿 비율.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=0 $P z_research/scripts/figures/plot_v11_dh_gif.py      # → RollOutV3/figures/v11_dh_gif/
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_window_readout import MODEL                     # noqa: E402  (모델 정의를 복제하지 않는다)

FR = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
META = FR / "metadata.csv"
OUT = ROOT / "z_research/RollOutV3/figures/v11_dh_gif"
TMP = Path(os.environ.get("DH_TMP", "/dev/shm/v11_dh"))
NS, CS, S, G, W = 32, 16, 256, 16, 288
MOT = {"static": "static_{}", "flat": "moving_{}_flat", "ramp": "moving_{}"}
TIM = {"visible": ("visible", "", "0"), "late k=2": ("occlusion", "late", "2"), "mid k=2": ("occlusion", "mid", "2")}


def cond_name(mo, vis, timing):
    c = MOT[mo].format(vis)
    return c + ("_mid" if timing == "mid" else "")


def arr(s):
    return np.array([float(v) for v in str(s).replace(",", " ").split()], np.float32)


def pick():
    """(motion, timing) 마다 vanish block 하나 → [(mo, tlabel, row_a, row_b, meta_a)]"""
    csv.field_size_limit(10 ** 9)
    R = [r for r in csv.DictReader(INDEX.open()) if r["violation_type"] == "vanish" and r["plausible"] == "1"]
    M = {m["name"]: m for m in csv.DictReader(META.open())}
    by = {}
    for r in R:
        by.setdefault(r["block_id"], {})[r["variant"]] = r
    out = []
    for mo in MOT:
        for tl, (vis, timing, k) in TIM.items():
            c = cond_name(mo, vis, timing)
            for b, d in by.items():
                a = d.get("pos_a")
                if not a or a["condition"] != c or a["sym_k"] != k or a["env"] != "concrete_brick":
                    continue
                m = M[a["video_id"]]
                if mo != "static" and m.get("travel_direction") != "l2r":
                    continue
                out.append((mo, tl, a, d["pos_b"], m)); break
            else:
                sys.exit(f"block 없음: {mo} {tl} ({c}, k={k})")
    return out


def extract(sel):
    """h 토큰 → D_h (n_pair, 16, 16, 16). 모델 한 번, GPU 한 장."""
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    rows = [x for s in sel for x in (s[2], s[3])]
    TMP.mkdir(parents=True, exist_ok=True)
    with (TMP / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    data = dict(root=str(TMP), index_csv="index.csv", frames_root=str(FR), frames_pattern="{file_name}/{frame:06d}.png",
                frames_start=0, frames_stride=3, n_frames=NS, resolution=256)
    dev = torch.device("cuda")
    bundle = build_from_config({**MODEL, "window_size": NS}, dev)
    ds = WMADataset(dict(data=data, model={**MODEL, "window_size": NS}, features={"cache_dir": "/tmp"}, surprise={}))
    D = []
    with torch.no_grad():
        for j in range(len(sel)):
            clips = torch.stack([ds.clip(2 * j), ds.clip(2 * j + 1)]).to(dev, dtype=bundle.dtype)
            h = bundle.target_encoder(clips); h = h[-1] if isinstance(h, list) else h
            h = F.layer_norm(h.float(), (h.size(-1),)).reshape(2, NS // 2, S, -1)
            D.append((h[0] - h[1]).abs().mean(-1).reshape(NS // 2, G, G).cpu().numpy())
            print(f"  {j+1}/{len(sel)} {sel[j][0]:6s} {sel[j][1]:9s} {sel[j][2]['video_id']}", flush=True)
    return np.stack(D)


def frame(r, s):
    return Image.open(FR / r["file_name"] / f"{3*s:06d}.png").convert("RGB")


def checks(sel, D):
    """(1) 짝 픽셀 차이가 물체 원 (반지름 30 px) 밖에 있는 비율 (2) D_h argmax 가 진실 칸 ±1 안인 튜블릿 비율 (보이는 튜블릿만)."""
    res = {}
    for j, (mo, tl, a, b, m) in enumerate(sel):
        px, py = arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])
        hs, he = float(m["hidden_start"] or 1e9), float(m["hidden_end"] or -1)
        outside = []
        for s in (0, 10, 20, 31):
            d = np.abs(np.asarray(frame(a, s), np.float32) - np.asarray(frame(b, s), np.float32)).max(-1) > 20
            yy, xx = np.mgrid[:W, :W]; far = (xx - px[s]) ** 2 + (yy - py[s]) ** 2 > 30 ** 2
            outside.append(float((d & far).sum() / max(d.sum(), 1)))
        hit, nvis = 0, 0
        for t in range(NS // 2):
            s2 = [2 * t, 2 * t + 1]
            if any(hs <= 3 * s <= he for s in s2):
                continue
            cx, cy = px[s2].mean() / W * G, py[s2].mean() / W * G
            if not (0 <= cx < G and 0 <= cy < G):
                continue
            iy, ix = np.unravel_index(D[j, t].argmax(), (G, G)); nvis += 1
            hit += abs(ix + 0.5 - cx) <= 1.5 and abs(iy + 0.5 - cy) <= 1.5
        res[f"{mo}|{tl}"] = dict(video_a=a["video_id"], video_b=b["video_id"],
                                 pixel_diff_outside_object=[round(x, 4) for x in outside],
                                 argmax_hit=f"{hit}/{nvis}")
    return res


def render(sel, D):
    OUT.mkdir(parents=True, exist_ok=True)
    P, gap, top, left = 170, 6, 46, 64
    vmax = float(np.percentile(D, 99.5)); vmin = float(np.percentile(D, 5))
    cmap = matplotlib.colormaps["magma"]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = small = ImageFont.load_default()
    mos, tls = list(MOT), list(TIM)
    Wt = left + len(tls) * (2 * P + gap) + (len(tls) - 1) * 14; Ht = top + 22 + len(mos) * (P + gap) + 24
    frames = []
    for s in range(NS):
        t = s // 2
        im = Image.new("RGB", (Wt, Ht), "white"); d = ImageDraw.Draw(im)
        ph = "context" if s < CS else "prediction window"
        d.text((left, 6), f"sample {s:2d} (raw {3*s:2d}) · tubelet {t:2d} · {ph}", fill="black", font=font)
        d.text((left, 24), "D_h = |h(object clip) - h(empty clip)| per token, mean over 1280 dims, shared color scale "
                           f"[{vmin:.2f}, {vmax:.2f}] · white ring = true position, dashed = hidden", fill="#444", font=small)
        for gi, tl in enumerate(tls):
            x0 = left + gi * (2 * P + gap + 14)
            d.text((x0 + P - 40, top + 4), tl, fill="black", font=font)
        for ri, mo in enumerate(mos):
            y0 = top + 22 + ri * (P + gap)
            d.text((6, y0 + P // 2 - 7), mo, fill="black", font=font)
            for gi, tl in enumerate(tls):
                j = next(i for i, x in enumerate(sel) if x[0] == mo and x[1] == tl)
                _, _, a, b, m = sel[j]
                x0 = left + gi * (2 * P + gap + 14)
                im.paste(frame(a, s).resize((P, P), Image.BILINEAR), (x0, y0))
                bg = np.asarray(frame(b, s).convert("L").resize((P, P), Image.BILINEAR), np.float32)[..., None] / 255
                hm = np.clip((D[j, t] - vmin) / (vmax - vmin), 0, 1)
                hm = np.asarray(Image.fromarray((hm * 255).astype(np.uint8)).resize((P, P), Image.NEAREST), np.float32) / 255
                rgb = cmap(hm)[..., :3]; al = 0.35 + 0.6 * hm[..., None]
                comp = (al * rgb + (1 - al) * 0.5 * bg) * 255
                im.paste(Image.fromarray(comp.astype(np.uint8)), (x0 + P + gap, y0))
                px, py = arr(m["object_px_x_by_sample"])[s] / W * P, arr(m["object_px_y_by_sample"])[s] / W * P
                hs, he = float(m["hidden_start"] or 1e9), float(m["hidden_end"] or -1)
                hidden = hs <= 3 * s <= he
                for xo in (x0, x0 + P + gap):
                    if 0 <= px < P and 0 <= py < P:
                        rr = 12
                        if hidden:
                            for k in range(0, 360, 40):
                                d.arc([xo + px - rr, y0 + py - rr, xo + px + rr, y0 + py + rr], k, k + 20, fill="white", width=2)
                        else:
                            d.ellipse([xo + px - rr, y0 + py - rr, xo + px + rr, y0 + py + rr], outline="white", width=2)
                if hidden:
                    d.text((x0 + 4, y0 + 3), "object hidden", fill="yellow", font=small)
                if s >= CS:
                    d.rectangle([x0 + P + gap, y0, x0 + 2 * P + gap - 1, y0 + P - 1], outline="#2a78d6", width=2)
        d.text((left, Ht - 20), "blue frame = prediction window (samples 16-31). left of each pair: object clip; right: D_h over the empty clip (grayscale)",
               fill="#444", font=small)
        frames.append(im)
    frames[0].save(OUT / "dh_v11.gif", save_all=True, append_images=frames[1:], duration=220, loop=0)
    frames[CS + 4].save(OUT / "dh_v11_still_s20.png")
    return vmin, vmax


def main():
    sel = pick()
    cache = OUT / "dh_maps.npz"
    ids = np.array([s[2]["video_id"] for s in sel])
    if cache.exists() and (np.load(cache)["video_a"] == ids).all():
        D = np.load(cache)["D"]
    else:
        D = extract(sel)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, D=D, video_a=ids, video_b=np.array([s[3]["video_id"] for s in sel]),
                            motion=np.array([s[0] for s in sel]), timing=np.array([s[1] for s in sel]))
    res = checks(sel, D)
    vmin, vmax = render(sel, D)
    res["_scale"] = dict(vmin=vmin, vmax=vmax)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"→ {OUT}/dh_v11.gif")


if __name__ == "__main__":
    main()
