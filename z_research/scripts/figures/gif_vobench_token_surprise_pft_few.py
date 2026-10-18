#!/usr/bin/env python3
"""vast AR vs vast AR + PFT 토큰별 surprise Δ GIF — 샘플 10 개만 빠르게 (2026-10-15, 사용자: "샘플 10 개 정도만").

split v2 test 의 late · permanence · 움직이는 물체 (flat_vel · flat_acc · ramp_down) block 을 objset 마다 5 개 (seed 0) 골라,
block 마다 pos_a · imp_ab 두 영상만 두 모델로 토큰별 e[t, i, j] = mean_채널 |p − LN(h)| 를 계산한다 (token_surprise_maps.py 의 AR 경로와 같은 식, batch 1).
  Δ = e_imp − e_pos (파랑 = 불가능 쪽을 더 놀라워함 = 정답 방향), 두 모델 · 10 block 공통 눈금 (|Δ| 99 분위).
그림 형식 · 함수는 gif_vobench_token_surprise_pft.py 와 같다. 열 = vast AR · vast AR + PFT, 행 = block.

  CUDA_VISIBLE_DEVICES=<gpu> python z_research/scripts/figures/gif_vobench_token_surprise_pft_few.py
  → z_research/VOBench_v1/vast_ar_pft/token_surprise/few_<objset>.gif
"""
import contextlib
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import gif_vobench_token_surprise_pft as G  # noqa: E402

MOT = {"flat_vel": "const. vel", "flat_acc": "accelerating", "ramp_down": "ramp down"}


def pick():
    rows = list(csv.DictReader(open(G.INDEX))); B = {}
    for r in rows: B.setdefault(r["block_id"], {})[r["variant"]] = r
    rng = np.random.default_rng(0); out = []
    for o in ("primitive", "realistic"):
        c = [v for v in B.values() if v["pos_a"]["objset"] == o and v["pos_a"]["occlusion"] == "late"
             and v["pos_a"]["violation"] == "permanence" and v["pos_a"]["motion"] in MOT]
        c = sorted(c, key=lambda v: (v["pos_a"]["motion"], v["pos_a"]["block_id"])); rng.shuffle(c)
        by = {}
        for v in c: by.setdefault(v["pos_a"]["motion"], []).append(v)
        sel = []
        while len(sel) < 5 and any(by.values()):          # 운동 셋을 번갈아
            for m in MOT:
                if by.get(m) and len(sel) < 5: sel.append(by[m].pop(0))
        out += sel
    return out


def token_e(run, vids):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices  # noqa: F401
    from evals.world_model_analysis.data import WMADataset
    from analysis import predictors as PV
    from analysis.predictors import ar_scoring as ARS
    cfg = yaml.safe_load((run / "_resolved.yaml").read_text())
    ds = WMADataset(cfg); idx = {r.video_id: k for k, r in enumerate(ds.records)}
    dev = torch.device("cuda:0"); bundle = build_from_config(cfg["model"], dev)
    assert PV.is_ar(bundle.predictor); S = cfg["surprise"]; ctx = int(S["context_length"]); ts, sp = bundle.tubelet_size, bundle.num_spatial_tokens
    ARS.check_ar_ready(bool(S.get("target_layer_norm", True)))
    ac = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(cfg["model"].get("autocast", "none")).lower())
    E = {}
    with torch.no_grad():
        for v in vids:
            x = ds.clip(idx[v])[None].to(dev, bundle.dtype)
            with (torch.autocast("cuda", dtype=ac) if ac else contextlib.nullcontext()):
                hb = ARS.encode_blocks(bundle.target_encoder, x, ts); p, h, _ = ARS.rollout_from_blocks(bundle.predictor, hb, ctx // ts, sp)
            E[v] = (p.float() - h.float()).abs().mean(-1).reshape(-1, 16, 16).cpu().numpy()      # (8 미래 튜블릿, 16, 16)
    del bundle; torch.cuda.empty_cache()
    return E


def main():
    sel = pick(); vids = [v[k]["video_id"] for v in sel for k in ("pos_a", "imp_ab")]
    E = {n: token_e(run, vids) for n, run in G.RUNS.items()}; names = list(G.RUNS)
    D = {n: [E[n][v["imp_ab"]["video_id"]] - E[n][v["pos_a"]["video_id"]] for v in sel] for n in names}
    VMAX = float(np.percentile(np.abs(np.stack([d for n in names for d in D[n]])), 99))
    for n in names: print(n, "쌍 Δ", " ".join(f"{d.mean():+.4f}" for d in D[n]))
    PX, G8, HEAD, LEFT = G.PX, 8, 46, 170
    for o in ("primitive", "realistic"):
        ii = [k for k, v in enumerate(sel) if v["pos_a"]["objset"] == o]
        W = LEFT + 2 * PX + G8; H = 26 + HEAD + len(ii) * (PX + 12) + 70
        fimp = {k: G.frames(sel[k]["imp_ab"]) for k in ii}; leg = G.legend_bar(VMAX, PX); out = []
        for s in range(32):
            im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
            dr.text((10, 6), f"VOBench_v1 · {o} · late occlusion · permanence (moving) · impossible clip with token Δ", fill=(0, 0, 0), font=G.FONT_B)
            fut = s >= 16; t = (s - 16) // 2
            dr.text((W - 300, 26), f"future sample {s - 15}/16 (tubelet t{t + 8})" if fut else f"context sample {s + 1}/16 (no tokens scored)",
                    fill=(0, 0, 0) if fut else (120, 120, 120), font=G.FONT)
            for ci, n in enumerate(names): dr.text((LEFT + ci * (PX + G8) + 4, 26 + 18), n, fill=(0, 0, 0), font=G.FONT_B)
            for ri, k in enumerate(ii):
                y = 26 + HEAD + ri * (PX + 12); v = sel[k]["pos_a"]
                dr.multiline_text((8, y + 6), f"{MOT[v['motion']]}\nk = {v['k']}\nblock {v['block_id']}", fill=(0, 0, 0), font=G.FONT_B, spacing=3)
                yy = y + 70
                for n in names:
                    tot = D[n][k].mean()
                    dr.multiline_text((8, yy), f"{n}\n pair Δ {tot:+.4f}", fill=(42, 120, 214) if tot > 0 else (235, 104, 52), font=G.FONT, spacing=2); yy += 40
                for ci, n in enumerate(names):
                    base = fimp[k][s]; arr = G.overlay(base, D[n][k][t], VMAX) if fut else base
                    im.paste(Image.fromarray(arr), (LEFT + ci * (PX + G8), y))
            im.paste(leg, (LEFT + PX // 2 + G8 // 2, H - 36))
            dr.text((8, H - 64), "blue: impossible more surprising (correct) · orange: possible more surprising\none colour scale for both models", fill=(60, 60, 60), font=G.FONT, spacing=2)
            out.append(im)
        G.OUT.mkdir(parents=True, exist_ok=True)
        src = np.concatenate([np.asarray(out[k]) for k in (0, 16, 22, 28, 31)], 0)
        pal = Image.fromarray(src).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in out]
        q[0].save(G.OUT / f"few_{o}.gif", save_all=True, append_images=q[1:], duration=[300] * 31 + [1500], loop=0, optimize=False)
        print("saved", G.OUT / f"few_{o}.gif")


if __name__ == "__main__":
    main()
