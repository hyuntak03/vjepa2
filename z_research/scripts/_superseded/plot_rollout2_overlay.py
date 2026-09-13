#!/usr/bin/env python3
"""RollOut_v2 — 미래 프레임 위에 "p 가 만든 위치" 를 얹는다. 클립당 세 줄.

  1행  프레임 + 마커:  흰 링 = 정답 궤적(plan 투영)   하늘색 점선 링 = 불가능 궤적(ledge/wall)
                     주황 = p 의 pooled 자 (p_B: ledge·wall 을 학습에서 뺀 것; 단일 미래 시나리오는 p_A)
                     초록 = h 로 학습한 토큰 decoder 를 p 토큰에 건 값     파랑 = 같은 decoder 를 h 토큰에 건 값
  2행  토큰 decoder 의 softmax 맵 — **p 토큰** (predictor 가 물체를 어느 칸에 두었나)
  3행  같은 맵 — **h 토큰** (실제 프레임을 본 인코더, 천장)
열 = 미래 튜블릿 0..7 의 첫 프레임 (raw 48 + 6t). p 는 최종 출력(predictor_norm → predictor_proj), h 는 LN(target_encoder) — 채점과 같은 특징.

  python z_research/scripts/figures/plot_rollout2_overlay.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_position as rp                                            # noqa: E402
import rollout2_token_decoder as td                                       # noqa: E402

FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2")
OUT = rp.ROOT / "z_research/RollOutV2/figures/overlay"
ORANGE, GREEN, BLUE, WHITE, CYAN = (235, 104, 52), (27, 175, 122), (42, 120, 214), (255, 255, 255), (80, 220, 255)
R = 288


def ring(d, x, y, col, r=7, w=2, dashed=False):
    if not dashed:
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=w); return
    for k in range(0, 360, 30):                       # 점선 링
        d.arc([x - r, y - r, x + r, y + r], start=k, end=k + 15, fill=col, width=w)


def dot(d, x, y, col, r=5):
    d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=WHITE, width=2)


def heat(frame, a, peak_xy):
    """a: (256,) softmax → 16x16 → 288 로 nearest 확대해 프레임에 섞는다."""
    m = a.reshape(16, 16).astype(np.float32); m = m / m.max()
    rgba = (matplotlib.colormaps["magma"](m) * 255).astype(np.uint8)
    hm = Image.fromarray(rgba[:, :, :3]).resize((R, R), Image.NEAREST)
    alpha = Image.fromarray((m * 200).astype(np.uint8)).resize((R, R), Image.NEAREST)
    out = frame.copy(); out.paste(hm, (0, 0), alpha)
    d = ImageDraw.Draw(out); x, y = peak_xy; d.rectangle([x - 8, y - 8, x + 8, y + 8], outline=WHITE, width=2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-scenario", type=int, default=2); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(a.device)
    idx = list(csv.DictReader(rp.INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((rp.CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}; rows = np.array([row_of[v] for v in vids])
    F = rp.pool(rows); res = R / 2
    px = np.stack([rp.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([rp.arr(r["px_y_by_sample"]) for r in idx])
    Lx = (px.reshape(-1, 16, 2).mean(2) / res - 1)[:, 8:]; Ly = (py.reshape(-1, 16, 2).mean(2) / res - 1)[:, 8:]
    sc = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); blk = np.array([r["block_id"] for r in idx])
    imp_of = {blk[i]: i for i in np.where(plaus == 0)[0]}
    # pooled 자 (rollout2_position 과 같은 split)
    rng = np.random.RandomState(a.seed); tr, ho = [], []
    for scn in rp.SCEN:
        ids = np.where((sc == scn) & (plaus == 1))[0]; ho += list(ids[hold[ids]]); fit = ids[~hold[ids]]
        cell = np.array([f"{prim[i]}|{sec[i]}" for i in fit])
        for c in np.unique(cell):
            i = fit[cell == c]; rng.shuffle(i); k = int(round(rp.TRAIN_FRAC * len(i))); tr += list(i[:k])
    tr, ho = np.array(sorted(tr)), np.array(sorted(ho)); tr_B = tr[~np.isin(sc[tr], ["ledge", "wall"])]
    fut = lambda s_, ids: F[s_][ids].reshape(-1, rp.D).astype(np.float64)
    W = {"p_A": {c: rp.fit_lstsq(fut("p", tr), L[tr].ravel()) for c, L in (("x", Lx), ("y", Ly))},
         "p_B": {c: rp.fit_lstsq(fut("p", tr_B), L[tr_B].ravel()) for c, L in (("x", Lx), ("y", Ly))}}
    # 토큰 decoder (h 학습, 저장본)
    model = td.TokenDecoder().to(dev); model.load_state_dict(torch.load(rp.ROOT / "z_research/RollOutV2/exp_results/token_decoder.pt", map_location=dev)); model.eval()
    P = np.load(rp.CACHE / "predictor.npy", mmap_mode="r"); H = np.load(rp.CACHE / "target.npy", mmap_mode="r")
    gx = model.gx.cpu().numpy(); gy = model.gy.cpu().numpy()

    @torch.no_grad()
    def tok(src, i):
        x = np.asarray(P[rows[i]]).reshape(8, 256, rp.D) if src == "p" else np.asarray(H[rows[i], 8 * 256:]).reshape(8, 256, rp.D)
        pr, at = model(torch.from_numpy(x).to(dev, torch.float32)); return pr.cpu().numpy(), at.cpu().numpy()

    OUT.mkdir(parents=True, exist_ok=True); frames = [48 + 6 * t for t in range(8)]
    # 최소형: 미래 프레임 8장 위에 **predictor 에서 읽은 위치(pooled 자)** 하나만. ledge·wall 은 학습에서 뺀 자(p_B).
    for scn in rp.SCEN:
        ids = ho[sc[ho] == scn]; wp = "p_B" if scn in ("ledge", "wall") else "p_A"
        pick = [ids[(prim[ids] == lv) & (sec[ids] == sv)][0] for lv in sorted(set(prim[ids]))[:1] for sv in sorted(set(sec[ids]))][:a.per_scenario]
        for i in pick:
            xp = {c: rp.apply_w(fut("p", [i]), W[wp][c]).reshape(8) for c in "xy"}
            canvas = Image.new("RGB", (R * 8, R + 24), "white"); d = ImageDraw.Draw(canvas)
            d.text((6, 5), f"{scn}  {idx[i]['video_id']}  primary={prim[i]:g} secondary={sec[i]:g}   orange = object position read from predictor output ({wp})   frames raw 48+6t", fill=(0, 0, 0))
            for t, f in enumerate(frames):
                im = Image.open(FRAMES / idx[i]["file_name"] / f"{f:06d}.png").convert("RGB"); dd = ImageDraw.Draw(im)
                dot(dd, (xp["x"][t] + 1) * res, (xp["y"][t] + 1) * res, ORANGE, r=10)
                dd.text((4, 4), f"t{t} f{f}", fill=(255, 255, 0)); canvas.paste(im, (R * t, 24))
            name = f"{scn}_{idx[i]['video_id']}.png"; canvas.save(OUT / name); print(f"  [saved] overlay/{name}")


if __name__ == "__main__":
    main()
