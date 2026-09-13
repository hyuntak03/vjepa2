#!/usr/bin/env python3
"""RollOut_v2 — h 로 학습한 **토큰 단위** 위치 decoder(token_decoder.pt)를 predictor 미래 8 튜블릿에 test 로만 건다.
  overlay_tok/  미래 프레임 8장 위에 predictor 에서 읽은 위치. 주황 점 = softmax 기대값 (좌상단 peak = 최대 토큰 확률)
  traj_tok/     마지막 미래 프레임 위에 궤적 두 줄 — 정답(흰) vs predictor(주황, 기대값), 점 = 튜블릿 0..7
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_token_decoder as td                                       # noqa: E402

FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2"); OUTROOT = td.ROOT / "z_research/RollOutV2/figures"
ORANGE, WHITE = (235, 104, 52), (255, 255, 255); R = 288


def main():
    dev = torch.device("cuda:0"); model = td.TokenDecoder().to(dev)
    model.load_state_dict(torch.load(td.ROOT / "z_research/RollOutV2/exp_results/token_decoder.pt", map_location=dev)); model.eval()
    idx = list(csv.DictReader(td.INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((td.CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}; rows = np.array([row_of[v] for v in vids])
    res = R / 2; px = np.stack([td.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([td.arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(-1, 16, 2).mean(2) / res - 1, py.reshape(-1, 16, 2).mean(2) / res - 1], -1)[:, 8:]
    sc = np.array([r["scenario"] for r in idx]); prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    P = np.load(td.CACHE / "predictor.npy", mmap_mode="r"); gx = model.gx.cpu().numpy(); gy = model.gy.cpu().numpy()
    for d in ("overlay_tok", "traj_tok"):
        (OUTROOT / d).mkdir(parents=True, exist_ok=True)
    frames = [48 + 6 * t for t in range(8)]
    for scn in td.SCEN:
        ids = np.where(hold & (plaus == 1) & (sc == scn))[0]
        pick = [ids[(prim[ids] == lv) & (sec[ids] == sv)][0] for lv in sorted(set(prim[ids]))[:1] for sv in sorted(set(sec[ids]))][:2]
        for i in pick:
            with torch.no_grad():
                pr, at = model(torch.from_numpy(np.asarray(P[rows[i]]).reshape(8, 256, td.D)).to(dev, torch.float32))
            pr = pr.cpu().numpy(); pk = at.argmax(-1).cpu().numpy(); pmax = at.max(-1).values.cpu().numpy(); gt = L[i]
            canvas = Image.new("RGB", (R * 8, R + 24), "white"); d = ImageDraw.Draw(canvas)
            d.text((6, 5), f"{scn}  {idx[i]['video_id']}  primary={prim[i]:g} secondary={sec[i]:g}   token decoder fit on h, applied to predictor tokens:  dot = softmax expectation", fill=(0, 0, 0))
            for t, f in enumerate(frames):
                im = Image.open(FRAMES / idx[i]["file_name"] / f"{f:06d}.png").convert("RGB"); dd = ImageDraw.Draw(im)
                x, y = (pr[t, 0] + 1) * res, (pr[t, 1] + 1) * res; dd.ellipse([x - 10, y - 10, x + 10, y + 10], fill=ORANGE, outline=WHITE, width=2)
                dd.text((4, 4), f"t{t} f{f}  peak {pmax[t]:.2f}", fill=(255, 255, 0)); canvas.paste(im, (R * t, 24))
            canvas.save(OUTROOT / "overlay_tok" / f"{scn}_{idx[i]['video_id']}.png")
            bg = Image.open(FRAMES / idx[i]["file_name"] / f"{frames[-1]:06d}.png").convert("RGB").resize((R * 2, R * 2), Image.NEAREST); dd = ImageDraw.Draw(bg)
            for traj, col in ((gt, WHITE), (pr, ORANGE)):
                pts = [((p[0] + 1) * res * 2, (p[1] + 1) * res * 2) for p in traj]; dd.line(pts, fill=col, width=4)
                for t, (x, y) in enumerate(pts):
                    dd.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(0, 0, 0)); dd.text((x + 9, y - 9), str(t), fill=col)
            dd.text((8, 8), f"{scn} {idx[i]['video_id']}   white = truth   orange = predictor (token decoder fit on h, test only)   dots = future tubelet 0..7", fill=(255, 255, 0))
            bg.save(OUTROOT / "traj_tok" / f"{scn}_{idx[i]['video_id']}.png")
        print(f"  [saved] {scn} x2 -> overlay_tok/, traj_tok/")


if __name__ == "__main__":
    main()
