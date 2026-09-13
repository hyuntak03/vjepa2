#!/usr/bin/env python3
"""RollOut_v2 — 학습셋-w (rollout2_test_readout.py 의 preds.npz) 로 읽은 predictor 위치를 프레임 위에 그린다.

  figures/<train>/<pooling>/<rep>/overlay/  미래 프레임 8장 (raw 48, 54, …, 90 = 튜블릿 앞 샘플) 위에 읽은 위치 (주황 원)
  figures/<train>/<pooling>/<rep>/traj/     마지막 미래 프레임 위에 궤적 두 줄 — 정답(흰) vs 읽기(주황), 점 = 튜블릿 0..7
  pooling ∈ spatial_pooling / attentive_pooling,  rep ∈ p (predictor) / z (context encoder 32 frames) / h (target encoder 32 frames)
시나리오마다 가능 클립 N 개 (primary 레벨을 고르게). 라벨 = 영어.

  python z_research/scripts/figures/plot_rollout2_readout.py [--per-scenario 3] [--pooling spatial_pooling|attentive_pooling] [--rep p|z|h]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_test_readout as rt                          # noqa: E402  (ROLLOUT2_TRAIN=v1|v2 → exp_results/<train>, figures/<train>)
ROOT = rt.ROOT
FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
ORANGE, WHITE, YEL = (235, 104, 52), (255, 255, 255), (255, 255, 0); R, RES = 288, 144.0
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
FR = [48 + 6 * t for t in range(8)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-scenario", type=int, default=3)
    ap.add_argument("--pooling", choices=["spatial_pooling", "attentive_pooling"], default="spatial_pooling")
    ap.add_argument("--rep", choices=["p", "z", "h"], default="p", help="p = predictor, z = context encoder 32 frames, h = target encoder 32 frames"); a = ap.parse_args()
    sub = "p/test" if (a.pooling == "spatial_pooling" and a.rep == "p") else a.rep
    PREDS = rt.RES_ROOT / f"{a.pooling}/{sub}/preds.npz"; FIG = rt.FIG_ROOT / f"{a.pooling}/{a.rep}"
    z = np.load(PREDS); pred, truth, vid = z["pred"], z["truth"], z["video_id"]
    idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    src = {"p": "predictor output p", "z": "context encoder z (32 frames)", "h": "target encoder h (32 frames)"}[a.rep]
    for d in ("overlay", "traj"):
        (FIG / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for scn in SCEN:
        ids = np.where((z["scenario"] == scn) & (z["plausible"] == 1))[0]
        prim = np.array([float(idx[vid[i]]["primary"]) for i in ids]); levels = np.unique(prim)
        pick = [rng.choice(ids[prim == lv]) for lv in levels[np.linspace(0, len(levels) - 1, a.per_scenario).round().astype(int)]]
        for i in pick:
            r = idx[vid[i]]; gt, pr = truth[i], pred[i]; tag = f"{scn}  {vid[i]}  primary={float(r['primary']):g} secondary={float(r['secondary']):g}"
            canvas = Image.new("RGB", (R * 8, R + 24), "white"); d = ImageDraw.Draw(canvas)
            d.text((6, 5), f"{tag}   orange = position read from {src} ({a.pooling} readout fit on RollOut_v2_training, test only)", fill=(0, 0, 0))
            for t, f in enumerate(FR):
                im = Image.open(FRAMES / r["file_name"] / f"{f:06d}.png").convert("RGB"); dd = ImageDraw.Draw(im)
                x, y = (pr[t, 0] + 1) * RES, (pr[t, 1] + 1) * RES; dd.ellipse([x - 10, y - 10, x + 10, y + 10], fill=ORANGE, outline=WHITE, width=2)
                dd.text((4, 4), f"t{t} f{f}", fill=YEL); canvas.paste(im, (R * t, 24))
            canvas.save(FIG / "overlay" / f"{scn}_{vid[i]}.png")
            bg = Image.open(FRAMES / r["file_name"] / f"{FR[-1]:06d}.png").convert("RGB").resize((R * 2, R * 2), Image.NEAREST); dd = ImageDraw.Draw(bg)
            for traj, col in ((gt, WHITE), (pr, ORANGE)):
                pts = [((p[0] + 1) * RES * 2, (p[1] + 1) * RES * 2) for p in traj]; dd.line(pts, fill=col, width=4)
                for t, (x, y) in enumerate(pts):
                    dd.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(0, 0, 0)); dd.text((x + 9, y - 9), str(t), fill=col)
            dd.text((8, 8), f"{tag}   white = truth   orange = {src} ({a.pooling} readout fit on training set)   dots = future tubelet 0..7", fill=YEL)
            bg.save(FIG / "traj" / f"{scn}_{vid[i]}.png")
        print(f"  [saved] {scn} x{len(pick)} -> {FIG.relative_to(ROOT)}/overlay/, traj/")


if __name__ == "__main__":
    main()
