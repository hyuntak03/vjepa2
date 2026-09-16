#!/usr/bin/env python3
"""RollOut_v2 — `plot_rollout2_readout.py` 의 traj PNG 를 움직이는 GIF 로. 같은 clip, 같은 자(preds.npz).

프레임 32 샘플 (raw 0, 3, …, 93) 을 차례로 보여 주며
  흰색   정답 궤적 (index 의 px 라벨, tubelet = 2 샘플 평균) — 문맥부터 현재 tubelet 까지 누적
  주황   읽은 위치 (preds.npz 의 pred, 미래 tubelet 8개) — 미래 구간(샘플 16~) 부터 누적
  아래   time bar (문맥 / 미래, 현재 위치 삼각형)
기본 대상은 figures/<train>/<pooling>/<rep>/traj/ 에 이미 있는 PNG 의 clip 들. 출력 figures/<train>/<pooling>/<rep>/traj_gif/<같은 이름>.gif

  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_traj_gif.py [--pooling attentive_pooling] [--rep p] [--clips ID ...]
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_test_readout as rt                          # noqa: E402

FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2")
INDEX = rt.ROOT / "data_csv/rollout_v2/index_probe.csv"
SAMPLES = list(range(0, 94, 3))
Z = 2; PW = 288 * Z; PAD, HEAD, FOOT = 24, 70, 92
BG, INK, MUTED = (18, 18, 22), (240, 240, 240), (150, 150, 160)
WHITE, ORANGE, GREEN, BLUE, FUT = (255, 255, 255), (235, 104, 52), (27, 175, 122), (42, 120, 214), (200, 200, 205)
COL = {"p": ORANGE, "z": GREEN, "h": BLUE}
NAME = {"p": "predictor p (sees only the first 16 frames)", "z": "context encoder z (32 frames)", "h": "target encoder h (32 frames)"}
DESC = {"ledge": "rolls off a ledge and falls", "wall": "rolls into a wall and stops", "ramp_a": "rolls down a ramp",
        "arc": "flies on a parabola", "fall": "is thrown upward and falls", "flat_v": "rolls at constant velocity",
        "flat_a": "rolls with constant acceleration"}
FONT = fm.findfont("DejaVu Sans"); FONT_B = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def trail(dd, pts, col, width, dot, last):
    if len(pts) > 1:
        dd.line(pts, fill=col, width=width)
    for x, y in pts[:-1]:
        dd.ellipse([x - dot, y - dot, x + dot, y + dot], fill=col)
    x, y = pts[-1]
    dd.ellipse([x - last, y - last, x + last, y + last], fill=col, outline=(0, 0, 0) if col == WHITE else WHITE, width=2)


def make_gif(r, pred, rep, pooling, out, ms):
    truth = np.stack([arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])], -1).reshape(16, 2, 2).mean(1) * Z   # (16, 2) 화면 px x Z
    pr = (pred + 1) * rt.RES * Z                                                                                # (8, 2)
    scen = r["scenario"]; W = PAD * 2 + PW; H = HEAD + PW + FOOT; frames = []
    for t, f in enumerate(SAMPLES):
        cv = Image.new("RGB", (W, H), BG); dd = ImageDraw.Draw(cv)
        dd.text((PAD, 12), f"RollOut v2 · {scen}: the object {DESC.get(scen, scen)}", fill=INK, font=font(19, True))
        dd.text((PAD, 40), f"{r['video_id']}   primary={float(r['primary']):g}  secondary={float(r['secondary']):g}", fill=MUTED, font=font(13))
        im = Image.open(FRAMES / r["file_name"] / f"{f:06d}.png").convert("RGB").resize((PW, PW), Image.NEAREST); di = ImageDraw.Draw(im)
        tub = t // 2
        trail(di, [tuple(truth[i]) for i in range(tub + 1)], WHITE, 3, 4, 10)
        j = tub - 8
        if j >= 0:
            trail(di, [tuple(pr[i]) for i in range(j + 1)], COL[rep], 4, 5, 11)
        else:
            di.text((12, PW - 30), "context frames: no prediction yet", fill=COL[rep], font=font(15, True))
        cv.paste(im, (PAD, HEAD))
        y0 = HEAD + PW + 16; x0, x1 = PAD, W - PAD - 260; seg = (x1 - x0) / 32
        for i in range(32):
            dd.rectangle([x0 + i * seg, y0, x0 + (i + 1) * seg - 2, y0 + 12], fill=BLUE if i < 16 else FUT)
        cx = x0 + (t + 0.5) * seg; dd.polygon([(cx, y0 + 14), (cx - 7, y0 + 26), (cx + 7, y0 + 26)], fill=WHITE)
        dd.text((x0, y0 + 30), "context (observed)", fill=BLUE, font=font(13)); dd.text((x0 + 16 * seg + 4, y0 + 30), "future", fill=FUT, font=font(13))
        dd.text((x0, y0 + 52), f"frame {f:2d} / 93", fill=MUTED, font=font(13))
        lx = W - PAD - 250
        dd.ellipse([lx, y0 + 2, lx + 14, y0 + 16], fill=WHITE); dd.text((lx + 22, y0), "true position", fill=INK, font=font(13))
        dd.ellipse([lx, y0 + 26, lx + 14, y0 + 40], fill=COL[rep]); dd.text((lx + 22, y0 + 24), f"read from {rep}", fill=INK, font=font(13))
        dd.text((lx, y0 + 50), NAME[rep], fill=MUTED, font=font(11))
        frames.append(cv)
    ref = Image.new("RGB", (W, H * 3)); [ref.paste(frames[i], (0, H * k)) for k, i in enumerate((4, 18, 31))]
    d = ImageDraw.Draw(ref)                                         # 마커 색이 팔레트에 확실히 들어가게 큰 견본
    for k, c in enumerate((COL[rep], WHITE, BLUE, FUT, BG, INK, MUTED)):
        d.rectangle([20 + 80 * k, 20, 90 + 80 * k, 200], fill=c)
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=[ms] * 31 + [1600], loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="attentive_pooling", choices=["attentive_pooling", "spatial_pooling"])
    ap.add_argument("--rep", default="p", choices=["p", "z", "h"])
    ap.add_argument("--clips", nargs="*", help="video_id 들. 기본 = traj/ 폴더의 PNG 와 같은 clip")
    ap.add_argument("--ms", type=int, default=380)
    a = ap.parse_args()
    fig = rt.FIG_ROOT / a.pooling / a.rep
    sub = "p/test" if (a.pooling == "spatial_pooling" and a.rep == "p") else a.rep
    z = np.load(rt.RES_ROOT / a.pooling / sub / "preds.npz")
    row = {v: i for i, v in enumerate(z["video_id"])}
    idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    if a.clips:
        jobs = [(f"{idx[c]['scenario']}_{c}", c) for c in a.clips]
    else:
        jobs = []
        for p in sorted((fig / "traj").glob("*.png")):
            c = re.search(r"(rollout2_.*)$", p.stem).group(1); jobs.append((p.stem, c))
    out = fig / "traj_gif"; out.mkdir(parents=True, exist_ok=True)
    for stem, c in jobs:
        make_gif(idx[c], z["pred"][row[c]], a.rep, a.pooling, out / f"{stem}.gif", a.ms)
        print(f"→ {out / (stem + '.gif')}", flush=True)


if __name__ == "__main__":
    main()
