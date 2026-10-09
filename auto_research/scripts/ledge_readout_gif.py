#!/usr/bin/env python3
"""RollOut_v2 ledge holdout 클립의 GIF — 앞 16 샘플 (encoder 가 본 문맥) 뒤 16 샘플 (못 본 미래) 위에 z 자 (과거만) 예측 · 진실 · 등속 외삽을 겹친다 (2026-10-02).
    python ledge_readout_gif.py --frames /local_datasets/world/world_analysis/RollOut_v2 --n 3 --out auto_research/figures/future_from_context
"""
import argparse, csv, re, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); RES = 144.0; T = 8
ap = argparse.ArgumentParser(); ap.add_argument("--frames", required=True); ap.add_argument("--n", type=int, default=3); ap.add_argument("--out", required=True); ap.add_argument("--scenario", default="ledge"); ap.add_argument("--rep", default="z"); ap.add_argument("--head", default="B")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
Z = np.load(ROOT / f"auto_research/exp_results/future_from_context/{'pB/' if (a.rep, a.head) == ('p', 'B') else ''}preds_{a.head}_{a.rep}.npz", allow_pickle=True)
idx = {r["video_id"]: r for r in csv.DictReader((ROOT / "data_csv/rollout_v2/index_probe.csv").open())}
arr = lambda s: np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])
sel = np.where(Z["scenario"] == a.scenario)[0]
# 속도 레벨이 다른 클립을 고른다 (primary 로 정렬해 고르게)
prim = np.array([float(idx[v]["primary"]) for v in Z["vid"][sel]]); order = sel[np.argsort(prim)]; pick = order[np.linspace(0, len(order) - 1, a.n).astype(int)]
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14); font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception: font = font_s = ImageFont.load_default()
frames_all = []; strip = []
for s in range(32):
    tiles = []
    for i in pick:
        vid = Z["vid"][i]; r = idx[vid]; d = Path(a.frames) / r["file_name"]; raw = int(arr(r["sample_frames"])[s])
        im = Image.open(d / f"{raw:06d}.png").convert("RGB").resize((288, 288)); dr = ImageDraw.Draw(im)
        px = arr(r["px_x_by_sample"]); py = arr(r["px_y_by_sample"])
        if s < 16:
            dr.rectangle([0, 0, 288, 20], fill=(30, 30, 30)); dr.text((6, 3), f"CONTEXT  sample {s+1}/16  (encoder input)", fill=(255, 255, 255), font=font)
        else:
            j = (s - 16) // 2; tr = (Z["truth"][i, j] + 1) * RES; pr = (Z["pred"][i, j] + 1) * RES
            last = (Z["ctx_last"][i] + 1) * RES
            # 등속 외삽: 문맥 마지막 두 슬롯 (샘플 12·13 → 14·15 평균) 차이로
            l6 = np.array([px[12:14].mean(), py[12:14].mean()]); l7 = np.array([px[14:16].mean(), py[14:16].mean()]); cv = l7 + (l7 - l6) * (j + 1)
            dr.rectangle([0, 0, 288, 20], fill=(120, 20, 20)); dr.text((6, 3), f"FUTURE  slot {j+1}/8  (NOT seen by encoder)", fill=(255, 255, 255), font=font)
            R_ = 9
            dr.ellipse([cv[0] - 5, cv[1] - 5, cv[0] + 5, cv[1] + 5], outline=(170, 170, 170), width=2)
            dr.ellipse([tr[0] - R_, tr[1] - R_, tr[0] + R_, tr[1] + R_], outline=(27, 175, 122), width=3)
            dr.ellipse([pr[0] - R_, pr[1] - R_, pr[0] + R_, pr[1] + R_], outline=(42, 120, 214), width=3); dr.line([pr[0] - 14, pr[1], pr[0] + 14, pr[1]], fill=(42, 120, 214), width=2); dr.line([pr[0], pr[1] - 14, pr[0], pr[1] + 14], fill=(42, 120, 214), width=2)
            dr.rectangle([0, 268, 288, 288], fill=(0, 0, 0)); dr.text((4, 271), f"blue + = {a.rep} readout   green o = truth   gray o = const-vel", fill=(255, 255, 255), font=font_s)
        dr.text((6, 24), f"speed {r['primary']}", fill=(255, 255, 0), font=font_s)
        tiles.append(im)
    W = Image.new("RGB", (288 * len(tiles) + 4 * (len(tiles) - 1), 288), (255, 255, 255))
    for k, t in enumerate(tiles): W.paste(t, (k * 292, 0))
    frames_all.append(W.resize((W.width * 2, W.height * 2), Image.NEAREST) if len(tiles) <= 2 else W)
    if s in (0, 8, 15, 17, 21, 25, 29, 31): strip.append(W)
dur = [250] * 16 + [450] * 16; gif = out / f"{a.scenario}_{a.rep}{a.head}_readout.gif"
frames_all[0].save(gif, save_all=True, append_images=frames_all[1:], duration=dur, loop=0)
S = Image.new("RGB", (strip[0].width, strip[0].height * len(strip)), (255, 255, 255))
for k, t in enumerate(strip): S.paste(t, (0, k * t.height))
S.save(out / f"{a.scenario}_{a.rep}{a.head}_strip.png"); print("saved", gif, "clips", [Z["vid"][i] for i in pick])
