#!/usr/bin/env python3
"""릴리즈 vs post-FT predictor 가 만든 미래를 한 clip 에서 나란히 GIF — 32 샘플 프레임 위에 진실(흰 원), 자의 선형 읽기(점), 자 attention 최댓값 칸(십자, 파라미터 0) + 히트맵. 릴리즈 = 파랑, post-FT = 주황.
⚠️ 자는 릴리즈 p 로 학습된 것이라 post-FT p 에서 선형 읽기(점)는 y 가 위로 치우친다 — post-FT 는 십자(attention 최댓값)로 읽을 것.

입력: predictor_training/predictor_IntPhysGenV11_PFT/exp_results/{release,pft}/{rollout_v2,v11}/preds.npz (pft_ruler_direct.py 산출)
미래 8 슬롯 = 샘플 16..31 (튜블릿 = 샘플 2 장). 슬롯 i 의 읽기는 샘플 16+2i, 17+2i 에 같이 그린다. 왼쪽 = 릴리즈, 오른쪽 = post-FT (히트맵은 그 슬롯의 자 attention, 256 토큰 → 16×16).
출력: predictor_training/predictor_IntPhysGenV11_PFT/figures/gif/<set>/<clip>.gif

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_pft_pair_gif.py --set v11 --clip v11_moving_occlusion_flat_k4_00000_pos_a
  $P z_research/scripts/figures/plot_pft_pair_gif.py --set rollout_v2 --clip rollout2_flat_v_00000_pos_roll --n 2   # --n: 시나리오/조건별 앞 n 개 자동
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                   # noqa: E402
SET = ROOT / "z_research/predictor_training/predictor_IntPhysGenV11_PFT"
FRAMES = {"v11": Path("/local_datasets/world/world_analysis/IntPhysGen_v11"), "rollout_v2": Path("/local_datasets/world/world_analysis/RollOut_v2")}
INDEX = {"v11": ROOT / "data_csv/intphysgen_v11_full/index_probe.csv", "rollout_v2": ROOT / "data_csv/rollout_v2/index_probe.csv"}
META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
PW, RES = 288, 144.0; WHITE, BLUE, ORANGE = (255, 255, 255), (42, 120, 214), (235, 104, 52)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception: FONT = ImageFont.load_default()


def load(setname):
    out = {}
    for pr in ("release", "pft"):
        z = np.load(SET / "exp_results" / pr / setname / "preds.npz", allow_pickle=True); vid, pred, att = z["video_id"], z["pred"], z["attn"]   # npz 는 키마다 다시 읽으므로 한 번만 꺼낸다
        out[pr] = {v: (pred[i], att[i]) for i, v in enumerate(vid)}
    return out


def truth_px(setname, row, meta):
    if setname == "v11":
        m = meta[row["video_id"]]; return np.stack([rt.arr(m["object_px_x_by_sample"]), rt.arr(m["object_px_y_by_sample"])], -1)
    return np.stack([rt.arr(row["px_x_by_sample"]), rt.arr(row["px_y_by_sample"])], -1)


def heat(a):
    g = (a.astype(np.float32).reshape(16, 16) / max(a.max(), 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(g).resize((PW, PW), Image.NEAREST)


HEAT, MS = True, 180


def panel(frame, t, tru, pred, att, color, label):
    im = frame.copy()
    if t >= 16 and HEAT:
        i = (t - 16) // 2
        h = heat(att[i]); im = Image.blend(im, Image.merge("RGB", (h, h, h)).point(lambda v: v), 0.35)
    dd = ImageDraw.Draw(im)
    tub = tru.reshape(16, 2, 2).mean(1)
    for j in range(min(t // 2 + 1, 16)):                              # 진실 궤적 (튜블릿 단위) 까지
        x, y = tub[j]; dd.ellipse([x - 3, y - 3, x + 3, y + 3], outline=WHITE, width=2)
    if t >= 16:
        for i in range((t - 16) // 2 + 1):
            x, y = (pred[i] + 1) * RES; dd.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)            # 선형 읽기 (post-FT p 에서는 y 편향 있음 — 단서)
            k = int(att[i].astype(np.float32).argmax()); cx, cy = (k % 16) * 18 + 9, (k // 16) * 18 + 9  # attention 최댓값 칸 (파라미터 0)
            dd.line([cx - 7, cy, cx + 7, cy], fill=color, width=3); dd.line([cx, cy - 7, cx, cy + 7], fill=color, width=3)
    dd.text((6, 4), label, fill=color, font=FONT); dd.text((6, PW - 18), f"sample {t:2d}  {'context' if t < 16 else 'future slot %d' % ((t-16)//2)}", fill=WHITE, font=FONT)
    return im


def make(setname, vid, data, rows, meta, out):
    r = rows[vid]; tru = truth_px(setname, r, meta); frs = []
    fnums = [int(v) for v in r["sample_frames"].split()] if "sample_frames" in r else list(range(0, 96, 3))
    for t, f in enumerate(fnums):
        fr = Image.open(FRAMES[setname] / r["file_name"] / f"{f:06d}.png").convert("RGB").resize((PW, PW), Image.NEAREST)
        L = panel(fr, t, tru, *data["release"][vid], BLUE, "release predictor"); R = panel(fr, t, tru, *data["pft"][vid], ORANGE, "post-FT (v11) predictor")
        cv = Image.new("RGB", (PW * 2 + 8, PW + 26), (20, 20, 20)); cv.paste(L, (0, 0)); cv.paste(R, (PW + 8, 0))
        dd = ImageDraw.Draw(cv); dd.rectangle([0, PW + 4, int((PW * 2 + 8) * (t + 1) / 32), PW + 22], fill=WHITE if t < 16 else ORANGE); dd.text((8, PW + 6), vid, fill=(0, 0, 0) if t < 16 else (255, 255, 255), font=FONT)
        frs.append(cv)
    out.mkdir(parents=True, exist_ok=True); p = out / f"{vid}.gif"
    frs[0].save(p, save_all=True, append_images=frs[1:], duration=[MS] * 31 + [1000], loop=0); print(f"→ {p}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, help="v11 | v11_shape | v11_color | rollout_v2"); ap.add_argument("--clip", nargs="*", default=[]); ap.add_argument("--n", type=int, default=0); ap.add_argument("--no-heat", action="store_true", help="attention 히트맵 끄기"); ap.add_argument("--ms", type=int, default=180, help="프레임당 ms (작을수록 빠름)"); a = ap.parse_args()
    global HEAT, MS; HEAT, MS = not a.no_heat, a.ms
    kind = "v11" if a.set.startswith("v11") else "rollout_v2"
    data = load(a.set); rows = {r["video_id"]: r for r in csv.DictReader(INDEX[kind].open())}; meta = {r["name"]: r for r in csv.DictReader(META.open())} if kind == "v11" else {}
    vids = list(a.clip)
    if a.n:
        seen = {}
        for v in data["pft"]:
            key = rows[v]["scenario"] if kind == "rollout_v2" else f'{rows[v]["condition"]}_k{rows[v]["sym_k"]}'
            if seen.get(key, 0) < a.n: seen[key] = seen.get(key, 0) + 1; vids.append(v)
    for v in vids: make(kind, v, data, rows, meta, SET / "figures/gif" / a.set)


if __name__ == "__main__":
    main()
