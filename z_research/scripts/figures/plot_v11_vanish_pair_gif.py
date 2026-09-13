#!/usr/bin/env python3
"""v11 — 한 클립에 대해 encoder(z) 와 predictor(p) 위치 읽기를 나란히 놓은 발표용 GIF.

`plot_v11_vanish_readout.py` 가 저장한 readout.npz (같은 학습셋 자, ROLLOUT2_TRAIN) 를 읽어 그린다. 왼쪽 = context encoder z (32 frames 전부 봄),
오른쪽 = predictor p (앞 16 frames 만 봄). 진실·읽기 모두 tubelet (2 frames) 단위. 아래 time bar (문맥 / 미래 / 가려진 구간).
출력: figures/<train>/v11_vanish/pair/<motion>_<timing>/<clip>.gif  (v2 는 pair/v2/)

  python z_research/scripts/figures/plot_v11_vanish_pair_gif.py --motion ramp --k 3 --clip v11_moving_occlusion_k3_02689_pos_a
  python z_research/scripts/figures/plot_v11_vanish_pair_gif.py --motion flat --timing early --k 4 --clip v11_moving_occlusion_flat_early_k4_02464_pos_a
  python z_research/scripts/figures/plot_v11_vanish_pair_gif.py --dataset v2 --clip rollout2_ledge_04126_pos_roll      # RollOut_v2 test 클립 (캐시에서 z·p 를 바로 읽음)
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                 # noqa: E402  (ROLLOUT2_TRAIN → figures/<train>)

FRAMES = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
FIG = rt.FIG_ROOT / "v11_vanish"
R, Z = 288, 2; PW = R * Z; GAP, PAD = 28, 24; HEAD, TITLE, FOOT = 64, 34, 84
BG, INK, MUTED = (18, 18, 22), (240, 240, 240), (150, 150, 160)
WHITE, GREEN, ORANGE, BLUE, FUT, HID = (255, 255, 255), (27, 175, 122), (235, 104, 52), (42, 120, 214), (200, 200, 205), (70, 70, 78)
SAMPLES = list(range(0, 94, 3))
FONT = fm.findfont("DejaVu Sans"); FONT_B = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def load(rep, motion, K, clip, timing="late"):
    z = np.load(FIG / rep / (motion if timing == "late" else f"{motion}_{timing}") / f"k{K}" / "readout.npz", allow_pickle=True)
    idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}; blk = idx[clip]["block_id"]
    i = int(np.where(z["block"] == blk)[0][0]); return z["pred"][i], z["truth"][i]          # pred (n_tub, 2) px, truth (16, 2) px (tubelet 평균)


def draw_panel(im, t, truth, pred, col, label_no_pred):
    dd = ImageDraw.Draw(im); tub = t // 2; n_tub = len(pred); j0 = 16 - n_tub
    tp = [(truth[i][0] * Z, truth[i][1] * Z) for i in range(tub + 1)]
    if len(tp) > 1:
        dd.line(tp, fill=WHITE, width=2)
    for x, y in tp[:-1]:
        dd.ellipse([x - 3, y - 3, x + 3, y + 3], outline=WHITE, width=1)
    x, y = tp[-1]; dd.ellipse([x - 10, y - 10, x + 10, y + 10], outline=WHITE, width=3)
    j = tub - j0
    if j >= 0:
        rp = [(pred[i][0] * Z, pred[i][1] * Z) for i in range(j + 1)]
        if len(rp) > 1:
            dd.line(rp, fill=col, width=3)
        for x, y in rp[:-1]:
            dd.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        x, y = rp[-1]; dd.ellipse([x - 11, y - 11, x + 11, y + 11], fill=col, outline=WHITE, width=2)
    else:
        dd.text((12, PW - 34), label_no_pred, fill=col, font=font(15, True))


def load_v2(clip):
    """RollOut_v2 (test) 클립: 캐시에서 z (16 튜블릿) 와 p (8 튜블릿) 를 자 (attn.pt) 로 바로 읽는다. 진실은 index 의 px 라벨 (튜블릿 평균)."""
    import json, torch
    from rollout2_attn_readout import AttnReadout
    idx = {r["video_id"]: r for r in csv.DictReader(rt.INDEX.open())}; r = idx[clip]
    C = Path("/local_datasets/world/world_analysis/cache"); out = {}
    for rep, cache, fname in (("z", C / "rollout_v2_ctx32_vith", "isolated_ctx_0_32.npy"), ("p", C / "rollout_v2_vith", "predictor.npy")):
        row = json.loads((cache / "meta.json").read_text())["video_ids"].index(clip)
        tok = torch.from_numpy(np.asarray(np.load(cache / fname, mmap_mode="r")[row])).cuda().float().reshape(-1, 256, 1280)
        m = AttnReadout().cuda(); m.load_state_dict(torch.load(rt.RES_ROOT / "attentive_pooling" / rep / "attn.pt", map_location="cuda")); m.eval()
        with torch.no_grad():
            out[rep] = (m(tok)[0].cpu().numpy() + 1) * 144.0
    truth = np.stack([arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])], -1).reshape(16, 2, 2).mean(1)
    return out["z"], out["p"], truth, r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["v11", "v2"], default="v11"); ap.add_argument("--motion", default="ramp"); ap.add_argument("--timing", default="late", choices=["late", "early", "mid"]); ap.add_argument("--k", default="3")
    ap.add_argument("--clip", required=True); ap.add_argument("--ms", type=int, default=380); a = ap.parse_args()
    global FRAMES
    if a.dataset == "v11":
        idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}; r = idx[a.clip]
        pz, truth = load("z", a.motion, a.k, a.clip, a.timing); pp, _ = load("p", a.motion, a.k, a.clip, a.timing)
        meta = {m["name"]: m for m in csv.DictReader((FRAMES / "metadata.csv").open())}[a.clip]                       # 가려진 샘플 = metadata hidden_start/end (raw frame) 에 드는 32 샘플
        hs, he = (float(meta["hidden_start"]), float(meta["hidden_end"])) if meta["hidden_start"] else (1, 0); hid = {t for t in range(32) if hs <= 3 * t <= he}
        cond = {"flat": "flat ground, constant velocity", "ramp": "ramp, constant acceleration", "static": "static, the object does not move"}[a.motion]
        when = {"late": "across the context/future boundary", "early": "early in the context only", "mid": "in the middle of the context only"}[a.timing]
        sub = f"IntPhysGen v11 · {cond} · occluded k={a.k}, {when}   (possible clip)"
    else:
        FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2"); pz, pp, truth, r = load_v2(a.clip); hid = set()
        scen = r["scenario"]; desc = {"ledge": "rolls off a ledge and falls", "wall": "rolls into a wall and stops", "ramp_a": "rolls down a ramp", "arc": "flies on a parabola",
                                      "fall": "is thrown upward and falls", "flat_v": "rolls at constant velocity", "flat_a": "rolls with constant acceleration"}.get(scen, scen)
        sub = f"RollOut v2 · {scen}: the object {desc}   (possible clip)"
    W = PAD * 2 + PW * 2 + GAP; H = HEAD + TITLE + PW + FOOT; frames = []
    for t, f in enumerate(SAMPLES):
        cv = Image.new("RGB", (W, H), BG); dd = ImageDraw.Draw(cv)
        dd.text((PAD, 14), "Where does the model put the object?", fill=INK, font=font(22, True))
        dd.text((PAD, 42), sub, fill=MUTED, font=font(14))
        for col_i, (im_x, ttl, col) in enumerate(((PAD, "Encoder  z  — sees all 32 frames", GREEN), (PAD + PW + GAP, "Predictor  p  — sees only the first 16 frames", ORANGE))):
            dd.rectangle([im_x, HEAD + 6, im_x + 8, HEAD + TITLE - 6], fill=col); dd.text((im_x + 16, HEAD + 6), ttl, fill=INK, font=font(17, True))
            im = Image.open(FRAMES / r["file_name"] / f"{f:06d}.png").convert("RGB").resize((PW, PW), Image.NEAREST)
            draw_panel(im, t, truth, pz if col_i == 0 else pp, col, "" if col_i == 0 else "no prediction yet (context frames)")
            cv.paste(im, (im_x, HEAD + TITLE))
        # footer: time bar + legend
        y0 = HEAD + TITLE + PW + 16; x0, x1 = PAD, W - PAD - 300; seg = (x1 - x0) / 32
        for i in range(32):
            dd.rectangle([x0 + i * seg, y0, x0 + (i + 1) * seg - 2, y0 + 12], fill=HID if i in hid else (BLUE if i < 16 else FUT))
        cx = x0 + (t + 0.5) * seg; dd.polygon([(cx, y0 + 14), (cx - 7, y0 + 26), (cx + 7, y0 + 26)], fill=WHITE)
        dd.text((x0, y0 + 30), "context (observed)", fill=BLUE, font=font(13)); dd.text((x0 + 16 * seg + 4, y0 + 30), "future", fill=FUT, font=font(13))
        if hid:
            dd.text((x0 + min(hid) * seg, y0 + 46), "object hidden", fill=(170, 170, 180), font=font(12))
        dd.text((x1 - 190, y0 + 30), f"frame {f:2d} / 93", fill=MUTED, font=font(13))
        lx = W - PAD - 270; dd.ellipse([lx, y0 + 2, lx + 14, y0 + 16], outline=WHITE, width=2); dd.text((lx + 22, y0), "true position", fill=INK, font=font(13))
        dd.ellipse([lx, y0 + 24, lx + 14, y0 + 38], fill=GREEN); dd.text((lx + 22, y0 + 22), "read from encoder z", fill=INK, font=font(13))
        dd.ellipse([lx, y0 + 46, lx + 14, y0 + 60], fill=ORANGE); dd.text((lx + 22, y0 + 44), "read from predictor p", fill=INK, font=font(13))
        frames.append(cv)
    ref = Image.new("RGB", (W, H * 3)); [ref.paste(frames[i], (0, H * j)) for j, i in enumerate((4, 18, 31))]
    d = ImageDraw.Draw(ref)                                          # 마커 색이 팔레트에 확실히 들어가도록 큰 견본을 그린다 (자홍 물체 등에 밀리지 않게)
    for k, c in enumerate((GREEN, ORANGE, WHITE, BLUE, FUT, HID, BG, INK)):
        d.rectangle([20 + 150 * k, 20, 160 + 150 * k, 200], fill=c)
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT); q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    out = FIG / "pair" / (f"{a.motion}_{a.timing}" if a.dataset == "v11" else "v2"); out.mkdir(parents=True, exist_ok=True); path = out / f"{a.clip}.gif"
    q[0].save(path, save_all=True, append_images=q[1:], duration=[a.ms] * 31 + [1600], loop=0); print(f"→ {path}  ({W}x{H}, {len(q)} frames)")


if __name__ == "__main__":
    main()
