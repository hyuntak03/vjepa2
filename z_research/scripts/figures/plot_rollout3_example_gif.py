#!/usr/bin/env python3
"""RollOut v3 — `attn` 자가 읽은 위치를 **z · h · p 열**로 보여 주는 예시 GIF.

`z` = context encoder 가 **창 전체를 보고** 읽은 위치 (predictor 에 입력을 주는 바로 그 encoder).
`h` = target encoder (EMA) 가 같은 창을 보고 읽은 위치 — **자가 이 창에 전이됐는지의 대조**.
`p` = predictor 가 **문맥만 보고 만든 미래**에서 읽은 위치.

⚠️ **물체가 없다고 하면 위치를 안 그린다** (2026-09-23 사용자 지시).
   자는 **attn**, 판정은 **presence 값 > 학습셋 문턱** (`summary.json`).
   v3 에는 음성이 0 개라 문턱 재조정은 못 하지만 **라벨 없는 검증**은 된다 — v3 는 물체가 내내
   화면 안이라 `z` 는 항상 "있다" 여야 한다. 그 오작동률은 `exp_results/windows/WINDOW_SUMMARY.md`.

- ○ 흰 테두리 = 진실 위치, ● 색 채움 = 자가 읽은 위치
- 문맥 구간은 테두리 파랑, 예측 구간은 주황. `p` 는 예측 구간에만 값이 있다
- `wall`·`ledge` 는 불가능 미래의 진실을 ○ 빨강으로 같이 찍는다 (문맥이 같으므로 p 는 하나뿐)

출력: z_research/RollOutV3/figures/examples/<scenario>_C<c>_P<p>.gif

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_rollout3_example_gif.py                    # 8 시나리오, C16/P32
  $P z_research/scripts/figures/plot_rollout3_example_gif.py --ctx 8 --prd 16 --speed 10.5
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, str(ROOT))
SRC = Path("/data2/local_datasets/world/world_analysis/RollOut_v3")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES as READOUT, fig   # noqa: E402  경로는 한 곳에서 (R3_DECODER / R3_OUT)
FIG = fig("examples")
SPLIT, RESN, CELL, S, D = 32, 144.0, 18.0, 256, 1280
# 자는 **attn** 으로 확정 (2026-09-23). 판정은 presence 값 > 학습셋 문턱.
# ⚠️ attn 의 지도(top1)는 v3 에서 평평해 판정에 못 쓴다 — presence 를 쓴다.
BIAS = json.loads((READOUT / "attn_bias_px.json").read_text())
SCEN = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
BG, INK, MUTED, DIM = (18, 18, 22), (240, 240, 240), (150, 150, 160), (70, 70, 78)
BLUE, ORANGE, GREEN, RED, WHITE = (42, 120, 214), (235, 104, 52), (27, 175, 122), (220, 70, 70), (255, 255, 255)
COL = {"z": BLUE, "h": GREEN, "p": ORANGE}
NAME = {"z": "z — context encoder (whole window)",
        "h": "h — target encoder, EMA (whole window)",
        "p": "p — predictor (context only)"}
FNT = lambda s, b=False: ImageFont.truetype(
    fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if b else "normal")), s)


def thresholds(d):
    """문턱을 **학습 산출물에서 읽는다.**

    ⚠️ 하드코딩하면 자를 다시 학습했을 때 **조용히 옛 문턱으로** 그려진다 (2026-09-24 에 이걸 막았다).
       값은 `summary.json` 의 `thr_val_fpr5` — 학습셋 val 음성 5 % 오탐 지점이다.
    """
    S = json.loads((d / "summary.json").read_text())
    return {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}


THR = None   # thresholds(READOUT) 로 채운다


def load_r3():
    spec = importlib.util.spec_from_file_location(
        "r3", ROOT / "z_research/scripts/analysis/rollout3_window_readout.py")
    m = importlib.util.module_from_spec(spec); sys.modules["r3"] = m; spec.loader.exec_module(m)
    return m


def pick(rows, scen, speed):
    cand = [r for r in rows if r["scenario"] == scen and r["role"] == "roll"]
    return min(cand, key=lambda r: (abs(float(r["primary"]) - speed), r["video_id"]))


def main(a):
    global THR
    THR = thresholds(READOUT)
    r3 = load_r3()
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    import decord  # noqa: F401

    rows = list(csv.DictReader((ROOT / "data_csv/rollout_v3/index.csv").open()))
    by_id = {r["video_id"]: i for i, r in enumerate(rows)}
    imp_of = {r["block_id"]: r for r in rows if r["role"] == "impossible_roll"}
    c, p_len = a.ctx, a.prd
    total, tc, tp = c + p_len, c // 2, p_len // 2
    lo = SPLIT - c
    dev = torch.device("cuda", 0)

    R = load_r3().__dict__["readout_cls"]()
    heads = {}
    for rep in a.reps:
        h = R("attn").to(dev).eval()
        h.load_state_dict(torch.load(READOUT / rep / "readout_attn.pt", map_location="cpu"))
        heads[rep] = h
    bundle = build_from_config({**r3.MODEL, "window_size": total}, dev)
    ds = WMADataset(dict(data={**r3.DATA, "frames_start": lo, "n_frames": total},
                         model={**r3.MODEL, "window_size": total},
                         features={"cache_dir": "/tmp"}, surprise={}))
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    arr = lambda s: np.array([float(v) for v in str(s).split()], np.float32)

    FIG.mkdir(parents=True, exist_ok=True)
    for scen in a.scenarios:
        rec = pick(rows, scen, a.speed); i = by_id[rec["video_id"]]
        with torch.no_grad():
            clip = ds.clip(i).unsqueeze(0).to(dev, dtype=bundle.dtype)
            o = extract_batch(clip, bundle, [{"base": "predictor"}], context_length=c,
                              mask_index=0, out_dtype=torch.float16)
            tok = {}
            if "p" in a.reps:
                tok["p"] = o["predictor"].reshape(tp, S, D).to(dev)
            for rep, mod in (("z", "context_encoder"), ("h", "target_encoder")):
                if rep not in a.reps:
                    continue
                f = getattr(bundle, mod)(clip); f = f[-1] if isinstance(f, list) else f
                tok[rep] = ln(f.float()).reshape(-1, S, D).half()          # 창 전체 튜블릿
            out = {}
            for rep, tk in tok.items():
                xy, pres, amap = heads[rep](tk.float())
                # ⚠️ 자의 출력은 정규화 좌표 (px/144 − 1) — 절대 px 는 **(v+1)*144** 다.
                #    차이만 볼 때는 오프셋이 상쇄돼 안 드러난다 (2026-09-23 에 여기서 144 px 어긋났다)
                # ⚠️ attn 의 좌표 편향을 뺀다 (학습셋 held-out 에서 잰 값, v3 라벨 미사용) — 2026-09-23
                b = np.array(BIAS[rep], np.float32)
                out[rep] = ((xy.cpu().numpy() + 1) * RESN - b, pres.cpu().numpy(), amap.max(-1).values.cpu().numpy())

        px, py = arr(rec["px_x_by_sample"]), arr(rec["px_y_by_sample"])
        inf = arr(rec["in_frame_by_sample"]) > 0
        qrec = imp_of.get(rec["block_id"]) if scen in ("wall", "ledge") else None
        qx, qy = (arr(qrec["px_x_by_sample"]), arr(qrec["px_y_by_sample"])) if qrec else (None, None)
        qin = (arr(qrec["in_frame_by_sample"]) > 0) if qrec else None

        PS, GX, HEAD, FOOT, PAD = a.panel, 14, 88, 72, 20
        W = PAD * 2 + len(a.reps) * PS + GX * (len(a.reps) - 1)
        H = HEAD + PS + FOOT
        sc_ = PS / 288.0
        frames = []
        for j in range(total):
            fr = lo + j; tb = j // 2
            im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
            d.text((PAD, 10), f"RollOut v3 · {scen}", font=FNT(16, True), fill=INK)
            d.text((PAD, 32), f"{rec['video_id']}   speed {rec['primary']} px/2f", font=FNT(10), fill=MUTED)
            d.text((PAD, 46), f"context {c}f [f{lo}-f{SPLIT-1}]  ·  prediction {p_len}f [f{SPLIT}-f{SPLIT+p_len-1}]",
                   font=FNT(10), fill=MUTED)
            phase = "CONTEXT" if fr < SPLIT else "PREDICTION"
            d.text((W - 200, 12), f"frame {fr:02d}   tubelet {tb:02d}", font=FNT(13, True), fill=INK)
            d.text((W - 200, 32), f"{phase}", font=FNT(12, True), fill=BLUE if fr < SPLIT else ORANGE)
            d.text((W - 200, 48), f"event @ f{SPLIT}", font=FNT(10), fill=MUTED)
            src = Image.open(SRC / "Images" / rec["file_name"] / f"{fr:06d}.png").convert("RGB")
            for col, rep in enumerate(a.reps):
                x0 = PAD + col * (PS + GX)
                im.paste(src.resize((PS, PS), Image.BILINEAR), (x0, HEAD))
                dd = ImageDraw.Draw(im)
                if inf[fr]:                                                # 진실
                    r_ = max(5, int(PS * 0.032))
                    dd.ellipse([px[fr]*sc_+x0-r_, py[fr]*sc_+HEAD-r_, px[fr]*sc_+x0+r_, py[fr]*sc_+HEAD+r_],
                               outline=WHITE, width=2)
                if qx is not None and fr > SPLIT and qin[fr]:              # 불가능 미래의 진실 (화면 안일 때만)
                    r_ = max(6, int(PS * 0.038))
                    dd.ellipse([qx[fr]*sc_+x0-r_, qy[fr]*sc_+HEAD-r_, qx[fr]*sc_+x0+r_, qy[fr]*sc_+HEAD+r_],
                               outline=RED, width=2)
                k = tb - tc if rep == "p" else tb                         # p 는 예측 구간에만 있다
                tag, note = "", ""
                if 0 <= k < len(out[rep][0]):
                    xy, pres, a1 = out[rep][0][k], out[rep][1][k], out[rep][2][k]
                    # ⚠️ 판정은 **지도**로. p 는 같은 튜블릿 z 의 top1 과 비교한다 (대조는 언제나 z)
                    ok = pres > THR[rep]
                    if ok:
                        r_ = max(5, int(PS * 0.030))
                        dd.ellipse([xy[0]*sc_+x0-r_, xy[1]*sc_+HEAD-r_, xy[0]*sc_+x0+r_, xy[1]*sc_+HEAD+r_],
                                   fill=COL[rep], outline=WHITE)
                        e = np.hypot(xy[0]-px[fr], xy[1]-py[fr]) if inf[fr] else np.nan
                        tag = f"present  {pres:+.1f}"
                        note = f"err {e:.0f} px" if inf[fr] else ""
                    else:
                        tag = f"ABSENT  {pres:+.1f}"
                        note = "position not drawn"
                else:
                    tag = "— (context only)" if rep == "p" else ""
                dd.rectangle([x0, HEAD, x0+PS-1, HEAD+PS-1],
                             outline=(BLUE if fr < SPLIT else ORANGE) if tag and not tag.startswith("—") else DIM,
                             width=3)
                dd.text((x0, HEAD - 18), NAME[rep], font=FNT(11, True), fill=COL[rep])
                dd.text((x0, HEAD + PS + 6), tag, font=FNT(12, True),
                        fill=RED if tag.startswith("ABSENT") else (COL[rep] if tag.startswith("present") else MUTED))
                dd.text((x0, HEAD + PS + 24), note, font=FNT(10), fill=MUTED)
            d.text((PAD, H - 30), "○ white = truth   ● filled = readout (attn, 2 heads)"
                                  + ("   ○ red = impossible future" if qx is not None else ""),
                   font=FNT(10), fill=MUTED)
            d.text((PAD, H - 16), "the position is hidden whenever the readout says the object is absent",
                   font=FNT(10), fill=MUTED)
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE))
        f_ = FIG / f"{scen}_C{c}_P{p_len}.gif"
        frames[0].save(f_, save_all=True, append_images=frames[1:], duration=a.ms, loop=0, optimize=True)
        print(f"→ {f_.name}  {rec['video_id']}  p present {100*(out['p'][1] > THR['p']).mean():.0f}% of {tp} tubelets")
    del bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=SCEN)
    ap.add_argument("--reps", nargs="+", default=["z", "h", "p"], choices=["z", "h", "p"])
    ap.add_argument("--ctx", type=int, default=16, choices=(4, 8, 16, 32))
    ap.add_argument("--prd", type=int, default=32, choices=(4, 8, 16, 32))
    ap.add_argument("--speed", type=float, default=8.25)
    ap.add_argument("--panel", type=int, default=300)
    ap.add_argument("--ms", type=int, default=140)
    main(ap.parse_args())
