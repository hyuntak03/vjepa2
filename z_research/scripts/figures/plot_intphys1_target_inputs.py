#!/usr/bin/env python3
"""IntPhys 1 · 88.89 칸 (skip2_w32) 에서 **target encoder 가 실제로 받는 32 장** 을 원 해상도 (256×256) 그대로 저장한다.

채점기와 같은 경로로 텐서를 만든다:
  cfg   = 88.89 실행의 _resolved.yaml (z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32/)
  clip  = WMADataset(cfg).clip(i)        PNG 288×288 → bilinear (antialias 없음) 256×256 → ImageNet 정규화
  창    = _intphys1_windows(... yaml 의 skip/창/C/stride/frame_budget ...)   (eval.py 와 같은 생성기)
  입력  = clip[:, context + future]      eval.py 의 `x = clips[v][:, full_of[st]]` 와 같다 → target_encoder(x)
저장할 때만 정규화를 되돌린다 (x·std + mean → 0..255). 그 밖의 가공은 없다.

출력 (z_research/Benchmarks/figures/intphys1_windows/target_inputs/):
  start{SS}.png   시작점 SS (raw) 창 하나. 행 = 영상 (pos/imp 쌍 3 개), 열 = 창 32 장 (256 px 원본).
                  테두리 파랑 = context / 주황 = future — 그 영상·시작점에서 Filtered 가 고른 C* 기준 (per_window.json 의 min).
                  target encoder 는 색과 무관하게 32 장 전부를 받는다. ▼ = 그 쌍의 pos/imp PNG 가 처음 갈린 뒤 첫 장.
  start{SS}.gif   같은 창을 32 장 순서대로 재생 (3 쌍 × pos|imp 격자, 칸마다 slot · raw · context/future 표시)

  python z_research/scripts/figures/plot_intphys1_target_inputs.py [--pairs O1_01_2:O1_01_1 O2_09_1:O2_09_3 O3_03_4:O3_03_2]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evals.world_model_analysis.data import WMADataset, MEAN, STD  # noqa: E402
from evals.world_model_analysis.eval import _intphys1_windows       # noqa: E402

RUN = ROOT / "z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32"
OUT = ROOT / "z_research/Benchmarks/figures/intphys1_windows/target_inputs"
COMBO, SKIP, WZ = "skip2_w32", 2, 32
BLUE, ORANGE, INK = (42, 120, 214), (235, 104, 52), (34, 34, 34)
FONT = font_manager.findfont("DejaVu Sans")


def font(sz):
    return ImageFont.truetype(FONT, sz)


def load():
    cfg = yaml.safe_load(open(RUN / "_resolved.yaml"))
    W = cfg["surprise"]["intphys1"]
    assert SKIP in W["frame_skips"] and WZ in W["window_sizes"]
    ds = WMADataset(cfg)
    cl = [m * WZ // 16 for m in W["context_mult"]]
    wins = _intphys1_windows(ds.n_frames, SKIP, WZ, cl, int(W["stride"]), int(cfg["model"].get("tubelet_size", 2)),
                             str(W.get("frame_budget", "full")))
    by = {}
    for C, cf, tf in wins:
        by.setdefault(cf[0], {})[C] = (cf, tf)
    for s in by:                                  # 같은 시작점이면 C 와 무관하게 target 입력이 같다 (eval.py full_of)
        fulls = {tuple(cf + tf) for cf, tf in by[s].values()}
        assert len(fulls) == 1 and len(next(iter(fulls))) == WZ
    return cfg, ds, by, cl


def chosen_C(vid):
    best = {}
    for combo, s, C, sur in json.load(open(RUN / "per_window.json"))["windows"][vid]:
        if combo == COMBO and (s not in best or sur < best[s][1]):
            best[s] = (C, sur)
    return {s: c for s, (c, _) in best.items()}


def to_uint8(clip):                               # (3, T, H, W) 정규화 텐서 → (T, H, W, 3) uint8
    x = (clip * STD + MEAN).clamp(0, 1)
    return (x.permute(1, 2, 3, 0).numpy() * 255 + 0.5).astype(np.uint8)


def first_div(a, b):
    d = np.nonzero([not np.array_equal(a[t], b[t]) for t in range(len(a))])[0]
    return int(d[0]) if len(d) else None


def simple(ds, by, vid, px=128):
    """행 = 창 (시작 raw), 열 = 그 창에서 target encoder 가 받는 32 장. 모델 입력 텐서를 정규화만 되돌려 px 로 줄인다."""
    idx = {r.video_id: i for i, r in enumerate(ds.records)}
    fr = to_uint8(ds.clip(idx[vid]))
    lw, top, gap, lab = 150, 50, 22, 18
    starts = sorted(by)
    img = Image.new("RGB", (lw + WZ * (px + 2), top + len(starts) * (px + gap)), (255, 255, 255)); dr = ImageDraw.Draw(img)
    dr.text((8, 8), f"{vid} · IntPhys 1 {COMBO} (88.89 cell): frames given to the target encoder, one row per window (number = raw frame index)",
            fill=INK, font=font(22))
    for r, s in enumerate(starts):
        cf, tf = next(iter(by[s].values()))
        y = top + r * (px + gap)
        dr.text((8, y + px // 2 - 12), f"window {r + 1}\nstart raw {s}", fill=INK, font=font(18))
        for k, f in enumerate(cf + tf):
            x = lw + k * (px + 2)
            img.paste(Image.fromarray(fr[f]).resize((px, px), Image.BILINEAR), (x, y))
            dr.text((x + px // 2 - 10, y + px + 1), str(f), fill=INK, font=font(lab - 4))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT.parent / f"target_encoder_frames_{vid}.png"
    img.save(out, optimize=True); print("→", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["O1_01_2:O1_01_1", "O2_09_1:O2_09_3", "O3_03_4:O3_03_2"],
                    help="pos:imp 쌍")
    ap.add_argument("--gif-ms", type=int, default=300)
    ap.add_argument("--simple", default="", help="영상 하나만: 행 = 창 9 개, 열 = target encoder 가 받는 32 장 (색·C 표시 없음)")
    a = ap.parse_args()
    cfg, ds, by, cl = load()
    if a.simple:
        return simple(ds, by, a.simple)
    idx = {r.video_id: i for i, r in enumerate(ds.records)}
    rows = []                                                      # (vid, role, frames uint8, C*, div)
    for pr in a.pairs:
        pos, imp = pr.split(":")
        fp, fi = to_uint8(ds.clip(idx[pos])), to_uint8(ds.clip(idx[imp]))
        dv = first_div(fp, fi)
        rows += [(pos, "possible", fp, chosen_C(pos), dv), (imp, "impossible", fi, chosen_C(imp), dv)]
    H = W_ = int(cfg["data"]["resolution"])
    assert rows[0][2].shape[1:3] == (H, W_)
    OUT.mkdir(parents=True, exist_ok=True)
    bd, tg, lw, top, rg = 5, 10, 330, 96, 26      # 테두리, 튜블릿 틈, 왼쪽 라벨 폭, 머리, 행 틈
    cell = W_ + 2 * bd
    Wc = lw + WZ * cell + (WZ // 2 - 1) * tg + 10
    Hc = top + len(rows) * (cell + rg) + 10
    f_t, f_c, f_r = font(30), font(17), font(22)
    summary = {}
    for s in sorted(by):
        frames = next(iter(by[s].values()))
        frames = frames[0] + frames[1]                              # raw 인덱스 32 개
        img = Image.new("RGB", (Wc, Hc), (255, 255, 255)); dr = ImageDraw.Draw(img)
        dr.text((12, 8), f"IntPhys 1 · {COMBO} (88.89 cell) · window start raw {s}: the 32 frames given to the target encoder "
                         f"(256x256 model input, de-normalized only). Blue = context / orange = future at C* chosen by Filtered; "
                         f"C candidates {cl}", fill=INK, font=f_t)
        xs = [lw + k * cell + (k // 2) * tg for k in range(WZ)]
        for k, f in enumerate(frames):
            dr.text((xs[k] + 4, top - 30), f"slot {k + 1} · raw {f}", fill=INK, font=f_c)
        summary[s] = {}
        for r, (vid, role, fr, Cs, dv) in enumerate(rows):
            y = top + r * (cell + rg)
            C = Cs[s]; summary[s][vid] = C
            dr.text((10, y + cell // 2 - 30), f"{vid}\n{role} · C*={C}", fill=INK, font=f_r)
            for k, f in enumerate(frames):
                dr.rectangle([xs[k], y, xs[k] + cell - 1, y + cell - 1], fill=BLUE if k < C else ORANGE)
                img.paste(Image.fromarray(fr[f]), (xs[k] + bd, y + bd))
            bx = xs[C] - tg // 2 - 1 if C < WZ else xs[-1] + cell
            dr.rectangle([bx - 2, y - 6, bx + 2, y + cell + 6], fill=INK)
            if dv is not None:
                ks = [k for k, f in enumerate(frames) if f >= dv]
                if ks:
                    cx = xs[ks[0]] + cell // 2
                    dr.polygon([(cx - 9, y - 14), (cx + 9, y - 14), (cx, y - 2)], fill=INK)
        img.save(OUT / f"start{s:02d}.png", optimize=True)
        # GIF — 쌍마다 한 행, pos | imp
        n = len(rows) // 2
        gw, gh, pad, hd = 2 * cell + 20 + 250, n * (cell + 34) + 60, 20, 60
        seq = []
        for k, f in enumerate(frames):
            g = Image.new("RGB", (gw, gh), (255, 255, 255)); d = ImageDraw.Draw(g)
            d.text((10, 8), f"start raw {s} · slot {k + 1}/32 · raw frame {f}", fill=INK, font=font(24))
            for pi in range(n):
                for c in range(2):
                    vid, role, fr, Cs, dv = rows[2 * pi + c]
                    x0, y0 = 250 + c * (cell + pad), hd + pi * (cell + 34)
                    ctx = k < Cs[s]
                    d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=BLUE if ctx else ORANGE)
                    g.paste(Image.fromarray(fr[f]), (x0 + bd, y0 + bd))
                    d.text((x0 + 4, y0 + cell + 2), f"{role} · {'context' if ctx else 'future'} (C*={Cs[s]})", fill=INK, font=font(17))
                d.text((8, hd + pi * (cell + 34) + cell // 2 - 12), rows[2 * pi][0][:5], fill=INK, font=font(24))
            seq.append(g)
        seq[0].save(OUT / f"start{s:02d}.gif", save_all=True, append_images=seq[1:], duration=a.gif_ms, loop=0)
    (OUT / "chosen_C.json").write_text(json.dumps({"pairs": a.pairs, "C_star_by_start_raw": summary,
                                                   "first_div_raw": {r[0]: r[4] for r in rows}}, indent=1))
    print(json.dumps(summary, indent=1)); print("→", OUT)


if __name__ == "__main__":
    main()
