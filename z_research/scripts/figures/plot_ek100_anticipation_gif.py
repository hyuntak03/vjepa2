#!/usr/bin/env python3
"""EK100 action anticipation 이 무슨 과제인지 보여주는 GIF — 문맥 / 1초 간격 / 정답 action.

한 GIF = 한 val segment. 세 구간을 8 fps 로 이어 붙인다 (재생 속도 = 실시간).

    CONTEXT   32 frames, 4.0s          모델이 보는 전부. 여기서 다음 action 을 맞혀야 한다
    GAP       1.0s                     아무도 안 본다 (anticipation time tau_a)
    ACTION    정답 구간 (최대 2.0s)      verb / noun / action 라벨이 이때 공개된다

문맥이 **action segment 시작 1.0초 전**에서 끝난다 (표준 EK100 프로토콜 =
`anticipation_point_mode: paper`). 프레임 인덱스는 채점 코드와 같은 산술로 뽑는다:
`af = sf - round(tau_a * fps)`, `indices = arange(af - 32*fstp, af, fstp)`, `fstp = int(fps/8)`.

프레임 위에 **모델 입력 영역**도 같이 그린다 — `center_crop`(256x256, 기본) 과
`cover`(256x448) 가 원본에서 어디를 쓰는지. 라벨은 영어다 (CLAUDE.md §8-4: 한글 폰트 없음).

    python z_research/scripts/figures/plot_ek100_anticipation_gif.py                  # 기본 6개
    python z_research/scripts/figures/plot_ek100_anticipation_gif.py --n 3 --video P01_11
    python z_research/scripts/figures/plot_ek100_anticipation_gif.py --list           # 후보만 본다
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
ANNOT = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
VIDEO_ROOT = Path("/data/dataset/EPIC-KITCHENS")
OUTDIR = REPO / "z_research/anticipation/EK100/figures/samples"

# 프로토콜 (configs/ek100_vith.yaml 과 같아야 한다)
FPC, FPS, TAU_A = 32, 8, 1.0
GAP_FRAMES = int(TAU_A * FPS)
ACTION_MAX_S = 2.0
DISP_W = 640  # GIF 가로

def _default_spatial():
    """본 config(ek100_vith)의 spatial 설정을 읽어 온다."""
    import yaml
    d = yaml.safe_load((REPO / "z_research/anticipation/EK100/configs/ek100_vith.yaml").read_text())
    e = d["experiment"]["data"]
    return e["spatial_mode"], int(e.get("width") or e["resolution"])


DEFAULT_MODE, DEFAULT_W = _default_spatial()

# 검증된 팔레트 (CLAUDE.md §8-4)
BLUE, ORANGE, GREEN, GRAY = (42, 120, 214), (235, 104, 52), (27, 175, 122), (130, 130, 130)

_MPL = Path(__import__("matplotlib").__file__).parent / "mpl-data/fonts/ttf"
F = lambda sz, bold=False: ImageFont.truetype(str(_MPL / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), sz)


def crop_box(w, h, out_h, out_w, mode):
    """dataloader 의 spatial 변환이 원본에서 쓰는 영역 (x0, y0, x1, y1)."""
    if mode == "center_crop":  # 짧은변 -> out*256/224 (= 292), 가운데 정사각
        s = (out_h * 256 / 224) / min(h, w)
    else:  # short_side: scale = max(out_h/h, out_w/w) -> 짧은변이 out_h 에 정확히 맞는다
        s = max(out_h / h, out_w / w)
    cw, ch = out_w / s, out_h / s
    return ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2)


def draw_frame(img, phase, t_rel, row, prog, boxes=True):
    """한 프레임에 타임라인·라벨·모델 입력 영역을 얹는다."""
    W, H = img.size
    bar_h, foot_h = 30, 74
    canvas = Image.new("RGB", (W, bar_h + H + foot_h), (16, 16, 16))
    canvas.paste(img, (0, bar_h))
    d = ImageDraw.Draw(canvas, "RGBA")

    # ---- 모델 입력 영역
    if boxes:
        sx = W / row["src_w"]
        # 어느 쪽이 기본인지는 config 에서 읽는다 (문서가 코드와 어긋나지 않게)
        for mode, (oh, ow), col, lab in (
            ("short_side", (256, DEFAULT_W), GREEN,
             f"short_side -> 256x{DEFAULT_W}   <- MODEL INPUT"),
            ("center_crop", (256, 256), GRAY,
             "center_crop 256x256 (official impl., 1.14x more zoom)"),
        ):
            x0, y0, x1, y1 = [v * sx for v in crop_box(row["src_w"], row["src_h"], oh, ow, mode)]
            y0, y1 = max(0, y0) + bar_h, min(H, y1) + bar_h
            d.rectangle([x0, y0, x1, y1], outline=col, width=3)
            d.text((x0 + 6, y0 + 4), lab, font=F(12, True), fill=col)

    # ---- 타임라인
    n_ctx, n_gap, n_act = row["n_ctx"], GAP_FRAMES, row["n_act"]
    tot = n_ctx + n_gap + n_act
    x = 0
    for n, col, lab in ((n_ctx, BLUE, "CONTEXT 4.0s"), (n_gap, GRAY, f"GAP {TAU_A}s"), (n_act, ORANGE, "ACTION")):
        w = W * n / tot
        d.rectangle([x, 0, x + w, bar_h - 8], fill=col)
        if w > 70:
            d.text((x + w / 2, (bar_h - 8) / 2), lab, font=F(12, True), fill=(255, 255, 255), anchor="mm")
        x += w
    cx = W * prog
    d.polygon([(cx - 6, bar_h - 8), (cx + 6, bar_h - 8), (cx, bar_h - 1)], fill=(255, 255, 255))

    # ---- 발/헤더
    y = bar_h + H + 6
    d.text((8, y), f"{row['narration_id']}   {row['video_id']} @ {row['fps']:.2f} fps   "
                   f"action segment {row['sf']}-{row['ef']}  ({row['dur']:.2f}s)",
           font=F(13), fill=(170, 170, 170))
    if phase == "context":
        d.text((8, y + 20), f"CONTEXT  t = {t_rel:+.2f}s  (model input: 32 frames @ 8 fps)",
               font=F(17, True), fill=BLUE)
        d.text((8, y + 44), "-> predict the NEXT action from this alone", font=F(14), fill=(210, 210, 210))
    elif phase == "gap":
        d.text((8, y + 20), f"GAP  t = {t_rel:+.2f}s   anticipation time tau_a = {TAU_A}s",
               font=F(17, True), fill=GRAY)
        d.text((8, y + 44), "never seen by the model", font=F(14), fill=(150, 150, 150))
    else:
        d.text((8, y + 20), f"ACTION  t = {t_rel:+.2f}s    ANSWER", font=F(17, True), fill=ORANGE)
        d.text((8, y + 44), f"verb '{row['verb']}' ({row['verb_class']})   "
                            f"noun '{row['noun']}' ({row['noun_class']})   "
                            f"action = ({row['verb_class']}, {row['noun_class']})",
               font=F(15, True), fill=(255, 235, 220))
    return canvas


def make_gif(vr, row, out, boxes=True):
    fps = row["fps"]
    fstp = int(fps / FPS)
    af = row["af"]
    idx_ctx = np.arange(af - FPC * fstp, af, fstp)
    idx_gap = np.arange(af, af + GAP_FRAMES * fstp, fstp)
    n_act = int(min(row["dur"], ACTION_MAX_S) * FPS)
    idx_act = np.arange(row["sf"], row["sf"] + n_act * fstp, fstp)
    idx = np.clip(np.concatenate([idx_ctx, idx_gap, idx_act]), 0, len(vr) - 1).astype(np.int64)
    phases = ["context"] * len(idx_ctx) + ["gap"] * len(idx_gap) + ["action"] * len(idx_act)
    row = {**row, "n_ctx": len(idx_ctx), "n_act": len(idx_act)}

    buf = vr.get_batch(idx).asnumpy()
    row["src_h"], row["src_w"] = buf.shape[1], buf.shape[2]
    dw = DISP_W
    dh = int(round(row["src_h"] * dw / row["src_w"] / 2) * 2)

    frames = []
    for i, (f, ph) in enumerate(zip(buf, phases)):
        img = Image.fromarray(f).resize((dw, dh), Image.BILINEAR)
        t_rel = (idx[i] - row["sf"]) / fps          # action 시작 기준 초
        frames.append(draw_frame(img, ph, t_rel, row, (i + 0.5) / len(idx), boxes))
    hold = [frames[-1]] * 6                         # 마지막 정답 프레임을 조금 붙잡는다
    frames[0].save(out, save_all=True, append_images=frames[1:] + hold,
                   duration=int(1000 / FPS), loop=0, optimize=True)
    return out, len(frames) + len(hold)


def candidates():
    tr = pd.read_csv(ANNOT / "EPIC_100_train.csv")
    va = pd.read_csv(ANNOT / "EPIC_100_validation.csv")
    info = pd.read_csv(ANNOT / "EPIC_100_video_info.csv").set_index("video_id")
    tact = set(zip(tr.verb_class, tr.noun_class))
    va = va[[(a, b) in tact for a, b in zip(va.verb_class, va.noun_class)]].copy()
    va["fps"] = va.video_id.map(info.fps)
    va["dur"] = (va.stop_frame - va.start_frame) / va.fps
    va["af"] = (va.start_frame - (TAU_A * va.fps).round()).astype(int)
    va["lead"] = va.af - FPC * (va.fps / FPS).astype(int)       # 문맥 시작
    # 문맥이 영상 안에 있고, action 이 너무 짧지도 길지도 않은 것
    ok = (va.lead > 0) & va.dur.between(1.0, 4.0)
    return va[ok].rename(columns={"start_frame": "sf", "stop_frame": "ef"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="만들 GIF 개수")
    ap.add_argument("--video", nargs="*", help="특정 video_id 만")
    ap.add_argument("--per-video", type=int, default=3, help="비디오 하나에서 몇 개 (열기 44s 를 아끼려고)")
    ap.add_argument("--no-boxes", action="store_true", help="모델 입력 영역 사각형 생략")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default=str(OUTDIR))
    a = ap.parse_args()

    c = candidates()
    if a.video:
        c = c[c.video_id.isin(a.video)]
    if a.list:
        print(c.groupby("video_id").size().sort_values(ascending=False).head(15))
        print(c[["narration_id", "video_id", "verb", "noun", "dur"]].head(20).to_string(index=False))
        return

    # verb 가 겹치지 않게 고르고, 비디오당 --per-video 개씩 묶는다
    rng = np.random.default_rng(0)
    c = c.sample(frac=1.0, random_state=0)
    picked, seen_verb, per = [], set(), {}
    for _, r in c.iterrows():
        if r.verb in seen_verb or per.get(r.video_id, 0) >= a.per_video:
            continue
        picked.append(r); seen_verb.add(r.verb); per[r.video_id] = per.get(r.video_id, 0) + 1
        if len(picked) >= a.n:
            break
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    from decord import VideoReader, cpu
    order = sorted(picked, key=lambda r: r.video_id)
    vr, cur = None, None
    for r in order:
        path = VIDEO_ROOT / r.video_id.split("_")[0] / "videos" / f"{r.video_id}.MP4"
        if cur != r.video_id:
            print(f"opening {path} ...", flush=True)
            vr, cur = VideoReader(str(path), num_threads=4, ctx=cpu(0)), r.video_id
        out = outdir / f"{r.narration_id}_{r.verb}_{r.noun}.gif".replace(" ", "-").replace(":", "")
        p, n = make_gif(vr, dict(r), out, boxes=not a.no_boxes)
        print(f"  {p.name}   {n} frames   verb={r.verb} noun={r.noun} dur={r.dur:.2f}s "
              f"({p.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
