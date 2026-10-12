#!/usr/bin/env python3
"""VOBench_v1 토큰별 surprise 차 Δ 를 불가능 영상 위에 — 왼쪽 V-JEPA 2 릴리즈 · 오른쪽 Ariel AR ep40 (2026-10-11).

입력: 두 채점 실행의 `token_surprise.npz` (z_research/scripts/analysis/token_surprise_maps.py — 토큰 평균 = per_video_surprise 검증 후 저장).
  Δ[t, i, j] = e_imp − e_pos   (matched pair: 문맥이 같아 p 가 같다; mean(Δ) = 채점 차)
  파랑 = Δ > 0 (그 토큰이 불가능 쪽을 더 놀랍게 본다 = 정답 방향) · 주황 = Δ < 0. 불투명도 = 0.9 · (|Δ| / vmax)^1.5.
  vmax = **하나** — 두 모델 · 두 objset · 미래 튜블릿 t8–t15 전부 같은 눈금 (late 클립 미래 토큰 |Δ| 를 모두 합친 99 분위; 사용자 지시 2026-10-11).
  ⚠️ AR 은 타깃이 블록별 LN(target_encoder) 이라 Δ 의 크기 자체는 다른 공간의 값이다 — 같은 눈금은 나란히 보기 위한 것.
  흰 원 = 가능 영상에서 물체가 있어야 할 자리 (label.json, 그 프레임의 중심 · 반경). 문맥 16 샘플은 덮지 않는다 (토큰이 미래에만).
행 = late · k = 2 의 전형 쌍 (그 조건 릴리즈 · AR 쌍 Δ 평균이 각자 중앙값에 가장 가까운 block): permanence 등속 (사라짐) · permanence 정지 · shape 등속.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/gif_vobench_token_surprise.py [_static]   → z_research/VOBench_v1/ariel_ar_ep40/surprise/figures/token_surprise_vs_vjepa2/<objset>.gif · _strip.png
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "z_research/VOBench_v1/exp_results"
RUNS = {"V-JEPA 2 (release)": EXP / "surprise_c16t32__vobench_v1_vith",
        "Ariel AR ep40": EXP / "surprise_c16t32__vobench_v1_vith_ariel_ar_ep40"}
INDEX = REPO / "data_csv/vobench_v1/index.csv"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
OUT = REPO / "z_research/VOBench_v1/ariel_ar_ep40/surprise/figures/token_surprise_vs_vjepa2"
RAW = [3 * i for i in range(32)]
BLUE, ORANGE = np.array([42, 120, 214]), np.array([235, 104, 52])
GAMMA, PX = 1.5, 256                                     # 512 원본을 256 으로 (모델 입력과 같은 격자: 16 px = 토큰 한 칸)
FONT = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 12)
FONT_B = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 13)
SETS = {   # 이름 → 행 (위반, 운동, 라벨). 출력 <objset><접미사>.gif
    "": [("permanence", "flat_vel", "permanence\nconst. vel\n(object vanishes)"),
         ("permanence", "flat_static", "permanence\nstatic\n(object vanishes)"),
         ("shape", "flat_vel", "shape\nconst. vel\n(shape changes)")],
    "_static": [("permanence", "flat_static", "permanence\nstatic\n(object vanishes)"),       # 2026-10-11 사용자: 전부 정지 물체로
                ("shape", "flat_static", "shape\nstatic\n(shape changes)"),
                ("color", "flat_static", "colour\nstatic\n(colour changes)")],
}


def frames(row):
    return [np.asarray(Image.open(FR / row["file_name"] / f"{f:06d}.png").convert("RGB").resize((PX, PX), Image.BILINEAR)) for f in RAW]


def overlay(img, d, vmax):
    a = np.clip(np.abs(d) / vmax, 0, 1) ** GAMMA * 0.9
    col = np.where((d > 0)[..., None], BLUE, ORANGE)
    up = lambda m: np.kron(m, np.ones((PX // 16, PX // 16) + ((1,) if m.ndim == 3 else ())))
    a, col = up(a)[..., None], up(col)
    return (img * (1 - a) + col * a).astype(np.uint8)


def legend_bar(vmax, w, h=30):
    im = Image.new("RGB", (w, h), "white"); dr = ImageDraw.Draw(im)
    xs = np.linspace(-vmax, vmax, w - 40)
    for i, v in enumerate(xs):
        a = min(abs(v) / vmax, 1) ** GAMMA * 0.9; c = BLUE if v > 0 else ORANGE
        dr.line([(20 + i, 2), (20 + i, 13)], fill=tuple(int(255 * (1 - a) + c[j] * a) for j in range(3)))
    dr.text((14, 15), f"-{vmax:.3f}", fill=(60, 60, 60), font=FONT); dr.text((w // 2 - 4, 15), "0", fill=(60, 60, 60), font=FONT)
    dr.text((w - 60, 15), f"+{vmax:.3f}", fill=(60, 60, 60), font=FONT)
    return im


def main(suffix=""):
    CONDS = SETS[suffix]
    E = {}
    for name, run in RUNS.items():
        z = np.load(run / "token_surprise.npz"); E[name] = dict(zip(z["video_id"], z["e"]))
    rows = list(csv.DictReader(INDEX.open()))
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block_id"], {})[r["variant"]] = r
    lab_cache = {}
    def obj_pos(row):                                    # 가능 arm (pos_a) 의 물체 중심 · 반경, 샘플마다 (256 격자)
        bdir = FR / Path(row["file_name"]).parent
        if bdir not in lab_cache:
            lab_cache[bdir] = json.load(open(bdir / "label.json"))["labels"]["A"]
        return [(e["u"] / 2, e["v"] / 2, e["r_px"] / 2, e["hidden"]) for e in (lab_cache[bdir][f] for f in RAW)]
    names = list(RUNS)
    D = lambda name, v: E[name][v["imp_ab"]["video_id"]] - E[name][v["pos_a"]["video_id"]]           # A 쌍 (있는데→있는)
    late_all = [v for v in blocks.values() if v["pos_a"]["occlusion"] == "late"]
    VMAX = float(np.percentile(np.abs(np.stack([D(n, v) for n in names for v in late_all])), 99))   # 공통 눈금 하나
    vmax = {n: VMAX for n in names}
    print(f"  공통 vmax {VMAX:.4f}")
    for o in ("primitive", "realistic"):
        late = [v for v in blocks.values() if v["pos_a"]["objset"] == o and v["pos_a"]["occlusion"] == "late"]
        spec = []
        for viol, mot, lab in CONDS:
            cand = [v for v in late if v["pos_a"]["violation"] == viol and v["pos_a"]["motion"] == mot and v["pos_a"]["k"] == "2"]
            sc = {n: np.array([D(n, v).mean() for v in cand]) for n in names}
            rk = sum(np.abs(sc[n] - np.median(sc[n])) / (np.std(sc[n]) + 1e-12) for n in names)
            v = cand[int(np.argmin(rk))]
            spec.append((lab, v, {n: D(n, v) for n in names}))
            print(f"  {o} {viol}/{mot}: block {v['pos_a']['source_block']} · " + " · ".join(f"{n} pair Δ {D(n, v).mean():+.4f}" for n in names))
        G, HEAD, LEFT = 8, 46, 190
        W = LEFT + 2 * PX + G; H = 26 + HEAD + len(spec) * (PX + 20) + 96
        fimp = [frames(v["imp_ab"]) for _, v, _ in spec]; pos = [obj_pos(v["pos_a"]) for _, v, _ in spec]
        leg = legend_bar(VMAX, PX)
        out = []
        for s in range(32):
            im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
            dr.text((10, 6), f"VOBench_v1 · {o} · late occlusion (k = 2) · impossible clip with token Δ", fill=(0, 0, 0), font=FONT_B)
            fut = s >= 16; t = (s - 16) // 2
            stage = f"future sample {s - 15}/16 (tubelet t{t + 8})" if fut else f"context sample {s + 1}/16 (no tokens scored)"
            dr.text((W - 300, 26), stage, fill=(0, 0, 0) if fut else (120, 120, 120), font=FONT)
            for ci, n in enumerate(names):
                dr.text((LEFT + ci * (PX + G) + 4, 26 + 18), n, fill=(0, 0, 0), font=FONT_B)
            for ri, (lab, v, Dm) in enumerate(spec):
                y = 26 + HEAD + ri * (PX + 20)
                dr.multiline_text((8, y + 6), lab, fill=(0, 0, 0), font=FONT_B, spacing=3)
                yy = y + 62
                for n in names:
                    tot = Dm[n].mean()
                    dr.multiline_text((8, yy), f"{n.split(' (')[0]}\n pair Δ {tot:+.4f}\n ({'correct' if tot > 0 else 'wrong'})",
                                      fill=(42, 120, 214) if tot > 0 else (235, 104, 52), font=FONT, spacing=2); yy += 52
                for ci, n in enumerate(names):
                    base = fimp[ri][s]
                    arr = overlay(base, Dm[n][t], vmax[n]) if fut else base
                    pim = Image.fromarray(arr); pd = ImageDraw.Draw(pim)
                    u, vv, r, hid = pos[ri][s]
                    if fut:
                        pd.ellipse([u - r - 2, vv - r - 2, u + r + 2, vv + r + 2], outline=(255, 255, 255), width=2)
                    im.paste(pim, (LEFT + ci * (PX + G), y))
            im.paste(leg, (LEFT + PX // 2 + G // 2, H - 36))
            dr.text((8, H - 92), "blue: impossible more surprising · orange: possible more surprising\nwhite ring: where the object is in the possible clip\none colour scale for both models and all tubelets",
                    fill=(60, 60, 60), font=FONT, spacing=2)
            out.append(im)
        OUT.mkdir(parents=True, exist_ok=True)
        bar = np.concatenate([np.asarray(legend_bar(max(vmax.values()), W))[:14]] * 12, 0)
        src = np.concatenate([np.asarray(out[k]) for k in (0, 16, 20, 24, 28, 31)] + [bar], 0)
        pal = Image.fromarray(src).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in out]
        q[0].save(OUT / f"{o}{suffix}.gif", save_all=True, append_images=q[1:], duration=[300] * 31 + [1500], loop=0, optimize=False)
        # 띠: 전체 시퀀스 16 튜블릿 (튜블릿 첫 샘플) — 문맥 t0–t7 은 원본만, 미래 t8–t15 는 Δ 겹침. 행 = 조건 × 모델, 눈금 공통
        cw, SEP = PX // 2 + 4, 14
        x_of = lambda tt: LEFT + tt * cw + (SEP if tt >= 8 else 0)
        Hs = 40 + len(spec) * len(names) * (PX // 2 + 6) + 44
        strip = Image.new("RGB", (x_of(15) + cw, Hs), "white"); ds = ImageDraw.Draw(strip)
        ds.text((LEFT + 3 * cw, 4), "context (t0–t7)", fill=(120, 120, 120), font=FONT_B)
        ds.text((x_of(8) + 3 * cw, 4), "future (t8–t15): token Δ", fill=(0, 0, 0), font=FONT_B)
        for tt in range(16):
            ds.text((x_of(tt) + 52, 22), f"t{tt}", fill=(0, 0, 0) if tt >= 8 else (120, 120, 120), font=FONT)
        k = 0
        for ri, (lab, v, Dm) in enumerate(spec):
            for n in names:
                y = 40 + k * (PX // 2 + 6); k += 1
                ds.multiline_text((6, y + 8), lab + "\n" + n, fill=(0, 0, 0), font=FONT, spacing=2)
                for tt in range(16):
                    s = 2 * tt
                    arr = overlay(fimp[ri][s], Dm[n][tt - 8], vmax[n]) if tt >= 8 else fimp[ri][s]
                    strip.paste(Image.fromarray(arr).resize((PX // 2, PX // 2)), (x_of(tt), y))
        ds.line([(x_of(8) - SEP // 2 - 2, 20), (x_of(8) - SEP // 2 - 2, Hs - 46)], fill=(0, 0, 0), width=2)
        strip.paste(legend_bar(VMAX, PX), (x_of(8) + 2 * cw, Hs - 38))
        ds.text((LEFT, Hs - 36), "blue: impossible more surprising · orange: possible more surprising · one scale", fill=(60, 60, 60), font=FONT)
        strip.save(OUT / f"{o}{suffix}_strip.png")
        print(f"  [saved] {OUT / (o + suffix)}.gif · _strip.png · vmax {VMAX:.4f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")      # "" = 기본 3 행 · "_static" = 정지 물체 3 위반
