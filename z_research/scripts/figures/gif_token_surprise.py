#!/usr/bin/env python3
"""토큰별 surprise 차 Δ = |p − h_imp| − |p − h_pos| 를 원본 프레임 위에 겹친 GIF (2026-10-07, v11_realistic · ledge).

입력: `<run>/token_surprise.npz` (z_research/scripts/analysis/token_surprise_maps.py — 토큰 평균이 per_block.json 과 같음을 검증해 저장)
한 줄 = matched pair 하나 (문맥이 같아 p 가 같다). 칸 = 가능 clip | 불가능 clip | Δ (두 프레임 50/50 겹친 위에).
  파랑 = Δ > 0 (그 토큰이 불가능 쪽을 더 놀랍게 본다 = 정답 방향) · 주황 = Δ < 0 (틀린 방향).
  불투명도 = 0.9 · (|Δ| / vmax)^GAMMA (GAMMA 1.5 — 배경 토큰의 작은 Δ 는 옅게, 큰 값은 진하게. 표시 규칙일 뿐 값은 그대로).
  vmax = 두 세트 미래 토큰 전체 |Δ| 의 99 분위 (GIF 끼리 같은 눈금). 문맥 16 샘플은 덮지 않는다 (토큰이 미래에만 있다).
  미래 샘플 2 장 = 튜블릿 하나 → 같은 Δ 지도.
예시 고르기: 조건마다 쌍 Δ 평균 (= 채점 차) 이 그 조건 중앙값에 가장 가까운 block (전형 사례). 가림은 k=2.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/gif_token_surprise.py      → z_research/v11_realistic/figures/token_surprise/*.gif · *_strip.png
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "z_research/v11_realistic/exp_results"
OUT = REPO / "z_research/v11_realistic/figures/token_surprise"
SETS = {
    "realistic": (EXP / "surprise_c16t32__v11_realistic_vith", REPO / "data_csv/intphysgen_v11_realistic/index.csv",
                  Path("/local_datasets/world/world_analysis/IntPhysGen_v11_realistic")),
    "ledge": (EXP / "surprise_c16t32__v11_realistic_ledge_vith", REPO / "data_csv/intphysgen_v11_realistic_ledge/index.csv",
              Path("/local_datasets/world/world_analysis/IntPhysGen_v11_realistic_ledge")),
}
RAW = [3 * i for i in range(32)]                 # 32 샘플 = raw 0, 3, …, 93
BLUE, ORANGE = np.array([42, 120, 214]), np.array([235, 104, 52])
GAMMA = 1.5
PX = 288                                         # 원본 해상도 (모델은 256 으로 전체 리사이즈 → 16×16 토큰이 화면 전체를 덮는다)
FONT = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
FONT_B = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 14)


def load(name):
    run, idx, fr = SETS[name]
    z = np.load(run / "token_surprise.npz")
    E = dict(zip(z["video_id"], z["e"]))
    rows = list(csv.DictReader(idx.open()))
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block_id"], {})[r["variant"]] = r
    return E, blocks, fr


def pair_delta(E, pos, imp):
    return E[imp["video_id"]] - E[pos["video_id"]]          # (8, 16, 16)


def obj_mask(row, md):
    """미래 튜블릿마다 물체 중심 (metadata object_px_*_by_sample, 288 px / 18 px 칸) 이 든 칸 + 이웃 3×3. 빈 장면 clip 은 전부 False."""
    M = np.zeros((8, 16, 16), bool)
    m = md.get(row["video_id"])
    if m is None or not m.get("object_px_x_by_sample"):
        return M
    xs = np.array(m["object_px_x_by_sample"].split(), float); ys = np.array(m["object_px_y_by_sample"].split(), float)
    if len(xs) < 32:
        return M
    for t in range(8):
        for q in (16 + 2 * t, 17 + 2 * t):
            if np.isnan(xs[q]) or not (0 <= xs[q] < PX and 0 <= ys[q] < PX):
                continue
            cx, cy = int(xs[q] // 18), int(ys[q] // 18)
            M[t, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
    return M


def frames(fr, row):
    return [np.asarray(Image.open(fr / row["file_name"] / f"{f:06d}.png").convert("RGB")) for f in RAW]


def overlay(img, d, vmax):
    """img (PX, PX, 3) 위에 Δ (16, 16) — 토큰 칸 그대로 (최근접 확대)."""
    a = np.clip(np.abs(d) / vmax, 0, 1) ** GAMMA * 0.9
    col = np.where((d > 0)[..., None], BLUE, ORANGE)
    up = lambda m: np.kron(m, np.ones((PX // 16, PX // 16) + ((1,) if m.ndim == 3 else ())))
    a, col = up(a)[..., None], up(col)
    return (img * (1 - a) + col * a).astype(np.uint8)


def legend_bar(vmax, w=PX, h=34):
    im = Image.new("RGB", (w, h), "white"); dr = ImageDraw.Draw(im)
    xs = np.linspace(-vmax, vmax, w - 40)
    for i, v in enumerate(xs):
        a = min(abs(v) / vmax, 1) ** GAMMA * 0.9; c = BLUE if v > 0 else ORANGE
        rgb = tuple(int(255 * (1 - a) + c[j] * a) for j in range(3))
        dr.line([(20 + i, 2), (20 + i, 14)], fill=rgb)
    dr.text((20, 17), f"-{vmax:.2f}", fill=(60, 60, 60), font=FONT)
    dr.text((w // 2 - 4, 17), "0", fill=(60, 60, 60), font=FONT)
    dr.text((w - 64, 17), f"+{vmax:.2f}", fill=(60, 60, 60), font=FONT)
    return im


def render(name, rows_spec, vmax, title):
    """rows_spec = [(라벨, pos_row, imp_row, fr, Δ (8,16,16), 물체 마스크 (8,16,16))] → GIF + 튜블릿 띠 PNG.
    색: 32 프레임 전부 **팔레트 하나** (프레임마다 팔레트를 새로 만들면 같은 Δ 가 프레임마다 다른 색이 된다 — 2026-10-07 수정)."""
    G, HEAD, LEFT = 6, 40, 150
    W = LEFT + 3 * PX + 2 * G
    H = 28 + HEAD + len(rows_spec) * (PX + 26) + 40
    fpos = [frames(fr, p) for _, p, _, fr, _, _ in rows_spec]
    fimp = [frames(fr, q) for _, _, q, fr, _, _ in rows_spec]
    leg = legend_bar(vmax)
    out = []
    for s in range(32):
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        dr.text((10, 6), title, fill=(0, 0, 0), font=FONT_B)
        fut = s >= 16; t = (s - 16) // 2
        stage = f"future sample {s - 15}/16  (tubelet t{t})" if fut else f"context sample {s + 1}/16  (no tokens to score)"
        dr.text((W - 330, 6), stage, fill=(0, 0, 0) if fut else (120, 120, 120), font=FONT)
        for ci, lab in enumerate(["possible", "impossible", "Δ = |p−h_imp| − |p−h_pos|"]):
            dr.text((LEFT + ci * (PX + G) + 4, 28 + 14), lab, fill=(0, 0, 0), font=FONT)
        for ri, (rlab, _, _, _, D, M) in enumerate(rows_spec):
            y = 28 + HEAD + ri * (PX + 26)
            a, b = fpos[ri][s], fimp[ri][s]
            blend = ((a.astype(np.float32) + b) / 2).astype(np.uint8)
            third = overlay(blend, D[t], vmax) if fut else blend
            for ci, arr in enumerate([a, b, third]):
                im.paste(Image.fromarray(arr), (LEFT + ci * (PX + G), y))
            tot = D.mean()
            nl = rlab.count("\n") + 1
            dr.multiline_text((8, y + 8), rlab, fill=(0, 0, 0), font=FONT_B, spacing=3)
            y2 = y + 8 + nl * 18 + 10
            dr.multiline_text((8, y2), f"pair mean Δ\n{tot:+.4f}\n({'correct' if tot > 0 else 'wrong'})",
                              fill=(42, 120, 214) if tot > 0 else (235, 104, 52), font=FONT, spacing=3)
            if fut:
                mo = D[t][M[t]].mean() if M[t].any() else float("nan")
                dr.multiline_text((8, y2 + 62), f"tubelet t{t} mean Δ\n all     {D[t].mean():+.4f}\n object  {mo:+.4f}\n backgr. {D[t][~M[t]].mean():+.4f}",
                                  fill=(60, 60, 60), font=FONT, spacing=3)
        im.paste(leg, (LEFT + 2 * (PX + G), H - 38))
        dr.text((LEFT + PX + G - 10, H - 34), "blue: impossible more surprising\norange: possible more surprising", fill=(60, 60, 60), font=FONT)
        out.append(im)
    OUT.mkdir(parents=True, exist_ok=True)
    # 공통 팔레트: 여러 프레임 + 색 눈금 전 구간 (면적을 크게 줘 눈금 색이 팔레트에 확실히 들어가게)
    bar = np.concatenate([np.asarray(legend_bar(vmax, w=W, h=34))[:16]] * 12, 0)
    src = np.concatenate([np.asarray(out[k]) for k in (0, 16, 20, 24, 28, 31)] + [bar], 0)
    pal = Image.fromarray(src).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in out]
    q[0].save(OUT / f"{name}.gif", save_all=True, append_images=q[1:], duration=[300] * 31 + [1500], loop=0, optimize=False)
    # 띠: 미래 8 튜블릿의 Δ (튜블릿 첫 샘플 프레임 위)
    strip = Image.new("RGB", (LEFT + 8 * (PX // 2 + 4), 24 + len(rows_spec) * (PX // 2 + 6)), "white"); ds = ImageDraw.Draw(strip)
    for t in range(8):
        ds.text((LEFT + t * (PX // 2 + 4) + 50, 4), f"t{t}", fill=(0, 0, 0), font=FONT)
    for ri, (rlab, _, _, _, D, _M) in enumerate(rows_spec):
        y = 24 + ri * (PX // 2 + 6)
        ds.multiline_text((8, y + 8), rlab, fill=(0, 0, 0), font=FONT_B, spacing=3)
        for t in range(8):
            s = 16 + 2 * t
            blend = ((fpos[ri][s].astype(np.float32) + fimp[ri][s]) / 2).astype(np.uint8)
            strip.paste(Image.fromarray(overlay(blend, D[t], vmax)).resize((PX // 2, PX // 2)), (LEFT + t * (PX // 2 + 4), y))
    strip.save(OUT / f"{name}_strip.png")
    print(f"  [saved] {OUT / name}.gif · _strip.png")


def typical(cands, key):
    """쌍 Δ 평균이 그 집합 중앙값에 가장 가까운 것."""
    v = np.array([key(c) for c in cands]); return cands[int(np.argmin(np.abs(v - np.median(v))))]


def decompose(data):
    """쌍 Δ 평균 (= 채점 차) 을 물체 토큰 · 배경 토큰 기여로 나눈다 → token_surprise/object_vs_background.{json,md}.
    물체 토큰 = 미래 튜블릿마다 두 clip 중 어느 쪽이든 물체 중심 (metadata object_px_*_by_sample, 288 px / 18 px 칸) 이 든 칸 + 이웃 3×3."""
    mask = obj_mask
    res = {}
    for name, pairs in [("ledge", [("ledge", "pos_fall", "imp_float")]), ("realistic", [("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")])]:
        E, B, fr = data[name]
        md = {r["name"]: r for r in csv.DictReader((fr / "metadata.csv").open())}
        for d, pv, iv in pairs:
            for b in B.values():
                D = pair_delta(E, b[pv], b[iv]); M = mask(b[pv], md) | mask(b[iv], md)
                key = f"ledge" if name == "ledge" else f"{b[pv]['condition']} {d}"
                res.setdefault(key, []).append((D[M].sum() / D.size, D[~M].sum() / D.size, M.mean(),
                                                np.abs(D[M]).mean() if M.any() else np.nan, np.abs(D[~M]).mean()))
    md_lines = ["| 조건 · 방향 | n | 쌍 Δ 평균 | 물체 토큰 기여 | 배경 토큰 기여 | 물체 토큰 비율 | 토큰당 |Δ| 물체 / 배경 |", "|---|---:|---:|---:|---:|---:|---:|"]
    out = {}
    for k in sorted(res):
        v = np.array(res[k]); o, g = v[:, 0].mean(), v[:, 1].mean()
        out[k] = dict(n=len(v), mean_delta=o + g, object=o, background=g, object_token_frac=float(v[:, 2].mean()),
                      abs_object=float(np.nanmean(v[:, 3])), abs_background=float(v[:, 4].mean()))
        md_lines.append(f"| {k} | {len(v)} | {o + g:+.4f} | {o:+.4f} | {g:+.4f} | {100 * v[:, 2].mean():.1f} % | "
                        f"{np.nanmean(v[:, 3]):.3f} / {v[:, 4].mean():.3f} |")
    (OUT / "object_vs_background.json").write_text(json.dumps(out, indent=1))
    (OUT / "object_vs_background.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


def main():
    data = {n: load(n) for n in SETS}
    EL, BL = data["ledge"][0], data["ledge"][1]
    ER, BR = data["realistic"][0], data["realistic"][1]
    allabs = np.concatenate([np.abs(pair_delta(EL, b["pos_fall"], b["imp_float"])).ravel() for b in BL.values()]
                            + [np.abs(pair_delta(ER, b[p], b[q])).ravel() for b in BR.values()
                               for p, q in (("pos_a", "imp_ab"), ("pos_b", "imp_ba"))])
    vmax = float(np.percentile(allabs, 99))
    print(f"vmax (|Δ| 99 분위, 두 세트) = {vmax:.4f}")
    log = {"vmax": vmax, "examples": {}}
    # ledge: 속도마다 z300 전형 사례
    E, B, fr = data["ledge"]
    md = {r["name"]: r for r in csv.DictReader((fr / "metadata.csv").open())}
    for v in (140, 180, 220):
        c = [b for b in B.values() if b["pos_fall"]["condition"] == f"grav_v{v}_z300"]
        b = typical(c, lambda b: pair_delta(E, b["pos_fall"], b["imp_float"]).mean())
        D = pair_delta(E, b["pos_fall"], b["imp_float"])
        name = f"ledge_v{v}_z300"
        render(name, [("falls vs\nfloats on", b["pos_fall"], b["imp_float"], fr, D, obj_mask(b["pos_fall"], md) | obj_mask(b["imp_float"], md))], vmax,
               f"Ledge, {v} cm/s, depth 300 — possible: falls / impossible: floats on")
        log["examples"][name] = dict(block=b["pos_fall"]["source_block"], pair_mean_delta=float(D.mean()))
    # vanish (실물): 조건 6 개, 한 GIF 에 A · B 두 줄 (같은 block)
    E, B, fr = data["realistic"]
    md = {r["name"]: r for r in csv.DictReader((fr / "metadata.csv").open())}
    for cond, lab in [("static_visible", "Static, no occluder"), ("static_occlusion", "Static, occluded at context end (k=2)"),
                      ("moving_visible_flat", "Moving flat, no occluder"), ("moving_occlusion_flat", "Moving flat, occluded at context end (k=2)"),
                      ("moving_visible", "Moving ramp, no occluder"), ("moving_occlusion", "Moving ramp, occluded at context end (k=2)")]:
        c = [b for b in B.values() if b["pos_a"]["condition"] == cond and ("occlusion" not in cond or b["pos_a"]["sym_k"] == "2")]
        b = typical(c, lambda b: pair_delta(E, b["pos_a"], b["imp_ab"]).mean())
        DA, DB = pair_delta(E, b["pos_a"], b["imp_ab"]), pair_delta(E, b["pos_b"], b["imp_ba"])
        MA = obj_mask(b["pos_a"], md) | obj_mask(b["imp_ab"], md); MB = obj_mask(b["pos_b"], md) | obj_mask(b["imp_ba"], md)
        render(f"vanish_{cond}", [("A: object\nstays vs\nvanishes", b["pos_a"], b["imp_ab"], fr, DA, MA),
                                  ("B: empty\nstays vs\nobject appears", b["pos_b"], b["imp_ba"], fr, DB, MB)], vmax, f"Object permanence — {lab}")
        log["examples"][f"vanish_{cond}"] = dict(block=b["pos_a"]["source_block"], A=float(DA.mean()), B=float(DB.mean()))
    (OUT / "examples.json").write_text(json.dumps(log, indent=1))
    decompose(data)


if __name__ == "__main__":
    main()
