#!/usr/bin/env python3
"""v3 전부 학습 모델이 VOBench late 사라짐 쌍에서 무엇을 예측하나 — 픽셀 decoder 그림 + 패치 L1 차 지도 GIF (2026-11-02, 사용자 요청).

쌍: VOBench late (split v2 시험 절반) permanence 사라짐 쌍 36 개 (운동 6 × 물체 모양 2 × 3). 문맥 = 원본 0, 3, …, 45 (16 장), 미래 = 48 … 93.
모델: 원래 (릴리즈) · v3 전부 (v3 학습 장면 7,136 개, 공통 설정) · VOBench 학습 (참고). encoder 는 셋 다 릴리즈 고정.
GIF 1 (_decoder.gif): [실제 가능 미래 | 실제 불가능 미래 | 정답 latent 를 decoder 로 그린 것 (decoder 상한) | 세 모델의 예측을 픽셀 decoder (z_pixel_decoder vob_grav_i1_vitb_both) 로 그린 것]
GIF 2 (_l1.gif): [실제 가능 미래 | 세 모델의 Δ 지도] — 토큰마다 Δ = |p − h_imp| − |p − h_pos| (채널 평균, h = LN(target_encoder) 미래 토큰).
   파랑 = Δ > 0 (그 토큰에서 불가능 쪽을 더 놀랍게 본다 = 정답 방향) · 주황 = Δ < 0. 불투명도 = 0.9 · (|Δ| / vmax)^1.5, vmax = 36 쌍 · 세 모델 공통 (|Δ| 99 분위).
   바탕 = 불가능 영상. 흰 원 = 가능 영상에서 물체가 있는 자리 (물체 있음 − 물체 없음 영상 픽셀 차의 중심 · 반경).
   머리줄의 correct / wrong = 쌍 판정 (미래 전체 토큰 평균 Δ 의 부호, 표준 채점과 같은 식).
  실행 (GPU 1): P=… $P predictor_lab/v3_ablation/page/gif_v3full_vob.py  → predictor_lab/results/v3_ablation/viz_v3full_vob/
  처음 보는 물체 split (2026-11-02): VIZ_SET=objholdout … → viz_objholdout/ (모델 = 원래 · 처음 보는 물체 PFT · v3 전부, 파일 이름 앞에 범주)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_pixel_decoder"))
import te_v11_score as te  # noqa: E402
import transfer_gap_r4_endprior as r4  # noqa: E402
from transfer_gap_te import resolve  # noqa: E402
from model import PixelDecoder  # noqa: E402

R = REPO / "z_training/runs"
MODE = os.environ.get("VIZ_SET", "split2")          # split2 = VOBench split v2 시험 late · objholdout = 처음 보는 물체 split 시험 late (2026-11-02)
if MODE == "objholdout":
    PREDS = [("release", str(R / "release_vith/latest.pt")), ("unseen-object PFT", str(R / "vobench_release_postft_objholdout/latest.pt")),
             ("v3 all (Blender)", str(R / "v3_full_s0/latest.pt"))]
    SETS_ = [f"data.root={REPO}/data_csv/vobench_v1_objholdout", "data.index_csv=index_test.csv",
             f"data.frames_root={REPO}/z_pixel_decoder/runs/_frames/VOBench_v1_256full"]
elif MODE == "v4flat":                              # 2026-11-02: v4 평면 (같은 장면 × 속도 4) 이 VOBench 평면 late 에서 무엇을 그리나 — [원본 | v3 전부 | v4 평면]
    PREDS = [("v3 all (1 speed / scene)", str(R / "v3_full_s0/latest.pt")), ("v4 flat (same scene x 4 speeds)", str(R / "v4vc_fam_s0/latest.pt"))]
    SETS_ = None
else:
    PREDS = [("release", str(R / "release_vith/latest.pt")), ("v3 all (Blender)", str(R / "v3_full_s0/latest.pt")),
             ("VOBench-trained", str(R / "vobench_release_postft_g/latest.pt"))]
    SETS_ = None
DEC = REPO / "z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt"
FR = REPO / "z_pixel_decoder/runs/_frames/VOBench_v1_256full"
OUT = REPO / {"objholdout": "predictor_lab/results/v3_ablation/viz_objholdout", "v4flat": "predictor_lab/results/v4vc/viz_vob_flat"}.get(
    MODE, "predictor_lab/results/v3_ablation/viz_v3full_vob")
RAW = [3 * k for k in range(32)]
C, T, S = 16, 32, 256
BLUE, ORANGE = np.array([42, 120, 214]), np.array([235, 104, 52])
FONT = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 12)
MOTION = {"flat_static": "static", "flat_vel": "flat const. vel", "flat_acc": "flat accelerating", "flat_dec": "flat decelerating",
          "ramp_up": "ramp up", "ramp_down": "ramp down"}


def frames(fn):
    return np.stack([np.asarray(Image.open(FR / fn / f"{f:06d}.png").convert("RGB").resize((256, 256), Image.BILINEAR)) for f in RAW])


def obj_track(pa, pb):
    """가능 영상 (물체 있음) − 물체 없음 영상의 픽셀 차 → 샘플마다 (중심 x, y, 반경) 또는 None"""
    out = []
    for a, b in zip(pa.astype(np.float32), pb.astype(np.float32)):
        m = np.abs(a - b).mean(-1) > 18
        if m.sum() < 12:
            out.append(None); continue
        ys, xs = np.nonzero(m)
        out.append((xs.mean(), ys.mean(), max(6.0, np.sqrt(m.sum() / np.pi))))
    return out


def overlay(img, d, vmax):
    a = (np.clip(np.abs(d) / vmax, 0, 1) ** 1.5 * 0.9)
    col = np.where((d > 0)[..., None], BLUE, ORANGE)
    a = np.kron(a, np.ones((16, 16)))[..., None]
    col = np.kron(col, np.ones((16, 16, 1)))
    return (img * (1 - a) + col * a).astype(np.uint8)


def label(im, x, text, color=(255, 255, 255), w=250):
    d = ImageDraw.Draw(im)
    d.rectangle([x, 0, x + w, 15], fill=(0, 0, 0))
    d.text((x + 3, 1), text, fill=color, font=FONT)


def circle(im, x0, tr):
    if tr is None:
        return
    cx, cy, r = tr
    ImageDraw.Draw(im).ellipse([x0 + cx - r, cy - r, x0 + cx + r, cy + r], outline=(255, 255, 255), width=2)


def main():
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, _ = resolve("vobench_v1_split2_test", "vith", SETS_ or r4.VOB, None, 0)
    sc = te.Scorer(cfg, dev, PREDS)
    W = te.WMADataset(cfg)
    st = torch.load(DEC, map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
    dec.load_state_dict(st["decoder"])
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

    by = {}
    for i, r in enumerate(W.records):
        if r.raw.get("violation_type") == "permanence" and r.raw.get("occlusion", "late") == "late":
            by.setdefault(r.block_id, {})[r.raw.get("variant")] = i
    blocks = sorted([b for b, v in by.items() if {"pos_a", "imp_ab", "pos_b"} <= set(v)
                     and (MODE != "v4flat" or W.records[v["pos_a"]].raw["motion"].startswith("flat"))],
                    key=lambda b: (W.records[by[b]["pos_a"]].raw["motion"], W.records[by[b]["pos_a"]].raw["objset"], int(b)))
    print(f"사라짐 쌍 {len(blocks)} 개", flush=True)

    rec, store = [], []
    for b in blocks:
        v = by[b]; raw = W.records[v["pos_a"]].raw
        clips = torch.stack([W.clip(v["pos_a"]), W.clip(v["imp_ab"])])
        with torch.inference_mode():
            z, h, ci, ti, _ = sc.encode(clips)
            hh = h.float().reshape(2, C // 2 + 8, S, -1)[:, C // 2:]                     # (2, 8, 256, D): 가능 · 불가능 미래
            maps, decs, verdict, objm = [], [], [], {}
            yg = dec(hh[0]) * STD + MEAN                                                  # 정답 (가능 미래) latent → decoder = decoder 상한
            gt_dec = (yg.clamp(0, 1).permute(0, 2, 3, 4, 1).reshape(16, 256, 256, 3) * 255).round().byte().cpu().numpy()
            dho = (hh[0] - hh[1]).abs().mean(-1)                                          # (8, 256) 물체가 있는 토큰에서 크다
            omask = dho > 0.5 * dho.max(-1, keepdim=True).values                          # 튜블릿마다 물체 토큰
            for k, mod in enumerate(sc.mods):
                p = sc.predict(mod, z, ci, ti).float().reshape(2, 8, S, -1)[0]                # 두 영상의 문맥이 같다 → 예측도 같다 (가능 쪽 것)
                e_pos = (p - hh[0]).abs().mean(-1); e_imp = (p - hh[1]).abs().mean(-1)
                dlt = (e_imp - e_pos).view(8, 16, 16).cpu().numpy()
                maps.append(dlt); verdict.append(bool(e_imp.mean() > e_pos.mean()))
                # 물체 토큰에서: 예측이 "물체 있는 미래" 쪽에 더 가까운 토큰 비율 · 그 토큰들의 평균 Δ (튜블릿 앞 4 · 뒤 4)
                closer = ((e_pos < e_imp) & omask).float().sum(-1) / omask.float().sum(-1).clamp(min=1)
                objm[PREDS[k][0]] = {"closer_to_object_early": float(closer[:4].mean()), "closer_to_object_late": float(closer[4:].mean()),
                                     "delta_obj": float(((e_imp - e_pos) * omask).sum() / omask.sum().clamp(min=1))}
                y = dec(p.view(8, S, -1)) * STD + MEAN                                      # (8, 3, 2, 256, 256)
                decs.append((y.clamp(0, 1).permute(0, 2, 3, 4, 1).reshape(16, 256, 256, 3) * 255).round().byte().cpu().numpy())
        fa, fi, fb = frames(raw["file_name"]), frames(W.records[v["imp_ab"]].raw["file_name"]), frames(W.records[v["pos_b"]].raw["file_name"])
        store.append((b, raw, fa, fi, obj_track(fa, fb), maps, decs, verdict, gt_dec))
        rec.append({"block": b, "holdout": raw.get("holdout"), "motion": raw["motion"], "objset": raw["objset"], "k": raw.get("k"), "obj": raw.get("obj_a"),
                    "correct": dict(zip([t for t, _ in PREDS], verdict)), "object_tokens": objm})
        print(b, raw["motion"], raw["objset"], verdict, flush=True)

    vmax = float(np.percentile(np.abs(np.concatenate([np.stack(x[5]).ravel() for x in store])), 99))
    tags = [t for t, _ in PREDS]
    for b, raw, fa, fi, tr, maps, decs, verdict, gt_dec in store:
        ho = raw.get("holdout")                                       # objholdout 시험: shape · color · mesh · both · seen
        name = (f"{ho}_" if ho else "") + f"{raw['motion']}_{raw['objset']}_k{raw.get('k')}_b{b}"
        title = (f"[{'unseen ' + ho if ho not in (None, 'seen', 'both') else ('unseen 2+' if ho == 'both' else 'seen object')}] " if ho else "") + \
                f"{MOTION.get(raw['motion'], raw['motion'])} · {raw['objset']} · {raw.get('obj_a')} · late k{raw.get('k')}"
        if MODE == "v4flat":                                          # 원본 (가능 미래) | 모델마다 예측을 decoder 로
            g1 = []
            for t in range(T):
                panels = [fa[t]] + [fa[t] if t < C else decs[k][t - C] for k in range(len(tags))]
                im = Image.fromarray(np.concatenate(panels, 1)).convert("RGB")
                heads = ["real video"] + [f"{tg}: prediction ({'correct' if verdict[k] else 'wrong'})" for k, tg in enumerate(tags)]
                for j, hd in enumerate(heads):
                    label(im, j * 256, hd + (" [ctx = real]" if t < C and j >= 1 else ""), (255, 255, 0) if (t >= C and j >= 1) else (255, 255, 255), w=255)
                    if t >= C:
                        circle(im, j * 256, tr[t])
                ImageDraw.Draw(im).text((3, 240), f"frame {RAW[t]}  {'context' if t < C else 'future'}  |  {title}  |  circle = true object", fill=(255, 255, 255), font=FONT)
                g1.append(im)
            g1[0].save(OUT / f"{name}.gif", save_all=True, append_images=g1[1:], duration=150, loop=0)
            continue
        # GIF 1: decoder
        g1 = []
        for t in range(T):
            panels = [fa[t], fi[t], fa[t] if t < C else gt_dec[t - C]] + [fa[t] if t < C else decs[k][t - C] for k in range(len(tags))]
            im = Image.fromarray(np.concatenate(panels, 1)).convert("RGB")
            heads = ["real: possible future", "real: impossible (vanish)", "true latent -> decoder (ceiling)"] + \
                    [f"{tg} prediction ({'correct' if verdict[k] else 'wrong'})" for k, tg in enumerate(tags)]
            for j, hd in enumerate(heads):
                label(im, j * 256, hd + (" [ctx]" if t < C and j >= 2 else ""), (255, 255, 0) if (t >= C and j >= 2) else (255, 255, 255))
            ImageDraw.Draw(im).text((3, 240), f"frame {RAW[t]}  {'context' if t < C else 'future'}  |  {title}", fill=(255, 255, 255), font=FONT)
            g1.append(im)
        g1[0].save(OUT / f"{name}_decoder.gif", save_all=True, append_images=g1[1:], duration=150, loop=0)
        # GIF 2: L1 Δ 지도
        g2 = []
        for t in range(T):
            panels = [fa[t]] + [fi[t] if t < C else overlay(fi[t].astype(np.float32), maps[k][(t - C) // 2], vmax) for k in range(len(tags))]
            im = Image.fromarray(np.concatenate(panels, 1)).convert("RGB")
            for j in range(len(panels)):
                if t >= C:
                    circle(im, j * 256, tr[t])
            heads = ["real: possible future"] + [f"{tg} ({'correct' if verdict[k] else 'wrong'})" for k, tg in enumerate(tags)]
            for j, hd in enumerate(heads):
                label(im, j * 256, hd, (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.text((3, 240), f"frame {RAW[t]}  {'context' if t < C else 'future'}  |  blue = impossible more surprising (right) · orange = possible more surprising (wrong) · circle = object in possible future",
                   fill=(255, 255, 255), font=FONT)
            g2.append(im)
        g2[0].save(OUT / f"{name}_l1.gif", save_all=True, append_images=g2[1:], duration=150, loop=0)
    json.dump({"vmax": vmax, "pairs": rec, "models": dict(PREDS), "decoder": str(DEC)}, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    import collections
    agg = collections.defaultdict(list)
    for r in rec:
        grp = ("static" if r["motion"] == "flat_static" else "moving") + (f"/{r['holdout']}" if r.get("holdout") else "")
        for tg, m in r["object_tokens"].items():
            agg[(grp, tg)].append((m["closer_to_object_early"], m["closer_to_object_late"], m["delta_obj"]))
    for (grp, tg), v in sorted(agg.items()):
        v = np.array(v); print(f"[object tokens] {grp:6s} {tg:18s} n={len(v)} closer-to-object early {v[:, 0].mean():.2f} late {v[:, 1].mean():.2f} | mean delta {v[:, 2].mean():+.4f}", flush=True)
    print("vmax", vmax, "->", OUT, "GIF_DONE", flush=True)


if __name__ == "__main__":
    main()
