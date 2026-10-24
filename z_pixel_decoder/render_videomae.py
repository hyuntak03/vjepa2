#!/usr/bin/env python3
"""VideoMAEv2-g 가 예측한 미래를 픽셀로 (2026-10-21, 사용자 지시: "VideoMAEv2 가 예측한 미래도 VoBench 에서 pixel space 로, 어차피 decoder 붙어 있잖아").

  source cvpr/env.sh; CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=$PWD $CVPR_PY z_pixel_decoder/render_videomae.py --sets late,gravity --n 8

채점 (testbed vobench_v1*_videomae2g_wc8t16_s6) 과 같은 입력: 원본 0, 6, …, 90 장 (16 장), 224² bilinear (antialias 없음), fp16 autocast,
문맥 8 장 (tubelet 4) 만 encoder 에 보이고 미래 8 장 (tubelet 4 × 256 패치) 을 MAE decoder 가 채운다 (analysis/model_loaders._VMae*).
VideoMAE 의 출력은 **패치마다 (채널별) 평균 0 · 분산 1 로 정규화한 픽셀** (2 × 14 × 14 × 3 = 1176) 이라 픽셀로 되돌리려면 패치 평균 · 표준편차가 필요하다.
모델은 미래 패치의 그 값을 모른다 → 두 가지로 되돌린다:
  GT stats   정답 미래 패치의 평균 · 표준편차 (VideoMAE 공식 시각화와 같다) — 정답의 색 · 밝기가 새어 들어간다 (구조만 모델 몫)
  ctx stats  문맥 마지막 tubelet 같은 자리 패치의 평균 · 표준편차 — 모델 입력 안의 정보만 쓴다
클립은 clips.py (render_future.py 와 같은 클립). VideoMAE 는 VOBench 로 학습한 적 없다 (도메인 밖).
산출: z_pixel_decoder/runs/vob_grav_i1_vitb_both/videomae/<set>/ (옛 runs/videomae2g_renders 에서 옮김) — grid.png (행 = 클립 × [원본 · GT stats · ctx stats], 열 = 문맥 끝 2 장 + 미래 8 장) · <clip>.gif · summary.json"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from einops import rearrange
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.model_loaders import build_videomae2                      # noqa: E402
from analysis.intphys2.surprise import _context_target_indices         # noqa: E402
from clips import pick_clips                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="late,gravity")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--root", default=os.environ.get("CVPR_VOBENCH", "/data2/local_datasets/world/Benchmark/VOBench_v1"))
ap.add_argument("--val-frac", type=float, default=0.1)                 # decoder (vob_grav_i1_vitb_both) val 표시용 — 같은 클립을 고르려고
ap.add_argument("--out", default=os.path.join(HERE, "runs/vob_grav_i1_vitb_both/videomae"))   # decoder 는 안 쓰지만 render_future 의 모델별 폴더 옆에 (10-21)
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/videomae2/models--OpenGVLab--VideoMAE2/snapshots/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e.pth")
T, C, R, P, TB = 16, 8, 224, 14, 2
G = R // P                                                              # 16 패치 / 변
NS = G * G                                                              # 256 / tubelet
IDX = [6 * k for k in range(T)]
ctx_m, tgt_m, pred_m, _ = build_videomae2({"checkpoint": CK, "decoder_depth": 4}, dev, num_frames=T, img_size=R)
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=TB, spatial_tokens=NS, batch_size=1, device=dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))
to8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def load(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0           # (3, T, H, W) [0, 1]
    return F.interpolate(x.transpose(0, 1), size=(R, R), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


def patches(x01):
    """(3, T, R, R) → (T/2·256, 2·14·14, 3) — _VMaeTarget 과 같은 배치 (토큰 = t h w, 패치 안 = p0 p1 p2, 채널 마지막)"""
    return rearrange(x01.unsqueeze(0), "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c", p0=TB, p1=P, p2=P)[0]


def unpatch(sq):
    """(4·256, 392, 3) → (8, 3, R, R) [0, 1]"""
    v = rearrange(sq.unsqueeze(0), "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)", t=(T - C) // TB, h=G, w=G, p0=TB, p1=P, p2=P)[0]
    return v.permute(1, 0, 2, 3).clamp(0, 1)


for s_ in a.sets.split(","):
    out = os.path.join(a.out, s_); os.makedirs(out, exist_ok=True)
    clips = pick_clips(s_, a.n, a.root, a.val_frac)
    summ, panels = [], []
    for c in clips:
        xs = [load(p) for _, p in c["futures"]]
        xn = ((xs[0] - MEAN) / STD).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = pred_m(ctx_m(xn, masks=[ci]), ci, ti)                                            # (1, 4·256, 1176) 정규화 패치
        z = rearrange(z.float()[0], "n (p c) -> n p c", c=3)                                     # _VMaeTarget 의 'b n (p c)' 와 같은 순서
        sq = patches(xs[0])                                                                      # (8·256, 392, 3) [0, 1]
        fut, last = sq[C // TB * NS:], sq[(C // TB - 1) * NS: C // TB * NS]
        stat = lambda q: (q.mean(-2, keepdim=True), q.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        mg, sg = stat(fut)
        ml, sl = stat(last); ml, sl = ml.repeat((T - C) // TB, 1, 1), sl.repeat((T - C) // TB, 1, 1)
        y_gt, y_ctx = unpatch(z * sg + mg), unpatch(z * sl + ml)
        # 채점 surprise (정규화 패치 공간 L1) — 미래마다
        sur = {lab: float((z - (patches(x)[C // TB * NS:] - stat(patches(x)[C // TB * NS:])[0]) / stat(patches(x)[C // TB * NS:])[1]).abs().mean())
               for (lab, _), x in zip(c["futures"], xs)}
        real = [x[:, C:].permute(1, 0, 2, 3) for x in xs]
        copy = xs[0][:, C - 1:C].permute(1, 0, 2, 3).expand(T - C, -1, -1, -1)
        rec = {"clip": c["name"], "dec_val": c["dec_val"], "surprise": sur,
               "psnr_gt_stats": {lab: psnr(y_gt, r) for (lab, _), r in zip(c["futures"], real)},
               "psnr_ctx_stats": {lab: psnr(y_ctx, r) for (lab, _), r in zip(c["futures"], real)}, "psnr_copy": psnr(copy, real[0])}
        summ.append(rec)
        ctx8 = to8(xs[0][:, :C].permute(1, 0, 2, 3))
        rows_ = [(f"orig {lab}", to8(x.permute(1, 0, 2, 3))) for (lab, _), x in zip(c["futures"], xs)]
        rows_ += [("VideoMAE (GT stats)", torch.cat([ctx8, to8(y_gt)])), ("VideoMAE (ctx stats)", torch.cat([ctx8, to8(y_ctx)]))]
        panels.append((c["name"], rows_, rec))
        frames = []
        grows = [("video", rows_[0][1]), rows_[-2], rows_[-1]]           # GIF (10-21): 영상 | 예측 (GT stats) | 예측 (ctx stats) — VideoMAE 는 정답이 픽셀이라 encoder 칸이 없다
        for t in range(T):
            im = Image.fromarray(torch.cat([rr[t] for _, rr in grows], -1).permute(1, 2, 0).numpy()); dr = ImageDraw.Draw(im)
            for j, (lab, _) in enumerate(grows):
                dr.rectangle([j * R, 0, j * R + 170, 14], fill=(0, 0, 0))
                raw = lab.startswith(("orig", "video"))
                dr.text((j * R + 3, 1), lab + (" (ctx)" if t < C and not raw else ""), fill=(255, 255, 0) if t >= C and not raw else (255, 255, 255))
            dr.text((3, R - 14), f"frame {IDX[t]}", fill=(255, 255, 255))
            frames.append(im)
        frames[0].save(os.path.join(out, c["name"].replace("/", "__") + ".gif"), save_all=True, append_images=frames[1:], duration=250, loop=0)
        print(f"{s_} {c['name']}: PSNR GT stats {list(rec['psnr_gt_stats'].values())[0]:.1f} · ctx stats {list(rec['psnr_ctx_stats'].values())[0]:.1f}"
              f" · copy {rec['psnr_copy']:.1f} | surprise " + " · ".join(f"{k} {v:.3f}" for k, v in sur.items()), flush=True)
    COLS, s, L, top = [6, 7] + list(range(8, 16)), 128, 260, 18
    nrow = sum(len(rr) for _, rr, _ in panels)
    sheet = Image.new("RGB", (L + s * len(COLS), top + s * nrow), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
    for j, t in enumerate(COLS):
        dr.text((L + j * s + 3, 3), f"f{IDX[t]}" + (" ctx" if t < C else ""), fill=(110, 110, 110) if t < C else (0, 0, 0))
    y = top
    for name, rr, rec in panels:
        for i, (lab, seq) in enumerate(rr):
            if i == 0:
                dr.text((3, y + 3), "/".join(name.split("/")[:3]), fill=(0, 0, 0)); dr.text((3, y + 17), name.split("/")[-2][:36], fill=(60, 60, 60))
            dr.text((3, y + s - 30), lab, fill=(0, 0, 160) if lab.startswith("VideoMAE") else (0, 0, 0))
            if lab.startswith("VideoMAE"):
                key = "psnr_gt_stats" if "GT" in lab else "psnr_ctx_stats"
                dr.text((3, y + s - 16), "PSNR " + " / ".join(f"{l} {v:.1f}" for l, v in rec[key].items()), fill=(90, 90, 90))
            for j, t in enumerate(COLS):
                sheet.paste(Image.fromarray(seq[t].permute(1, 2, 0).numpy()).resize((s, s)), (L + j * s, y))
            y += s
        dr.line([(0, y - 1), (sheet.width, y - 1)], fill=(120, 120, 120), width=2)
    sheet.save(os.path.join(out, "grid.png"))
    json.dump({"set": s_, "checkpoint": CK, "clips": summ}, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    m = lambda k: np.mean([list(r[k].values())[0] if isinstance(r[k], dict) else r[k] for r in summ])
    print(f"[{s_}] 평균 PSNR ({len(summ)} clip, 가능 미래) GT stats {m('psnr_gt_stats'):.2f} · ctx stats {m('psnr_ctx_stats'):.2f} · copy {m('psnr_copy'):.2f}")
    print("산출:", out)
