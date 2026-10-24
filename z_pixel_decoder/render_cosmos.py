#!/usr/bin/env python3
"""Cosmos-Predict1-4B (자기회귀, 영상만 조건) 가 VOBench 문맥에서 만드는 픽셀 미래 (2026-10-21, 사용자: "cosmos 로 VoBench 돌려서 실제로 어떤 pixel future 만드는지").

  환경은 vjepa2 가 아니라 cosmos-predict1 (python 3.10, /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1):
  source /data/hyuntak/anaconda3/bin/activate cosmos-predict1; cd /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1
  CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PWD python /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_pixel_decoder/render_cosmos.py --sets late,gravity --n 8
  (다른 노드: z_pixel_decoder/cosmos.sbatch)

모델 규약 (cosmos_predict1/autoregressive, base 추론 = 텍스트 없음): 문맥 9 장 → 미래 24 장 (토크나이저 DV8x16x16 — 시간 8 배 · 공간 16 배 압축,
33 장 = latent 5 칸 중 문맥 2 칸), 640 × 1024. 샘플링 temperature 1.0 · top_p 0.8 (공식 기본) · seed 0 — **확률적이다** (seed 하나 = 표본 하나).
VOBench 맞추기:
  시간  원본 100 장에서 stride 2 — 문맥 raw 30, 32, …, 46 (9 장) · 미래 raw 48, 50, …, 94 (24 장). V-JEPA 의 미래 구간 (raw 48–93) 과 같고 위반 갈림 (48) 이 경계.
        V-JEPA 문맥은 raw 0–45 (16 장, stride 3) — Cosmos 는 그중 뒤쪽 30–46 만 본다 (모델이 받는 문맥이 9 장 고정).
  공간  512² 정사각 → 640² 로 줄여 좌우에 검은 띠 (pillarbox) 를 붙여 640 × 1024. 공식 전처리 (비율 유지 + 가운데 자르기) 는 위아래 37.5 % 를 잘라
        바닥 근처 물체 (late · 낙하 끝) 가 화면 밖으로 나간다. 출력은 가운데 640² 를 다시 잘라 256² 로 비교한다.
  diffusion decoder 는 끈다 (토큰을 토크나이저 decoder 로 바로 — 모델 자체가 만든 것만 보려고). guardrail 도 끈다.
비교: PSNR (미래 24 장, 256²) — 모델 · 복사 (문맥 마지막 장) · 토크나이저 상한 (정답 33 장을 encode → decode, '오른쪽 encoder 칸').
ledge 는 낙하 · 부유 두 미래 중 픽셀 MSE 가 작은 쪽을 'pixel picks' 로 적는다 (latent 채점이 아니다).
산출: z_pixel_decoder/runs/vob_grav_i1_vitb_both/cosmos_ar4b/<set>/ — grid.png · <clip>.gif (영상 | Cosmos | tokenizer encode→decode) · summary.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clips import pick_clips                                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="late,gravity")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--root", default=os.environ.get("CVPR_VOBENCH", "/data2/local_datasets/world/Benchmark/VOBench_v1"))
ap.add_argument("--checkpoint_dir", default="/data/hyuntak/project/2026/2027_cvpr/cosmos-predict1/checkpoints")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val-frac", type=float, default=0.1)                 # clips.py 가 decoder val 클립을 먼저 고르는 데 쓴다 (render_future 와 같은 클립)
ap.add_argument("--out", default=os.path.join(HERE, "runs/vob_grav_i1_vitb_both/cosmos_ar4b"))
ap.add_argument("--list-frames", action="store_true", help="쓸 프레임 경로만 찍고 끝 (다른 노드용 복사)")
a = ap.parse_args()
CTX = list(range(30, 47, 2))                                           # 9 장
FUT = list(range(48, 95, 2))                                           # 24 장
IDX = CTX + FUT
assert len(CTX) == 9 and len(IDX) == 33                                # 토크나이저 입력 33 장 (NUM_TOTAL_FRAMES)
H, W, S = 640, 1024, 256
PADL = (W - H) // 2
CLIPS = {s_: pick_clips(s_, a.n, a.root, a.val_frac) for s_ in a.sets.split(",")}
if a.list_frames:
    for s_, cs in CLIPS.items():
        for c in cs:
            for _, d in c["futures"]:
                print("\n".join(f"{d}/{i:06d}.png" for i in IDX))
    sys.exit(0)

from cosmos_predict1.autoregressive.inference.world_generation_pipeline import ARBaseGenerationPipeline  # noqa: E402  (cosmos-predict1 환경)
from cosmos_predict1.autoregressive.utils.inference import NUM_TOTAL_FRAMES                   # noqa: E402
from cosmos_predict1.autoregressive.configs.inference.inference_config import SamplingConfig  # noqa: E402
assert NUM_TOTAL_FRAMES == len(IDX)
torch._C._jit_set_texpr_fuser_enabled(False)                           # base.py 와 같다
pipe = ARBaseGenerationPipeline(inference_type="base", checkpoint_dir=a.checkpoint_dir, checkpoint_name="Cosmos-Predict1-4B",
                                disable_diffusion_decoder=True, offload_guardrail_models=False, offload_diffusion_decoder=False,
                                offload_network=False, offload_tokenizer=False, disable_guardrail=True, parallel_size=1)
SAMP = SamplingConfig(echo=True, temperature=1.0, top_p=0.8, compile_sampling=True)        # validate_args 의 기본값
VT = pipe.model.tokenizer.video_tokenizer
psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))


def load(d):
    """→ (3, 33, 640, 1024) [-1, 1] pillarbox · (33, 3, 256, 256) [0, 1] 비교용"""
    x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX])).permute(0, 3, 1, 2).float().cuda() / 255
    sq = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    big = F.pad(sq, (PADL, W - H - PADL), value=-1.0)
    small = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
    return big.permute(1, 0, 2, 3), small


def to_small(v):
    """(33, 3, 640, 1024) [0, 1] → 가운데 정사각 → (33, 3, 256, 256)"""
    return F.interpolate(v[..., PADL:PADL + H], size=(S, S), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)


to8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
COLS, s, L, top = [7, 8, 9, 12, 15, 18, 21, 24, 27, 30, 32], 128, 260, 36
for s_, cs in CLIPS.items():
    out = os.path.join(a.out, s_); os.makedirs(out, exist_ok=True)
    summ, panels = [], []
    for c in cs:
        t0 = time.time()
        bigs, smalls = zip(*[load(p) for _, p in c["futures"]])
        inp = bigs[0].clone()
        inp[:, 9:] = inp[:, 8:9]                                                                   # 공식과 같이 문맥 뒤는 마지막 장 반복 (토크나이저가 33 장을 요구)
        gen = pipe.generate(inp_vid=inp.unsqueeze(0), sampling_config=SAMP, num_input_frames=9, seed=a.seed)   # (33, 640, 1024, 3) uint8
        y = to_small(torch.from_numpy(gen).permute(0, 3, 1, 2).float().cuda() / 255)
        with torch.no_grad():
            _, ind = VT.encode(bigs[0].unsqueeze(0).cuda())
            rec = to_small((VT.decode(ind)[0].float().permute(1, 0, 2, 3) * 0.5 + 0.5).clamp(0, 1))      # 토크나이저 상한
        labs = [lab for lab, _ in c["futures"]]
        rec_ = {"clip": c["name"], "seed": a.seed, "sec": round(time.time() - t0, 1),
                "psnr": {lab: psnr(y[9:], sm[9:]) for lab, sm in zip(labs, smalls)},
                "psnr_tokenizer": psnr(rec[9:], smalls[0][9:]), "psnr_copy": psnr(smalls[0][8:9].expand(24, -1, -1, -1), smalls[0][9:]),
                "psnr_ctx_echo": psnr(y[:9], smalls[0][:9])}
        if len(labs) > 1:
            mse = {lab: float(((y[9:] - sm[9:]) ** 2).mean()) for lab, sm in zip(labs, smalls)}
            rec_["pixel_mse"] = mse; rec_["pixel_picks"] = min(mse, key=mse.get)
        summ.append(rec_)
        rows = [(f"orig {lab}", to8(sm), None) for lab, sm in zip(labs, smalls)]
        notes = [("PSNR " + " / ".join(f"{l} {v:.1f}" for l, v in rec_["psnr"].items()), (90, 90, 90))]
        if "pixel_picks" in rec_:
            notes.append(("pixel picks: " + rec_["pixel_picks"], (150, 60, 0)))
        rows += [("tokenizer enc->dec", to8(rec), [(f"PSNR {rec_['psnr_tokenizer']:.1f}  (tokenizer ceiling)", (90, 90, 90))]),
                 ("Cosmos-Predict1-4B", to8(y), notes)]
        panels.append((c["name"], rows))
        frames = []                                                                                  # GIF: 영상 | Cosmos | tokenizer
        g = [("video", rows[0][1]), ("Cosmos-4B", rows[-1][1]), ("tokenizer enc->dec", rows[-2][1])]
        for t in range(33):
            im = Image.fromarray(torch.cat([q[t] for _, q in g], -1).permute(1, 2, 0).numpy()); dr = ImageDraw.Draw(im)
            for j, (lab, _) in enumerate(g):
                raw = lab == "video"
                dr.rectangle([j * S, 0, j * S + 200, 14], fill=(0, 0, 0))
                dr.text((j * S + 3, 1), lab + (" (ctx)" if t < 9 and not raw else ""), fill=(255, 255, 0) if t >= 9 and not raw else (255, 255, 255))
            dr.text((3, S - 14), f"frame {IDX[t]}", fill=(255, 255, 255))
            frames.append(im)
        frames[0].save(os.path.join(out, c["name"].replace("/", "__") + ".gif"), save_all=True, append_images=frames[1:], duration=125, loop=0)
        print(f"{s_} {c['name']}: PSNR " + " · ".join(f"{l} {v:.1f}" for l, v in rec_["psnr"].items()) +
              f" | tokenizer {rec_['psnr_tokenizer']:.1f} · copy {rec_['psnr_copy']:.1f}" + (f" | pixel picks {rec_['pixel_picks']}" if "pixel_picks" in rec_ else "") +
              f" ({rec_['sec']} s)", flush=True)
    first = lambda r: list(r["psnr"].values())[0]
    m = {k: float(np.mean([f(r) for r in summ])) for k, f in [("cosmos", first), ("tokenizer", lambda r: r["psnr_tokenizer"]), ("copy", lambda r: r["psnr_copy"])]}
    nrow = sum(len(rr) for _, rr in panels)
    sh = Image.new("RGB", (L + s * len(COLS), top + s * nrow), (255, 255, 255)); d = ImageDraw.Draw(sh)
    d.text((3, 3), f"Cosmos-Predict1-4B (AR, video only, seed {a.seed})  ·  {s_}  ·  mean PSNR (possible future, 24 frames) {m['cosmos']:.2f} dB  ·  "
                   f"copy {m['copy']:.2f}  ·  tokenizer {m['tokenizer']:.2f}", fill=(0, 0, 0))
    for j, t in enumerate(COLS):
        d.text((L + j * s + 3, top - 15), f"f{IDX[t]}" + (" ctx" if t < 9 else ""), fill=(110, 110, 110) if t < 9 else (0, 0, 0))
    y0 = top
    for name, rr in panels:
        for i, (lab, seq, notes) in enumerate(rr):
            if i == 0:
                d.text((3, y0 + 3), "/".join(name.split("/")[:3]), fill=(0, 0, 0)); d.text((3, y0 + 17), name.split("/")[-2][:36], fill=(60, 60, 60))
            d.text((3, y0 + s - 30 - 14 * max(len(notes or []) - 1, 0)), lab, fill=(0, 0, 160) if notes else (0, 0, 0))
            for q, (txt, col) in enumerate(notes or []):
                d.text((3, y0 + s - 16 - 14 * q), txt, fill=col)
            for j, t in enumerate(COLS):
                sh.paste(Image.fromarray(seq[t].permute(1, 2, 0).numpy()).resize((s, s)), (L + j * s, y0))
            y0 += s
        d.line([(0, y0 - 1), (sh.width, y0 - 1)], fill=(120, 120, 120), width=2)
    sh.save(os.path.join(out, "grid.png"))
    json.dump({"set": s_, "model": "Cosmos-Predict1-4B (= Cosmos-1.0-Autoregressive-4B, md5 5fdc62fc…)", "ctx_raw": CTX, "fut_raw": FUT,
               "sampling": {"temperature": 1.0, "top_p": 0.8, "seed": a.seed}, "mean": m, "clips": summ},
              open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[{s_}] 평균 PSNR (가능 미래 24 장, {len(summ)} clip) Cosmos {m['cosmos']:.2f} · 복사 {m['copy']:.2f} · 토크나이저 상한 {m['tokenizer']:.2f} → {out}", flush=True)
