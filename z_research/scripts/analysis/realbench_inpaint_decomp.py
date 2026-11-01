#!/usr/bin/env python3
"""real_bench 점수가 inpainting 흔적 때문인가 — 미래 칸을 물체 / 지운 영역 (물체 밖) / 손대지 않은 칸으로 나눠 판정 기여를 잰다
(2026-11-01, 사용자: "inpainting 이 문제일까?").

배경: 전체 판정이 배경 칸 판정과 같았다 (물체 칸은 미래 칸의 1.3 %, z_analysis/results/realbench_object_cells).
배경 차이의 원인 후보:
  (1) inpainting 흔적 — 물체를 지울 때 그림자까지 넓혀 지우고 다시 그린 영역 (removal 마스크 중 물체 밖)
  (2) encoder 의 전역 섞기 — 지운 영역 밖 픽셀은 원본 그대로인데 (README: "지운 영역 밖의 픽셀은 원본을 그대로 복사") ViT 가 화면 전체를 섞어
      물체 유무가 다른 칸 latent 까지 바꾼다 → inpainting 이 아니라 물체 유무 자체가 번진 것
칸 분류 (미래 블록마다, 두 장 중 하나라도 비율 > 0.2):
  O = SAM2 물체 칸 · R = removal 마스크 칸 중 O 아닌 것 (inpainting 만) · U = 나머지 (픽셀이 원본과 같다)
재는 것 (짝 A = 계속 있음 vs 사라짐, 짝 B = 처음부터 없음 vs 생겨남):
  정답 차이 |h_가능 − h_불가능| 칸 평균 (분류별) · 판별 기여 = Σ_분류 (|p − h_불가능| − |p − h_가능|) / 전체 칸 수 (분류별 합 = 전체 판별 차이)
  분류별로만 판정했을 때 정답률 · 전체 판정의 부호를 정한 분류
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/realbench_inpaint_decomp.py
출력: z_analysis/results/realbench_inpaint_decomp/summary.json
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "z_analysis/results/realbench_inpaint_decomp"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
from analysis.intphys2.model import build_from_config
SRC = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads"
FR = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
NF, S, G, NB = 32, 256, 16, 8; NCTX = NB * S
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/real_bench_quads/index.csv"))))
by = {(r["video"], r["quad_variant"]): r for r in rows}
vids = sorted({r["video"] for r in rows})


def cell_masks(v):
    r = by[(v, "possible_present")]
    z = np.load(os.path.join(SRC, r["file_name"].rsplit("/", 1)[0], "masks.npz")); T, H, W = z["shape"]
    fut = [int(x) for x in r["fut_frames"].split()]
    out = {}
    for key in ("sam2", "removal"):
        m = np.unpackbits(z[key], axis=-1)[..., :W].astype(np.float32)
        a = np.zeros((NB, S), bool)
        for t in range(NB):
            for f in fut[2 * t:2 * t + 2]:
                a[t] |= (F.adaptive_avg_pool2d(F.interpolate(torch.from_numpy(m[f])[None, None], size=(256, 256), mode="nearest"), (G, G))[0, 0].numpy() > 0.2).ravel()
        out[key] = a
    O = out["sam2"]; R = out["removal"] & ~O; U = ~(O | R)
    return dict(O=O.ravel(), R=R.ravel(), U=U.ravel())


def clip(r):
    fr = np.stack([np.asarray(Image.open(os.path.join(FR, r["file_name"], f"{j:06d}.png")).convert("RGB")) for j in range(NF)])
    x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2).float().div_(255)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    return (x - MEAN) / STD


mc = dict(arch_name="vit_huge", checkpoint=CK, img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True, uniform_power=False,
          dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
          predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32", autocast="float16")
bundle = build_from_config(mc, dev)
ci = torch.arange(NCTX, device=dev)[None]; ti = torch.arange(NCTX, 2 * NCTX, device=dev)[None]
rec = []
for v in vids:
    M = {k: torch.from_numpy(x).to(dev) for k, x in cell_masks(v).items()}
    for pair, pos, imp in (("A", "possible_present", "impossible_vanish"), ("B", "possible_absent", "impossible_appear")):
        rp, ri = by[(v, pos)], by[(v, imp)]
        x = torch.cat([clip(rp), clip(ri)]).to(bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x[:1], masks=[ci])
            h = F.layer_norm(bundle.target_encoder(x)[:, NCTX:].float(), (1280,))
            p = bundle.predictor(z, ci, ti, mask_index=0).float()[0]
        lp, li = (p - h[0]).abs().mean(-1), (p - h[1]).abs().mean(-1)
        dh = (h[0] - h[1]).abs().mean(-1)
        d = li - lp
        rec.append(dict(video=v, pair=pair, quality=rp["quality"], event=rp["condition"],
                        frac={k: float(m.float().mean()) for k, m in M.items()},
                        dh={k: float(dh[m].mean()) if m.any() else float("nan") for k, m in M.items()},
                        contrib={k: float(d[m].sum() / d.numel()) for k, m in M.items()},
                        cat_acc={k: float(d[m].mean() > 0) if m.any() else float("nan") for k, m in M.items()},
                        total=float(d.mean())))
json.dump(rec, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
med = lambda xs: float(np.nanmedian(xs))
for pr in "AB":
    q = [r for r in rec if r["pair"] == pr]
    print(f"\n짝 {pr} ({len(q)} 영상) — 전체 정답률 {100 * np.mean([r['total'] > 0 for r in q]):.1f}")
    print(f"  {'분류':24s} {'칸 비율':>8s} {'정답 차이 |Δh|':>14s} {'판별 기여 (중앙)':>16s} {'이 분류만 정답률':>16s} {'전체와 부호 같음':>16s}")
    for k, nm in (("O", "물체"), ("R", "지운 영역 (물체 밖)"), ("U", "손대지 않음 (픽셀 같음)")):
        same = np.mean([np.sign(r["contrib"][k]) == np.sign(r["total"]) for r in q])
        print(f"  {nm:24s} {100 * med([r['frac'][k] for r in q]):7.1f}% {med([r['dh'][k] for r in q]):14.4f} {med([r['contrib'][k] for r in q]):+16.2e} "
              f"{100 * np.nanmean([r['cat_acc'][k] for r in q]):15.1f}% {100 * same:15.1f}%")
    share = [abs(r["contrib"]["U"]) / (abs(r["contrib"]["O"]) + abs(r["contrib"]["R"]) + abs(r["contrib"]["U"]) + 1e-12) for r in q]
    print(f"  판별 기여 절대값 중 '손대지 않음' 몫 중앙 {100 * med(share):.0f} %")
