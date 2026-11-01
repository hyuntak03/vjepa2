#!/usr/bin/env python3
"""real_bench attention 개입 — 미래 토큰이 문맥의 물체 칸을 더 보게 하면 긴 가림 뒤 물체를 이어 그리나 (2026-11-01, 사용자:
"미래 마스크가 이전 컨텍스트 물체 보도록 attn score 강제로 높여서 인터벤션 해봐").

배경: real_bench 짝 A (계속 있음 vs 사라짐) 에서 물체 칸 예측이 '없음' 정답에 더 가깝다 (z_analysis/results/realbench_object_cells).
개입: predictor 12 층 · 모든 head 의 attention logit 에 β 를 더한다 (SDPA attn_mask, 학습 없음):
  all     — 미래 토큰 전부 → 문맥에서 물체가 보이던 칸 (SAM2 마스크, 그 블록 두 장 모두 물체 칸 비율 > 0.2)
  oracle  — **정답 미래 물체 칸** (계속 있음 영상의 SAM2) 의 미래 토큰만 → 문맥 물체 칸. 정답 위치를 쓰므로 판별 상한 (CLAUDE.md 관문 4)
  placebo — 미래 토큰 전부 → 같은 블록 · 같은 개수의 물체 아닌 문맥 칸 (씨앗 고정)
짝 A 만 (문맥 = 원본, 물체가 보인다). 짝 B 는 문맥이 물체를 지운 영상이라 과거 물체 칸이 없다.
점수: 전체 칸 · 물체 칸만 (미래 물체 칸) 의 정답률 = s(사라짐) > s(계속 있음), 물체 칸에서 예측이 '있음' 정답에 더 가까운 비율.
채점 규약은 cvpr testbed 와 같다 (문맥 16 · 미래 16, float32 + fp16 autocast).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/realbench_attn_bias.py
출력: z_analysis/results/realbench_attn_bias/summary.json
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="vith,permanence_wrop_C")
ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/realbench_attn_bias"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
from analysis.intphys2.model import build_from_config
SRC = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads"
FR = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
NF, S, G, NB = 32, 256, 16, 8
NCTX = NB * S; N = 2 * NCTX
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/real_bench_quads/index.csv"))))
by = {(r["video"], r["quad_variant"]): r for r in rows}
vids = sorted({r["video"] for r in rows})


def cells_of(v, which):
    """which = ctx | fut → (8, 256) bool: 그 블록 두 장 모두 (ctx) / 어느 한 장 (fut) 에서 물체 칸 비율 > 0.2."""
    r = by[(v, "possible_present")]
    z = np.load(os.path.join(SRC, r["file_name"].rsplit("/", 1)[0], "masks.npz")); T, H, W = z["shape"]
    m = np.unpackbits(z["sam2"], axis=-1)[..., :W].astype(np.float32)
    fr = [int(x) for x in r[f"{which}_frames"].split()]
    out = np.zeros((NB, S), bool)
    for t in range(NB):
        cs = [F.adaptive_avg_pool2d(F.interpolate(torch.from_numpy(m[f])[None, None], size=(256, 256), mode="nearest"), (G, G))[0, 0].numpy() > 0.2
              for f in fr[2 * t:2 * t + 2]]
        out[t] = (np.logical_and(*cs) if which == "ctx" else np.logical_or(*cs)).ravel()
    return out


def clip(r):
    fr = np.stack([np.asarray(Image.open(os.path.join(FR, r["file_name"], f"{j:06d}.png")).convert("RGB")) for j in range(NF)])
    x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2).float().div_(255)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    return (x - MEAN) / STD


rng = np.random.default_rng(0)
CTX = {v: cells_of(v, "ctx") for v in vids}
FUT = {v: cells_of(v, "fut") for v in vids}
PLAC = {}
for v in vids:
    pl = np.zeros((NB, S), bool)
    for b in range(NB):
        k = int(CTX[v][b].sum())
        if k:
            pl[b, rng.choice(np.nonzero(~CTX[v][b])[0], k, replace=False)] = True
    PLAC[v] = pl
res = {}
for name in a.models.split(","):
    mc = dict(arch_name="vit_huge", checkpoint=CK, img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True, uniform_power=False,
              dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
              predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32", autocast="float16")
    if name != "vith":
        mc["predictor_checkpoint"] = os.path.join(REPO, "z_training/runs", name, "latest.pt")
    bundle = build_from_config(mc, dev)
    BIAS = {"m": None}
    for blk in bundle.predictor.predictor_blocks:
        f0 = blk.forward
        def fw(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, _f=f0):
            return _f(x, mask=mask, attn_mask=BIAS["m"] if BIAS["m"] is not None else attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        blk.forward = fw
    ci = torch.arange(NCTX, device=dev)[None]; ti = torch.arange(NCTX, N, device=dev)[None]
    per = []
    for v in vids:
        if not CTX[v].any():
            continue
        rp, ri = by[(v, "possible_present")], by[(v, "impossible_vanish")]
        x = torch.cat([clip(rp), clip(ri)]).to(bundle.dtype)
        keys_obj = torch.from_numpy(np.nonzero(CTX[v].ravel())[0]).to(dev)
        keys_pl = torch.from_numpy(np.nonzero(PLAC[v].ravel())[0]).to(dev)
        q_or = torch.from_numpy(np.nonzero(FUT[v].ravel())[0]).to(dev) + NCTX
        fo = torch.from_numpy(FUT[v].ravel()).to(dev)
        out = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x[:1], masks=[ci])
            h = F.layer_norm(bundle.target_encoder(x)[:, NCTX:].float(), (1280,))      # (2, 2048, D): [계속 있음, 사라짐]
            for kind in ["base"] + [f"{k}_b{bt:g}" for k in ("all", "oracle", "placebo") for bt in a.betas]:
                if kind == "base":
                    BIAS["m"] = None
                else:
                    k, bt = kind.split("_b"); bt = float(bt)
                    m = torch.zeros(1, 1, N, N, device=dev, dtype=torch.float16)
                    keys = keys_pl if k == "placebo" else keys_obj
                    if k == "oracle":
                        if len(q_or) == 0:
                            continue
                        m[0, 0, q_or[:, None], keys[None, :]] = bt
                    else:
                        m[0, 0, NCTX:, keys] = bt
                    BIAS["m"] = m
                p = bundle.predictor(z, ci, ti, mask_index=0).float()[0]
                BIAS["m"] = None
                lp, li = (p - h[0]).abs().mean(-1), (p - h[1]).abs().mean(-1)        # (2048,)
                out[kind] = dict(all=float(li.mean() - lp.mean()), obj=float(li[fo].mean() - lp[fo].mean()) if fo.any() else float("nan"),
                                 yes_closer=float((lp[fo].mean() < li[fo].mean()).float()) if fo.any() else float("nan"))
        per.append(dict(video=v, event=rp["condition"], n_ctx_obj=int(CTX[v].sum()), n_fut_obj=int(FUT[v].sum()), res=out))
        print(f"[{name}] {v} 끝", flush=True)
    res[name] = per
    del bundle; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
acc = lambda xs: 100 * np.mean([x > 0 for x in xs if x == x]) if any(x == x for x in xs) else float("nan")
for name, per in res.items():
    print(f"\n{name} · 짝 A {len(per)} 영상 (문맥에 물체 칸이 있는 것) · 정답률 % [전체 칸 / 물체 칸] · 물체 칸에서 '있음' 쪽 %")
    kinds = list(per[0]["res"])
    for k in kinds:
        q = [x["res"][k] for x in per if k in x["res"]]
        print(f"  {k:12s} {acc([y['all'] for y in q]):5.1f} / {acc([y['obj'] for y in q]):5.1f} · '있음' 쪽 {100 * np.nanmean([y['yes_closer'] for y in q]):5.1f} (n {len(q)})")
