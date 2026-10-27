"""릴리즈 predictor 예측 길이 ablation — 한 번에 예측하는데 왜 걸음마다 물체가 사라지나, 짧게 예측하면 괜찮나 (2026-10-25, 학습 없음).
사용자: "release predictor 는 one shot prediction 인데 왜 저럼..? prediction length ablation 해봐. 짧게 prediction 하면 문제 없는지".

데이터: phys3d_single_v1 val 96 장 영상 (pred_object_vector.py 와 같은 목록 · 라벨 · 물체 칸 · 배경 정의). 원본 0, 3, … 장 (4 fps).
문맥 16 장 (블록 0..7) 고정, 예측 M ∈ {1, 2, 4, 8} 블록 (2 · 4 · 8 · 16 장). 창 T = 16 + 2M 장.
  예측 = predictor(context_encoder(문맥 16 장), 미래 M 블록 자리) · 정답 = LN(target_encoder(창 T 장 한 번에)) 의 미래 블록 (각 M 마다 자기 창)
  같은 미래 블록 (걸음 k = 1..M) 을 M 끼리 비교한다.
지표 (참 자리 칸, 물체가 문맥에서 없던 새 자리 · bg 있는 블록만):
  obj_cos   cos(e_P, e_Y), e_P = (P − μ) − β(bg − μ) (예측에서 배경 성분 뺀 나머지), e_Y = Y − bg (encoder 물체 벡터)
  o_true    칸별 (X − μ) 를 [(bg − μ), (Ō − μ)] 에 회귀한 Ō 계수의 참 자리 평균 (Ō = 마지막 문맥 블록 물체 칸 평균) — 예측 · 정답
  dec_s     decoder 물체 세기: 정답 블록의 물체 칸만 예측으로 바꿔 그린 그림이 물체 지운 그림과 다른 정도 / 정답이 다른 정도
  l1_obj · l1_bg  물체 칸 · 배경 칸 토큰 L1 (예측 vs 정답)
μ 는 M = 8 정답 토큰 평균 (모든 M 에 같은 μ).

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_pred_length.py
출력: auto_research/results/release_pred_length.json
"""
import glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
from analysis.intphys2.model import build_from_config                     # noqa: E402
from evals.world_model_analysis.eval import _context_target_indices       # noqa: E402
from model import PixelDecoder                                            # noqa: E402

P3 = "/workspace/dataset/training_data/phys3d_single_v1"
OUT = os.path.join(REPO, "auto_research/results/release_pred_length.json")
C, S, G, D, NBC = 16, 256, 16, 1280, 8
MS = (1, 2, 4, 8)
IDX = [3 * k for k in range(32)]
DECODER = os.path.join(REPO, "z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt")
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
dev = torch.device("cuda")


def block_labels(st):
    pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    cc = np.arange(G) * 16 + 8.0; out = []
    for b in range(16):
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        ok = bool(min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9)
        box = np.array([min(bb[f0, 0], bb[f1, 0]), min(bb[f0, 1], bb[f1, 1]), max(bb[f0, 2], bb[f1, 2]), max(bb[f0, 3], bb[f1, 3])])
        cells = []
        if np.isfinite(box).all():
            cols = np.nonzero((cc >= box[0]) & (cc <= box[2]))[0]; rows = np.nonzero((cc >= box[1]) & (cc <= box[3]))[0]
            cells = [int(r * G + c) for r in rows for c in cols]
        out.append({"ok": ok, "cells": cells, "box": box})
    return out


files = [m["clip"] for m in json.load(open("/workspace/auto_research_cache/pred_objvec/val_meta.json"))]
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
        "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
        "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
rel = build_from_config(MCFG, dev)
st_ = torch.load(DECODER, map_location="cpu", weights_only=False); mc = st_["cfg"]["model"]
dec = PixelDecoder(in_dim=D, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st_["decoder"]); del st_
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
IND = {M: _context_target_indices(ctx_frames=C, tgt_frames=2 * M, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev) for M in MS}
cos = lambda a, b: F.cosine_similarity(a, b, dim=-1)
import decord

# 1 차: 토큰 계산 (편마다 M 4 개)
cache = []
for i, f in enumerate(files):
    lab = block_labels(np.load(f[:-4] + "_states.npz"))
    fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
    xn = ((torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD).unsqueeze(0)
    rec = {"lab": lab, "M": {}}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for M in MS:
            ci, ti = IND[M]; x = xn[:, :, : C + 2 * M]
            z = rel.context_encoder(x, masks=[ci]); p = rel.predictor(z, ci, ti, mask_index=0); h = rel.target_encoder(x)
            rec["M"][M] = (LN(h.float())[0].view(NBC + M, S, D).half(), p.float()[0].view(M, S, D).half())
    cache.append(rec)
    if i % 50 == 0:
        log(f"{i}/{len(files)}")
mu = torch.stack([r["M"][8][0].float().mean((0, 1)) for r in cache]).mean(0)

rows = []
for i, rec in enumerate(cache):
    lab = rec["lab"]; last = lab[NBC - 1]
    cov = [set(l["cells"]) for l in lab]
    for M in MS:
        Y_all, P_all = rec["M"][M][0].float(), rec["M"][M][1].float()
        bgm = torch.full((S, D), float("nan"), device=dev)
        for c in range(S):
            qs = [q for q in range(NBC) if c not in cov[q]]
            if qs:
                bgm[c] = Y_all[qs, c].mean(0)
        if not (last["ok"] and last["cells"]):
            continue
        ob = Y_all[NBC - 1, last["cells"]].mean(0)
        allobj = set().union(*cov[: NBC + M]); bgc = [c for c in range(S) if c not in allobj]
        for k in range(1, M + 1):
            b = NBC + k - 1; L = lab[b]; cells = L["cells"]
            if not L["ok"] or not cells or torch.isnan(bgm[cells]).any():
                continue
            Y, P, B = Y_all[b, cells], P_all[k - 1, cells], bgm[cells]
            pb, bb = (P - mu).reshape(-1), (B - mu).reshape(-1); beta = (pb @ bb) / (bb @ bb)
            eP, eY = (P - mu) - beta * (B - mu), Y - B
            def oscore(X):
                x, bq, o = X - mu, B - mu, (ob - mu).expand_as(X)
                bbq, oo, bo, xb, xo = (bq * bq).sum(1), (o * o).sum(1), (bq * o).sum(1), (x * bq).sum(1), (x * o).sum(1)
                return float(((bbq * xo - bo * xb) / (bbq * oo - bo ** 2)).mean())
            r = {"clip": i, "M": M, "step": k, "obj_cos": float(cos(eP, eY).mean()), "o_true_pred": oscore(P), "o_true_gt": oscore(Y),
                 "l1_obj": float((P - Y).abs().mean()), "l1_bg": float((P_all[k - 1, bgc] - Y_all[b, bgc]).abs().mean()) if bgc else float("nan")}
            Bt = Y_all[b].clone(); Bt[cells] = B; Xo = Y_all[b].clone(); Xo[cells] = P
            with torch.no_grad():
                y_ = (dec(torch.stack([Y_all[b], Bt, Xo])) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
            x0, y0, x1, y1 = [int(np.clip(v, 0, 255)) for v in L["box"]]
            if x1 > x0 and y1 > y0:
                reg = lambda t: t[..., y0:y1 + 1, x0:x1 + 1]
                dY = float((reg(y_[0]) - reg(y_[1])).abs().mean())
                if dY > 1e-4:
                    r["dec_s"] = float((reg(y_[2]) - reg(y_[1])).abs().mean()) / dY
            rows.append(r)
out = {}
for M in MS:
    out[M] = {}
    for k in range(1, M + 1):
        R = [r for r in rows if r["M"] == M and r["step"] == k]
        f = lambda key: round(float(np.nanmedian([r.get(key, np.nan) for r in R])), 3)
        out[M][k] = {"n": len(R), **{key: f(key) for key in ("obj_cos", "o_true_pred", "o_true_gt", "dec_s", "l1_obj", "l1_bg")}}
json.dump(out, open(OUT, "w"), indent=1)
for M in MS:
    print("M", M, " | ".join(f"k{k}: cos {v['obj_cos']:.2f} o {v['o_true_pred']:.2f}/{v['o_true_gt']:.2f} dec {v['dec_s']:.2f} l1o {v['l1_obj']:.3f} l1b {v['l1_bg']:.3f} (n{v['n']})" for k, v in out[M].items()))
log("saved", OUT)
