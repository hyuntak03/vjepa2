#!/usr/bin/env python3
"""[T] 개입 준비: 정답 블록 토큰의 잡음을 줄이는 인코딩 (16 비트 → 32 비트 계산, 무작위 평활 = 작은 픽셀 잡음 K 번 평균) 이
   (1) 잡음 바닥을 얼마나 낮추나 (2) 물체 대비는 남기나 (3) 평균 보상 · 운반 보상을 어떻게 바꾸나 — 학습 없이 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/smooth_target_check.py --n 24

phys3d val 캐시의 픽셀 (블록 16 개 창) 을 다시 인코딩한다. 변형:
  bf16       기존 캐시와 같은 방식 (확인용)
  fp32       같은 입력, 32 비트 계산
  rs{K}_bf16 픽셀에 가우시안 잡음 (표준편차 sig/255) 을 K 번 더해 각각 인코딩 → 블록별 LN 뒤 평균  (Cohen 등 2019 의 무작위 평활 꼴)
  rs{K}_fp32 같은 것을 32 비트로
재는 것 (칸별 차원 평균 L1):
  noise   물체가 2 칸 안에 한 번도 안 온 칸의 |h_{b+1} − h_b|           (잡음 바닥)
  objstat 물체가 정지한 쌍에서 물체 안쪽 칸의 |h_{t+1} − h_t|
  scale   서로 다른 창의 같은 칸 거리 (단위)
  contrast 물체 안쪽 칸 (덮임 ≥ 0.8) 의 토큰 vs 그 칸에 물체 없던 블록 평균
  보상: 물체가 움직인 쌍에서 copy · mean8 · 운반 오라클 (배경 = 물체 없던 문맥 평균, 물체 칸 = 궤적 평균) 의 화면 평균 손실."""
import argparse, glob, json, os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--sig", type=float, default=2.0, help="평활 잡음 표준편차 (/255)")
ap.add_argument("--variants", default="bf16,fp32,rs_bf16,rs_fp32")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/smooth_target_check.json"))
a = ap.parse_args()
dev = torch.device("cuda")
G, NB, S = 16, 16, 256
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


@torch.no_grad()
def encode(pix, fp32, K, sig):                       # pix (32,256,256,3) uint8 → (16,256,1280) float32
    x = torch.from_numpy(np.ascontiguousarray(pix)).to(dev).permute(3, 0, 1, 2)[None].float() / 255
    outs = []
    for k in range(max(K, 1)):
        xx = x + (torch.randn_like(x) * sig / 255 if K > 0 else 0)
        xn = (xx - MEAN) / STD
        if fp32:
            h = AR.encode_blocks(enc, xn, 2)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = AR.encode_blocks(enc, xn.to(torch.bfloat16), 2)
        outs.append(h.float()[0].view(NB, S, -1))
    return torch.stack(outs).mean(0).cpu().numpy()


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


lab = np.load(os.path.join(a.cache, "lab_val_s0.npz"))
PX = np.load(os.path.join(a.cache, "pix_val_s0.npy"), mmap_mode="r")
HC = np.load(os.path.join(a.cache, "h_val_target_s0.npy"), mmap_mode="r")
# 움직이는 물체가 있는 창을 고른다 (평가에 쓸 쌍이 생기도록)
cand = []
for w in range(PX.shape[0]):
    sp = np.linalg.norm(np.diff(lab["pos"][w][1::2], axis=0), axis=-1)
    cand.append((np.nanmean(sp > 8), w))
wins = [w for _, w in sorted(cand, reverse=True)[: a.n]]
VAR = {"bf16": (False, 0), "fp32": (True, 0), "rs_bf16": (False, a.K), "rs_fp32": (True, a.K)}
VAR = {k: VAR[k] for k in a.variants.split(",")}
acc = {v: {k: [] for k in ("noise", "objstat", "contrast", "copy", "mean8", "oracle", "cache_diff")} for v in VAR}
Hs = {v: [] for v in VAR}
for i, w in enumerate(wins):
    cvf = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0)); cvb = np.maximum(cvf[0::2], cvf[1::2])
    ok = lab["inframe_frac"][w] > 0.9
    vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
    pos = lab["pos"][w]
    never = ~dil((cvb > 0).any(0), 2).reshape(-1)
    for v, (fp32, K) in VAR.items():
        torch.manual_seed(1000 + i)
        h = encode(PX[w], fp32, K, a.sig)
        Hs[v].append(h[:, ::17].copy())                                   # 척도용 (칸 일부)
        if v == "bf16":
            acc[v]["cache_diff"].append(float(np.abs(h - HC[w].astype(np.float32)).mean()))
        if never.sum():
            acc[v]["noise"].append(float(np.abs(h[1:, never] - h[:-1, never]).mean()))
        inside = (cvb >= 0.8).reshape(NB, -1)
        free = (cvb == 0).reshape(NB, -1)
        cnt = free.sum(0)
        bg = (h * free[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]
        cm = inside & (cnt > 0)[None]
        if cm.sum():
            acc[v]["contrast"].append(float(np.abs(h[cm.nonzero()[0], cm.nonzero()[1]] - bg[cm.nonzero()[1]]).mean()))
        for t in range(7, NB - 1):
            d = pos[2 * t + 3] - pos[2 * t + 1]; dn = np.linalg.norm(d)
            if not np.isfinite(dn):
                continue
            y = h[t + 1]
            if dn < 1 and vis_b[t] and vis_b[t + 1]:
                ins = inside[t] & inside[t + 1]
                if ins.sum():
                    acc[v]["objstat"].append(float(np.abs(y[ins] - h[t][ins]).mean()))
                continue
            if not (dn > 8 and vis_b[t] and vis_b[t + 1]):
                continue
            ctx = h[t - 7:t + 1]
            fr = (cvb[t - 7:t + 1] == 0).reshape(8, -1); c8 = fr.sum(0)
            bg8 = np.where((c8 > 0)[:, None], (ctx * fr[..., None]).sum(0) / np.maximum(c8, 1)[:, None], h[t])
            orc = bg8.copy(); acc_o, n_o = np.zeros_like(bg8), 0
            for b in range(t - 7, t + 1):
                if not vis_b[b]:
                    continue
                db = pos[2 * t + 3] - pos[2 * b + 1]
                if not np.isfinite(db).all():
                    continue
                acc_o += np.roll(np.roll(h[b].reshape(G, G, -1), int(round(db[1] / 16)), 0), int(round(db[0] / 16)), 1).reshape(S, -1); n_o += 1
            dst = (cvb[t + 1] > 0).reshape(-1)
            if n_o:
                orc[dst] = acc_o[dst] / n_o
            acc[v]["copy"].append(float(np.abs(h[t] - y).mean())); acc[v]["mean8"].append(float(np.abs(ctx.mean(0) - y).mean()))
            acc[v]["oracle"].append(float(np.abs(orc - y).mean()))
    print(f"{i + 1}/{len(wins)}", flush=True)
res = {}
for v in VAR:
    H = np.stack(Hs[v])                                                   # (n,16,16,D)
    scale = float(np.abs(H[1:] - H[:-1]).mean())                          # 다른 창 같은 칸 (창 순서 이웃)
    r = {k: float(np.mean(x)) for k, x in acc[v].items() if x}
    r["scale_other_window"] = scale
    r["n_mov_pairs"] = len(acc[v]["copy"])
    if "copy" in r:
        r["avg_reward_pct"] = 100 * (r["copy"] - r["mean8"]) / r["copy"]
        r["transport_reward_pct"] = 100 * (r["mean8"] - r["oracle"]) / r["copy"]
    res[v] = r
    print(v, json.dumps({k: round(x, 4) for k, x in r.items()}), flush=True)
res["args"] = vars(a); res["windows"] = wins
json.dump(res, open(a.out, "w"), indent=1)
