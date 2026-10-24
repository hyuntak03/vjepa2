#!/usr/bin/env python3
"""고리 A ↔ B 다리 [E]: 위치 불확실성 → (칸 · 차원별) 중앙값 토큰의 물체 존재감 곡선 (auto_research, 2026-10-21, 리더 최우선).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_posunc_median.py --source phys3d --n_obj 16
  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_posunc_median.py --source vobench --n_obj 16

물체 하나마다 배경 이미지 Bg 와 물체 픽셀 (마스크 M) 을 얻는다:
  phys3d  : 물체가 움직이는 창의 시간 중앙값 = 배경, 한 장에서 |장 − 배경| > 6 단계 (bbox 안) = 물체
  vobench : 물체 영속 위반 block (가림 없음) 의 pos_a 장 0 (물체) 과 pos_b 장 0 (빈 장면) — 차 > 8 단계 = 물체 (그림자 포함)
표본 k = 1..32: δ_k ~ 균등 [−σ, σ]² (정수 px), 물체를 기준 자리 + δ_k 에 붙인 블록
  정지 블록 = (C(δ), C(δ)) · 움직이는 블록 = (C(δ), C(δ + (8, 0)))  (블록 안 8 px)
  noise: 0 = 깨끗 · 1 = 표본마다 ±1/255 섭동 추가 (nuisance 포함)
각 σ (0 · 1 · 2 · 4 · 8 · 16 px) 에서 32 표본의 블록 토큰 → 칸 · 차원별 **중앙값 토큰** (L1 최적) · **평균 토큰** (L2 최적).
존재감 (칸 c, 물체 칸 = σ=0 마스크 덮임 > 0.3): p(c) = ⟨T(c) − B(c), T0(c) − B(c)⟩ / |T0(c) − B(c)|²
  (T0 = σ=0 깨끗한 토큰, B = 배경만 있는 같은 종류 블록 토큰). 1 = 물체 그대로, 0 = 배경. 물체 칸 평균.
  + |T − T0| / |B − T0| (물체 칸, L1 비) · 기준 자리 ±2 칸 안 최대 cos(T(c), 물체 템플릿) (템플릿 = T0 물체 칸 평균, 배경 기준 cos 와 함께)
같은 것을 픽셀 공간 (픽셀 중앙값 이미지 자체 · 그 이미지를 인코딩한 토큰) 과 무작위 초기화 encoder 로."""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="phys3d", choices=["phys3d", "vobench"])
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n_obj", type=int, default=16)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--sigmas", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
ap.add_argument("--v", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda"); G, R = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
SD = _load_checkpoint(CK)["target_encoder"]
ENC = {}
for name in ("target", "random"):
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if name == "target":
        _load_state_dict(e, SD, tag="t")
    ENC[name] = e.to(dev, torch.bfloat16).eval()
del SD
gen = torch.Generator(device=dev).manual_seed(0)
rng = np.random.default_rng(0)


@torch.no_grad()
def enc(name, px):                    # px (B,2,R,R,3) float 0..255 → (B,256,D)
    x = (px.div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    out = []
    for i in range(0, x.size(0), 16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = ENC[name](x[i:i + 16].to(torch.bfloat16)).float()
        out.append(F.layer_norm(h, (h.size(-1),)))
    return torch.cat(out)


# ---------------- 물체 모으기: (Bg (R,R,3), Obj (R,R,3), M (R,R) bool)
objs = []
if a.source == "phys3d":
    files = sorted(glob.glob(os.path.join(a.p3, "pix_val_s*.npy")))
    cand = []
    for f in files:
        sh = f.split("_s")[-1].split(".")[0]
        lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz"))
        for w in range(lab["bbox"].shape[0]):
            bb = lab["bbox"][w]
            if not np.isfinite(bb).all():
                continue
            cx = (bb[:, 0] + bb[:, 2]) / 2; cy = (bb[:, 1] + bb[:, 3]) / 2; size = (bb[:, 2] - bb[:, 0]).mean()
            if np.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > 3 * size and 20 < size < 64:
                cand.append((f, w, sh))
    rng.shuffle(cand)
    for f, w, sh in cand:
        if len(objs) >= a.n_obj:
            break
        lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz")); bb = lab["bbox"][w]
        fr = torch.from_numpy(np.asarray(np.load(f, mmap_mode="r")[w])).float().to(dev)
        Bg = fr.median(0).values
        ok = [i for i in range(32) if bb[i, 0] > 40 and bb[i, 2] < R - 56 and bb[i, 1] > 40 and bb[i, 3] < R - 40 and lab["vis_frac"][w][i] > 0.95]
        if not ok:
            continue
        i = ok[len(ok) // 2]; x0, y0, x1, y1 = [int(v) for v in bb[i]]
        diff = (fr[i] - Bg).abs().mean(-1)
        M = torch.zeros(R, R, dtype=torch.bool, device=dev)
        M[y0 - 2:y1 + 3, x0 - 2:x1 + 3] = diff[y0 - 2:y1 + 3, x0 - 2:x1 + 3] > 6
        if M.sum() < 200:
            continue
        objs.append((Bg, fr[i], M, f"phys3d/{os.path.basename(f)}/{w}"))
else:
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["obj_b"] == "empty" and r["occlusion"] == "visible"]
    by = {}
    for r in rows:
        by.setdefault(r["source_block"], {})[r["variant"]] = r
    def load(rel, fi):
        x = torch.from_numpy(np.asarray(Image.open(os.path.join("/workspace/dataset/benchmarks/VOBench_v1", rel, f"{fi:06d}.png")).convert("RGB"))).to(dev)
        x = F.interpolate(x.permute(2, 0, 1)[None].float(), size=(R, R), mode="bilinear", align_corners=False)[0]
        return x.round().clamp(0, 255).permute(1, 2, 0)
    for sb, v in sorted(by.items()):
        if len(objs) >= a.n_obj or "pos_a" not in v or "pos_b" not in v:
            continue
        for fi in (45, 24, 0, 69, 93):                     # 물체가 가장자리에서 떨어진 장을 고른다
            A, B = load(v["pos_a"]["file_name"], fi), load(v["pos_b"]["file_name"], fi)
            M = (A - B).abs().mean(-1) > 8
            ys, xs = torch.where(M)
            if M.sum() < 150 or M.float().mean() > 0.15 or ys.min() < 40 or ys.max() > R - 40 or xs.min() < 40 or xs.max() > R - 56:
                continue
            objs.append((B, A, M, f"vobench/{sb}/f{fi}"))
            break
print(f"[{a.source}] 물체 {len(objs)} 개", flush=True)


def comp(Bg, Obj, M, dy, dx):
    Ms = torch.roll(M, (dy, dx), (0, 1)); src = torch.roll(Obj, (dy, dx), (0, 1))
    out = Bg.clone(); out[Ms] = src[Ms]
    return out


def presence(T, T0, B, objc):         # T,T0,B (256,D) → 물체 칸 평균 존재감 · L1 비
    num = ((T - B) * (T0 - B)).sum(-1); den = ((T0 - B) ** 2).sum(-1).clamp_min(1e-6)
    p = (num / den)[objc].mean()
    l1 = ((T - T0).abs().mean(-1) / (B - T0).abs().mean(-1).clamp_min(1e-6))[objc].mean()
    return float(p), float(l1)


def maxcos(T, tmpl, near):
    c = F.cosine_similarity(T, tmpl[None], dim=-1)
    return float(c[near].max())


RES = []
for oi, (Bg, Obj, M, tag) in enumerate(objs):
    cov = M.float().view(G, 16, G, 16).mean(dim=(1, 3)).reshape(-1)
    objc = cov > 0.3
    if objc.sum() < 1:
        continue
    oy, ox = torch.where(cov.view(G, G) > 0)
    near = torch.zeros(G, G, dtype=torch.bool, device=dev)
    near[max(int(oy.min()) - 2, 0):int(oy.max()) + 3, max(int(ox.min()) - 2, 0):int(ox.max()) + 3] = True; near = near.view(-1)
    pixobj = M.float()
    rec = {"tag": tag, "obj_px": int(M.sum()), "obj_cells": int(objc.sum())}
    for kind in ("static", "move"):
        v = 0 if kind == "static" else a.v
        bgblk = torch.stack([Bg, Bg])[None]
        blk0 = torch.stack([comp(Bg, Obj, M, 0, 0), comp(Bg, Obj, M, 0, v)])[None]
        for name in ("target", "random"):
            B_ = enc(name, bgblk)[0]; T0 = enc(name, blk0)[0]
            tmpl = T0[objc].mean(0)
            rec[f"{kind}/{name}/bg_maxcos"] = maxcos(B_, tmpl, near)
            for s in a.sigmas:
                d = rng.integers(-s, s + 1, size=(a.K, 2)) if s > 0 else np.zeros((a.K, 2), int)
                blks = torch.stack([torch.stack([comp(Bg, Obj, M, int(dy), int(dx)), comp(Bg, Obj, M, int(dy), int(dx) + v)]) for dy, dx in d])   # (K,2,R,R,3)
                for noise in (0, 1):
                    x = blks if noise == 0 else (blks + torch.randint(-1, 2, blks.shape, device=dev, generator=gen).float()).clamp(0, 255)
                    H = enc(name, x)                                                          # (K,256,D)
                    Tmed = H.median(0).values; Tmean = H.mean(0)
                    for agg, T in (("median", Tmed), ("mean", Tmean)):
                        p, l1 = presence(T, T0, B_, objc)
                        rec[f"{kind}/{name}/n{noise}/s{s}/{agg}/p"] = p
                        rec[f"{kind}/{name}/n{noise}/s{s}/{agg}/l1ratio"] = l1
                        rec[f"{kind}/{name}/n{noise}/s{s}/{agg}/maxcos"] = maxcos(T, tmpl, near)
                    # 표본 하나 (평균 내기 전) 의 존재감: 개별 토큰은 물체를 담고 있나
                    ps = [presence(H[k], T0, B_, objc)[0] for k in range(0, a.K, 8)]
                    rec[f"{kind}/{name}/n{noise}/s{s}/single/p"] = float(np.mean(ps))
                    if name == "target" and noise == 0:
                        # 픽셀 공간: 픽셀 중앙값 이미지 자체의 존재감 + 그것을 인코딩한 토큰
                        pm = blks.median(0).values                                            # (2,R,R,3)
                        P0 = blk0[0]; PB = bgblk[0]
                        pv = lambda X: X.permute(1, 2, 0, 3).reshape(R, R, -1)
                        num = ((pv(pm) - pv(PB)) * (pv(P0) - pv(PB))).sum(-1); den = ((pv(P0) - pv(PB)) ** 2).sum(-1)
                        numc = num.view(G, 16, G, 16).sum(dim=(1, 3)).reshape(-1); denc = den.view(G, 16, G, 16).sum(dim=(1, 3)).reshape(-1).clamp_min(1e-6)
                        rec[f"{kind}/pixel/s{s}/median/p"] = float((numc / denc)[objc].mean())
                        pmn = blks.mean(0)
                        num = ((pv(pmn) - pv(PB)) * (pv(P0) - pv(PB))).sum(-1)
                        rec[f"{kind}/pixel/s{s}/mean/p"] = float((num.view(G, 16, G, 16).sum(dim=(1, 3)).reshape(-1) / denc)[objc].mean())
                        for nm2 in ("target", "random"):
                            Tp = enc(nm2, pm[None])[0]
                            T0b = T0 if nm2 == "target" else enc("random", blk0)[0]; Bb = B_ if nm2 == "target" else enc("random", bgblk)[0]
                            rec[f"{kind}/pixmedian_enc_{nm2}/s{s}/p"] = presence(Tp, T0b, Bb, objc)[0]
    RES.append(rec)
    print(f"[{oi}] {tag} 물체 {rec['obj_px']} px · 칸 {rec['obj_cells']} | 정지 · target 중앙값 존재감 σ: " +
          " ".join(f"{s}:{rec[f'static/target/n0/s{s}/median/p']:.2f}" for s in a.sigmas) +
          " | 픽셀 중앙값: " + " ".join(f"{s}:{rec[f'static/pixel/s{s}/median/p']:.2f}" for s in a.sigmas) +
          " | random 중앙값: " + " ".join(f"{s}:{rec[f'static/random/n0/s{s}/median/p']:.2f}" for s in a.sigmas), flush=True)

keys = sorted(set(k for r in RES for k in r if k not in ("tag", "obj_px", "obj_cells")))
summ = {k: float(np.mean([r[k] for r in RES if k in r])) for k in keys}
summ["n_obj"] = len(RES); summ["obj_px_mean"] = float(np.mean([r["obj_px"] for r in RES])); summ["obj_cells_mean"] = float(np.mean([r["obj_cells"] for r in RES]))
print(f"\n== {a.source} 요약 (물체 {len(RES)} 개, 평균 {summ['obj_px_mean']:.0f} px · 칸 {summ['obj_cells_mean']:.1f})")
for kind in ("static", "move"):
    print(f" [{kind}] σ = " + " ".join(f"{s:>5d}" for s in a.sigmas))
    for lab_, key in (("target 중앙값 (깨끗)", "target/n0/s{s}/median/p"), ("target 중앙값 (+±1/255)", "target/n1/s{s}/median/p"),
                      ("target 평균 (깨끗)", "target/n0/s{s}/mean/p"), ("target 표본 하나", "target/n0/s{s}/single/p"),
                      ("random 중앙값", "random/n0/s{s}/median/p"), ("random 평균", "random/n0/s{s}/mean/p"),
                      ("픽셀 중앙값", "pixel/s{s}/median/p"), ("픽셀 평균", "pixel/s{s}/mean/p"),
                      ("픽셀 중앙값 → target", "pixmedian_enc_target/s{s}/p"), ("픽셀 중앙값 → random", "pixmedian_enc_random/s{s}/p"),
                      ("target 중앙값 L1/(배경 L1)", "target/n0/s{s}/median/l1ratio"), ("target 중앙값 최대 cos", "target/n0/s{s}/median/maxcos")):
        vals = [summ.get(f"{kind}/" + key.format(s=s)) for s in a.sigmas]
        print(f"   {lab_:28s} " + " ".join(f"{v:5.2f}" if v is not None else "   — " for v in vals))
    print(f"   배경 토큰 최대 cos (기준) target {summ.get(f'{kind}/target/bg_maxcos', float('nan')):.2f} · random {summ.get(f'{kind}/random/bg_maxcos', float('nan')):.2f}")
out = a.out or os.path.join(os.path.dirname(__file__), f"../results/enc_posunc_median_{a.source}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"summary": summ, "records": RES, "sigmas": a.sigmas, "K": a.K, "v": a.v}, open(out, "w"), indent=1)
print("→", out)
