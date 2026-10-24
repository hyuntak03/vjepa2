#!/usr/bin/env python3
"""고리 A ↔ B 다리 [E] 2 차: "제자리 (1−π) / 이동 (π)" 두 갈래 섞기에서 중앙값 (L1 최적) · 평균 (L2 최적) 토큰의 새 자리 존재감 · 옛 자리 잔상 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_mixture_median.py --source phys3d
  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_mixture_median.py --source vobench

물체 16 개 (enc_posunc_median.py 와 같은 고르기). 정지 물체 블록. 이동 d = 24 · 48 px (가로).
표본 K 개: 각 표본은 배경 블록 하나 위에 물체를 x0 (제자리) 또는 x0 + d (이동) 에 붙인 것. π 마다 이동 round(πK) 개 + 제자리 나머지 (배경은 표본끼리 겹치지 않게 나눠 씀).
nuisance 조건 (표본마다 무엇이 다른가):
  clean      모든 표본이 같은 배경 (3D 합성: 시간 중앙값, VOBench: 빈 장면 한 장) → 토큰 두 종류
  pm1        같은 배경 + 표본마다 ±1/255 섭동
  real       표본마다 그 clip 의 실제 연속 두 장 (3D 합성: 원래 물체 자리를 시간 중앙값으로 메움 / VOBench: 빈 장면 arm 의 장들) → 압축 · 렌더 · 번짐 nuisance 가 표본마다 새로
  real_other (3D 합성만) 실제 장 그대로 — 원래 물체가 다른 곳 (기준 영역에서 24 px 넘게 떨어진 장만) 에서 움직이며 번짐을 만든다
  natural    자연 영상 (HT100M, 정지 칸이 가장 많은 창) 의 연속 두 장을 배경으로
존재감 = ⟨T − B̄, T̄ref − B̄⟩ / |T̄ref − B̄|² (새 자리: 이동 물체만 덮는 칸, ref = π=1 집계 / 옛 자리: 제자리 물체만 덮는 칸, ref = π=0 집계),
B̄ = 같은 배경들의 물체 없는 표본을 같은 방식 (중앙값 / 평균) 으로 집계한 것. 픽셀 공간도 같은 식."""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="phys3d", choices=["phys3d", "vobench"])
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--nat", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--n_obj", type=int, default=16)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda"); G, R = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(e, _load_checkpoint(CK)["target_encoder"], tag="t"); e = e.to(dev, torch.bfloat16).eval()
gen = torch.Generator(device=dev).manual_seed(0)
rng = np.random.default_rng(0)
PIS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
DS = (24, 48)


@torch.no_grad()
def enc(px):                          # (B,2,R,R,3) float 0..255 → (B,256,D)
    x = (px.div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    out = []
    for i in range(0, x.size(0), 16):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = e(x[i:i + 16].to(torch.bfloat16)).float()
        out.append(F.layer_norm(h, (h.size(-1),)))
    return torch.cat(out)


def paste(bg2, Obj, M, dx):           # bg2 (2,R,R,3) → 물체를 두 장에 같은 자리로
    Ms = torch.roll(M, dx, 1); src = torch.roll(Obj, dx, 1)
    out = bg2.clone(); out[:, Ms] = src[Ms]
    return out


# ---------------- 물체 · 실제 배경 블록 모으기
objs = []   # (Bg_ref (R,R,3), Obj, M, real_blocks (n,2,R,R,3), other_blocks or None, tag)
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
        # real: 장마다 원래 물체 bbox (4 px 팽창) 를 시간 중앙값으로 메운다
        filled = fr.clone()
        for k in range(32):
            if np.isfinite(bb[k]).all():
                a0, b0, a1, b1 = [int(v) for v in bb[k]]
                filled[k, max(b0 - 4, 0):b1 + 5, max(a0 - 4, 0):a1 + 5] = Bg[max(b0 - 4, 0):b1 + 5, max(a0 - 4, 0):a1 + 5]
        real = torch.stack([filled[k:k + 2] for k in range(31)])
        # real_other: 원래 물체가 기준 영역 (x0 ~ x1 + 48) 에서 24 px 넘게 떨어진 장의 연속 두 장
        oth = []
        for k in range(31):
            far = True
            for kk in (k, k + 1):
                if not np.isfinite(bb[kk]).all():
                    continue
                a0, b0, a1, b1 = bb[kk]
                if not (a1 < x0 - 24 or a0 > x1 + 48 + 24 or b1 < y0 - 24 or b0 > y1 + 24):
                    far = False
            if far:
                oth.append(fr[k:k + 2])
        objs.append((Bg, fr[i], M, real, torch.stack(oth) if len(oth) >= 8 else None, f"phys3d/{os.path.basename(f)}/{w}"))
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
        for fi in (45, 24, 0, 69, 93):
            A, B = load(v["pos_a"]["file_name"], fi), load(v["pos_b"]["file_name"], fi)
            M = (A - B).abs().mean(-1) > 8
            ys, xs = torch.where(M)
            if M.sum() < 150 or M.float().mean() > 0.15 or ys.min() < 40 or ys.max() > R - 40 or xs.min() < 40 or xs.max() > R - 56:
                continue
            frs = torch.stack([load(v["pos_b"]["file_name"], 3 * k) for k in range(32)])
            objs.append((B, A, M, torch.stack([frs[k:k + 2] for k in range(31)]), None, f"vobench/{sb}/f{fi}"))
            break
# 자연 영상 배경: HT100M 수집 창 중 정지 칸 비율 상위 3 개
NAT = []
zf = sorted(glob.glob(os.path.join(a.nat, "ht100m_target_encoder_s*.npz")))
cands = []
for f in zf:
    z = np.load(f)
    for i in range(z["pixD"].shape[0]):
        cands.append(((z["pixD"][i] < 1).mean(), f, i))
# 정지 화면 (슬라이드 · 정지 영상) 을 피하고, 카메라가 고정이면서 일부가 움직이는 창: 정지 칸 비율 0.4–0.8 중 가장 높은 3 개
cands = sorted([c for c in cands if 0.4 <= c[0] <= 0.8], reverse=True)
for frac, f, i in cands[:3]:
    P = np.load(f[:-4] + "_pix.npy", mmap_mode="r")
    fr = torch.from_numpy(np.asarray(P[i])).float().to(dev)                                  # (16,R,R,3)
    NAT.append(torch.stack([fr[k:k + 2] for k in range(15)]))
print(f"[{a.source}] 물체 {len(objs)} 개 · 자연 배경 창 정지 칸 비율 {[round(c[0], 2) for c in cands[:3]]}", flush=True)


def proj(T, B_, Tr, cells):
    num = ((T - B_) * (Tr - B_)).sum(-1); den = ((Tr - B_) ** 2).sum(-1).clamp_min(1e-6)
    return float((num / den)[cells].mean())


def pixproj(Pm, PB, Pr, cells):       # (2,R,R,3) 칸별
    pv = lambda X: X.permute(1, 2, 0, 3).reshape(R, R, -1)
    num = ((pv(Pm) - pv(PB)) * (pv(Pr) - pv(PB))).sum(-1).view(G, 16, G, 16).sum(dim=(1, 3)).reshape(-1)
    den = ((pv(Pr) - pv(PB)) ** 2).sum(-1).view(G, 16, G, 16).sum(dim=(1, 3)).reshape(-1).clamp_min(1e-6)
    return float((num / den)[cells].mean())


RES = []
for oi, (Bg, Obj, M, real, oth, tag) in enumerate(objs):
    rec = {"tag": tag}
    for d in DS:
        cov0 = M.float().view(G, 16, G, 16).mean(dim=(1, 3)).reshape(-1)
        cov1 = torch.roll(M, d, 1).float().view(G, 16, G, 16).mean(dim=(1, 3)).reshape(-1)
        old = (cov0 > 0.3) & (cov1 < 0.05); new = (cov1 > 0.3) & (cov0 < 0.05)
        if old.sum() < 1 or new.sum() < 1:
            continue
        conds = {"clean": None, "pm1": None, "real": real, "natural": NAT[oi % len(NAT)]}
        if oth is not None:
            conds["real_other"] = oth
        for cname, bgs in conds.items():
            if bgs is None:
                K = 32
                base = torch.stack([Bg, Bg])[None].expand(3 * K, -1, -1, -1, -1).clone()
            else:
                n = bgs.shape[0]; K = n
                idx = np.concatenate([rng.permutation(n) for _ in range(3)])
                base = bgs[idx]
            bgS, bgG, bgB = base[:K], base[K:2 * K], base[2 * K:3 * K]
            stay = torch.stack([paste(b, Obj, M, 0) for b in bgS]); go = torch.stack([paste(b, Obj, M, d) for b in bgG]); bonly = bgB.clone()
            if cname == "pm1":
                stay, go, bonly = [(x + torch.randint(-1, 2, x.shape, device=dev, generator=gen).float()).clamp(0, 255) for x in (stay, go, bonly)]
            if cname == "clean":
                hs, hg, hb = enc(stay[:1]).expand(K, -1, -1), enc(go[:1]).expand(K, -1, -1), enc(bonly[:1]).expand(K, -1, -1)
            else:
                hs, hg, hb = enc(stay), enc(go), enc(bonly)
            for agg in ("median", "mean"):
                f_ = (lambda H: H.median(0).values) if agg == "median" else (lambda H: H.mean(0))
                B_, Tg, Ts = f_(hb), f_(hg), f_(hs)
                fp = (lambda X: X.median(0).values) if agg == "median" else (lambda X: X.mean(0))
                PB, PG, PS = fp(bonly), fp(go), fp(stay)
                for pi in PIS:
                    ng = int(round(pi * K))
                    H = torch.cat([hg[:ng], hs[ng:]])
                    T = f_(H)
                    rec[f"d{d}/{cname}/pi{pi}/{agg}/new"] = proj(T, B_, Tg, new)
                    rec[f"d{d}/{cname}/pi{pi}/{agg}/old"] = proj(T, B_, Ts, old)
                    if cname in ("real", "pm1"):
                        X = torch.cat([go[:ng], stay[ng:]])
                        Pm = fp(X)
                        rec[f"d{d}/pix_{cname}/pi{pi}/{agg}/new"] = pixproj(Pm, PB, PG, new)
                        rec[f"d{d}/pix_{cname}/pi{pi}/{agg}/old"] = pixproj(Pm, PB, PS, old)
            # 표본 하나의 nuisance 크기 (물체 칸 · 먼 칸): 표본끼리 토큰 차 / 물체 대비
            if cname != "clean":
                contrast = (Tg - B_).abs().mean(-1)[new].mean()
                spread = (hg[1:] - hg[:-1]).abs().mean(-1)[:, new].mean() / np.sqrt(2)
                rec[f"d{d}/{cname}/nuis_over_contrast"] = float(spread / contrast)
    RES.append(rec)
    msg = " | ".join(f"{c}: " + " ".join(f"{rec.get(f'd24/{c}/pi{pi}/median/new', float('nan')):.2f}" for pi in PIS)
                     for c in ("clean", "pm1", "real", "natural"))
    print(f"[{oi}] {tag} d24 중앙값 새 자리 존재감 (π 0→1) {msg}", flush=True)

keys = sorted(set(k for r in RES for k in r if k != "tag"))
summ = {k: float(np.mean([r[k] for r in RES if k in r])) for k in keys}
cnt = {k: int(sum(k in r for r in RES)) for k in keys}
summ["n_obj"] = len(RES)
print(f"\n== {a.source} 요약 (물체 {len(RES)} 개) — 존재감 (1 = 물체 그대로, 0 = 배경)")
for d in DS:
    print(f" [이동 {d} px] π =" + "".join(f"{pi:>6}" for pi in PIS) + "   (nuisance / 물체 대비)")
    for c in ("clean", "pm1", "real", "real_other", "natural", "pix_pm1", "pix_real"):
        for agg in ("median", "mean"):
            for where in ("new", "old"):
                vals = [summ.get(f"d{d}/{c}/pi{pi}/{agg}/{where}") for pi in PIS]
                if any(v is None for v in vals):
                    continue
                nc = summ.get(f"d{d}/{c}/nuis_over_contrast")
                print(f"   {c:10s} {agg:6s} {'새 자리' if where == 'new' else '옛 자리'} " + "".join(f"{v:6.2f}" for v in vals)
                      + (f"   {nc:.2f}" if nc is not None and where == "new" and agg == "median" else "") + f"  (n {cnt[f'd{d}/{c}/pi0.5/{agg}/{where}']})")
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../results/enc_mixture_median_{a.source}.json")
json.dump({"summary": summ, "records": RES, "pis": PIS}, open(out, "w"), indent=1)
print("→", out)
