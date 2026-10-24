#!/usr/bin/env python3
"""고리 6 [E]: 세 갈래 섞기 — 간다 (p) · 남는다 (q) · 사라진다 (r = 1 − p − q) — 의 칸 · 차원별 중앙값 (L1 최적) 토큰 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_mixture3_median.py --source phys3d
  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_mixture3_median.py --source vobench

물체 16 개 (enc_mixture_median.py 와 같은 고르기), 정지 물체 블록, 이동 d = 24 px (가로).
표본 = 배경 블록 하나 + 물체 상태 하나: 간다 (물체 @ x0 + d) · 남는다 (@ x0) · 사라진다 (물체 없음, 배경만).
배경 (nuisance 조건):
  natural  자연 영상 (HT100M) 한 장면의 연속 두 장 — 정지 칸 비율 0.4–0.8 인 수집 창 3 개의 원본을 4 fps 로 64 장 다시 읽어 블록 63 개 (장면 전환 없는 구간)
  real     그 clip 의 실제 연속 두 장 (3D 합성: 원래 물체 자리를 시간 중앙값으로 메움 / VOBench: 빈 장면 arm) — 블록 31 개
갈래마다 배경 블록 전부를 한 번씩 쓴 표본 집합을 인코딩하고, (p, q) 마다 간다 round(pK) · 남는다 round(qK) · 사라진다 나머지를 모아
칸 · 차원별 중앙값 · 평균. 존재감 (enc_mixture_median.py 와 같은 사영 지표):
  새 자리 = ⟨T − B̄, T̄go − B̄⟩/|T̄go − B̄|² (이동 물체만 덮는 칸) · 옛 자리 = ⟨T − B̄, T̄stay − B̄⟩/|T̄stay − B̄|² (제자리 물체만 덮는 칸)
  B̄ · T̄go · T̄stay = 사라진다 · 간다 · 남는다 표본만 같은 방식으로 집계한 것."""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="phys3d", choices=["phys3d", "vobench"])
ap.add_argument("--n_obj", type=int, default=16)
ap.add_argument("--d", type=int, default=24)
ap.add_argument("--out", default=None)
a3 = ap.parse_args()
_saved = list(sys.argv)
sys.argv = [sys.argv[0], "--source", a3.source, "--n_obj", str(a3.n_obj)]
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "enc_mixture_median.py")).read()
exec(compile(_src.split("# 자연 영상 배경: HT100M")[0], "enc_mixture_median.py", "exec"))   # 인코더 · 물체 · real 배경 · paste
sys.argv = _saved
import decord   # noqa: E402

# ---------------- 자연 배경: 수집 창의 원본을 4 fps 로 64 장 다시 읽는다
zf = sorted(glob.glob(os.path.join(a.nat, "ht100m_target_encoder_s*.npz")))
cands = []
for f in zf:
    z = np.load(f); meta = json.load(open(f[:-4] + "_meta.json"))
    for i in range(z["pixD"].shape[0]):
        fr_ = (z["pixD"][i] < 1).mean()
        if 0.4 <= fr_ <= 0.8:
            cands.append((fr_, meta[i]))
cands.sort(key=lambda c: -c[0])
NAT = []
for fr_, m in cands:
    if len(NAT) >= 3:
        break
    vr = decord.VideoReader(m["path"], num_threads=2)
    sd = m["stride"]; s0 = max(0, min(m["start"], len(vr) - 1 - sd * 63))
    idx = [s0 + sd * k for k in range(64) if s0 + sd * k < len(vr)]
    x = torch.from_numpy(vr.get_batch(idx).asnumpy())
    H_, W_ = x.shape[1:3]; s = min(H_, W_); y0, x0_ = (H_ - s) // 2, (W_ - s) // 2
    x = x[:, y0:y0 + s, x0_:x0_ + s].permute(0, 3, 1, 2).float().to(dev)
    x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).round().clamp(0, 255).permute(0, 2, 3, 1)
    # 장면 전환 (연속 장 평균 |Δ| > 25 단계) 없는 가장 긴 구간
    dd = (x[1:] - x[:-1]).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    cuts = [0] + [k + 1 for k in np.where(dd > 25)[0]] + [len(x)]
    seg = max(zip(cuts[:-1], cuts[1:]), key=lambda c: c[1] - c[0])
    if seg[1] - seg[0] < 32:
        continue
    xs = x[seg[0]:seg[1]]
    NAT.append(torch.stack([xs[k:k + 2] for k in range(len(xs) - 1)]))
print(f"[{a.source}] 물체 {len(objs)} 개 · 자연 배경 블록 수 {[n.shape[0] for n in NAT]}", flush=True)

GRID = [round(0.1 * i, 1) for i in range(11)]
PQ = [(p, q) for p in GRID for q in GRID if p + q <= 1.0 + 1e-9]


def proj(T, B_, Tr, cells):
    num = ((T - B_) * (Tr - B_)).sum(-1); den = ((Tr - B_) ** 2).sum(-1).clamp_min(1e-6)
    return float((num / den)[cells].mean())


d = a3.d
RES = []
for oi, (Bg, Obj, M, real, oth, tag) in enumerate(objs):
    cov0 = M.float().view(G, 16, G, 16).mean(dim=(1, 3)).reshape(-1)
    cov1 = torch.roll(M, d, 1).float().view(G, 16, G, 16).mean(dim=(1, 3)).reshape(-1)
    old = (cov0 > 0.3) & (cov1 < 0.05); new = (cov1 > 0.3) & (cov0 < 0.05)
    if old.sum() < 1 or new.sum() < 1:
        continue
    rec = {"tag": tag}
    for cname, bgs in (("natural", NAT[oi % len(NAT)]), ("real", real)):
        K = bgs.shape[0]
        pg, ps, pv = rng.permutation(K), rng.permutation(K), rng.permutation(K)
        go = torch.stack([paste(bgs[k], Obj, M, d) for k in pg]); st = torch.stack([paste(bgs[k], Obj, M, 0) for k in ps]); va = bgs[pv].clone()
        hg, hs, hv = enc(go), enc(st), enc(va)
        rec[f"{cname}/K"] = K
        for agg in ("median", "mean"):
            f_ = (lambda H: H.median(0).values) if agg == "median" else (lambda H: H.mean(0))
            B_, Tg, Ts = f_(hv), f_(hg), f_(hs)
            for p, q in PQ:
                ng, ns = int(round(p * K)), int(round(q * K)); nv = max(K - ng - ns, 0)
                H = torch.cat([hg[:ng], hs[:ns], hv[:nv]])
                T = f_(H)
                rec[f"{cname}/{agg}/p{p}_q{q}/new"] = proj(T, B_, Tg, new)
                rec[f"{cname}/{agg}/p{p}_q{q}/old"] = proj(T, B_, Ts, old)
    RES.append(rec)
    print(f"[{oi}] {tag} natural 중앙값 (p,q)=(0.5,0.4): 새 {rec['natural/median/p0.5_q0.4/new']:.2f} 옛 {rec['natural/median/p0.5_q0.4/old']:.2f} · "
          f"(0.5,0.5): 새 {rec['natural/median/p0.5_q0.5/new']:.2f} 옛 {rec['natural/median/p0.5_q0.5/old']:.2f}", flush=True)

keys = sorted(set(k for r in RES for k in r if k != "tag"))
summ = {k: float(np.mean([r[k] for r in RES if k in r])) for k in keys}
summ["n_obj"] = len(RES)
print(f"\n== {a.source} 요약 (물체 {len(RES)} 개, 이동 {d} px) — 존재감 새 자리 / 옛 자리")
for cname in ("natural", "real"):
    for agg in ("median", "mean"):
        print(f" [{cname} · {agg}]  행 = p (간다), 열 = q (남는다)")
        print("        " + "".join(f"   q={q:<4}" for q in GRID))
        for p in GRID:
            cells = []
            for q in GRID:
                if p + q <= 1.0 + 1e-9:
                    cells.append(f"{summ[f'{cname}/{agg}/p{p}_q{q}/new']:4.2f}/{summ[f'{cname}/{agg}/p{p}_q{q}/old']:4.2f}")
                else:
                    cells.append("    —    ")
            print(f"  p={p:<4} " + " ".join(cells))
out = a3.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../results/enc_mixture3_median_{a.source}.json")
json.dump({"summary": summ, "records": RES, "grid": GRID, "d": d}, open(out, "w"), indent=1)
print("→", out)
