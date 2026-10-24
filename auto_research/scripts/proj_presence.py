#!/usr/bin/env python3
"""[P] [E] 의 사영 존재감을 학습된 AR 출력에 (auto_research, 2026-10-21). 두 갈래 섞기 (제자리 1−π / 이동 π) 와 같은 자로 맞댄다.

  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/proj_presence.py --run natural_ar_vw_state latest.pt

사영 존재감 (enc_mixture_median.py 와 같은 식) = 칸별 ⟨T − B̄, R − B̄⟩ / |R − B̄|² 의 칸 평균.
  B̄ (물체 없는 기준) = 그 칸에 물체가 닿지 않았던 문맥 블록 (t−7..t) 토큰의 평균. 그런 블록이 없는 칸은 뺀다.
  새 자리 칸 = 블록 t+1 덮임 > 0.3 이고 블록 t 덮임 < 0.05 (두 장 중 최대), R = 정답 h_{t+1}.
  옛 자리 칸 = 블록 t 덮임 > 0.3 이고 블록 t+1 덮임 < 0.05, R = h_t.
대상: (1) 모델 TF 한 걸음 예측 (t = 7..14, 움직인 쌍: 이동 > 8 px · 두 블록 보임)
      (2) 학습 없는 두 갈래 섞기: 이동 갈래 = h_{t+1} + nuisance 표본, 제자리 갈래 = h_t + nuisance 표본 (K = 15 · 갈래마다),
          nuisance 표본 = 같은 창 먼 칸의 연속 블록 차 / √2 (칸 · 블록 무작위). π 마다 이동 round(πK) 개 + 제자리 나머지 → 칸별 중앙값 · 평균.
      (3) 굴린 예측 걸음별 (블록 8+j): 새 자리 = 블록 8+j 덮임 > 0.3 & 블록 7 < 0.05 (R = h_{8+j}), 옛 자리 = 블록 7 > 0.3 & 블록 8+j < 0.05 (R = h_7)."""
import argparse, glob, json, os, sys
import numpy as np, torch, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--K", type=int, default=15)
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/proj_presence"))
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K, NB, L = 256, 16, 8, 8, 16, 8
PIS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
tag = f"{a.run[0]}__{a.run[1].replace('.pt', '')}"
os.makedirs(a.out, exist_ok=True)
D_ = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
model = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
model = model.to(dev).eval()
gen = torch.Generator(device=dev); gen.manual_seed(0)


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    m = m.reshape(G, G).copy()
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m.reshape(-1)


def proj(T, B_, R_, cells):
    """T · B_ · R_ (S, D) 또는 (n, S, D) → 칸별 사영의 칸 평균 (cells bool S)."""
    num = ((T - B_) * (R_ - B_)).sum(-1); den = ((R_ - B_) ** 2).sum(-1).clamp_min(1e-6)
    return (num / den)[..., cells].mean(-1)


TF, MIX, RO = [], [], []
for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    for w in range(X.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        cmax = np.maximum(cv[0::2], cv[1::2]).reshape(NB, S)                       # 블록별 칸 덮임 (두 장 중 최대)
        touch = cmax > 0
        ok = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        far = ~dil(touch.any(0), 2)
        if far.sum() < 10:
            continue
        pos = lab["pos"][w]
        h = torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_tf = model.forward_seq(h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0)).float()[0].view(NB - 1, S, -1)
            p_ro = model.rollout_kv(h[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        farc = torch.from_numpy(np.where(far)[0]).to(dev)
        noise_pool = ((h[1:, farc] - h[:-1, farc]) / 2 ** 0.5).reshape(-1, h.size(-1))   # 실제 nuisance 표본
        for t in range(C - 1, NB - 1):
            d = pos[2 * t + 3] - pos[2 * t + 1]
            if not (visb[t] and visb[t + 1] and np.isfinite(d).all() and np.linalg.norm(d) > 8):
                continue
            new = (cmax[t + 1] > 0.3) & (cmax[t] < 0.05); old = (cmax[t] > 0.3) & (cmax[t + 1] < 0.05)
            free = ~touch[t - L + 1:t + 1]                                         # (8, S) 물체가 안 닿은 문맥 블록
            cnt = free.sum(0)
            new &= cnt > 0; old &= cnt > 0
            if not (new.any() and old.any()):
                continue
            fm = torch.from_numpy(free).to(dev).float()
            B_ = (h[t - L + 1:t + 1] * fm[..., None]).sum(0) / torch.from_numpy(np.maximum(cnt, 1)).to(dev).float()[:, None]
            nm, om = torch.from_numpy(new).to(dev), torch.from_numpy(old).to(dev)
            rec = dict(t=t, dn=float(np.linalg.norm(d)), win=f"{sh}:{w}",
                       new=float(proj(p_tf[t], B_, h[t + 1], nm)), old=float(proj(p_tf[t], B_, h[t], om)),
                       new_copy=float(proj(h[t], B_, h[t + 1], nm)), old_copy=float(proj(h[t], B_, h[t], om)),
                       new_true=float(proj(h[t + 1], B_, h[t + 1], nm)), old_true=float(proj(h[t + 1], B_, h[t], om)),
                       new_mean8=float(proj(h[t - L + 1:t + 1].mean(0), B_, h[t + 1], nm)), old_mean8=float(proj(h[t - L + 1:t + 1].mean(0), B_, h[t], om)))
            TF.append(rec)
            # 두 갈래 섞기 (학습 없음): 표본 = 갈래 토큰 + 실제 nuisance (같은 칸에 독립 표본)
            Kk = a.K
            cells = nm | om
            ci = torch.where(cells)[0]
            ix = torch.randint(0, noise_pool.shape[0], (2 * Kk, len(ci)), device=dev, generator=gen)
            nz = noise_pool[ix]                                                    # (2K, n, D)
            # 공정판: 이동 갈래 = 옮긴 복사 (h_t 를 정답 변위 정수 칸만큼) — 입력만으로 만든 "완전 운반" (정답의 잡음을 공유하지 않는다)
            mc = torch.roll(h[t].view(G, G, -1), shifts=(int(round(d[1] / 16)), int(round(d[0] / 16))), dims=(0, 1)).reshape(S, -1)
            rec.update(new_mc=float(proj(mc, B_, h[t + 1], nm)), old_mc=float(proj(mc, B_, h[t], om)))
            mrec = {}
            for src, gtok in (("true", h[t + 1]), ("mc", mc)):
                go = gtok[ci][None] + nz[:Kk]; st = h[t][ci][None] + nz[Kk:]
                for pi in PIS:
                    ng = int(round(pi * Kk))
                    Hs = torch.cat([go[:ng], st[ng:]])
                    for agg in ("median", "mean"):
                        T = Hs.median(0).values if agg == "median" else Hs.mean(0)
                        Tf = h[t + 1].clone(); Tf[ci] = T                           # 칸 자리 맞추기
                        mrec[f"{src}/{agg}/{pi}"] = (float(proj(Tf, B_, h[t + 1], nm)), float(proj(Tf, B_, h[t], om)))
            MIX.append(mrec)
        # 굴린 예측 걸음별
        if visb[C - 1]:
            free = ~touch[:C]; cnt = free.sum(0)
            fm = torch.from_numpy(free).to(dev).float()
            B_ = (h[:C] * fm[..., None]).sum(0) / torch.from_numpy(np.maximum(cnt, 1)).to(dev).float()[:, None]
            for j in range(K):
                b = C + j
                D = float(np.linalg.norm(pos[2 * b + 1] - pos[2 * C - 1]))
                if not (visb[b] and np.isfinite(D) and D > 16):
                    continue
                new = (cmax[b] > 0.3) & (cmax[C - 1] < 0.05) & (cnt > 0); old = (cmax[C - 1] > 0.3) & (cmax[b] < 0.05) & (cnt > 0)
                if not (new.any() and old.any()):
                    continue
                nm, om = torch.from_numpy(new).to(dev), torch.from_numpy(old).to(dev)
                RO.append(dict(j=j, new=float(proj(p_ro[j], B_, h[b], nm)), old=float(proj(p_ro[j], B_, h[C - 1], om)),
                               new_mean8=float(proj(h[:C].mean(0), B_, h[b], nm)), old_mean8=float(proj(h[:C].mean(0), B_, h[C - 1], om))))
    print(f, len(TF), len(RO), flush=True)

out = {"run": a.run, "n_tf": len(TF), "tf": {}, "mix": {}, "ro": {}}
m = lambda k: float(np.mean([r[k] for r in TF]))
for k in ("new", "old", "new_copy", "old_copy", "new_true", "old_true", "new_mean8", "old_mean8", "new_mc", "old_mc"):
    out["tf"][k] = m(k)
rng = np.random.RandomState(0)
wins = np.array([r["win"] for r in TF]); uw = np.unique(wins); ixw = {u: np.where(wins == u)[0] for u in uw}
bs = []
for _ in range(500):
    sel = np.concatenate([ixw[u] for u in rng.choice(uw, len(uw))])
    bs.append([np.mean([TF[i]["new"] for i in sel]), np.mean([TF[i]["old"] for i in sel]), np.mean([TF[i]["new"] + TF[i]["old"] for i in sel])])
out["tf"]["ci_new_old_sum"] = np.percentile(bs, [2.5, 97.5], axis=0).tolist()
print(f"\n== {tag}: 사영 존재감 (TF 한 걸음, 움직인 쌍 {len(TF)})")
print(f"  모델     새 자리 {out['tf']['new']:.3f} · 옛 자리 {out['tf']['old']:.3f} · 합 {out['tf']['new'] + out['tf']['old']:.3f}   95 % {np.round(out['tf']['ci_new_old_sum'], 3).tolist()}")
print(f"  복사     새 {out['tf']['new_copy']:.3f} · 옛 {out['tf']['old_copy']:.3f} | 정답 새 {out['tf']['new_true']:.3f} · 옛 {out['tf']['old_true']:.3f} | 8 블록 평균 새 {out['tf']['new_mean8']:.3f} · 옛 {out['tf']['old_mean8']:.3f}")
print(f"  옮긴 복사 (입력만으로 만든 완전 운반) 새 {out['tf']['new_mc']:.3f} · 옛 {out['tf']['old_mc']:.3f}")
print("  두 갈래 섞기 (실제 nuisance 표본 K = 15 / 갈래): π → (새, 옛, 합). 이동 갈래 = 정답 h_{t+1} (정답 잡음 공유) / 옮긴 복사 (공정판)")
for src in ("true", "mc"):
    for agg in ("median", "mean"):
        row = []
        for pi in PIS:
            nw = float(np.mean([r[f"{src}/{agg}/{pi}"][0] for r in MIX])); od = float(np.mean([r[f"{src}/{agg}/{pi}"][1] for r in MIX]))
            out["mix"][f"{src}/{agg}/{pi}"] = (nw, od)
            row.append(f"π{pi}: {nw:.2f}/{od:.2f}/{nw + od:.2f}")
        print(f"   {src:4s} {agg:6s} " + " · ".join(row))
print("  굴린 예측 걸음별 (움직임 > 16 px): 새 · 옛 · 합 | 8 블록 평균 새 · 옛")
for j in range(K):
    R_ = [r for r in RO if r["j"] == j]
    if len(R_) < 5:
        continue
    e = {k: float(np.mean([r[k] for r in R_])) for k in ("new", "old", "new_mean8", "old_mean8")}
    out["ro"][j] = dict(n=len(R_), **e)
    print(f"   걸음 {j + 1}: n {len(R_):3d}  {e['new']:.3f} · {e['old']:.3f} · {e['new'] + e['old']:.3f} | {e['new_mean8']:.3f} · {e['old_mean8']:.3f}")
json.dump(out, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
