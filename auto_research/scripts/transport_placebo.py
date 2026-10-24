#!/usr/bin/env python3
"""[L] 운반 신뢰 b 도구 검사 — 위약 (90° 돌린 흐름) · 복사 정답 (auto_research, 2026-10-22 KST 16 시).

사용자가 붙여 준 다른 기계의 결과: 토큰 옮기기 b 는 "복사를 정답으로 넣어도" 0.14–0.37 이 나와 도구 검사에 떨어지고,
위약 (실제 방향 − 90° 돌린 방향) 을 빼면 자연 영상 데이터 보상 0.15 vs 블록 AR 0.005–0.011 이라 "모델 ≈ 데이터 최적" 이 틀린다.
같은 일이 이 기계의 class_transport 회귀 (SSv2 수동 · 행위자 캐시, 기준 블록 AR) 에서도 일어나는지 본다.

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=<g> /venv/vjepa/bin/python -W ignore auto_research/scripts/transport_placebo.py

회귀는 class_transport 와 같다 (y ≈ a·m + b·W, 움직이는 칸 |U[t+1]| ≥ 0.5, t = 4 · 5 · 6). 새로 넣은 것:
  W_rot  = h_t 를 U[t+1] 을 90° 돌린 흐름으로 옮긴 것 (같은 크기, 틀린 방향) — 위약
  y_copy = h_t (다음 블록이 아무것도 안 움직였다고 친 복사 정답) — 도구가 옳으면 b_true ≈ 0 이어야 한다
  위약 뺀 운반 = b_true − b_rot (데이터 · 모델 각각)"""
import argparse, json, os, sys
import numpy as np, torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/ssv2_classes")
ap.add_argument("--groups", nargs="+", default=["passive", "actor"])
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


from analysis.intphys2.model import _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
dev = torch.device("cuda")
D_ = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
model = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
model = model.to(dev).eval()

KINDS = ("true", "rot")
YS = ("data", "model", "copy")
res = {}
rng = np.random.RandomState(0)
for grp in a.groups:
    Hs = np.load(os.path.join(a.cache, f"{grp}_h.npy"), mmap_mode="r"); Us = np.load(os.path.join(a.cache, f"{grp}_U.npy")).astype(np.float32)
    REC = []
    for w in range(Hs.shape[0]):
        h = np.asarray(Hs[w]).astype(np.float32); U = Us[w].reshape(-1, S, 2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hx = torch.from_numpy(h).to(dev)
            p_all = model.forward_seq(hx[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1).cpu().numpy()
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            m = h[t - 4:t + 1].mean(0)
            ys = {"data": h[t + 1], "model": p_all[t], "copy": h[t]}
            r = dict(win=w, n=int(sel.sum()))
            for kind in KINDS:
                Wk = warp(h[t], U[t + 1] if kind == "true" else U[t + 1] @ R90.T)
                X = np.stack([m[sel].ravel(), Wk[sel].ravel()], 1).astype(np.float64)
                r[f"XtX_{kind}"] = X.T @ X
                for nm, y in ys.items():
                    r[f"Xty_{kind}_{nm}"] = X.T @ y[sel].ravel().astype(np.float64)
            REC.append(r)
        if w % 100 == 0:
            print(f"{grp} {w}/{Hs.shape[0]}", flush=True)

    def bfit(recs, kind, nm):
        A = sum(r[f"XtX_{kind}"] for r in recs); bv = sum(r[f"Xty_{kind}_{nm}"] for r in recs)
        return np.linalg.solve(A, bv)[1]

    def boot(recs, fn, nb=300):
        wins = np.array([r["win"] for r in recs]); uw = np.unique(wins); ix = {u: np.where(wins == u)[0] for u in uw}
        v = [fn([recs[i] for i in np.concatenate([ix[u] for u in rng.choice(uw, len(uw))])]) for _ in range(nb)]
        return np.percentile(v, [2.5, 97.5]).tolist()

    e = {"n_windows": len(set(r["win"] for r in REC)), "n_steps": len(REC), "n_cells": int(sum(r["n"] for r in REC))}
    for nm in YS:
        for kind in KINDS:
            e[f"b_{kind}_{nm}"] = float(bfit(REC, kind, nm))
        e[f"net_{nm}"] = e[f"b_true_{nm}"] - e[f"b_rot_{nm}"]
        e[f"ci_net_{nm}"] = boot(REC, lambda rr, n_=nm: bfit(rr, "true", n_) - bfit(rr, "rot", n_))
    e["ratio_net_model_over_data"] = e["net_model"] / e["net_data"] if e["net_data"] != 0 else None
    res[grp] = e
    print(f"== {grp}: 창 {e['n_windows']} · 칸 {e['n_cells']}")
    for nm in YS:
        print(f"   {nm:5s} b_true {e[f'b_true_{nm}']:.3f} · b_rot {e[f'b_rot_{nm}']:.3f} · 위약 뺀 {e[f'net_{nm}']:.3f} {np.round(e[f'ci_net_{nm}'], 3).tolist()}")
    print(f"   위약 뺀 모델 / 데이터 = {e['ratio_net_model_over_data']:.2f}")
outp = a.out or os.path.join(REPO, "auto_research/results/transport_placebo.json")
json.dump(res, open(outp, "w"), indent=1)
print("저장:", outp)
