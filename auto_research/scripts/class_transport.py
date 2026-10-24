#!/usr/bin/env python3
"""[P] H2 (조건) 판정 — 수동 물리 vs 행위자 부류에서 데이터 최적 운반 신뢰 b 와 기준 AR 의 b (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/class_transport.py [--model 1]

입력: cache_ssv2_classes.py 캐시 ({passive,actor}_h.npy · _U.npy). 정의는 [T] transfer_beta.py 와 같다:
  블록 8 개 창, t = 4 · 5 · 6, 정답 y = h_{t+1}, m = 같은 칸 5 블록 평균 (h_{t−4..t}), 움직이는 칸 = |U[t+1](c)| ≥ 0.5 칸.
  y ≈ a·m + b·W 최소제곱 (칸 · 차원 풀링). W 세 가지:
    참 (true)          = h_t 를 실제 흐름 U[t+1] 로 옮긴 것 (가까운 칸, 화면 밖은 제자리) — 정답 블록을 봐야 아는 변위
    지속 오일러 (pers) = 같은 칸의 직전 흐름 U[t] 로 옮긴 것
    지속 라그랑주 (lag) = 블록 t 의 칸 s 내용을 s − U[t](s) 로 보냄 (등속 외삽) — 문맥만으로 할 수 있는 운반
  --model 1: 같은 회귀를 기준 블록 AR 의 TF 한 걸음 예측 p_{t+1} (문맥 = 블록 0..t) 에도 → 모델 b. 데이터 b 와의 비.
  카메라 정지 창 (블록 t · t+1 흐름 칸 중앙값 크기 < 0.3 칸) 만 따로도 낸다. 구간 = 창 단위 bootstrap 95 %."""
import argparse, json, os, sys
import numpy as np, torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/ssv2_classes")
ap.add_argument("--groups", nargs="+", default=["passive", "actor"])
ap.add_argument("--model", type=int, default=1)
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
KINDS = ("true", "pers", "lag")


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def splat(ht, u0):
    d = np.floor(P0 - u0).astype(int)
    ok = (d >= 0).all(1) & (d < G).all(1)
    j = d[:, 1] * G + d[:, 0]
    acc = np.zeros_like(ht); n = np.zeros(S)
    np.add.at(acc, j[ok], ht[ok]); np.add.at(n, j[ok], 1)
    out = ht.copy(); m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


model = None
if a.model:
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

res = {}
rng = np.random.RandomState(0)
for grp in a.groups:
    Hs = np.load(os.path.join(a.cache, f"{grp}_h.npy"), mmap_mode="r"); Us = np.load(os.path.join(a.cache, f"{grp}_U.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(a.cache, f"{grp}_meta.json")))
    REC = []                                                    # 창 × t 마다 충분통계 (kind 별, y = data / model)
    for w in range(Hs.shape[0]):
        h = np.asarray(Hs[w]).astype(np.float32); U = Us[w].reshape(-1, S, 2)
        if model is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hx = torch.from_numpy(h).to(dev)
                p_all = model.forward_seq(hx[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1).cpu().numpy()
        for t in (4, 5, 6):
            mag = np.linalg.norm(U[t + 1], axis=1)
            sel = mag >= 0.5
            if sel.sum() < 2:
                continue
            cam = max(np.median(np.linalg.norm(U[t], axis=1)), np.median(mag)) < 0.3
            m = h[t - 4:t + 1].mean(0)
            ys = {"data": h[t + 1]}
            if model is not None:
                ys["model"] = p_all[t]
            r = dict(win=w, t=t, cam_still=bool(cam), n=int(sel.sum()), label=meta[w]["label"])
            for kind in KINDS:
                Wk = warp(h[t], U[t + 1]) if kind == "true" else (warp(h[t], U[t]) if kind == "pers" else splat(h[t], U[t]))
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

    out = {}
    for sub, recs in (("all", REC), ("cam_still", [r for r in REC if r["cam_still"]])):
        if len(recs) < 20:
            continue
        e = {"n_steps": len(recs), "n_windows": len(set(r["win"] for r in recs)), "n_cells": int(sum(r["n"] for r in recs))}
        for kind in KINDS:
            e[f"b_data_{kind}"] = float(bfit(recs, kind, "data")); e[f"ci_data_{kind}"] = boot(recs, lambda rr, k=kind: bfit(rr, k, "data"))
            if model is not None:
                e[f"b_model_{kind}"] = float(bfit(recs, kind, "model")); e[f"ci_model_{kind}"] = boot(recs, lambda rr, k=kind: bfit(rr, k, "model"))
        out[sub] = e
        line = f"== {grp} [{sub}] 창 {e['n_windows']} · 걸음 {e['n_steps']} · 움직이는 칸 {e['n_cells']}"
        print(line)
        for kind in KINDS:
            s_ = f"   b_{kind:4s} 데이터 {e[f'b_data_{kind}']:.3f} [{e[f'ci_data_{kind}'][0]:.3f}, {e[f'ci_data_{kind}'][1]:.3f}]"
            if model is not None:
                s_ += f" · 기준 AR {e[f'b_model_{kind}']:.3f} [{e[f'ci_model_{kind}'][0]:.3f}, {e[f'ci_model_{kind}'][1]:.3f}] · 비 (모델/데이터) {e[f'b_model_{kind}'] / e[f'b_data_{kind}']:.2f}"
            print(s_)
        if model is not None:
            print(f"   모델 b_true / 데이터 b_lag (모델이 문맥 최적만큼 옮기나) {e['b_model_true'] / e['b_data_lag']:.2f}")
    res[grp] = out
outp = a.out or os.path.join(REPO, "auto_research/results/class_transport.json")
json.dump(res, open(outp, "w"), indent=1)
print("저장:", outp)
