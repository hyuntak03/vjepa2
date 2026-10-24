#!/usr/bin/env python3
"""[P] 다중 가설 · L1 이어 학습 판정의 자연 영상판 ([T] 13:30 사전 등록의 "자연 영상: 출력마다 b vs 데이터 b") (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/hyp_natural.py

데이터: [E] 수집 캐시 (블록 8 개 창, K710 · SSv2 · HT100M) + [T] natural_transport_ctx5 의 칸 흐름 (transfer_beta.py 와 같은 입력).
출력 (TF 한 걸음, 문맥 = 블록 0..t, t = 4 · 5 · 6): 기준 AR · L1 이어 학습 · 다중 가설 (가설 k 마다).
  b = 움직이는 칸 (실제 흐름 ≥ 0.5 칸) 에서 출력 ≈ a·(같은 칸 5 블록 평균) + b·(실제 흐름으로 옮긴 h_t) 의 b. 데이터 b 는 정답에 같은 회귀.
  가설 사이 차이의 자리: |p_k − 가설 평균| 의 차원 평균을 움직이는 칸 / 정지 칸 (흐름 < 0.1 칸) 에서. 승리 가설 (정답 L1 최소, 창 단위) 몫."""
import json, os, sys
import numpy as np, torch, yaml

torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402
EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(REPO, "auto_research/results/natural_transport_ctx5")
FT = "/workspace/auto_research_cache/ar_ft"
dev = torch.device("cuda")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def build():
    D_ = os.path.join(REPO, "z_training/runs/natural_ar_vw_state")
    pc = yaml.safe_load(open(os.path.join(D_, "config.yaml")))["model"]["predictor"]
    return vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                                num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))


MODELS = {}
m = build(); m.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                                if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); MODELS["base"] = (m.to(dev).eval(), None)
for nm in ("ft_l1", "ft_wta3"):
    d = torch.load(os.path.join(FT, nm + ".pt"), map_location="cpu", weights_only=False)
    m = build(); m.load_state_dict(_clean_backbone_key(d["predictor"])); m = m.to(dev).eval()
    MODELS[nm] = (m, None if d.get("codes") is None else d["codes"].to(dev))
ZV = {"v": None}
for nm, (m, _) in MODELS.items():
    orig = m._ar_embed_obs

    def patched(x, orig=orig):
        e = orig(x)
        return e if ZV["v"] is None else e + ZV["v"][:, None, :].to(e.dtype)
    m._ar_embed_obs = patched

OUTS = [("base", None), ("ft_l1", None)] + [("ft_wta3", k) for k in range(3)]
res = {}
for src in ("k710", "ssv2", "ht100m"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    acc = {o: [np.zeros((2, 2)), np.zeros(2)] for o in [("data", None)] + OUTS}
    dif = {"mov": [], "sta": []}; wins = []
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = np.asarray(T_[w, 0]).astype(np.float32); Uw = U[w0 + w].reshape(8, S, 2)
            hx = torch.from_numpy(h).to(dev)
            P = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for nm, k in OUTS:
                    m, codes = MODELS[nm]
                    ZV["v"] = None if k is None else codes[k][None]
                    P[(nm, k)] = m.forward_seq(hx[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1).cpu().numpy()
                    ZV["v"] = None
            e3 = [np.abs(P[("ft_wta3", k)][3:7] - h[4:8]).mean() for k in range(3)]; wins.append(int(np.argmin(e3)))
            for t in (4, 5, 6):
                mag = np.linalg.norm(Uw[t + 1], axis=1); sel = mag >= 0.5
                if sel.sum() < 2:
                    continue
                mm = h[t - 4:t + 1].mean(0); Wt = warp(h[t], Uw[t + 1])
                X = np.stack([mm[sel].ravel(), Wt[sel].ravel()], 1).astype(np.float64)
                XtX = X.T @ X
                for o in [("data", None)] + OUTS:
                    y = h[t + 1] if o[0] == "data" else P[o][t]
                    acc[o][0] += XtX; acc[o][1] += X.T @ y[sel].ravel().astype(np.float64)
                hy = np.stack([P[("ft_wta3", k)][t] for k in range(3)]); dv = np.abs(hy - hy.mean(0)).mean((0, 2))   # (S,)
                st = np.linalg.norm(Uw[t + 1], axis=1) < 0.1
                dif["mov"].append(float(dv[sel].mean()))
                if st.any():
                    dif["sta"].append(float(dv[st].mean()))
        w0 += T_.shape[0]
    r = {f"{o[0]}{'' if o[1] is None else '_h' + str(o[1])}": float(np.linalg.solve(*acc[o])[1]) for o in acc}
    r["hyp_diff_moving"] = float(np.mean(dif["mov"])); r["hyp_diff_static"] = float(np.mean(dif["sta"]))
    r["win_share"] = (np.bincount(wins, minlength=3) / len(wins)).tolist()
    res[src] = r
    print(f"== {src} (창 {len(wins)}): b — 데이터 {r['data']:.3f} · 기준 {r['base']:.3f} · L1 이어 {r['ft_l1']:.3f} · 가설 0 / 1 / 2 {r['ft_wta3_h0']:.3f} / {r['ft_wta3_h1']:.3f} / {r['ft_wta3_h2']:.3f}"
          f" | 가설 사이 차이 (차원 평균 |p_k − 평균|) 움직이는 칸 {r['hyp_diff_moving']:.4f} · 정지 칸 {r['hyp_diff_static']:.4f} | 승리 몫 {np.round(r['win_share'], 2).tolist()}", flush=True)
json.dump(res, open(os.path.join(REPO, "auto_research/results/hyp_natural.json"), "w"), indent=1)
