#!/usr/bin/env python3
"""[P] 메커니즘: 블록 AR 이 문맥 안정도 s 를 계산할 경로가 있나 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/stab_mechanism.py [--ft ft_l1 ...]

s 정의 ([T] context_reliability.py 와 같다): 움직이는 칸 c (실제 흐름 |U[t+1](c)| ≥ 0.5 칸, t = 4 · 5 · 6), 그 내용의 블록 t 자리 j = c + U[t+1](c),
  블록 t−1 자리 p = j + U[t](j). s = cos(h_t(j), h_{t−1}(p)). 데이터: [E] 캐시 (블록 8 개 창) + natural_transport_ctx5 흐름. 자연 = K710 + HT100M, 3D 합성.
(b) 주 근거 — 선형 판독: TF 위치 t (블록 t 를 넣고 블록 t+1 을 내는 자리) 의 층별 은닉 (0 = 입력 임베딩 · 1..12 층 뒤, 384 차원) 을
    query 둘 (새 자리 c · 내용 자리 j) 에서 모아 ridge 로 s 를 읽는다 (창 짝수 = 맞춤 · 홀수 = 평가, λ 는 맞춤 반 안에서 5 겹 교차검증).
    기준선: ① 입력 토큰 h_t(j) 하나 (1280, 블록 t−1 을 못 봄 = "비교 없이 읽히는 몫") ② 상한 h_t(j) ⊙ h_{t−1}(p) (내적의 성분 = 비교 그 자체).
    은닉의 R² 가 ① 보다 크면 모델이 블록 사이 비교를 (일부) 계산한다.
(a) 보조 — attention: 새 자리 query (블록 t, 칸 c) 가 블록 t−1 의 내용 자리 p (3×3) · 블록 t 의 내용 자리 j (3×3) · 블록 t−1 의 같은 칸 c 에 준 질량 (층별, 헤드 평균 / 최대)."""
import argparse, json, os, sys
import numpy as np, torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--ft", nargs="*", default=["ft_l1"])
ap.add_argument("--per", type=int, default=40, help="(창, t) 마다 뽑는 움직이는 칸 수")
ap.add_argument("--only_ft", type=int, default=0, help="1 = 기준 · 합성 전용은 건너뛰고 --ft 만 (결과 json 에 덧붙인다)")
a = ap.parse_args()
torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                        # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                  # noqa: E402
import src.models.utils.modules as MOD                                         # noqa: E402
dev = torch.device("cuda")
EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(REPO, "auto_research/results/natural_transport_ctx5")
G, S, NL = 16, 256, 12
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
LAYERS = [0, 2, 4, 6, 8, 10, 12]
pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
build = lambda: vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                                     num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
SRC = {"base": os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"),
       "phys3d_only_e10": os.path.join(REPO, "z_training/runs/natural_ar_vw_state_phys3d_only/e10.pt")}
SRC.update({n: f"/workspace/auto_research_cache/ar_ft/{n}.pt" for n in a.ft})
if a.only_ft:
    SRC = {n: f"/workspace/auto_research_cache/ar_ft/{n}.pt" for n in a.ft}

REC_A = {"on": False, "w": []}
_sdpa = torch.nn.functional.scaled_dot_product_attention


def sdpa(q, k, v, *args, **kw):
    if REC_A["on"]:
        REC_A["w"].append(torch.softmax((q.float() @ k.float().transpose(-1, -2)) / q.size(-1) ** 0.5, dim=-1)[0])   # (h, nq, M)
    return _sdpa(q, k, v, *args, **kw)


MOD.F.scaled_dot_product_attention = sdpa


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def nb3(j):
    y, x = divmod(int(j), G)
    return [yy * G + xx for yy in range(max(0, y - 1), min(G, y + 2)) for xx in range(max(0, x - 1), min(G, x + 2))]


def collect(model, srcs):
    rng = np.random.RandomState(0)
    out = {k: [] for k in ("s", "win", "hin", "prod", "src")}
    for L_ in LAYERS:
        out[f"hc{L_}"] = []; out[f"hj{L_}"] = []
    att = []
    for src in srcs:
        U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                h = torch.from_numpy(np.asarray(T_[w, 0]).astype(np.float32)).to(dev)
                caches = None
                for b in range(7):                                                   # 블록 0..6 을 차례로 (block-causal KV)
                    rec = b in (4, 5, 6)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        x = model._ar_embed_obs(h[b:b + 1].reshape(1, S, -1))
                        hs = [x[0].float()]
                        new = []
                        REC_A.update(on=rec, w=[])
                        for i, blk in enumerate(model.predictor_blocks):
                            x, kv = blk.forward_kv(x, torch.arange(b * S, (b + 1) * S, device=dev).unsqueeze(0), G, G, None if caches is None else caches[i])
                            new.append(kv); hs.append(x[0].float())
                        REC_A["on"] = False
                        caches = new
                    if not rec:
                        continue
                    t = b
                    u1 = U[w0 + w, t + 1].reshape(S, 2); u0 = U[w0 + w, t].reshape(S, 2)
                    mov = np.linalg.norm(u1, axis=1) >= 0.5
                    js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
                    sel = np.where(mov & ok1 & ok0)[0]
                    if sel.size == 0:
                        continue
                    if sel.size > a.per:
                        sel = rng.choice(sel, a.per, replace=False)
                    c = torch.from_numpy(sel).to(dev); j = torch.from_numpy(js[sel]).to(dev); p = torch.from_numpy(jp[sel]).to(dev)
                    s = torch.nn.functional.cosine_similarity(h[t][j], h[t - 1][p], dim=-1)
                    out["s"].append(s.cpu().numpy()); out["win"].append(np.full(len(sel), w0 + w)); out["src"].append(np.full(len(sel), src))
                    out["hin"].append(h[t][j].half().cpu().numpy()); out["prod"].append((h[t][j] * h[t - 1][p]).half().cpu().numpy())
                    for L_ in LAYERS:
                        out[f"hc{L_}"].append(hs[L_][c].half().cpu().numpy()); out[f"hj{L_}"].append(hs[L_][j].half().cpu().numpy())
                    # attention: 새 자리 query (블록 t, 칸 c) → key 블록 t−1 (지연 1) 의 p 3×3 · 블록 t 의 j 3×3 · 블록 t−1 의 같은 칸 c
                    Wl = torch.stack(REC_A["w"])                                                     # (12, h, 256, (t+1)*256)
                    rows = []
                    for ci, jj, pp in zip(sel, js[sel], jp[sel]):
                        kp = [(t - 1) * S + q for q in nb3(pp)]; kj = [t * S + q for q in nb3(jj)]; kc = [(t - 1) * S + int(ci)]
                        m_p = Wl[:, :, ci, kp].sum(-1); m_j = Wl[:, :, ci, kj].sum(-1); m_c = Wl[:, :, ci, kc].sum(-1)   # (12, h)
                        rows.append(torch.stack([m_p, m_j, m_c]).cpu().numpy())
                    att.append(np.stack(rows))                                                      # (n, 3, 12, h)
            w0 += T_.shape[0]
        print(f"  {src} 끝 (표본 {sum(len(x) for x in out['s'])})", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(att)


def ridge_r2(X, y, win):
    tr = (win % 2 == 0); te = ~tr
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd; ym = y[tr].mean()
    Xtr, ytr = Xs[tr], y[tr] - ym
    # λ: 맞춤 반 안에서 창 단위 5 겹
    wt = win[tr]; folds = (wt // 2) % 5
    best, bl = -1e9, None
    for lam in (1e0, 1e1, 1e2, 1e3, 1e4):
        sc = []
        for f in range(5):
            a_, b_ = folds != f, folds == f
            A = Xtr[a_].T @ Xtr[a_] + lam * np.eye(X.shape[1]); wv = np.linalg.solve(A, Xtr[a_].T @ ytr[a_])
            r = ytr[b_] - Xtr[b_] @ wv; sc.append(1 - (r ** 2).sum() / ((ytr[b_] - ytr[b_].mean()) ** 2).sum())
        if np.mean(sc) > best:
            best, bl = np.mean(sc), lam
    A = Xtr.T @ Xtr + bl * np.eye(X.shape[1]); wv = np.linalg.solve(A, Xtr.T @ ytr)
    r = (y[te] - ym) - Xs[te] @ wv
    return float(1 - (r ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()), bl


OUTP = os.path.join(REPO, "auto_research/results/stab_mechanism.json")
res = json.load(open(OUTP)) if (a.only_ft and os.path.exists(OUTP)) else {}
for name, path in SRC.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)["predictor"]
    m = build(); m.load_state_dict({k: v for k, v in _clean_backbone_key(sd).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); m = m.to(dev).eval()
    res[name] = {}
    for grp, srcs in (("natural", ["k710", "ht100m"]), ("phys3d", ["phys3d"])):
        D, A = collect(m, srcs)
        s, win = D["s"].astype(np.float64), D["win"]
        e = {"n": int(len(s)), "s_mean": float(s.mean()), "s_sd": float(s.std())}
        e["R2_input_hj"], _ = ridge_r2(D["hin"].astype(np.float64), s, win)
        e["R2_upper_prod"], _ = ridge_r2(D["prod"].astype(np.float64), s, win)
        for L_ in LAYERS:
            e[f"R2_hidden_c_L{L_}"], _ = ridge_r2(D[f"hc{L_}"].astype(np.float64), s, win)
            e[f"R2_hidden_j_L{L_}"], _ = ridge_r2(D[f"hj{L_}"].astype(np.float64), s, win)
        am = A.mean(0)                                                                   # (3, 12, h)
        e["att_prev_content_mean"] = am[0].mean(-1).tolist(); e["att_prev_content_headmax"] = am[0].max(-1).tolist()
        e["att_cur_content_mean"] = am[1].mean(-1).tolist(); e["att_prev_samecell_mean"] = am[2].mean(-1).tolist()
        res[name][grp] = e
        print(f"\n== {name} · {grp} (표본 {e['n']}, s 평균 {e['s_mean']:.3f} · 표준편차 {e['s_sd']:.3f})")
        print(f"   R² 입력 토큰 h_t(j) 하나 {e['R2_input_hj']:.3f} · 상한 h_t(j)⊙h_(t−1)(p) {e['R2_upper_prod']:.3f}")
        print("   R² 은닉 (새 자리 c)   " + " · ".join(f"L{L_} {e[f'R2_hidden_c_L{L_}']:.3f}" for L_ in LAYERS))
        print("   R² 은닉 (내용 자리 j) " + " · ".join(f"L{L_} {e[f'R2_hidden_j_L{L_}']:.3f}" for L_ in LAYERS))
        print("   attention 새 자리 → 블록 t−1 내용 자리 3×3 (헤드 평균 / 최대) " + " ".join(f"{x:.3f}/{y:.3f}" for x, y in zip(e["att_prev_content_mean"], e["att_prev_content_headmax"])))
        print("   attention 새 자리 → 블록 t 내용 자리 3×3 (헤드 평균) " + " ".join(f"{x:.3f}" for x in e["att_cur_content_mean"]) + " | 블록 t−1 같은 칸 " + " ".join(f"{x:.3f}" for x in e["att_prev_samecell_mean"]))
        json.dump(res, open(OUTP, "w"), indent=1)
