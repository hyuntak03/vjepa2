#!/usr/bin/env python3
"""굴리는 동안 "속도만" 다시 넣어 주면 물체가 덜 사라지고 옮겨지나 — 인과 개입 (2026-10-27, 사용자 요청).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/velocity_intervention.py --run natural_ar_vw_state_fps10_c16k8 e6.pt --alpha 1 2 4

전제 측정 (velocity_where_predictor.py): 정답 블록 입력 · TF 은닉에는 물체 속도가 있다 (물체 3×3 R² 0.5), 굴린 걸음 은닉은 전 층 0.000.
개입: 상태 되먹임 rollout (rollout_kv 와 같은 계산) 의 매 굴리기 걸음 입력 u (블록 256 칸 × 384) 에
   u ← u + α · Π_v(e_last)          Π_v = 속도 부분공간 (2 차원) 으로의 투영, e_last = 마지막 문맥 블록의 입력 임베딩 (관측, 미래 정보 아님)
   속도 부분공간 = 관측 입력 임베딩 (물체 3×3 · 전체 칸) → 물체 속도 ridge 의 두 방향 (맞춤 clip 에서만 정함, 확인 clip 에 개입).
   물체 외형은 주지 않는다 (2 차원 성분만). 모든 칸에 건다 (속도가 물체 주변 + 화면 전체에 퍼져 있음).
위약: 같은 크기로 속도 부분공간을 90° 돌린 방향 (vx ↔ vy, 부호) · 무작위 2 차원 부분공간.
잰다 (걸음별, phys3d val 확인 clip, 블록 0.2 초): ① 정답 칸 물체다움 o = d_bg/(d_obj+d_bg) (vanish_median.py 와 같은 정의) ② 옛 자리 물체다움
  ③ 운반 이득 (transport_gain.py 정의, 파라미터 없는 템플릿 cos 읽기) ④ 굴린 은닉의 속도 R² (개입 뒤 속도가 실제로 읽히나 — 조작 확인)."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 2.0, 4.0])
ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.08); torch.manual_seed(0)
G, S, C, K, ST = 16, 256, 8, 8, 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0]); pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                            num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred.load_state_dict(_clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]), strict=True)
pred = pred.to(dev).eval()
assert pred.ar_feedback == "state"

paths = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4") if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
paths = [paths[i] for i in np.random.default_rng(0).permutation(len(paths))[: a.n]]
half = len(paths) // 2
cell = lambda q: int(np.clip(q[1] // 16, 0, G - 1)) * G + int(np.clip(q[0] // 16, 0, G - 1))
def nb3(c):
    r, q = divmod(c, G); return [rr * G + qq for rr in range(max(0, r - 1), min(G, r + 2)) for qq in range(max(0, q - 1), min(G, q + 2))]

def load(p):
    st = np.load(p[:-4] + "_states.npz"); vr = decord.VideoReader(p, num_threads=2)
    idx = [ST * k for k in range(2 * (C + K))]
    x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    sc = 256 / x.shape[-1]
    if x.shape[-1] != 256: x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
    return st, idx, sc, h

# ---- 1. 속도 부분공간: 맞춤 clip 의 관측 입력 임베딩 (물체 3×3 평균) → 속도 ------------------------------------
X, Y = [], []
for p in paths[:half]:
    st, idx, sc, h = load(p)
    with torch.no_grad():
        e = pred._ar_embed_obs(h)                                       # (nb, 256, 384) — 관측 입력과 같은 임베딩
    for b in range(C + K):
        f1 = idx[2 * b + 1]
        if st["vis_frac"][f1, 0] > 0.3 and np.isfinite(st["pos"][f1, 0]).all():
            X.append(e[b, nb3(cell(st["pos"][f1, 0] * sc))].mean(0).cpu().numpy()); Y.append(st["vel"][f1, 0] * ST)
X, Y = np.array(X), np.array(Y)
mu = X.mean(0); Xc = X - mu
lam = 10.0 * np.trace(Xc.T @ Xc) / Xc.shape[1]
W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (Y - Y.mean(0)))     # (384, 2)
Q, _ = np.linalg.qr(W)                                                                   # 속도 부분공간 정규 직교 기저 (384, 2)
Qt = torch.tensor(Q, device=dev, dtype=torch.float32)
Qrand = torch.linalg.qr(torch.randn(384, 2, device=dev))[0]
print(f"속도 부분공간: 맞춤 블록 {len(X)} · 맞춤 R² {1 - ((Y - Y.mean(0) - Xc @ W) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum():.3f}", flush=True)

def proj(u, B, rot=False):                       # u (S, 384) → 부분공간 성분 (S, 384)
    c = u @ B                                    # (S, 2)
    if rot: c = torch.stack([-c[:, 1], c[:, 0]], 1)
    return c @ B.T

def rollout(h, mode, alpha):
    """rollout_kv 와 같은 계산 + 매 굴리기 걸음 입력에 α·Π(e_last) 를 더한다. return 걸음별 예측 (K, S, 1280), 굴린 은닉 (12 층 마지막, K−1)."""
    caches, p, st_ = None, None, None
    for b in range(C):
        p, caches, st_ = pred._ar_step_kv(h[b:b + 1], torch.arange(b * S, (b + 1) * S, device=dev)[None], caches)
    e_last = pred._ar_embed_obs(h[C - 1:C])[0]
    add = {"none": None, "vel": proj(e_last, Qt), "rot": proj(e_last, Qt, rot=True), "rand": proj(e_last, Qrand)}[mode]
    preds, hids = [p[0]], []
    pos = torch.arange((C - 1) * S, C * S, device=dev)[None]
    for _ in range(1, K):
        pos = pos + S
        u = st_ + pred.type_embed[1]
        if add is not None: u = u + alpha * add[None]
        p, caches, st_ = pred._ar_step_kv(u, pos, caches, embedded=True)
        preds.append(p[0]); hids.append(st_[0])
    return torch.stack(preds), torch.stack(hids)

# ---- 2. 확인 clip 에 개입 --------------------------------------------------------------------------------------
modes = [("none", 0.0)] + [(m, al) for al in a.alpha for m in ("vel", "rot", "rand")]
R = {m: [] for m in modes}
for p in paths[half:]:
    st, idx, sc, h = load(p)
    fl = idx[2 * C - 1]
    if not (st["vis_frac"][fl, 0] > 0.3 and np.isfinite(st["pos"][fl, 0]).all()): continue
    old = cell(st["pos"][fl, 0] * sc); hc = h[C - 1]
    tmpl = F.normalize(hc[nb3(old)].mean(0), dim=0)
    for m in modes:
        with torch.no_grad():
            pr, hid = rollout(h, *m)
        for k in range(K):
            b = C + k; f1 = idx[2 * b + 1]; pos_ = st["pos"][f1, 0] * sc
            if not (st["vis_frac"][f1, 0] > 0.3 and np.isfinite(pos_).all()): continue
            tr = cell(pos_); d_true = np.linalg.norm(pos_ - st["pos"][fl, 0] * sc)
            if tr == old or d_true < 16: continue
            d = lambda u, v: float((u - v).abs().mean())
            o_t = d(pr[k, tr], hc[tr]) / (d(pr[k, tr], h[b, tr]) + d(pr[k, tr], hc[tr]) + 1e-9)
            o_o = d(pr[k, old], h[b, old]) / (d(pr[k, old], hc[old]) + d(pr[k, old], h[b, old]) + 1e-9)
            cs = (F.normalize(pr[k], dim=1) @ tmpl).cpu().numpy()                     # 템플릿 cos → 최대 칸 주변 가중 중심
            j = int(cs.argmax()); nn_ = nb3(j); w = np.clip(cs[nn_], 0, None) + 1e-9
            cxy = np.array([(np.array(nn_) % G * 16 + 8) @ w / w.sum(), (np.array(nn_) // G * 16 + 8) @ w / w.sum()])
            dp, dt = cxy - (np.array([old % G, old // G]) * 16 + 8), pos_ - st["pos"][fl, 0] * sc
            gain = float(dp @ dt / (dt @ dt))
            vel_r = hid[k - 1][nb3(tr)].mean(0).cpu().numpy() if k > 0 else None
            R[m].append((k, o_t, o_o, gain, vel_r, st["vel"][f1, 0] * ST))
print(f"\n확인 clip {len(paths) - half} · 1 칸 넘게 움직인 블록. 물체다움 1 = 물체 쪽, 0 = 배경 쪽 (중앙값) · 운반 이득 (중앙값, 1 = 정확히 옮김)")
print(f"{'개입':12s} {'정답칸 o':>9s} {'옛자리 o':>9s} {'운반 이득':>9s} {'걸음 1 / 4 / 8 정답칸 o':>24s}   n")
for m, rows in R.items():
    if not rows: continue
    A_ = np.array([(r[0], r[1], r[2], r[3]) for r in rows])
    st_ = " / ".join(f"{np.median(A_[A_[:, 0] == k, 1]):.3f}" if (A_[:, 0] == k).sum() > 5 else "—" for k in (0, 3, 7))
    print(f"{m[0] + (f' α{m[1]:g}' if m[1] else ''):12s} {np.median(A_[:, 1]):9.3f} {np.median(A_[:, 2]):9.3f} {np.median(A_[:, 3]):9.3f} {st_:>24s}   {len(A_)}")
