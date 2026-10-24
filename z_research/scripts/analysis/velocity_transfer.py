#!/usr/bin/env python3
"""속도 읽기 head 이식 — 문맥 토큰에서 배운 속도 head 가 예측 토큰에서도 통하나 (2026-10-23, 사용자 질문:
"context 에서 쓰인 속도 표현이랑 prediction 에서 쓰인 게 같은 표현인지").

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/velocity_transfer.py \
      --run natural_ar_vw_state latest.pt --run natural_ar_vw_state_phys3d_event latest.pt [--limit 0]

데이터: phys3d_single_v1 val (학습에 안 쓴 clip), transport_gain.py 와 같은 창 (r0 ∈ {0, 2}, 간격 3, 문맥 블록 0..7 → 미래 8..15).
라벨: 블록마다 화면 속도 (vx, vy) = 그 블록 두 프레임의 states.vel 평균 (px/프레임). 물체가 보이는 (vis_frac > 0.3) 블록만.
표현 (블록마다 1280 차원 하나로 모음 — 화면 전체 256 칸 평균 · 정답 물체 자리 3×3 평균 두 가지):
  ctx   = 문맥 블록 1..7 의 LN(target_encoder(블록))          (AR 의 입력)
  fut   = 미래 블록 8..15 의 LN(target_encoder(블록))         (대조군: 시점만 다름)
  ro    = 모델의 굴린 예측 미래 블록 8..15
  tf    = 모델의 TF 한 걸음 예측 (정답 과거 다 보고 다음 블록)
head = ridge 선형 회귀 (입력 표준화는 학습 표현의 평균 · 표준편차로, alpha 는 학습 clip 안 5-fold 로 고름). clip 단위 반반 split (seed 0).
보고: R² (vx · vy 평균). 학습 → 시험 = ctx→ctx · ctx→fut · ctx→ro · ctx→tf · ro→ro · tf→tf.
⚠️ 3×3 은 **정답** 물체 자리 — 예측이 물체를 다른 데 두면 그 칸에 물체가 없을 수 있다 (그래서 화면 전체 평균을 같이 본다)."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K, ST = 256, 16, 8, 8, 3
T = 2 * (C + K)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()


def load_pred(run, ck):
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    assert pc.get("kind") == "ar", f"{run}: kind=ar 만"
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k_) is not None:
            kw[k_] = pc[k_]
    m = vit_prefix_predictor(**kw)
    w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
         if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m.to(dev).eval()


models = [(r, load_pred(r, c)) for r, c in a.run]
cells3 = lambda p: (lambda cx, cy: [y * G + x for y in range(max(0, cy - 1), min(G, cy + 2)) for x in range(max(0, cx - 1), min(G, cx + 2))])(
    int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1)))
paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
if a.limit:
    paths = paths[: a.limit]
# D[rep][pool] = list of (feature 1280, label 2, clip, moving)
D = {}
def add(rep, blk, pos, v, ci):
    if not (np.isfinite(pos).all() and np.isfinite(v).all()):
        return
    for pool, t in (("all", blk.mean(0)), ("obj3", blk[cells3(pos)].mean(0))):
        D.setdefault((rep, pool), []).append((t.float().cpu().numpy(), v, ci, float(np.linalg.norm(v) > 1.0)))

for ci, path in enumerate(paths):
    st = np.load(path[:-4] + "_states.npz")
    vr = decord.VideoReader(path, num_threads=2)
    for r0 in (0, 2):
        idx = [r0 + ST * k for k in range(T)]
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        xn = (x - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, xn.to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
        lab = []
        for b in range(C + K):
            f0, f1 = idx[2 * b], idx[2 * b + 1]
            vis = st["vis_frac"][f0, 0] > 0.3 and st["vis_frac"][f1, 0] > 0.3
            pos = st["pos"][f1, 0]; v = (st["vel"][f0, 0] + st["vel"][f1, 0]) / 2
            lab.append((vis, pos, v))
        for b in range(1, C):
            if lab[b][0]:
                add("ctx", hb[b], lab[b][1], lab[b][2], ci)
        for j in range(K):
            if lab[C + j][0]:
                add("fut", hb[C + j], lab[C + j][1], lab[C + j][2], ci)
        idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
        for name, m in models:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_ro = m.rollout_kv(hb[:C].reshape(1, C * S, -1), idx_ctx, K).float()[0].view(K, S, -1)
                p_tf = m.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)).float()[0]
            p_tf = p_tf.view(C + K - 1, S, -1)[C - 1:]
            for j in range(K):
                if lab[C + j][0]:
                    add(f"ro:{name}", p_ro[j], lab[C + j][1], lab[C + j][2], ci)
                    add(f"tf:{name}", p_tf[j], lab[C + j][1], lab[C + j][2], ci)
    if ci % 50 == 0:
        print(f"{ci}/{len(paths)} clip", flush=True)

clips = np.unique([r[2] for r in D[("ctx", "all")]])
rng = np.random.RandomState(0); perm = rng.permutation(clips)
train_c = set(perm[: len(perm) // 2])


def arr(rep, pool, split):
    rows = [r for r in D[(rep, pool)] if (r[2] in train_c) == (split == "train")]
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))


def fit(X, Y, cl):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = Y.mean(0)
    best = None
    folds = np.array_split(np.random.RandomState(1).permutation(np.unique(cl)), 5)
    for al in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        sc = []
        for f in folds:
            te = np.isin(cl, f); tr = ~te
            W = np.linalg.solve(Xs[tr].T @ Xs[tr] + al * np.eye(Xs.shape[1]), Xs[tr].T @ (Y[tr] - Y[tr].mean(0)))
            sc.append(r2(Xs[te] @ W + Y[tr].mean(0), Y[te]))
        if best is None or np.mean(sc) > best[0]:
            best = (np.mean(sc), al)
    W = np.linalg.solve(Xs.T @ Xs + best[1] * np.eye(Xs.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym, best[1]


def r2(P, Y):
    return float(np.mean([1 - ((P[:, k] - Y[:, k]) ** 2).sum() / ((Y[:, k] - Y[:, k].mean()) ** 2).sum() for k in range(Y.shape[1])]))


print(f"\nclip {len(clips)} (학습 {len(train_c)} · 시험 {len(clips) - len(train_c)}). R² (vx · vy 평균) — 시험 clip. 괄호 = 움직이는 블록만 (|v| > 1 px/프레임)")
for pool in ("all", "obj3"):
    print(f"\n[{'화면 전체 평균' if pool == 'all' else '정답 물체 자리 3×3 평균'}]")
    X, Y, cl, _ = arr("ctx", pool, "train"); h_ctx, al = fit(X, Y, cl)
    def show(nm, h, rep):
        Xt, Yt, _, mv = arr(rep, pool, "test"); P_ = h(Xt)
        print(f"  {nm:46s} {r2(P_, Yt):6.3f} ({r2(P_[mv > 0], Yt[mv > 0]):6.3f})   n {len(Yt)}")
    show(f"문맥 head (alpha {al:g}) → 문맥", h_ctx, "ctx")
    show("문맥 head → 실제 미래 (대조군)", h_ctx, "fut")
    for name, _ in models:
        show(f"문맥 head → {name} 굴린 예측", h_ctx, f"ro:{name}")
        show(f"문맥 head → {name} TF 한 걸음", h_ctx, f"tf:{name}")
        for mode in ("ro", "tf"):
            X2, Y2, cl2, _ = arr(f"{mode}:{name}", pool, "train"); h2, al2 = fit(X2, Y2, cl2)
            show(f"{'굴린 예측' if mode == 'ro' else 'TF'} head (자기, alpha {al2:g}) → {name} {'굴린 예측' if mode == 'ro' else 'TF'}", h2, f"{mode}:{name}")
