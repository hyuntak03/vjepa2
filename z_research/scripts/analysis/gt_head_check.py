#!/usr/bin/env python3
"""정답 토큰 (문맥 + 미래, LN(target_encoder(블록))) 으로 학습한 attention 상태 head 가 (1) 처음 보는 clip 의 정답 토큰,
(2) AR 예측 토큰에서 위치 · 속도를 얼마나 읽나 (2026-10-24, 사용자: "head 를 채점 타깃 · predictor 입력으로 학습").
학습 없이 head 만 학습한다 — predictor 학습 전 확인 (head 가 믿을 만한 목표인가 · 지금 AR 예측이 얼마나 어긋나나).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/gt_head_check.py \
      --run natural_ar_vw_state latest.pt --run natural_ar_vw_state_phys3d_event latest.pt [--ntrain 400]

head: 블록마다 x = LN(토큰), query 1 개 attention (SingleStateHead 와 같은 식) → z (1280) → 위치 Linear(2) · 속도 Linear(2).
학습 데이터: phys3d_single_v1 **train** clip ntrain 개의 문맥 + 미래 블록 중 물체가 보이는 블록 (vis_frac > 0.3). 손실 L1 (위치 /256, 속도 /8).
평가: phys3d_single_v1 **val** (transport_gain.py 와 같은 창). 블록 종류: 보임 / 가림 (hidden) / 다시 나타남 (그 창에서 가림 다음 첫 보임).
보고: 위치 오차 px 중앙값 · 움직이는 블록 (|v| > 1) 방향 cos 중앙값 · 정답 칸 attention (균등 = 1)."""
import argparse, glob, os, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1")
ap.add_argument("--run", nargs=2, action="append", default=[], metavar=("RUN", "CKPT"))
ap.add_argument("--ntrain", type=int, default=400)
ap.add_argument("--epochs", type=int, default=30)
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
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    m = vit_prefix_predictor(**kw)
    m.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
    return m.to(dev).eval()


def clips(split, n=0):
    ps = sorted(p for p in glob.glob(os.path.join(a.root, split, "clips", "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
    if n:
        random.Random(0).shuffle(ps); ps = ps[:n]
    return ps


def windows(path):
    st = np.load(path[:-4] + "_states.npz")
    vr = decord.VideoReader(path, num_threads=2)
    for r0 in (0, 2):
        idx = [r0 + ST * k for k in range(T)]
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
        lab = []
        for b in range(C + K):
            f0, f1 = idx[2 * b], idx[2 * b + 1]
            vis = bool(st["vis_frac"][f0, 0] > 0.3 and st["vis_frac"][f1, 0] > 0.3)
            hid = bool(st["hidden"][f1, 0] > 0.5) and bool(st["present"][f1, 0] > 0.5) and bool(st["inframe_frac"][f1, 0] > 0.5)
            lab.append((vis, hid, st["pos"][f1, 0].astype(np.float32), ((st["vel"][f0, 0] + st["vel"][f1, 0]) / 2).astype(np.float32)))
        yield hb, lab


class Head(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim)); self.log_tau = nn.Parameter(torch.zeros(()))
        self.pos = nn.Linear(dim, 2); self.vel = nn.Linear(dim, 2)

    def forward(self, x):                                   # (n, S, D) → pos px (n,2), vel (n,2), attn (n,S)
        x = F.layer_norm(x.float(), (x.size(-1),))
        w = torch.softmax((x @ self.query[0]) * (x.size(-1) ** -0.5) * self.log_tau.exp(), -1)
        z = (w[..., None] * x).sum(1)
        return self.pos(z) * 256, self.vel(z) * 8, w


# ---- 1. head 학습: train clip 의 정답 토큰 (문맥 + 미래, 보이는 블록) ------------------------------------
X, P, V = [], [], []
tr = clips("train", a.ntrain)
for i, p in enumerate(tr):
    for hb, lab in windows(p):
        for b, (vis, hid, pos, vel) in enumerate(lab):
            if vis and np.isfinite(pos).all() and np.isfinite(vel).all():
                X.append(hb[b].half().cpu()); P.append(pos); V.append(vel)
    if i % 100 == 0:
        print(f"train {i}/{len(tr)} · 블록 {len(X)}", flush=True)
X = torch.stack(X); P = torch.tensor(np.stack(P)); V = torch.tensor(np.stack(V))
head = Head().to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
for ep in range(a.epochs):
    perm = torch.randperm(len(X)); tot = 0
    for j in range(0, len(X), 256):
        k = perm[j:j + 256]
        pp, vv, _ = head(X[k].to(dev))
        loss = (pp - P[k].to(dev)).abs().mean() / 256 + (vv - V[k].to(dev)).abs().mean() / 8
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(k)
    if ep % 5 == 4 or ep == a.epochs - 1:
        print(f"head epoch {ep + 1}: 학습 손실 {tot / len(X):.4f} (블록 {len(X)})", flush=True)
del X
head.eval()

# ---- 2. 평가: val clip 의 정답 토큰 · AR 예측 토큰 -----------------------------------------------------
models = [(r, load_pred(r, c)) for r, c in a.run]
R = {}
def add(key, blocks, labs):
    with torch.no_grad():
        pp, vv, w = head(blocks)
    for i, (kind, pos, vel) in enumerate(labs):
        if kind is None or not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            continue
        cell = int(np.clip(pos[1] // 16, 0, G - 1)) * G + int(np.clip(pos[0] // 16, 0, G - 1))
        R.setdefault((key, kind), []).append((pp[i].cpu().numpy(), vv[i].cpu().numpy(), pos, vel, float(w[i, cell]) * S))

for i, p in enumerate(clips("val")):
    for hb, lab in windows(p):
        kinds = []
        for b, (vis, hid, pos, vel) in enumerate(lab):
            reap = vis and b > 0 and any(lab[c][1] for c in range(max(0, b - 3), b))
            kinds.append(("다시 나타남" if reap else "보임") if vis else ("가림" if hid else None))
        lab_ = [(kinds[b], lab[b][2], lab[b][3]) for b in range(C + K)]
        add("정답 문맥", hb[:C], lab_[:C]); add("정답 미래", hb[C:], lab_[C:])
        for name, m in models:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_ro = m.rollout_kv(hb[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
                p_tf = m.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)).float()[0]
            add(f"{name} TF 한 걸음", p_tf.view(C + K - 1, S, -1)[C - 1:], lab_[C:])
            add(f"{name} 굴린 예측", p_ro, lab_[C:])
    if i % 100 == 0:
        print(f"val {i}", flush=True)

print("\n정답 토큰으로 학습한 attention head — 위치 오차 px 중앙값 / 방향 cos 중앙값 (움직이는 블록) / 정답 칸 attention (균등 1) / n")
keys = sorted({k for k, _ in R}, key=lambda s: (not s.startswith("정답"), s))
for key in keys:
    cells = []
    for kind in ("보임", "다시 나타남", "가림"):
        rows = R.get((key, kind), [])
        if not rows:
            cells.append(f"{kind} —"); continue
        pp = np.stack([r[0] for r in rows]); vv = np.stack([r[1] for r in rows]); pt = np.stack([r[2] for r in rows]); vt = np.stack([r[3] for r in rows])
        err = np.median(np.linalg.norm(pp - pt, axis=1)); mv = np.linalg.norm(vt, axis=1) > 1
        cos = np.median((vv[mv] * vt[mv]).sum(1) / (np.linalg.norm(vv[mv], axis=1) * np.linalg.norm(vt[mv], axis=1) + 1e-9)) if mv.sum() > 5 else float("nan")
        cells.append(f"{kind} {err:5.1f} px / {cos:4.2f} / {np.median([r[4] for r in rows]):5.1f} / {len(rows)}")
    print(f"  {key:40s} " + "  |  ".join(cells))
