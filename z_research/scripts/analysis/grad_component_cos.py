#!/usr/bin/env python3
"""고정 상태 판독기 손실의 성분끼리, 그리고 latent 손실과 predictor gradient 가 어느 방향을 보는지 잰다 (2026-10-26, 사용자 요청).
gradient 크기 맞춤 run 에서 가중 후 판독기 gradient 가 성분 목표의 합보다 작게 나와 (0.8x vs 무관하면 약 1.2x) 성분끼리 상쇄되는지 확인.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/grad_component_cos.py \
      --run natural_ar_vw_state_phys3d_event_gthead_frozen_gm <ckpt> [--head phys3d_state_head_pretrain/e20.pt] [--nb 10]

데이터: phys3d_single_v1 val (학습에 안 쓴 clip), 창 = raw 0 부터 간격 3, 문맥 블록 8 → 예측 블록 8 (학습 사건 창과 같은 구성), batch 4 clip.
손실: AR.ar_losses (학습과 같은 TF + 굴린 latent L1) · SHM.ar_state_loss_gt (고정 head, 보이는 블록, 성분 parts).
항: lat · tf_/ar_ × pos/dir/spd/ex · 성분 합 (pos = tf_pos + ar_pos …) · 가중 판독기 (run 의 마지막 grad_match 가중치).
보고: batch 마다 항끼리 cos 를 재 평균 (± 표준편차) + 기준 cos (같은 항, 다른 batch 끼리 — 고차원 잡음 바닥)."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR, state_head as SHM                                                       # noqa: E402
from z_training.data.phys2d_sup_labels import load_clip, window_labels                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--head", default="phys3d_state_head_pretrain/e20.pt")
ap.add_argument("--nb", type=int, default=24, help="batch 수")
ap.add_argument("--bs", type=int, default=1)
a = ap.parse_args()
dev = torch.device("cuda"); torch.manual_seed(0)
torch.cuda.set_per_process_memory_fraction(0.08)      # 약 7.6 GB — 학습과 같은 GPU 를 같이 쓴다
S, C, K, ST = 256, 8, 8, 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0])
cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
pc = cfg["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                            num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
ck = torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)
pred.load_state_dict({k: v for k, v in _clean_backbone_key(ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
pred = pred.to(dev).train()
hk = torch.load(os.path.join(REPO, "z_training/runs", a.head), map_location="cpu", weights_only=False)
head = SHM.build_head(hk["state_head_cfg"], 1280); head.load_state_dict(hk["state_head"]); head = head.to(dev).eval().requires_grad_(False)
SHc = cfg.get("state_head", {})
W = {}
mj = os.path.join(D, "monitor.jsonl")
if os.path.exists(mj):
    for l in open(mj):
        g = json.loads(l).get("gprobe", {})
        if "w_pos" in g:
            W = {c: g["w_" + c] for c in ("pos", "dir", "spd", "ex")}
print(f"{a.run[0]}:{a.run[1]} (epoch {ck.get('epoch')}) · 가중치 {W}", flush=True)

paths = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4")
               if load_clip(p)["present"].shape[0] >= ST * (2 * (C + K) - 1) + 1)
rng = np.random.default_rng(0); paths = [paths[i] for i in rng.permutation(len(paths))]
ps = [q for q in pred.parameters() if q.requires_grad]


def flat(l):
    g = torch.autograd.grad(l, ps, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(q)).reshape(-1).float() for x, q in zip(g, ps)]).cpu()   # CPU 에 (학습과 같은 GPU 를 쓸 때 메모리)


names = None; COS = []; NORM = []; G_prev = None; BASE = []
for b in range(a.nb):
    xs, labs = [], []
    for p in paths[b * a.bs:(b + 1) * a.bs]:
        clip = load_clip(p)
        labs.append({k: torch.from_numpy(v) for k, v in SHM.pack_labels(window_labels(clip, start=0, n_frames=2 * (C + K), stride=ST, tubelet=2, reduce="last")).items()})
        vr = decord.VideoReader(p, num_threads=2)
        x = torch.from_numpy(vr.get_batch([ST * k for k in range(2 * (C + K))]).asnumpy()).permute(3, 0, 1, 2).float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
        xs.append(x)
    x = torch.stack(xs).to(dev)
    lab = {k: torch.stack([l[k] for l in labs]).to(dev) for k in labs[0]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outs = {}
        lat, jl, sl, R = AR.ar_losses(pred, h, S, C, K, float(cfg.get("loss", {}).get("loss_exp", 1.0)), int(cfg.get("loss", {}).get("rollout_steps", 8)), out=outs)
        parts = {}
        SHM.ar_state_loss_gt(head, h, outs["p_tf"], outs["p_ar"], lab, S, C, R, SHc.get("w") or {}, SHc.get("split") or {}, fit_gt=False, parts=parts)
    terms = {"lat": lat, "lat_tf": jl, "lat_ar": sl}
    for c in ("pos", "dir", "spd", "ex"):
        terms[c] = parts[f"tf_{c}"] + parts[f"ar_{c}"]
    terms.update(parts)
    if W:
        terms["head_w"] = sum(W[c] * terms[c] for c in ("pos", "dir", "spd", "ex") if W[c] > 0)
    G = {n: flat(l) for n, l in terms.items()}
    names = list(G)
    M = torch.stack([G[n] for n in names]); Mn = F.normalize(M, dim=1)
    COS.append((Mn @ Mn.T).cpu().numpy()); NORM.append(M.norm(dim=1).cpu().numpy())
    if G_prev is not None:
        BASE.append({n: float(F.cosine_similarity(G[n], G_prev[n], dim=0)) for n in names})
    G_prev = {n: G[n].clone() for n in names}
    del G, M, Mn, outs, parts, terms, lat, jl, sl; torch.cuda.empty_cache()
    print(f"batch {b + 1}/{a.nb}", flush=True)

COS = np.stack(COS); NORM = np.stack(NORM)
mu, sd = COS.mean(0), COS.std(0)
idx = {n: i for i, n in enumerate(names)}
show = ["lat", "pos", "dir", "ex", "spd"] + (["head_w"] if W else [])
lab_ = {"lat": "latent", "pos": "위치", "dir": "방향", "ex": "존재정체", "spd": "속력", "head_w": "가중판독기"}
print(f"\n‖∇‖ 평균 (batch {a.nb} 개): " + " · ".join(f"{lab_.get(n, n)} {NORM[:, idx[n]].mean():.3f}" for n in show))
print("\ncos 평균 ± 표준편차 (batch 마다 잰 값)")
print(f"{'':10s}" + "".join(f"{lab_[n]:>14s}" for n in show))
for r in show:
    print(f"{lab_[r]:10s}" + "".join(f"{mu[idx[r], idx[c]]:>8.3f}±{sd[idx[r], idx[c]]:<5.2f}" for c in show))
print("\n기준: 같은 항, 연달은 두 batch 끼리 cos 평균 (잡음 바닥)")
print("  " + " · ".join(f"{lab_.get(n, n)} {np.mean([b_[n] for b_ in BASE]):.3f}" for n in show))
print("\n굴린 / TF 나눠서 (성분끼리 cos 평균)")
sub = ["ar_pos", "ar_dir", "ar_ex", "tf_pos", "tf_dir", "tf_ex", "lat_ar", "lat_tf"]
print(f"{'':8s}" + "".join(f"{n:>9s}" for n in sub))
for r in sub:
    print(f"{r:8s}" + "".join(f"{mu[idx[r], idx[c]]:>9.3f}" for c in sub))
if W:
    v = {c: W[c] * NORM[:, idx[c]].mean() for c in ("pos", "dir", "ex")}
    print(f"\n가중 후 크기: 위치 {v['pos']:.3f} · 방향 {v['dir']:.3f} · 존재정체 {v['ex']:.3f} · 가중 판독기 합 {NORM[:, idx['head_w']].mean():.3f} · latent {NORM[:, idx['lat']].mean():.3f}"
          f"  (서로 무관하면 합 ≈ {np.sqrt(sum(x ** 2 for x in v.values())):.3f})")
