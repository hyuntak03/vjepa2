#!/usr/bin/env python3
"""predictor 안에서 물체 속도가 어디 (어느 칸 · 어느 층) 에 있나 — 속도 개입 실험 설계 전 (2026-10-27, 사용자 요청).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/velocity_where_predictor.py --run natural_ar_vw_state_fps10_c16k8 e6.pt --n 200

데이터: phys3d_single_v1 val (속도 라벨), 30 fps 3 장 간격 32 장 = 16 블록 (0.2 초). 문맥 8 블록 → 굴린 8 걸음 (rollout_kv, 상태 되먹임).
상태: 블록 (Transformer block) 12 층 각각의 출력 (384) — forward_kv 를 감싸 기록. 0 층 = 입력 임베딩.
  관측 상태 = 문맥 채우기 걸음 b (입력 = 정답 블록 b, 그 블록의 현재 속도),  굴린 상태 = 굴리기 걸음 r (입력 = 자기 상태, 블록 C−1+r 의 현재).
영역 (그 블록의 실제 물체 위치 기준): 물체 3×3 칸 평균 · 먼 칸 (물체에서 체비셰프 거리 ≥ 5 칸) 평균 · 전체 256 칸 평균.
라벨: 그 블록 마지막 장의 화면 속도 (states.vel × 3 = 0.1 초당 px, 2 차원). 물체가 보이는 블록만.
ridge (clip 반 맞춤 / 반 확인) R²: ① 관측 → 관측 ② 관측에서 맞춘 판독기 → 굴린 상태 (이식) ③ 굴린 → 굴린 (자기 학습)."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True)
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.08)
G, S, C, K, ST = 16, 256, 8, 8, 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0]); pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                            num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"), n_registers=int(pc.get("n_registers", 0)))
pred.load_state_dict(_clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]), strict=True)
pred = pred.to(dev).eval()
print(f"{a.run[0]}:{a.run[1]}", flush=True)
REC = []
for li, blk in enumerate(pred.predictor_blocks):
    f0 = blk.forward_kv
    def fk(x, mask, H, W, past=None, _f=f0, _li=li):
        y, kv = _f(x, mask, H, W, past)
        if _li == 0: REC.append([x[:, :S].float()])          # 0 층 = 입력 (임베딩 / 되먹임 상태)
        REC[-1].append(y[:, :S].float())
        return y, kv
    blk.forward_kv = fk

def region_feats(st, pos):        # st (L, 256, 384) → dict 영역 → (L, 384)
    cx, cy = int(np.clip(pos[0] // 16, 0, G - 1)), int(np.clip(pos[1] // 16, 0, G - 1))
    r, c = np.arange(S) // G, np.arange(S) % G
    near = (np.abs(r - cy) <= 1) & (np.abs(c - cx) <= 1); far = np.maximum(np.abs(r - cy), np.abs(c - cx)) >= 5
    return {"물체 3×3": st[:, near].mean(1), "먼 칸": st[:, far].mean(1), "전체 평균": st.mean(1)}

paths = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4") if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
paths = [paths[i] for i in np.random.default_rng(0).permutation(len(paths))[: a.n]]   # (기본 --n 은 전부)
OBS, RO = [], []          # (clip, 영역 dict, vel)
for ci, p in enumerate(paths):
    st = np.load(p[:-4] + "_states.npz"); vr = decord.VideoReader(p, num_threads=2)
    idx = [ST * k for k in range(2 * (C + K))]
    x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    if x.shape[-1] != 256: x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()
        REC.clear()
        pred.rollout_kv(h[:, : C * S], torch.arange(C * S, device=dev)[None], K)
    for s, layers in enumerate(REC):                       # 걸음 s: 0..C-1 관측, C..C+K-2 굴림
        blkid = s if s < C else s                          # 관측 걸음 s = 블록 s, 굴리기 걸음 (s-C+1) = 블록 C-1+(s-C+1) = s
        f1 = idx[2 * blkid + 1]
        if not (st["vis_frac"][f1, 0] > 0.3 and np.isfinite(st["pos"][f1, 0]).all()): continue
        feats = region_feats(torch.stack(layers)[:, 0].cpu().numpy(), st["pos"][f1, 0] * (256 / vr[0].shape[1]))
        (OBS if s < C else RO).append((ci, feats, st["vel"][f1, 0] * ST))
    if ci % 50 == 0: print(f"{ci}/{len(paths)}", flush=True)

def ridge(Xa, Ya):
    """표준화 + 맞춤 반 안에서 다시 반으로 나눠 lam 고르기 (384 차원 · 표본 수백이라 규제가 필요). return (mu, sd, W)."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd
    def fit(Z_, Y_, lam):
        Z1 = np.c_[Z_, np.ones(len(Z_))]; return np.linalg.solve(Z1.T @ Z1 + lam * np.eye(Z1.shape[1]), Z1.T @ Y_)
    n = len(Z) // 2; best = None
    for lam in (1, 10, 100, 1e3, 1e4, 1e5):
        W = fit(Z[:n], Ya[:n], lam); P = np.c_[Z[n:], np.ones(len(Z) - n)] @ W
        sc = 1 - ((Ya[n:] - P) ** 2).sum() / ((Ya[n:] - Ya[n:].mean(0)) ** 2).sum()
        if best is None or sc > best[0]: best = (sc, lam)
    return mu, sd, fit(Z, Ya, best[1])
def r2(M, X, Y):
    mu, sd, W = M; P = np.c_[(X - mu) / sd, np.ones(len(X))] @ W
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
half = len(paths) // 2
print(f"\n관측 블록 {len(OBS)} · 굴린 블록 {len(RO)} · 확인 반 R² (속도 2 차원)")
print(f"{'영역':8s} {'층':>3s} {'① 관측→관측':>11s} {'② 관측→굴림':>11s} {'③ 굴림→굴림':>11s}")
for reg in ("물체 3×3", "먼 칸", "전체 평균"):
    for L in range(13):
        sel = lambda D, tr: (np.array([d[1][reg][L] for d in D if (d[0] < half) == tr]), np.array([d[2] for d in D if (d[0] < half) == tr]))
        Xa, Ya = sel(OBS, True); Xb, Yb = sel(OBS, False); Ra, Rya = sel(RO, True); Rb, Ryb = sel(RO, False)
        W = ridge(Xa, Ya); Wr = ridge(Ra, Rya)
        print(f"{reg:8s} {L:3d} {r2(W, Xb, Yb):11.3f} {r2(W, Rb, Ryb):11.3f} {r2(Wr, Rb, Ryb):11.3f}")
