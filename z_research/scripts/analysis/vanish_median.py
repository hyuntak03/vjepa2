#!/usr/bin/env python3
"""움직이는 물체는 왜 굴린 예측에서 사라지나 — "속력을 잃어 위치가 퍼지면 L1 최적 (칸별 중앙값) 이 배경" 가설 검사 (2026-10-27, 사용자 요청).

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vanish_median.py --run natural_ar_vw_state latest.pt

데이터: phys3d_single_v1 val (학습에 안 쓴 clip), 창 raw r0 ∈ {0, 2} 간격 3, 문맥 블록 8 → 굴린 예측 8 걸음 (transport_gain.py 와 같은 창).
토큰은 전부 LN(target_encoder(블록)) 공간. 칸 = 16 px. 블록 b 의 물체 칸 = states.pos (블록 마지막 장).
물체다움 (파라미터 없음): 칸 c 에서 o = d_bg / (d_obj + d_bg),  d = 토큰 L1 평균.
  d_obj = |예측(c) − 정답 미래 토큰(c)| (물체가 있는 토큰),  d_bg = |예측(c) − 문맥 마지막 블록 토큰(c)| (그 칸에 아직 물체가 없던 토큰 = 배경)
  → 1 = 물체 쪽, 0 = 배경 쪽, 0.5 = 중간. 정답 미래 토큰에 걸면 1, 문맥 토큰에 걸면 0 (자 검사).
옛 자리 (문맥 마지막 블록의 물체 칸) 도 같은 식: 예측이 옛 물체 (문맥 토큰) 와 지금 그 칸의 배경 (정답 미래 토큰) 중 어디에 가까운가.
조건: 그 블록에 물체가 보이고 (vis_frac > 0.3), 문맥 끝에서 정답 칸까지 1 칸 (16 px) 넘게 움직였을 때 (정답 칸 ≠ 옛 칸 이고 정답 칸이 문맥 끝에 배경).
보고 ① 걸음별 정답 칸 / 옛 자리 / 정답 칸 5×5 최대 물체다움  ② 이동 거리 · 물체 속력 구간별 정답 칸 물체다움 (사라짐이 거리 · 속력과 함께 오나)
  ③ 정지 물체 (이동 < 4 px) 대조: 같은 칸에서 예측이 정답 (물체) 쪽인가."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--memfrac", type=float, default=0.08)
a = ap.parse_args()
dev = torch.device("cuda")
if a.memfrac: torch.cuda.set_per_process_memory_fraction(a.memfrac)
S, G, C, K, ST = 256, 16, 8, 8, 3
T = 2 * (C + K)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                            num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
ck = torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)
pred.load_state_dict({k: v for k, v in _clean_backbone_key(ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
pred = pred.to(dev).eval()
print(f"{a.run[0]}:{a.run[1]} (epoch {ck.get('epoch')})", flush=True)

paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
cell = lambda q: int(np.clip(q[1] // 16, 0, G - 1)) * G + int(np.clip(q[0] // 16, 0, G - 1))
def nb5(c):
    r, q = divmod(c, G)
    return [rr * G + qq for rr in range(max(0, r - 2), min(G, r + 3)) for qq in range(max(0, q - 2), min(G, q + 3))]
MOV, STAT = [], []     # MOV: (걸음, 거리 px, 속력 px/장, o_true, o_old, o_max5, o_gt_true(자 검사), o_ctx_true(자 검사))
for ci, path in enumerate(paths):
    st = np.load(path[:-4] + "_states.npz")
    vr = decord.VideoReader(path, num_threads=2)
    for r0 in (0, 2):
        idx = [r0 + ST * k for k in range(T)]
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
            p_ro = pred.rollout_kv(hb[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        f_last = idx[2 * C - 1]
        if not (st["vis_frac"][f_last, 0] > 0.3 and np.isfinite(st["pos"][f_last, 0]).all()):
            continue
        old = cell(st["pos"][f_last, 0]); hc = hb[C - 1]
        for k in range(K):
            b = C + k; f1 = idx[2 * b + 1]
            pos = st["pos"][f1, 0]
            if not (st["vis_frac"][f1, 0] > 0.3 and np.isfinite(pos).all()):
                continue
            tr = cell(pos); hf = hb[b]; p = p_ro[k]
            dist = float(np.linalg.norm(pos - st["pos"][f_last, 0])); spd = float(np.linalg.norm(st["vel"][f1, 0]))
            d = lambda u, v: float((u - v).abs().mean())
            o = lambda c, obj, bg: d(p[c], bg[c]) / (d(p[c], obj[c]) + d(p[c], bg[c]) + 1e-9)
            if tr == old:
                if dist < 4:      # 정지 대조: 같은 칸, 예측이 정답 (물체) 쪽인가 — 배경 기준이 없으니 정답 대비 거리만
                    STAT.append((k, d(p[tr], hf[tr]), d(hc[tr], hf[tr])))
                continue
            o_true = o(tr, hf, hc)                          # 정답 칸: 물체 (정답 미래) vs 배경 (문맥 끝)
            o_old = o(old, hc, hf)                          # 옛 자리: 옛 물체 (문맥 끝) vs 지금 배경 (정답 미래)
            o_max5 = max(o(c, hf, hc) for c in nb5(tr) if c != old)
            gt_t = d(hf[tr], hc[tr]) / (0 + d(hf[tr], hc[tr]) + 1e-9)      # = 1 (자 검사)
            MOV.append((k, dist, spd, o_true, o_old, o_max5, gt_t, 1 - o_true if False else 0.0))
    if ci % 100 == 0:
        print(f"{ci}/{len(paths)} clip · 움직임 블록 {len(MOV)} · 정지 {len(STAT)}", flush=True)

M = np.array(MOV)
print(f"\n움직인 블록 {len(M)} (문맥 끝에서 1 칸 넘게, 물체 보임). 물체다움 o: 1 = 정답 미래 (물체) 쪽, 0 = 배경 (문맥 끝 그 칸) 쪽")
print(f"  {'걸음':>4s} {'정답칸 o':>9s} {'o<0.5 비율':>10s} {'옛자리 o':>9s} {'정답 5×5 최대 o':>15s} {'평균 거리 px':>12s}   n")
for k in range(K):
    m = M[:, 0] == k
    if m.sum() < 10: continue
    print(f"  {k + 1:4d} {np.median(M[m, 3]):9.3f} {100 * np.mean(M[m, 3] < 0.5):9.1f}% {np.median(M[m, 4]):9.3f} {np.median(M[m, 5]):15.3f} {np.mean(M[m, 1]):12.1f}   {m.sum()}")
print("  옛자리 o: 1 = 예측이 옛 물체를 그 자리에 남김 (복사), 0 = 그 자리를 배경으로 비움")
print("\n이동 거리 구간별 (문맥 끝 → 그 블록) 정답 칸 물체다움 중앙값 · o<0.5 비율")
for lo, hi in ((16, 32), (32, 48), (48, 64), (64, 96), (96, 400)):
    m = (M[:, 1] >= lo) & (M[:, 1] < hi)
    if m.sum() >= 10: print(f"  {lo:3d}~{hi:3d} px  o {np.median(M[m, 3]):.3f}  배경 쪽 {100 * np.mean(M[m, 3] < 0.5):5.1f}%  옛자리 o {np.median(M[m, 4]):.3f}   n {m.sum()}")
print("\n물체 속력 구간별 (px/장, 그 블록) — 같은 거리 구간 (16~48 px) 안에서")
mm = (M[:, 1] >= 16) & (M[:, 1] < 48)
for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 100)):
    m = mm & (M[:, 2] >= lo) & (M[:, 2] < hi)
    if m.sum() >= 10: print(f"  {lo:3d}~{hi:3d}  o {np.median(M[m, 3]):.3f}  배경 쪽 {100 * np.mean(M[m, 3] < 0.5):5.1f}%   n {m.sum()}")
if STAT:
    Sa = np.array(STAT)
    print(f"\n정지 물체 대조 (이동 < 4 px, 같은 칸) n {len(Sa)}: |예측 − 정답| 중앙 {np.median(Sa[:, 1]):.3f} · 참고 |문맥 끝 − 정답| {np.median(Sa[:, 2]):.3f}")
    for k in range(K):
        m = Sa[:, 0] == k
        if m.sum() >= 10: print(f"   걸음 {k + 1}: |예측 − 정답| {np.median(Sa[m, 1]):.3f}  n {m.sum()}")
