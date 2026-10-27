#!/usr/bin/env python3
"""감독 모델의 학습된 state head (1280 차원 AR 출력에 붙은 attention pooling head) 를 그대로 문맥 · 실제 미래 · 예측 토큰에 씌운다 (2026-10-23, 사용자 질문).

  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/state_head_transfer.py --run natural_ar_vw_state_phys3d_event latest.pt

데이터 · 창은 velocity_transfer.py 와 같다 (phys3d_single_v1 val, 문맥 블록 0..7 → 미래 8..15).
head 는 학습 때 AR 출력 (TF · 굴린 예측) 에만 손실을 받았다 — 문맥 · 실제 미래 토큰 (LN(target_encoder)) 은 학습에서 본 적이 없다.
보고 (보이는 블록만): 위치 오차 px (중앙값) · 움직이는 블록 (|v| > 1 px/프레임) 의 방향 cosine (중앙값) · 속력 Spearman.
라벨 단위: 위치 = states.pos / 256 (head 출력 × 256 = px), 방향 = 단위 벡터, 속력 = 그 run 의 라벨 정의를 모르므로 순위 상관만 본다."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR, state_head as SHM                                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--run", nargs=2, required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--head", default=None, help="다른 run 의 head 를 씌운다 (z_training/runs 기준 경로, 예: phys3d_state_head_pretrain/e20.pt) — 2026-10-26")
ap.add_argument("--memfrac", type=float, default=0.0)
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
ck = torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)
pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
          num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred = vit_prefix_predictor(**kw)
pred.load_state_dict({k: v for k, v in _clean_backbone_key(ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
pred = pred.to(dev).eval()
_hk = torch.load(os.path.join(REPO, "z_training/runs", a.head), map_location="cpu", weights_only=False) if a.head else ck
head = SHM.build_head(_hk["state_head_cfg"], 1280); head.load_state_dict(_hk["state_head"]); head = head.to(dev).eval()
print(f"state head ← {a.head or a.run[0]} · predictor ← {a.run[0]}:{a.run[1]} (epoch {ck.get('epoch')})", flush=True)

paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
if a.limit:
    paths = paths[: a.limit]
R = {k: [] for k in ("ctx", "fut", "ro", "tf")}     # (pred pos px 2, pred vel 3, true pos px 2, true vel 2, attn on true cell)


@torch.no_grad()
def run_head(blocks):                                # (n, S, D) → pos px (n,2), vel (n,3), attn (n,S)
    o = head(blocks.reshape(1, -1, blocks.size(-1)).float(), S)
    return (o["pos"][0, :, 0] * 256).cpu().numpy(), o["vel"][0, :, 0].cpu().numpy(), o["attn"][0, :, 0].cpu().numpy()


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
            p_tf = pred.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)).float()[0]
        p_tf = p_tf.view(C + K - 1, S, -1)[C - 1:]
        lab = []
        for b in range(C + K):
            f0, f1 = idx[2 * b], idx[2 * b + 1]
            lab.append((st["vis_frac"][f0, 0] > 0.3 and st["vis_frac"][f1, 0] > 0.3, st["pos"][f1, 0], (st["vel"][f0, 0] + st["vel"][f1, 0]) / 2))
        cellof = lambda q: int(np.clip(q[1] // 16, 0, G - 1)) * G + int(np.clip(q[0] // 16, 0, G - 1))
        old_ok = lab[C - 1][0] and np.isfinite(lab[C - 1][1]).all()          # 2026-10-24: 옛 자리 = 문맥 마지막 블록의 물체 칸
        for rep, blocks, bs in (("ctx", hb[:C], range(C)), ("fut", hb[C:], range(C, C + K)), ("ro", p_ro, range(C, C + K)), ("tf", p_tf, range(C, C + K))):
            pp, vv, aa = run_head(blocks)
            for i, b in enumerate(bs):
                vis, pos, v = lab[b]
                if vis and np.isfinite(pos).all() and np.isfinite(v).all():
                    cell = cellof(pos)
                    oc = cellof(lab[C - 1][1]) if old_ok else None
                    j = int(aa[i].argmax()); top = np.array([j % G * 16 + 8, j // G * 16 + 8], np.float32)
                    _sa = np.array([(aa[i] * (np.arange(S) % G * 16 + 8)).sum(), (aa[i] * (np.arange(S) // G * 16 + 8)).sum()], np.float32)   # 2026-10-26 soft-argmax 위치 (칸 좌표 가중평균)
                    R[rep].append((pp[i], vv[i], pos, v, aa[i][cell] * S,
                                   aa[i][oc] * S if oc is not None else np.nan,
                                   np.linalg.norm(lab[C - 1][1] - pos) if old_ok else np.nan,     # 옛 자리 ↔ 정답 거리 (px)
                                   np.linalg.norm(top - pos), np.linalg.norm(top - lab[C - 1][1]) if old_ok else np.nan,
                                   b - C if rep in ("ro", "tf", "fut") else -1, _sa, top))
    if ci % 50 == 0:
        print(f"{ci}/{len(paths)} clip", flush=True)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


print("\n같은 state head 를 씌운 결과 (보이는 블록). 위치 오차 px 중앙값 · 움직이는 블록 방향 cos 중앙값 · 속력 Spearman · 정답 칸 attention (균등 = 1)")
print(f"  {'토큰':30s} {'위치 오차':>8s} {'방향 cos':>8s} {'속력 ρ':>7s} {'정답칸 attn':>10s}   n")
names = {"ctx": "문맥 (LN target enc, 학습 때 안 봄)", "fut": "실제 미래 (학습 때 안 봄)", "ro": "굴린 예측 (학습 때 봄)", "tf": "TF 한 걸음 (학습 때 봄)"}
for rep in ("ctx", "fut", "tf", "ro"):
    rows = R[rep]
    pp = np.stack([r[0] for r in rows]); vv = np.stack([r[1] for r in rows]); pt = np.stack([r[2] for r in rows]); vt = np.stack([r[3] for r in rows]); at = np.array([r[4] for r in rows])
    err = np.linalg.norm(pp - pt, axis=1)
    mv = np.linalg.norm(vt, axis=1) > 1.0
    d = vv[mv, :2]; cos = (d * vt[mv]).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(vt[mv], axis=1) + 1e-9)
    print(f"  {names[rep]:30s} {np.median(err):8.1f} {np.median(cos):8.2f} {spearman(vv[mv, 2], np.linalg.norm(vt[mv], axis=1)):7.2f} {np.median(at):10.1f}   {len(rows)} (움직임 {mv.sum()})")


# ---- 2026-10-24: 굴린 예측 걸음별 — head 가 정답 자리 / 옛 자리 중 어디를 보나 (물체가 1 칸 넘게 움직인 블록만) ----
print("\n물체가 1 칸 (16 px) 넘게 움직인 블록 (옛 자리 = 문맥 마지막 블록 물체 칸). attention ×256 (균등 1) · attention 최대 칸이 정답 / 옛 자리에서 몇 px")
print(f"  {'토큰':12s} {'걸음':>4s} {'위치오차':>8s} {'정답칸':>7s} {'옛칸':>7s} {'최대칸-정답':>10s} {'최대칸-옛':>9s}   n")
for rep in ("fut", "tf", "ro"):
    rows = [r for r in R[rep] if np.isfinite(r[6]) and r[6] > 16]
    for st_ in ([None] + list(range(K))):
        rr = [r for r in rows if st_ is None or r[9] == st_]
        if len(rr) < 10:
            continue
        g = lambda k: np.nanmedian([r[k] for r in rr])
        err = np.median([np.linalg.norm(r[0] - r[2]) for r in rr])
        print(f"  {names[rep][:12]:12s} {('전체' if st_ is None else str(st_ + 1)):>4s} {err:8.1f} {g(4):7.1f} {g(5):7.1f} {g(7):10.1f} {g(8):9.1f}   {len(rr)}")


# ---- 2026-10-26: 위치를 attention 이 가리키는 칸 좌표로 읽기 (soft-argmax, 파라미터 없음) vs 지금 선형 읽기 ----
print("\n위치 읽기 비교 (보이는 블록, 오차 px 중앙값): 선형 (z → Linear) · soft-argmax (Σ attention × 칸 중심) · argmax 칸 중심")
print(f"  {'토큰':30s} {'선형':>7s} {'soft-argmax':>12s} {'argmax':>8s}   | 1 칸 넘게 움직인 블록: {'선형':>6s} {'soft':>6s} {'argmax':>7s}  n")
for rep in ("ctx", "fut", "tf", "ro"):
    rows = R[rep]
    e_lin = np.array([np.linalg.norm(r[0] - r[2]) for r in rows]); e_sa = np.array([np.linalg.norm(r[10] - r[2]) for r in rows]); e_am = np.array([np.linalg.norm(r[11] - r[2]) for r in rows])
    mv = np.array([np.isfinite(r[6]) and r[6] > 16 for r in rows])
    f = lambda e: f"{np.median(e[mv]):6.1f}" if mv.any() else "   —  "
    print(f"  {names[rep]:30s} {np.median(e_lin):7.1f} {np.median(e_sa):12.1f} {np.median(e_am):8.1f}   | {'':22s}{f(e_lin)} {f(e_sa)} {f(e_am)}  {mv.sum()}")
