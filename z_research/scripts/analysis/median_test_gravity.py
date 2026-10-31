#!/usr/bin/env python3
"""블록 AR 의 굴린 예측은 "가능한 미래들의 칸별 중앙값" 인가 (2026-10-28, vll7 가설 검사, 사용자 요청).

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/median_test_gravity.py --run natural_ar_vw_state latest.pt

데이터: VOBench_v1 gravity 의 arc 블록 (72 블록) — 같은 문맥 (raw 0–45, splice 48) 에 미래 9 개 (가능 1 = 중력 포물선 · 불가능 8 = 솟음 · 떠 있음 ·
  직선 · 멈춤 1/2/4/8/12 장 뒤). 물체 위치가 미래마다 갈린다 = 실제로 갈라진 미래 세트. 3 장 간격 32 장 = 블록 16 (문맥 8 → 미래 8, 0.2 초).
토큰: 블록별 LN(target_encoder). 예측 p = AR rollout (문맥 8 블록, 같은 문맥이라 미래 9 개에 공통).
비교 기준 (걸음 · 칸마다): ① 칸별 중앙값 (9 미래, 차원별) ② 칸별 평균 ③ 가장 가까운 미래 하나 ④ 가능 미래 (pos) ⑤ 복사 (마지막 문맥 블록).
칸 나누기: 9 미래 사이의 칸별 퍼짐 (차원 평균 |x − 중앙값|) 상위 10 % = "갈리는 칸" (물체가 미래마다 다른 자리에 있는 칸), 하위 50 % = "공통 칸".
보고: 기준마다 |p − 기준| (L1 평균) 와 가장 가까운 기준의 비율. 가설 (L1 최적 = 칸별 중앙값) 이 맞으면 갈리는 칸에서 ① 이 가장 가깝다."""
import argparse, collections, csv, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
from app.vjepa_frozen import ar as AR
ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs=2, required=True); a = ap.parse_args()
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.1)
S, C, K = 256, 8, 8
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0]); pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                            num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred.load_state_dict(_clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]), strict=True)
pred = pred.to(dev).eval()
ROOT = "/workspace/dataset/benchmarks/VOBench_v1/gravity"
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_gravity/index.csv"))) if r["law"] == "arc"]
blocks = collections.defaultdict(list)
for r in rows: blocks[r["block_id"]].append(r)

def enc(fn):
    x = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(ROOT, fn, f"{3 * k:06d}.png")).convert("RGB"))) for k in range(2 * (C + K))])
    x = F.interpolate(x.to(dev).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)[None] / 255
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)

NAMES = ["① 중앙값", "② 평균", "③ 가장 가까운 미래", "④ 가능 미래", "⑤ 복사 (문맥 끝)", "⑥ 위약: 다른 블록 9 미래 평균", "⑦ 위약: 문맥 8 블록 시간 평균", "⑧ 위약: 가능 미래 8 블록 시간 평균"]
acc = {reg: {k: [] for k in range(K)} for reg in ("갈리는 칸", "공통 칸")}      # (걸음) → list of (5 거리)
win = {reg: np.zeros(len(NAMES)) for reg in acc}
ctx_check = []
prev_mean = None                                                   # 위약 ⑥: 바로 앞 블록 (다른 장면) 의 9 미래 평균
for bid, arms in blocks.items():
    H = {r["variant"]: enc(r["file_name"]) for r in arms}
    if "pos" not in H or len(H) < 5: continue
    hp = H["pos"]
    ctx_check.append(max(float((H[v][:C] - hp[:C]).abs().max()) for v in H))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = pred.rollout_kv(hp[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev)[None], K).float()[0].view(K, S, -1)
    Fut = torch.stack([H[v][C:] for v in H])                      # (9, K, S, D)
    med = Fut.median(0).values; mean = Fut.mean(0)
    spread = (Fut - med[None]).abs().mean(-1).mean(0)            # (K, S)
    for k in range(K):
        q = torch.quantile(spread[k], torch.tensor([0.5, 0.9], device=dev))
        for reg, m in (("갈리는 칸", spread[k] >= q[1]), ("공통 칸", spread[k] <= q[0])):
            d = lambda ref: float((p[k, m] - ref[m]).abs().mean())
            dfut = [d(Fut[i, k]) for i in range(Fut.size(0))]
            ds = [d(med[k]), d(mean[k]), min(dfut), d(hp[C + k]), d(hp[C - 1]),
                  d(prev_mean[k]) if prev_mean is not None else float("nan"), d(hp[:C].mean(0)), d(hp[C:].mean(0))]
            acc[reg][k].append(ds); win[reg][int(np.nanargmin(ds))] += 1
    prev_mean = mean
print(f"{a.run[0]}:{a.run[1]} · arc 블록 {len(ctx_check)} · 미래 9 개 · 문맥 토큰 최대 차이 (미래끼리) {max(ctx_check):.2e}\n")
for reg in acc:
    print(f"[{reg}] |p − 기준| L1 평균 (걸음 1 · 4 · 8 · 전체) · 가장 가까운 비율")
    for j, n in enumerate(NAMES):
        per = [np.nanmean([x[j] for x in acc[reg][k]]) for k in (0, 3, 7)]
        allv = np.nanmean([x[j] for k in range(K) for x in acc[reg][k]])
        print(f"   {n:18s} {per[0]:.4f} · {per[1]:.4f} · {per[2]:.4f} · {allv:.4f}   가장 가까움 {100 * win[reg][j] / win[reg].sum():5.1f}%")
    print()
